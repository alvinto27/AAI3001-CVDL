"""CPU-only core parity, stored legacy samples, and one-core timings on VM."""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import os
import sys
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

os.environ["OMP_NUM_THREADS"] = "1"
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "LSB-Dataset-Generator"))
import numpy as np
from PIL import Image
import lsb_engine as engine
import lsb_batch as batch
import lsb_settings as settings


def original(path):
    spec = importlib.util.spec_from_file_location("old_engine", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def public(record):
    return {k:v for k,v in vars(record).items() if not k.startswith("_")}


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def property_chunk(job):
    old_path, begin, end = job
    old = original(old_path)
    counts, shapes = Counter(), Counter()
    for i in range(begin, end):
        generator = np.random.default_rng(41000 + i)
        # Every Cartesian settings combination recurs, not just chance coverage.
        mode = ("lowest", "plane")[i % 2]
        k = i // 2 % 8 + 1
        method = ("replace", "match")[i // 16 % 2]
        placement = ("random", "contiguous", "stride")[i // 32 % 3]
        shape = ((256, 256), (512, 512), (257, 263), (33, 65), (17, 19))[i % 5]
        if i == 192:
            shape = (3280, 4948)
        carrier = generator.integers(0, 256, size=(*shape, 3), dtype=np.uint8)
        snapshot = carrier.copy()
        stride = int(generator.integers(1, 17)) if placement == "stride" else 1
        bits = k if mode == "lowest" else 1
        capacity = ((carrier.size - 1) // stride + 1) * bits // 8
        count = int(generator.integers(1, min(capacity, 4096) + 1))
        units = (count * 8 + bits - 1) // bits
        maximum_start = carrier.size - 1 - (units - 1) * stride
        start = None if i % 3 else int(generator.integers(0, maximum_start + 1))
        kwargs = dict(bit_mode=mode, k=k, method=method, placement=placement,
                      stride=stride, start=start, payload=("random_bytes", "letters")[i // 96 % 2])
        kwargs["payload_bytes" if i // 192 % 2 else "bpp"] = (
            count if i // 192 % 2 else
            (count + float(generator.uniform(-0.49, 0.49))) * 8 / (shape[0] * shape[1]))
        rngs = [np.random.default_rng(i + 731) for _ in range(4)]
        expected, expected_mask, expected_result = old.embed(carrier, rng=rngs[0], **kwargs)
        full, mask, result = engine.embed(carrier, rng=rngs[1], **kwargs)
        fast, metadata, changed = engine.embed_fast(carrier, rng=rngs[2], return_changed_units=True, **kwargs)
        plain, plain_meta, absent = engine.embed_fast(carrier, rng=rngs[3], **kwargs)
        require(all(np.array_equal(expected, a) for a in (full, fast, plain)), f"pixel case {i}")
        require(np.array_equal(expected_mask, mask), f"selection mask case {i}")
        require(public(expected_result) == public(result) == metadata == plain_meta, f"metadata case {i}")
        require(all(r.bit_generator.state == rngs[0].bit_generator.state for r in rngs[1:]), f"RNG case {i}")
        reference_changes = np.flatnonzero(carrier.reshape(-1) != fast.reshape(-1))
        require(np.array_equal(changed, reference_changes), f"changed units case {i}")
        require(not changed.flags.writeable and absent is None, f"optional units case {i}")
        require(np.array_equal(carrier, snapshot), f"clean array mutation {i}")
        counts[f"{mode}/k{k}/{method}/{placement}"] += 1
        shapes[f"{shape[0]}x{shape[1]}"] += 1
    return dict(combinations=dict(counts), shapes=dict(shapes))


def properties(path, count, workers):
    chunks = [(str(path), begin, min(begin + 500, count)) for begin in range(0, count, 500)]
    total, shapes = Counter(), Counter()
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for result in pool.map(property_chunk, chunks):
            total.update(result["combinations"])
            shapes.update(result["shapes"])
    require(sum(total.values()) == count and len(total) == 96, "incomplete Cartesian coverage")
    return dict(cases=count, settings_combinations=len(total),
                minimum_cases_per_combination=min(total.values()), shapes=dict(shapes))


def legacy(root, count):
    config = batch.read_config(root / "configs/raise512-run1.yaml")
    with (root / "Datasets/raise512-run1/labels.csv").open(newline="", encoding="utf-8") as stream:
        rows = [r for r in csv.DictReader(stream) if r["label"] == "1"]
    # Uniform spread across all covers, plus mandatory groups if needed.
    chosen = {int(x) for x in np.linspace(0, len(rows)-1, count, dtype=int)}
    for field in ("placement", "method", "bit_mode"):
        for value in {r[field] for r in rows}:
            chosen.add(next(i for i,r in enumerate(rows) if r[field] == value))
    groups = Counter()
    for index in sorted(chosen):
        row = rows[index]
        with Image.open(root / "Datasets/raise512-run1/clean" / f"{row['cover_id']}.png") as image:
            carrier = np.array(image)
        key = tuple(int(row[f"embed_seed_{f}"]) for f in ("seed", "cover_index", "variant", "stream"))
        # Shared settings redraw preserves requested bpp and all legacy settings.
        kwargs, _ = settings.draw_legacy_settings(config, carrier,
                    np.random.default_rng(settings.legacy_seed_sequence(key[0], key[1], key[2], 0)))
        kwargs["rng"] = np.random.default_rng(settings.legacy_seed_sequence(*key))
        fast, meta, _ = engine.embed_fast(carrier, **kwargs)
        with Image.open(root / "Datasets/raise512-run1" / row["file"]) as image:
            require(np.array_equal(fast, np.asarray(image)), f"stored pixel mismatch {row['cover_id']}")
        for field in ("bit_mode", "k", "method", "placement", "start", "stride", "payload_kind",
                      "payload_bytes", "embedded_bits", "units_used", "units_changed", "bpp_actual"):
            require(str(batch._csv_value(meta[field])) == row[field], f"stored metadata mismatch {field}")
        groups[f"{row['bit_mode']}/{row['method']}/{row['placement']}"] += 1
    require(len(groups) == 12, "legacy sample missing settings groups")
    return dict(rows=len(chosen), groups=dict(groups), mismatches=0)


def ordering_benchmark():
    results = []
    for name, total in (("512", 512*512*3), ("D7000", 3280*4948*3)):
        for density in (0.002083333333, 0.020833333333, 0.166666666667):
            n = int(total * density)
            chosen = np.random.default_rng(17).choice(total, size=n, replace=False)
            times = {"sort": [], "mark": []}
            expected = np.sort(chosen)
            for trial in range(6):
                for method in ("sort", "mark"):
                    sample = chosen.copy()  # outside timer; actual choice owns its array
                    start = time.perf_counter()
                    if method == "sort":
                        sample.sort()
                        ordered = sample
                    else:
                        marks = np.zeros(total, dtype=bool)
                        marks[sample] = True
                        ordered = np.flatnonzero(marks)
                    elapsed = time.perf_counter() - start
                    require(np.array_equal(ordered, expected), "ordering changed selection")
                    if trial:
                        times[method].append(elapsed)
            results.append(dict(shape=name, selected=n, density=density,
                                sort_ms=float(np.median(times["sort"]))*1000,
                                mark_ms=float(np.median(times["mark"]))*1000))
    return results


def speed(root):
    from lsb_source import load_photo
    with (root / "data/raise/photo_manifest.csv").open(newline="", encoding="utf-8") as stream:
        photo = next(r for r in csv.DictReader(stream)
                     if r["camera_model"] == "NIKON D7000" and r["source_height"] == "3280")
    results = []
    for name, shape in (("256", (256,256)), ("512", (512,512)), ("D7000", (3280,4948))):
        carrier = (load_photo(photo, root) if name == "D7000" else
                   np.random.default_rng(13).integers(0,256,size=(*shape,3),dtype=np.uint8))
        for placement in ("contiguous", "stride", "random"):
            kwargs = dict(bit_mode="lowest", k=1, method="match", placement=placement,
                          stride=2, bpp=0.5, payload="random_bytes", start=None)
            timings = {"embed": [], "fast": [], "fast_changed": []}
            for trial in range(8):
                for route in timings:
                    rng = np.random.default_rng(17)
                    start = time.perf_counter()
                    if route == "embed":
                        result = engine.embed(carrier, rng=rng, **kwargs)
                    else:
                        result = engine.embed_fast(carrier, rng=rng, return_changed_units=route=="fast_changed", **kwargs)
                    elapsed = time.perf_counter() - start
                    if trial:
                        timings[route].append(elapsed)
                    del result
            results.append(dict(shape=name, placement=placement,
                                source_photo=photo["photo_id"] if name == "D7000" else "synthetic_uint8",
                                **{route: dict(median_ms=float(np.median(t))*1000,
                                              images_per_second=1/float(np.median(t)))
                                   for route,t in timings.items()}))
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--old-engine", type=Path, required=True)
    parser.add_argument("--cases", type=int, default=20000)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    require(args.cases >= 20000, "at least 20,000 cases required")
    report = dict(properties=properties(args.old_engine, args.cases, args.workers))
    print("Property cases passed", flush=True)
    report["legacy"] = legacy(args.repo_root, 500)
    print("Stored legacy sample passed", flush=True)
    cpu = min(os.sched_getaffinity(0))
    os.sched_setaffinity(0, {cpu})
    report.update(benchmark_cpu=cpu, omp=os.environ["OMP_NUM_THREADS"],
                  ordering=ordering_benchmark(), timings=speed(args.repo_root), numpy=np.__version__, status="PASS")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
