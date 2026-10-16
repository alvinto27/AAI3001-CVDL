"""CPU-only Mode 1 replay gates and OTF identity checks; no dataset writes."""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import multiprocessing as mp
import os
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

os.environ["OMP_NUM_THREADS"] = "1"
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "LSB-Dataset-Generator"))
import numpy as np
from PIL import Image
from lsb_batch import Config, read_config
from lsb_source import load_photo, configure_source_worker, grid_axes
from lsb_mode1 import TileSpec, OTFSeed, make_sample, replay_legacy


def rows(path):
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def check(ok, message):
    if not ok:
        raise ValueError(message)


def orientation(record):
    return "landscape" if int(record["source_width"]) > int(record["source_height"]) else "portrait"


def verify_photo(root, record, tiles, configuration):
    root = Path(root)
    cfg = Config(**configuration)
    start = time.perf_counter()
    photo = load_photo(record, root)
    decode = time.perf_counter() - start
    generation = comparison = 0.0
    placements = Counter()
    for row in tiles:
        start = time.perf_counter()
        clean, _ = replay_legacy(photo, record, row, cfg, "clean")
        stego, metadata = replay_legacy(photo, record, row, cfg, "stego")
        generation += time.perf_counter() - start
        start = time.perf_counter()
        for kind, actual in (("clean", clean), ("stego", stego)):
            with Image.open(root / "Datasets/generated/raise512-run1-records" / row[f"{kind}_path"]) as image:
                check(image.mode == "RGB" and np.array_equal(actual, np.asarray(image)),
                      f"STOP: {kind} mismatch {row['cover_id']} photo {row['photo_id']}")
        comparison += time.perf_counter() - start
        placements[metadata["embedding"]["placement"]] += 1
    return dict(photo_id=record["photo_id"], pairs=len(tiles), decode=decode,
                generation=generation, comparison=comparison, placements=dict(placements),
                camera_orientation=f"{record['camera_model']}/{orientation(record)}")


def replay(root, photos, tiles, cfg, workers):
    grouped = defaultdict(list)
    for tile in tiles:
        grouped[tile["photo_id"]].append(tile)
    start = time.perf_counter()
    results = []
    with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("spawn"),
                             initializer=configure_source_worker) as pool:
        futures = [pool.submit(verify_photo, str(root), photo, grouped[photo["photo_id"]], cfg.as_dict())
                   for photo in photos if photo["photo_id"] in grouped]
        for future in as_completed(futures):
            results.append(future.result())
            if len(results) % 100 == 0:
                print(f"Verified {len(results)}/{len(futures)} photos", flush=True)
    wall = time.perf_counter() - start
    placement = Counter()
    for result in results:
        placement.update(result["placements"])
    return dict(pairs=sum(r["pairs"] for r in results), photos=len(results), mismatches=0,
                wall_seconds=wall, worker_seconds={f:sum(r[f] for r in results)
                    for f in ("decode", "generation", "comparison")}, placements=dict(placement),
                camera_orientation=sorted({r["camera_orientation"] for r in results}), workers=workers)


def signature(sample):
    pixels, meta = sample
    return hashlib.sha256(pixels.tobytes()).hexdigest(), json.dumps(meta, sort_keys=True)


def otf_photo(job):
    root, record = job
    photo = load_photo(record, Path(root))
    cfg = Config(seed=1, input="unused", output="unused", bit_mode=["lowest", "plane"], k=[1,2,3,4,5,6,7,8], method=["replace", "match"],
                 placement=["random", "contiguous", "stride"], stride=[1,8], size={"bpp":[0.05,0.5]})
    outputs, cases = {}, 0
    for size in (512,256):
        ys, xs = grid_axes(record, "overlap_last", size)
        spec = TileSpec("overlap_last", size, len(ys)-1, len(xs)-1)
        for purpose in ("train", "val", "test"):
            base = signature(make_sample(photo, record, spec, cfg, OTFSeed(7,purpose,0), "stego"))
            repeated = signature(make_sample(photo, record, spec, cfg, OTFSeed(7,purpose,0), "stego"))
            changed_seed = signature(make_sample(photo, record, spec, cfg, OTFSeed(8,purpose,0), "stego"))
            changed_epoch = signature(make_sample(photo, record, spec, cfg, OTFSeed(7,purpose,1), "stego"))
            check(base == repeated, "repeat changed")
            if purpose == "train":
                check(base[0] != changed_seed[0] and base[0] != changed_epoch[0], "train seed/epoch unchanged")
            else:
                check(base == changed_seed == changed_epoch, "evaluation seed/epoch affected output")
            outputs[f"{size}/{purpose}"] = base
            cases += 1
        fixed = [signature(make_sample(photo, record, spec, cfg, OTFSeed(7,"train",e,fixed_variants=True), "stego"))
                 for e in (0,1,5,100)]
        check(all(s == fixed[0] for s in fixed), "fixed variants changed")
        overflow = Config(seed=1, input="unused", output="unused", bit_mode="plane", k=1, method="replace", placement="stride", stride=1000,
                          size={"bpp":10.0})
        _, meta = make_sample(photo, record, spec, overflow, OTFSeed(7,"train",0), "stego")
        info = meta["settings"]
        check(info["clipped"] and info["effective_bpp"] < info["requested_bpp"], "lower-to-fit not recorded")
        outputs[f"{size}/clipped"] = (info["requested_bpp"], info["effective_bpp"], info["payload_bytes"])
    return record["photo_id"], outputs, cases


def otf(root, photos, div):
    chosen = {}
    for record in photos:
        chosen.setdefault((record["camera_model"], orientation(record)), record)
    jobs = [(str(root), r) for r in list(chosen.values()) + div[:2]]
    start = time.perf_counter()
    first = {r[0]:r[1:] for r in map(otf_photo, jobs)}
    second = {r[0]:r[1:] for r in map(otf_photo, reversed(jobs))}
    with ProcessPoolExecutor(max_workers=4, mp_context=mp.get_context("spawn"), initializer=configure_source_worker) as pool:
        parallel = {r[0]:r[1:] for r in pool.map(otf_photo, reversed(jobs))}
    check(first == second == parallel, "call order/process count changed samples")
    return dict(photos=len(jobs), sizes=[512,256], purpose_cases=sum(v[1] for v in first.values()),
                fixed_epoch_values=[0,1,5,100], clipping_cases=len(jobs)*2,
                clipped_examples={k:{s:v[0][s] for s in ("512/clipped","256/clipped")} for k,v in first.items()},
                serial_orders=2, parallel_workers=4, mismatch_count=0, wall_seconds=time.perf_counter()-start)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo-root", type=Path, required=True)
    p.add_argument("--gate", choices=["sample","full","otf"], required=True)
    args = p.parse_args()
    root = args.repo_root.resolve()
    photos = rows(root / "data/raise/photo_manifest.csv")
    tiles = rows(root / "data/raise/legacy_tile_manifest.csv")
    cfg = read_config(root / "configs/raise512-run1.yaml")
    if args.gate == "otf":
        report = otf(root, photos, rows(root / "data/div2k/photo_manifest.csv"))
    else:
        if args.gate == "sample":
            seen, selected_photos = set(), set()
            # Start with all six camera/orientation groups; avoid 500 RAW decodes.
            for photo in photos:
                key = photo["camera_model"], orientation(photo)
                if key not in seen:
                    selected_photos.add(photo["photo_id"])
                    seen.add(key)
            selected = [r for r in tiles if r["photo_id"] in selected_photos]
            for photo in photos:
                if len(selected) >= 500:
                    break
                if photo["photo_id"] not in selected_photos:
                    selected_photos.add(photo["photo_id"])
                    selected.extend(r for r in tiles if r["photo_id"] == photo["photo_id"])
            # Preserve mandatory groups even when the last photo is partial.
            tiles = selected[:500]
        report = replay(root, photos, tiles, cfg, 8)
        check(report["pairs"] == (500 if args.gate == "sample" else 49796), "wrong pair count")
        check(len(report["camera_orientation"]) == 6 and set(report["placements"]) == {"random","contiguous","stride"},
              "coverage missing")
    print(json.dumps(dict(status="PASS", gate=args.gate, **report), indent=2), flush=True)


if __name__ == "__main__":
    main()
