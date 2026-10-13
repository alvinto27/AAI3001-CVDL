"""Read-only full source verification on the team VM; no cache or GPU work."""
from __future__ import annotations

import argparse
import csv
import json
import multiprocessing as mp
import os
import sys
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

# Set before importing rawpy, directly or through source/preparation modules.
os.environ["OMP_NUM_THREADS"] = "1"
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "LSB-Dataset-Generator"))
import numpy as np
import PIL
from PIL import Image
import lsb_prepare
from lsb_source import configure_source_worker, grid_axes, iter_tiles, load_photo, topleft_crop


def read_rows(path):
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def check(condition, message):
    if not condition:
        raise ValueError(message)


def verify_one(root_string, record, legacy, coverage):
    root = Path(root_string)
    started = time.perf_counter()
    pixels = load_photo(record, root)
    decoded = time.perf_counter() - started
    clean_count = 0
    if record["source"] == "raise":
        by_position = {(int(t["tile_row"]), int(t["tile_col"])): t for t in legacy}
        check(len(by_position) == len(legacy), "duplicate legacy positions")
        cropped = topleft_crop(pixels, record)
        check(cropped.shape == (int(record["legacy512_height"]), int(record["legacy512_width"]), 3),
              "legacy crop shape mismatch")
        for tile in iter_tiles(pixels, record, "topleft_crop", 512):
            original = by_position.pop((tile.row, tile.column))
            check((tile.y, tile.x) == (int(original["pixel_y"]), int(original["pixel_x"])),
                  "legacy start mismatch")
            with Image.open(root / "Datasets/raise512-run1" / original["clean_path"]) as image:
                check(image.mode == "RGB", "reference clean tile is not RGB")
                reference = np.asarray(image)
            check(reference.dtype == np.uint8 and np.array_equal(tile.pixels, reference),
                  f"clean pixel mismatch: {original['clean_path']}")
            clean_count += 1
        check(not by_position, "unmatched legacy rows")
        if coverage:
            # Preparation retains the same shared numerical path and metadata.
            prepared = lsb_prepare.decode_source(root / record["master_path"])
            check(np.array_equal(prepared.pixels, pixels), "preparation/source pixels differ")
            check((prepared.camera_make, prepared.camera_model) ==
                  (record["camera_make"], record["camera_model"]), "preparation camera metadata differs")
    else:
        with Image.open(root / record["master_path"]) as image:
            reference = np.array(image, copy=True)
        check(reference.dtype == np.uint8 and np.array_equal(reference, pixels), "DIV2K pixel mismatch")
    counts = {}
    for size in (512, 256):
        ys, xs = grid_axes(record, "overlap_last", size)
        check(ys[-1] + size == pixels.shape[0] and xs[-1] + size == pixels.shape[1], "edge mismatch")
        covered = np.zeros(pixels.shape[:2], dtype=bool) if coverage else None
        count = 0
        for tile in iter_tiles(pixels, record, tile_size=size):
            check(tile.pixels.shape == (size, size, 3) and np.shares_memory(tile.pixels, pixels),
                  "tile copied/padded or wrong shape")
            if coverage:
                covered[tile.y:tile.y + size, tile.x:tile.x + size] = True
            count += 1
        check(count == int(record[f"overlap{size}_rows"]) * int(record[f"overlap{size}_cols"]),
              "overlap count mismatch")
        if coverage:
            check(covered.all(), "uncovered real pixels")
        counts[str(size)] = count
    return dict(photo_id=record["photo_id"], source=record["source"], camera=record["camera_model"],
                decode_seconds=decoded, total_seconds=time.perf_counter() - started,
                clean_tiles=clean_count, overlap=counts, coverage=coverage)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    check(1 <= args.workers <= 8, "use 1..8 workers on the team VM")
    root = args.repo_root.resolve()
    raise_rows = read_rows(root / "data/raise/photo_manifest.csv")
    div_rows = read_rows(root / "data/div2k/photo_manifest.csv")
    check(len(raise_rows) == 1000 and len(div_rows) == 900, "wrong source inventory")
    legacy = defaultdict(list)
    for tile in read_rows(root / "data/raise/legacy_tile_manifest.csv"):
        legacy[tile["photo_id"]].append(tile)
    samples = {}
    for row in raise_rows:
        orientation = "landscape" if int(row["source_width"]) > int(row["source_height"]) else "portrait"
        samples.setdefault((row["camera_model"], orientation), row["photo_id"])
    check(len(samples) == 6, "missing camera/orientation coverage sample")
    selected = set(samples.values())
    started = time.perf_counter()
    results = []
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=mp.get_context("spawn"),
                             initializer=configure_source_worker) as executor:
        jobs = {executor.submit(verify_one, str(root), row, legacy.get(row["photo_id"], []),
                                row["photo_id"] in selected): row["photo_id"]
                for row in raise_rows + div_rows}
        for future in as_completed(jobs):
            result = future.result()
            results.append(result)
            if len(results) % 100 == 0:
                print(f"Verified {len(results)}/1900 photos", flush=True)
    wall = time.perf_counter() - started
    summary = dict(status="PASS", workers=args.workers, omp_num_threads=os.environ["OMP_NUM_THREADS"],
                   wall_seconds=wall, coverage_samples={" / ".join(k): v for k, v in sorted(samples.items())},
                   preparation_source_matches=len(selected), sources={}, camera_decode_seconds={})
    for name, expected in (("raise", (1000, 49796, 65184, 243120)), ("div2k", (900, 0, 10960, 42744))):
        rows = [r for r in results if r["source"] == name]
        counts = (len(rows), sum(r["clean_tiles"] for r in rows),
                  sum(r["overlap"]["512"] for r in rows), sum(r["overlap"]["256"] for r in rows))
        check(counts == expected, f"source totals mismatch: {name}/{counts}")
        summary["sources"][name] = dict(photos=counts[0], clean_tiles_equal=counts[1],
                                         overlap512=counts[2], overlap256=counts[3])
    for camera in sorted({r["camera"] for r in results if r["source"] == "raise"}):
        times = [r["decode_seconds"] for r in results if r["camera"] == camera]
        summary["camera_decode_seconds"][camera] = dict(photos=len(times), median=float(np.median(times)),
                                                       p90=float(np.percentile(times, 90)))
    import rawpy
    summary["versions"] = dict(numpy=np.__version__, pillow=PIL.__version__, rawpy=rawpy.__version__,
                                libraw=str(rawpy.libraw_version))
    print(json.dumps(summary, sort_keys=True, indent=2), flush=True)


if __name__ == "__main__":
    main()
