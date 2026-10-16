"""Full legacy settings redraw plus a three-cover production generation check."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import sys
import time
from collections import Counter
from pathlib import Path

os.environ["OMP_NUM_THREADS"] = "1"
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "LSB-Dataset-Generator"))
import numpy as np
from PIL import Image
import lsb_batch as batch
import lsb_engine as engine
import lsb_settings as settings

FIELDS = ("bit_mode", "k", "method", "placement", "start", "start_random", "stride",
          "payload_kind", "payload_bytes", "size_kind", "size_value")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--scratch-root", type=Path, required=True)
    args = parser.parse_args()
    root, scratch = args.repo_root.resolve(), args.scratch_root.resolve()
    if scratch.is_relative_to(root) or scratch.exists():
        raise ValueError("use a NEW scratch directory outside the checkout")
    cfg = batch.read_config(root / "configs/raise512-run1.yaml")
    image = np.zeros((512, 512, 3), dtype=np.uint8)
    mismatches, examples, placements = Counter(), [], Counter()
    rows = 0
    started = time.perf_counter()
    with (root / "Datasets/generated/raise512-run1-records/labels.csv").open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            if row["label"] != "1":
                continue
            seed, index, variant = int(row["seed"]), int(row["cover_index"]), int(row["variant"])
            if seed != cfg.seed:
                raise ValueError("stored seed differs from run1 YAML")
            rng = np.random.default_rng(settings.legacy_seed_sequence(seed, index, variant, 0))
            kwargs, meta = settings.draw_legacy_settings(cfg, image, rng)
            payload_bytes = (int(kwargs["payload_bytes"]) if "payload_bytes" in kwargs
                             else settings._bpp_payload_bytes(kwargs["bpp"], 512 * 512))
            start, stride = None, None
            if kwargs["placement"] != "random":
                embed_rng = np.random.default_rng(settings.legacy_seed_sequence(seed, index, variant, 1))
                # Own engine helpers: advance exactly its payload prefix, then
                # reproduce its placement/start draw. Do not embed/change pixels.
                engine._make_payload(kwargs["payload"], payload_bytes, embed_rng)
                bits_per_unit = kwargs["k"] if kwargs["bit_mode"] == "lowest" else 1
                units = (payload_bytes * 8 + bits_per_unit - 1) // bits_per_unit
                _, start, stride = engine._select_units(image.size, units, kwargs["placement"],
                                                        kwargs["start"], kwargs["stride"], embed_rng)
            actual = {field: kwargs[field] for field in ("bit_mode", "k", "method", "placement")}
            actual.update(start=start, start_random=cfg.start == "random", stride=stride,
                          payload_kind=kwargs["payload"], payload_bytes=payload_bytes,
                          size_kind=meta["size_kind"], size_value=meta["size_value"])
            for field in FIELDS:
                value = str(batch._csv_value(actual[field]))
                if value != row[field]:
                    mismatches[field] += 1
                    if len(examples) < 10:
                        examples.append(dict(cover=index, field=field, expected=row[field], actual=value))
            rows += 1
            placements[row["placement"]] += 1
    if rows != 49796 or mismatches:
        raise ValueError(f"redraw failed: rows={rows}, mismatches={dict(mismatches)}, examples={examples}")
    redraw_seconds = time.perf_counter() - started
    # Two harmless non-image stubs preserve the original discovery indices
    # 3/4/5. No worker/engine/seed code is mocked or replaced.
    covers, output = scratch / "covers", scratch / "output"
    covers.mkdir(parents=True)
    (covers / "done.jsonl").write_text("")
    (covers / "manifest.csv").write_text("")
    samples = []
    with (root / "Datasets/generated/raise512-run1-records/metadata.csv").open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            if row["label"] == "stego" and int(row["cover_index"]) in (3, 4, 5):
                samples.append(row)
    if len(samples) != 3:
        raise ValueError("missing original covers 3/4/5")
    for row in samples:
        shutil.copyfile(root / "Datasets/generated/raise512-run1-records/clean" / f"{row['cover_id']}.png",
                        covers / row["source_file"])
    config = batch.Config(**{**cfg.as_dict(), "input": str(covers), "output": str(output)})
    generated = batch.generate(config, workers=1)
    checks = []
    for row in samples:
        actual_path = output / row["file"]
        reference_path = root / "Datasets/generated/raise512-run1-records" / row["file"]
        with Image.open(actual_path) as actual, Image.open(reference_path) as reference:
            pixels_equal = np.array_equal(np.asarray(actual), np.asarray(reference))
        bytes_equal = actual_path.read_bytes() == reference_path.read_bytes()
        if not pixels_equal or not bytes_equal:
            raise ValueError(f"three-cover identity failed: {row['cover_id']}; pixels={pixels_equal}, PNG={bytes_equal}")
        checks.append(dict(cover_id=row["cover_id"], placement=row["placement"], pixels_equal=True,
                           png_bytes_equal=True, sha256=hashlib.sha256(actual_path.read_bytes()).hexdigest()))
    print(json.dumps(dict(status="PASS", redraw_rows=rows,
                          fields=FIELDS, mismatches={field: 0 for field in FIELDS},
                          placements=dict(placements), redraw_seconds=redraw_seconds,
                          generation=generated, three_covers=checks, numpy_version=np.__version__), indent=2))


if __name__ == "__main__":
    main()
