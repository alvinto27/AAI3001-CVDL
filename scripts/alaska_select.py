#!/usr/bin/env python3
"""Freeze an even, reproducible ALASKA#2 TIFF selection from the camera ID index.

The ID ranges are authoritative; the publisher's count column has two errors.
Run from any directory: python scripts/alaska_select.py
"""
from __future__ import annotations

import csv
import hashlib
import re
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INDEX = ROOT / "data/alaska/ALASKAv2Base_by_DSC.txt"
OUTPUT = ROOT / "data/alaska/selection_v1.csv"
INDEX_SHA256 = "cd4bf8b9a82c8422f86c94d82966c6a423ba2e4748667aa970777fc7b0e91a54"
VERSION = "ALASKA_v2_TIFF_VariousSize_COLOR/selection_v1"
SEED = 0
CAP = 250
BASE_URL = "https://alaska.utt.fr/DATASETS/ALASKA_v2_TIFF_VariousSize_COLOR"
LINE = re.compile(r"\s*(\d+)\s*::\s*(\d{5})\s*-->\s*(\d{5})\s+\./([\w-]+)\s*")


def model_name(group: str) -> str:
    # All suffixes here identify the same camera model; retain the exact group
    # in each CSV row for a future camera-held-out split.
    name = re.sub(r"_Lens[12]$|_device[123]$", "", group)
    if name in {"Pentax_K10D_DNG", "Pentax_K10D_PEF"}:
        return "Pentax_K10D"
    return name


def selected_ids(ids: list[int], model: str) -> list[int]:
    if len(ids) <= CAP:
        return ids
    # One ID from each equal-population bin. The hash only breaks choices
    # within that bin; thus picks span each model's entire ordered ID list.
    chosen = []
    for k in range(CAP):
        group = ids[k * len(ids) // CAP : (k + 1) * len(ids) // CAP]
        chosen.append(min(group, key=lambda number: (
            hashlib.sha256(f"{VERSION}\0{SEED}\0{model}\0{number:05d}".encode()).digest(), number
        )))
    return chosen


def main() -> None:
    digest = hashlib.sha256(INDEX.read_bytes()).hexdigest()
    if digest != INDEX_SHA256:
        raise ValueError(f"index changed: {digest}; expected {INDEX_SHA256}")
    by_model: dict[str, list[int]] = defaultdict(list)
    original_group: dict[int, str] = {}
    excluded: Counter[str] = Counter()
    mismatches = []
    last = 0
    for line in INDEX.read_text(encoding="utf-8").splitlines()[1:]:
        match = LINE.fullmatch(line)
        if not match:
            raise ValueError(f"invalid camera range: {line!r}")
        stated, first, end = (int(match[i]) for i in (1, 2, 3))
        group = match[4]
        if first != last + 1 or end < first:
            raise ValueError(f"gap/overlap in ID ranges: {line!r}")
        last = end
        if stated != end - first + 1:
            mismatches.append((group, stated, end - first + 1))
        if group.endswith(("_FLATS", "_TEXTURES")):
            excluded[f"{group}: calibration flat/texture, not a natural photograph"] += end - first + 1
            continue
        if group == "RAW_wesaturate":
            excluded[f"{group}: mixed raw processing, no single camera identity"] += end - first + 1
            continue
        model = model_name(group)
        for number in range(first, end + 1):
            by_model[model].append(number)
            original_group[number] = group
    if last != 80005 or len(by_model) != 39 or sum(excluded.values()) != 6153:
        raise ValueError(f"unexpected inventory: last={last}, models={len(by_model)}, excluded={sum(excluded.values())}")
    if mismatches != [("Canon_EOS-60D", 3678, 2750), ("Canon_EOS-500D", 1872, 2800)]:
        raise ValueError(f"unexpected publisher count mismatches: {mismatches}")
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    counts: Counter[str] = Counter()
    with OUTPUT.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(("alaska_id", "camera_group", "camera_model", "url"))
        for model in sorted(by_model):
            for number in selected_ids(by_model[model], model):
                image_id = f"{number:05d}"
                writer.writerow((image_id, original_group[number], model, f"{BASE_URL}/{image_id}.tif"))
                counts[model] += 1
    print(f"Index SHA-256: {digest}; version={VERSION}, seed={SEED}, cap={CAP}")
    print("Publisher count mismatches (declared, ID-range count):", mismatches)
    print("Excluded (group, IDs, reason):")
    for label, count in sorted(excluded.items()):
        print(f"  {label}: {count}")
    print("Camera model | available | selected")
    for model in sorted(by_model):
        print(f"{model} | {len(by_model[model])} | {counts[model]}")
    print(f"TOTAL models={len(by_model)} eligible={sum(map(len, by_model.values()))} "
          f"excluded={sum(excluded.values())} selected={sum(counts.values())}")
    print(f"Selection SHA-256: {hashlib.sha256(OUTPUT.read_bytes()).hexdigest()}")
    print("No train/val/test assignment. Nikon_D90 also occurs in RAISE; Leica_Mono has a monochrome sensor.")


if __name__ == "__main__":
    main()
