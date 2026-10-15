#!/usr/bin/env python3
"""ALASKA#2 selection v2: camera-held-out test, seen-camera train/val/test, block-disjoint.

Input : data/alaska/ALASKAv2Base_by_DSC.txt (ID -> camera), data/alaska/selection_v1.csv
        (to prefer photos already on the PC) and a header scan of the Pi download made by
        scripts/alaska_scan.py (file present, SHA-256 from the Pi SHA256SUMS, width, height).
Output: data/alaska/selection_v2.csv

  python3 scripts/alaska_select_v2.py --meta pi_meta.csv pc_v1_meta.csv [--out data/alaska/selection_v2.csv]

Rules (all seeded by VERSION+SEED+model+block, independent of file order):
* Excluded: *_FLATS, *_TEXTURES, RAW_wesaturate (as in v1) and Leica_Mono (monochrome).
* UNSEEN models: every photo of these models can only be test_unseen. UNSEEN_CAP per model,
  v1 photos first.
* Other models ("seen"): the total photo budget TOTAL is spread evenly over the models
  ("water-fill": every model gets the same number, a model with fewer photos gives all it has and
  the remainder goes to the others). Each model total is split train/val/test_seen in the ratio
  TRAIN:VAL:TEST_SEEN.
* Leakage rule: a model's photos, in ID order, are cut into BLOCKs of BLOCK consecutive photos.
  A whole block belongs to one split. Blocks of different splits are never next to each other
  (a block-wide gap stays empty) unless a model is too small to allow it (reported).
* Blocks are chosen by (more v1 photos first, then seeded hash); inside the chosen blocks, v1 photos
  are taken first.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from alaska_select import INDEX, INDEX_SHA256, LINE, ROOT, model_name  # noqa: E402

VERSION = "ALASKA_v2_TIFF_VariousSize_COLOR/selection_v2"
SEED = 0
BLOCK = 10
UNSEEN = ["Google_Pixel-XL", "Samsung_GalaxyS9plus-SM-G965F", "Huawei_P20lite-ANE-LX1",
          "Panasonic_DMC-TZ60", "Ricoh_GRDigital-III", "Canon_EOS-60D", "Sony_A7R", "Pentax_K50"]
DROP_MODELS = {"Leica_Mono"}
UNSEEN_CAP = 250
TRAIN, VAL, TEST_SEEN = 13500, 1000, 1700
TOTAL = TRAIN + VAL + TEST_SEEN
V1 = ROOT / "data/alaska/selection_v1.csv"
BASE_URL = "https://alaska.utt.fr/DATASETS/ALASKA_v2_TIFF_VariousSize_COLOR"
COLUMNS = ["alaska_id", "camera_group", "camera_model", "split", "block", "width", "height",
           "tiles512", "bytes", "sha256", "in_v1", "url"]


def h(*parts) -> bytes:
    return hashlib.sha256("\0".join(map(str, (VERSION, SEED) + parts)).encode()).digest()


def read_index() -> dict[int, tuple[str, str]]:
    digest = hashlib.sha256(INDEX.read_bytes()).hexdigest()
    if digest != INDEX_SHA256:
        raise ValueError(f"index changed: {digest}")
    out = {}
    for line in INDEX.read_text(encoding="utf-8").splitlines()[1:]:
        m = LINE.fullmatch(line)
        if not m:
            raise ValueError(f"bad index line {line!r}")
        first, end, group = int(m[2]), int(m[3]), m[4]
        if group.endswith(("_FLATS", "_TEXTURES")) or group == "RAW_wesaturate":
            continue
        model = model_name(group)
        if model in DROP_MODELS:
            continue
        for n in range(first, end + 1):
            out[n] = (group, model)
    return out


def water_fill(avail: dict[str, int], total: int) -> dict[str, int]:
    lo, hi = 0, max(avail.values())
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if sum(min(a, mid) for a in avail.values()) <= total:
            lo = mid
        else:
            hi = mid - 1
    give = {m: min(a, lo) for m, a in avail.items()}
    extra = total - sum(give.values())
    for m in sorted((m for m in avail if avail[m] > lo), key=lambda m: h("extra", m))[:extra]:
        give[m] += 1
    return give


def split_model(photos: list[dict], quota: dict[str, int], model: str) -> tuple[dict[int, str], dict[int, int], int]:
    """Return (id -> split, id -> block, number of adjacent-block fallbacks)."""
    blocks = [photos[i:i + BLOCK] for i in range(0, len(photos), BLOCK)]
    owner: dict[int, str] = {}
    fallbacks = 0
    for split in ("test_seen", "val", "train"):
        need = quota[split]
        got = 0

        def blocked(b: int) -> bool:
            return any(owner.get(x, split) != split for x in (b - 1, b + 1))

        while got < need:
            free = [b for b in range(len(blocks)) if b not in owner]
            clean = [b for b in free if not blocked(b)]
            pool = clean
            if not pool:
                pool, fallbacks = free, fallbacks + 1
            if not pool:
                break
            best = min(pool, key=lambda b: (-sum(p["in_v1"] for p in blocks[b]), h(model, "block", b)))
            owner[best] = split
            got += len(blocks[best])
    ids_split, ids_block = {}, {}
    for split in ("test_seen", "val", "train"):
        mine = [(b, p) for b, bl in enumerate(blocks) if owner.get(b) == split for p in bl]
        mine.sort(key=lambda bp: (not bp[1]["in_v1"], h(model, "photo", bp[1]["id"])))
        for b, p in mine[:quota[split]]:
            ids_split[p["id"]], ids_block[p["id"]] = split, b
    return ids_split, ids_block, fallbacks


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--meta", required=True, type=Path, nargs="+",
                    help="alaska_scan.py CSVs: Pi download (without v1) and PC copy of v1")
    ap.add_argument("--allow-partial", action="store_true", help="testing only: skip unseen models with no photos")
    ap.add_argument("--out", type=Path, default=ROOT / "data/alaska/selection_v2.csv")
    args = ap.parse_args()
    index = read_index()
    v1 = {int(r["alaska_id"]) for r in csv.DictReader(V1.open(newline=""))}
    meta, missing_ids = {}, []
    rows_in = [r for path in args.meta for r in csv.DictReader(path.open(newline=""))]
    if len({r["alaska_id"] for r in rows_in}) != len(rows_in):
        raise ValueError("an ID occurs in two --meta files")
    for r in rows_in:
        n = int(r["alaska_id"])
        if n in index:
            if (r["bits"], r["compression"], r["photometric"], r["samples"]) != ("8/8/8", "1", "2", "3"):
                raise ValueError(f"{n}: not uncompressed RGB8: {r}")
            meta[n] = r
    for n in sorted(index):
        if n not in meta:
            missing_ids.append(n)
    by_model: dict[str, list[dict]] = defaultdict(list)
    for n in sorted(meta):
        r = meta[n]
        by_model[index[n][1]].append({"id": n, "in_v1": n in v1, "w": int(r["width"]), "h": int(r["height"])})
    absent = [m for m in UNSEEN if m not in by_model]
    if absent and not args.allow_partial:
        raise ValueError(f"UNSEEN models with no photos: {absent}")
    unseen_use = [m for m in UNSEEN if m in by_model]
    seen = sorted(m for m in by_model if m not in UNSEEN)
    rows = []
    # Unseen: v1 photos first, then the seeded hash.
    unseen_split = {}
    for m in unseen_use:
        pick = sorted(by_model[m], key=lambda p: (not p["in_v1"], h(m, "unseen", p["id"])))[:UNSEEN_CAP]
        for p in pick:
            unseen_split[p["id"]] = "test_unseen"
    # Seen.
    share = water_fill({m: len(by_model[m]) for m in seen}, TOTAL)
    split_of, block_of, fallback = dict(unseen_split), {}, {}
    quotas = {}
    for m in seen:
        t = share[m]
        val = round(t * VAL / TOTAL)
        tst = round(t * TEST_SEEN / TOTAL)
        q = {"val": val, "test_seen": tst, "train": t - val - tst}
        quotas[m] = q
        s, b, fb = split_model(by_model[m], q, m)
        split_of.update(s)
        block_of.update(b)
        fallback[m] = fb
    for n in sorted(split_of):
        group, model = index[n]
        r = meta[n]
        w, hh = int(r["width"]), int(r["height"])
        rows.append([f"{n:05d}", group, model, split_of[n], block_of.get(n, ""), w, hh,
                     (w // 512) * (hh // 512), r["bytes"], r["sha256"], int(n in v1), f"{BASE_URL}/{n:05d}.tif"])
    rows.sort(key=lambda x: (x[2], x[0]))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8", newline="") as f:
        wr = csv.writer(f, lineterminator="\n")
        wr.writerow(COLUMNS)
        wr.writerows(rows)
    audit(rows, by_model, quotas, fallback, missing_ids, args.meta)
    return 0


def audit(rows, by_model, quotas, fallback, missing_ids, meta_path) -> None:
    assert len({r[0] for r in rows}) == len(rows), "duplicate IDs"
    models = defaultdict(lambda: defaultdict(lambda: [0, 0, 0, 0]))   # model -> split -> photos,tiles,bytes,v1
    for r in rows:
        c = models[r[2]][r[3]]
        c[0] += 1; c[1] += r[7]; c[2] += int(r[8]); c[3] += r[10]
    # Leakage audit: nearest ID of another split inside the same model.
    min_gap = {}
    for m in models:
        ids = sorted((int(r[0]), r[3]) for r in rows if r[2] == m)
        gaps = [b[0] - a[0] for a, b in zip(ids, ids[1:]) if a[1] != b[1]]
        min_gap[m] = min(gaps) if gaps else None
    for p in meta_path:
        print(f"meta={p.name} sha256={hashlib.sha256(p.read_bytes()).hexdigest()}")
    print(f"eligible present={sum(len(v) for v in by_model.values())} missing eligible IDs={len(missing_ids)}")
    print("\nSplit | photos | tiles512 | GB | in_v1")
    tot = Counter()
    for sp in ("train", "val", "test_seen", "test_unseen"):
        c = [sum(models[m][sp][i] for m in models if sp in models[m]) for i in range(4)]
        print(f"{sp} | {c[0]} | {c[1]} | {c[2]/1e9:.2f} | {c[3]}")
        tot.update(dict(enumerate(c)))
    print(f"ALL | {tot[0]} | {tot[1]} | {tot[2]/1e9:.2f} | {tot[3]}")
    print("\nModel | avail | train(tiles) | val | test_seen | test_unseen | min ID gap across splits | adjacent fallbacks | note")
    for m in sorted(models):
        g = models[m]
        cells = [f"{g[s][0]}({g[s][1]})" if s == "train" else str(g[s][0]) for s in ("train", "val", "test_seen", "test_unseen")]
        note = ""
        if m in quotas:
            q = quotas[m]
            short = {s: q[s] - g[s][0] for s in q if g[s][0] < q[s]}
            note = f"SHORT {short}" if short else ""
            if len(by_model[m]) < 560:
                note += f" small model (all {len(by_model[m])} photos in play)"
        print(f"{m} | {len(by_model[m])} | {' | '.join(cells)} | {min_gap[m]} | {fallback.get(m, '-')} | {note}")
    if missing_ids:
        print(f"\nMissing eligible IDs (first 20): {missing_ids[:20]}")


if __name__ == "__main__":
    raise SystemExit(main())
