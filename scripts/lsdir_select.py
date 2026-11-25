#!/usr/bin/env python3
"""LSDIR HR subset selection v1: train/val/test, stratified by shard, block-disjoint.

Input : the Pi inventory /srv/datasets/lsdir/inventory.csv (made by scripts/pi/lsdir_extract.py;
        columns shard,path,bytes,width,height,color_type,bit_depth,sha256). Not committed (9.8 MB);
        its SHA-256 is checked below.
Output: data/lsdir/selection_v1.csv

  python3 scripts/lsdir_select.py --inventory inventory.csv [--out data/lsdir/selection_v1.csv]

Rules (seeded by VERSION+SEED+shard+block; independent of row order):
* Quotas TRAIN/VAL/TEST are split over the 17 shards in proportion to shard size (largest
  remainder, seeded tie-break), so every shard gives about 1/17 of each split.
* Leakage rule: in each shard, photos in ID order are cut into BLOCKs of 10 consecutive IDs. A whole
  block goes to one split. Blocks of different splits are never next to each other (an empty
  block separates them, so the smallest ID gap across splits is 11). Reason: neighbouring IDs share
  the exact image size 3.3x more often than random pairs (3.79 % vs 1.14 %, falls to the random
  level by about 50 IDs), and the only byte-identical pair (0021599, 0021600) has consecutive IDs.
* Blocks are chosen in seeded-hash order: test first, then val, then train. The last block of a split
  is trimmed to the quota by seeded hash.
* Duplicate pair: 0022000/0021600.png is never used. 0022000/0021599.png may only be in train
  (a block holding it cannot be val or test).
"""
from __future__ import annotations

import argparse
import csv
import hashlib
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VERSION = "LSDIR_HR_train/selection_v1"
SEED = 0
BLOCK = 10
TRAIN, VAL, TEST = 7000, 1500, 2500
INVENTORY_SHA256 = "31371bb2f01512139a70724794049ca9574274c1b95d20d50790e7825f12b0dc"
INVENTORY_ROWS = 84991
DUP_DROP = "0022000/0021600.png"        # byte-identical to DUP_KEEP
DUP_KEEP = "0022000/0021599.png"        # train only
COLUMNS = ["lsdir_id", "shard", "path", "split", "block", "width", "height", "tiles512", "bytes", "sha256"]
ORDER = ("test", "val", "train")


def h(*parts) -> bytes:
    return hashlib.sha256("\0".join(map(str, (VERSION, SEED) + parts)).encode()).digest()


def apportion(total: int, sizes: dict[str, int], tag: str) -> dict[str, int]:
    whole = sum(sizes.values())
    exact = {s: total * n / whole for s, n in sizes.items()}
    out = {s: int(v) for s, v in exact.items()}
    left = total - sum(out.values())
    for s in sorted(sizes, key=lambda s: (-(exact[s] - out[s]), h(tag, s)))[:left]:
        out[s] += 1
    return out


def pick_shard(shard: str, photos: list[dict], quota: dict[str, int]) -> tuple[dict[str, tuple[str, int]], int]:
    blocks: dict[int, list[dict]] = defaultdict(list)
    for p in photos:
        blocks[(p["num"] - 1) // BLOCK].append(p)
    order = sorted(blocks, key=lambda b: h(shard, "block", b))
    owner: dict[int, str] = {}
    chosen: dict[str, tuple[str, int]] = {}
    fallbacks = 0
    for split in ORDER:
        got = 0
        while got < quota[split]:
            def ok(b: int, strict: bool) -> bool:
                if b in owner:
                    return False
                if split != "train" and any(p["path"] == DUP_KEEP for p in blocks[b]):
                    return False
                return not strict or all(owner.get(x, split) == split for x in (b - 1, b + 1))
            pool = [b for b in order if ok(b, True)]
            if not pool:
                pool = [b for b in order if ok(b, False)]
                fallbacks += 1
            if not pool:
                raise ValueError(f"{shard}: not enough blocks for {split}")
            b = pool[0]
            owner[b] = split
            usable = sorted((p for p in blocks[b] if p["path"] != DUP_DROP), key=lambda p: h(shard, "photo", p["path"]))
            for p in usable[:quota[split] - got]:
                chosen[p["path"]] = (split, b)
                got += 1
    return chosen, fallbacks


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--inventory", required=True, type=Path)
    ap.add_argument("--out", type=Path, default=ROOT / "data/lsdir/selection_v1.csv")
    args = ap.parse_args()
    digest = hashlib.sha256(args.inventory.read_bytes()).hexdigest()
    if digest != INVENTORY_SHA256:
        raise ValueError(f"inventory changed: {digest}")
    rows = list(csv.DictReader(args.inventory.open(newline="")))
    if len(rows) != INVENTORY_ROWS or len({r["path"] for r in rows}) != len(rows):
        raise ValueError("unexpected inventory rows or repeated paths")
    by_shard: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        if (r["color_type"], r["bit_depth"]) != ("2", "8"):
            raise ValueError(f"not RGB 8-bit: {r['path']}")
        r["num"] = int(Path(r["path"]).stem)
        by_shard[r["shard"]].append(r)
    sizes = {s: len(v) for s, v in by_shard.items()}
    quotas = {s: {} for s in sizes}
    for split, total in (("train", TRAIN), ("val", VAL), ("test", TEST)):
        for s, n in apportion(total, sizes, split).items():
            quotas[s][split] = n
    out, fallbacks = [], {}
    for s in sorted(by_shard):
        chosen, fallbacks[s] = pick_shard(s, by_shard[s], quotas[s])
        for r in by_shard[s]:
            if r["path"] in chosen:
                split, b = chosen[r["path"]]
                w, hh = int(r["width"]), int(r["height"])
                out.append([Path(r["path"]).stem, s, r["path"], split, b, w, hh, (w // 512) * (hh // 512),
                            r["bytes"], r["sha256"]])
    out.sort(key=lambda x: x[2])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8", newline="") as f:
        wr = csv.writer(f, lineterminator="\n")
        wr.writerow(COLUMNS)
        wr.writerows(out)
    audit(out, fallbacks, digest, args.out)
    return 0


def audit(out, fallbacks, digest, path) -> None:
    assert len({r[2] for r in out}) == len(out) and len({r[9] for r in out}) == len(out)
    assert DUP_DROP not in {r[2] for r in out}
    print(f"inventory sha256={digest}")
    print("\nSplit | photos | tiles512 | bytes")
    for sp in ("train", "val", "test"):
        sel = [r for r in out if r[3] == sp]
        print(f"{sp} | {len(sel)} | {sum(r[7] for r in sel)} | {sum(int(r[8]) for r in sel)}")
    print(f"all | {len(out)} | {sum(r[7] for r in out)} | {sum(int(r[8]) for r in out)}")
    print("\nShard | train | val | test | min ID gap across splits | fallbacks")
    for s in sorted(fallbacks):
        sel = sorted((int(r[0]), r[3]) for r in out if r[1] == s)
        c = Counter(sp for _, sp in sel)
        gaps = [b[0] - a[0] for a, b in zip(sel, sel[1:]) if a[1] != b[1]]
        print(f"{s} | {c['train']} | {c['val']} | {c['test']} | {min(gaps)} | {fallbacks[s]}")
    dup = [r[3] for r in out if r[2] == DUP_KEEP]
    print(f"\n{DUP_KEEP}: {dup[0] if dup else 'not selected'}; {DUP_DROP}: never selected")
    print(f"selection sha256={hashlib.sha256(path.read_bytes()).hexdigest()}")


if __name__ == "__main__":
    raise SystemExit(main())
