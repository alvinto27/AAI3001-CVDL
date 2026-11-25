#!/usr/bin/env python3
"""List ALASKA TIFF facts from file headers only (no pixel decoding).

Reads width, height, bits, compression, colour model and channel count from the
first TIFF directory, and the SHA-256 from a SHA256SUMS file. Standard library only,
so it runs on the Pi and on the PC.

  python3 alaska_scan.py DIR SHA256SUMS OUT.csv [--verify] [--only IDS.txt]

--verify re-hashes each file and keeps only files whose hash equals SHA256SUMS.
--only restricts the scan to the 5-digit IDs listed in IDS.txt (one per line,
or the first CSV column of a file with a header line starting "alaska_id").
Files without a SHA256SUMS line (also *.part) are skipped and counted.
"""
from __future__ import annotations

import csv
import hashlib
import re
import struct
import sys
from pathlib import Path

HEAD = ["alaska_id", "width", "height", "bits", "compression", "photometric", "samples", "bytes", "sha256"]
SIZES = {1: 1, 3: 2, 4: 4}
FMT = {1: "B", 3: "H", 4: "I"}


def tiff_facts(path: Path) -> dict[int, tuple]:
    with path.open("rb") as f:
        head = f.read(8)
        end = {b"II": "<", b"MM": ">"}.get(head[:2])
        if end is None or struct.unpack(end + "H", head[2:4])[0] != 42:
            raise ValueError("not a classic TIFF")
        f.seek(struct.unpack(end + "I", head[4:8])[0])
        (count,) = struct.unpack(end + "H", f.read(2))
        tags = {}
        for _ in range(count):
            tag, typ, n, val = struct.unpack(end + "HHI4s", f.read(12))
            if tag in (256, 257, 258, 259, 262, 277) and typ in SIZES:
                size = SIZES[typ] * n
                if size <= 4:
                    raw = val[:size]
                else:
                    pos = f.tell()
                    f.seek(struct.unpack(end + "I", val)[0])
                    raw = f.read(size)
                    f.seek(pos)
                tags[tag] = struct.unpack(end + FMT[typ] * n, raw)
        return tags


def main() -> int:
    folder, sums, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
    verify = "--verify" in sys.argv[4:]
    only = None
    if "--only" in sys.argv:
        lines = Path(sys.argv[sys.argv.index("--only") + 1]).read_text().splitlines()
        only = {x.split(",")[0].strip() for x in lines if re.fullmatch(r"\d{5}(,.*)?", x.strip())}
    known = {}
    for line in sums.read_text().splitlines():
        m = re.fullmatch(r"([a-f0-9]{64})  (?:.*/)?([0-9]{5})\.tif", line)
        if m:
            known[m[2]] = m[1]
    rows, skipped, bad = [], 0, []
    for ident in sorted(known):
        if only is not None and ident not in only:
            continue
        path = folder / f"{ident}.tif"
        if not path.is_file():
            skipped += 1
            continue
        try:
            if verify:
                h = hashlib.sha256()
                with path.open("rb") as f:
                    for block in iter(lambda: f.read(1 << 20), b""):
                        h.update(block)
                if h.hexdigest() != known[ident]:
                    bad.append(ident)
                    continue
            t = tiff_facts(path)
            rows.append([ident, t[256][0], t[257][0], "/".join(map(str, t[258])), t[259][0],
                         t[262][0], t[277][0], path.stat().st_size, known[ident]])
        except (ValueError, KeyError, struct.error, OSError) as exc:
            bad.append(f"{ident}:{exc}")
    tmp = out.with_suffix(".tmp")
    with tmp.open("w", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(HEAD)
        w.writerows(rows)
    tmp.replace(out)
    print(f"rows={len(rows)} listed_but_missing={skipped} bad={len(bad)} {bad[:5]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
