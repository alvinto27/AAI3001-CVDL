#!/usr/bin/env python3
"""Resumable LSDIR extractor + inventory for the Pi (standard library only).

Per shard: stream `gzip -dc | tar`, write each PNG, hash it, read the PNG
header. A shard is done only when the whole archive is read (gzip CRC ok),
every file has the size from its tar header, and the shard CSV is written.
Then `done/<shard>.json` is written. A restart redoes only shards without it.
"""
from __future__ import annotations

import csv
import datetime as dt
import hashlib
import json
import os
import shutil
import statistics
import struct
import subprocess
import sys
import tarfile
import time
from collections import Counter, defaultdict
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from pathlib import Path

from common import setup, status

ROOT = Path('/srv/datasets/lsdir')
WORK = ROOT / 'extract'            # STATUS, log, per-shard CSV, done markers
DONE = WORK / 'done'
PART = WORK / 'csv'
PROG = WORK / 'progress'
OUT = {'hr': ROOT / 'hr', 'val1': ROOT / 'val1'}
SHARDS = [f'shard-{i:02d}' for i in range(17)] + ['val1']
EXPECTED_TRAIN = 84991
WORKERS = 2
MIN_FREE = 100 * 1024**3
HEAD = ['shard', 'path', 'bytes', 'width', 'height', 'color_type', 'bit_depth', 'sha256']
START = dt.datetime.now(dt.timezone.utc)
PNG_SIG = b'\x89PNG\r\n\x1a\n'


class DiskGuard(RuntimeError):
    pass


def guard(extra: int = 0) -> None:
    free = shutil.disk_usage(ROOT).free
    if free < MIN_FREE + extra:
        raise DiskGuard(f'DISK_GUARD: free {free} B is below 100 GiB reserve (+{extra} B incoming)')


def png_header(head: bytes) -> tuple[str, str, str, str]:
    if len(head) >= 26 and head[:8] == PNG_SIG and head[12:16] == b'IHDR':
        w, h, depth, ctype = struct.unpack('>IIBB', head[16:26])
        return str(w), str(h), str(ctype), str(depth)
    return '', '', '', ''


def safe_rel(name: str, strip: str) -> str | None:
    parts = [p for p in name.split('/') if p not in ('', '.')]
    if strip and parts and parts[0] == strip:
        parts = parts[1:]
    if not parts or '..' in parts or name.startswith('/'):
        return None
    return '/'.join(parts)


def extract_shard(shard: str) -> dict:
    """Run in a worker process. Raises on any problem; returns a summary dict."""
    kind = 'val1' if shard == 'val1' else 'hr'
    strip = 'val1' if kind == 'val1' else ''   # val1 archive has a redundant top 'val1/'
    dest = OUT[kind]
    dest.mkdir(parents=True, exist_ok=True)
    os.nice(0)
    guard(15 * 1024**3 if kind == 'hr' else 2 * 1024**3)
    t0 = time.time()
    rows, nbytes, n = [], 0, 0
    seen = set()
    proc = subprocess.Popen(['gzip', '-dc', str(ROOT / f'{shard}.tar.gz')],
                            stdout=subprocess.PIPE, bufsize=1 << 20)
    try:
        with tarfile.open(fileobj=proc.stdout, mode='r|') as tar:
            for member in tar:
                if member.isdir():
                    rel = safe_rel(member.name, strip)
                    if rel:
                        (dest / rel).mkdir(parents=True, exist_ok=True)
                    continue
                if not member.isreg():
                    raise ValueError(f'{shard}: unexpected non-regular member {member.name}')
                rel = safe_rel(member.name, strip)
                if rel is None or rel in seen:
                    raise ValueError(f'{shard}: bad or repeated path {member.name}')
                seen.add(rel)
                target = dest / rel
                target.parent.mkdir(parents=True, exist_ok=True)
                tmp = target.with_name(target.name + '.part')
                h = hashlib.sha256()
                head = b''
                size = 0
                src = tar.extractfile(member)
                with tmp.open('wb') as out:
                    for block in iter(lambda: src.read(1 << 20), b''):
                        if len(head) < 32:
                            head = (head + block)[:32]
                        h.update(block)
                        out.write(block)
                        size += len(block)
                if size != member.size:
                    raise ValueError(f'{shard}: {rel} wrote {size} B, header says {member.size} B')
                os.replace(tmp, target)
                if target.stat().st_size != size:
                    raise ValueError(f'{shard}: {rel} size changed after write')
                w, hh, ct, bd = png_header(head)
                rows.append([shard, rel, size, w, hh, ct, bd, h.hexdigest()])
                nbytes += size
                n += 1
                if n % 200 == 0:
                    guard()
                    (PROG / f'{shard}.json').write_text(json.dumps({'images': n, 'bytes': nbytes}))
        while proc.stdout.read(1 << 20):   # drain tar padding so gzip can check its CRC
            pass
    except BaseException:
        proc.kill()
        raise
    finally:
        proc.stdout.close()
        rc = proc.wait()
    if rc != 0:
        raise RuntimeError(f'{shard}: gzip exit {rc} (archive damaged?)')
    os.sync()
    tmp_csv = PART / f'{shard}.csv.tmp'
    with tmp_csv.open('w', newline='') as f:
        wr = csv.writer(f)
        wr.writerow(HEAD)
        wr.writerows(rows)
    os.replace(tmp_csv, PART / f'{shard}.csv')
    info = {'shard': shard, 'images': n, 'bytes': nbytes, 'seconds': round(time.time() - t0, 1),
            'finished': dt.datetime.now(dt.timezone.utc).isoformat()}
    tmp_done = DONE / f'{shard}.json.tmp'
    tmp_done.write_text(json.dumps(info))
    os.replace(tmp_done, DONE / f'{shard}.json')       # done marker comes last
    (PROG / f'{shard}.json').unlink(missing_ok=True)
    return info


def done_info() -> dict[str, dict]:
    return {p.stem: json.loads(p.read_text()) for p in DONE.glob('*.json')}


def progress(log, state: str, note: str = '') -> None:
    done = done_info()
    imgs = sum(d['images'] for d in done.values())
    nb = sum(d['bytes'] for d in done.values())
    for p in PROG.glob('*.json'):
        try:
            d = json.loads(p.read_text())
            imgs += d['images']
            nb += d['bytes']
        except (OSError, ValueError):
            pass
    status(WORK, state=state, done=imgs, total=EXPECTED_TRAIN + 250, bytes_=nb, failures=0,
           started=START, note=f'shards_done={len(done)}/{len(SHARDS)} {note}'.strip(),
           baseline_bytes=BASE[0])


BASE = [0]


def merge() -> None:
    for name, shards, dest in (('inventory.csv', SHARDS[:-1], ROOT / 'inventory.csv'),
                               ('inventory_val1.csv', ['val1'], ROOT / 'inventory_val1.csv')):
        tmp = dest.with_suffix('.tmp')
        with tmp.open('w', newline='') as out:
            wr = csv.writer(out)
            wr.writerow(HEAD)
            for s in shards:
                with (PART / f'{s}.csv').open(newline='') as f:
                    rd = csv.reader(f)
                    next(rd)
                    wr.writerows(rd)
        os.replace(tmp, dest)


def pct(vals, q):
    vals = sorted(vals)
    return vals[min(len(vals) - 1, int(q * len(vals)))]


def report() -> str:
    lines = []
    for title, path in (('HR train', ROOT / 'inventory.csv'), ('val1', ROOT / 'inventory_val1.csv')):
        rows = list(csv.DictReader(path.open(newline='')))
        paths = Counter(r['path'] for r in rows)
        dup_paths = {k: v for k, v in paths.items() if v > 1}
        by_sha = defaultdict(list)
        for r in rows:
            by_sha[r['sha256']].append(r['path'])
        dup_sha = {k: v for k, v in by_sha.items() if len(v) > 1}
        non_png = [r for r in rows if not r['width']]
        not_rgb8 = [r for r in rows if r['width'] and (r['color_type'], r['bit_depth']) != ('2', '8')]
        ok = [r for r in rows if r['width']]
        ws = [int(r['width']) for r in ok]
        hs = [int(r['height']) for r in ok]
        big = sum(1 for w, h in zip(ws, hs) if w >= 512 and h >= 512)
        lines += [f'## {title}', '', '| item | value |', '|---|---|',
                  f'| images | {len(rows)} |', f'| total bytes | {sum(int(r["bytes"]) for r in rows)} |',
                  f'| duplicate paths | {len(dup_paths)} |',
                  f'| duplicate SHA-256 values (groups) | {len(dup_sha)} |',
                  f'| files in duplicate groups | {sum(len(v) for v in dup_sha.values())} |',
                  f'| non-PNG files | {len(non_png)} |',
                  f'| PNG but not RGB 8-bit | {len(not_rgb8)} |',
                  f'| both sides >= 512 | {big} |']
        if ok:
            for nm, v in (('width', ws), ('height', hs)):
                lines.append(f'| {nm} min / median / max | {min(v)} / {int(statistics.median(v))} / {max(v)} |')
        kinds = Counter((r['color_type'], r['bit_depth']) for r in ok)
        lines += [f'| colour type, bit depth counts | {dict(kinds)} |', '']
        if title == 'HR train':
            lines.append(f'Expected 84,991: {"MATCH" if len(rows) == EXPECTED_TRAIN else "MISMATCH"} '
                         f'(diff {len(rows) - EXPECTED_TRAIN:+d})')
            tj = ROOT / 'train.json'
            if tj.is_file():
                lines.append(compare_train(tj, rows))
            lines.append('')
        for k, v in list(dup_sha.items())[:10]:
            lines.append(f'- dup sha {k[:12]}: {v[:4]}')
        for r in (non_png + not_rgb8)[:10]:
            lines.append(f'- odd file: {r["path"]} type={r["color_type"]} depth={r["bit_depth"]}')
        lines.append('')
    return '\n'.join(lines)


def compare_train(tj: Path, rows) -> str:
    try:
        data = json.loads(tj.read_text())
        items = data if isinstance(data, list) else next(iter(data.values()))
        names = {str(x).lstrip('./') for x in items}
        have = {r['path'] for r in rows}
        stems = {Path(n).with_suffix('').as_posix() for n in names}
        hstems = {Path(p).with_suffix('').as_posix() for p in have}
        return (f'train.json entries {len(names)}; missing on disk {len(stems - hstems)}; '
                f'on disk not listed {len(hstems - stems)}')
    except Exception as exc:  # noqa: BLE001 - report text only
        return f'train.json compare failed: {exc}'


def main() -> int:
    for d in (WORK, DONE, PART, PROG):
        d.mkdir(parents=True, exist_ok=True)
    log = setup(WORK, 'lsdir_extract')
    todo = [s for s in SHARDS if not (DONE / f'{s}.json').is_file()]
    BASE[0] = sum(d['bytes'] for d in done_info().values())
    log.info('start; %d shards to do: %s', len(todo), todo)
    try:
        guard(2 * 15 * 1024**3)
    except DiskGuard as exc:
        log.error('%s', exc)
        progress(log, 'disk_guard', str(exc))
        return 78
    failed = False
    if todo:
        with ProcessPoolExecutor(max_workers=WORKERS) as pool:
            queue, active = list(todo), {}
            while queue or active:
                while queue and len(active) < WORKERS and not failed:
                    s = queue.pop(0)
                    active[pool.submit(extract_shard, s)] = s
                if not active:
                    break
                fin, _ = wait(active, timeout=30, return_when=FIRST_COMPLETED)
                progress(log, 'running', f'active={sorted(active.values())}')
                for fut in fin:
                    s = active.pop(fut)
                    try:
                        info = fut.result()
                        log.info('done %s', info)
                    except DiskGuard as exc:
                        log.error('%s', exc)
                        progress(log, 'disk_guard', str(exc))
                        failed = True
                        code = 78
                    except Exception as exc:  # noqa: BLE001
                        log.error('FAILED %s: %r', s, exc)
                        failed = True
                        code = 1
                if failed:
                    queue.clear()
        if failed:
            progress(log, 'failed', 'see lsdir_extract.log; restart redoes unfinished shards')
            return code
    merge()
    text = report()
    (WORK / 'REPORT.md').write_text(text)
    progress(log, 'finished', 'inventory.csv, inventory_val1.csv, extract/REPORT.md written')
    log.info('finished')
    print(text)
    return 0


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == 'report':
        print(report())
        sys.exit(0)
    sys.exit(main())
