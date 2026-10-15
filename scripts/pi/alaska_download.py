#!/usr/bin/env python3
"""Unattended Pi-only ALASKA remainder, excluding the selected PC subset."""
from __future__ import annotations

import csv
import datetime as dt
import hashlib
import os
import re
import shutil
import subprocess
import threading
import time
from collections import deque
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path
from zoneinfo import ZoneInfo

from PIL import Image

from common import MIN_FREE, check_space, setup, status, summary

ROOT = Path('/srv/datasets/alaska')
IMAGES = ROOT / 'tiff_various_color'
IMAGES.mkdir(parents=True, exist_ok=True)
SOURCE = Path(__file__).with_name('selection_v1.csv')
START_AT = dt.datetime(2026, 10, 15, 13, 15, tzinfo=ZoneInfo('Asia/Singapore'))
BASE = 'https://alaska.utt.fr/DATASETS/ALASKA_v2_TIFF_VariousSize_COLOR'
START = dt.datetime.now(dt.timezone.utc)
LOG = setup(ROOT, 'alaska')
TOTAL = 80005 - 9723
WORKERS = 8
SINGLE_MAX = 4 * 1024**2  # reserve enough for each in-flight TIFF plus a 50 GiB floor
DIGESTS = ROOT / 'SHA256SUMS'
failures = 0
failure_lock = threading.Lock()
stop = threading.Event()
verified: dict[int, tuple[str, int]] = {}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024**2), b''):
            h.update(block)
    return h.hexdigest()


def tiff_ok(path: Path) -> bool:
    try:
        with Image.open(path) as image:
            if (image.format != 'TIFF' or image.mode != 'RGB' or min(image.size) < 512
                    or image.tag_v2.get(258) != (8, 8, 8)
                    or image.tag_v2.get(277) != 3):
                return False
            image.load()  # reject truncated/corrupt TIFF pixel data before rename
            return True
    except (OSError, ValueError, SyntaxError):
        return False


def record(state: str, note: str = '') -> None:
    status(ROOT, state=state, done=len(verified), total=TOTAL,
           bytes_=sum(size for _, size in verified.values()), failures=failures,
           started=START, note=note)


def selection() -> set[int]:
    with SOURCE.open(newline='', encoding='utf-8') as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != 9723:
        raise ValueError('wrong PC selection count')
    picked = {int(row['alaska_id']) for row in rows}
    if len(picked) != 9723 or min(picked) < 1 or max(picked) > 80005:
        raise ValueError('invalid/duplicate PC selection')
    return picked


def existing(ids: list[int]) -> None:
    known = {}
    if DIGESTS.is_file():
        for line in DIGESTS.read_text().splitlines():
            match = re.fullmatch(r'([a-f0-9]{64})  tiff_various_color/([0-9]{5})\.tif', line)
            if not match:
                raise ValueError('invalid SHA256SUMS record')
            known[int(match[2])] = match[1]
    for number in ids:
        path = IMAGES / f'{number:05d}.tif'
        if number not in known or not path.is_file():
            continue
        if tiff_ok(path) and sha256(path) == known[number]:
            verified[number] = (known[number], path.stat().st_size)
    # Remove stale entries and duplicates after a crash. Unverified final files
    # are re-fetched; verified files remain unchanged.
    with DIGESTS.open('w') as stream:
        for number, (digest, _) in sorted(verified.items()):
            stream.write(f'{digest}  tiff_various_color/{number:05d}.tif\n')
    LOG.info('restart verified %d/%d files', len(verified), TOTAL)
    record('running', 'verified prior downloads')


def fetch(number: int) -> tuple[int, str, int, str]:
    name = f'{number:05d}.tif'
    part = IMAGES / (name + '.part')
    final = IMAGES / name
    for attempt in range(3):
        if shutil.disk_usage(ROOT).free < MIN_FREE + WORKERS * SINGLE_MAX:
            return number, 'disk_guard', 0, ''
        part.unlink(missing_ok=True)  # restart only this interrupted object at byte zero
        request = subprocess.run(
            ['curl', '--silent', '--show-error', '--location', '--connect-timeout', '12',
             '--max-time', '90', '--output', str(part), '--write-out', '%{http_code}',
             f'{BASE}/{name}'], capture_output=True, text=True,
        )
        code = request.stdout.strip()
        if request.returncode == 0 and code == '200' and tiff_ok(part):
            digest = sha256(part)
            size = part.stat().st_size
            os.replace(part, final)
            return number, digest, size, ''
        part.unlink(missing_ok=True)
        # No hard stop for HTTP 429/5xx or timeouts: retry this file and
        # eventually defer it so healthy files can still make progress.
        reason = f'curl={request.returncode}, http={code}, valid_tiff=no_or_unchecked'
        global failures
        with failure_lock:
            failures += 1
        if attempt < 2:
            delay = (600, 1200)[attempt]
            LOG.warning('retry %s %s; wait %ds', name, reason, delay)
            stop.wait(delay)
            if stop.is_set():
                return number, 'stopped', 0, ''
    LOG.warning('defer %s after 3 failures', name)
    return number, 'deferred', 0, ''


def main() -> int:
    global failures
    now = dt.datetime.now(ZoneInfo('Asia/Singapore'))
    while now < START_AT:
        left = (START_AT - now).total_seconds()
        record('waiting', f'ALASKA starts at {START_AT.isoformat()}, {int(left)}s left')
        time.sleep(min(60, max(1, left)))
        now = dt.datetime.now(ZoneInfo('Asia/Singapore'))
    excluded = selection()
    ids = [i for i in range(1, 80006) if i not in excluded]
    assert len(ids) == TOTAL
    try:
        check_space(ROOT, WORKERS * SINGLE_MAX)
    except RuntimeError as exc:
        LOG.error('%s', exc)
        record('disk_guard', str(exc))
        return 78
    existing(ids)
    todo = deque(n for n in ids if n not in verified)
    deferred = deque()
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        active = {}
        def fill() -> None:
            while todo and len(active) < WORKERS:
                number = todo.popleft()
                active[pool.submit(fetch, number)] = number
        fill()
        while active or deferred:
            if not active:
                record('retrying', f'{len(deferred)} deferred files; wait 30 min')
                LOG.warning('retry %d deferred files after 30 min', len(deferred))
                time.sleep(1800)
                todo, deferred = deferred, deque()
                fill()
                continue
            finished, _ = wait(active, return_when=FIRST_COMPLETED, timeout=60)
            if not finished:
                record('running', f'in_flight={len(active)}, deferred={len(deferred)}')
                continue
            for future in finished:
                number = active.pop(future)
                ident, digest, size, reason = future.result()
                if reason == 'disk_guard':
                    stop.set()
                    record('disk_guard', f'free space below 50 GiB at {number:05d}')
                    LOG.error('disk guard at %s', number)
                    # No new jobs; thread pool waits for remaining workers to end.
                    return 78
                assert ident == number
                if reason == 'deferred':
                    deferred.append(number)
                    record('retrying', f'{number:05d} deferred; others continue')
                    continue
                verified[number] = (digest, size)
                with DIGESTS.open('a') as stream:
                    stream.write(f'{digest}  tiff_various_color/{number:05d}.tif\n')
                    stream.flush()
                    os.fsync(stream.fileno())
                if len(verified) % 100 == 0:
                    LOG.info('verified %d/%d bytes=%d', len(verified), TOTAL,
                             sum(value[1] for value in verified.values()))
                    record('running', f'in_flight={len(active)}, deferred={len(deferred)}')
            fill()
    if len(verified) != TOTAL:
        raise RuntimeError(f'incomplete: {len(verified)}/{TOTAL}')
    with DIGESTS.open('w') as stream:
        for number, (digest, _) in sorted(verified.items()):
            stream.write(f'{digest}  tiff_various_color/{number:05d}.tif\n')
    record('done', 'all files verified')
    summary(ROOT, done=len(verified), total=TOTAL,
            bytes_=sum(size for _, size in verified.values()), failures=failures, started=START)
    LOG.info('complete all %d TIFFs', TOTAL)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
