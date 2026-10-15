#!/usr/bin/env python3
"""Unattended, verified HF model-repo archive acquisition for the Pi."""
from __future__ import annotations

import datetime as dt
import hashlib
import os
import signal
import subprocess
import time
from pathlib import Path

from common import check_space, setup, status, summary

ROOT = Path('/srv/datasets/lsdir')
TABLE = Path(__file__).with_name('lsdir-sha256.tsv')
HF = Path.home() / 'hf-env/bin/hf'
START = dt.datetime.now(dt.timezone.utc)
LOG = setup(ROOT, 'lsdir')
EXPECTED = {}
for line in TABLE.read_text().splitlines():
    if line.startswith('#') or not line.strip():
        continue
    name, size, digest = line.split()
    EXPECTED[name] = (int(size), digest)
if len(EXPECTED) != 17:
    raise ValueError('expected 17 pinned shard digests')
NAMES = [f'shard-{i:02d}.tar.gz' for i in range(17)] + ['val1.tar.gz']
verified: dict[str, tuple[str, int]] = {}
failures = 0


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024**2), b''):
            h.update(block)
    return h.hexdigest()


def record(state: str, note: str = '') -> None:
    status(ROOT, state=state, done=len(verified), total=len(NAMES),
           bytes_=sum(size for _, size in verified.values()), failures=failures,
           started=START, note=note)


def run_hf(name: str, expected_size: int, force: bool = False) -> bool:
    # No token in argv, environment overrides, stdout, stderr, or the log.
    env = dict(os.environ, HF_HUB_DISABLE_PROGRESS_BARS='1', HF_XET_NUM_CONCURRENT_RANGE_GETS='4',
               HF_XET_CHUNK_CACHE_SIZE_BYTES='0')
    check_space(ROOT, expected_size)
    command = [str(HF), 'download', 'ofsoundof/LSDIR', name, '--local-dir', str(ROOT)]
    if force:
        command.append('--force-download')
    process = subprocess.Popen(command, env=env, stdout=subprocess.DEVNULL,
                               stderr=subprocess.DEVNULL, start_new_session=True)
    while True:
        try:
            code = process.wait(timeout=60)
            return code == 0
        except subprocess.TimeoutExpired:
            try:
                check_space(ROOT)
            except RuntimeError:
                os.killpg(process.pid, signal.SIGTERM)
                process.wait()
                raise
            record('running', name)


def inspect(name: str) -> bool:
    path = ROOT / name
    if not path.is_file():
        return False
    size = path.stat().st_size
    if name in EXPECTED and size != EXPECTED[name][0]:
        return False
    value = digest(path)
    if name in EXPECTED and value != EXPECTED[name][1]:
        return False
    verified[name] = (value, size)
    return True


def main() -> int:
    global failures
    # Xet may give an interrupted archive a new temporary suffix on restart.
    # Service KillMode=control-group guarantees no old downloader still writes
    # here; remove only abandoned incomplete files, never completed archives.
    for stale in (ROOT / '.cache/huggingface/download').glob('*.incomplete'):
        LOG.warning('remove abandoned partial %s bytes=%d', stale.name, stale.stat().st_size)
        stale.unlink()
    # Verify on every restart; do not trust metadata alone after a power cut.
    for name in NAMES:
        if inspect(name):
            LOG.info('verified existing %s', name)
    record('running', 'checking archives')
    pending = [name for name in NAMES if name not in verified]
    round_ = 0
    while pending:
        later = []
        for name in pending:
            size = EXPECTED.get(name, (765_000_000, ''))[0]
            for attempt in range(2):
                record('running', f'downloading {name} attempt {attempt + 1}')
                LOG.info('download %s attempt %d', name, attempt + 1)
                try:
                    ok = run_hf(name, size, force=attempt == 1)
                except RuntimeError as exc:
                    LOG.error('%s', exc)
                    record('disk_guard', name)
                    return 78
                if ok and inspect(name):
                    LOG.info('verified %s bytes=%d', name, verified[name][1])
                    record('running', f'verified {name}')
                    break
                path = ROOT / name
                if ok and path.exists():
                    # A present but wrong hash/size must never be retained.
                    LOG.error('size/SHA-256 mismatch %s attempt %d', name, attempt + 1)
                    path.unlink()
                    continue
                # Transient failures never exhaust the service's retries.
                failures += 1
                delay = min(1800, 600 * min(failures, 3))
                LOG.warning('temporary failure %s; retry in %ds', name, delay)
                record('retrying', f'{name}: waiting {delay}s')
                time.sleep(delay)
            else:
                failures += 1
                LOG.error('two invalid attempts %s; retry in a later pass', name)
                later.append(name)
                record('retrying', f'{name}: deferred after second mismatch')
        pending = later
        if pending:
            round_ += 1
            delay = min(1800, 600 * round_)
            LOG.warning('retry pass for %d files after %ds', len(pending), delay)
            time.sleep(delay)
    with (ROOT / 'SHA256SUMS').open('w') as stream:
        for name in NAMES:
            stream.write(f'{verified[name][0]}  {name}\n')
    record('done', 'all files checked')
    summary(ROOT, done=len(verified), total=len(NAMES), bytes_=sum(size for _, size in verified.values()),
            failures=failures, started=START)
    LOG.info('complete %d archives', len(NAMES))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
