"""Shared CPU-suite lock and read-only checks for long-suite fixtures (Linux)."""
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import hashlib
from importlib.metadata import PackageNotFoundError, version
import json
import math
import os
from pathlib import Path
import stat
import subprocess
import sys
from tempfile import NamedTemporaryFile
import time

# Container-local, independent of checkout/worktree. Never remove this inode.
LOCK_PATH = Path('/tmp/cvdl-cpu-test-suite.lock')
REQUIRED_FIXTURES = (
    'tests/fixtures/srm_kernels_published.npy',
    'notebooks/04_baseline_kv_xunet_v3.ipynb',
    'data/raise/photo_manifest.csv', 'data/div2k/photo_manifest.csv',
    'data/alaska/photo_manifest.csv', 'data/lsdir/photo_manifest.csv',
    'data/test/standard-256-v1/summary.json',
    'data/hard-test/256-v1/summary.json',
    'data/val/60da4391043e8ce48ff2/summary.json',
    'runs/srm-e50-paired-20261023T022339Z-4518/checkpoints/'
    'srm-e50-paired-20261023T022339Z-4518/data_run_settings.json',
    'runs/verification/rgb-channel-analysis-2026-10-24/recipe.json',
)


@contextmanager
def suite_lock(root, selection, *, timeout=120., path=LOCK_PATH, stream=None):
    """Wait at most timeout seconds; kernel ownership, not PID text, is authority."""
    if not math.isfinite(timeout) or timeout < 0:
        raise ValueError('lock timeout must be finite and nonnegative')
    stream = sys.stderr if stream is None else stream
    fd = os.open(path, os.O_RDWR | os.O_CREAT | os.O_CLOEXEC | os.O_NOFOLLOW, 0o600)
    acquired = False
    try:
        if not stat.S_ISREG(os.fstat(fd).st_mode):
            raise ValueError('suite lock must be a regular file')
        deadline = time.monotonic() + timeout
        announced = False
        while True:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                acquired = True
                break
            except BlockingIOError:
                os.lseek(fd, 0, os.SEEK_SET)
                owner = os.read(fd, 4096).decode('utf-8', errors='replace').strip() or '(owner details pending)'
                if not announced:
                    print(f'CPU suite locked by {owner}; wait limit {timeout:g}s', file=stream, flush=True)
                    announced = True
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError(f'CPU suite lock refused: owner {owner}')
                time.sleep(min(.1, remaining))
        owner = dict(pid=os.getpid(), started_utc=datetime.now(timezone.utc).isoformat(),
                     checkout=str(Path(root).resolve()), selection=selection)
        os.ftruncate(fd, 0)
        os.lseek(fd, 0, os.SEEK_SET)
        os.write(fd, (json.dumps(owner) + '\n').encode())
        yield
    finally:
        if acquired:
            fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)


def check_fixtures(root, *, required=REQUIRED_FIXTURES):
    """Cheap known-path smoke, not full data integrity validation. Never repair."""
    root = Path(root).resolve()
    errors = []
    try:
        checkout = subprocess.check_output(
            ['git', 'rev-parse', '--show-toplevel'], cwd=root, stderr=subprocess.PIPE, text=True)
        if Path(checkout.strip()).resolve() != root:
            errors.append('Git pointer resolves to a different checkout')
    except (OSError, subprocess.CalledProcessError):
        errors.append('Git pointer is not usable in this container')
    for name in required:
        path = root / name
        try:
            if not path.is_file():
                raise OSError('missing, dangling or not a regular file')
            with path.open('rb') as stream:
                stream.read(1)
        except OSError as error:
            errors.append(f'{name}: {error}')
    # External data links are allowed. This cache-refusal fixture must instead
    # resolve physically inside the checkout, even when the cache does not exist.
    cache = root / 'notebooks/triton-test-cache'
    try:
        if not cache.resolve().is_relative_to(root):
            errors.append('notebooks/triton-test-cache resolves outside checkout')
    except (OSError, RuntimeError):
        errors.append('notebooks/triton-test-cache cannot resolve')
    if errors:
        raise ValueError('CPU suite fixture check refused; fix paths manually (no files changed):\n  '
                         + '\n  '.join(errors))


def evidence_path(root):
    return Path(root) / 'runs/.scratch/full-suite-evidence.json'


def runtime_identity():
    # Distribution metadata avoids importing torch (and touching GPU state).
    return dict(python=sys.version, torch=version('torch'))


def full_evidence_key(root):
    """None on dirty/unreadable inputs; byte hashes, never size/mtime shortcuts."""
    root = Path(root)
    def git(*args):
        return subprocess.check_output(['git', *args], cwd=root, stderr=subprocess.PIPE).strip()
    try:
        if git('status', '--porcelain', '--untracked-files=all'):
            return None
        tree = git('rev-parse', 'HEAD^{tree}').decode('ascii')
        hashes = []
        for name in REQUIRED_FIXTURES:
            with (root / name).open('rb') as stream:
                hashes.append((name, hashlib.file_digest(stream, 'sha256').hexdigest()))
        key = dict(tree=tree, fixtures_sha256=hashlib.sha256(
            json.dumps(hashes, separators=(',', ':')).encode()).hexdigest(), runtime=runtime_identity())
        if git('status', '--porcelain', '--untracked-files=all'):
            return None
        return key
    except (OSError, ValueError, subprocess.CalledProcessError, PackageNotFoundError):
        return None


def record_full_pass(root, before, result):
    """Called only by a complete FULL run; failed/focused/dirty runs never qualify."""
    if before is None or not result.wasSuccessful() or result.testsRun <= 0:
        return False
    after = full_evidence_key(root)
    if after != before:
        return False
    path = evidence_path(root)
    # Refuse evidence locations that would dirty this checkout themselves.
    relative = path.relative_to(root)
    ignored = subprocess.run(['git', 'check-ignore', '-q', '--', str(relative)], cwd=root,
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    if ignored.returncode:
        return False
    payload = dict(version=1, level='full', passed=True, key=after,
                   tests=result.testsRun, skipped=len(result.skipped))
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = None
    try:
        with NamedTemporaryFile('w', dir=path.parent, prefix='.full-evidence-', delete=False) as stream:
            temp = Path(stream.name)
            json.dump(payload, stream, indent=2)
            stream.write('\n')
        os.replace(temp, path)
    finally:
        if temp is not None and temp.exists():
            temp.unlink()
    return True


def matching_full_evidence(root):
    """Read-only local reuse; never changes the classifier's required level."""
    try:
        evidence = json.loads(evidence_path(root).read_text())
        if (not isinstance(evidence, dict) or evidence.get('version') != 1 or evidence.get('level') != 'full'
                or evidence.get('passed') is not True or type(evidence.get('tests')) is not int
                or evidence['tests'] <= 0):
            return False
        current = full_evidence_key(root)
        return current is not None and evidence.get('key') == current
    except (OSError, ValueError, TypeError):
        return False
