"""Versioned, atomic RGB8 photo cache; strict loads never decode on a miss."""
from __future__ import annotations

import contextlib
import fcntl
import hashlib
import json
import os
import re
import tempfile
import time
from pathlib import Path, PurePosixPath

import numpy as np
import PIL
from lsb_raw import raw_settings_record
from lsb_source import configure_source_worker, load_photo

FORMAT_VERSION = 1
SOURCE_DECODE_VERSION = 1


class CacheError(ValueError):
    """Missing, stale, corrupt, or incompatible cache entry; rebuild explicitly."""


def array_sha256(array):
    if not array.flags.c_contiguous:
        raise CacheError("cache array must be C-contiguous")
    return hashlib.sha256(memoryview(array).cast("B")).hexdigest()


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4*1024*1024), b""):
            digest.update(block)
    return digest.hexdigest()


def decode_version(source_type):
    result = dict(source_decode_version=SOURCE_DECODE_VERSION, numpy=np.__version__,
                  pillow=PIL.__version__, rawpy=None, libraw=None, raw_settings=None,
                  raw_noise_hook="not_applicable")
    if source_type == "nef":
        configure_source_worker()
        import rawpy
        result.update(rawpy=rawpy.__version__, libraw=list(rawpy.libraw_version),
                      raw_settings=raw_settings_record(), raw_noise_hook="disabled")
    elif source_type == "tiff_rgb8":
        result.update(tiff_policy="uncompressed-RGB8-orientation1-stored-samples-no-ICC-transform-v1")
    elif source_type != "png8":
        raise CacheError(f"unsupported source_type {source_type}")
    return result


def _paths(row, root, repo):
    photo_id = row["photo_id"]
    if not re.fullmatch(r"[A-Za-z0-9_-]+", photo_id):
        raise CacheError("unsafe photo_id")
    relative = PurePosixPath(row["master_path"])
    if relative.is_absolute() or ".." in relative.parts or "\\" in str(relative):
        raise CacheError("master_path must be repository-relative")
    root = Path(root)
    return root / f"{photo_id}.npy", root / f"{photo_id}.json", Path(repo) / relative


def _read_json(path):
    try:
        with Path(path).open(encoding="utf-8") as stream:
            record = json.load(stream)
        if not isinstance(record, dict):
            raise ValueError("cache record must be an object")
        return record
    except (OSError, ValueError) as exc:
        raise CacheError(f"missing/invalid cache record {path}: {exc}") from exc


def _check_header(root):
    header = _read_json(Path(root) / "cache.json")
    if header != dict(format_version=FORMAT_VERSION, dtype="uint8", order="C", channels=3):
        raise CacheError("incompatible cache format/header; use a new cache root")


def _sync_directory(root):
    fd = os.open(root, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _atomic_json(path, record):
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                         prefix=f".{path.name}-", suffix=".tmp", delete=False) as stream:
            temporary = Path(stream.name)
            os.fchmod(stream.fileno(), 0o664)
            json.dump(record, stream, sort_keys=True, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        _sync_directory(path.parent)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _clean_interrupted_temps(path):
    # Call only while holding the header/entry lock for this exact target.
    for abandoned in path.parent.glob(f".{path.name}-*.tmp"):
        abandoned.unlink()


def initialize_cache(root):
    root = Path(root)
    root.mkdir(mode=0o2775, parents=True, exist_ok=True)
    (root / ".locks").mkdir(mode=0o2775, exist_ok=True)
    with (root / ".locks/header.lock").open("a+b") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        _clean_interrupted_temps(root / "cache.json")
        if (root / "cache.json").exists():
            _check_header(root)
        else:
            _atomic_json(root / "cache.json", dict(format_version=FORMAT_VERSION,
                                                   dtype="uint8", order="C", channels=3))


@contextlib.contextmanager
def _entry_lock(root, photo_id, *, writing=False):
    path = Path(root) / ".locks" / f"{photo_id}.lock"
    try:
        stream = path.open("a+b" if writing else "rb")
    except OSError as exc:
        raise CacheError(f"missing cache entry lock for {photo_id}; build explicitly") from exc
    with stream:
        fcntl.flock(stream, fcntl.LOCK_EX if writing else fcntl.LOCK_SH)
        yield


def _load_unlocked(row, root, repo, *, mmap, verify_hash, verify_master_hash):
    _check_header(root)
    path, record_path, master = _paths(row, root, repo)
    record = _read_json(record_path)
    shape = [int(row["source_height"]), int(row["source_width"]), 3]
    expected = dict(format_version=FORMAT_VERSION, photo_id=row["photo_id"], source=row["source"],
                    source_type=row["source_type"], noise=row["noise"], master_path=row["master_path"],
                    shape=shape, dtype="uint8", order="C", decode=decode_version(row["source_type"]))
    for key, value in expected.items():
        if record.get(key) != value:
            raise CacheError(f"{row['photo_id']}: stale/mismatched {key}; rebuild explicitly")
    try:
        stat = master.stat()
        if (record["master_size"], record["master_mtime_ns"]) != (stat.st_size, stat.st_mtime_ns):
            raise CacheError(f"{row['photo_id']}: master size/mtime changed")
        if verify_master_hash and file_sha256(master) != record["master_sha256"]:
            raise CacheError(f"{row['photo_id']}: master SHA-256 changed")
        array = np.load(path, mmap_mode="r" if mmap else None, allow_pickle=False)
        if array.shape != tuple(shape) or array.dtype != np.uint8 or not array.flags.c_contiguous:
            raise CacheError(f"{row['photo_id']}: array shape/dtype/order mismatch")
        if verify_hash and array_sha256(array) != record["array_sha256"]:
            raise CacheError(f"{row['photo_id']}: array SHA-256 mismatch")
    except (OSError, ValueError, KeyError) as exc:
        if isinstance(exc, CacheError):
            raise
        raise CacheError(f"{row['photo_id']}: invalid cache data: {exc}") from exc
    return array


def load_cached(row, cache_root, repo_root, *, mmap=False, verify_hash=True,
                verify_master_hash=True):
    """Strict by default: header, identity, source versions, shape, master and array.

    mmap=True returns a read-only mapping. False returns an owned array.
    Hash checks may be disabled explicitly after session validation, but cheap
    metadata/version/shape/master-stat checks always run. Never falls back to RAW.
    """
    _paths(row, cache_root, repo_root)  # validate IDs before lock path construction
    with _entry_lock(cache_root, row["photo_id"]):
        return _load_unlocked(row, cache_root, repo_root, mmap=mmap, verify_hash=verify_hash,
                              verify_master_hash=verify_master_hash)


def build_entry(row, cache_root, repo_root):
    """Build or strictly validate-and-skip one entry; serialize concurrent writers.

    Array rename precedes record commit while holding the entry lock. Failure
    leaves no partial array; a missing/stale record is rejected and resumable.
    Persistent zero-byte lock files are intentional, not abandoned temporaries.
    """
    initialize_cache(cache_root)
    path, record_path, master = _paths(row, cache_root, repo_root)
    with _entry_lock(cache_root, row["photo_id"], writing=True):
        _clean_interrupted_temps(path)
        _clean_interrupted_temps(record_path)
        if record_path.exists():
            try:
                existing = _load_unlocked(row, cache_root, repo_root, mmap=True,
                                          verify_hash=True, verify_master_hash=True)
                return dict(photo_id=row["photo_id"], status="skipped", bytes=existing.nbytes)
            except CacheError:
                pass
        stat = master.stat()
        master_hash = file_sha256(master)
        start = time.perf_counter()
        array = load_photo(row, Path(repo_root))
        decoded = time.perf_counter() - start
        if (master.stat().st_size, master.stat().st_mtime_ns) != (stat.st_size, stat.st_mtime_ns):
            raise CacheError("master changed during cache build")
        record = dict(format_version=FORMAT_VERSION, photo_id=row["photo_id"], source=row["source"],
                      source_type=row["source_type"], noise=row["noise"], shape=list(array.shape), dtype="uint8", order="C",
                      array_sha256=array_sha256(array), master_path=row["master_path"],
                      master_size=stat.st_size, master_mtime_ns=stat.st_mtime_ns,
                      master_sha256=master_hash, decode=decode_version(row["source_type"]))
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}-", suffix=".tmp",
                                             delete=False) as stream:
                temporary = Path(stream.name)
                os.fchmod(stream.fileno(), 0o664)
                np.save(stream, array, allow_pickle=False)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, path)
            _atomic_json(record_path, record)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
        return dict(photo_id=row["photo_id"], status="built", bytes=array.nbytes, decode_seconds=decoded)
