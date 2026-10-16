> Archived 2026-10-16: done; verification remains for reference. See [findings](../findings.md).

# Full-photo session cache

Status: plan step 6 complete. Cache implementation, build, integrity checks,
full replay, read measurements, and all tests passed on the team VM. Cache root is a required script argument. The approved
team cache is `/srv/aai3001/otf-cache/v1/`, outside the checkout.

## Format and safety

`lsb_cache.py` stores one C-contiguous RGB8 `{photo_id}.npy` and one
`{photo_id}.json` record per photo. `cache.json` is the single format header:
version 1, uint8, C order, three channels. Each entry records photo/source/type,
noise policy, shape, array SHA-256 (raw C-order pixel bytes, not NPY header),
master path/size/mtime/SHA-256, and decode version.

Decode records include NumPy/Pillow versions. NEF records also include
rawpy/LibRaw versions, the shared RAW settings record, and the disabled noise
hook. PNG records mark RAW fields/hook as not applicable. Increment the source
decode version if numerical source decoding changes beyond the recorded settings.

Writers hold per-photo file locks, write a temporary NPY in the cache folder,
flush/fsync it, rename it, then atomically commit the JSON record. JSON/header
writes also flush/fsync and rename. Directory sync completes the commit.
Readers take a shared entry lock; they cannot see an in-progress replacement.
A crash before record commit leaves an entry that strict loading rejects and
an explicit build can repair. Temporary files are removed on handled failures;
resume also removes interrupted entry/header temp files under their locks.
Zero-byte files in `.locks/` are intentional persistent locks, not temp litter.

The builder skips only entries whose shape/version, source identity/versions,
array hash, and current master hash pass. It repairs invalid entries explicitly.
The loader **never** decodes on a missing or stale entry:

```python
pixels = load_cached(photo_manifest_row, cache_root, repo_root, mmap=True)
```

Default checks are strict: format/header, identity, source/noise/decode versions,
shape/dtype/order, master size/mtime/hash, and array hash. `mmap=True` returns a
read-only mapping; normal loading returns an owned array. After a full session
validation, callers may explicitly set `verify_hash=False` and
`verify_master_hash=False` to avoid full scans for each tile. Cheap metadata,
version, shape and master-stat checks still run. This relaxed mode is not a
new integrity validation; do not use it to accept unverified or changed data.

`build_cache.py` uses spawned processes with OMP=1, one decode per built entry,
file locks for concurrent writers, a disk-space preflight, and group-readable/
writable cache files. The shared cache root uses group aai3001 and setgid.
Masters and existing datasets are never changed or deleted.

The cache may be outside the checkout. It may be inside the checkout only when
`git check-ignore -q <cache-root>` succeeds from the repository root. The
builder refuses an in-checkout path when Git reports that it is not ignored,
when the check fails, or when Git is unavailable. The default worker limit
remains 1 through 8. The G4-only setup uses explicit `--allow-high-workers`
with `--workers "$(($(nproc)-8))"`: the opt-in accepts at most 64 workers
and cannot exceed detected CPUs. Budget about 1 GiB RAM per builder process;
40 builders need up to about 40 GiB on the 180 GB G4. Use fewer if memory
pressure appears. Output bytes do not depend on worker count.
`check_cache.py` and the loader had no separate cache-location
rule to change; they continue to validate cache contents and never create a
cache implicitly.

## Commands

Run copied code from VM scratch, not from a writable VM checkout. Before the
full build, run the decode gate alone; then tests and build:

```sh
OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 \
  /srv/aai3001/AAI3001-CVDL/.venv/bin/python scripts/check_cache.py \
  --repo-root /srv/aai3001/AAI3001-CVDL \
  --cache-root /srv/aai3001/otf-cache/v1 --gate decode
# Same arguments with --gate verify, replay, read after the build.
OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 \
  /srv/aai3001/AAI3001-CVDL/.venv/bin/python scripts/build_cache.py \
  --repo-root /srv/aai3001/AAI3001-CVDL \
  --cache-root /srv/aai3001/otf-cache/v1 --workers 8
```

For tests, set TMPDIR to a temporary subfolder **inside** the approved cache
root. Remove that test folder after tests. Long jobs can run in the background
with logs in code scratch; keep the real cache when cleaning scratch.

On the PC, the Git-ignored cache is inside the checkout and uses the container
path `/work/Datasets/generated/otf-cache/v1`:

```sh
podman exec -w /work aai3001 python scripts/build_cache.py \
  --repo-root /work --cache-root /work/Datasets/generated/otf-cache/v1 --workers 8
podman exec -w /work aai3001 python scripts/check_cache.py \
  --repo-root /work --cache-root /work/Datasets/generated/otf-cache/v1 --gate verify
```

The builder runs `git check-ignore` before it creates or resumes this cache.

## PC cache build (2026-10-14)

The PC container did not include Git, so Git 2.43.0 was installed in the
container before the build. `git check-ignore` then confirmed that
`/work/Datasets/generated/otf-cache/v1` is ignored by the checkout. The build used the
maximum allowed eight workers, OMP=1, and completed in 170.430413 seconds.
It created 1,900 NPY files, 1,900 entry JSON records, one header JSON, and
1,901 persistent lock files. Cache content is 52,763,054,963 bytes
(about 52.76 GB or 49.15 GiB); allocated storage is 52,773,285,888 bytes.

The strict `check_cache.py --gate verify` pass completed in 13.751310 seconds.
It checked all 1,900 arrays and current master files: zero hash mismatches,
52,761,268,224 array bytes, and all source/version/shape checks passed.
The VM was initially off, so the comparison was pending at build time. After
it started, all 1,900 PC and VM entry JSON records matched field-for-field
once `master_mtime_ns` (machine-local) was removed. SHA-256 of the sorted
`photo_id -> normalized-record-SHA-256` map was
`bde842a00a9060d2a3b5e6c9e214a5d6f976a52f3e494f503a1290a93c8ae482`
on both machines; zero missing or mismatched records. The comparison read the
VM cache only; it did not modify that cache. Git 2.43.0 was installed with
`apt` in the PC container for the ignored-path check. That installation is
lost if container `aai3001` is recreated; install Git there again before
building an in-checkout cache.

## Isolated decode scaling

No other agent Python/training work was observed before this gate; VM load
was 0.02. Each camera used eight photos, reused at 1/4/8 workers. OMP=1. Wall
seconds/photo includes worker startup and concurrent work. Mean decode seconds
is the average latency inside each worker. These are not forced cold reads:
there is no permission to drop the kernel page cache.

| Camera | Processes | Wall s/photo | Photos/s | Mean decode s/photo |
| --- | ---: | ---: | ---: | ---: |
| D7000 | 1 | 2.295154 | 0.435701 | 2.258741 |
| D7000 | 4 | 0.634204 | 1.576781 | 2.322040 |
| D7000 | 8 | 0.557997 | 1.792126 | 3.862801 |
| D90 | 1 | 1.780787 | 0.561549 | 1.746286 |
| D90 | 4 | 0.487100 | 2.052966 | 1.786185 |
| D90 | 8 | 0.433170 | 2.308565 | 2.945337 |
| D40 | 1 | 0.871789 | 1.147067 | 0.841254 |
| D40 | 4 | 0.258991 | 3.861141 | 0.878366 |
| D40 | 8 | 0.241273 | 4.144684 | 1.473693 |

Eight workers improve aggregate speed only modestly above four, while increasing
individual decode latency. This small batch is not a sustained loader benchmark.

## Build and correctness

Initial free space: 106,553,708,544 bytes. Build versions: NumPy 2.5.3,
Pillow 12.3.0, rawpy 0.27.1, LibRaw 0.22.1. Final inventory: 1,900 NPY files,
1,900 entry JSON records plus one header, zero temporary files. Free space after
all checks: 53,778,096,128 bytes (build-time free space is listed below).

| Check | Result |
| --- | --- |
| Full 1,900-photo build | Eight workers, OMP=1, wall 554.714458 s; 1,900 newly built entries. Raw arrays 52,761,268,224 bytes; all files 52,763,054,963 bytes; allocated 52,773,380,096 bytes (about 52.77 GB / 49.15 GiB). Free after 53,779,742,720 bytes (about 50.08 GiB). |
| All array/master hashes | 1,900 arrays and current masters match their recorded hashes; zero mismatches. Eight workers, wall 364.835579 s; source/version/shape checks also passed. |
| All 49,796 cached clean/stego pairs | 1,000 cached RAISE photos; zero clean or stego pixel mismatches against stored PNGs; zero NEF decodes. Validation wall 382.593263 s. Worker-summed read 247.985202 s, cut+embed/hash 382.633263 s, PNG read+compare 2,419.567626 s. |
| Cache-only read+cut+embed wall time | 154.843332 s, eight workers, all 49,796 pairs; no reference PNG reads, zero NEF decodes. Every per-photo aggregate stego hash matches the pixel-validation pass. Worker-summed read 917.356991 s, cut+embed/hash 315.774474 s. |
| Tests | 74 passed: 40 source/manifest/settings/fast/Mode 1/cache tests plus 34 engine/batch/prepare tests. Six cache tests cover interrupted atomic writes, resume, missing-record commit, stale source/master, shape/hash/version rejection, readonly mmap and no decode fallback. |

The replay gate has two passes. First it verifies every clean/stego pair against
stored PNG pixels and records per-photo aggregate stego hashes. Second it
repeats cache-read/cut/embed without PNG reads and checks the hashes. The second
wall time is the requested cache-read + cut + embed cost, with small hash work
included. Source NEFs are stat-checked but never decoded on either replay pass.

## Read speed and disk observations

Read gates use one pinned CPU and all 1,900 photos in fixed random
order, first/second passes for normal NPY loads and read-only mmap followed by
copy. A random real 512 tile is copied from one mmap per random photo to force
page access. Sequential disk observation streams all cache files through an
8 MiB buffer; the total is larger than RAM.

Logical MB/s uses pixel bytes (decimal MB). Physical MB/s uses the process's
`/proc/self/io` read_bytes change and includes kernel read-ahead. Memory snapshots
show total/available RAM and cached pages. Estimated cached fractions from
logical minus physical reads are only estimates, not exact page-hit counts.
A second pass does not imply a fully warm cache: the complete cache exceeds RAM.
No advertised disk ceiling or controlled cold-cache rate is claimed.

| Full-photo route | Pass | Seconds | Logical MB/s | Photos/s | Kernel read GB | Physical MB/s | Likely cached |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Normal np.load | 1 | 131.466140 | 401.330 | 14.452 | 27.004 | 205.410 | 48.8% |
| Normal np.load | 2 | 131.764855 | 400.420 | 14.420 | 27.073 | 205.462 | 48.7% |
| mmap then copy | 1 | 149.082971 | 353.905 | 12.745 | 30.542 | 204.868 | 42.1% |
| mmap then copy | 2 | 173.199774 | 304.627 | 10.970 | 35.297 | 203.792 | 33.1% |

Each full-photo pass uses 52,761,268,224 logical pixel bytes. Total RAM was
33,646,940,160 bytes (31.34 GiB), with about 30.9–31.7 GB in the kernel page
cache during these passes. Available memory was about 31.9–32.2 GB because
cache pages are reclaimable. The complete 52.76 GB photo cache does not fit.
The percentages are rough estimates from read_bytes, not exact hit ratios.

| Other read case | Result |
| --- | --- |
| One random real 512 tile per random photo, mmap + tile copy | 1,900 tiles in 23.782536 s: 79.8906 tiles/s. Logical 1,494,220,800 bytes, 62.8285 MB/s; actual read_bytes 4,646,244,352 bytes, 195.3637 physical MB/s. |
| Large sequential stream, all NPY files, 8 MiB blocks | 52,761,511,424 file bytes in 172.079792 s: 306.6107 logical MB/s. Actual read_bytes 35,375,628,288 bytes: 205.5769 physical MB/s. |

Random tile storage reads were about 3.11 times the tile payload despite a
partly warm cache. Page granularity and read-ahead matter for scattered photo
access. Full-photo normal loads were faster than mmap-then-copy here, but page
residency differed; this is not an isolated algorithm comparison. The observed
storage-read rate was about 204–206 MB/s for full/sequential passes, not the
advertised or maximum disk speed. These results support keeping photo-local
tile work before the planned bounded cross-photo shuffle; no loader was added.
