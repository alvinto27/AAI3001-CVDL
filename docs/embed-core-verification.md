# Shared embed core and mask-free path

Status: plan step 4 complete. CPU-only VM verification. No loader,
cache, training, GPU work, or full-dataset generation was added.

## API and numerical path

`lsb_engine.py` has one `_embed_core()` for validation, payload generation,
unit selection, field padding, replace/match, and pixel writes. Random calls,
arguments, and order are unchanged. `embed()` keeps its signature and returns
`(stego, selection_mask, EmbedResult)`. Random extraction still has the same
private selected-unit cache. Selection masks still mark selection, not change.

`embed_fast()` has the same embedding arguments plus the keyword
`return_changed_units=False`. It returns:

```text
(stego_uint8, public_metadata_dict, changed_units_or_None)
```

The dictionary has all 12 public `EmbedResult` fields, without the private
selection cache. No pixel mask or `EmbedResult` object is constructed on this
route. `units_changed` stays: one Boolean comparison and count over selected
values is cheap and required for metadata parity. Only when asked, the core
filters selected units by that comparison to create increasing, read-only
`np.intp` changed indices. Otherwise no changed-index array is made.

Both routes copy the carrier. Read-only and non-contiguous clean views are
safe. No `out=` or in-place API was added. A changed unit `u` maps to pixel
`u // 3`; the future mode 2 label step can find every tile containing it.
This step returns the required indices but does not yet implement tile labels.
Fast metadata cannot be passed to `extract()` because it omits selection data.

No other speed idea was implemented. A possible later check is whether the
extra `.copy()` after advanced indexing in `before = flat[selected].copy()`
is needed. It remains unchanged here and needs approval before modification.

## Correctness checks

The unmodified pre-step-4 engine was copied from git to VM scratch as an
independent reference. Property checks compared original embed, new embed,
fast with indices, and fast without indices.

| Check | Result |
| --- | --- |
| Random property cases | 20,000 passed; all 96 bit-mode/k/method/placement combinations, at least 208 cases each. |
| Shapes | 4,000 at 256x256; 4,000 at 512x512; 3,999 at 257x263; 4,000 each at 33x65 and 17x19; one at full D7000 3280x4948. |
| Settings | Both bit modes, k=1..8, replace/match, all placements, given/drawn starts, both payload kinds, bytes/bpp with fractional-byte requests. |
| Pixels, metadata, RNG | All four routes agree exactly. Original/new selection masks agree. Caller clean pixels do not change. |
| Changed units | Exact equality to full-carrier `flatnonzero(clean != stego)` in row-major order. Optional-off result is always None. |
| Stored legacy sample | 503 original stego rows, all 12 mode/method/placement groups; zero pixel or public-metadata mismatches. Legacy settings and embedding keys used. |
| Unit tests | 29 source/manifest/settings/fast tests plus 34 existing engine/batch/prepare tests: 63 passed. |

The first mask-allocation test attempt accidentally mocked RNG initialization
as well. The RNG is now created before the allocation mock; no production
change was needed. Validation errors/types/messages and unchanged RNG state
are checked for invalid requests. Fast calls without indices are tested to
allocate no pixel mask.

## One-core speed

Team VM: Intel Xeon at 2.20 GHz, four cores/eight vCPUs. Benchmarks were pinned
to logical CPU 0, with `OMP_NUM_THREADS=1`, NumPy 2.5.3. One warm-up and seven
measured calls per route/case; median reported. Synthetic RGB8 tiles are used
at 256/512. Full-photo rows use actual decoded D7000 `r00617aa1t`, 3280x4948.
Source decode is outside timing. Inputs remain unchanged and are reused.

Fixed settings: lowest k=1, match, 0.5 bpp, random bytes, drawn start; stride
placement uses stride 2. Calls include carrier copy, selection, embedding,
and their requested output work, not source I/O, labels, augmentation, queues,
or tensor conversion. These are images/s **per pinned core**, not loader or
GPU throughput. `embed` below is the current public mask-producing wrapper.

| Image | Placement | embed ms / images/s | fast ms / images/s | fast + indices ms / images/s |
| --- | --- | ---: | ---: | ---: |
| 256 | Contiguous | 1.543 / 648.16 | 1.406 / 711.10 | 1.669 / 599.12 |
| 256 | Stride | 1.536 / 650.89 | 1.392 / 718.40 | 1.673 / 597.64 |
| 256 | Random | 2.187 / 457.21 | 2.106 / 474.91 | 2.337 / 427.92 |
| 512 | Contiguous | 5.453 / 183.40 | 4.926 / 202.99 | 5.833 / 171.44 |
| 512 | Stride | 5.450 / 183.48 | 4.942 / 202.33 | 5.838 / 171.29 |
| 512 | Random | 8.682 / 115.18 | 8.125 / 123.08 | 9.075 / 110.19 |
| Full D7000 | Contiguous | 423.636 / 2.361 | 376.607 / 2.655 | 432.861 / 2.310 |
| Full D7000 | Stride | 424.434 / 2.356 | 376.714 / 2.655 | 436.326 / 2.292 |
| Full D7000 | Random | 880.040 / 1.136 | 832.807 / 1.201 | 889.597 / 1.124 |

Mask-free calls without indices are faster in these checks. Returning changed
indices adds filtering/allocation work and is slower than the mask path here.
It supplies label information; no speed gain is claimed for that output.
Downstream label work is not timed in this step.

## Boolean-mark ordering: rejected

The candidate keeps the exact `rng.choice` subset, marks its units in a
Boolean array, then reads `flatnonzero` to recover increasing order. Every
candidate output equalled the existing sorted indices. It was slower at all
six tested size/density cases, so production keeps `selected.sort()` unchanged.
No new selection algorithm, density-based dispatch, or RNG call was added.

Selection/copy setup is outside these stage-only timers. Each method orders
an owned index array; mark timing includes allocation, marking, and readback.
One warm-up and five measured trials per case:

| Size | Selected units | Density | sort ms | Boolean mark ms |
| --- | ---: | ---: | ---: | ---: |
| 512 | 1,638 | 0.2083% | 0.0112 | 0.2289 |
| 512 | 16,383 | 2.0833% | 0.1510 | 0.7149 |
| 512 | 131,072 | 16.6667% | 1.0060 | 1.1649 |
| Full D7000 | 101,433 | 0.2083% | 0.7947 | 23.3454 |
| Full D7000 | 1,014,339 | 2.0833% | 9.4952 | 61.5984 |
| Full D7000 | 8,114,720 | 16.6667% | 92.5260 | 183.6307 |

## Re-run safely

Copy code and an unmodified pre-step-4 engine reference to VM scratch. Use
the existing VM Python; do not write into the VM checkout. For example:

```sh
OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 \
  /srv/aai3001/AAI3001-CVDL/.venv/bin/python scripts/check_embed_fast.py \
  --repo-root /srv/aai3001/AAI3001-CVDL \
  --old-engine "$PWD/old_engine.py" --cases 20000 --workers 4
```

Property checks use four CPU workers; the timing stage then pins its own
process to one CPU. Reports go to standard output. Remove scratch code,
logs, references, and temporary test outputs when finished. No source images
or retained dataset records are changed.
