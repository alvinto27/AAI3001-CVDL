# Shared settings draws and seed keys

Status: plan step 3 complete, 2026-10-13. `lsb_settings.py` is the shared
settings/seed module. `lsb_batch.py` imports its original private helper names
and exception class, so existing callers keep those names. No loader or
embedding optimisation is implemented here.

## Legacy compatibility

The choice, stride, byte conversion, capacity, and legacy size helpers retain
their original logic. One shared settings body draws in this order:

```text
bit_mode -> k -> method -> placement -> stride (only for stride placement)
         -> payload_kind -> size
```

Fixed choices do not consume a choice draw. Legacy size ranges limit their
upper bound to capacity before drawing. A fixed value or minimum above
capacity still raises `SizeDoesNotFit`. A carrier smaller than one payload
byte still fails. The only structural addition is a size callback; its
default is the unchanged legacy size helper.

Legacy key, unchanged:

```text
SeedSequence([seed, original_cover_index, variant, stream])
stream: settings=0, embedding=1
```

Do not append a version, epoch, source index, or purpose to this key.
`legacy_seed_sequence()` and `draw_legacy_settings()` expose the legacy route.
The batch worker continues to replace the settings RNG with the separate
embedding RNG before it calls the engine.

## New OTF key, version 1

`otf_seed_sequence()` uses explicit named keyword arguments. **Entropy** is
the ordered integer list supplied to NumPy's `SeedSequence`:

```text
[version, purpose, seed, epoch, carrier, photo_index,
 grid_id, tile_size, tile_row, tile_col, variant, stream]
```

| Part | Integer value / policy |
| --- | --- |
| Version | 1. A future key change must use a new version. |
| Purpose | `train`=1, `val`=2, `test`=3. CSV `val` means validation. |
| Seed | Training uses `run_seed`. Validation/test always use fixed evaluation seed 1, not `run_seed`. |
| Epoch | Training uses `epoch`, or 0 with `fixed_variants=True`. Validation/test always use 0. |
| Carrier | `tile`=1 (mode 1), `photo`=2 (mode 2). |
| Photo index | Frozen globally unique index: RAISE 0..999; DIV2K 1000..1899. |
| Grid | `none`=0, `topleft_crop`=1, `overlap_last`=2. |
| Tile size | Overlap tile keys: 256 or 512. Top-left tile keys: 512 only. |
| Tile row / column | Non-negative row/column from the frozen grid. |
| Variant | Positive integer, initially 1. |
| Stream | `settings`=0, `embedding`=1. |

With `carrier="photo"`, grid ID, tile size, row, and column are all set to 0.
Thus mode 2 produces the same full-photo embedding stream for 256 and 512
extraction. Carrier and purpose still separate the random streams. Do not
include worker IDs, rank, scheduling order, or training source weights.
Keep the recorded NumPy version for exact replay; a key alone does not freeze
future RNG transformation changes. See the
[NumPy SeedSequence reference](https://numpy.org/doc/stable/reference/random/bit_generators/generated/numpy.random.SeedSequence.html).

Validation/test ignore both supplied `run_seed` and `epoch`. Their settings
profile and variant must also remain fixed outside this helper. Purpose
separates their keys. `fixed_variants=True` affects training epoch only;
changing training run seed still changes its streams.

A new OTF key with grid code 1 is **not** legacy replay. Exact v6 replay uses
the four-integer legacy key. Use the versioned OTF key only for new runs.

## New lower-to-fit path

Call `draw_otf_settings()` explicitly for new OTF runs. It shares all choices
and their draw order with the legacy function, but uses a new size callback:

1. Draw the requested size from its configured range, or take its fixed value.
2. Convert bpp to payload bytes with v6 rounding and one-byte minimum.
3. Limit the requested bytes to the drawn settings' capacity.
4. Return the effective size and requested/effective size records.

The record includes `requested_size_kind`, `requested_size_value`,
`requested_bpp`, `requested_payload_bytes`, `payload_bytes`, `effective_bpp`,
`capacity_payload_bytes`, and `clipped`. Effective bpp uses the actual rounded
bytes and carrier pixel count. For mode 2, the caller passes the full photo.
A below-byte request may rise to the existing one-byte minimum. A carrier
unable to hold one byte fails; it does not receive padding or a zero payload.

Unlike legacy upper-range clipping, new ranges draw a requested value first
and then clip its bytes. This change is isolated to the new function/key
route. Original batch generation never opts into it.

## Optional weighted training draws

A config can set two explicit options. Both are absent by default.

- `k_weights` maps each configured `k` to a finite, non-negative weight.
  The weights must have a positive sum. The sampler divides each weight by
  that sum. Every configured `k` must occur exactly once in the mapping.
- `bpp_mixture` is a list of `{weight, bpp}` components. The sampler first
  selects a component by weight, then draws uniformly from its `bpp` range.
  Every component range must be inside `size.bpp`. Byte rounding and carrier
  capacity limits still apply after the requested draw.

Without these options, the config record and all random draws stay unchanged.
With them, the training data record adds `training_sampling`: the version,
config SHA-256, and complete resolved config. Each sampled mixture draw also
records `bpp_mixture_component`. Validation and test use the separate frozen
`evaluation_config`, never the training config.

[`configs/train-hard-v1.yaml`](../../configs/train-hard-v1.yaml) sets
`k_weights: {1: 3, 2: 1, 3: 1, 4: 1}`. Its two bpp components have equal weights:
`[0.05, 0.15]` and `[0.05, 0.50]`. Thus 61.111% of requested payloads are below
0.15 bpp, not 50%: the full-range component also includes small payloads.

| Requested bpp bin | Default share | Hard share |
| --- | ---: | ---: |
| 0.05–0.10 | 11.111% | 30.556% |
| 0.10–0.15 | 11.111% | 30.556% |
| 0.15–0.35 | 44.444% | 22.222% |
| 0.35–0.50 | 33.333% | 16.667% |

The [SRM-hard notebook](../../notebooks/experiments/srm-hard.ipynb) retains the
30-epoch SRM model recipe. It uses 216,000 pairs per epoch and equal source
weights. The source scheduler uses exact quotas, not independent random source
choices: RAISE, ALASKA and LSDIR each supply 72,000 pairs. It samples without
replacement until a source pool is exhausted. The approved 256-overlap pools
contain 170,356 / 121,704 / 140,863 tiles respectively, so no source pool cycles
within an epoch. The original SRM recipe supplies 72,000 / 72,000 / 36,000 pairs
in that source order. Other stego choices remain uniform and unchanged.

## VM checks

Environment: NumPy 2.5.3. All execution was in VM scratch, not the checkout.

| Check | Result |
| --- | --- |
| Complete legacy redraw | 49,796 stego rows; zero mismatches in every one of 11 fields. |
| Fields | `bit_mode`, `k`, `method`, `placement`, `start`, `start_random`, `stride`, `payload_kind`, `payload_bytes`, `size_kind`, `size_value`. |
| Placements checked | Random 16,564; stride 16,623; contiguous 16,609. |
| Redraw elapsed | 13.215151 seconds initially; 13.333465 seconds on final-code repeat. Correctness check, not an embedding benchmark. |
| New seed/settings tests | 8 passed: repeat streams/settings, changed identity/epoch/variant, fixed evaluation, legacy formula, photo cross-size identity, capacity and byte rounding. |
| Prior manifest/source tests | 18 passed. |
| Existing batch/engine/prepare tests | 34 passed, including 9 batch tests. |
| Total unit tests | 60 passed. |
| Three-cover production `generate()` | Covers 000003/000004/000005, random/stride/contiguous; all 3 stego PNGs are pixel-identical **and file-byte-identical** to stored references. |

Stored `start` is an embedding-stream result, not a settings-choice result.
The redraw check calls the engine's own `_make_payload()` and
`_select_units()` for the 33,232 contiguous/stride rows. It produces no stego
images during that check. Random-placement start/stride are absent by the
engine contract. No payload-prefix or start RNG logic is copied. The only
mirrored arithmetic is units-used calculation from pre-step-4
`lsb_engine.py` at commit `967cfd4`, lines 302-303 and 307 (bits per unit and
ceiling division). The later [core refactor](../archive/embed-core-verification.md)
changes line numbers, not that calculation.

The three-cover check uses the real batch `generate()` entry point and engine.
Two empty non-image files precede the three original tile names in scratch,
preserving original discovery indices 3/4/5 without mocking the worker or
seed code. They are skipped; only three image covers are generated.

Re-run from a scratch code copy with the existing VM Python:

```sh
OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 \
  /srv/aai3001/AAI3001-CVDL/.venv/bin/python scripts/archive/check_settings.py \
  --repo-root /srv/aai3001/AAI3001-CVDL \
  --scratch-root "$PWD/check-output-new"
```

The scratch root must be new and outside the checkout. Remove its generated
images/logs after verification. No full dataset embedding, speed work,
training, GPU work, or cache creation is part of this step.
