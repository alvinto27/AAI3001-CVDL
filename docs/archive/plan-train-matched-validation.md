# Plan: train-matched validation (work list #1)

Status: **implemented, opt-in** (2026-10-18). Existing notebooks and legacy validation are unchanged. Design by Worker_1 (2026-10-18); user decisions below override the design where they differ.

Source decision: [2026-10-17 handoff](handoff-2026-10-17.md), section 2, "Validation": validation follows the run's training set.

## Problem

| | Today (`checkpoint_validation='capped'`) | Wanted |
| --- | --- | --- |
| Tile size | 512 | Training tile size (256) |
| Grid | Source-record trim for ALASKA/LSDIR | Overlap-last, as in training |
| k, bpp, stego type | Frozen run1 evaluation config (k 1-8) | The run's training config (incl. `k_weights`, `bpp_mixture`) |
| Source mix | RAISE 5,000 / ALASKA 1,535 / LSDIR 1,500 (RAISE 62%) | The run's training `source_mix` |
| Frozen record | `validation_pairs.csv` with `cover_id,source` only (hash `6bb3c8ef...`) | Full per-pair rows and a recipe hash |

Current path: `cvdl/settings.py:162-181` (`tile_size`, `train_tile_size`, `checkpoint_validation`), `LSB-Dataset-Generator/lsb_torch.py:141-162,207-215,417-471` (`FixedPairs` uses `evaluation_config`), `LSB-Dataset-Generator/lsb_notebook.py:12-25,65-108` (capped sampling, seeds 2401/2402), `cvdl/runtime.py:147-155`, `scripts/evaluate_run.py:188-192`.

## User decisions (2026-10-18)

| # | Decision |
| ---: | --- |
| 1 | **Size = 10% of the training pairs per epoch**, derived from the run's settings. Quotas follow the training source mix exactly. Validation scales when training pairs per epoch change. The builder records its rounding rule in the recipe. |
| 2 | **Pool = the existing frozen `val` photo split only.** No split changes. Unique tiles, no repeats. If a quota exceeds a source pool, the builder **fails** with a clear error (no reuse, no cap). |
| 3 | **No augmentation.** k, bpp (incl. `k_weights` / `bpp_mixture`) and stego-type draws follow the training config. |
| 4 | **DIV2K threshold-calibration panel is out of scope.** DIV2K is not trained, so it is not in train-matched validation. It belonged to the per-source threshold task (#4); since 2026-10-18 per-source thresholds are reference only, and the user dropped the DIV2K reference cut-off: no DIV2K validation panel. |

Validation is for checkpoint selection, overfitting curves, and (later) per-source thresholds. The standard test stays one fixed design for all models (work list #2).

## Expected sizes

| Recipe | Train pairs/epoch | Val pairs (10%) | RAISE | ALASKA | LSDIR |
| --- | ---: | ---: | ---: | ---: | ---: |
| SRM (`srm.ipynb`, mix .4/.4/.2) | 180,000 | 18,000 | 7,200 | 7,200 | 3,600 |
| `srm-hard` (`srm-hard.ipynb`, 72k each) | 216,000 | 21,600 | 7,200 | 7,200 | 7,200 |

Validation pools at 256 overlap-last (measured by Worker_1 from `data/<source>/photo_manifest.csv` with `lsb_train_grid.axes(record, 256, 'overlap_last')`, no decoding):

| Source | Val photos | Unique 256 tiles |
| --- | ---: | ---: |
| RAISE | 150 | 36,484 |
| DIV2K | 120 | 5,720 |
| ALASKA | 1,000 | 8,966 |
| LSDIR | 1,500 | 30,584 |

ALASKA is the limit: at 10%, ALASKA training above about 89,660 pairs per epoch makes the builder fail.

Photo split (unchanged): RAISE 701/150/149, DIV2K 680/120/100, LSDIR 7,000/1,500/2,500, ALASKA 13,500/1,000/1,700 seen + 2,000 unseen (train/val/test).

## Design

- **Settings:** opt-in `checkpoint_validation='train_matched'` plus `validation_recipe='<id>'`. Existing modes (`standard`, `combined`, `capped`) and their defaults do not change.
- **Files:** tracked `data/val/<recipe-id>/{recipe.json, pairs.csv, summary.json}`. Never overwrite; a semantic change makes a new directory.
- **Builder:** `scripts/build_validation.py --notebook PATH [--check]`. It reads literal settings from the notebook and does not execute it. `--check` rebuilds and must be identical.
- **Recipe identity:** SHA-256 of canonical JSON: schema/sampler version; training tile size; overlap-last geometry version; sources and normalized quotas; resolved embedding config (k, `k_weights`, bpp, `bpp_mixture`, stride, start, payload, ...); pairs fraction and rounding rule; selection seed; source manifest and split hashes. Exclude model, LR, epochs, workers, hardware, `run_seed`, and YAML input/output paths. Keep the raw YAML SHA as provenance. SRM-30, SRM-e50 and 256 KV share one recipe; `srm-hard` has its own.
- **Rows:** pair id, cover id, source photo, source, split, tile size, grid row/col/x/y; k, bit mode, method, placement, stride, payload, requested bpp and mixture component, payload bytes, effective bpp, clipping, embedding seed key. `summary.json` holds the CSV SHA, row counts per source, and realised k/bpp/type histograms.
- **Selection:** exact per-source quotas; uniform sampling of unique validation tiles without replacement within each source; stable order; fixed seed 1 in its own domain (`train-matched-val-v1`, `select`, source). Settings and embedding use the existing OTF `purpose='val'`, epoch 0, eval seed 1, drawn from the training config through the shared draw functions.
- **Geometry:** explicit 256 overlap-last validation adapter. Do not set global `tile_size=256` or repoint `evaluation_config`; keep the legacy restrictions (`lsb_torch.py:62-63`, `lsb_mode1.py:49-59`) unchanged.
- **Runtime:** refuses a missing or mismatched recipe and never builds or writes the list during training. The run record stores the recipe id, full hash and pair-list SHA. QUICK uses a deterministic source-balanced subset and records the subset hash and the parent hash.
- **Metrics:** per-source validation metrics driven by the source list in the new mode (today hardcoded in `cvdl/lit.py:75-78,94-103`). Checkpoint metric stays minimum P_E. Legacy output unchanged.
- **Evaluator:** uses the run's saved validation by default. An explicit `--validation-recipe ID` rescoring path is optional in this task (only if small); it records original vs requested recipe and writes to a fresh output folder.
- **Compatibility:** legacy branches add no record keys, files or RNG draws. `refactor_golden.py compare` stays ALL EQUAL without recapture. Old capped hash `6bb3c8ef...` unchanged.

## Implementation and use

The builder is [`scripts/build_validation.py`](../../scripts/build_validation.py).
It reads literal notebook settings without executing the notebook. It resolves the
full training recipe, not the PC QUICK subset. It does not read image pixels.
The runtime and replay adapter are in [`cvdl/validation.py`](../../cvdl/validation.py).

| Data recipe | Frozen id | Pairs | ALASKA / LSDIR / RAISE |
| --- | --- | ---: | --- |
| SRM, SRM-e50, 256 overlap KV | [`60da4391043e8ce48ff2`](../../data/val/60da4391043e8ce48ff2/) | 18,000 | 7,200 / 3,600 / 7,200 |
| SRM-hard | [`bf7ede374cab01f73f5d`](../../data/val/bf7ede374cab01f73f5d/) | 21,600 | 7,200 / 7,200 / 7,200 |

Rounding: round the total `training_pairs / 10` to the nearest integer, with
half rounded up. Multiply this total by each normalized source weight. Take
the integer parts, then assign the remaining pairs to the largest fractional
remainders. Break ties by source name. Thus the total is exact and each source
quota differs by less than one pair from its ideal share. All approved source
shares are exact. `recipe.json` records this rule and the full training pair count.

To verify both committed lists on the PC:

```sh
podman exec -w /work aai3001 python scripts/build_validation.py --notebook notebooks/experiments/srm.ipynb --check
podman exec -w /work aai3001 python scripts/build_validation.py --notebook notebooks/experiments/srm-hard.ipynb --check
```

Without `--check`, the builder creates a missing recipe directory. It refuses
to replace an existing directory with different bytes. Runtime never builds or
writes these files. A missing or mismatched recipe stops data construction.
The full CSV stores the settings draw, embedding seed and coordinates for replay.
`summary.json` stores file hashes, input-pool counts and realized histograms.

For a **new** training notebook, keep its training data settings and add:

```python
checkpoint_validation='train_matched',
validation_recipe='bf7ede374cab01f73f5d',  # only with the SRM-hard data recipe
```

Do not use an id for a different training recipe. Model, LR, epoch count, worker
count and training seed do not change the id. Changes to embedding settings,
source mix, training pair count, tile geometry or frozen source records do.
The existing `tile_size=512` legacy test setting stays unchanged; validation
uses `train_tile_size=256` through the separate adapter. No validation augmentation
is applied. QUICK records both the full-list hash and its deterministic
source-balanced subset hash.

The evaluator reads `validation_recipe` from saved run settings by default.
The explicit `--validation-recipe` rescoring option is **deferred**. Per-source
threshold tables still use their legacy calibration panels; changing those
panels is work list #4, not part of this implementation. No DIV2K panel was built.

## Out of scope

- Editing existing notebooks (incl. `srm-hard.ipynb`; that is work list #7).
- New 256 standard test (#2), hard test at 256 (#3), per-source thresholds and the DIV2K calibration panel (#4).
- Any G4 or training run.

## Verification

Implementation checks on 2026-10-18: **286 tests passed**; the normal golden
compare printed **ALL EQUAL** without recapture. Both builders passed `--check`
twice. Six real pairs per recipe replayed with identical bytes through 0-worker
and 2-worker loaders. The old 8,035-pair capped validation hash is unchanged.
Initial metadata build times were 1.093 s (SRM) and 1.896 s (SRM-hard); these
are not inference or validation-pass timings.

- `podman exec -w /work aai3001 python -m unittest discover -s tests` passes (270 before).
- `podman exec -w /work aai3001 python scripts/refactor_golden.py compare` prints ALL EQUAL.
- Builder `--check` twice gives identical files; old capped hash unchanged.
- CPU replay: a few pairs of each new list through the loader with 0 and 2 workers give identical bytes.
- Tests: recipe hash canonical and order-independent; cosmetic YAML changes keep the id, semantic changes change it; mismatch rejection; no runtime writes; exact quotas, unique tiles, val-only photos; quota above pool fails; independence from training seed, epoch, model and workers; settings replay incl. weighted k and bpp mixture; QUICK subset hashes.
