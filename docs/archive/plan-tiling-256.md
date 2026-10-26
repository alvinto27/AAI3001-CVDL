# Tiling at 256 pixels: three arms and a 256 test panel (design)

Status: ~~three 256 arms prepared, not launched~~ **done**: runs 4-6 (2026-10-17), [result](../results/result-training-tiling.md). ~~The 256 test panel is design only~~ superseded by the built [standard-256 suite](../active/plan-standard-test-256.md) and [hard-256 test](../active/plan-hard-test-256.md).
Parent record: [training tiling](../active/plan-training-tiling.md).

## What changed in code

| Item | Change |
| --- | --- |
| Setting `train_tile_size` | New, default `None` (= `tile_size`). Value 256 needs an explicit `train_grid`. |
| `tile_size` | Stays 512. Validation, tests and extra panels still use it. |
| `Corpus(train_tile=...)` | Used only for training pools, tile specs, plan, plan summary. |
| `data/train-grids/v1/summary-256.json` | Plan digests and pool counts at 256, all three rules. `summary.json` (512) is unchanged. |
| Preflight | Checks the 256 summary when `train_tile_size` is 256. |

The old guard "new sources require trim512 tiles" still protects `tile_size`. It is not the
training tile size, so it stays. Default behaviour and the 512 arms are byte-identical
(golden compare ALL EQUAL; preflight differs only by the new inert key `train_tile_size`).

## Budget (same pixels per step and per epoch as 512)

Batch 128 images at 256 = 32 images at 512 (pixels). `tiles_per_epoch` 180,000 at 256 = 45,000 at 512.
Other settings unchanged (30 epochs, LR 3e-4, drops 5×0.3 and 20×0.1, k 1–4, mix 40/20/40, seed 0).

Pool sizes at 256 (training photos only):

| Rule | RAISE | ALASKA | LSDIR |
| --- | ---: | ---: | ---: |
| overlap-last / centre-pad | 170,356 | 121,704 | 140,863 |
| centre-trim | 148,782 | 84,556 | 86,256 |

Draws per epoch: RAISE 72,000, ALASKA 72,000, LSDIR 36,000. The smallest pool
(ALASKA trim, 84,556) is larger than its draw, so **no pool cycles**.

Steps per epoch (computed with the real schedule): 512 arms 2,813 (1 worker) / 2,814–2,816 (6 workers);
256 arms 2,813 / 2,815. Equal.

## 256 test panel (design only)

Goal: score the 256 models and the 512 models on the same frozen test photos cut at 256.
Every model is fully convolutional, so it can read 256 or 512 tiles.

Needed:

1. **Evaluation grid at 256.** Recorded ALASKA/LSDIR starts are `trim512` and cannot be reused.
   Proposed rule: split each frozen 512 test tile into its four 256 quadrants. Same photos, same
   pixels as the 512 panel, no overlap, no new photo selection. (Alternative: overlap-last at 256 for all
   sources; it adds tiles the 512 panel never saw.)
2. **New frozen pair list and settings.** Payload and bpp draws are keyed with `tile_size`, so 256 keys differ
   automatically. Draw them once, save the pair list, settings and SHA-256, and never change them. Expect about
   4× the pairs (about 43,800 standard, plus bpp-1 and the five source panels).
3. **Eval profile/path.** A new eval profile (e.g. `tile256_quadrants`) in `FixedPairs`/`Corpus` that does not touch
   `overlap_full_mix`; panels named like `test-256`, run by `scripts/evaluate_run.py` on the PC for all models.
4. **Reporting.** AUC/P_E, FA/miss at 0.5 and at the validation threshold, per source/k/bpp, for 512 and 256 models side by side.
   The threshold stays the one calibrated on the frozen 512 validation. A 256 validation set is not planned.
5. **Tests and cost.** Unit tests for quadrant geometry/digest; one PC evaluation pass per model (about 4× pairs, quarter
   the pixels each).
