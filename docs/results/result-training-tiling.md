# Result: training tiling (6 arms, 2026-10-16/17)

Design: [plan-training-tiling](../active/plan-training-tiling.md) and [plan-tiling-256](../archive/plan-tiling-256.md).
Runs and job IDs: [run queue](../active/run-queue.md). One seed per arm.

## Setup

- Recipe newdata-1 (wide XuNet w4, no normalisation, LR 3e-4, 30 epochs, mix RAISE 0.4 / LSDIR 0.2 / ALASKA 0.4, k 1-4).
- Arms change only the training tile grid: **overlap** (last tile pushed back inside the photo), **trim** (centre, partial tiles dropped), **pad** (centre, partial tiles zero-padded after embedding; the clean twin is padded the same).
- 512 arms: batch 32, 45,000 tiles/epoch. 256 arms: batch 128, 180,000 tiles/epoch (same pixels per step, 4x more distinct tiles), same LR.
- Validation and all tests are frozen 512 tiles (standard test `fdd989f7...`, capped validation `6bb3c8ef...`). All 9 frozen identities are equal across the 6 arms.
- Evaluation on the PC. From 256 trim on: one tool, `scripts/evaluate_run.py` at `c13ee00` (proved exactly equal to the earlier pinned scripts on 256 overlap: max |delta prob| 0.0).

## Test P_E (%, minimum P_E over thresholds on each test panel; test-optimised, not validation-calibrated)

> Correction 2026-10-18: these P_E values are the **minimum P_E over thresholds on each test panel itself** (test-optimised, optimistic; `cvdl/metrics.py:84-96`), not validation-calibrated and not threshold 0.5. Standard test, P_E % (fixed 0.5 / capped-validation min-P_E cut-off / test minimum): 256 overlap 2.31 / 2.27 / 2.25; SRM-30 1.86 / 1.69 / 1.67; SRM-e50 1.67 / 1.63 / 1.57. See the [2026-10-18 journal](../journal/2026-10-18/README.md).

| Test | newdata-1 | 512 overlap | 512 trim | 512 pad | **256 overlap** | 256 trim | 256 pad |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Standard | 2.54 | 2.80 | 4.07 | 4.32 | **2.25** | 3.55 | 3.60 |
| bpp-1 | 2.09 | 2.30 | 3.51 | 3.79 | **1.90** | 3.33 | 3.11 |
| ALASKA seen | 16.62 | 16.28 | 16.28 | 16.39 | **15.75** | 16.31 | 16.20 |
| ALASKA unseen | 13.88 | 13.78 | **13.54** | 14.21 | 13.67 | 13.91 | 13.99 |
| LSDIR | 6.64 | 7.02 | 7.26 | 7.35 | 6.60 | **6.46** | 6.70 |

AUC (standard): 512 overlap .9937, trim .9915, pad .9902; 256 overlap .9965, trim .9937, pad .9936.

## Per-source 5%-FA cut-offs from validation (test FA / miss / k=1 miss, %)

| Source | 512 overlap | 256 overlap | 256 trim | 256 pad |
| --- | --- | --- | --- | --- |
| RAISE | 5.2 / 1.8 / 5.5 | 5.0 / 1.0 / 2.8 | 7.1 / 1.6 / 4.5 | 7.1 / 1.7 / 5.1 |
| DIV2K | 8.7 / 5.1 / 21.3 | 10.8 / 4.1 / 19.5 | 9.4 / 3.4 / 14.8 | 8.9 / 4.0 / 19.5 |
| ALASKA seen | 4.8 / 30.7 / 69.6 | 5.4 / 29.2 / 68.1 | 4.9 / 30.4 / 71.7 | 4.6 / 31.1 / 70.5 |
| ALASKA unseen | 4.7 / 24.7 / 64.4 | 6.7 / 21.3 / 62.3 | 6.0 / 22.2 / 61.8 | 5.2 / 24.0 / 67.2 |
| LSDIR | 3.9 / 10.3 / 49.1 | 3.5 / 9.9 / 47.3 | 3.1 / 10.2 / 44.8 | 3.3 / 10.1 / 47.0 |

## Padding check (capped: 18,000 cases, same cases for all models, `padding_pairs_sha256` `2d96da55...`)

Missed stego, unpadded -> padded, at each model's validation cut-off:

| Padding | 512 overlap | 512 pad | 256 overlap |
| --- | --- | --- | --- |
| strip 50% | 4.1 -> 9.8 | 4.6 -> 6.7 | 3.3 -> 8.5 |
| corner 50% | 5.0 -> 9.5 | 5.3 -> 7.0 | 3.0 -> 7.7 |

Zeros dilute the evidence in every model. Training with padding cuts the extra misses by about 60%, but costs accuracy on unpadded tiles.

## Conclusions

- **Overlap is the best grid at both tile sizes.** Trim and pad give about 45-60% more errors on the standard test.
- **256 training tiles improve all three arms**, although tests use 512 tiles. 256 overlap is the best model so far.
- The 256 recipe changes three things together (tile size, 4x distinct tiles, batch 128). One seed per arm.
- 256 models overshoot the 5% FA target on test for DIV2K and ALASKA unseen: keep per-source cut-offs.
- Deployment: cover the full photo with overlap tiles; do not pad.

## Speed (G4)

| Run | Steps/s | Run time | Limit |
| --- | ---: | ---: | --- |
| 512 arms | 54.6 | ~40 min | GPU |
| 256 arms | 32.7-33.4 | ~55 min | loader (main-process message receive; see `scripts/profile_loader.py`) |
