# Ideas board

Status (2026-10-24): open ideas only. Done and closed items (#1, #2, #2b, #3, #9, #10, #16, #17, #19) moved to [done ideas](../archive/ideas-board-done.md). Nothing here starts without a user decision. Created 2026-10-18.

Evidence: the dated [journal](../journal/README.md); the [run queue](run-queue.md) has every run.

## Experiments (need training)

| # | Idea | Why | Evidence / risk | Cost | State |
| ---: | --- | --- | --- | --- | --- |
| 20 | **Separate-then-reunion colour front end** (WISERNet idea, Zeng et al., IEEE TIFS 2019; raised by the user 2026-10-24, kept as a viable try): keep the RGB colours apart longer before mixing them | Photo content is strongly correlated across R, G and B; our LSB changes are not (the generator writes independent bits into the R, G and B LSBs of each chosen pixel, `lsb_engine.py`). A layer that adds colours together grows content about 3x but independent noise only about 1.7x. Camera noise is also correlated across colours after demosaicing, so colour relations might help the noisy-camera false alarms (a hope, not shown) | SRM-e50 already filters each colour separately (`SRMFilter`, 30 frozen kernels per colour, 90 maps), but `conv1` (90 -> 16) mixes all colours at once. A learned layer could learn colour differences by itself, so the gain is an empirical question. Published gains were on other embedding methods, not LSB replacement. Two separate tests: **(B) delay mixing**: `conv1` per colour (`groups=3`), mixing from `conv2` (small code change, cheaper conv1, the core claim; suggested first); **(A) learnable SRM kernels** started from SRM (overlaps #10's learnable residual filters; care with the FP32 guard and the clamp at 3; slow PC pilot). Not covered by arm A (that delayed spatial pooling) | Per arm: R1 recipe (paired), 50 epochs from scratch, about 2.5 h, about $4.2; frozen rule vs R1 (hard k=1 AUC + standard guard) plus noisy-camera FA | Idea (user 2026-10-24); no plan yet |
| 4 | **Untangle `srm-hard`**: one run that changes only the k/bpp mix (keep 40/40/20 photos), or only the photo shares | `srm-hard` changed both; we cannot tell which caused the small k=1 gain and the higher ALASKA false alarms | vs SRM-30: hard k1 low+noisy AUC standard .905 vs .882, LSDIR .671 vs .651; ALASKA seen P_E 21.95 vs 20.03 | 1 G4 run, ~$3 | To review |
| 5 | **Seed repeats** (e.g. SRM-e50 x 2 extra seeds) | Several model gaps are under 1 point; one seed cannot show whether they are real | Pending since 2026-10-17. Declined for the architecture tests (user 2026-10-19); single-seed results need a minimum gain instead (D6) | 2 G4 runs, ~$6-9 | To review |
| 11 | **RGB channel analysis**: embed in R only, G only, B only, and spread over RGB at the same total bpp; test whether explicit cross-channel features help | Our payload is spread over 3 channels (~1.7% of values per channel at 0.05 bpp); unknown whether the model uses cross-channel relations | Needs a new frozen test panel (generator channel option to check) and inference; pairs well with item 1's per-channel statistics | Build + GPU inference ~15 min per model | To review (external suggestion) |
| 12 | **CNN + statistical features** (**raised** by check 1, 2026-10-19; **on hold by the user 2026-10-19**: feels close to cheating for a CNN project, since hand-built replacement statistics would do the work; kept on the list): add classic replacement statistics as inputs, or a separate fusion classifier | Only if item 1 finds signal the CNN misses | Item 1: WS beats SRM-e50 in 10 of 12 noisy ALASKA replacement cells; WS score drifts with noise | Code + 1 G4 run | On hold (user) |
| 18 | **More epochs / early stopping** (raised by the user 2026-10-22): up to 80 epochs with early stopping, or plateau-driven LR drops then an early stop | SRM-30 -> SRM-e50 helped (3.07% -> 2.54%), but the gain came from a longer middle LR phase: after the last drop (epoch 33) SRM-e50 val P_E stayed flat .043-.048 and train loss moved .226 -> .217; run 11 (+15 epochs at LR 3e-5) gave nothing | Fixed drops + early stop saves only 1-2 epochs; an adaptive schedule could help but needs its own matched test; validation P_E is noisy (epoch 18 jump to .084), so patience must be several epochs; about 245-290 min, no resume | about $7-8 | **shelved by the user 2026-10-22** until evidence shows more epochs are needed; #17 stays at a fixed 50 epochs |
| 6 | Front-end upgrade / BF16 / longer training | Older pending items. Arm A (2026-10-19) costs 2.3x compute, so BF16 matters for heavier models | Hold until item 5 is settled; front-end ladder in the [architecture plan](plan-architecture-tests.md) | - | On hold |

## Photo routing

| # | Idea | Question it answers | Evidence | First steps | State |
| ---: | --- | --- | --- | --- | --- |
| 15 | **Route by processing history** (user idea, 2026-10-21): a front-end classifier (decompressed JPEG with quality/table estimate, JPEG then resized, never compressed, synthetic) sends each photo to an expert (JPEG-compatibility, resampling-aware, the current noise CNN), with soft mixture weights. The router must use features that survive embedding (block grid, table estimate) | Would routing help on real JPEG-origin uploads? | Our weakest panel (ALASKA) is never compressed, so routing would not change it. DIV2K/LSDIR may be JPEG-origin (not checked) | First checks if resumed: a history audit of the sources, and a JPEG-origin stress test of run 13 (about 1 h on the PC) | **Dropped for now (user)** |

## Evaluation fixes

| # | Idea | Why | State |
| ---: | --- | --- | --- |
| 13 | **Save the logit margin in the evaluation** (next to `prob_stego`) and use it for AUC and thresholds | float32 softmax saturates: 26% of SRM-e50 validation scores are exactly 1.0. With mean pooling these are almost all stego (52.6% of stego, 0.0% of clean exactly 1.0; clean >= 0.999: 0.1%), so clean-vs-stego ties are rare and AUC and the 0.45 threshold are barely affected. It matters for max/top-k pooling, where clean tiles can saturate too. Low priority for the main evaluation | To review (found in check 2, 2026-10-19) |
| 14 | **Test the current compile guards** (added 2026-10-20 from the test audit): add direct tests, with mocked compile and backend calls, for the compile and cache checks in `cvdl/settings.py` and the in-place compile placement in `cvdl/runtime.py` | The G4 runs with `compile=default`, but only frozen notebook 04's compile snippets have tests (`tests/test_notebook_compile.py`); the current guards have none | To review (user: keep on the board for now) |

## Report topics (no work)

| # | Topic | Evidence |
| ---: | --- | --- |
| 7 | One deployment threshold costs ALASKA | SRM-e50 at its validation cut-off: ALASKA seen FA 16.3%, noisy ALASKA hard cells FA 35-38% |
| 8 | Training sources matter more than architecture for unseen data | wide-R2 (RAISE+DIV2K only) ALASKA seen P_E 32.65%, FA 44.4% |
