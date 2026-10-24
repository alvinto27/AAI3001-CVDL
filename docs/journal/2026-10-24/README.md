# 2026-10-24

## Submission model changed to run 18

### Did

- Showed the user run 18's cell figure (`runs/verification/cell-curriculum50-paired-2026-10-23/cells/cells.png`): an exact hit on the RAISE contiguous band, a partial hit on the ALASKA contiguous band, flat on the random-placement tile (as for run 13 and R2).

### Decided

- **User: run 18 is the submission model for now** (`srm-cell-aux-curriculum50-paired-20261023T133537Z-1366`, `last.ckpt`, SHA-256 `eee9c63a...0b80`, eval `b3d9a49137b5dbd7bd5a`, tile cut-off .5094). It replaces run 13.
  - Standard P_E 2.62% (run 13 2.97%, R2 2.43%); cell head .867 (run 13 .863, R2 .860); contiguous cell recall / FA 51.5% / 3.8% (R2 52.7% / 4.5%).
  - It failed its frozen rule against R2 (no established difference either way); the choice is the user's.

- Worker_2: photo verdict rule (c) re-fitted for run 18 (`0a5f417`: only model and cut-off source parameters; `fit()` and `score()` identical to `da411c1`; serial batch 8 as for run 13). Same 448 validation + 576 TEST photos; GPU cache 24.7 min; rule frozen (SHA-256 `05d61c52...9bf61a`, logit cut-off -.0564, calibration FA 11/224 = 4.9%) before the TEST cache was read.

### Found

| TEST photos, rule (c) | Run 13 | Run 18 |
| --- | ---: | ---: |
| Clean photos flagged | 8.5% | 8.7% |
| Messages missed | 9.3% | 8.4% |
| P_E / AUC | 8.9% / .960 | 8.5% / .965 |
| ALASKA unseen FA | 17.2% | 16.4% |
| Hard k=1 0.05-0.15: miss / P_E | 38.1% / 23.0% | 33.0% / 20.5% |

- Marks as for run 13: FA <= 7% fails (50 of 576), hard P_E <= 25% passes. Tile scores still add value over cell features (P_E -1.7 points [-2.5, -0.8]). Differences between the models are point estimates only.

- Bundle step: the `pipeline` builder refused the yx-written photo rule (its identity record has checked cut-off sources instead of `map_kind` / `evaluation_id`). Orchestrator chose a small builder change over a re-fit, so the tested rule stays exactly the same. Worker_2 stopped responding during this work; Worker_3 reviewed the uncommitted work and finished it.
- `pipeline` `bfbbdcc`: builder route `yx_cutoff_sources` (accepted only if checkpoint SHA, tile threshold ID and value, cells-summary hash and cell cut-off all match; one refusal test per mismatch); **run 18 bundle `deploy/run18-rule-c-v1/` is the shipped default** (`deploy.json` `81723296...3dcb`, `weights.pt` `fb8d1957...0650`); the run 13 bundle stays selectable with `--bundle`. FULL 411 pass.
- Checks: an independent rebuild is byte-identical; 6 validation photos give the same verdicts as the frozen run 18 values (max rule-logit difference 2e-6); the fresh CPU container answers `/analyze` on DIV2K 0281 with `not_detected`, as the reference.

- Cause of the format mismatch: the photo scripts were made general twice, separately (`pipeline` `0968a34` on 2026-10-21; `yx` `0a5f417` today), because the re-fit task went to `yx` without pointing at `pipeline`'s version. User chose to unify now.
- Worker_3 unified them: `pipeline` `9adb589`, then `yx` `c05f5c0`. Six files are byte-identical on both branches (`cvdl/photo_inference.py`, `cvdl/pixels.py`, `scripts/photo_heatmap.py`, `scripts/photo_verdict_cache.py`, `scripts/photo_verdict_fit.py`, `tests/test_photo_verdict.py`), based on `pipeline`'s version: `--run` + `--evaluation-id` replace `--tile-calibration`; new caches keep the native identity fields and add the hashed cut-off sources; a local SHA helper removes the `cvdl.pipeline` import. pipeline FULL 415 pass; yx FAST 436 pass. On both branches the run 13 and run 18 photo lists reproduce byte for byte (select-only, CPU); frozen rules, both bundles and the builder are unchanged.

### Next

- Later: merge `yx` / `pipeline` into `main`.

## Ideas board clean-up and next item

- Ideas board: done items moved to `docs/archive/ideas-board-done.md`; #15 copied from `pipeline`; #4 and #5 dropped (user); #6 merged into #18 (user); #20 added: separate-then-reunion colour front end (WISERNet, Zeng et al. 2019), kept as a viable try. The board is identical on both branches.
- User: do #11 (RGB channel analysis, test only) first, before #20. Worker_3: read-only plan.
- #11 plan (Worker_3, scratch `runs/.scratch/rgb-channel-plan-2026-10-24/PLAN.md`): 480 test photos (48 noisy + 48 other per source/split), one tile each; k 1/2; 0.05/0.15/0.50 bpp; contiguous and random; the same message per photo in every variant. Variants: R only, G only, B only, same-pixel RGB (main), and today's RGB layout built fresh (bridge; it covers about 1/3 as many pixels). A correlated-change control was dropped: LSB replacement cannot give equal changes in R, G and B without biasing the pixels. Frozen rule on run 18 (k=1, random, 0.05/0.15): a colour effect needs |dAUC| >= .02 with a 98.3% interval excluding 0; all within +/-.02 = no large effect; wide = inconclusive.
- Orchestrator to the user: #11 can show a colour blind spot but cannot show whether #20 would help (density and colour relations are tangled). **User: do #11 anyway (curiosity).** Amendment: estimate the interval width from existing scores first; use 2 tiles per photo if 48 photos per group are too few.

## #11 RGB channel analysis: result

### Did

- Worker_3: code `9361d8b` (an explicit-unit replacement entry point in `lsb_engine.py`, old generator output byte-identical on 192 recorded cases; `cvdl/channel_analysis.py`; build and evaluate scripts). FULL 484 pass. Panel frozen before scoring (recipe SHA-256 `3604b153...6e11`): 480 test photos, one tile each, 48 per source/split/noise group; 5,760 stegos per arm; 29,280 images per model. Width check from existing scores: expected adjusted half-width .019 (kept one tile per photo; actual .009).
- PC GPU scoring, about 29 s per model (R1, run 18, SRM-e50), then the frozen report once. Output `runs/verification/rgb-channel-analysis-2026-10-24/`.

### Found

Primary aggregate (k=1, random, 0.05 + 0.15 bpp, mean AUC over the 10 source/noise groups):

| Model | R only | G only | B only | Same-pixel RGB | Today's RGB layout |
| --- | ---: | ---: | ---: | ---: | ---: |
| Run 18 (primary) | .660 | **.775** | .662 | .725 | .721 |
| R1 | .657 | .783 | .645 | .718 | .716 |
| SRM-e50 | .629 | .772 | .641 | .707 | .704 |

| Run 18 contrast vs same-pixel RGB | dAUC | 98.3% interval | Miss / P_E change at the frozen cut-off |
| --- | ---: | --- | ---: |
| R only | -.065 | [-.074, -.056] | P_E +6.0 points |
| G only | +.050 | [+.041, +.058] | P_E -5.1 points |
| B only | -.063 | [-.072, -.054] | P_E +4.8 points |

- **Frozen rule verdict (run 18): EFFECT** for all three colours. The same pattern holds for R1 and SRM-e50: a message only in red or only in blue is clearly harder to find; only in green, clearly easier.
- R only vs G only vs B only is a fair comparison (same pixels, same density), so the models really are more sensitive to green than to red and blue. Possible cause (not tested): camera sensors have the least noise in green (two green sites per Bayer block, highest sensitivity) and the most in blue, so a change stands out most in green.
- Today's RGB layout vs same-pixel RGB: no clear difference (per-rate intervals include 0), so the pixel footprint of today's generator does not matter here.
- Run 18 cell head (contiguous k=1): the same order (G .78-.82, R .68-.74, B .65-.71 within-tile AUC); clean-cell FA about 5%.
- Limits: single-colour vs RGB mixes 3x density with colour relations, so the RGB contrasts do not show whether #20 would help. Test photos were used before.

### Decided

- Blind spot to report: hiding only in red or blue raises the miss rate by about 5-6 points for all three models. No model or threshold change from this check.

## #20 started

- Ideas board clean-up: #11 moved to done; #15 and #14 dropped (user). Open: #20, #12 (on hold), #18 (shelved), #13, report topics #7 and #8.
- User: work on #20 now. Worker_3: read-only plan for arm B (conv1 per colour, mixing from conv2), R1's recipe, frozen rule vs R1.
- #20 arm B plan (Worker_3; Worker_3 then stopped responding): B36, 12 learned maps per colour in a grouped conv1, mixing from conv2 (B18 = 53% compute, B48 = 127%). **User approved** with: width 36; rule `wiser_b` = both `paired_r1` routes vs R1 (the plan had only the hard route); #11 panel re-scored as context. Native grouped conv1 on ROCm (the conv1 workaround supports groups=1 only). Worker_4: implementation, FULL, PC pilot; stop before the G4.
- Worker_4: B36 code `68b7c3b` + `e77538e` (13-line model class; notebook = R1's with name + model changed; native grouped conv1 on ROCm; rule `wiser_b` also checks the same training data as R1 and a complete 140,650-step run; #11 context route verified all 29,280 images byte-identical). FULL 500 pass. PC pilot: grouped conv vs CPU FP64 worst error 4.7e-7 (limit 1e-4); batch 16: B36 27.5 steps/s, 1.2 GiB vs R1 architecture 14.6 steps/s, 9.6 GiB (the R1 architecture needs the slow ROCm conv1 workaround; not a G4 forecast); finite loss and gradients; compile + reload pass. Run 19 launch handed to Run_Supervisor (user told).
- Run 19 attempt 1 `srm-e50-paired-coloursep36-20261024T100818Z-8639` failed after 56 s (exit 1; about $0.1): `RuntimeError: Expected canUse32BitIndexMath(input) && canUse32BitIndexMath(output)` in the compiled Lightning sanity-check validation step, at a `groups=3`, bias-free convolution (most likely the frozen SRM conv, not the new conv1). R1 ran the same SRM conv with no error. Hypothesis: 512-tile validation batches exceed 32-bit indexing on the native depthwise kernel, and the grouped conv1 changes the compiler's layout choice. Worker_4: diagnosis and a fix proposal; no relaunch before review.
- Run 19 diagnosis (Worker_4, torch 2.14.1 sources): the failing op is the frozen SRM conv (output 128 x 90 x 512 x 512 = 3.02e9 elements > 2^31) in the 512-tile sanity validation. On NVIDIA, FP32 NCHW depthwise goes to the native kernel, which needs 32-bit indexing of the output; the cuDNN check looks only at input and weight. R1's compiled inference graph gets channels-last (cuDNN), but B36's grouped conv1 tips Inductor's inference layout estimate to NCHW. Compile off would not help. Training (256 tiles) and post-training evaluation (32 images) are below the limit. Fix chosen (Orchestrator): an explicit opt-in evaluation batch cap `val_batch_images=64` in the B36 notebook only; training recipe unchanged; `wiser_b` permits only this field to differ.
- Run 19 attempt 2 `srm-e50-paired-coloursep36-20261024T112444Z-2678` (`3e24f72`): the evaluation cap worked (validation passed). **Stopped by the frozen speed rule** at epoch 3: 12.10 steps/s vs R1 17.8-18.4 (B36 forward arithmetic is 98% of R1, but the grouped full-resolution conv1 runs on slower library paths); about 215-220 min projected (rule: 180). About $0.76; attempt 1 about $0.1. Epoch-4 checkpoint on the VM disk. Pending (user): run longer, try a speed fix, or close #20.
- **User: option A**: rerun B36 fresh at the slower speed with longer limits (`--timeout-min 260 --max-run-minutes 280`; about 215-220 min, about $6), unsupervised overnight; the speed stop rule is lifted for this run (a cost rule, not a result rule). Not chosen: a short channels-last speed test first (about $0.3). A preemption tonight is not resumed until the next session.
