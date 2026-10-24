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
