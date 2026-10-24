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

### Next

- Worker_2: build the run 18 deploy bundle on `pipeline` and make it the shipped default (run 13 bundle stays selectable).
