# 2026-10-25

## Run 19 (#20 arm B, B36) delivered

### Did

- Checked run 19 attempt 3 `srm-e50-paired-coloursep36-20261024T115642Z-1325`: done, exit 0, 50 epochs (140,650 steps), 11:57:46-15:36:37Z (about 219 min, about $6); no preemption; VM TERMINATED, 300 min / STOP. Best `best-epoch48-val_pe0.0572-val_auc0.9860.ckpt` (capped validation; R1 .0444 / .9917). First delivery with the permission fix `bf84b6c`: all Pi files mode 644 (no `sudo chmod` needed).
- Run_Supervisor copied it to the PC: 24 files, Pi = PC SHA for best `2f45fb4d…f0d5` and last `de703a96…c028` (matches `attempts.jsonl`); wall 218.7 min; 12.04 steps/s for the whole run (R1 17.8-18.4). Then Worker_4: calibration mapping entry, evaluation, `wiser_b` verdict vs R1, #11 context re-score.
- Worker_4: calibration mapping entry `b54f63e` (`60da4391043e8ce48ff2`, `post_hoc_256`; the 12 older entries byte-identical; FAST 547 pass). Evaluation `1bf72fdd2579bc0da80c` (batch 32, conv1 workaround off, 4 workers); best is epoch 49 of 50. Evidence `runs/verification/wiser-b-eval-2026-10-25/`.

### Found

| Measure | R1 | B36 (run 19) | B36 − R1 [95%] |
| --- | ---: | ---: | --- |
| Standard P_E | 2.55% | 3.66% | +1.11 points [+0.80, +1.44] |
| Standard AUC | .9964 | .9928 | −.0037 |
| Hard k=1 low (primary) AUC | .757 | .693 | −.064 [−.072, −.055] |
| Hard k=1 low P_E | 36.2% | 40.7% | |
| Hard all panels AUC | .850 | .805 | |

- **Rule `wiser_b`: FAIL.** Route a (standard better) and route b (hard better) each fail all three checks.
- #11 panel re-score, context only (`runs/verification/rgb-channel-b36-context-2026-10-25/`): red/green/blue-only AUC .614 / .730 / .594 vs run 18 .660 / .775 / .662. B36 does not close the red/blue blind spot; it is worse on all three.

### Decided

- #20 arm B is closed: delaying colour mixing hurts our LSB detector (and costs 1.5x the G4 time). Early mixing in conv1 stays. Arm A (learnable SRM) waits for a user decision.

### Next

- User: #20 arm A or close #20; the Pi watchdog setup (step 1 audit).
