# 2026-10-25

## Run 19 (#20 arm B, B36) delivered

### Did

- Checked run 19 attempt 3 `srm-e50-paired-coloursep36-20261024T115642Z-1325`: done, exit 0, 50 epochs (140,650 steps), 11:57:46-15:36:37Z (about 219 min, about $6); no preemption; VM TERMINATED, 300 min / STOP. Best `best-epoch48-val_pe0.0572-val_auc0.9860.ckpt` (capped validation; R1 .0444 / .9917). First delivery with the permission fix `bf84b6c`: all Pi files mode 644 (no `sudo chmod` needed).
- Run_Supervisor copied it to the PC: 24 files, Pi = PC SHA for best `2f45fb4d…f0d5` and last `de703a96…c028` (matches `attempts.jsonl`); wall 218.7 min; 12.04 steps/s for the whole run (R1 17.8-18.4). Then Worker_4: calibration mapping entry, evaluation, `wiser_b` verdict vs R1, #11 context re-score.
