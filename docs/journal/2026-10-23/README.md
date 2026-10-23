# 2026-10-23

## Paired training runs start

### Did

- User: start the runs. R1 (`srm-e50-paired.ipynb`, `d28e2bb`, run queue row 15) handed to Run_Supervisor (GCE limit 300 min / STOP checked, no change). Worker_2: decision tooling for the frozen R1, R2 and curriculum50 rules while R1 trains.
- Run_Supervisor: R1 `srm-e50-paired-20261023T022339Z-4518` launched 02:23:39Z after a dry-run question (`embedding_schedule` None, as srm-e50; the Orchestrator's brief had said "original" by mistake). Epoch 1: 18.4 steps/s, about 3.0 min per epoch (same as SRM-e50); projection about 156 min.
- Worker_2: `781a9bf` `scripts/training_decision.py` with the frozen rule sets `paired_r1`, `curriculum50`, `paired_r2` (standard ΔP_E at each model's own cutoff and hard k=1 low ΔAUC, 2000 shared photo-cluster draws; R2 head-vs-head cell AUC; curriculum50 checks the schedule identity before the epoch >= 8 condition). Reproduces the recorded run 10 (+.018 [+.013, +.023]) and run 11 (−.001 [−.005, +.002]) intervals and the run 13 cell gain; arm-A outputs byte-identical. Also: curriculum50 runs can now be evaluated at `eval_batch_size=32` (evaluation layer only; training code byte-identical to `77f0ecd`). FAST 446 pass.
- R1 done: 02:24:30-04:57:40Z (153.2 min, about $4.2), exit 0. Best epoch 39, capped val P_E .0444, AUC .9917 (SRM-e50: epoch 43, .0432, .9917). Delivered, SHA-256 Pi = PC. Worker_2: evaluation and frozen verdict.
