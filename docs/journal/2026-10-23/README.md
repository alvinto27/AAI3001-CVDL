# 2026-10-23

## Paired training runs start

### Did

- User: start the runs. R1 (`srm-e50-paired.ipynb`, `d28e2bb`, run queue row 15) handed to Run_Supervisor (GCE limit 300 min / STOP checked, no change). Worker_2: decision tooling for the frozen R1, R2 and curriculum50 rules while R1 trains.
- Run_Supervisor: R1 `srm-e50-paired-20261023T022339Z-4518` launched 02:23:39Z after a dry-run question (`embedding_schedule` None, as srm-e50; the Orchestrator's brief had said "original" by mistake). Epoch 1: 18.4 steps/s, about 3.0 min per epoch (same as SRM-e50); projection about 156 min.
- Worker_2: `781a9bf` `scripts/training_decision.py` with the frozen rule sets `paired_r1`, `curriculum50`, `paired_r2` (standard ΔP_E at each model's own cutoff and hard k=1 low ΔAUC, 2000 shared photo-cluster draws; R2 head-vs-head cell AUC; curriculum50 checks the schedule identity before the epoch >= 8 condition). Reproduces the recorded run 10 (+.018 [+.013, +.023]) and run 11 (−.001 [−.005, +.002]) intervals and the run 13 cell gain; arm-A outputs byte-identical. Also: curriculum50 runs can now be evaluated at `eval_batch_size=32` (evaluation layer only; training code byte-identical to `77f0ecd`). FAST 446 pass.
- R1 done: 02:24:30-04:57:40Z (153.2 min, about $4.2), exit 0. Best epoch 39, capped val P_E .0444, AUC .9917 (SRM-e50: epoch 43, .0432, .9917). Delivered, SHA-256 Pi = PC. Worker_2: evaluation and frozen verdict.

### Found

- R1 evaluation `6ef349895655136bef90` (best epoch 39, SHA checked). Deployment cutoff from `60da` through a new mapping entry (`cbc124e`, approved; treated as SRM-e50; old 10 entries unchanged).

| P_E % | SRM-e50 | R1 paired |
| --- | ---: | ---: |
| Standard-256 | 2.54 | 2.55 |
| ALASKA seen | 19.42 | 18.99 |
| ALASKA unseen | 16.30 | 15.58 |
| LSDIR | 9.39 | 8.84 |
| Standard AUC | .99637 | .99642 |
| Hard k=1 low AUC (pooled, 3,024 pairs) | .7450 | **.7574** |

- Frozen rule `paired_r1`: hard ΔAUC +.0123 [+.0081, +.0168]; standard ΔP_E +.005 points [-.164, +.174]. Route a fails (no standard gain), **route b passes**. Final: **PASS** (a hard-case gain, not a standard gain).
- Context (k=1 noisy-camera false alarms, own cutoffs): most fall a little (seen noisy 34.6 -> 32.3%), but unseen noisy rises 30.3 -> 31.9% and LSDIR low noisy 5.9 -> 6.6%. Noisy ALASKA false alarms stay high.

### Decided

- By the frozen rule: R2 starts from R1 best (`b00cd4b6...f180`); the curriculum run uses the paired variant (run queue 16, 17).
