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
- Worker_2: `59f0566` R2 notebook starts from R1 best (path on the VM `runs-out/...`, SHA `b00cd4b6...f180`), test expectation updated for exactly these two values; FULL 446 pass at `59f0566` (run in parallel with the launch; notebook-only change).
- R2 `srm-cell-aux-paired-20261023T054148Z-4299`: 05:43:44-06:32:16Z, 48.5 min, exit 0; start checkpoint found and SHA verified on the VM. Evaluation next (Worker_2). Curriculum50 paired (row 17, `77f0ecd`) handed to Run_Supervisor.
- R2 evaluation `4e5d3592a9f84387cc51` (last, SHA checked; own `60da` cutoff .4527):

| Measure | Run 13 (last) | R2 (last) | Δ [95%] |
| --- | ---: | ---: | --- |
| Standard-256 P_E % | 2.97 | **2.43** | -0.53 points [-0.84, -0.22] |
| ALASKA seen / unseen / LSDIR P_E % | 18.95 / 16.21 / 9.26 | 19.10 / 15.83 / 9.08 | context |
| Hard k=1 low AUC | .7513 | .7505 | -.001 [-.006, +.004] |
| Cell head, mean per-eligible-stego-tile AUC (128 tiles) | .8629 | .8597 | -.003 [-.015, +.009] |
| Cell recall / FA at own 5% validation cutoff (all placements) | 15.9% / 3.8% | 16.9% / 4.8% | context |

- Frozen rule `paired_r2`: head gain +.02 FAIL, lower bound > 0 FAIL; standard guard, hard guard, cell FA <= 7% pass. **Final: FAIL** (no localization gain). The standard tile score improved clearly; R2's standard P_E (2.43%) is also below SRM-e50's 2.54% (not a frozen comparison).
- Curriculum50 paired preempted (spot) at 08:26:10Z, about epoch 37 of 50; nothing delivered; fresh relaunch 1 of 3 (same commit and command). About 105 min / $2.9 lost.
- Orchestrator error found: G4 resume had been shelved together with arm G (2026-10-22), although it is not GBRAS-specific. User: restore it as ideas #19; Worker_2 read-only plan.
- Worker_2 resume plan (12-20 h). Found: `last.ckpt` saving is not fully crash-safe (fsspec temporary file in the default temp folder, then a move; no fsync). User: **slim version** approved ([plan](../../active/plan-g4-resume.md)), about 4-6 h.
- Worker_2: slim resume `58aa081`: `submit.sh resume <job>` (retained GCE metadata, then the disk job record checked on boot; attempt counter), resume only from the interrupted state with the same commit, notebook and settings and a readable `last.ckpt` (SHA recorded); `DurableCheckpointIO` for every checkpoint save (same-folder temp, fsync, `os.replace`, folder fsync); `attempts.jsonl`; same Pi folder. Bug fixed on the way: on a service stop the old runner could record an interrupted run as exit 0; now 143 / interrupted. A CPU run interrupted at the 7 -> 8 stage boundary and resumed equals the uninterrupted run exactly. FULL 458 pass. Install on the VM needs a maintenance boot after the curriculum run.
- Curriculum50 paired relaunch `srm-e50-curriculum50-paired-20261023T090435Z-3325` done: 158.3 min, exit 0; stage switch at epoch 8 with exact 90,000 / 90,000 counts; best capped val P_E .0419, AUC .9922 (R1 .0444 / .9917). Delivered, Pi = PC. Worker_2: calibration mapping entry (approved, as R1), evaluation and the frozen `curriculum50` verdict vs R1. Run_Supervisor: maintenance boot to install the resume runner (user told).
- Run_Supervisor: resume runner installed on the VM (maintenance boot; 4 stockouts first; up 12:06:36Z, stop 12:09:36Z, about $0.1). `~/.local/lib/cvdl-g4/job.sh` `f4aa9aa7...` and `job_run.last-good` `3fe4bbfb...` equal the files of `58aa081`. `runs-out/` untouched. VM TERMINATED, 300 min / STOP.
- Curriculum50 paired evaluation `a615397d39af1655d9ca` (best, epoch 40 one-based; schedule identity and the 50 epochs of counts verified: 5.13 M original + 3.87 M staged draws):

| P_E % | SRM-e50 | R1 paired | Curriculum50 paired |
| --- | ---: | ---: | ---: |
| Standard-256 | 2.54 | 2.55 | 2.57 |
| ALASKA seen | 19.42 | 18.99 | 19.72 |
| ALASKA unseen | 16.30 | 15.58 | 16.68 |
| LSDIR | 9.39 | 8.84 | 9.35 |
| Hard k=1 low AUC | .7450 | .7574 | .7568 |

- Frozen rule `curriculum50` vs R1: hard ΔAUC -.0006 [-.0056, +.0043] (needs +.01, lower bound > 0): FAIL; standard guard passes; epoch >= 8 passes. **Final: FAIL.** On top of pairing, the 50-epoch curriculum adds nothing measurable; seen-ALASKA noisy false alarms are higher (37.0% vs 32.3%).

### Decided (end of day, by the frozen rules)

- Pairing (#16) helps the hard cases (+.012) at no standard cost and no speed cost: use paired batches for future SRM training.
- The 50-epoch curriculum (#17) gives no gain over pairing alone. Not adopted.
- R2 fails its localization rule but has the lowest standard P_E so far (2.43%) with the same heatmap quality as run 13. Pending (user): switch the submission model to R2 or keep run 13.
- User: keep R2 in mind as the submission model (no switch yet). Approved run 18: a cell fine-tune from the Curriculum50 paired best, otherwise exactly R2; frozen rule `paired_r2` with R2 as the reference. Not chosen: a 50-epoch curriculum with the cell head from scratch (several differences at once).
- Run 18 `srm-cell-aux-curriculum50-paired-20261023T133537Z-1366` (`a7dfdaa`; starts from Curriculum50 best, epoch 40 one-based): 8 stockouts, then 13:37:14-14:25:42Z, 48.5 min, exit 0. **Bug in `58aa081`:** `last.ckpt`, the best checkpoint and `attempts.jsonl` reached the Pi as mode 600 (`tempfile.mkstemp`; `os.replace` keeps the mode), so `rpi400` could not read them; the user ran `sudo find ... -perm 600 -exec chmod 644` for this job. Copied, Pi = PC SHA. Worker_2: permission fix, then the run 18 evaluation. No G4 launch until the fix is in place.
