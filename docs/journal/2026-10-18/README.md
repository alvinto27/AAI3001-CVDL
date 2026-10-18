## 2026-10-18 — Project survey, train-matched validation design approved

Recorded by the Orchestrator.

### Did

- Worker_1 surveyed the project (read-only) and wrote a design for the train-matched validation (work list #1). The Orchestrator checked the survey claims against the files.
- Design record: [train-matched validation plan](../../active/plan-train-matched-validation.md).
- Container `aai3001` started (it was stopped).
- Train-matched validation built by Worker_1: `131492b` (opt-in `checkpoint_validation='train_matched'` + `validation_recipe`; `cvdl/validation.py`, `scripts/build_validation.py`, 16 new tests). Frozen lists:
  - SRM recipe `60da4391043e8ce48ff2`: 18,000 pairs (RAISE 7,200 / ALASKA 7,200 / LSDIR 3,600).
  - `srm-hard` recipe `bf7ede374cab01f73f5d`: 21,600 pairs (7,200 each).
- Verification (Worker_1): 286 tests pass (415 s); `refactor_golden.py compare` ALL EQUAL (GPU visible, approved by the Orchestrator); builder `--check` REBUILD IDENTICAL twice; old capped hash `6bb3c8ef...` unchanged; 6 pairs per recipe byte-identical at 0 and 2 loader workers.
- Orchestrator re-check: both `--check` REBUILD IDENTICAL; `test_train_matched_validation` + `test_g4_jobs` 27 tests OK; all rows from `val` photos only, no repeated tiles, all 256.

- `srm-hard` notebook moved to train-matched validation (`a0e72cd`, Run_Supervisor): PC QUICK smoke pass (record has recipe `bf7ede374cab01f73f5d`, per-source val metrics); dry run PASS. Launched on the G4 at 04:38:54Z (first attempt, no STOCKOUT): `srm-hard-20261018T043854Z-8467`. Running.

### Found

- Validation pools at 256 overlap-last (val split only): RAISE 36,484 tiles (150 photos), DIV2K 5,720 (120), ALASKA 8,966 (1,000), LSDIR 30,584 (1,500).
- Photo split (train/val/test): RAISE 701/150/149, DIV2K 680/120/100, LSDIR 7,000/1,500/2,500, ALASKA 13,500/1,000/1,700 seen + 2,000 unseen.
- The tiling result table labels its P_E as "validation min-P_E cut-off" (`docs/results/result-training-tiling.md` line 14); the 2026-10-17 handoff section 4 labels the same numbers "threshold 0.5". Resolved (side job A, Worker_2; Orchestrator re-checked SRM-e50: fixed 0.5 = 1.6703%, matches): both labels are wrong.
- Stale records: `README.md` calls notebook 04 the current baseline; `docs/status.md` says the Pi is online (handoff: shut down); the 2026-10-17 journal "Next" still says launch `srm-hard`; run queue row 8 has no result and "After the runs" still asks for the padding check.

- Realised draws match the training configs. SRM: k 24.9/25.2/24.7/25.1%; bpp bins 11.4/10.7/44.6/33.3% (target 11.1/11.1/44.4/33.3). `srm-hard`: k 50.2/16.6/16.6/16.6%; bpp bins 30.7/30.2/22.8/16.3% (target 30.6/30.6/22.2/16.7). Stego types near equal.
- Capacity clipping: 1,310 / 18,000 (7.3%) SRM, 899 / 21,600 (4.2%) `srm-hard`, from the unchanged shared draw policy.
- A CPU-only golden compare differs in 6 machine fields (device, ROCm fix flag); with the GPU visible it is ALL EQUAL.

- Side job A (Worker_2, read-only): every published test P_E (tiling table, handoff section 4, SRM-30, SRM-e50) is the **minimum P_E over thresholds on that test panel itself**, optimised separately for each panel and subgroup (`cvdl/metrics.py:84-96`, called per frame from `cvdl/post_fit.py:14-16` and `cvdl/evaluation.py:185-196`). It is not threshold 0.5 and not a validation cut-off. The comparison JSONs also hold a separate `at_capped_val_min_pe.pe_at_cutoff` (`cvdl/panels.py:95`).

  | Run | Panel | Fixed 0.5 | Full-panel val cut-off | Capped val cut-off | Test minimum (published) |
  | --- | --- | ---: | ---: | ---: | ---: |
  | 256 overlap | standard | 2.31 | 2.32 | 2.27 | 2.25 |
  | 256 overlap | ALASKA seen | 16.50 | 16.37 | 15.92 | 15.75 |
  | SRM-30 | standard | 1.86 | 1.69 | 1.69 | 1.67 |
  | SRM-30 | ALASKA seen | 15.11 | 14.92 | 14.75 | 13.86 |
  | SRM-e50 | standard | 1.67 | 1.70 | 1.63 | 1.57 |
  | SRM-e50 | ALASKA seen | 14.20 | 13.54 | 13.24 | 13.10 |

  Order SRM-e50 < SRM-30 < 256 overlap holds at every validation cut-off. The SRM-e50 gain over SRM-30 shrinks (standard: 0.10 points test-optimised, 0.06 at the capped cut-off, and reverses at the full-panel cut-off: 1.70 vs 1.69). Labels corrected in the tiling result, the 2026-10-17 handoff and journal, and status.

### Decided

- User: validation size = **10% of the training pairs per epoch**, exact training source mix, from the existing `val` photo split; no split changes; unique tiles; the builder fails if a pool is too small. SRM recipe 18,000 pairs (7,200 / 7,200 / 3,600); `srm-hard` 21,600 (7,200 each).
- User: no augmentation; k, bpp and stego type follow the training config.
- DIV2K threshold-calibration panel moves to the per-source threshold task (#4).
- User: delete leftovers (old `runs/.scratch/` worktrees except `srm-hard-launch`, speed/profile/BF16/map-demo runs, `g4-prime-1`, `bt2-job`, `early-pc`; `g4-v4-r1b` and `g4-v4-r2` only if no record links them).
- User: side job A (resolve the threshold label) after B.
- User: **thresholds** (replaces the 2026-10-17 "per-source thresholds are the main report"). Deployment cannot know the source, so it can use only one threshold. Main report = **one threshold at min P_E on the run's validation**, frozen and applied to every test panel. Per source: FA and miss at that one threshold, and AUC. Per-source thresholds stay only as a labelled reference in the report notebook ("if the source were known"). No DIV2K reference cut-off (user: skip DIV2K; no DIV2K validation panel).

- Cleanup done by Worker_1: 7 old worktrees and the speed, profile, BF16 and map-demo outputs deleted; `runs/` 2.4 GB -> 2.0 GB (409 MiB freed). Kept because tracked files use them: `g4-prime-1` and `g4-v4-r2` (golden compare), `g4-v4-r1b` (findings, archived plans), `early-pc` (V1/V3 results).

- User: report notebook template (#5) after tasks 2-4, not before.
- User: launch `srm-hard` now (after its notebook uses the new validation and checks pass), supervised by Run_Supervisor; Worker_2 works on task #2 in parallel.
- User: standard test at 256 approved as designed by Worker_2 ([plan](../../active/plan-standard-test-256.md)): new suite is the default with no flag; counts in multiples of 96 (10,944 / 2,688 / 3,168 / 5,280); photo-balanced tile selection; k 5-8 opt-in on the same tiles (k + 4).

### Next

- Evaluator `--validation-recipe` rescoring path: deferred (needed for rescoring old models on the new validation).
- `srm-hard` notebook update to the new validation (work list #7). Pending.
- Side job A. Pending.
