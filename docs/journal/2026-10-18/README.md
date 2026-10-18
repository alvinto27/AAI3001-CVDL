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
- Standard test at 256 built by Worker_2: `bd36403` (`cvdl/standardtest.py`, `scripts/build_standard_test.py`, `configs/test-256-v1.yaml`, 8 frozen lists, recipe `5045c1076ab3f540a61c`). `evaluate_run.py runs/<run>` now runs it by default; `--higher-k` adds k 5-8; `--test-suite legacy-512` keeps the old suite. Hard test reported as pending until #3. Verification: 306 tests pass; golden ALL EQUAL (GPU visible); builder `--check` byte-identical twice (2.4 s). Orchestrator check: counts 10,944 / 2,688 / 3,168 / 5,280, k exactly one quarter, every k x type cell equal (RAISE 101, DIV2K 13, ALASKA seen 28, unseen 33, LSDIR 55), every test photo used, no repeated tile, k 5-8 lists match row for row with k + 4.

- Hard test at 256 built by Worker_2: `a56c275` (`cvdl/hardnoise.py`, `cvdl/hardtest256.py`, `scripts/measure_hard_noise_256.py`, `scripts/build_hard_test_256.py`, `data/hard-test/256-v1/`, recipe `d6d91a5a98dae856d1ae`). Noise scan: 9,219 photos, 206,811 tiles, 198.9 s on 8 CPU workers (pilot projected 203.7 s); RAM peak ~17 GB. Verification: 322 tests pass; golden ALL EQUAL (GPU visible); builder `--check` byte-identical twice; hard v1 `--check` still identical. Orchestrator check: 32 cells x 1,008, every noisy row above its source edge, every low row in bpp 0.05-0.15, test splits only, at most 1 use per tile in a cell.
- `srm-hard` epoch-2 projection (Run_Supervisor): 18.3 steps/s (GPU-bound); warm epoch 207 s (train 183.7 s, validation 22.4 s for 21,600 pairs); total ~105-110 min vs timeout 165.

- One deployment threshold built by Worker_2: `68e4f0b`. Default evaluation scores the validation list first, fixes one pooled min-P_E cut-off (float64; ties: smaller FA), and applies it to every standard-256 row, hard-256 cell and k 5-8 panel; per-source references separate (no DIV2K). Mapping `configs/evaluation-calibration-v1.json`: wide-R2 -> new list `8328cc6b0673cade0312` (RAISE 15,239 / DIV2K 2,761, shares from its saved training record: 45,674 / 8,276 pairs per epoch); runs 2-10 -> `60da4391043e8ce48ff2`; all labelled `post_hoc_256`. Verification: 338 tests pass; golden ALL EQUAL (GPU visible); builder `--check` identical. Orchestrator check: all 10 mapped checkpoint SHA-256 values match the files in `runs/`; wide-R2 list is val-only, 256, no repeated tile.

- `srm-hard` done: 04:42:59-06:28:44Z (105.6 min, exit 0), VM TERMINATED, delivered to the Pi, copied to `runs/srm-hard-20261018T043854Z-8467/` (Pi = PC SHA-256 for both checkpoints; Orchestrator re-hashed). Best epoch 28: val P_E .1595, AUC .9325 on its own train-matched validation (k=1 50%, half of bpp 0.05-0.15), so not comparable with older val P_E. Last epoch per-source val P_E: RAISE .041, ALASKA .282, LSDIR .165.
- SRM-e50 pilot of the new default evaluation (Worker_2): 4.2 min on the PC (validation 77 s, standard 42 s, source panels 37 s, hard 91 s). One cut-off 0.4506 (post_hoc_256, list `60da...`; validation P_E 10.29%). Standard-256 P_E at that cut-off 2.54% (RAISE 1.87, DIV2K 7.81), ALASKA seen 19.42% (FA 16.3%), unseen 16.30%, LSDIR 9.39%. k=1 misses: standard 9.8%, ALASKA seen 49.7%, LSDIR 38.7%. Hard: ALASKA seen k1 low+noisy FA 37.9% / miss 59.8%; LSDIR k1 low+noisy miss 82.1%. Training records unchanged (SHA). Not like-for-like with the old 512 numbers.

- `srm-hard` evaluated (Run_Supervisor, 4 min 12 s, eval id `a08cade1bcae87ff827c`, train_matched cut-off 0.4905; validation FA 10.8% / miss 21.1% / P_E 15.95%). Training records unchanged (SHA). Against SRM-e50 at each model's own validation cut-off, P_E %: standard 3.10 vs 2.54 (DIV2K 10.82 vs 7.81), ALASKA seen 21.95 vs 19.42, unseen 19.16 vs 16.30, LSDIR 9.77 vs 9.39. Misses fall (ALASKA seen k1 45.1 vs 49.7; LSDIR k1 30.9 vs 38.7) but false alarms rise on every panel (ALASKA seen 21.7 vs 16.3). AUC flat or lower except small gains on k=1 low-bpp cells (standard k1 low_noisy .905 vs .887; LSDIR k1 low_noisy .671 vs .653). ALASKA noisy k=1 cells stay at chance for both (AUC ~.52-.56). Caveat: srm-hard trained 30 epochs, SRM-e50 50; the like-for-like 30-epoch comparison is SRM-30 (needs rescoring, #6).

- Rescore (#6) by Run_Supervisor, 06:40-07:11Z: 8 of 9 old models done with the new default suite (all `post_hoc_256`, list `60da...`); training records unchanged (25/25 SHA OK). P_E % at each model's validation cut-off (standard / ALASKA seen / unseen / LSDIR): SRM-30 3.07 / 20.03 / 17.31 / 9.63; SRM-e50 2.54 / 19.42 / 16.30 / 9.39; srm-hard 3.10 / 21.95 / 19.16 / 9.77; overlap-256 3.43 / 21.21 / 18.75 / 11.27; trim-256 4.82 / 21.89 / 19.11 / 10.86; pad-256 5.07 / 22.19 / 19.27 / 11.27; overlap-512 5.61 / 23.94 / 21.43 / 12.62; trim-512 6.48 / 23.49 / 21.31 / 12.95; pad-512 6.53 / 24.20 / 20.93 / 12.88; newdata-1 5.13 / 24.29 / 21.21 / 12.64. Like-for-like 30-epoch check, srm-hard vs SRM-30: standard equal (3.10 vs 3.07), ALASKA worse (seen 21.95 vs 20.03), hard k1 low+noisy AUC better on standard (.905 vs .882) and LSDIR (.671 vs .651), ALASKA k1 low+noisy at chance for every model (.51-.52).
- ~~Pending: wide-R2 failed (rc 1): `settings_for_run` read a missing `checkpoint_validation` key.~~ Fixed in `dd9867a` (Worker_2): an exact-SHA migration for this one record derives standard validation, no source mix, no fixed epoch quota from its saved notebook (22,476 validation images = all 512 val tiles; 3,372 steps = 53,950 pairs); other incomplete records still fail. 353 tests pass; golden ALL EQUAL. wide-R2 evaluated (3.4 min, `post_hoc_256`, list `8328...`, cut-off 0.664; validation P_E 3.87%): standard 3.78, RAISE 3.08, DIV2K 9.21, ALASKA seen 32.65 (FA 44.4%), unseen 31.08, LSDIR 14.02. Never trained on ALASKA or LSDIR. Rescore #6 complete: all 10 old models plus srm-hard on the new suite.

- Report notebook template (#5) built by Worker_2: `716b2cd` (`notebooks/templates/report.ipynb`, `cvdl/reporting.py`, `scripts/build_report.py`, guide [report-notebook.md](../../active/report-notebook.md)). Reads saved outputs only, CPU only, one cut-off per model, optional reference model. 347 tests pass. Demos (git-ignored): `runs/srm-e50-20261017T111310Z-0214/report.ipynb`; `runs/srm-hard-20261018T043854Z-8467/report.ipynb` with reference SRM-30. Headline rows equal the CSV values exactly.
- The SRM-e50 ALASKA reference threshold equals the global cut-off (0.45057889819145203): checked on saved scores; both have a unique optimum on the same ALASKA stego score; not a bug.

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
- Hard-256 noise edges (all val-split 256 tiles, model-independent): RAISE 1.691, DIV2K 2.442, ALASKA 2.613, LSDIR 2.516 (v1 from one model: 1.691, 2.409, 2.600, 2.368). Noisy test pools: RAISE+DIV2K 10,507 / 41,088, ALASKA seen 3,475 / 15,313, unseen 3,278 / 18,070, LSDIR 12,980 / 50,586: no tile reuse needed (v1 needed ALASKA reuse).

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

- User: hard test at 256 approved as designed by Worker_2 ([plan](../../active/plan-hard-test-256.md)): noise edges from all validation tiles per source (incl. DIV2K, noise only), candidates from all test tiles, overlap with standard-256 allowed, 8-worker CPU noise scan after a timing pilot.

- User: threshold design approved ([plan](../../active/plan-validation-threshold.md)): old checkpoints get a post-hoc 256 overlap-last calibration (option C): runs 2-10 share list `60da4391043e8ce48ff2` (18,000 pairs); wide-R2 gets its own RAISE+DIV2K list (18,000, shares from its training records). Rescore only the 10 documented models. Ties: smaller FA; unmapped runs fail, no fallback.

- User on the demo reports: ids and hashes are not useful; tables are too dense and it is not clear what to look at. Approved to try a question-first layout ([plan](../../active/plan-report-readability.md)): headline numbers, one chart per question, full tables in a collapsed appendix, plain labels, ids hidden in metadata.

- B1 (noisy ALASKA k=1 at chance for all models): Orchestrator read the SRM-e50 hard predictions at its validation cut-off. k=1 misses: contiguous (packed) payloads are easier than random (scattered) ones when not noisy (standard 8-10% vs 26-28%; ALASKA 42-49% vs 67-70%), but on noisy ALASKA both are missed (53-60% vs 58-60%). LSB replacement and +/-1 matching are missed equally (ALASKA noisy k1: 53-60% vs 58-60%), so the model does not appear to use the known replacement fingerprint. Per-cell labels add little for scattered low-bpp payloads (almost every 32 px cell holds a change). User: run a classic-detector check (Weighted Stego on the noisy ALASKA k=1 hard tiles, replace vs match; CPU only) **later, after the user returns**. Pending. Per-cell training plan stays on hold and is out of date (512 tiles, old tests). Follow-up (user question): contiguous k=1 misses at low bpp are still high outside RAISE/DIV2K (SRM-e50: standard 12.2%, ALASKA seen 50.6%, unseen 45.5%, LSDIR 43.5%; random 40.5 / 79.2 / 75.0 / 68.5%; 336 stego tiles per cell). So per-cell labels could carry information for packed payloads on ALASKA and LSDIR (many clean 32 px cells); they still do not address the noisy case. Why (Orchestrator reading of `LSB-Dataset-Generator/lsb_engine.py` `_select_units`): contiguous = one run of consecutive RGB units from a random start; at k=1, 0.13 bpp on a 256 tile that is a full-width band of about 11 rows (~4% of the tile), about half its LSBs changed. The tile score averages the 8 x 8 cell logits, so the band is ~5% of the vote and is outvoted by textured cells. Proposed second check for after the user returns (Pending, not approved): re-score existing checkpoints with max or top-k cell pooling instead of the mean, threshold on validation, compare contiguous/random misses and noisy-clean FA on the hard test (GPU inference only, no training).

### Next

- Evaluator `--validation-recipe` rescoring path: deferred (needed for rescoring old models on the new validation).
- `srm-hard` notebook update to the new validation (work list #7). Pending.
- Side job A. Pending.
