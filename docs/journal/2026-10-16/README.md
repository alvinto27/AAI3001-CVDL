## 2026-10-16 — New-source data, Pi freeze, first mixed-source G4 run

Recorded by the Orchestrator (the Scribe session stopped on 2026-10-15).

### Did

- User decisions at the start: ALASKA option B (only Canon_PowerShot-G11 and
  Nikon_D90 train-only) and the LSDIR subset 7,000 / 1,500 / 2,500 photos.
- Worker_3 regenerated ALASKA selection v2 with option B (`ac098cf`):
  `data/alaska/selection_v2.csv` SHA-256
  `d3f122d637b09d9c65e7cafd8592f211c8b8381a3b3fa80303363bde5255206d`
  (supersedes `437fcfef...ff85`), 18,200 rows, 28 models in val/test.
- Worker_3 made LSDIR selection v1 (`4d6f27b`): `data/lsdir/selection_v1.csv`
  SHA-256 `c7e1b3209e96f1eda6ba59c4896ddbd639fffb7797e63048b5c4d674d65c9dbf`,
  11,000 rows, input `inventory.csv` SHA-256 `31371bb2...b0dc` (not
  committed). Splits stratified by shard (each shard train 411-412, val
  88-89, test 147-148) and assigned by blocks of 10 consecutive IDs with a
  minimum cross-split ID gap of 11 (Orchestrator confirmed globally).
- Pi -> PC copies started in parallel at ~09:00 +08 (ALASKA ~7.6 MiB/s,
  LSDIR ~3.7 MB/s). The Pi froze at ~09:07-09:10: no LAN ping, no SSH, sudo
  did not run on its console. The user powered it off and connected the USB
  SSD (ext4, label `datafs`) to the PC at `/run/media/yangxuan/datafs`.
  Both copies resumed from the SSD (read-only use): LSDIR ~228-240 MB/s,
  ALASKA ~192 MiB/s.
- Records built with full master verification: LSDIR `2659b3d`, ALASKA
  `daf893c` (builder `verified_masters=true` for both).
- Upload PC -> G4 in a maintenance boot (start 01:28Z): the Pi internet
  upload measured 0.10-0.25 MB/s (while busy) against the PC 12.7-14 MB/s, so
  the upload went from the PC. One or two rsync streams gave ~4.6 MB/s
  (long round trip to us-central1); 12 parallel rsync streams gave ~15 MB/s.
  Upload 01:36-02:27Z, 0 errors; all 29,200 files verified on the G4 (size +
  SHA-256); 66 GB free after. In the same boot the bootstrap `job.sh` was
  installed (SHA-256 `89cc83e9...b1f2af`; `job_run.last-good`
  `a49f0300...0034`; `CVDL_G4_STAGE` count 3). VM stop took 02:27:49 ->
  02:33:55Z. Maintenance boot cost ~$1.8 (estimate).
- Worker_2 updated `docs/archive/plan-new-data-code.md` and prepared
  `scripts/g4/jobs/newdata-1.env` (`717dcf8`): R2-wide recipe plus
  `V3_SOURCE_MIX=raise:0.4,lsdir:0.2,alaska:0.4`, `V3_TILES_PER_EPOCH=45000`,
  capped validation (5,000 RAISE + 1,535 ALASKA + 1,500 LSDIR = 8,035 pairs,
  pair-list SHA `6bb3c8ef...e4d5ca`), persistent workers, extra panels
  (ALASKA val/test_seen/test_unseen, LSDIR val/test, bpp-1).
- First launch 02:34:44Z failed at 02:36:14Z in the 2-batch sanity
  validation: `ValueError: capped validation has no alaska examples` (the
  sanity pass sees only RAISE). VM shut down by itself at 02:41:14Z after the
  5-minute delivery retry (Pi offline); ~$0.20. Fix `1a6a6bf`: per-source
  metrics are skipped only while `trainer.sanity_checking`; opt-in
  `V3_LIMIT_TRAIN_BATCHES` for a PC pre-flight; new unit test. PC pre-flight
  with the exact full 8,035-pair list passed (2,009 validation batches).
- Worker_3 Pi fixes (`3a6b1fd`, `docs/active/pi-host-changes.md`): `/srv/swapfile`
  (8 GiB on the USB SSD) removed, fstab line commented; zram 2 GiB kept.
  Persistent journal (`/etc/systemd/journald.conf.d/90-cvdl-persistent.conf`,
  SystemMaxUse=200M); the old journal was volatile by an RPi default, so the
  log of the frozen boot is lost.
- Relaunch 02:50:18Z, job
  `...wide-w4_mixr2-20261016T024859Z-9032`, commit `3a6b1fd`. Exit 0 at
  03:36:23Z, delivered 03:37:26Z, VM stopped by itself. ~47 VM min, ~$1.28
  (estimate). Worker_2 result `docs/results/result-newdata-1.md` (`ab9d3f1`).

### Found

- Run timing: epoch 0 cold (first train wait 83.9 s, train 109.4 s, I/O wait
  41.9%, reads 275.7 MiB/s; 7.4 it/s at first). Warm epochs: 54.5 steps/s,
  train 51.6 s for 2,813 steps, validation first-batch wait ~1.4 s (was
  18.8 s), validation 18-19 s for 8,035 pairs, GPU util 100%, ~414-427 W of
  600 W. No late slowdown in epochs 24-29 (train 51.5-51.8 s; the wide-R2 run
  had 84-111 s). Both speed fixes are confirmed on the G4. The sampler groups
  tiles by photo (one decode per photo group).
- Best epoch 25 (zero-based), capped val P_E 0.0574 (RAISE 0.0207, ALASKA
  0.1564, LSDIR 0.0757). Spikes at epochs 8, 10, 12, 14. Final train loss
  0.2736 above combined val loss 0.1746: still underfitting.
- New panels, old wide-R2 checkpoint -> mixed checkpoint (same pair IDs;
  Orchestrator re-computed from prediction files):

  | Panel | AUC | FA at 0.5 | k=1 miss at 0.5 | Miss at 5% FA |
  | --- | --- | --- | --- | --- |
  | ALASKA test_unseen | 0.7982 -> 0.9340 | 47.48% -> 13.80% | 31.30% -> 49.11% | not reachable (see correction) -> 25.31% |
  | ALASKA test_seen | 0.7842 -> 0.9159 | 49.74% -> 15.03% | 28.32% -> 52.21% | not reachable (see correction) -> 31.53% |
  | LSDIR test | 0.9628 -> 0.9742 | 7.91% -> 3.55% | 38.80% -> 47.48% | 11.94% -> 8.66% |

  "Miss at 5% FA" uses the threshold at the 95th percentile of each panel's
  clean scores (an analysis view, not a deployable threshold). ALASKA
  unseen P_E 0.23359 -> 0.13876; LSDIR test P_E 0.08151 -> 0.06635.
- Correction (same day, from check B): the first Orchestrator view gave the
  old model 21.25% / 22.24% ALASKA miss "at 5% FA". That was wrong. The old
  model scores 25.63% (unseen) and 28.25% (seen) of clean ALASKA test tiles
  exactly 1.0, so its 95th percentile is 1.0 and the real FA at that cut is
  25.6% / 28.3%, not 5%. No threshold gives the old model 5% FA on ALASKA
  without flagging nothing (score saturation; 418/1,535 = 27.2% of clean
  ALASKA val tiles at 1.0; the mixed model has 7/1,535).
- Interpretation (Orchestrator, corrected): the mixed model is better on
  ALASKA at both fixed and strict false-alarm levels; the old model is
  over-confident on new cameras. LSDIR improves on all measures. LSDIR is not an easy source (test P_E
  0.066).
- Standard test regressed: P_E 0.019533 -> 0.025370; k=1 0.046439 ->
  0.059008; DIV2K 0.052932 -> 0.059039.
- ALASKA per camera at threshold 0.5: 21/28 seen and 5/8 unseen cameras have
  FA above 10%; Sigma_SD1Merrill seen P_E 0.384 (FA 31.4%), Sigma_SD10 0.312.

- Check B (Worker_2, `042903d`, `docs/results/result-check-b.md`, PC only; both
  checkpoints scored on the same standard val, 11,238 pairs, settings SHA
  `2d24d423...`). Thresholds chosen on validation only:
  - Mixed model, ALASKA val 5%-FA cut-off 0.815754: test FA 5.97% (seen) /
    5.93% (unseen), drift +0.97 / +0.93 pp. Miss 29.72% / 24.20%; k=1 miss
    68.14% / 66.16%. P_E-optimal val cut-off 0.651967: test FA 9.10% / 8.79%,
    miss 24.74% / 19.63%.
  - One shared mixed 5%-FA cut-off (0.408610) gives ALASKA test FA 29.53% /
    24.36% against ~1.33% on RAISE: per-source calibration is necessary.
  - Old wide-R2: strict 5% FA on ALASKA is not reachable (score saturation,
    418/1,535 clean val tiles at exactly 1.0).
  - Tile noise (std of the clipped KV input) predicts the clean score inside
    all 36 ALASKA cameras (median Spearman 0.546 seen / 0.544 unseen). By
    val-derived noise quartile Q1 -> Q4: seen FA at 0.5 7.6% -> 23.1%, k=1
    miss 31.0% -> 75.0%; unseen FA 7.6% -> 21.3%, k=1 miss 34.4% -> 65.4%.
  - Figure `docs/figures/check-b/alaska-clean-score-contrasts.png`: high-score
    clean tiles have strong, textured KV residuals; their cell maps show many
    hot cells spread over the textured areas.
- Conclusion of B: two problems. Calibration (solved by per-source
  thresholds that hold on unseen cameras) and a model problem: natural noise
  and texture read as embedding, so noisy tiles raise both false alarms and
  k=1 misses. A threshold cannot fix the second.

### Decided

- Upload to the US VM with many parallel streams (12 gave ~15 MB/s).
- One heavy job at a time on the Pi; no swap on the USB SSD.
- Keep wide-R2 as the reference for the unchanged standard test; keep the
  mixed checkpoint for new-source research. Do not tune thresholds on test
  panels.
- User order for next work: B (free checks: ALASKA false-alarm review and
  per-source thresholds chosen on validation) -> A (per-cell training) -> C
  (more capacity / longer training, bf16).

### Afternoon: per-cell design, hold, documentation tidy-up

- Did: Worker_2 wrote the per-cell training design
  (`a6170ed`, [plan](../../active/plan-per-cell-training.md)): separate 1x1
  cell head, hard "any changed pixel" cell label with a reliability weight,
  50% region payloads, loss `tile CE + 0.2 x cell BCE`, pass marks fixed
  before a run. Not approved.
- Decided: the user put all research work on hold and asked for a
  documentation tidy-up first.
- Did: Worker_2 reorganised `docs/` (`45ec074`): `status.md`, `findings.md`,
  one journal folder per day, `active/`, `results/`, `archive/`. Nothing was
  deleted; the day folders join back to the old journal text. 309 links
  resolve; 93/93 tests pass.
- Decided: from now on, `docs/status.md` replaces the research checklist as
  the record that changes in the same commit as the journal.
- Did: Worker_2 changed run storage (`99ca3a1`). Notebook 05 keeps only
  `last.ckpt` (overwritten each epoch) and one best checkpoint; headless runs
  keep one full `run.log` and print progress at most every 45 s. PC-only
  `run_notebook.py --resume`: a QUICK run stopped after epoch 0 and resumed to
  epoch 1 with the same settings SHA (`e8a8aeb3...248d`). 97/97 tests pass.
- Did: removed 122 per-epoch checkpoints (445,284,516 B) after a check that
  the Pi holds full copies of the 5 G4 runs; moved early PC checkpoints to
  `runs-out/early-pc/`; deleted 9 scratch task logs. `runs-out` 1.1 GB ->
  695 MB.
- Decided: G4 preemption keeps the fresh-ID restart. G4 resume across boots
  is postponed until C (longer runs).
- Did: Worker_3 merged `runs-out/`, `checkpoints/` and `notebooks/runs/` into
  one git-ignored `runs/` folder (`5881bbd`, `7463ce1`, `7d56dbe`; tracked
  result files stay tracked). Notebooks 01, 03, 04 and 05 and the scripts
  write to `runs/`. The G4 keeps its internal `runs-out/` paths (installed
  `job.sh`, job-ID records); the PC copy lands in `runs/<job-id>/`. Removed
  20 early per-epoch checkpoints, one duplicate checkpoint and the old
  `pc-headless-2ep` test run (~19 MB). 97/97 tests pass.
- Decided: the user discarded the private notebook 04 edits (test settings
  only). The Orchestrator dropped all 4 notebook 04 stashes.
- Did: PR #2 to `main` (branch `runs-folder-main`) with only the same folder
  move for main's files: https://github.com/alvinto27/AAI3001-CVDL/pull/2.
  Done: the user merged PR #2 (`0981ce7`) and deleted the branch.
- Did: Worker_3 reorganised the PC data folders (`76f6257`, `a26ebda`):
  `Datasets/originals/{raise,div2k,alaska,lsdir}` (72 GB) and
  `Datasets/generated/raise512-run1-records` (117 MB); `ImageSets/` removed.
  `master_path` rewritten in 9 record CSVs. Proof: standard test hash
  `fdd989f7...06837`, capped validation SHA `6bb3c8ef...e4d5ca`, 4 RAISE/DIV2K
  panels, 5 new-source panels, fixed train pairs (81,288), and decoded pixels
  of 24 photos are all unchanged; all masters present with correct size and
  SHA. Deleted the otf-cache (50 GB) and the raise512-run1 images (37.5 GB).
  All 22,097 ALASKA files kept (user). PC free space 700 -> 778 GB.
- Decided: `V3_SOURCE` defaults to `decode` (notebooks 04 and 05); `cache` is
  an explicit option and the cache must be rebuilt to use it.
- Pending: the G4 disk and `scripts/g4/{setup.sh,upload_masters.sh}` still
  use the old layout. Reorganise the G4 before the next G4 job (code from
  `76f6257` on needs the new layout). Then a PR for `main`.
- Found: the `runs/` change had removed `runs-out/` from `.gitignore`; on the
  G4 this would make the checkout dirty and the next job would refuse to
  start. Worker_3 restored it (`9798b56`).
- Did: G4 data reorganisation (`9798b56`: `scripts/g4/reorg_data_layout.sh`,
  `delivered_manifest.sh`, updated `setup.sh` and `upload_masters.sh`).
  Maintenance boot 06:00:55Z -> stopped 06:10:43Z (~10 min, ~$0.27
  estimate). Dry run, then apply 06:03-06:04Z: 5 renames into
  `Datasets/originals/`, `~/otf-cache` deleted (50 GB), per-epoch
  checkpoints and tb logs of 8 delivered job folders deleted (427.3 MB, each
  file matched the Pi by size and SHA-256). Master check pass (RAISE 1,000
  and DIV2K 900 present; ALASKA 18,200 and LSDIR 11,000 size + SHA OK). G4
  checkout detached at `9798b56`, clean. `.g4-state` and the bootstrap install
  unchanged (39 files, same SHA-256; `job.sh` `89cc83e9...`). G4 free disk 66
  -> 116 GB.
- Decided: `main` stays as it is until `yx` goes into `main` as a whole.
- Did: code survey (Worker_3). Found: the model classes in
  `scripts/map_demo.py` (used by `eval_checkpoint.py` and check B) are a
  rewritten copy of the notebook 05 cell. Parity check on 4 real checkpoints
  x 64 real tiles (RAISE, DIV2K, ALASKA, LSDIR), CPU and GPU: logits and cell
  maps differ by exactly 0; strict state_dict loads. Past results are not
  affected. Permanent test `tests/test_model_copy_parity.py` (`ac8810a`).
- Found: the 34 tests in `LSB-Dataset-Generator/tests` ran only from inside
  that folder and were never part of the suite; they pass.
- Did: code tidy-up (`d6a3875`): 8 one-off check / selection scripts to
  `scripts/archive/` (they need the deleted PNG dataset), `check_notebook_quick.py`
  deleted, generator tests moved into `tests/`, Mode 2 code and its 12 tests
  archived (dropped with MIL), notebooks 01 and 03 to `notebooks/archive/`.
  Active scripts, `scripts/g4/` and `scripts/pi/` unchanged. Suite: 122
  tests pass from the repo root (100 + 34 - 12 archived).

- Found: a test merge of `yx` into `main` conflicts in about a dozen files
  (add/add). Cause: PR #1 copied files into `main` without their `yx`
  history. Not caused by PR #2. The later `yx` -> `main` merge needs one
  careful PR.
- Decided: the G4 keeps its internal `runs-out/` name for now (installed
  `job.sh`, job-ID records). Rename it to match the PC `runs/` with the next
  real G4 job, not in a separate boot.
- Did: started the notebook refactor (user request). Goal: move the code
  that notebooks 04 and 05 share (data, models, Lightning module, callbacks,
  evaluation, plots) into one importable package, so each notebook holds
  only its experiment. Worker_3 is doing a survey and design first; nothing
  changes until the proof rules (same hashes, batches, model outputs,
  checkpoint keys, and training losses) are agreed.
- Decided (user): notebook 04 is frozen as the v3 baseline; the refactor
  starts from notebook 05; package name `cvdl`; each run records its git
  commit. Phase 0 (capture of the "before" values) started.
- Decided (user): experiment settings live in the notebook's top cell (in
  git), not in G4 env files. Machine settings (data workers PC 6/4 vs G4
  32/16, QUICK, precision limits) come from an automatic per-machine
  profile. One-off PC overrides via `run_notebook.py --set`, recorded in the
  run. Settings have one definition each (type, default, range); unknown
  names or bad values stop the run; each run saves the final settings, the
  source of each value, and the commit. A G4 job then names a notebook and
  a commit instead of an env file.

- Discussed: the user finds the final held-out test large. Facts: the
  standard test has 10,956 pairs from only 249 photos (RAISE about 65 tiles
  per photo); all post-fit scoring took about 6 of 45 min (~$0.16) in the
  mixed run. Decided: keep the standard test unchanged. Idea from the user:
  weight evaluation towards hard cases. Orchestrator view: a dedicated hard
  test set with about 1,000 pairs per hard group (95% margin about +/-3% on
  a 60% miss rate, against +/-11% for today's 81-pair noisiest-quarter k=1
  group), reported separately, instead of one weighted score. Status: idea
  only; the user wants to discuss it further; not automatic.
- Found: QUICK runs do not cap the extra test panels (each runs the full
  panel, e.g. 10,956 bpp-1 pairs). Recorded in the refactor golden as a known
  defect, to be fixed on purpose in the settings phase.

- Did: refactor phase 0 (`bc7eba2`): before-values and a re-runnable
  checker `scripts/refactor_golden.py`. CPU 20-step reference run 28 s
  (first try took 22+ min on 1 thread in the full bpp-1 panel; fixed to 8
  threads, no extra panels). Notebook 04 got only its frozen banner.
- Found: the GPU is not bit-exact across processes, also with unchanged
  code (5 repeats: 2 exact, others flip; score shifts up to 2.9e-4, logit
  929 changes by 6e-5). A module audit showed identical code ran.
- Decided (user): slim checks. Exact and mandatory: settings, data, CPU
  20-step run, 4-checkpoint CPU outputs; GPU package-vs-notebook in one
  process (exact 0). GPU QUICK is a smoke test only (files, counts, keys),
  run once at the end of the refactor.
- Did: refactor phase 1 (`a8f1a7d`): `cvdl/models.py`, `cvdl/metrics.py`;
  `map_demo`, `eval_checkpoint`, `speed_matrix`, `check_b_*` import them.
  Checker ALL EQUAL (re-run by the Orchestrator); 130 tests; a saved bpp-1
  panel re-scores exactly. Design: `active/plan-cvdl-package.md`.

### Next

- Done: B checks (`042903d`).
- Done: bookkeeping tidy-up of docs, runs, data (PC and G4) and code.
- In progress: notebook refactor; phases 0-1 done, next phase 2 (data).
- On hold: A, per-cell training (design written, not approved).
- On hold: C, capacity / epochs / bf16.
- Pending: start a new Scribe session (the old Scribe alias shows the
  Orchestrator session ID).
