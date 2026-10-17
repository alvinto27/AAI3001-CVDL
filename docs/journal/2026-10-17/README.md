## 2026-10-17 — Training tiling complete, one evaluation tool, SRM runs, hard test v1

Recorded by the Orchestrator.

### Did

- G4 runs (one per boot, all exit 0, all copied to the PC with Pi = PC SHA-256):
  - 512 pad `tiling-pad-20261017T013055Z-7210` (01:31-02:11Z, 54.65 steps/s, GPU-bound).
  - 256 overlap `tiling-overlap-256-20261017T022212Z-9778` (02:22-03:18Z; 2 capacity STOCKOUTs before).
  - 256 trim `tiling-trim-256-20261017T032025Z-8579` (03:20-04:16Z).
  - 256 pad `tiling-pad-256-20261017T043145Z-9721` (04:31-05:27Z; 3 STOCKOUTs before).
  - SRM `srm-20261017T052857Z-4521` (05:29-07:03Z; commit `0d851b6`). Evaluated.
  - SRM-e50 `srm-e50-20261017T111310Z-0214` (start 11:13Z after 1 STOCKOUT; commit `2841fdf`; timeout 165 min, boot limit 180 min). Done 11:13-13:47Z (153.1 min; ~$4.3). Best epoch 43, val P_E .0432, AUC .9917. Pi=PC SHA verified.
- Capacity retry: `runs/.scratch/retry-256.sh`, 4-min waits, at most 15 attempts, fresh job ID each.
- PC evaluation of all 6 tiling arms by Worker_4: [result](../../results/result-training-tiling.md).
- Code (Worker_5 unless noted):
  - `970d337` padding check with worker DataLoader and `--padding-pairs` cap (default 2,000; 18,000 cases). Arm 1 full check reproduced: 0 decision flips in 131,472 rows. 7.0 min full, 2 min capped (was 12.3 min).
  - `f0bb81d` temperature scaling (`--temperature`).
  - `0e05dcb`, `496b5a7` tanh saturation (`--tanh-saturation`), stage selection `--only` / `--skip-main`, refuse to overwrite without `--overwrite`.
  - `c13ee00` main stage writes validation files, standard-validation panels and the per-source cut-off tables. Worker_4 verified on 256 overlap: 9 files, 113,860 rows, max |delta prob| 0.0; 35-row check-B CSV byte-identical.
  - `95be1aa` `scripts/profile_loader.py` (256 loader profile).
  - `1503aac`, `ee401be`, `9c3addd`, `d721e06` (Worker_4): SRM test update, `--settings-from` for the main stage, path-relative golden compare, restricted evaluation `--set`.
  - `7567501` (Worker_4): the evaluator no longer writes into a run's `data_run_settings.json`; the 6 tiling records restored from the Pi copies (SHA equal; old bytes kept in each evaluation folder).
  - `863d9fb`, `efde4df` (Worker_5): packed worker messages (`pack_blocks`, default 16, 1 = off) and PC loader scripts with a worker cap, memory guard and clean exit. Batches identical to pack 1 at 1, 6 and 12 real workers for the 256 and 512 recipes (epochs 0 and 1); 32/64 workers by an ordering model. 248 tests, golden ALL EQUAL.
  - `0d851b6` (Worker_4) SRM notebook filled with the 256 overlap recipe; only `name` and `front_end` differ.
- Worker_5 32-worker loader hash runs filled the PC RAM (about 20 GB, swap 100%) and the kernel killed a golden capture; the Orchestrator killed the runs and their orphans. New rules: at most 12 real workers on the PC, a memory guard in the loader scripts, and ask Worker_4 before heavy PC work.
- Removed 3 orphan DataLoader workers (left by a killed padding check at 01:49Z; about 4.3 GB of swap).
- `srm-hard` notebook `795fd93` (Worker_4): opt-in `k_weights` and `bpp_mixture` in the generator settings; old configs draw exactly as before (fingerprints equal); validation/test hashes unchanged. 20,000-draw check: source 33.6/33.2/33.2%, k 49.8/16.7/16.9/16.6%, bpp bins 30.5/30.9/21.7/16.9%. 256 tests, golden ALL EQUAL, G4 preflight and PC QUICK pass; dry run PASS. Estimate 112 min. Launch tomorrow on the user's go (command in the run queue).
- Hard test v1 `d74d716` (Worker_5): `data/hard-test/v1/` (4 frozen pair lists + `summary.json` with SHA-256). 32 cells x 1,008 pairs (24 balanced stego combinations x 42). Test-split photos only; noise edges from validation only. ALASKA noisy cells use 572/574 distinct tiles, each at most 2x; clean FA counts distinct tiles. Evaluator stage on by default (`--no-hard` off, `--only hard` for finished runs). 270 tests, golden ALL EQUAL, rebuild check identical (Orchestrator rerun).

### Found

- Tiling: overlap is the best grid at 512 and 256; 256 training tiles improve all arms. 256 overlap standard P_E 2.25% (512 overlap 2.80%, newdata-1 2.54%). Details in the result record.
- Temperature: wide-R2 T = 113 (clean ALASKA validation scores tied at 1.0: 27% -> 0%; 5% FA becomes reachable). Mixed-source models T = 0.98-1.30. Logits reach 300-660 (wide-R2 3,000-5,000), so float32 sigmoid ties at 1.0 come from rounding.
- 256 loader: G4 32.8 steps/s, GPU about 40-60%, host CPU about 1/3, no disk wait. Profile: 97% of main-process samples in the batch-join thread; per-message receive cost; 32 messages per step at 256 vs 8 at 512. Synthetic ceiling 35 steps/s at 256, 70 at 512.
- SRM on the G4: 18.4 steps/s, GPU 100% at about 458 W (GPU-bound). Run time 93.1 min of 105. Best epoch 29 (last), val P_E .0466 (256 KV .0543).
- SRM test (PC, `d721e06`, `--set rocm_conv1_fix=off --set eval_batch_size=32`), P_E % 256 KV -> SRM: standard 2.25 -> 1.67, bpp-1 1.90 -> 1.21, ALASKA seen 15.75 -> 13.86, ALASKA unseen 13.67 -> 11.70, LSDIR 6.60 -> 5.16. AUC improves on all 5.
- SRM per-source 5%-FA (test miss / k=1 miss %, KV -> SRM): RAISE 1.0/2.8 -> 0.6/2.0; DIV2K 4.1/19.5 -> 2.9/16.6; ALASKA seen 29.2/68.1 -> 25.0/64.9; ALASKA unseen 21.3/62.3 -> 18.4/58.3; LSDIR 9.9/47.3 -> 6.5/31.9.
- SRM tanh: inputs beyond +/-3 at tanh1 20.8% clean / 26.4% stego (KV wide-R2 0.5/0.7%); tanh2 4.5%. Logit p99 clean 6.8 / stego 202. Temperature T = 0.82. Padding still dilutes evidence (padded misses lower than KV in 6/6 cases; extra miss from padding larger in 4/6).
- wide-R2 tanh: tanh1 0.5/0.7%, tanh2 8.8/9.0% beyond +/-3; logit p99 clean 462. Tanh is not saturated in KV models; the large values come after tanh.
- Evaluation wrote `diagnostic_pair_ids` into the training data record of 6 tiling run folders (audit `runs/verification/training-data-record-write-audit.json`). wide-R2 clone and SRM restored.

- SRM weakness breakdown (Worker_4, CPU only, saved predictions; `runs/srm-20261017T052857Z-4521/post-training/breakdown/`). Miss at per-source validation FA5:
  - by k (1/2/3/4): RAISE 1.98/0/0/0; DIV2K 16.57/1.20/0/0.79; ALASKA seen 64.90/37.77/22.54/20.26; ALASKA unseen 58.27/27.59/16.53/11.63; LSDIR 31.86/8.47/2.59/1.19 (%).
  - k=1 by clean-tile noise quarter Q1 -> Q4: ALASKA seen 38.0 -> 92.5% (n 100/80); unseen 34.4 -> 80.3% (125/81); LSDIR 18.8 -> 76.2% (165/63).
  - k=1 by effective bpp 0.05-0.10 -> 0.35-0.50: ALASKA unseen 84.2 -> 48.2% (38/108); LSDIR 60.0 -> 19.1% (70/168).
  - k=1 stride harder than contiguous (ALASKA unseen 69.4 vs 49.6%); replace harder than match (LSDIR 38.2 vs 25.6%). Factors not adjusted for each other.
  - Clean FA rises with texture (RAISE own-source quarters Q1 1.5% -> Q4 9.8%).
- The standard test uses k 1-8 and bpp 0.05-0.5 (bpp-1 panel: up to 1.0); training uses k 1-4. About half of the test pairs are k 5-8, and k=1 is about 12.5%.
- With one shared threshold (5% FA on mixed validation), SRM clean FA on test: RAISE 1.8%, DIV2K 7.8%, LSDIR 6.0%, ALASKA 18-20%.

- SRM-e50 vs SRM-30 (best epoch 43 vs 29; P_E at 0.5): standard 1.57 vs 1.67%; bpp-1 1.17 vs 1.21%; ALASKA seen 13.10 vs 13.86%; ALASKA unseen 11.02 vs 11.70%; LSDIR 5.04 vs 5.16%. AUC within 0.006.
- SRM-e50 vs SRM-30 on hard test v1: miss lower in 25 of 32 cells, higher in 5, equal in 2; most gains 1-3 points. No cell has non-overlapping 95% Wilson intervals: a small, consistent trend, one seed each. Hard-test clean FA is higher for e50 in some noisy cells (RAISE+DIV2K noisy about .14 vs .10; ALASKA seen noisy .077 vs .047). ALASKA k1 low+noisy stays near chance (AUC .53-.54).
- Hard test v1 results (miss at per-source validation FA5, SRM vs 256 overlap KV): SRM is better in 31 of 32 cells. Examples: LSDIR k1 all .328 vs .484; LSDIR k2 low+noisy .351 vs .583; ALASKA unseen k1 all .617 vs .637. ALASKA k1 low+noisy is near chance for both (AUC .52-.54; miss .88-.93).
- Incident: the first hard-test scoring used the ordered loader, which buffered about 21 GB because the hard pairs spread over all photos. Worker_5 stopped it; `score()` now reads one task per photo. Second run held about 18 GB available.

### Decided

- User: SRM uses the 256 overlap recipe (train tile 256, overlap_last, batch 128, 180,000 tiles/epoch). No PC speed test; read the speed on the G4 and stop early if 30 epochs cannot fit. Kept running.
- User: build the loader fix (workers send packs of blocks; batches must stay byte-identical; can be turned off).
- User: Worker_4 runs every post-training check with the one tool.
- Orchestrator: `--settings-from` must also feed the main stage (code fix) for old runs such as wide-R2; no hand-made settings files.
- User: SRM-e50 (50 epochs, LR drops scaled to epochs 8 and 33). User raised the G4 max run duration to 3 h (10,800 s).
- User: hard test v1 = k 1 and 2; per source panel x {all bpp, low bpp 0.05-0.15, noisiest quarter, low bpp and noisy}; about 1,000 pairs per cell; test-split photos only; reported next to the unchanged standard test. Worker_5 builds it.
- User: default evaluation = standard suite + hard suite. Padding, temperature and tanh checks only on request.
- User: hard training notebook `srm-hard` = SRM recipe, 30 epochs, with per epoch RAISE 72,000 / ALASKA 72,000 / LSDIR 72,000 pairs (LSDIR doubled; 216,000 total); k=1 50%, k=2-4 one sixth each; half of the bpp draws from 0.05-0.15, half from 0.05-0.5; stego type unchanged. No texture-weighted sampling in this run. Prepare launch-ready today; launch tomorrow on the user's go. Worker_4 builds it.
- Rule: after every PC run or kill, check `pgrep -af spawn_main` for orphan workers.
- User (late): validation must follow the training set, so that the training and validation curves show overfitting. The final holdout test needs a major update: 256 x 256 tiles only, bpp 0.05 <= x <= 0.5, k 1-4 (user); k 5-8 as a separate opt-in check (SRM-e50 k8 misses 3.0-4.0% at threshold 0.5, above k4). Same total pairs as today, all on k 1-4 (user: option b). Other scope to agree. `srm-hard` launch on hold until the new validation exists (to confirm).

### Next

- `srm-hard` (`795fd93`): launch on the user's go (command in the run queue). Then evaluate with the default suite (standard + hard v1) and compare with SRM-30 and SRM-e50.
- Pending user items: delete about 600 MB of old runs; check whether `job-console.log` duplicates `run.log`; the `yx` to `main` PR.
- Loader fix: pushed. PC speed: synthetic ceiling at 256 34 -> 76 steps/s, main-process time per step 27.5 -> 8.3 ms; real PC loader unchanged (decode-bound at 6-12 workers). On the G4, SRM-e50 (pack 16) trained at 18.36 steps/s, the same as SRM-30 (pack 1, 18.4): SRM is GPU-bound, so no gain is visible. A loader-bound 256 KV run is needed to measure it. Pending.
- Missing stages on old models: done. Combined 9-model table `runs/verification/post-training-model-comparison-20261017.csv`. Temperature helps only 256 overlap and SRM (slightly); tanh saturation only in SRM tanh1.
