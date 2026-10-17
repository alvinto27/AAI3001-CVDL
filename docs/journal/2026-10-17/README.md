## 2026-10-17 — Training tiling complete, one evaluation tool, SRM run

Recorded by the Orchestrator.

### Did

- G4 runs (one per boot, all exit 0, all copied to the PC with Pi = PC SHA-256):
  - 512 pad `tiling-pad-20261017T013055Z-7210` (01:31-02:11Z, 54.65 steps/s, GPU-bound).
  - 256 overlap `tiling-overlap-256-20261017T022212Z-9778` (02:22-03:18Z; 2 capacity STOCKOUTs before).
  - 256 trim `tiling-trim-256-20261017T032025Z-8579` (03:20-04:16Z).
  - 256 pad `tiling-pad-256-20261017T043145Z-9721` (04:31-05:27Z; 3 STOCKOUTs before).
  - SRM `srm-20261017T052857Z-4521` (start 05:29Z; commit `0d851b6`). Pending.
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
- Removed 3 orphan DataLoader workers (left by a killed padding check at 01:49Z; about 4.3 GB of swap).

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

### Decided

- User: SRM uses the 256 overlap recipe (train tile 256, overlap_last, batch 128, 180,000 tiles/epoch). No PC speed test; read the speed on the G4 and stop early if 30 epochs cannot fit. Kept running.
- User: build the loader fix (workers send packs of blocks; batches must stay byte-identical; can be turned off).
- User: Worker_4 runs every post-training check with the one tool.
- Orchestrator: `--settings-from` must also feed the main stage (code fix) for old runs such as wide-R2; no hand-made settings files.
- Rule: after every PC run or kill, check `pgrep -af spawn_main` for orphan workers.

### Next

- SRM evaluated: best model so far. Next G4 run: user decision (longer SRM needs resume across boots; seed repeat). Pending.
- Loader fix: pushed. PC speed: synthetic ceiling at 256 34 -> 76 steps/s, main-process time per step 27.5 -> 8.3 ms; real PC loader unchanged (decode-bound at 6-12 workers). G4 gain not measured yet. Pending.
- Missing stages on old models: done. Combined 9-model table `runs/verification/post-training-model-comparison-20261017.csv`. Temperature helps only 256 overlap and SRM (slightly); tanh saturation only in SRM tanh1.
