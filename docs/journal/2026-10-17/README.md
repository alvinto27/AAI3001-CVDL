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
  - `0d851b6` (Worker_4) SRM notebook filled with the 256 overlap recipe; only `name` and `front_end` differ.
- Removed 3 orphan DataLoader workers (left by a killed padding check at 01:49Z; about 4.3 GB of swap).

### Found

- Tiling: overlap is the best grid at 512 and 256; 256 training tiles improve all arms. 256 overlap standard P_E 2.25% (512 overlap 2.80%, newdata-1 2.54%). Details in the result record.
- Temperature: wide-R2 T = 113 (clean ALASKA validation scores tied at 1.0: 27% -> 0%; 5% FA becomes reachable). Mixed-source models T = 0.98-1.30. Logits reach 300-660 (wide-R2 3,000-5,000), so float32 sigmoid ties at 1.0 come from rounding.
- 256 loader: G4 32.8 steps/s, GPU about 40-60%, host CPU about 1/3, no disk wait. Profile: 97% of main-process samples in the batch-join thread; per-message receive cost; 32 messages per step at 256 vs 8 at 512. Synthetic ceiling 35 steps/s at 256, 70 at 512.
- SRM on the G4: 18.4 steps/s, GPU 100% at about 458 W (GPU-bound). Projected 93.6 min of 105.

### Decided

- User: SRM uses the 256 overlap recipe (train tile 256, overlap_last, batch 128, 180,000 tiles/epoch). No PC speed test; read the speed on the G4 and stop early if 30 epochs cannot fit. Kept running.
- User: build the loader fix (workers send packs of blocks; batches must stay byte-identical; can be turned off).
- User: Worker_4 runs every post-training check with the one tool.
- Orchestrator: `--settings-from` must also feed the main stage (code fix) for old runs such as wide-R2; no hand-made settings files.
- Rule: after every PC run or kill, check `pgrep -af spawn_main` for orphan workers.

### Next

- SRM result and evaluation. Pending.
- Loader fix (Worker_5). Pending.
- `--settings-from` fix, then missing stages on old models (wide-R2, newdata-1, 512 arms). Pending.
- Tanh saturation results across models. Pending.
