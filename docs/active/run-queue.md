# G4 run queue

The Orchestrator keeps this list current. Add a run when the user approves it.
Change its state in the same step as the launch, the end, or the evaluation.
One G4 run per boot. Since 2026-10-17 the boot limit is 180 min: use `--timeout-min 165 --max-run-minutes 180` for long runs (shorter runs may keep timeout 105).

States: `queued` (approved, waiting) · `preparing` (code or notebook not ready)
· `running` · `delivered` (on the Pi, not evaluated) · `evaluated` · `failed`.

| # | Run (notebook) | Purpose | Commit | State | Job ID | Result |
| ---: | --- | --- | --- | --- | --- | --- |
| 1 | `experiments/tiling-overlap.ipynb` | Tiling arm 1: overlap-last, 512 | `8fff50d` | evaluated (14:00-14:40Z; 39m49s; best epoch 29, val P_E .0586) | `tiling-overlap-20261016T135926Z-8501` | `runs/<job>/pc-comparison/` (standard P_E 2.80%; padding: misses about double at 50% pad) |
| 2 | `experiments/tiling-trim.ipynb` | Tiling arm 2: centre trim, 512 | `8fff50d` | evaluated (14:41-15:22Z; 39m53s notebook time) | `tiling-trim-20261016T144117Z-3222` | evaluated: standard P_E 4.07% |
| 3 | `experiments/tiling-pad.ipynb` | Tiling arm 3: centre zero-pad, 512 | `8fff50d` | evaluated (01:31-02:11Z; 40m05s; best epoch 27, val P_E .0668; 54.65 steps/s, GPU-bound) | `tiling-pad-20261017T013055Z-7210` | evaluated: standard P_E 4.32% (overlap 2.80, trim 4.07); padding-trained model cuts padding misses about 60% |
| 4 | `experiments/tiling-overlap-256.ipynb` | Tiling arm 1 at 256: overlap-last (batch 128, 180,000 tiles/epoch, LR 3e-4) | `30bac1f` | evaluated (02:22-03:18Z; 55m36s; 32-33 steps/s, loader-bound) | `tiling-overlap-256-20261017T022212Z-9778` | evaluated: standard P_E 2.25% (512 overlap 2.80); best so far |
| 5 | `experiments/tiling-trim-256.ipynb` | Tiling arm 2 at 256: centre trim (same 256 budget) | `30bac1f` | evaluated (03:20-04:16Z; 55m38s; best epoch 27, val P_E .0630) | `tiling-trim-256-20261017T032025Z-8579` | evaluated: standard P_E 3.55% (256 overlap 2.25) |
| 6 | `experiments/tiling-pad-256.ipynb` | Tiling arm 3 at 256: centre zero-pad (same 256 budget) | `30bac1f` | evaluated (04:31-05:27Z; 54m56s; best epoch 26, val P_E .0641; 33.1 steps/s, loader-bound) | `tiling-pad-256-20261017T043145Z-9721` | evaluated: standard P_E 3.60% |
| 7 | `experiments/srm.ipynb` | SRM front end (30 fixed filters per channel) with the 256 overlap recipe | `0d851b6` | evaluated (05:29-07:03Z; 93.1 min; 18.4 steps/s, GPU-bound; best epoch 29, val P_E .0466 vs 256 KV .0543) | `srm-20261017T052857Z-4521` | evaluated: standard P_E 1.67% (256 KV 2.25); better on all 5 tests and all sources; `runs/<job>/post-training/` |
| 8 | `experiments/srm-e50.ipynb` | SRM, 50 epochs, LR drops scaled to epochs 8 and 33 (user 2026-10-17); boot limit 180 min, timeout 165 | `2841fdf` | done (11:13-13:47Z; 153.1 min; best epoch 43, val P_E .0432, AUC .9917); evaluated (standard + hard v1) | `srm-e50-20261017T111310Z-0214` | — |
| 9 | `experiments/srm-hard.ipynb` | SRM hard training: per epoch RAISE/ALASKA/LSDIR 72,000 each (216,000), k weights 3:1:1:1, bpp 0.5 x U(0.05,0.15) + 0.5 x U(0.05,0.5) (`configs/train-hard-v1.yaml`); 30 epochs; ~112 min. Checkpoint validation: train-matched recipe `bf7ede374cab01f73f5d` (21,600 pairs, 256). Timeout 165, boot limit 180 | `a0e72cd` | **running** (launched 2026-10-18 04:38:54Z, VM start 04:42:34Z, first attempt; supervised by Run_Supervisor) | `srm-hard-20261018T043854Z-8467` | — |

## Next (approved in principle, not queued)

- None. (The SRM front end became runs 7-9.)

## PC task queue (free, no G4)

| Task | State | Note |
| --- | --- | --- |
| SRM notebook build | done (`632dd22`, `97487b8`, `0d851b6`) | run 7 |
| Padding check fix | done (`970d337`) | worker DataLoader, 2,000 pairs per shape and level (18,000 cases) |
| Temperature scaling | done (`f0bb81d`) | opt-in stage; helps only wide-R2 (ties) and slightly 256 overlap / SRM |
| tanh saturation | done (`0e05dcb`) | opt-in stage; only SRM tanh1 saturates (20.8% / 26.4%) |
| 256 loader speed | done (`863d9fb`, `efde4df`) | `pack_blocks=16` default; batches identical; G4 gain not measured (SRM GPU-bound) |
| SRM weakness breakdown | done | `runs/srm-20261017T052857Z-4521/post-training/breakdown/` |
| Hard test v1 | done (`d74d716`) | `data/hard-test/v1/`; default evaluation stage |
| `srm-hard` notebook | done (`795fd93`) | run 9, launch-ready |
| Photo-level decision | deferred by the user | — |

## After the runs

- Evaluate each delivered run on the PC with `scripts/evaluate_run.py` (plus
  the padding check). Then set its state to `evaluated` and link the result.
- Reports: 3-arm tiling comparison at 512 (runs 1-3) and at 256 (runs 4-6);
  256 vs 512 per arm (1/4, 2/5, 3/6) on the 512 tests and the 256 test panel.

## Done

| Run | Job ID | Result |
| --- | --- | --- |
| — | — | — |
