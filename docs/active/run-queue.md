# G4 run queue

The Orchestrator keeps this list current. Add a run when the user approves it.
Change its state in the same step as the launch, the end, or the evaluation.
One G4 run per boot. Timeout 105 min; the VM stops 120 min after boot.

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
| 8 | `experiments/srm-e50.ipynb` | SRM, 50 epochs, LR drops scaled to epochs 8 and 33 (user 2026-10-17); boot limit 180 min, timeout 165 | `2841fdf` | running (start 11:13Z; attempt 2 after 1 STOCKOUT; ~154 min) | `srm-e50-20261017T111310Z-0214` | — |
| 9 | `experiments/srm-hard.ipynb` | SRM hard training: per epoch RAISE/ALASKA/LSDIR 72,000 each (216,000), k weights 3:1:1:1, bpp 0.5 x U(0.05,0.15) + 0.5 x U(0.05,0.5) (`configs/train-hard-v1.yaml`); 30 epochs; ~112 min | `795fd93` | queued: **launch-ready, launch on the user's go** (dry run PASS 11:51Z from a clean worktree). Command: `cd runs/.scratch/srm-hard-launch && scripts/g4/submit.sh run --notebook notebooks/experiments/srm-hard.ipynb --commit 795fd93ffe1c287b56083fed11729bbe9cee7cdd --timeout-min 165 --max-run-minutes 180` | — | — |

## Next (approved in principle, not queued)

- **SRM filter bank instead of the KV filter** (front-end test, aimed at weak
  k = 1 detection). Tiling strategy and tile size are decided after runs 1-6.
  Notebook build (user decision): Worker_5, right after the 256 notebooks.
  `tile_size`, `train_grid`, `batch_size` and `tiles_per_epoch` stay unset
  ("must be set"; run and G4 pre-flight fail while unset). Default front end
  stays `kv`; SRM = 30 fixed filters per colour channel, same +/-3 clip.

## PC task queue (free, no G4)

| Task | State | Note |
| --- | --- | --- |
| SRM notebook build | done (`632dd22`, `97487b8`); user chose 2026-10-17: train tile 256, grid overlap_last, batch 128, 180,000 tiles/epoch (same as 256 overlap). G4 run approved after the 256 pad evaluation, no PC speed test: read the G4 speed after warm epoch 2; stop the VM early if 30 epochs cannot finish inside 105 min | values to fill in by Worker_4 |
| **Fix the padding check first (blocks all padding checks)** | approved, first task on 2026-10-17 | the 2026-10-16 check ran in one process (~74% of one core, no data workers; 65,736 cases for one model). Fix: multi-worker loader for decode/embed (like the main evaluation) and a cap of 2,000 pairs per shape and level, the same seeded pairs for every model. Do not resume or rerun any padding check until this is fixed and reviewed. |
| Temperature scaling in `scripts/evaluate_run.py` | approved, queued (next free worker) | one T fitted on validation; report calibration, 1.0 ties and FA5 reachability before/after; add only (old outputs unchanged); also run on wide-R2, newdata-1 and the tiling models |
| tanh saturation measurement | approved, queued (next free worker) | existing models, PC; share of tanh inputs with abs > 3 per layer, clean vs stego; a G4 run only if a fix is tested later |
| Photo-level decision | deferred by the user | — |
| 256 loader speed | approved 2026-10-17: Worker_5 implements the fix after its profile (in parallel with the SRM run) | 256 overlap on the G4: 32-33 steps/s vs 54.6 at 512 (GPU 40-55%, host CPU about 1/3, main process about 72% of one core, workers about 1/3 core each, no disk wait). Hypothesis: per-tile hand-off overhead (4x more items per step). Fix idea: workers build whole batches; profile on the PC first; prove identical data hashes. Runs 5-6 keep the current code. |

## After the runs

- Evaluate each delivered run on the PC with `scripts/evaluate_run.py` (plus
  the padding check). Then set its state to `evaluated` and link the result.
- Reports: 3-arm tiling comparison at 512 (runs 1-3) and at 256 (runs 4-6);
  256 vs 512 per arm (1/4, 2/5, 3/6) on the 512 tests and the 256 test panel.

## Done

| Run | Job ID | Result |
| --- | --- | --- |
| — | — | — |
