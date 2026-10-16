# G4 run queue

The Orchestrator keeps this list current. Add a run when the user approves it.
Change its state in the same step as the launch, the end, or the evaluation.
One G4 run per boot. Timeout 105 min; the VM stops 120 min after boot.

States: `queued` (approved, waiting) · `preparing` (code or notebook not ready)
· `running` · `delivered` (on the Pi, not evaluated) · `evaluated` · `failed`.

| # | Run (notebook) | Purpose | Commit | State | Job ID | Result |
| ---: | --- | --- | --- | --- | --- | --- |
| 1 | `experiments/tiling-overlap.ipynb` | Tiling arm 1: overlap-last, 512 | `8fff50d` | evaluated (14:00-14:40Z; 39m49s; best epoch 29, val P_E .0586) | `tiling-overlap-20261016T135926Z-8501` | `runs/<job>/pc-comparison/` (standard P_E 2.80%; padding: misses about double at 50% pad) |
| 2 | `experiments/tiling-trim.ipynb` | Tiling arm 2: centre trim, 512 | `8fff50d` | delivered (14:41-15:22Z; 39m53s notebook time) | `tiling-trim-20261016T144117Z-3222` | evaluate 2026-10-17 (main eval; padding check after the fix) |
| 3 | `experiments/tiling-pad.ipynb` | Tiling arm 3: centre zero-pad, 512 | `8fff50d` | queued: paused until 2026-10-17 (user) | — | — |
| 4 | `experiments/tiling-overlap-256.ipynb` | Tiling arm 1 at 256: overlap-last (batch 128, 180,000 tiles/epoch, LR 3e-4) | `30bac1f` | queued (dry run OK) | — | — |
| 5 | `experiments/tiling-trim-256.ipynb` | Tiling arm 2 at 256: centre trim (same 256 budget) | `30bac1f` | queued (dry run OK) | — | — |
| 6 | `experiments/tiling-pad-256.ipynb` | Tiling arm 3 at 256: centre zero-pad (same 256 budget) | `30bac1f` | queued (dry run OK) | — | — |

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
| SRM notebook build | done (`632dd22`, `97487b8`); 5 values unset; PC needs `rocm_conv1_fix=off`; G4 time untested | tile size, grid, batch, tiles/epoch unset |
| **Fix the padding check first (blocks all padding checks)** | approved, first task on 2026-10-17 | the 2026-10-16 check ran in one process (~74% of one core, no data workers; 65,736 cases for one model). Fix: multi-worker loader for decode/embed (like the main evaluation) and a cap of 2,000 pairs per shape and level, the same seeded pairs for every model. Do not resume or rerun any padding check until this is fixed and reviewed. |
| Temperature scaling in `scripts/evaluate_run.py` | approved, queued (next free worker) | one T fitted on validation; report calibration, 1.0 ties and FA5 reachability before/after; add only (old outputs unchanged); also run on wide-R2, newdata-1 and the tiling models |
| tanh saturation measurement | approved, queued (next free worker) | existing models, PC; share of tanh inputs with abs > 3 per layer, clean vs stego; a G4 run only if a fix is tested later |
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
