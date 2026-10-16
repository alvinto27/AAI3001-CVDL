# G4 run queue

The Orchestrator keeps this list current. Add a run when the user approves it.
Change its state in the same step as the launch, the end, or the evaluation.
One G4 run per boot. Timeout 105 min; the VM stops 120 min after boot.

States: `queued` (approved, waiting) · `preparing` (code or notebook not ready)
· `running` · `delivered` (on the Pi, not evaluated) · `evaluated` · `failed`.

| # | Run (notebook) | Purpose | Commit | State | Job ID | Result |
| ---: | --- | --- | --- | --- | --- | --- |
| 1 | `experiments/tiling-overlap.ipynb` | Tiling arm 1: overlap-last, 512 | `8fff50d` | running (start 14:00Z, 2026-10-16) | `tiling-overlap-20261016T135926Z-8501` | — |
| 2 | `experiments/tiling-trim.ipynb` | Tiling arm 2: centre trim, 512 | `8fff50d` | queued | — | — |
| 3 | `experiments/tiling-pad.ipynb` | Tiling arm 3: centre zero-pad, 512 | `8fff50d` | queued | — | — |
| 4 | `experiments/tiling-overlap-256.ipynb` | Tiling arm 1 at 256: overlap-last (batch 128, 180,000 tiles/epoch, LR 3e-4) | — | preparing (Worker_5) | — | — |
| 5 | `experiments/tiling-trim-256.ipynb` | Tiling arm 2 at 256: centre trim (same 256 budget) | — | preparing (Worker_5) | — | — |
| 6 | `experiments/tiling-pad-256.ipynb` | Tiling arm 3 at 256: centre zero-pad (same 256 budget) | — | preparing (Worker_5) | — | — |

## Next (approved in principle, not queued)

- **SRM filter bank instead of the KV filter** (front-end test, aimed at weak
  k = 1 detection). Tiling strategy and tile size are decided after runs 1-6.
  Notebook build (user decision): Worker_5, right after the 256 notebooks.
  `tile_size`, `train_grid`, `batch_size` and `tiles_per_epoch` stay unset
  ("must be set"; run and G4 pre-flight fail while unset). Default front end
  stays `kv`; SRM = 30 fixed filters per colour channel, same +/-3 clip.

## After the runs

- Evaluate each delivered run on the PC with `scripts/evaluate_run.py` (plus
  the padding check). Then set its state to `evaluated` and link the result.
- Reports: 3-arm tiling comparison at 512 (runs 1-3) and at 256 (runs 4-6);
  256 vs 512 per arm (1/4, 2/5, 3/6) on the 512 tests and the 256 test panel.

## Done

| Run | Job ID | Result |
| --- | --- | --- |
| — | — | — |
