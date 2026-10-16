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
| 4 | `experiments/tiling-overlap-256.ipynb` | Overlap-last at 256 (batch 128, 180,000 tiles/epoch, LR 3e-4); compare with run 1 on 512 and 256 tests | — | preparing (Worker_5) | — | — |

## After the runs

- Evaluate each delivered run on the PC with `scripts/evaluate_run.py` (plus
  the padding check). Then set its state to `evaluated` and link the result.
- Reports: 3-arm tiling comparison (runs 1-3); 256 vs 512 comparison (runs
  1 and 4) on the 512 tests and the 256 test panel.

## Done

| Run | Job ID | Result |
| --- | --- | --- |
| — | — | — |
