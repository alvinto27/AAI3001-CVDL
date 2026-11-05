# 2026-11-05 — run 21 attribution heatmap scored

## Did

- Scored the run 21 BEST signed covariance attribution map (`class_score`) on the frozen cell fixture `configs/cell-diagnostic-v1.json`. This is the same 192 validation / 256 test pairs used for run 18 and run 22. User approved this local PC check.
- Checked the run 21 BEST SHA-256 before the run: `4ba2a4b59a535b53ab9c0235c54231476b45e2193982bf28f7b7f2171d9898cf` (matches the [run21 plan](../../active/plan-run21-cell-heatmap.md)).
- Command (container, ROCm GPU, 2 workers, 4 pairs per batch):

```sh
podman exec -w /work aai3001 python -B -u scripts/cell_eval.py \
  runs/srm-e50-paired-gapcov32-20261026T115421Z-1917/checkpoints/srm-e50-paired-gapcov32-20261026T115421Z-1917/best-epoch43-val_pe0.0329-val_auc0.9946.ckpt \
  --manifest configs/cell-diagnostic-v1.json --device cuda --workers 2 --batch-pairs 4 --figure \
  --output runs/verification/run21-attribution-cell-eval-2026-11-05/cells
```

  **Exit 0, 65.1 s.** No code change. GPU memory and `/dev/shm` back at baseline; no leftover workers.
- Ran a one-off saved-score paired comparison (CPU) of run 21 attribution against run 18 aux, run 22 aux and run 22 attribution. The script checks identical pair order, targets and valid-negative masks. It reproduces the published run 18 (.8669) and run 22 (.8635) primary values exactly.

## Found

Primary metric: mean per-tile within-stego AUC, 128 contiguous test tiles, 2,000 source-photo resamples, seed 20261021. FA and recall at each map's own validation 5%-FA cutoff, pooled test.

| Map | Within-tile AUC [95%] | Pooled-cell AUC | Test cell FA | Recall |
| --- | --- | --- | ---: | ---: |
| Run 18 trained head (shipped) | .8669 [.8381, .8931] | .8664 | 4.21% | 16.14% |
| Run 22 trained head | .8635 [.8350, .8911] | .8735 | 4.51% | 16.69% |
| Run 22 attribution | .7595 [.7245, .7946] | .7786 | 4.76% | 15.92% |
| **Run 21 attribution** | **.7518 [.7176, .7859]** | .7694 | 4.69% | 14.72% |

| Paired difference (within-tile AUC) | Point [95%] |
| --- | --- |
| Run 21 attribution − run 18 trained head | **−.1151 [−.1406, −.0902]** |
| Run 21 attribution − run 22 trained head | **−.1117 [−.1370, −.0870]** |
| Run 21 attribution − run 22 attribution | −.0077 [−.0224, +.0062] |

- The run 21 attribution map is clearly worse at localization than either trained head (about −.11 AUC; interval far from zero).
- Run 22's extra head training did not change its attribution map in a clear way (−.008, interval crosses zero).
- Figure `cells/cells.png`: attribution finds the changed band on a flat RAISE tile, but marks only one cell on a textured ALASKA tile.
- Limits: one fixed fixture (not whole-photo performance); contiguous placement only for AUC; one training seed; attribution is tile-global context (256×256), not a local mask.

## Decided

- No decision. Run 18 stays the submission and `pipeline` default.

## Next

- **Pending (user):** submission choice. Run 21 alone would ship a map about .11 AUC worse than run 18. Options include run 18 (current), run 22, or run 21 verdict + run 18 heatmap (two models).

Evidence (git-ignored): `runs/verification/run21-attribution-cell-eval-2026-11-05/` (`cells/`, `cell.log`, `cell.exit`, `compare.py`, `compare.log`, `comparison.json`).
