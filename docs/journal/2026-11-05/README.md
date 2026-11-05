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

- **User: ship run 22** as the submission model (replaces run 18). Run 21 alone was not chosen because of its weaker map.

## Run 22 shipped

### Did

- Photo rule (c) re-fitted for run 22 with the frozen method of runs 13/18. Same photos, splits, variants and seeds as run 18 (checked: `photos` lists identical); only the cut-offs are run 22's own (tile .5245211124 from `a1ab2faad8f27bca309e`, cell .3114905656 from the run 22 cell summary). Photo list SHA-256 `192006e2...900b`.
- GPU cache, 1,024 photos × 13 versions: **22.4 min, exit 0**. CPU fit/freeze on validation only: frozen rule SHA-256 **`b5d8ba6f1e0faf85575aa4e8d8fddfdae394fe86ff0fa4884b759421131130fc`**, unchanged after the single TEST score. Rule (c) logit cut-off .6958.
- `pipeline` `044c1ad`: ported `xunet_wide_gapcov32` inference (`cvdl/models.py`, `cov_pool.py`, `cell_diagnostics.py`, `photo_inference.py` byte-identical to `yx`). Bundle contract accepts the covariance model only with its exact `cov_pool_identity` and the trained-head map. Built `deploy/run22-rule-c-v1` (`weights.pt` `6d58acdb...42c1`, `deploy.json` `3aae1259...3203`); made it the default. Run 18 and run 13 bundles stay selectable. Dockerfile and `.dockerignore` include `cov_pool.py` and the new bundle.
- Tests on `pipeline`: 2 new covariance-bundle contract tests; parity replay now covers run 22 (default) and run 18. FULL **419 tests OK, exit 0, 526.8 s**. A first FULL run had 3 environment failures (worktree lacked `runs/.scratch` and older run folders); with temporary links to the main `runs/` they pass; links removed after.
- Podman CPU image: `/health` reports `run22-rule-c`; DIV2K 0281 `not_detected`, about 1.05 s, logit within 1.5×10⁻⁷ of project CPU. Test image removed.

### Found

| TEST photos (576), rule (c) | Run 18 | Run 22 |
| --- | ---: | ---: |
| Clean photos flagged (FA) | 8.681% | **5.556%** |
| Messages missed | 8.362% | 10.185% |
| P_E | 8.521% | 7.870% |
| AUC | .96475 | .96686 |
| ALASKA unseen FA | 16.406% | **8.594%** |
| Hard k=1 .05-.15 bpp P_E | 20.536% | 22.592% |
| Practical mark (FA <= 7% and hard P_E <= 25%) | FAIL | **PASS** |

- Paired photo bootstrap vs run 18 (descriptive; panel-stratified, 2,000 draws, seed 20261021): FA **−3.125 [−5.382, −1.037]**, miss **+1.823 [+1.013, +2.705]**, P_E −0.651 [−1.635, +0.311].
- Validation parity (CPU app vs cached GPU features), run 22: tile delta 0.0; feature delta ≤ 1.729×10⁻⁶; logit delta ≤ 4.472×10⁻⁶; 6/6 verdicts match.

### Next

- **Pending (user):** push `yx` and `pipeline`; merge to `main` (C1).
- HEIC input is still not supported by the app (PNG/JPEG/TIFF only).

Evidence (git-ignored): `runs/verification/photo-verdict-run22-2026-11-05/` (`REPORT.md`, `frozen-rules.json`, `compare-run18.json`, `compare_run18.py`, cache/fit/score logs, `pipeline-full*.log`).

Evidence (git-ignored): `runs/verification/run21-attribution-cell-eval-2026-11-05/` (`cells/`, `cell.log`, `cell.exit`, `compare.py`, `compare.log`, `comparison.json`).
