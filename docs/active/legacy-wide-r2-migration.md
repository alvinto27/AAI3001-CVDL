# Wide-R2 legacy settings migration

The historical wide-R2 run has no `run_settings.json`. Its old data record also
has no `checkpoint_validation`, `settings.source_mix`, or
`settings.tiles_per_epoch`. The evaluator must not use general defaults for
these missing fields.

Run: `xunet_v4_clip3_nonorm_lr0.0003_warm1_drops5x0.3-20x0.1_k1to4_e30_wide-w4_r2-20261015T121753Z-29499`.
Its checkpoint folder uses the same name without the final timestamp suffix.

## Saved evidence

Paths below are relative to that run in `runs/`.

- `checkpoints/<checkpoint-name>/data_run_settings.json`: RAISE and DIV2K;
  512 overlap-last; full training photo fraction; one variant; no source weights.
  SHA-256: `bb9945fbd236497823c6b8e84bdf3f68b9e9300b39ed7683d5372154ed0b4ebb`.
- `checkpoints/<checkpoint-name>/model_run_settings.json`: 3,372 training steps,
  batch size 32 images. This equals `ceil(2 * 53,950 / 32)`, where 53,950 is
  the full selected training tile count recorded in the wide calibration
  `training-evidence.json` under `data/val/8328cc6b0673cade0312/`.
  SHA-256: `9835987926ba07f439e262c7bd6d3d5a66d3edc58ab42cc847dd3e5e8eb5d24c`.
- `executed.ipynb`, cell 2, source lines 187-193: the saved `RunSettings` call
  has no source mixture or epoch quota. `NotebookData(ROOT, settings,
  quick=QUICK, profile=EVAL_PROFILE)` has no alternative validation selection.
- The same notebook, cell 12, source lines 94-98: `val_dataset` is
  `otf_data.val`; the validation loader comes directly from `otf_data.loaders`.
  Its saved output reports 703 validation batches.
- Cell 19, source line 530: training receives that `val_loader`.
- Cell 21, saved output: **22,476 validation images**, or **11,238 pairs**.
  This is all 9,782 RAISE plus 1,456 DIV2K validation tiles at 512. It is not
  a capped or mixed-source validation list.
- Notebook SHA-256:
  `431f0d2188f2641f0cc0a9b1af04a18e6dd4be9ae420f196164ed3447f52ed64`.

## Resolved values and limits

Only this exact set of three missing fields, with all three evidence hashes
unchanged, resolves to:

```text
checkpoint_validation = standard
source_mix = None
tiles_per_epoch = None
```

`scripts/evaluate_run.py` checks the evidence, prints the migration, and records
it in the evaluation provenance. It does not change the saved training records.
Missing or changed evidence, or other missing required fields, causes a clear
error. A matching run name alone is not evidence.

This migration restores the original settings only. Deployment calibration
still uses the frozen checkpoint mapping: best checkpoint
`9ba63252e8eecbf398f646f2d69641b70ad37c914ed8b7d6b4665dda54a961e1`
uses recipe `8328cc6b0673cade0312`, labelled `post_hoc_256`.
