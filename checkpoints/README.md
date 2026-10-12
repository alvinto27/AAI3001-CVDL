# Checkpoints

This folder stores local training outputs. The outputs are not tracked. Only
this README is tracked.

## Run folders

Notebook 04 creates one folder per run. The full run name is built from
`CLIP_T` and `NORM`; `_quick` is added when `QUICK=True`.

| Example | Meaning |
| --- | --- |
| `baseline_kv_xunet_v3_clip3_group` | V3c: `CLIP_T=3`, `NORM="group"` |
| `baseline_kv_xunet_v3_clip3_group_quick` | The same settings with the quick test mode |

A run folder can contain:

- `epochNN-*.ckpt`, the checkpoint for each epoch.
- `best-*.ckpt`, the checkpoint selected by validation P_E.
- `last.ckpt`, the resume checkpoint.
- `test_predictions.csv` and `test_predictions_per_setting.csv`.
- Four per-setting heatmaps and `training_curves.png`.
- `logs/csv/run/metrics.csv` and `logs/tb/run/` TensorBoard events.

Older `lightning_logs/version_N/` folders hold outputs from v1, v2, and v3a.
The current local checkpoint tree is about 26 MB.

## Watch TensorBoard

Notebook 04 writes CSV and TensorBoard logs inside each run folder. From the
ROCm container, use:

```sh
podman exec aai3001 tensorboard \
  --logdir /work/checkpoints --bind_all --port 6006
```

From the VM repository root, use:

```sh
uv run tensorboard --logdir checkpoints --port 6006
```

Open `http://localhost:6006`. Notebook 04 documents the optional SSH tunnel
for the VM. Stop TensorBoard with `Ctrl+C` when finished.

## Do not commit

Never commit `.ckpt` files, prediction CSV files, heatmaps, training curves,
TensorBoard events, or other run outputs. Keep only this README in git.
