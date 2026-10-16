# Run outputs

`runs/` is the folder for notebook checkpoints, logs, and figures. One
subfolder per run, named after the run. There is no separate "working"
checkpoints folder any more: notebooks write directly into
`runs/<run-name>/` while a run is active, and the same folder holds the
finished result.

Small outputs from these three runs are tracked in git:

| Run folder | Version |
| --- | --- |
| `baseline_kv_xunet_v3_clip3_batch/` | V3a: clipping and BatchNorm |
| `baseline_kv_xunet_v3_noclip_group/` | V3b: no clipping and GroupNorm |
| `baseline_kv_xunet_v3_clip3_group/` | V3c: clipping and GroupNorm |

Each folder contains test predictions, a per-setting CSV table, four heatmap
PNGs, and `logs/csv/run/metrics.csv` with small `hparams.yaml` files. V3c also
has `training_curves.png` and the executed notebook
`04_v3c_clip3_group.ipynb`. The older runs have a `NOTE.md` that gives the
original metrics path and links to the report plots.

## Model files are on the VM

Model files live at:

```text
/srv/aai3001/AAI3001-CVDL/checkpoints/<run>/
```

(That path is the VM's own working checkout, unrelated to this repository's
`runs/`.) Fetch a best checkpoint from the repository root with:

```sh
scp 'aai-team@gcp-aai3001:/srv/aai3001/AAI3001-CVDL/checkpoints/baseline_kv_xunet_v3_clip3_group/best-*.ckpt' \
  runs/baseline_kv_xunet_v3_clip3_group/
```

Use `last.ckpt` to resume training. Use `best-*.ckpt` for evaluation. Keep
local model copies out of git.

## Not tracked

Git ignores all of `runs/` by default (`.ckpt` files, per-run `logs/tb/`
TensorBoard events, `lightning_logs/`, and everything else new); the files
listed above stayed tracked through the move from `checkpoints/`. Do not
stage other run folders.

## Watch TensorBoard

From the ROCm container:

```sh
podman exec aai3001 tensorboard \
  --logdir /work/runs --bind_all --port 6006
```

From the VM repository root:

```sh
uv run tensorboard --logdir runs --port 6006
```

Open `http://localhost:6006`. Notebook 04 gives the optional SSH tunnel for
the VM. Stop TensorBoard with `Ctrl+C` when finished.
