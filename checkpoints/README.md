# Checkpoints and run outputs

Use `checkpoints/<run>/` only as the notebook's working folder while a run is active. Put all finished PC and G4 run results under `runs-out/<run>/`. Notebook 05 headless runs already write their working checkpoints inside that result folder; keep only `last.ckpt` for resume and the single best validation-P_E checkpoint for evaluation. Do not move tracked files or the user's `baseline_kv_xunet_v3_clip3_group_otf/` folder.

Small outputs from these three runs are tracked in git:

| Run folder | Version |
| --- | --- |
| `baseline_kv_xunet_v3_clip3_batch/` | V3a: clipping and BatchNorm |
| `baseline_kv_xunet_v3_noclip_group/` | V3b: no clipping and GroupNorm |
| `baseline_kv_xunet_v3_clip3_group/` | V3c: clipping and GroupNorm |

Each folder contains test predictions, a per-setting CSV table, four heatmap
PNGs, and `logs/csv/run/metrics.csv` with small `hparams.yaml` files. V3c also
has `training_curves.png`. The older runs have a `NOTE.md` that gives the
original metrics path and links to the report plots.

## Model files are on the VM

Model files live at:

```text
/srv/aai3001/AAI3001-CVDL/checkpoints/<run>/
```

The VM holds the full v3a, v3b and v3c run folders, including the `.ckpt`
files. Fetch a best checkpoint from the repository root with:

```sh
scp 'aai-team@gcp-aai3001:/srv/aai3001/AAI3001-CVDL/checkpoints/baseline_kv_xunet_v3_clip3_group/best-*.ckpt' \
  runs-out/early-pc/baseline_kv_xunet_v3_clip3_group/
```

Use `last.ckpt` to resume training. Use `best-*.ckpt` for evaluation. Keep
local model copies out of git.

## Not tracked

Git ignores `.ckpt` files, per-run `logs/tb/` TensorBoard events, old
`lightning_logs/`, and `_quick` run folders. Local-only V1/V2 files and the old
executed V3b notebook were moved to `runs-out/early-pc/`, with the same folder names. Tracked records remain here. Do not stage other run folders.

## Watch TensorBoard

From the ROCm container:

```sh
podman exec aai3001 tensorboard \
  --logdir /work/runs-out --bind_all --port 6006
```

From the VM repository root:

```sh
uv run tensorboard --logdir runs-out --port 6006
```

Open `http://localhost:6006`. Notebook 04 gives the optional SSH tunnel for
the VM. Stop TensorBoard with `Ctrl+C` when finished.
