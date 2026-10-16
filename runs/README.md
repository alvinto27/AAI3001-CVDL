# Run outputs

`runs/` is the single folder for every PC and G4 run result: notebook
checkpoints, logs, figures, and (for headless jobs) `status.json` /
`executed.ipynb` / `nvidia.csv`. One subfolder per run, named after the run
or job ID. There is no separate "working" checkpoints folder any more:
notebooks write directly into `runs/<run-name>/` while a run is active, and
the same folder holds the finished result.

Small outputs from three of these runs are tracked in git:

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

`baseline_kv_xunet_v3_clip3_group_otf/` is the user's own OTF run (all
per-epoch checkpoints kept intentionally); it is not tracked in git and is
not touched by housekeeping passes here. It carries both executed notebooks
that produced it, `04_v3c_otf_2026-10-13.ipynb` (VM/L4) and
`04_v3c_otf_pc_2026-10-13.ipynb` (PC).

## Model files are on the VM

Model files for the frozen v3a/v3b/v3c runs live at:

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
listed above stayed tracked through the move from the old `checkpoints/` and
`runs-out/` folders. Do not stage other run folders.

## G4 batch runner: VM-internal paths are unchanged

`scripts/g4/job.sh`, `scripts/g4/job_run.sh`, and the VM-side `speed`
subcommand of `scripts/g4/setup.sh` still write to `runs-out/` **on the VM**
(for example `runs-out/.g4-state`, `runs-out/<job-id>/`). That is the VM's
own checkout, separate from this repository's `runs/`, and is left as-is.
When a finished job's result folder is copied from the Pi
(`/srv/cvdl-results/<job-id>/`) onto this PC, it lands under
`runs/<job-id>/`, not `runs-out/<job-id>/`. See
`docs/active/g4-batch-runner.md`.

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
