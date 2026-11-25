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

## Report runs: summaries tracked

The README's model-history ladder compares these three runs. Their summary
files are tracked (added with `git add -f`) so that every number in the
report can be checked against a file.

| Run folder | Model | Evaluation |
| --- | --- | --- |
| `g4-prime-1/` | V3c baseline (KV XuNet + GroupNorm), re-scored | `29fb97d0c311a0eca3b1` |
| `srm-e50-paired-gapcov32-20261026T115421Z-1917/` | Run 21: SRM filters, paired batches, covariance pooling | `cdc91a4b6d63a1517064` |
| `srm-cell-aux-paired-gapcov32-20261029T063502Z-5452/` | Run 22, the shipped model: run 21 + trained cell head | `a1ab2faad8f27bca309e` |

Each folder has:

| Path | Content |
| --- | --- |
| `evaluation/standard-256-v1/<evaluation>/summary.json`, `main_summary.csv`, `diagnostic_summary.csv` | Standard-test scores (P_E, AUC, false alarms, misses) |
| `…/panels/<panel>/` | The same scores for ALASKA seen, ALASKA unseen, LSDIR and the standard panel |
| `…/hard-test/hard_summary.json`, `hard_cells.csv` | Hard-test scores (k=1, low payload) |
| `…/validation/calibration.json` | The decision threshold and how it was chosen |
| `…/evaluation_settings.json` | The test suite, recipe and checkpoint used |
| `checkpoints/<run>/logs/csv/run/metrics.csv`, `training_curves.png` | Loss and accuracy per epoch |
| `checkpoints/<run>/run_settings.json`, `data_run_settings.json`, `model_run_settings.json` | Training settings and the exact training data selection |
| `report.ipynb` / `report_22.ipynb` (runs 21 and 22) | The executed report notebook with all outputs |

Not tracked: checkpoints (the shipped weights are in
`deploy/run22-rule-c-v1/`), per-tile prediction CSVs (about 90 MB per run),
TensorBoard events and machine logs.

## Model files

The checkpoints of the frozen v3a/v3b/v3c runs (and the user's OTF run) were
on the team VM at `/srv/aai3001/AAI3001-CVDL/checkpoints/<run>/`. That old VM
checkout was removed on 2026-11-25, after a full copy was checked by SHA-256
(checkpoints, dataset records, logs). The copy is outside git on the
maintainer's PC, in `runs/vm-backup-2026-11-25/`. Ask for a file if you need
it.

The shipped model's weights are tracked in `deploy/run22-rule-c-v1/`
(`weights.pt`, `model.onnx`).

Use `last.ckpt` to resume training. Use `best-*.ckpt` for evaluation. Keep
local model copies out of git.

## Not tracked

Git ignores all of `runs/` by default (`.ckpt` files, per-run `logs/tb/`
TensorBoard events, `lightning_logs/`, and everything else new); the files
listed above stayed tracked through the move from the old `checkpoints/` and
`runs-out/` folders, and the report-run summaries were added with
`git add -f`. Do not stage other run folders.

## G4 batch runner: VM-internal paths are unchanged

`scripts/g4/job.sh`, `scripts/g4/job_run.sh`, and the VM-side `speed`
subcommand of `scripts/g4/setup.sh` still write to `runs-out/` **on the VM**
(for example `runs-out/.g4-state`, `runs-out/<job-id>/`). That is the VM's
own checkout, separate from this repository's `runs/`, and is left as-is.
When a finished job's result folder is copied from the Pi
(`/srv/cvdl-results/<job-id>/`) onto this PC, it lands under
`runs/<job-id>/`, not `runs-out/<job-id>/`. See
`docs/archive/g4-batch-runner.md`.

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
