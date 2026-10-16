# AAI3001-CVDL

## Set up the Python environment

The project uses [uv](https://docs.astral.sh/uv/) and Python 3.12. `uv.lock` pins every package version. On Linux and Windows, PyTorch comes from the CUDA 13.0 index (`cu130`). On macOS, PyTorch comes from PyPI.

```sh
uv sync                                  # create .venv and install the locked packages
uv run python -c "import torch; print(torch.cuda.is_available())"
uv run python train.py                   # run a script in the environment
```

To add a package, run `uv add <package>`. Commit the changed `pyproject.toml` and `uv.lock` together.

`LSB-Dataset-Generator/README.md` documents the v6 image pipeline.
Run `prepare.py` to create RGB PNG tiles, then `generate.py` to create clean/stego pairs.
Use `generate.py --verify` to check deterministic replay.
See [the RAISE 512 rebuild guide](docs/archive/rebuild-raise512-run1.md) for the complete dataset command sequence.

## Notebooks

| File | Purpose |
| --- | --- |
| [`04_baseline_kv_xunet_v3.ipynb`](notebooks/04_baseline_kv_xunet_v3.ipynb) | Current training notebook. It has `CLIP_T` and `NORM` switches. V3c, with `CLIP_T=3` and `NORM="group"`, is the current baseline. |
| [`04_v3c_clip3_group.ipynb`](notebooks/runs/04_v3c_clip3_group.ipynb) | executed v3c run with outputs |
| [`04_v3c_otf_2026-10-13.ipynb`](notebooks/runs/04_v3c_otf_2026-10-13.ipynb) | executed first OTF v3c run (VM, L4) with outputs |
| [`02_eval_breakdown.ipynb`](notebooks/02_eval_breakdown.ipynb) | Per-setting evaluation of a `test_predictions.csv` file. |
| [`01_baseline_kv_xunet.ipynb`](notebooks/01_baseline_kv_xunet.ipynb) and [`03_baseline_kv_xunet_v2.ipynb`](notebooks/03_baseline_kv_xunet_v2.ipynb) | V1 and v2 history. They stay because the reports link them. |
| [`rocm_conv1_wgrad.py`](notebooks/rocm_conv1_wgrad.py) | ROCm-only conv1 speed fix used by notebook 04. |

Current results: [V3 comparison report](docs/results/results-v3.md).

Notebook 04 loader controls:

| Environment variable | Default and meaning |
| --- | --- |
| `V3_DATA` | `otf`; `png` keeps the original dataset path. |
| `V3_MAIN_THREADS` | 1 main-process torch CPU thread. |
| `V3_WORKERS` | 6 OTF training workers; PNG retains its platform-based count. |
| `V3_PRIME_NEXT_EPOCH` | `off` by default. Set `on` to refill the same persistent training workers during fit validation; it does not start workers for the sanity check or final epoch. Training images and order do not change. |
| `V3_EVAL_WORKERS` | 4 for val, test and train-eval; each worker owns whole-photo evaluation tasks and stops after each pass. |
| `V3_DIAG_WORKERS` | Defaults to `V3_EVAL_WORKERS`; workers that read the 1,000 train and 1,000 validation pairs for the final diagnostics cell. One worker task owns one photo, and the tile bytes and order do not change. |
| `V3_PREFETCH_IMAGES` | 128 images ahead per worker; 32 training groups of four. Evaluation rounds the lead to whole-photo tasks (one task at 512 validation). |
| `V3_PREFETCH` | Unset; optional training group-count override. Explicit 4 means only 16 images ahead. |
| `V3_SOURCE`, `V3_CACHE_ROOT` | Decode unless a cache root is supplied. |
| `V3_COMPILE` | `off` by default; `default` compiles the training network in place. Other modes are rejected. TF32 flags stay at their native values. |

Ordered delivery remains enabled. See [all controls and saved run settings](docs/archive/notebook-otf-verification.md#controls) and [compile verification and safety checks](docs/archive/notebook-compile-verification.md).

### Run notebook 04 without Jupyter

Create an env file with `QUICK` and any `V3_*` controls. Use a new output folder
for each run. The runner sets `V3_OUTPUT_ROOT` inside that folder; do not put it
in the env file.

```text
QUICK=False
V3_DATA=otf
V3_SOURCE=cache
V3_CACHE_ROOT=/path/to/otf-cache/v1
V3_RUN_NAME=my-run
```

```sh
python scripts/run_notebook.py --env-file run.env --output runs-out/my-run
```

`--notebook` selects another input notebook; `--set V3_WORKERS=4` overrides an
env-file value. The runner reads the source notebook but does not edit it. It
saves `executed.ipynb` and `status.json` after each code cell, and writes a
live `run.log`. The status heartbeat updates every 10 seconds. Check
`status.json` for `done` and exit code 0 before using results. A failed or
interrupted run exits non-zero and keeps its partial notebook and traceback.
Notebook checkpoints, predictions, figures and settings records remain under
`runs-out/my-run/checkpoints/<run name>/`; the status file lists them. Git
ignores `runs-out/`. The runner refuses a nonempty output folder.

## Data and citations

The `raise512-run1` source is RAISE. Cite it as [D.-T. Dang-Nguyen, C. Pasquini, V. Conotter, G. Boato, “RAISE – A Raw Images Dataset for Digital Image Forensics,” ACM Multimedia Systems, Portland, Oregon, March 18-20, 2015.](ImageSets/README.md#citations-and-terms) RAISE is for **non-commercial research and educational purposes**.

DIV2K has [frozen source records](docs/active/frozen-source-records.md); the builder reads its PNG headers. Notebook 04 now uses the [OTF loader by default](docs/archive/notebook-otf-verification.md), with a PNG fallback. Full OTF training is a user action. Cite it as [E. Agustsson and R. Timofte, “NTIRE 2017 Challenge on Single Image Super-Resolution: Dataset and Study,” IEEE Conference on Computer Vision and Pattern Recognition (CVPR) Workshops, July 2017.](ImageSets/README.md#citations-and-terms) The official page says DIV2K is for **academic research purposes only** and that copyright belongs to the original owners.

See the [ImageSets README](ImageSets/README.md) for source details, terms, and download links.

### RAISE records in git (`data/raise/`)

- `selection.csv` and `split_seed0.csv`: selected source photos and fixed splits.
- `photo_manifest.csv` and `legacy_tile_manifest.csv`: source identities and tile grids.
- `run1_labels.csv.gz` and `run1_metadata.csv.gz`: deterministic gzip copies of
  `Datasets/raise512-run1/labels.csv` and `metadata.csv`. See
  [hashes and decompression instructions](docs/active/frozen-source-records.md#preserved-run1-labels-and-metadata).

Dataset `masks/` and `records/` are not kept in git. Masks can be regenerated.
Dataset files remain local; this record preservation does not delete them.

## Train on the Spot VM

The `gcp-aai3001` VM is a Spot VM. Google can stop it at any time, with a 120-second notice.

### Team rule

Only run training that can resume on the VM. If a run cannot resume and the VM is preempted, the person who started it runs it again.

### Use the template

1. Copy `train_template.py` and edit only the two `EDIT` sections: the model and the data.
2. Run it in `tmux`, so it continues after you disconnect:
   ```sh
   uv run python my_train.py --run-name my-run --epochs 20
   ```
3. If the run stops, run the same command again. It continues from `checkpoints/my-run/last.ckpt`.
4. To start fresh, use a new `--run-name`.

The template uses PyTorch Lightning. It saves `last.ckpt` after every epoch (model, optimizer, and progress), and it also keeps the best epoch by `val_loss`. A preemption loses at most the epoch in progress.

### Final runs

For a long run that must not stop, change the VM to the standard provisioning model in the GCP console before you start. A standard VM is not preempted, but it costs more per hour. Change it back to Spot after the run.
