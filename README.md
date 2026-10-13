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
See [the RAISE 512 rebuild guide](docs/rebuild-raise512-run1.md) for the complete dataset command sequence.

## Notebooks

| File | Purpose |
| --- | --- |
| [`04_baseline_kv_xunet_v3.ipynb`](notebooks/04_baseline_kv_xunet_v3.ipynb) | Current training notebook. It has `CLIP_T` and `NORM` switches. V3c, with `CLIP_T=3` and `NORM="group"`, is the current baseline. |
| [`04_v3c_clip3_group.ipynb`](notebooks/runs/04_v3c_clip3_group.ipynb) | executed v3c run with outputs |
| [`02_eval_breakdown.ipynb`](notebooks/02_eval_breakdown.ipynb) | Per-setting evaluation of a `test_predictions.csv` file. |
| [`01_baseline_kv_xunet.ipynb`](notebooks/01_baseline_kv_xunet.ipynb) and [`03_baseline_kv_xunet_v2.ipynb`](notebooks/03_baseline_kv_xunet_v2.ipynb) | V1 and v2 history. They stay because the reports link them. |
| [`rocm_conv1_wgrad.py`](notebooks/rocm_conv1_wgrad.py) | ROCm-only conv1 speed fix used by notebook 04. |

Current results: [V3 comparison report](docs/results-v3.md).

## Data and citations

The `raise512-run1` source is RAISE. Cite it as [D.-T. Dang-Nguyen, C. Pasquini, V. Conotter, G. Boato, “RAISE – A Raw Images Dataset for Digital Image Forensics,” ACM Multimedia Systems, Portland, Oregon, March 18-20, 2015.](ImageSets/README.md#citations-and-terms) RAISE is for **non-commercial research and educational purposes**.

DIV2K has [frozen source records](docs/frozen-source-records.md); the builder reads its PNG headers. Training integration is Pending. Cite it as [E. Agustsson and R. Timofte, “NTIRE 2017 Challenge on Single Image Super-Resolution: Dataset and Study,” IEEE Conference on Computer Vision and Pattern Recognition (CVPR) Workshops, July 2017.](ImageSets/README.md#citations-and-terms) The official page says DIV2K is for **academic research purposes only** and that copyright belongs to the original owners.

See the [ImageSets README](ImageSets/README.md) for source details, terms, and download links.

### RAISE records in git (`data/raise/`)

- `selection.csv` and `split_seed0.csv`: selected source photos and fixed splits.
- `photo_manifest.csv` and `legacy_tile_manifest.csv`: source identities and tile grids.
- `run1_labels.csv.gz` and `run1_metadata.csv.gz`: deterministic gzip copies of
  `Datasets/raise512-run1/labels.csv` and `metadata.csv`. See
  [hashes and decompression instructions](docs/frozen-source-records.md#preserved-run1-labels-and-metadata).

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
