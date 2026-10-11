# AAI3001-CVDL

## Set up the Python environment

The project uses [uv](https://docs.astral.sh/uv/) and Python 3.12. `uv.lock` pins every package version. On Linux and Windows, PyTorch comes from the CUDA 13.0 index (`cu130`). On macOS, PyTorch comes from PyPI.

```sh
uv sync                                  # create .venv and install the locked packages
uv run python -c "import torch; print(torch.cuda.is_available())"
uv run python train.py                   # run a script in the environment
```

To add a package, run `uv add <package>`. Commit the changed `pyproject.toml` and `uv.lock` together.

## Pending: update uv.lock

`pyproject.toml` now lists pandas, matplotlib, and rawpy. `uv.lock` is out of date.
Pandas and matplotlib support notebooks. rawpy supports RAW input to `LSB-Dataset-Generator/prepare.py`.
On a device with uv, such as the gcp-aai3001 VM, run `uv lock`.
Check that torch and torchvision still use the `pytorch-cu130` index.
Run `uv sync` and `uv run python -c "import pandas, matplotlib, rawpy, torch; print(torch.cuda.is_available())"`.
Commit `pyproject.toml` if uv changed it and `uv.lock` together.
Delete this section after the lock update.
Until then, `uv sync` updates the lock by itself and `uv sync --locked` fails.

`LSB-Dataset-Generator/README.md` documents the v6 image pipeline.
Run `prepare.py` to create RGB PNG tiles, then `generate.py` to create clean/stego pairs.
Use `generate.py --verify` to check deterministic replay.

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
