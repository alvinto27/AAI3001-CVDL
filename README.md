# AAI3001-CVDL

## Set up the Python environment

The project uses [uv](https://docs.astral.sh/uv/) and Python 3.12. `uv.lock` pins every package version. On Linux and Windows, PyTorch comes from the CUDA 13.0 index (`cu130`). On macOS, PyTorch comes from PyPI.

```sh
uv sync                                  # create .venv and install the locked packages
uv run python -c "import torch; print(torch.cuda.is_available())"
uv run python train.py                   # run a script in the environment
```

To add a package, run `uv add <package>`. Commit the changed `pyproject.toml` and `uv.lock` together.

`LSB-Dataset-Generator/` is a standalone tool. It keeps its own `requirements.txt` and also runs in this environment.
