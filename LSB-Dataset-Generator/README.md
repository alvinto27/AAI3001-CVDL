# LSB Dataset Generator

Standalone desktop GUI and command-line generator, version **3.0.0**. It produces clean/stego PNG pairs, exact Boolean modification masks, selected embedding locations and the payload rate used for each image. It performs no dataset splitting, training, resizing or inference.

## Open on this computer

Double-click **Launch GUI.vbs** in this folder. No browser or server is needed. The launcher finds Python by itself, checking a `.venv` folder here, the path in `.runtime-path.txt` if that file exists, the Windows Python launcher, an installed Python, and finally `pythonw.exe` on `PATH`. If it finds nothing it lists the locations it searched. **No particular Python version is required.** Any Python 3.11 or newer with Tcl/Tk works, and an existing installation can be used as it is. You can also start `gui.py` yourself with any Python that has the dependencies.

1. Browse to your input folder. Only static **8-bit RGB PNG** is accepted.
2. Choose an output parent folder. The GUI suggests a new run subfolder and you can edit its path. The output folder must be new or empty and separate from the input tree.
3. Select **Fixed rate** or **Random range**.
   - Fixed rate uses the same rate for every image, for example `0.4` selects 40 percent of the available RGB-channel LSB positions.
   - Random range draws one rate per image between the minimum and maximum fields.
   - Both bounds stay editable and the defaults `0.1` and `0.9` are only starting values. Any range satisfying `0 < min <= max <= 1` is accepted, for example `0.1` to `0.8`.
4. Set the run seed. One seed controls the whole run.
5. Click **Generate dataset**. The log reports skipped inputs and write failures. **Cancel** finishes the current image before stopping, and completed samples remain available.
6. Use **Open output folder** for the generated files.

At larger Windows display scales, use the right-hand page scrollbar to reach the full results area. The log has its own scrollbar.

**Save config** and **Load config** use YAML. Relative paths in a loaded YAML file resolve beside that file. A saved effective run configuration points to the original input and output paths, so choose a new output before rerunning it.

## Set up on another Windows computer

Install any **Python 3.11 or newer** with Tcl/Tk support. Version 3.12 and 3.13 are both known to work. If the launcher already finds an installed Python that has NumPy, Pillow and PyYAML, no further setup is needed.

To keep the dependencies separate and pinned, create a local environment in this folder instead:

```powershell
py -3 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe gui.py
```

The launcher prefers `.venv\Scripts\pythonw.exe` when that folder exists, and otherwise uses whatever Python it finds. `requirements.txt` pins the versions for that optional environment only. The ignored `.runtime-path.txt`, `.vendor/` and `.tcl/` items are local conveniences for particular computers, not part of the portable source package. No administrator installation is required by the application itself. If startup fails, the launcher displays an error and writes `startup-error.log`.

## Command line

The CLI uses exactly the same core as the GUI and can run without a display:

```powershell
python generate.py --config example_config.yaml
python generate.py --input C:\data\sources --output C:\data\new-run --payload-rate 0.4 --run-seed 42
python generate.py --version
```

The example configuration expects `example_sources/`, which contains three clearly synthetic test images. It is a smoke test, not an experimental dataset. Edit the configuration or use the GUI for real sources. The example output folder already holds a completed smoke run on this computer, so delete it or change `output_dir` before running the example again, because existing output is never overwritten. Range mode is configured through YAML or the GUI. `--config` supplies the full configuration and the direct flags are used when no configuration file is given. Ctrl+C requests cancellation after the current image. Exit code 0 means completed, and 1 means cancelled or failed.

## Output structure

```text
output/
├── clean/000001.png
├── stego/000001.png
├── masks/000001.npy
├── selected_locations/000001.npy
├── metadata.csv
├── run_config.yaml
├── run_summary.json
└── generation.log
```

| Output | Contents |
| --- | --- |
| `clean/<source_id>.png` | Source image saved again through the same PNG path as stego, original dimensions |
| `stego/<source_id>.png` | Image after LSB replacement |
| `masks/<source_id>.npy` | Boolean `H × W × 3` array, true only where that channel value actually changed |
| `selected_locations/<source_id>.npy` | Unique flat RGB-channel indices selected for embedding |
| `metadata.csv` | One row per generated pair |
| `run_config.yaml` | Effective configuration of the run |
| `run_summary.json` | Run status and counts |
| `generation.log` | Run start, configuration, skipped inputs, failures and final counts |

Read NumPy files with `np.load(path, allow_pickle=False)`. A flat index `i` maps to row `i // (width * 3)`, column `(i // 3) % width` and channel `i % 3`, in R/G/B order.

Source IDs are sequential six-digit numbers assigned in processing order, so `000001` is the first successfully generated image of the run. The original discovery path is kept in `source_file` for every row.

## Payload semantics

- `N = payload_length(rate, W * H * 3)`, computed as `Decimal(str(rate)) * capacity` with `ROUND_HALF_UP`.
- One run-level `numpy.random.default_rng(run_seed)` generator is created when generation starts. It supplies the range-mode payload rates, the payload bits and the embedding locations in processing order.
- The embedding operation is `(old & 254) | bit`, so only the lowest bit of a channel can change and no higher bit is ever touched.
- Every RGB channel is eligible and each location is selected at most once.
- In fixed mode every successful image uses the configured rate. In range mode each successful image draws `rng.uniform(payload_min, payload_max)`, and the value used is written to `payload_rate` in `metadata.csv`.

The same input set, processing order, run seed and configuration reproduce the same random sequence. Per-image reproducibility that is independent of file order is not provided.

## Mask and selected-location meaning

The two arrays answer different questions and are intentionally different.

- The mask is the exact difference between the saved clean and stego images. It has one Boolean value per channel and is true only where the value changed. It is the ground truth for segmentation and localisation.
- The selected locations are the positions offered to the embedder. A selected position whose stored bit already matched the payload bit does not change, so `mask.sum()` is usually smaller than `len(selected)` and every true mask position is contained in the selected locations.

## Metadata schema

```csv
source_id,source_file,clean_file,stego_file,mask_file,selected_locations_file,payload_rate
000001,img1.png,clean/000001.png,stego/000001.png,masks/000001.npy,selected_locations/000001.npy,0.4
```

| Column | Meaning |
| --- | --- |
| `source_id` | Sequential run identifier, also the artifact file name |
| `source_file` | Path of the original file relative to the input folder |
| `clean_file` | Run-relative path of the clean PNG |
| `stego_file` | Run-relative path of the stego PNG |
| `mask_file` | Run-relative path of the modification mask |
| `selected_locations_file` | Run-relative path of the selected locations |
| `payload_rate` | Rate actually used for this image |

Values that are constant for the whole run are stored once in `run_config.yaml` or `run_summary.json` instead of in every row.

## Input and failure behaviour

- Original size is preserved. There is no resizing, cropping or format conversion.
- RGBA, grayscale, palette, 16-bit, animated, transparency-bearing, corrupt and non-PNG files are rejected rather than converted.
- The pixel safety limit is **16 million pixels per image**. Processing is serial and memory use grows with image size and payload rate.
- An invalid or failed source is logged with its reason, counted as skipped and does not stop the run. If writing one sample fails part way through, its artifacts are deleted and it gets no metadata row.
- An existing non-empty output folder is never overwritten, and the input and output folders must not overlap.
- `run_summary.json` reports `completed`, `cancelled` or `failed`, together with the discovered, generated and skipped counts. A run that generates no sample is reported as `failed`.

## Tests

```powershell
python -m unittest discover -s tests -v
```

GUI tests need a normal desktop session. `tests/test_core.py` and `tests/test_batch.py` work without one. `Validation-Report.md` records the earlier 2.0.0 implementation and is kept as history.

Implementation, tests and documentation were produced with AI assistance. Team review is required before academic submission or experimental use.
