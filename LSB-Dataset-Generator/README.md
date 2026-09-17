# LSB Dataset Generator

Standalone desktop GUI and command-line generator, version **2.0.0**. Produces clean/stego PNG pairs, exact Boolean change masks, selected locations and reproducibility records. It performs no dataset splitting, training or inference.

## Open on this computer

Double-click **Launch GUI.vbs** in this folder. The local launcher is configured for the installed Python runtime. No browser or server is needed. You can also run `gui.py` using a Python environment with the listed dependencies.

1. Browse to your input folder. Only static **8-bit RGB PNG** is accepted.
2. Choose an output parent folder. The GUI suggests a new run subfolder; you can edit its path. A manually entered output must be new or empty and separate from the input tree.
3. Select **Fixed rate** or **Random range**. For example, `0.4` selects 40% of the available RGB-channel LSB positions. About half those selected values may actually change; the exact count is recorded.
4. Keep or change the seed, optionally name the source dataset, and choose skip/stop behavior.
5. Click **Generate dataset**. The log reports rejected inputs and generation failures. **Cancel** finishes the current image before stopping; completed files and a partial summary remain available.
6. Use **Open output folder** for the files, or **Verify saved run** to reload artifacts and replay every generated sample.

At larger Windows display scales, use the right-hand page scrollbar to reach the full results area. The log has its own scrollbar.

**Save config** and **Load config** use YAML. Relative paths in a loaded YAML file resolve beside that file. A saved effective run configuration points to the original input/output paths; choose a new output before rerunning it.

## Set up on another Windows computer

Install Python 3.12 with Tcl/Tk support. In this folder, create a local environment and install the pinned packages:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe gui.py
```

The launcher prefers `.venv\Scripts\pythonw.exe`. The ignored `.runtime-path.txt`, `.vendor/` and `.tcl/` folders are local conveniences for this computer, not the portable source package. If moving the folder, use the virtual environment setup above. No administrator installation is required by the application itself. If startup fails, the launcher displays an error and writes `startup-error.log`.

## Command line

The CLI uses exactly the same core as the GUI and can run without a display:

```powershell
python generate.py --config example_config.yaml
python generate.py --input C:\data\sources --output C:\data\new-run --payload-rate 0.4 --run-seed 42
python generate.py --verify C:\data\new-run
python generate.py --version
```

The example configuration expects `example_sources/`; this folder contains three clearly synthetic test images. It is a smoke test, not an experimental dataset. Edit the configuration or use the GUI for your real sources. Range mode is configured through YAML or the GUI. `--config` supplies the full configuration; direct flags do not override it. Ctrl+C requests cancellation after the current image. Exit code 0 means completed (possibly with explicitly reported rejected files) or verified; failed/cancelled runs return 1.

## Output contract

| Output | Contents |
| --- | --- |
| `clean/<source_id>.png` | Canonical RGB source, saved through the same pipeline as stego |
| `stego/<source_id>.png` | LSB-replaced image, original dimensions |
| `masks/<source_id>.npy` | Boolean H × W × 3 array; true only where the channel actually changed |
| `selected_locations/<source_id>.npy` | Ordered int64 flat RGB-channel indices selected for embedding |
| `metadata.csv` | One row per successful clean/stego pair, labelled `stego`; includes all required source, rate, seed, count and artifact fields |
| `run_config.yaml` | Complete effective configuration |
| `run_summary.json` | Counts, status, runtime versions, run ID and UTC timestamps |
| `generation.log` | Complete progress, rejected source paths/reasons and failures |

Read NumPy files with `np.load(path, allow_pickle=False)`. A flat index `i` maps to row `i // (width * 3)`, column `(i // 3) % width`, and channel `i % 3`. Channel order is R/G/B. Masks are verification ground truth, not training images.

Every source is hashed from `b'lsb-rgb8-v2\0'`, big-endian uint32 width and height, and row-major decoded RGB bytes using SHA-256. The full digest is its stable `source_id`. Renames and PNG metadata changes therefore preserve identity. Duplicate canonical images are rejected/skipped and logged; original discovery paths remain in the log. `sample_id` adds a unique run ID so separate runs have distinct sample records.

## Exact reproducibility

- `N = Decimal(str(rate)) * (W * H * 3)`, rounded with `ROUND_HALF_UP`.
- A 128-bit embedding seed is the first 16 bytes of SHA-256 of ASCII `lsb-2.0.0|embedding|<run_seed>|<source_id>`, interpreted big-endian.
- One NumPy `Generator(PCG64(seed))` generates `N` uint8 bits with `integers(0, 2, size=N, dtype=uint8)`, then selects `N` unique indices with `choice(capacity, size=N, replace=False)`.
- The operation is `(old & 254) | bit`. No higher-order bit changes.
- Range mode derives a separate seed using purpose `rate`, then draws one uniform rate between the configured bounds. Equal bounds yield that fixed rate. The actual per-sample rate is saved in CSV.
- Preserve version 2.0.0 and the recorded NumPy version for replay. This rebuilt generator makes **no compatibility claim with the removed v1.x generator**.
- Both clean and stego images are newly saved as RGB PNG with compression level 6 and no copied ancillary metadata. Pixel arrays, not original PNG file bytes, define canonical identity and reproducibility.

The saved-run verifier needs only the run folder, this generator version and compatible dependencies. Original source files are not needed. It validates dimensions, RGB representation, counts, source identity, mask equality, unique selection, LSB-only differences, configuration-derived seeds/rates and exact replay. It also checks artifact paths stay inside the run folder and summary totals match the CSV. This verifies internal consistency, not cryptographic authenticity against deliberate replacement of an entire run.

## Input and failure policy

- Preserve original size. No resizing, cropping or format conversion.
- Reject RGBA, grayscale, palette, 16-bit RGB (checked from the PNG header before decoding), animation, transparency metadata, corrupt images and non-PNG files. Non-PNG files in the selected folder are counted as rejected, so use a dedicated image folder.
- Maximum **16 million pixels per image**. Processing is serial, but large images still need substantial memory, especially at high payload rates. Memory/resource failures are logged as failures; this tool is not a streaming image encoder.
- Zero-selected or zero-change samples are retained and explicitly flagged. No forced flips or silent relabelling. Review/exclude them as appropriate in the later training workflow.
- A generation/verification failure removes only that attempt's sample artifacts; it creates no successful CSV row. No existing run is overwritten, and no automatic resume/merge is attempted.
- `completed_with_rejections` means valid files completed under skip policy; review the reasons. `failed` means generation failed, stop policy was triggered, or no samples were generated. `cancelled` and `total_unprocessed` identify partial runs.

## Development and evidence

```powershell
python -m unittest discover -s tests -v
```

GUI tests need a normal desktop session. `tests/test_core.py` and `tests/test_batch.py` work without one. See [Implementation-Plan.md](Implementation-Plan.md) and [Validation-Report.md](Validation-Report.md) for execution checkpoints and evidence. The supplied specification is retained as [LSB-Dataset-Generator-Software-Specification.md](LSB-Dataset-Generator-Software-Specification.md).

Generated with OpenAI Codex assistance: implementation, tests and documentation. Team review is required before academic submission or experimental use.
