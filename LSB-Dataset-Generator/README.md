# LSB Dataset Generator

Standalone desktop GUI and command-line generator, version **5.0.0**. It produces clean/stego PNG pairs using random-start sequential 1-bit RGB embedding and one Boolean pixel-level selected-region mask per pair. It performs no dataset splitting, training, resizing or inference.

## Open on this computer

Double-click **Launch GUI.vbs** in this folder. No browser or server is needed. The launcher finds Python by itself, checking a `.venv` folder here, the path in `.runtime-path.txt` if that file exists, the Windows Python launcher, an installed Python, and finally `pythonw.exe` on `PATH`. If it finds nothing it lists the locations it searched. **No particular Python version is required.** Any Python 3.11 or newer with Tcl/Tk works, and an existing installation can be used as it is. You can also start `gui.py` yourself with any Python that has the dependencies.

1. Browse to your input folder. Only static **8-bit RGB PNG** is accepted.
2. Choose an output parent folder. The GUI suggests a new run subfolder and you can edit its path. The output folder must be new or empty and separate from the input tree.
3. Select **Fixed characters** or **Random range**.
   - Fixed mode uses the same positive whole-number character count for every image. `40` embeds 40 mixed-case ASCII letters, occupying 320 channel bits.
   - Range mode independently draws a uniform integer count per valid source, including both bounds. Defaults are `10` and `90`. Equal bounds behave as a fixed count.
   - Counts must be positive integers and minimum must not exceed maximum. A count too large for an image skips that image without truncation or resampling.
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
python generate.py --input C:\data\sources --output C:\data\new-run --character-count 40 --run-seed 42
python generate.py --input C:\data\sources --output C:\data\range-run --payload-mode range --min-character-count 10 --max-character-count 90
python generate.py --version
```

The example configuration expects `example_sources/`, which contains three clearly synthetic test images. It is a smoke test, not an experimental dataset. Edit the configuration or use the GUI for real sources. Choose a new output path for every run. Existing output is never overwritten. Range mode is available through CLI flags, YAML and the GUI. `--config` supplies the full configuration and the direct flags are used when no configuration file is given. Ctrl+C requests cancellation after the current image. Exit code 0 means completed, and 1 means cancelled or failed.

## Output structure

```text
output/
├── clean/000001.png
├── stego/000001.png
├── masks/000001.npy
├── metadata.csv
├── run_config.yaml
├── run_summary.json
└── generation.log
```

| Output | Contents |
| --- | --- |
| `clean/<source_id>.png` | Source image saved again through the same PNG path as stego, original dimensions |
| `stego/<source_id>.png` | Image after LSB replacement |
| `masks/<source_id>.npy` | Boolean `H × W` array, true where at least one channel belongs to the selected interval |
| `metadata.csv` | One row per generated pair |
| `run_config.yaml` | Effective configuration of the run |
| `run_summary.json` | Run status and counts |
| `generation.log` | Run start, configuration, skipped inputs, failures and final counts |

Load mask files with `np.load(path, allow_pickle=False)`. A flat channel index `i` maps to row `i // (width * 3)`, column `(i // 3) % width` and channel `i % 3`, in R/G/B order. Selection follows R, G, B within each pixel, left-to-right, then the next row. Selected-index files are not exported.

Source IDs are sequential six-digit numbers assigned in processing order, so `000001` is the first successfully generated image of the run. The original discovery path is kept in `source_file` for every row.

## Payload semantics

- Generate a separate message for each image. Every character is independently sampled uniformly from all 52 uppercase and lowercase English letters.
- Encode each letter as one ASCII byte and read its eight bits from left to right, most significant message bit first. No header, terminator, padding, compression or encryption is added.
- Capacity is `C = H * W * 3` bits. A count `N` needs `L = 8 * N` bits. Maximum characters are `floor(C / 8)`. For example, a 128 × 128 image holds 6,144 characters.
- If the chosen count exceeds capacity, skip and log that image without sample files or a metadata row. Do not truncate, clamp, resample or wrap.
- Choose `S` uniformly from `0` through `C - L`, inclusive, then embed exactly `[S, S + L)`. Partial boundary pixels are allowed.
- One run-level `numpy.random.default_rng(run_seed)` supplies range counts, starts and letter choices in processing order. Fixed mode uses `character_count`. Range mode uses inclusive `min_character_count` and `max_character_count`.
- The embedding operation is `(old & 254) | bit`. Preserve all upper seven image bits and every channel outside the interval. Matching LSBs need not flip.
- Messages are not retained in metadata, logs or separate files. The stego PNG necessarily contains the embedded text bits. There is no payload-recovery interface.
- Mixed-case letters have structured bit patterns. Results do not automatically generalise to arbitrary or encrypted payloads, and no improvement in detector performance is promised.
- Uniform valid starts do not give uniform channel coverage. Equal counts occupy different capacity percentages in differently sized images.

Identical inputs, processing order, configuration, generator version and seed reproduce images, masks and per-sample choices. File-order-independent reproduction and reproduction of old-version outputs are not provided.

## Region masks and compatibility

The mask marks every pixel touched by the selected interval, including partial boundary pixels and selected pixels whose LSBs already matched the payload. It is derived from selection, not image differences or a rectangular bounding box. It can remain true even when the entire stego image equals the clean image. `mask.sum()` counts region pixels, not selected channels or actual changes.

Historical version 3 outputs use scattered embedding and `H × W × 3` actual-change masks. Versions 4 and 5 export only `H × W` selected-region masks. Existing datasets are not converted, deleted or overwritten. Mask consumers require separately authorised compatibility work before using these outputs. Masks and generation metadata must not become detector inputs.

Configuration uses `character_count`, `min_character_count` and `max_character_count`. Zero, negative, fractional, Boolean and floating-point YAML counts such as `40.0` are rejected. Old percentage and fractional-rate fields fail ordinary unknown-field validation. Old percentage and fractional-rate CLI flags are not accepted.

`06_Scripts/build_manifest.py` accepts current `payload_percentage` metadata and divides it by 100 to preserve fractional `payload_rate` in its outputs. Historical `payload_rate` metadata remains supported. Reuse the permanent `source_split.csv` and supply the new run through `--generated-root`. Mask-loader and training compatibility work remains separate. Existing datasets retain their historical payload and mask meaning.

## Metadata schema

```csv
source_id,source_file,clean_file,stego_file,mask_file,character_count,encoded_byte_count,payload_percentage,start_channel,embedding_length
000001,img1.png,clean/000001.png,stego/000001.png,masks/000001.npy,8,8,0.13020833333333334,12,64
```

| Column | Meaning |
| --- | --- |
| `source_id` | Sequential run identifier, also the artifact file name |
| `source_file` | Path of the original file relative to the input folder |
| `clean_file` | Run-relative path of the clean PNG |
| `stego_file` | Run-relative path of the stego PNG |
| `mask_file` | Run-relative path of the selected-region mask |
| `character_count` | Selected number of letters |
| `encoded_byte_count` | ASCII bytes, equal to character count |
| `payload_percentage` | Calculated measurement only, `100 * embedding_length / C` |
| `start_channel` | Zero-based start in flat RGB-channel order |
| `embedding_length` | Selected channel count, exclusive end is start plus length |

`run_config.yaml` records the effective seed and character-count configuration. `run_summary.json` records `generator_version: 5.0.0`, `payload_encoding: ASCII`, `message_bit_order: most_significant_bit_first`, `text_policy: uniform_independent_A-Z_a-z` and `placement_policy: sequential_random_start_no_wrap`, alongside the seed, count configuration and run counts. GUI suggested run names include `text-v5`.

## Input and failure behaviour

- Original size is preserved. There is no resizing, cropping or format conversion.
- RGBA, grayscale, palette, 16-bit, animated, transparency-bearing, corrupt and non-PNG files are rejected rather than converted.
- The pixel safety limit is **16 million pixels per image**. Processing is serial and memory use grows with image size and character count.
- An invalid or failed source is logged with its reason, counted as skipped and does not stop the run. If writing one sample fails part way through, its artifacts are deleted and it gets no metadata row.
- An existing non-empty output folder is never overwritten, and the input and output folders must not overlap.
- `run_summary.json` reports `completed`, `cancelled` or `failed`, together with the discovered, generated and skipped counts. A run that generates no sample is reported as `failed`.

## Verification

The existing `tests/` suite describes the historical fractional-rate and actual-change-mask contract and has not been migrated. It is not version 5 validation. No new automated test infrastructure was added. Focused synthetic smoke checks and their limitations are recorded in the project's `04_Outputs/AAI3001-Text-LSB-Generator-Validation.md`. `Validation-Report.md` records the earlier 2.0.0 implementation and is kept as history.

Implementation, tests and documentation were produced with AI assistance. Team review is required before academic submission or experimental use.
