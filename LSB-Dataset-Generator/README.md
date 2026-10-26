# LSB Dataset Generator v6

## What the tool does

The tool creates clean and stego image pairs for CNN steganalysis.
A **least significant bit (LSB)** is a low-value bit in an image byte.
The tool changes selected LSBs to store a payload.

```text
source photos ──prepare.py──> 8-bit RGB PNG tiles
                                  │
                                  └──generate.py──> clean + stego + mask files
                                                         │
                                                         └──--verify
```

Use the tools in this order:

1. Use `prepare.py` to decode photos and create fixed-size PNG tiles.
2. Use `generate.py` to create clean and stego images.
3. Use `generate.py --verify` to replay and check every stego image.

A tile is a square crop from one source photo.
The tool does not resize a tile.

The shared full-photo source API is in `lsb_source.py`. It reads frozen RAISE
and DIV2K records and cuts manifest-backed overlap or legacy tile views.
It does not yet embed or load training batches. See the
[source API and verification record](../docs/archive/source-verification.md).
The existing `prepare.py` workflow below is unchanged.

`lsb_settings.py` now owns shared settings draws and generation seed keys.
The existing batch route keeps its legacy draws and four-integer seeds.
See the [settings and seed record](../docs/active/settings-and-seeds.md) for the
separate versioned OTF route and its lower-to-fit size policy.

`lsb_engine.embed_fast()` uses the same numerical core as `embed()` but skips
the pixel mask and selection cache. It returns public metadata and optional
actual changed-unit indices. See the [embed-core verification](../docs/archive/embed-core-verification.md)
for its API and measured CPU timings.

`lsb_mode1.make_sample()` composes full-photo tile access, shared settings/seeds,
and mask-free embedding. `replay_legacy()` accepts a frozen legacy tile row.
Both return owned RGB8 pixels and provenance metadata. See
[Mode 1 verification](../docs/archive/mode1-verification.md) for the API and replay checks.
No loader or torch Dataset is included in Mode 1.

`lsb_cache.build_entry()` and `scripts/build_cache.py` build/resume the separate
full-photo RGB8 NPY cache. `load_cached()` strictly rejects stale or missing
entries without decoding in their place. See
[session cache verification](../docs/archive/cache-verification.md) for format, commands,
validation, read timings, and the approved VM cache location.

`lsb_torch.RunSettings`, `Corpus`, `TrainingImages`, and `FixedPairs` provide
the Mode 1 PyTorch boundary. Masters are decoded by default; the verified session
cache is optional. Training uses a bounded uint8 shuffle buffer and stable
sample keys; evaluation exposes fixed pairs and a metadata frame. See
[loader verification](../docs/archive/torch-loader-verification.md) for the notebook
contract, epoch controls, source selection, counts, and throughput checks.
Notebook 04 now connects these helpers in [step 8](../docs/archive/notebook-otf-verification.md).
The loader transfers uint8 groups across workers; the notebook converts to
float32 on the device. `tensor_dtype="float32"` and `group_images=0` retain
reference paths for checks.

## Setup

The standalone tool needs these packages:

| Package | Use |
| --- | --- |
| NumPy | Arrays and random number generation |
| Pillow | PNG and photo decoding |
| PyYAML | YAML run files |
| rawpy | RAW photo decoding |

The main team project uses `uv` and `pyproject.toml`.
The team `uv` environment lists NumPy, Pillow, PyYAML, and rawpy in `pyproject.toml`.

Use any other Python environment with:

```sh
python -m pip install -r requirements.txt
```

Run the tool with plain Python:

```sh
python prepare.py --help
```

## Prepare source photos

Run this command:

```sh
python prepare.py --input SOURCE_DIR --output TILE_DIR [OPTIONS]
```

| Option | Type | Default | Meaning |
| --- | --- | --- | --- |
| `--input` | folder | required | Source folder searched recursively |
| `--output` | folder | required | Tile folder |
| `--tile` | positive integer | `512` | Tile width and height in pixels |
| `--max-tiles-per-image` | positive integer | all tiles | Maximum random tile count per source |
| `--seed` | non-negative integer | `0` | Tile selection seed |
| `--workers` | positive integer | CPU count | Process worker count |
| `--png-compress-level` | integer `0` to `9` | `1` | PNG compression level |

The tool accepts these extensions:

- RAW: `.nef`, `.nrw`, `.cr2`, `.cr3`, `.arw`, `.dng`, `.orf`, `.raf`, `.rw2`, `.pef`, `.srw`, and other listed rawpy extensions.
- PNG, JPEG, TIFF, BMP, and WebP.

RAW files use these fixed rawpy settings:

| Setting | Value |
| --- | --- |
| White balance | Camera white balance |
| Auto brightness | Disabled |
| Output colour | sRGB |
| Output depth | 8 bits per sample |
| Demosaic | AHD |
| Orientation | rawpy default handling |

The settings and rawpy version are recorded in `run_config.yaml`.
Pillow applies EXIF orientation to non-RAW files.
The tool converts accepted non-RAW files to 8-bit RGB.

The tool refuses these inputs:

- Grayscale images.
- Palette images with transparency.
- RGBA images with any alpha value below 255.
- 16-bit images.
- Animated images.
- Corrupt files and unsupported extensions.

The tool does not invent colour for grayscale input.
The tool does not resize an image.
The tool crops a top-left, non-overlapping grid.
It drops partial right and bottom edges.

If `--max-tiles-per-image` is set, the tool selects tiles with:

```text
SeedSequence([seed, image_index])
```

The selection does not depend on worker count.

### Prepare output

```text
TILE_DIR/
├── source_r0_c0.png
├── source_r0_c1.png
├── manifest.csv
├── run_config.yaml
├── done.jsonl
└── run.log
```

`r` is the zero-based tile row.
`c` is the zero-based tile column.
Each tile is written to a temporary file in the output folder.
The tool renames the temporary file only after the PNG is complete.

`manifest.csv` has one row per tile:

| Column | Meaning |
| --- | --- |
| `tile_file` | Tile file name |
| `source_file` | Source path relative to the input folder |
| `source_format` | `raw`, `png`, `jpeg`, `tiff`, `bmp`, or `webp` |
| `camera_make` | Camera make from RAW IFD0 or EXIF |
| `camera_model` | Camera model from RAW IFD0 or EXIF |
| `source_width` | Width after orientation |
| `source_height` | Height after orientation |
| `tile_row` | Zero-based grid row |
| `tile_col` | Zero-based grid column |
| `pixel_x` | Tile left position |
| `pixel_y` | Tile top position |

The source path identifies the photo.
Use it to split train and test data by photo.
Do not split by tile.

The tool resumes when the output contains an identical `run_config.yaml`.
It skips a source when its recorded tiles are present.
It retries an unexpected error after removing that source's partial tiles.
It records bad input and no-complete-tile results as completed skips.
Ctrl+C keeps completed ledger entries and leaves no temporary tile files.

## Generate clean and stego images

Run this command:

```sh
python generate.py --config run.yaml [--workers N]
```

Use `--verify OUTPUT_DIR` instead of `--config` to verify an existing run:

```sh
python generate.py --verify OUTPUT_DIR [--workers N]
```

Verification uses all available CPU cores by default. Use `--workers N` to set the process count. It prints progress about every five percent. The report and failure details do not depend on the worker count.

### Configuration keys

The configuration uses YAML.
Unknown keys are rejected.
A scalar is fixed for every stego image.
A list is a set of choices.
The tool draws one list choice uniformly for each stego image.

| Key | Type and allowed values | Default |
| --- | --- | --- |
| `seed` | non-negative integer | required |
| `input` | folder | required |
| `output` | folder | required |
| `png_compress_level` | one integer from `0` to `9` | `1` |
| `variants_per_cover` | positive integer | `1` |
| `write_masks` | Boolean | `true` |
| `bit_mode` | `lowest`, `plane`, or a list of these | `lowest` |
| `k` | integer `1` to `8`, or a list of these | `1` |
| `method` | `replace`, `match`, or a list of these | `replace` |
| `placement` | `contiguous`, `stride`, `random`, or a list | `contiguous` |
| `stride` | positive integer or `[min, max]` inclusive range | `1` |
| `payload` | `random_bytes`, `letters`, or a list | `random_bytes` |
| `start` | `random` or a non-negative integer | `random` |
| `size` | exactly one size form | required |

`stride` is used only when the drawn placement is `stride`.
A stride range draws one integer uniformly from both endpoints.

`size` accepts these forms:

```yaml
size:
  bpp: [0.02, 0.08]
```

```yaml
size:
  payload_bytes: [10, 40]
```

A range draws uniformly from both endpoints.
A single `bpp` or `payload_bytes` value is also valid.
A scalar `size: 40` means `payload_bytes: 40`.
A scalar `size: 0.04` means `bpp: 0.04`.

**Bits per pixel (bpp)** is the requested payload byte count multiplied by eight and divided by image pixel count.
The engine converts bpp to a byte count by rounding and enforcing a one-byte minimum.

The runner draws settings with:

```text
SeedSequence([seed, cover_index, variant, 0])
```

The runner passes a separate embed stream to `lsb_engine.embed`:

```text
SeedSequence([seed, cover_index, variant, 1])
```

The embed seed is stored in the stego PNG JSON.
Worker count and processing order do not change the output.

If a drawn size exceeds capacity, the runner caps the upper range at the largest size that fits.
If the minimum size does not fit, the runner skips that variant and logs the reason.

### Complete example `run.yaml`

```yaml
seed: 123
input: example_sources
output: /tmp/aai3001-v6-example-run
png_compress_level: 1
variants_per_cover: 2
write_masks: true
bit_mode: [lowest, plane]
k: [1, 2, 3]
method: [replace, match]
placement: [contiguous, stride, random]
stride: [1, 3]
payload: [random_bytes, letters]
start: random
size:
  bpp: [0.02, 0.08]
```

Run the example from this folder with:

```sh
python generate.py --config example_config.yaml
python generate.py --verify /tmp/aai3001-v6-example-run
```

## Embedding rules

The **carrier** is the input `H × W × 3` uint8 RGB array.
A **carrier unit** is one channel byte.
Units use row-major order with R, G, B inside every pixel.
The flat order is `R0, G0, B0, R1, G1, B1, ...`.

A payload uses MSB-first bit order.
The first payload bit is the most significant bit of the first payload byte.

| Mode | Stored payload bits per selected unit |
| --- | --- |
| `lowest`, `k` | The `k` lowest bits of the byte |
| `plane`, `k` | Bit plane `k`, which is bit `k - 1` with bit 1 as the LSB plane |

**Replacement** directly replaces the target bits.

**Matching** keeps a value when its target bits already match.
For lowest-bit matching, the result is the nearest valid value with the target `k`-bit field.
For plane matching, a mismatch uses `v - 2^(k-1)` or `v + 2^(k-1)`.
The direction is random when both values are valid.
The direction is forced at byte boundaries.
For lowest-bit `k=8`, every target byte has one valid value, so matching equals replacement.

The final selected unit receives random padding bits when the payload does not fill its field.
Padding bits are not part of the extracted payload.

| Placement | Selected units |
| --- | --- |
| `contiguous` | `start + i` |
| `stride` | `start + i * stride` |
| `random` | A sorted subset without replacement |

For contiguous and stride placement, the legal non-wrapping start is:

```text
0 <= start <= total_units - 1 - (units_used - 1) * stride
```

The engine draws a legal start when `start: random`.
Random placement does not use `start`.

The `random_bytes` payload uses uniformly generated byte values.
The `letters` payload uses uniformly generated ASCII `A-Z` and `a-z` bytes.

The mask is an `H × W` one-bit PNG.
It is true for every pixel with at least one selected channel.
It is true even when matching or replacement leaves the selected value unchanged.
The mask is based on selection, not on changed pixels.

## Output layout and metadata

```text
output/
├── clean/000001.png
├── stego/000001_1.png
├── masks/000001_1.png
├── metadata.csv
├── run_config.yaml
├── done.jsonl
└── run.log
```

`masks/` is empty when `write_masks: false`.
Clean and stego PNGs both contain a `stego` PNG text chunk with JSON.

Clean JSON contains:

| Field | Meaning |
| --- | --- |
| `label` | `clean` |
| `cover_id` | Six-digit cover identifier |
| `source_file` | Relative source path |
| `size` | `[height, width, channels]` |

Stego JSON contains the clean fields plus:

- `cover_index`, `variant`, and `generator_version`.
- `seed` and `embed_seed` with the entropy needed for replay.
- Every public `EmbedResult` field: `bit_mode`, `k`, `method`, `placement`, `start`, `stride`, `payload_kind`, `payload_bytes`, `embedded_bits`, `units_used`, `units_changed`, and `bpp_actual`.
- `start_random`, which tells replay whether the start was drawn.
- `size_kind` and `size_value`, which record the requested size form.

`metadata.csv` has one row for every clean and stego PNG.
Its columns are flat scalar columns:

```text
label,cover_id,cover_index,variant,source_file,height,width,channels,generator_version,seed,embed_seed_seed,embed_seed_cover_index,embed_seed_variant,embed_seed_stream,bit_mode,k,method,placement,start,start_random,stride,payload_kind,payload_bytes,embedded_bits,units_used,units_changed,bpp_actual,size_kind,size_value,file,mask_file
```

The embed seed fields are separate columns.
The CSV does not store JSON inside a cell.

The runner writes one source result at a time to `done.jsonl`.
Only successful covers, bad input, and size skips enter the ledger.
Unexpected errors remove partial files and are retried on the next run.
The runner rebuilds `metadata.csv` from the ledger.

## Read one generated image

Use `replay()` with the clean array and the stego JSON.
The function does not need `run_config.yaml`.

```python
import json
from pathlib import Path

import numpy as np
from PIL import Image

from lsb_batch import replay

root = Path("/tmp/aai3001-v6-example-run")
with Image.open(root / "clean/000001.png") as image:
    clean = np.asarray(image).copy()
with Image.open(root / "stego/000001_1.png") as image:
    metadata = json.loads(image.text["stego"])

replayed, mask, result = replay(clean, metadata)
print(replayed.shape, mask.shape, result.payload_bytes)
```

## Verify a run

`--verify` reads every stego PNG JSON. It uses all available CPU cores by default; use `--workers N` to choose the process count. It prints progress about every five percent.
It reads the matching clean PNG.
It rebuilds the payload and selected units from `embed_seed`.
It checks every `EmbedResult` field.
It checks the replayed stego pixels byte for byte.
It extracts the payload from the replay and stored stego images.
It reports passed and failed counts.
A non-zero failure count returns exit code 2.

## Caveats for CNN use

- Split train and test data by source photo.
- Split by camera when camera metadata is available.
- Never split only by tile.
- Never split only by variant.
- The `letters` payload has a structured bit pattern.
- JPEG-origin covers keep JPEG compression traces after PNG preparation.
- Metadata and masks are not model inputs.
- Keep clean and stego files from one source photo in the same split.

## Tests

All tests live in the repository's `tests/` folder. Run them from the repository root:

```sh
podman exec -w /work aai3001 python -B scripts/test_cpu.py  # full project
podman exec -w /work aai3001 python -B scripts/test_cpu.py --tests tests.test_lsb_batch tests.test_lsb_engine tests.test_lsb_prepare
```

The retired Mode 2 modules are in `archive/`; their tests are in `tests/archive/` and are not run by default.

The focused generator tests use synthetic photos. They do not need downloaded RAW files.
Use the shared runner for every suite, including focused checks. It holds one container-wide lock; do not bypass it with raw unittest.
