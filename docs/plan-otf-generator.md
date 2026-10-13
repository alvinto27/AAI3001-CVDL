# Plan: on-the-fly LSB generator

Status: design only. Implementation has not started.

The user approved the NEF master set on 2026-10-13. A **NEF** is a Nikon
RAW file. It keeps the sensor data before colour processing. Use the 1,000
selected RAISE NEFs as the only uploaded master set.

**On the fly (OTF)** means that a loader makes stego images during training.
A **cover** is an image before embedding. A **stego** image contains an
embedded payload. **Embedding** stores payload bits in selected image values.
A **tile** is a square crop. The default is 512 x 512. **Bits per pixel
(bpp)** means payload bits divided by the number of pixels in the cover.
RGB means red, green, and blue. **Least significant bits (LSBs)** are the
low-value bits of an integer image value.

Use the same image functions for OTF loading and for writing a pre-built
dataset. Do not keep a second embedding implementation in the loader.

| Term | Meaning |
| --- | --- |
| CPU | Central processing unit. It runs source decoding and the loader. |
| GPU | Graphics processing unit. It runs model training. |
| Virtual CPU, or vCPU | A processing thread allocated to a virtual machine. |
| `uint8` | Unsigned 8-bit integer array values, from 0 to 255. |
| `float32` | 32-bit floating-point values used by the current model. |
| MiB / GiB | Binary memory units: 1 MiB is 1,048,576 bytes; 1 GiB is 1,024 MiB. |

Kaggle execution is not a current priority. Two-GPU support is future work.
Do not implement a notebook multi-GPU strategy in the first loader work.

Related records:

- [Project journal and measured results](journal.md)
- [Current v3c baseline](results-v3.md)
- [Existing generator](../LSB-Dataset-Generator/README.md)
- [RAISE rebuild record](rebuild-raise512-run1.md)
- [Source citation and terms](../ImageSets/README.md#citations-and-terms)
- [Noise proposal, not active](plan-noisy-covers.md)

## 1. Fixed requirements

| Item | Required behaviour |
| --- | --- |
| Master | The 1,000 NEFs in `data/raise/selection.csv`. Keep the RAISE research and education restrictions. |
| Source | Decode with the exact `lsb_prepare.py` settings. Keep the full oriented 8-bit RGB photo. Do not resize or pad it. |
| Grid decision, user, 2026-10-13 | All new non-legacy runs use `overlap_last`. Cover every real photo pixel. Use no image padding and drop no edges. The last tile on each axis ends at the photo edge. |
| Legacy exception | Mode 1 legacy replay uses `topleft_crop` with the original 512 grid. Its clean tiles and cropped pixels must equal `raise512-run1`, byte for byte. |
| Grid record | Record `grid_policy`, tile size, source dimensions, and every tile start in provenance and the evaluation profile. |
| Bit conversion | If a source step produces 16 bits, take the high byte: `x >> 8`. Never divide by 257. |
| Mode 1 | Cut tiles, then embed. Bpp applies to each tile. Legacy replay keeps the original cover indices, variants, seeds, and settings rules. New runs use overlapped tiles. |
| Mode 2 | Embed the full oriented photo once, without a crop. Bpp applies to the full photo pixel count, not to each tile or the sum of overlapped tile areas. Then cut overlapped tiles. |
| Mode 2 tile label | Label 1 only when at least one RGB value in that tile changes. A changed overlap pixel marks every tile that contains it. Otherwise, label 0. All tiles from a clean photo have label 0. |
| Mode 2 tile-size identity | With the same source pixels, settings profile, seeds, variant, and epoch, a 256 run and a 512 run make the same full stego photo. Tile size affects only extraction and tile labels. |
| Identity | Keep each source-photo ID and the existing source-photo split. Never split twins, variants, or tiles from one photo across sets. |
| Noise | No noise now. Keep a disabled RAW-domain source step for later work. |
| Output | Both modes can load during training or write a dataset from the same transformation functions. |

A **variant** is one embedding made from a cover. A **seed** is the integer
input that starts a repeatable random-number stream. **Replay** means to
make the same output again from its recorded settings and seeds.

Mode 1 compatibility includes the existing v6 label convention: a generated
stego file has label 1. Do not silently apply the new mode 2 label rule to an
old mode 1 dataset. Record an actual-change flag in both modes. A change to
mode 1 labels needs a separate decision and a new dataset version.

## 2. Source and data flow

### Source settings

| rawpy setting | Value |
| --- | --- |
| White balance | `use_camera_wb=True` |
| Automatic brightness | `no_auto_bright=True` |
| Output colour | `rawpy.ColorSpace.sRGB` |
| Output depth | `output_bps=8` |
| Demosaic method | `rawpy.DemosaicAlgorithm.AHD` |
| Orientation | rawpy default, as in `lsb_prepare.py` |

**Demosaic** converts the sensor colour samples to RGB pixels. The future
RAW-noise step belongs before this operation. It must not modify a NEF file.
The step is absent from the current numerical path.

```text
NEF master + frozen photo record
              |
       open RAW sensor data
              |
       [future RAW-noise step: OFF]
              |
       rawpy, exact source settings
              |
       full oriented 8-bit RGB photo
              |
       common full cover -----------------------------+
              |                                       |
         Mode 1                                  Mode 2
       choose grid                           embed full photo once
       legacy: crop                          bpp per full photo
       new: overlap                                   |
       cut tiles                             cut overlapped tiles
              |                              256 or 512: same
       embed each tile                       embedded photo
       bpp per tile
              |                                       |
       clean + stego tiles                   clean + stego tiles
              |                              actual-change labels
              +-------------------+-------------------+
                                  |
                         common tile records
                         /                 \
                  dataset writer      training adapter
                  PNG + records       augmentation + batches
```

A **cache** is a local copy of decoded cover pixels. It is not another master.
A full-photo cache read can replace source decoding. Its pixels must equal
the direct source path. Only mode 1 legacy replay crops to the old grid.
New runs keep every photo pixel and use `overlap_last`. Do not cache stego
pixels for OTF training. A new grid defines a new evaluation profile; it is
not legacy replay.

### Photo and tile records

Freeze two small tables before implementation. Keep them in `data/raise/`.
Export them from the existing selection, preparation records, and labels.
Do not derive a new cover index from the order of a training subset.

| Proposed table | Required fields |
| --- | --- |
| `photo_manifest.csv` | Stable string `photo_id`, stable integer `photo_index`, relative NEF path, camera, full oriented source height and width, legacy crop and grid, split. Derive new grid starts from full dimensions and the recorded policy. |
| `tile_manifest.csv` | Legacy-only original `cover_id` and `cover_index`, photo ID, original tile row and column, original tile file name, original clean and stego relative paths. |
| New-run tile records | Photo ID, `grid_policy`, tile size, grid row and column, actual `pixel_y` and `pixel_x` starts, tile dimensions, variant, and split. Do not assign old cover IDs to new edge tiles. |

Use the source-file stem as the photo ID. Assign and freeze the integer photo
index once, from the full sorted source list. Never assign it from worker
order. Keep the original, possibly non-continuous, tile cover indices.

`data/raise/split_seed0.csv` stores file-to-split rows. Join these rows to the
tile records to get each photo's split. Assert that each photo has one split.
Do not create a new random split. Include these small records with the code
and the single NEF upload.

### Grid policies and checked counts

The old `topleft_crop` grids are complete only within their top-left crop.
They contain 49,796 tiles. Keep these counts for mode 1 legacy replay:

| Camera | Photos | Legacy landscape grid, rows x columns | Legacy portrait grid | Legacy tiles |
| --- | ---: | --- | --- | ---: |
| D7000 | 714 | 6 x 9 | 9 x 6 | 38,556 |
| D90 | 278 | 5 x 8 | 8 x 5 | 11,120 |
| D40 | 8 | 3 x 5 | 5 x 3 | 120 |

Only this legacy route drops partial right and bottom edges. No new run does.

For `overlap_last`, let `L` be one full photo side and `T` the tile size.
Require `L >= T`. Let `n = ceil(L / T)`; `ceil` rounds up to an integer.
For each axis, use these starts in order:

```text
start(i) = i * T       for 0 <= i < n - 1
start(n - 1) = L - T  for the last tile

example: L = 1,100, T = 512
[0 .......... 511]
                [512 .......... 1,023]
                   [588 .............. 1,099]
                   ^ last tile overlaps its neighbour
```

A tile has exactly `T` real pixels on each axis. If `L` is a multiple of `T`,
there is no overlap. If `L == T`, use one tile starting at 0. Reject a photo
side smaller than `T`; do not pad or resize it. Starts are strictly increasing.
Only the final pair can overlap on each axis.

Checked grids from the audited oriented source dimensions follow. Dimensions
are height x width; grid counts are rows x columns. Both orientations have
the same tiles per photo.

| Camera / orientation | Full height x width | Photos | 512 grid | Tiles/photo | 256 grid | Tiles/photo |
| --- | --- | ---: | --- | ---: | --- | ---: |
| D7000 landscape | 3280 x 4948 | 570 | 7 x 10 | 70 | 13 x 20 | 260 |
| D7000 portrait | 4948 x 3280 | 144 | 10 x 7 | 70 | 20 x 13 | 260 |
| D90 landscape | 2868 x 4310 | 194 | 6 x 9 | 54 | 12 x 17 | 204 |
| D90 portrait | 4310 x 2868 | 84 | 9 x 6 | 54 | 17 x 12 | 204 |
| D40 landscape | 2014 x 3039 | 6 | 4 x 6 | 24 | 8 x 12 | 96 |
| D40 portrait | 3039 x 2014 | 2 | 6 x 4 | 24 | 12 x 8 | 96 |

| Camera | Photos | 512 tiles over the source set | 256 tiles over the source set |
| --- | ---: | ---: | ---: |
| D7000 | 714 | 49,980 | 185,640 |
| D90 | 278 | 15,012 | 56,712 |
| D40 | 8 | 192 | 768 |
| Total | 1,000 | 65,184 | 243,120 |

These totals count one tile grid per photo, not both clean and stego outputs.
For one variant and both outputs, the totals are 130,368 at 512 and 486,240 at
256. No image has fake edge pixels.

## 3. Shared modules and file changes

Names below are proposed. No files in this table are implemented by this plan.
Keep NumPy image work separate from PyTorch training work. NumPy is the array
library. PyTorch is the training library.

| File | Planned change or purpose |
| --- | --- |
| Existing `LSB-Dataset-Generator/lsb_prepare.py` | Keep one RAW decode settings definition. Expose full-photo decode. Keep the existing crop operation and `prepare.py` results for legacy replay. Reserve a disabled pre-demosaic source hook for future RAW noise. |
| Existing `LSB-Dataset-Generator/lsb_engine.py` | Put embedding in one internal core. Keep `embed()` and its three return values compatible. Add a training entry point that can omit the selection mask and public metadata object. It can return actual changed indices when requested. |
| New `LSB-Dataset-Generator/lsb_sampling.py` | Share settings draws, size limits, and seed construction. Move, do not rewrite, the current logic from `lsb_batch.py`. Keep compatibility wrappers for its existing private helpers. |
| New `LSB-Dataset-Generator/lsb_source.py` | Read frozen photo records. Select direct NEF or full 8-bit cache. Validate settings, full source dimensions, and cache provenance. |
| New `LSB-Dataset-Generator/lsb_pipeline.py` | Common transforms and explicit grid policies. Keep the old crop only for mode 1 legacy replay. Cut overlapped tiles for new runs. Calculate actual-change labels for every containing tile. No file writing or training augmentation. |
| Existing `LSB-Dataset-Generator/lsb_batch.py` | Use the shared sampling and embedding core for the existing tile-input runner. Add a NEF-backed writer that calls the common pipeline. Keep v6 generation and replay valid. |
| Existing `LSB-Dataset-Generator/generate.py` | Keep the current configuration route. Add explicit mode and source options for the new NEF-backed route. Do not infer a mode from a file extension. |
| New `LSB-Dataset-Generator/lsb_loader.py` | PyTorch photo work items, tile output iterator, batch assembly, and separate augmentation. Use the common pipeline. |
| New `scripts/build_otf_cache.py` | Build and validate a session cache. Report progress and disk use. |
| New `scripts/benchmark_otf.py` | Check source speed, embedding speed, labels, output tiles/s, and worker memory. |
| New `configs/otf-default.yaml` | Record current sampling defaults, separate training/evaluation profiles, `overlap_last` for new runs, tile size, and training-photo fraction. Add an explicit legacy replay profile with `topleft_crop`. |
| Existing `scripts/build_labels.py` | Keep legacy mode 1 labels. Add an explicit schema route for mode 2 labels from common tile records, not from the word `stego` in a path. |
| Existing notebook 04 | Later add a loader choice. Keep the v3c model, settings, and existing loader available for comparison. |
| Tests and documentation | Add source, pipeline, loader, cache, and replay tests. Update generator instructions when implementation changes them. |

**Provenance** is the record of how an output was made. Writer and loader
records must include mode, source ID, full dimensions, grid policy, actual
tile starts, split, variant, effective settings, seed scheme, and algorithm
version. Record the old crop only for mode 1 legacy replay. Do not feed these
records to the model.

The writer saves unaugmented images. The training adapter applies flips and
90-degree rotations after the common pipeline. Thus, writer and loader pixels
can be compared before augmentation. **Augmentation** makes training examples
with these geometric changes. It does not change an embedding seed.

For mode 2, keep an embedded photo's identity separate from its tile label.
A tile under an embedded-photo path can still have label 0. Record both the
photo embedding flag and the actual-change tile flag. The writer must consume
these flags from the pipeline. It must not recalculate labels by file name.

## 4. Speed work and exactness limits

The [journal](journal.md) contains the full measurements. These tests used
warm local files and CPU-only loaders on a Ryzen 7 7700. The PC has 16 logical
threads and 8 physical cores. They are not Kaggle or Colab measurements.
The photo benchmarks used the old cropped, non-overlapped route. Keep them
as references, not as speed results for the new full-photo overlap route.
More photo pixels are embedded and more tiles are emitted in the new route.

| Work item | Expected benefit | Risk | Required proof |
| --- | --- | --- | --- |
| Mask-free embedding | Avoid selection-mask allocation and metadata construction. Tile timings were about 0.668 ms for the reduced path and 0.736 ms for the full path. The full timing also included some setup. Treat the gain as approximate. | A second implementation could change random draws or byte-boundary matching. | Compare full and reduced paths for all bit modes, k values, methods, payloads, and placements. Compare pixels, changed counts, and subsequent random draws. |
| Labels from changed flat indices | Avoid the full-photo RGB comparison. The measured combined cut-and-label step was about 147 ms/photo. Only part of that time can be removed. Measure label time separately before claiming a gain. | Selected but unchanged values could be counted. Row boundaries or several changed channels in one pixel could be mapped incorrectly. | Compare labels and changed-pixel counts with a full `stego != clean` reference, including tile edges. |
| Faster random placement | Random, 0.5 bpp, one bit per value, and match took about 586 ms/D7000 photo. Contiguous took about 126 ms. Selection and sorting are candidates for improvement. Gain is Pending measurement. | A new subset draw or an unsorted subset changes both the random stream and payload assignment. | First keep the exact `rng.choice` draws. Test a faster integer sort with exactly the same sorted indices. Require identical final pixels and random-stream state. |
| 8-bit source cache | Avoid repeated RAW decode. The old cropped route gave 269.6 tiles/s from direct NEFs and 609.5 from `.npy` in separate tests. New overlap gains are Pending. | Stale source settings can change covers. A legacy cropped cache lacks real edge pixels. Disk space can be insufficient. | Compare each full cache entry with oriented decode and all extracted rectangles with direct source slices. Reject legacy cropped cache for new runs. |
| Small tile batches | Avoid sending a full float32 photo bag through the worker queue. | A tile stream can lose tiles, mix photo splits, or give duplicate work to distributed workers. | Count every expected tile once. Compare outputs across worker and device counts. Measure memory and full training speed. |

A **selection mask** marks pixels whose channels were selected. It is not a
changed-pixel mask. Keep its existing meaning for the writer and v6 tests.

The core already holds the selected flat channel indices and their values
before and after embedding. Keep only indices where `before != after` for
mode 2 labels. Use the full photo width `W` to get pixel coordinates. A simple
`pixel // T` tile assignment is wrong in an overlap region.

```text
changed channel index u
      -> pixel index p = u // 3
      -> pixel row y = p // W, pixel column x = p % W
      -> all row starts R[j] with R[j] <= y < R[j] + T
      -> all column starts S[k] with S[k] <= x < S[k] + T
      -> every pair (j, k) of containing row/column intervals
      -> tile index j * grid_columns + k
      -> mark EVERY containing tile label 1
```

At most two row intervals and two column intervals contain one pixel. A
changed pixel can mark two tiles at an edge overlap and four at a corner
overlap. The inequalities include each tile start and exclude its end.
For fast lookup, find the last axis start at or before the coordinate, then
check it and its preceding start if that predecessor exists. Keep only
intervals that contain the coordinate. Test this lookup against a search of
all tile rectangles.

Process mapping in bounded blocks. Do not make a full RGB difference array.
Remove duplicate pixel indices when changed-pixel counts are requested.
Several changed channels in one pixel give one changed pixel in each
containing tile. The same real pixel can count once in each of several tiles;
that is intentional overlap, not duplicate channel counting.

Mode 1 new runs embed each overlapped tile independently. The clean overlap
pixels agree, but separately embedded copies can differ. Do not claim that
these stego tiles reconstruct one consistent full stego photo. Mode 2 gives
one consistent embedded photo shared by all its overlapping tiles.

Do not remove the sort from random placement in the compatible mode. Do not
replace `rng.choice` and call the result v6-compatible. A different random
selection algorithm is a possible later mode 2 version only. It needs its own
algorithm ID, determinism tests, distribution checks, and user approval.

## 5. Run configuration for short tests

A run can change its sampling settings without a dataset rebuild. The master
and decoded full-photo cache do not depend on embedding settings. Keep
training settings and evaluation settings in separate configuration blocks.

| Run field | Required support | Default and legacy rule |
| --- | --- | --- |
| `k` | Fixed integer, choice list, or weighted choices. | Uniform ordered list `[1, 2, 3, 4, 5, 6, 7, 8]`. |
| `bit_mode` | Fixed mode, choice list, or weighted choices. | Uniform ordered list `[lowest, plane]`. |
| `method` | Fixed method, choice list, or weighted choices. | Uniform ordered list `[replace, match]`. |
| `placement` | Fixed placement, choice list, or weighted choices. | Uniform ordered list `[contiguous, stride, random]`. |
| `stride` | Fixed positive integer or inclusive integer range. | Uniform range `[2, 16]`, not two choices. |
| `bpp` | Positive fixed value or uniform range, with the selected capacity policy. | Uniform range `[0.05, 0.5]`, limited by v6 capacity rules. |
| `payload_kind` | Fixed kind, choice list, or weighted choices. | Uniform ordered list `[random_bytes, letters]`. |
| `start` | Fixed legal start or random legal start. | Random. |
| `tile_size` | Positive configured square size, initially 512 or 256; not greater than the smaller photo side. | 512. Legacy replay requires 512. |
| `grid_policy` | Explicit policy stored with output and evaluation records. | `overlap_last` for all new runs in both modes. `topleft_crop` only for mode 1 legacy replay. |
| `train_photo_fraction` | Fraction greater than 0 and at most 1. | 1: all train photos. |
| `train_subset_seed` | Separate fixed seed for selecting train photos. | 0; does not enter embedding streams. |
| Generation profile | Mode, seed scheme, generator seed, and variants. | New-run profile uses `overlap_last`. Separate mode 1 legacy profile uses original cover indices, generator seed 1, variant 1, and `topleft_crop`. |

The sampling defaults use the original settings draw functions and list
order. Exact old output also requires the explicit legacy profile. Do not
implement uniform choices with a new weighted-choice function.
It can consume different random values. Weighted choices are a new sampler
version. Validate weights and record that version. Fixed settings also need
a recorded configuration because they consume fewer random draws.

Select a training fraction by photo ID, with its separate subset seed. Keep
all chosen photos in the original train split. Record the selected IDs.
Prefer nested subsets: a smaller fraction uses the start of the same seeded
photo order. Do not reduce validation or test when the training fraction
changes. A deliberate small evaluation profile must have a separate ID.

For every new run, tile size changes only the `overlap_last` extraction grid.
Both 256 and 512 grids cover the same full photo, without padding or dropped
edges. In mode 1, bpp applies to each extracted tile. In mode 2, bpp and
payload size use the full photo area. Do not sum overlapping tile areas for
payload size. Record the grid and effective payload byte count.

Mode 2 settings and embedding streams must not include tile size or tile
order. With an equal photo-generation profile, seeds, variant, and epoch,
the 256 and 512 configurations make the same stego photo. They then extract
different tile samples. This settles the former question about which source
crop to use for 256 runs: use the full photo for all new runs.

Original cover indices apply only to mode 1 `topleft_crop` replay. For new
mode 1 tiles, identify a tile by `(photo_id, grid_policy, tile_size, pixel_y,
pixel_x)` and use the extended seed scheme. Even a 512 `overlap_last` run is
not `raise512-run1` replay. A full-photo cache supports both grids without a
cache or dataset rebuild.

An **evaluation profile** is a fixed configuration for validation and test.
Record its mode, sampling values and weights, sampler version, tile size,
grid policy, tile starts, full source dimensions and versions, fixed seed
scheme, seeds, variants, and full photo split. Compare runs only when these
profiles match. Run name, training fraction, training epoch, training shuffle,
worker count, and GPU count must
not change evaluation pixels. Write the resolved profile beside predictions.
The same profile must work for OTF evaluation and a pre-built evaluation set.

### Tile-level metric counting

Count each extracted real tile as one sample. Overlapping tiles are separate
samples even when they share pixels. Include clean and unchanged embedded
photo tiles with label 0. Do not deduplicate their overlap pixels or give them
fractional weights in standard tile metrics. Deduplicate only repeated sample
IDs caused by a loader or distributed sampler error. A sample ID includes
photo ID, mode, grid policy, tile size, actual start, variant, and clean or
embedded role. Equal pixels in different real output items do not merge.

Pixels near the last row or column can influence several tile samples. Tiles
from one photo are therefore related; they are not independent statistical
observations. Photos with more tiles also contribute more samples. Keep all
tiles of a photo in its original split. Record per-photo tile counts and the
grid policy in evaluation output. A legacy-crop score is not directly
comparable with an overlap score. A 256 tile score is not the same evaluation
as a 512 tile score, even when mode 2 uses the same full stego photo.
Photo-level MIL metrics remain separate future work.

## 6. Seeds, variants, and epochs

An **epoch** is one pass through the training samples. Validation and test
must use fixed settings and pixels across epochs. Never use test results to
select settings or a checkpoint.

NumPy `SeedSequence` builds independent random streams. Preserve the exact
four-integer entropy sequence for mode 1 legacy replay:

| Use | Seed sequence or policy |
| --- | --- |
| Mode 1 legacy settings | `[generator_seed, original_cover_index, variant, 0]` |
| Mode 1 legacy embedding | `[generator_seed, original_cover_index, variant, 1]` |
| Mode 1 v6 replay | Generator seed 1, variant 1, and `configs/raise512-run1.yaml`. Settings draw order, capacity limits, payload draw order, and NumPy algorithm must stay compatible. |
| Mode 2 fixed settings | `[generator_seed, stable_photo_index, variant, 0]` |
| Mode 2 fixed embedding | `[generator_seed, stable_photo_index, variant, 1]` |
| Extended mode 1 | `[generator_seed, 1, grid_policy_id, photo_index, tile_size, pixel_y, pixel_x, variant, epoch, stream_id]`. Use for every new overlapped tile, a new sampler version, or changing training variants. Freeze the numeric grid-policy IDs. |
| Extended mode 2 | `[generator_seed, 2, photo_index, variant, epoch, stream_id]`. Use for a new sampler version or changing training variants. Never include tile size, grid indices, or tile starts. |
| Fixed extended evaluation | Keep the recorded variant and epoch value fixed, normally epoch 0. Extended entropy is not mode 1 legacy replay. |
| Augmentation and ordering | Use separate streams. Include epoch and tile identity for repeatable augmentation. Do not consume settings or embedding streams. |

Record a `seed_scheme` name to distinguish legacy and extended seeds. Mode 2
always uses the full oriented photo as the carrier. Its new output therefore
differs from the earlier cropped-photo benchmark even with the same entropy.
Record the full-photo algorithm version. Tile size and extraction policy are
not embedding entropy. A numeric mode ID does not enter the legacy sequence.
Never append an epoch value to legacy entropy, even epoch 0. That would change
the output.

For fixed generation, an epoch changes only sample order and augmentation.
For changing variants, an epoch changes only the training generation streams.
Keep validation and test at their recorded fixed variant IDs. Pass the epoch
to persistent workers explicitly. Update it before work for the next epoch
starts. Do not depend on a worker's private mutable random state.

A **persistent worker** stays alive between epochs. A **distributed rank**
identifies one training process. Work assignment may use rank and worker ID.
Generation seeds must not use either value. Output pixels must not depend on
worker count, scheduling, device count, or restart order.

Record the epoch, sample order, variant policy, and consumed position in the
training checkpoint. For a shuffle queue, save its state or rebuild the same
ordered tile IDs and skip completed items. Count an item as consumed only
when its training step is complete, not when a worker prepares it. Include
partly consumed photos. A restart must not make a different validation set.
The pre-built writer accepts the same variant and epoch inputs as the loader.
It records them in the output. It does not make hidden fresh seeds.

## 7. Loader interface and batch shapes

The first model use is tile-level classification. **Multiple instance learning
(MIL)** is later work. In MIL, a model uses a bag of items to classify the bag.
Here, one bag is exactly one whole photo. Do not combine tiles from different
photos into one bag.

Separate a photo work item from a model batch:

| Layer | Mode 1 | Mode 2 |
| --- | --- | --- |
| CPU worker work item | One full decoded photo. Use the legacy crop only for replay; otherwise cut overlapped tiles. Each tile has its recorded identity and start. | One full decoded photo and one photo variant. Embed once before any tile extraction. Cut all clean and embedded overlapped tiles. |
| Tile output item | One tile image, label, and tile record with its actual start and grid policy. A clean/stego pair can share the same decode. | One tile image, actual-change label, and tile record. Include unmodified tiles from embedded photos. Overlap pixels retain their full-photo values. |
| Training batch | Fixed `B x 3 x T x T` float32 values in the original 0-255 range; labels `B`. `T` is the run's tile size. | Same shape and range. Photo ID and tile coordinates stay in side records. |
| Later bag output | Group tiles by photo. Preserve clean and generated photo variant IDs separately. | One clean-photo bag or one embedded-photo bag. Keep the photo embedding flag and each tile label. |

`B` is the configured image batch size. Float32 is a 32-bit floating-point
value. Keep the source and embedding arrays as `uint8`, which is an unsigned
8-bit integer. Convert tiles only at the training boundary.

**Prefetch** means to prepare work before the consumer requests it. Start
with `prefetch_factor=1`. Set `OMP_NUM_THREADS=1` before rawpy loads in every
worker process. Setting it after OpenMP starts is not sufficient. OpenMP is
the native CPU thread system used by LibRaw. Set PyTorch CPU threads to 1 in
workers. Use a spawn process start method for RAW decoding.

Assign whole photos to workers. Do not decode a NEF once for every tile.
Use a bounded tile shuffle queue to mix photos after generation. Preserve
independent augmentation for clean and stego tiles. A pair-aware loading
option can share decode work, but it must not silently replace the notebook's
independent image ordering.

Keep mode 1 labels balanced when all clean/stego outputs are retained. In the
old cropped mode 2 sample, 69.03% of embedded-photo tiles had label 1. With
both clean and embedded outputs, about 34.5% had label 1. These values are not
measurements of the new full-photo overlap route. The fraction changes with
grid, placement, k, and bpp. Measure it for each new run.

Do not relabel an unchanged tile to make a 50/50 batch. Options are natural
class frequency, loss weights, or training-only balanced sampling. Report
natural-frequency validation and test results in every case. Keep duplicate
clean pixels from unmodified embedded tiles visible in the records.

### Fixed shapes for later devices

A **tensor processing unit (TPU)** is a training accelerator. **XLA** is a
compiler used by some accelerator runtimes. It can require fixed input shapes.
No TPU or XLA implementation is part of this work.

- Keep tile size and image batch size fixed. Define the last-batch policy.
- Do not pad a source photo, tile image, or bag with fake image pixels.
  Keep every evaluation sample. CPU/GPU evaluation can use a short final
  batch. For a fixed-shape compiler, use a separate final-batch shape rather
  than inventing pixels.
- For later MIL, calculate maximum bag length `N_max` from the grid policy
  and full source records. For `overlap_last`, it is 70 at 512 and 260 at
  256. For legacy mode 1 at 512, it remains 54. `N_max` is a capacity limit,
  not a request to add fake tiles.
- Proposed MIL batching: group bags by actual tile count `N`. Each group
  has shape `B_bag x N x 3 x T x T`, tile labels, and one photo label per
  bag. Use a small `B_bag` to fit device memory. At 512 the new counts are
  70, 54, and 24; at 256 they are 260, 204, and 96. A compiler can prepare
  these fixed shapes separately. This batching choice is future work.
- Changing grid policy or tile size requires a new recorded shape limit.
  Never use a legacy 54-tile limit for a new 512 overlap bag.

| Policy | Tile size | D7000 bag | D90 bag | D40 bag | `N_max` |
| --- | ---: | ---: | ---: | ---: | ---: |
| Legacy mode 1 `topleft_crop` | 512 | 54 | 40 | 15 | 54 |
| New `overlap_last` | 512 | 70 | 54 | 24 | 70 |
| New `overlap_last` | 256 | 260 | 204 | 96 | 260 |

### Worker memory

Memory below is calculated from RGB arrays, not a new throughput benchmark.
Overlap repeats some real pixels in several tiles. Full-photo source and cache
size stay independent of tile size.

| Camera | Full RGB cover, `uint8`, MiB | 512 one float32 bag, MiB | 512 clean + embedded, MiB | 256 one float32 bag, MiB | 256 clean + embedded, MiB |
| --- | ---: | ---: | ---: | ---: | ---: |
| D7000 | 46.43 | 210 | 420 | 195 | 390 |
| D90 | 35.37 | 162 | 324 | 153 | 306 |
| D40 | 17.51 | 72 | 144 | 72 | 144 |

The old cropped D7000 route used 162 MiB per float32 bag and 324 MiB for
both outputs. The new 512 overlap route uses 420 MiB for both outputs,
about 29.6% more. These numbers exclude embedding work arrays and queues.

Old cropped-mode worker peaks were about 1.1 GiB with cached/stitch sources
and about 1.5 GiB in the NEF/16-bit master test. They do not measure the new
full-photo overlap route. New worker peaks and throughput are Pending.

**Resident memory** is memory currently held by a process. The measured
process peaks include shared pages. They are not unique private memory.
Prefer small tile batches in worker queues. Keep at most a small configured
number of photo arrays in each worker. Measure the parent queue, pinned
memory, and total process memory as well as each worker. Do not assume that
16 workers are safe on every 30 GB machine.

### Future Kaggle support: two GPUs and four shared CPUs

**Distributed data parallel (DDP)** runs one training process for each GPU.
Each process trains the same model on a different part of the data. The
processes combine their gradients. Kaggle has two T4 GPUs and four shared
CPUs for this target configuration.

The user deferred Kaggle execution and notebook DDP setup. The first loader
work is single-process training. Keep loader classes in importable modules
so a later script can use Lightning's `ddp` strategy with two GPU processes.
Choose and test any notebook strategy separately when that work is approved.
It is not part of the current work steps.

| Resource | Initial setting |
| --- | --- |
| Model processes | Future: one process per T4 GPU, two total. No notebook strategy selected now. |
| Loader workers | About two per GPU process: four total, not four per GPU. Reduce to one per GPU if CPU or memory contention appears. |
| Native source threads | `OMP_NUM_THREADS=1` in each worker, set before rawpy loads. |
| Training batch | Set `B` per GPU. The global image batch is `2 * B` unless gradient accumulation changes it. Record both sizes. |
| Source cache | Build once. Use a process barrier, which makes both ranks wait until the cache is ready. Both ranks then read it. Do not build it twice. |
| Queues | Prefetch 1 and bounded tile buffers per rank. Count memory for both ranks and all four workers. |

A **distributed sampler** assigns different source items to different ranks.
For a standard indexed photo dataset, let Lightning install its distributed
sampler. Set the sampler epoch before the training iterator starts. Do not
also shard the same photo list manually.

The planned tile stream can need a custom iterable loader. An **iterable
loader** yields items in sequence instead of serving an item by numeric
index. Lightning cannot automatically assign every custom iterable. In this
case, turn off automatic distributed sampling only for that explicit route.
Assign each photo to one rank, then to one worker in that rank. Test the
complete union of outputs for duplicates and missing IDs. Keep generation
seeds based on photo/tile identity, not rank or worker ID.

Ranks must take the same number of training steps. Photo grids have different
tile counts. Plan equal step counts with an explicit training-only
last-batch or sampling policy. Do not add fake image pixels to equalise steps.
Do not let one rank stop while the other waits for gradient communication. Evaluation must count each real sample once.
Do not count repeated sampler padding or masked entries as real images.
Gather validation scores and labels by stable sample ID before calculating
global P_E and AUC. P_E is the minimum average false-alarm and miss rate;
AUC is the area under the receiver-operating-characteristic curve. Averaging
rank-local P_E or AUC does not give the global metric.

Initial two-T4 speed estimate: assume one T4 supplies about 50-75% of the
measured single-L4 rate of 215 images/s. This gives about 108-161 per T4 and
215-323 images/s for two T4s before communication costs. This ratio is a
planning assumption, not a measured hardware ratio. Use about 320 output
tiles/s combined, or 160 per rank, as an initial loader target. Also report
340 as the PC comparison target. Measure the real combined GPU consumption
rate and both ranks' loader wait time before choosing the final target.

## 8. Session cache and platform policy

The master stays NEF. A session cache stores the exact full oriented source
before tile extraction or the legacy-only crop. Both generation modes use
the same full-photo entries. Tile size, grid policy, and embedding distribution
do not require a new full-photo cache.

| Cache option | Use | Estimated disk for all 1,000 photos |
| --- | --- | ---: |
| Full oriented `uint8` RGB `.npy`, one photo per file | Proposed speed-first format. Keep the edge pixels so a run can change tile size. NumPy can read it directly or use a read-only memory map. A memory map reads file pages when needed instead of copying the complete file at open. Benchmark both read methods. | About 42.11 GiB, calculated from the source dimensions. |
| Full oriented 8-bit RGB PNG, compression 1 | Disk-saving fallback. Uses Pillow. Slower decode. Still exactly preserves cover values. Keep the edge pixels. | Pending measurement; the cropped 512-grid cache was estimated at 19.39 GiB, or 20.82 GB. Full files will be larger. |
| Optional legacy cropped cache | Only for mode 1 `topleft_crop` replay with an explicit crop-grid key. Never use it for a new `overlap_last` run; missing edge pixels cannot be recovered. | About 36.5 GiB for `.npy` at the original 512 grid. |
| In-memory photo cache | Optional small bounded cache only. Do not load the entire corpus into 30 GB RAM. | Set a memory limit; no full-corpus default. |

The full `.npy` size uses the audited full source dimensions and three bytes
per pixel; it excludes small headers. The cropped PNG estimate uses the
earlier 19.86 MiB median times 1,000. Do not use that cropped estimate as a
full-source cache promise. A cache for only one split is smaller. Record the
actual file count and bytes when built.

On the PC, 4 processes decoded 42 NEFs to 8 bits in 10.25 seconds. A linear
extrapolation gives about 4.1 minutes for 1,000 photos. This excludes 8-bit PNG
encoding and cache validation. The complete cache build time is Pending
measurement on each platform. Do not state 4.1 minutes as a Kaggle guarantee.

| Target | Initial source policy | Initial worker policy and speed check |
| --- | --- | --- |
| PC, 16 logical threads | Direct NEF if measured throughput is sufficient; otherwise build a session cache. | Try 4, 8, and 16 workers. Require at least 340 output tiles/s for current PC demand. The old cropped direct-NEF test measured 269.6 at 4 workers; overlap speed is Pending. |
| Team NVIDIA L4 VM, 8 virtual CPUs | Test direct NEF and cache. | Try 4 and 8 workers. Require at least 215 tiles/s for the measured single-L4 demand. PC scaling is not a VM test. |
| Kaggle, two T4 GPUs, 4 shared CPUs | Prefer one-time cache, subject to disk space. Upload only NEFs and the small code/records. | Start two loader workers per GPU process, four total. Initial combined target about 320 tiles/s is an estimate. Also report 340 for comparison. Measure actual two-GPU demand. |
| Free Colab | Inspect the allocated CPU, disk, and accelerator first. Prefer cache on a weak CPU allocation. | Test 1, 2, and up to the available CPUs. Record demand and source throughput for that session. No fixed free-Colab capacity is assumed. |

CPU count alone does not select the source path. Benchmark it against the
consumer's measured image rate. The 4-worker PC master-source test gave
269.6 tiles/s for NEFs on the old cropped grid. Remeasure the new full-photo
overlap route. A CPU with fewer or slower cores can need the cache.

Use a caller-supplied local `cache_root`. Example session locations are
`/kaggle/working/otf-cache`, `/content/otf-cache`, and a local PC/VM scratch
directory. Check available disk before writing. Do not assume a platform's
current disk allowance. Do not write a cache into the read-only master path.
Mounted Google Drive is not the default per-batch read path; copy needed files
to local session storage when permitted.

Cache rules:

1. Record master-set version, photo ID, source dimensions, source file size,
   decode settings, rawpy version, and LibRaw version. Record a full-source
   cache as uncropped. For a legacy cropped cache, record `topleft_crop`
   and its original grid.
   Record noise as disabled. Use a canonical configuration record, not a
   directory name alone.
2. Write one entry to a temporary file, validate it, then rename it. Add it to
   the completed-entry record only after the rename.
3. Resume complete entries. Rebuild incomplete entries. Reject entries with
   different provenance or array shape, depth, or colour order.
4. Allow read-only shared cache entries when several training ranks use one
   machine. Only one builder writes a given entry.
5. Keep the cache until the user ends the session or requests removal. Remove
   temporary build files. Do not upload it as a second master.
6. A future RAW-noise configuration changes the source provenance. Never reuse
   today's noise-free cache for different RAW pixels. If RAW noise changes
   each epoch, a fixed decoded cache cannot replace that source operation.

## 9. Tests and speed acceptance

Test image pixels before augmentation. PNG container bytes and compression
can differ even when image pixels match. For v6 metadata, compare the existing
public fields as well as pixels. A new pipeline version must not misrepresent
itself as the old v6 writer.

| Test | Pass condition |
| --- | --- |
| Frozen records | 1,000 photo IDs and one split per photo. Legacy has 49,796 original tiles. New grids have 65,184 at 512 and 243,120 at 256. No missing or duplicate tile identities. Shared pixels are not duplicate tile IDs. |
| Source identity | Direct NEF full decode gives the same legacy top-left crop and stored clean tile pixels. First use the 42-photo sample; run all 1,000 before release. Repeat for each supported rawpy/LibRaw build. |
| Source/cache identity | Each full cache entry equals direct oriented decode. All sources give the same overlapped tiles at 512 and 256 and the same mode 2 embedded photo. Reject legacy cropped cache for new runs. |
| Overlap grid | Exact multiples, one-tile axes, partial edges, both orientations, and side smaller than T. Every real pixel is covered. Starts are ordered, legal, and end at the real edge. No padding or dropped pixels. |
| Mode 1 replay | With original config, seed, cover index, and variant, clean and embedded tile pixels equal `raise512-run1`. First check a representative sample, then all 49,796 pairs. Verify v6 public metadata and masks when requested. |
| Shared writer/loader | Both consume the same records and seeds. Before augmentation, compare pixels, labels, IDs, effective settings, and variant records. |
| Reduced embed path | Full and mask-free paths give identical stego pixels and identical random-stream state. Selection masks keep the existing selection meaning. |
| Mode 2 determinism | Repeat the same full-photo seed. Compare the embedded photo, cut tiles, labels, and records across worker counts, caches, and ranks. With equal generation inputs, full stego pixels must also match between 256 and 512. |
| Label rule | No actual change gives all labels 0, including selected but unchanged values. Test one changed channel, duplicate changed channels in one pixel, byte boundaries, and intervals. A changed overlap pixel labels both edge tiles or all four corner tiles. All clean-photo tiles remain 0. |
| Changed-index labels | For every tile rectangle, labels and distinct changed-pixel counts equal a full tile-comparison reference. Test overlap starts and exclusive ends, exact multiples, last row/column, last photo pixel, random placement, and high k. |
| Capacity | Test requested size above capacity, range clipping, and minimum-size failure. One-bit plane/stride 16 holds at most about 0.1875 bpp. Record the effective bpp. |
| Worker/epoch rules | Persistent workers use the requested epoch. Fixed validation/test pixels do not change. Restart does not lose or repeat planned samples. Distributed workers do not produce duplicate work. |
| Fixed shapes | Test 512 and 256 grids and their new bag limits. A short evaluation batch retains every real tile. Future bag batches have only real tiles and use their actual count; no fake image padding. |
| Short-run configuration | Sampling defaults plus the explicit legacy profile replay v6. All new profiles default to overlap. Test choices, bpp ranges, both tile sizes, and train-photo fractions. Fixed evaluation profiles include grid policy and starts. |
| Tile metrics | Count each extracted rectangle once, including each overlap tile. Shared pixels do not reduce sample weights. Profile records include policy and starts. Loader/sampler duplicate IDs do not count twice. |
| Future two-GPU route | When approved, both ranks receive different photos. Combined evaluation has each real tile once and global metrics. Step counts are equal. Total loader workers respect the four-CPU budget. Notebook strategy tests remain deferred. |
| Cache failures | Interrupted writes, missing files, changed configuration, wrong colour order, and insufficient disk fail safely or rebuild. |

The speed script must report:

- Machine, CPU affinity, worker count, library versions, source, mode, seed
  policy, grid policy, tile size, memory limits, batch size, and prefetch setting.
- Separate source, settings, embed, label, tile conversion, and total times.
  Report median and p90; p90 is the time below which 90% of measurements fall.
- At least 30 photos for photo timings, and at least 200 tiles for tile timings.
  Include all three cameras and both orientations when available.
- Fixed slow settings: random placement, 0.5 bpp, match, lowest k=1, and plane
  k=1/k=8. Also measure the original mixed settings.
- At least three throughput trials. Include source loading, changed-index
  labels, augmentation, float32 conversion, and queue transfer. Report both
  cold and warm local-file results. Count only real output tiles. Report
  full-photo pixels/s too; overlap changes tiles/photo and repeats pixels.
- Output tiles/s at 4, 8, and 16 workers where supported. A clean tile and an
  embedded tile each count as one output image. State class fractions too.
- Peak worker and total resident memory, queue use, and cache disk bytes.
- A separate test while real training runs. Loader-only speed does not prove
  that training has no CPU or memory contention.

Targets are at least 340 tiles/s on the PC and 215 on the single-L4 VM.
Measure the two-T4 consumer rate and use it as the Kaggle acceptance target.
For Colab, use the actual allocated accelerator's measured demand. Use the
cache if the direct NEF path cannot sustain that demand.

## 10. Dependencies

Checked `pyproject.toml` for this plan. It already lists the following packages:

| Package | Planned use |
| --- | --- |
| rawpy | Exact NEF decode. The benchmark used rawpy 0.27.1 and LibRaw 0.22.1. |
| NumPy | Arrays, seeds, embedding, changed-index labels, `.npy` cache. |
| Pillow | 8-bit PNG cache and pre-built dataset output. |
| PyYAML | Configuration files. |
| PyTorch and torchvision | Tile loader and training boundary. |
| Lightning | Existing training, checkpoint, and logger integration. |
| pandas, matplotlib, TensorBoard | Existing notebook tables, plots, and monitoring. |

No additional package is required for the proposed NEF/8-bit-cache design.
OpenCV and PyPNG were installed only in the local container for measurements.
Do not add them for this design. If a future 16-bit RGB PNG source is approved,
a teammate must add a tested 16-bit decoder, for example
`opencv-python-headless`, and update the lock through the team's CUDA workflow.
Pillow alone does not preserve a 16-bit RGB array.

Check package availability on Kaggle and Colab. Keep recorded decoder versions
for exactness tests. Do not replace their GPU PyTorch build only to install
rawpy. This plan does not change `pyproject.toml` or `uv.lock`. The local AMD
machine continues to use the ROCm container and must not run `uv`.

## 11. Open decisions for the user

1. Training variants: fixed embedding each epoch, or new settings/payloads
   each epoch? How many variants per tile or photo? Validation and test stay
   fixed in either case.
2. Mode 2 class balance: keep natural frequency, use loss weights, or use
   training-only balanced tile sampling? Keep all zero-change tile records.
3. Capacity policy for new mode 2 runs: keep v6 range clipping, or reject a
   requested bpp that cannot fit the selected stride? Mode 1 legacy replay
   must keep the old clipping policy.
4. Session cache: speed-first `.npy` or smaller 8-bit PNG when disk is limited?
   Set the disk limit and split scope. Direct NEF remains available.
5. Tile ordering: use a bounded cross-photo shuffle queue, or accept photo-local
   ordering first? Keep the independent-image baseline as a comparison.
6. Future random-selection change: allow a new mode 2 algorithm version only
   if the compatible sort improvement is insufficient? Legacy mode 1 stays
   exact. MIL model, pooling, and loss choices remain later work.
7. Short-run evaluation profiles: retain an explicit legacy profile, add a
   fixed hard-setting overlap profile, or use both? Tile size and grid policy
   are part of each profile. Both mode 2 sizes use the same full embedded
   photo when generation inputs match, but their tile metrics differ.
8. Future fixed-shape MIL batching: use the proposed groups by actual tile
   count, or another method that uses only real tile images? Do not add fake
   source pixels or fake bag tiles. Choose this when MIL/device work starts.

Settled by the overlap decision: new runs cover the full photo at both tile
sizes. There is no choice between a cropped 256 source and a cropped 512
source for mode 2. The legacy grid is an explicit replay exception, not the
new-run default.

## 12. Work steps

Each step has one check before the next step starts.

| Order | Small task | Check |
| ---: | --- | --- |
| 1 | Export and freeze photo/tile records from existing data. | Counts, grids, IDs, and source-photo split pass. |
| 2 | Expose exact full NEF decode and the legacy-only crop. Keep noise disabled. | Legacy source identity on the 42-photo sample; full edge pixels are retained for new runs. |
| 2a | Add explicit grid policies and the overlap start calculation. | Checked counts, full pixel coverage, no image padding, and legal edge starts at both tile sizes. |
| 3 | Move settings/seed helpers to the shared sampling module. Keep wrappers. | Existing v6 tests and sample replay are unchanged. |
| 4 | Refactor one embedding core and add the mask-free return path. | Full/reduced pixels and random-stream state match. |
| 5 | Add changed-index labels for all containing overlap tiles. Keep the selection-mask API unchanged. | One/two/four-tile label tests and full tile-comparison tests pass. |
| 6 | Add mode 1 legacy replay and new overlap extraction. | Original-index replay passes on all old pairs; new tile IDs and geometry pass. |
| 7 | Add full-photo mode 2 to the common pipeline. | Determinism, capacity, changed-index labels, and same-photo identity across 256/512 pass. |
| 8 | Add session `.npy` cache and resume records. | Direct/cache identity and interrupted-write tests pass. |
| 9 | Add the optional 8-bit PNG cache format. | Same identity tests; report build time and disk bytes. |
| 10 | Connect the pre-built writer and label builder to common records. | Writer and pipeline agree in both modes; v6 route stays valid. |
| 11 | Add photo work assignment and fixed tile batches to the loader. | Counts, shape, memory, and 1/4-worker output tests pass. |
| 12 | Add run-configurable settings, tile size, grid policy, and train-photo fractions. | Explicit legacy replay and fixed overlap evaluation profiles pass; no dataset rebuild. |
| 13 | Add selected epoch/variant and shuffle policies. | Persistent-worker, restart, and fixed-evaluation tests pass. |
| 14 | Add the speed script; test changed-index and sort improvements. | Exactness is unchanged; record measured gains, not estimates. |
| 15 | Test direct/cache source policy on PC and the team VM. Test Colab when a session is available. | Platform demand, disk, and memory checks pass. Kaggle execution is deferred. |
| 16 | Connect notebook 04 for single-process training, without changing v3c. | Short run, complete evaluation count, and resume pass. No notebook DDP setup. |
| 17 | Prepare the bag adapter interface, not the MIL model. | One photo per bag; actual tile counts, overlap limits, and no fake image padding. |
| Later | Add two-rank photo assignment and equal-step policy when approved. | No duplicate work; global evaluation metrics and fixed shapes pass. Choose the execution strategy at that time. |

Before release, run complete legacy source identity and mode 1 replay checks.
Also check full-photo overlap coverage and mode 2 cross-size identity. Record
all measured results in the journal. Update generator documentation with the
implementation, not before its behaviour is verified.
