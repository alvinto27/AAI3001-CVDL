# Plan: on-the-fly LSB generator

Status: design only. Implementation has not started.

The user approved two cover sources on 2026-10-13: 1,000 selected RAISE
NEFs and 900 DIV2K HR PNGs, for both training and testing. A **NEF** is a
Nikon RAW file. It keeps sensor data before colour processing. **HR** means
high resolution. Keep NEFs as the sole RAISE master; do not add converted
RAISE masters. Each lossless 8-bit RGB DIV2K PNG is its own master.

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

User decisions, 2026-10-13: implement mode 1 OTF first for the tile CNN.
A **CNN** is a convolutional neural network used to classify tile images.
Implement mode 2 second for the future photo-aggregation model. Keep its full
design now. All team implementation tests and training run on the team L4 VM.
Kaggle execution and notebook multi-GPU setup are not current work.

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
| Masters | RAISE: 1,000 NEFs in `data/raise/selection.csv`. DIV2K: 800 train HR and 100 validation HR PNGs. Keep each source's research restrictions and citation. |
| Source | RAISE: exact `lsb_prepare.py` decode. DIV2K `png8`: Pillow, strict 8-bit RGB validation, no conversion. Keep full pixels; no resize or padding. |
| Source pool | Default: combined RAISE and DIV2K sets at natural tile proportions. Mix their training tiles in the same cross-photo shuffle queue. Retain `source` in every record. |
| Grid decision, user, 2026-10-13 | All new non-legacy runs use `overlap_last`. Cover every real photo pixel. Use no image padding and drop no edges. The last tile on each axis ends at the photo edge. |
| Legacy exception | Mode 1 legacy replay uses `topleft_crop` with the original 512 grid. Its clean tiles and cropped pixels must equal `raise512-run1`, byte for byte. |
| Grid record | Record `grid_policy`, tile size, source dimensions, and every tile start in provenance and the evaluation profile. |
| Bit conversion | For a NEF step producing 16 bits, use `x >> 8`, never divide by 257. `png8` rejects 16-bit files; it does not convert them. |
| Mode 1 | Cut tiles, then embed. Bpp applies to each tile. Legacy replay keeps the original cover indices, variants, seeds, and settings rules. New runs use overlapped tiles. |
| Mode 2 | Embed the full oriented photo once, without a crop. Bpp applies to the full photo pixel count, not to each tile or the sum of overlapped tile areas. Then cut overlapped tiles. |
| Mode 2 tile label | Label 1 only when at least one RGB value in that tile changes. A changed overlap pixel marks every tile that contains it. Otherwise, label 0. All tiles from a clean photo have label 0. |
| Mode 2 tile-size identity | With the same source pixels, settings profile, seeds, variant, and epoch, a 256 run and a 512 run make the same full stego photo. Tile size affects only extraction and tile labels. |
| Identity | Keep each source-photo ID and the existing source-photo split. Never split twins, variants, or tiles from one photo across sets. |
| Noise | No noise now. Disabled future RAW hook for RAISE only. DIV2K `png8`: noise is not applicable; no rawpy or RAW-noise hook. |
| Training variants | Support `fixed` and `per_epoch` as run settings. Validation and test stay fixed. |
| Class frequency | Keep natural frequency. Do not add class weights or balanced class sampling. The tile CNN uses mode 1 first; mode 2 supports future photo aggregation. |
| Capacity | Lower the payload to fit in both modes. Keep exact v6 clipping for legacy replay. Record requested size and effective bpp. |
| Cache | Full-photo `.npy` session cache for RAISE. Initially read DIV2K masters directly; test whether its optional `.npy` cache helps. Caches are not new masters. |
| Tile ordering | Use a bounded cross-photo shuffle queue. |
| Evaluation | Keep the full original embedding mix and a fixed evaluation profile. Make harder training examples by changing training settings only. |
| Execution | Team VM through `ssh gcp-aai3001`; no new PC, Kaggle, or Colab runs for this work. |
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

RAISE uses source type `nef` and the exact preparation settings below.

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

DIV2K uses source type `png8`. Read with Pillow and verify PNG, mode `RGB`,
and PNG header bit depth 8 with RGB colour type. Reject palette, grayscale,
alpha, and higher-depth files; do not use `.convert("RGB")` to hide a mismatch.
Keep stored pixel orientation, dimensions, and integer values. Do not resize,
apply colour profiles, or adjust gamma. It needs no rawpy call and no RAW
hook. Record noise as `not_applicable`, not as a disabled RAW operation.

```text
RAISE NEF + photo record                DIV2K PNG + photo record
          |                                       |
    open RAW sensor data                   Pillow: strict png8
          |                                no RAW/noise hook
    [future RAW noise: OFF]                       |
          |                                       |
    rawpy, exact settings                         |
          +-------------------+-------------------+
                              |
                 full 8-bit RGB cover, source recorded
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

Freeze a combined source manifest and preserve the original RAISE records.
Proposed new combined location: `data/otf/`. Existing records in `data/raise/`
remain unchanged. Export their identities from selection, preparation,
labels, and splits. Never derive a cover index from a training subset.

| Proposed table | Required fields |
| --- | --- |
| `data/otf/photo_manifest.csv` | `source` (`raise` / `div2k`), `source_type` (`nef` / `png8`), stable `photo_id`, unique integer `photo_index`, relative master path, full height/width, camera where known, noise applicability, split, and 512/256 overlap grid starts/counts. Legacy crop/grid exists only for RAISE. |
| Legacy tile manifest | Original `cover_id` and `cover_index`, RAISE photo ID, original row/column, tile file name, and clean/stego paths. Preserve the existing mapping. |
| New-run tile records | Source, photo ID/index, policy, tile size, grid row/column, actual `pixel_y`/`pixel_x`, dimensions, variant, and split. Do not assign old cover IDs to new edge tiles or DIV2K. |

RAISE photo IDs remain their NEF stems. Freeze RAISE indices 0..999 from the
full sorted RAISE list; do not re-sort it together with DIV2K. DIV2K photo IDs
are `div2k_0001`..`div2k_0900`. Reserve indices 1000..1899, using
`1000 + numeric_DIV2K_ID - 1`. These are generation identities, not worker,
subset, source-weight, or queue positions. Assert uniqueness across sources.

Join `data/raise/split_seed0.csv` to RAISE tile records and then photos.
Assert one split per photo. Preserve the existing RAISE split exactly.
DIV2K needs its own proposed image split below. Never put tiles or variants
of one source image in different sets. Include all frozen small records with
the code; master image files remain untracked.

### DIV2K split proposal and source conditions

**Proposal, not an approved split:** use official validation IDs 0801..0900
as the 100-photo test set. Split official train IDs 0001..0800 by image,
85% train / 15% validation: 680 / 120 photos. Sort IDs, use
`np.random.default_rng(0).permutation(800)`, assign the first 680 positions
to train, and the remaining 120 to validation. Freeze the resulting rows
once approved; do not redraw each epoch. The counts below use this proposal
and NumPy 2.5.3. No split file was created by this plan task.

Default train, validation, and test each combine the corresponding RAISE
and DIV2K photo sets. Training tiles enter one cross-photo shuffle queue;
there is no default source oversampling or source weighting. Validation/test
include every real tile from both sources once. Their order may be mixed
without changing generation seeds. Keep per-source reporting alongside the
combined result. Source mix weights or RAISE-only training are optional run
settings, not evaluation changes.

DIV2K is for **academic research only**; copyright remains with the original
image owners. Cite E. Agustsson and R. Timofte, "NTIRE 2017 Challenge on
Single Image Super-Resolution: Dataset and Study," CVPR Workshops, July 2017,
as recorded in the [source terms](../ImageSets/README.md#citations-and-terms).
Do not treat the download as permission to redistribute the images.

DIV2K was collected from the Internet. Its PNGs may retain an earlier JPEG
history, including 8 x 8 block traces in low bits. PNG is lossless storage,
not evidence of an uncompressed acquisition history. This is a cover-source
difference from RAISE. Compare clean-tile false-alarm rates by source at the
same validation-selected threshold. Use training with/without DIV2K as a
controlled comparison, not a reason to alter test pixels.

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

DIV2K counts below come from all 900 actual local PNG headers, checked read-only
in the ROCm container on 2026-10-13, not a generator or training run. All files
were 8-bit RGB; stored width ranges 1116..2040 and height 648..2040. The
shortest side is 648, so every source supports both tile sizes without padding.
Sizes vary; a single example cannot estimate the corpus. Recheck the VM
inventory against the frozen source records before implementation.

| DIV2K official folder | Photos | 512 tiles | 256 tiles |
| --- | ---: | ---: | ---: |
| `DIV2K_train_HR` | 800 | 9,732 | 37,936 |
| `DIV2K_valid_HR` | 100 | 1,228 | 4,808 |
| Total | 900 | 10,960 | 42,744 |

| DIV2K proposed split, seed 0 | Photos | 512 tiles | 256 tiles |
| --- | ---: | ---: | ---: |
| Train | 680 | 8,276 | 32,216 |
| Validation | 120 | 1,456 | 5,720 |
| Test | 100 | 1,228 | 4,808 |

These are single grids, before clean/stego duplication. With one variant,
DIV2K alone yields 21,920 output tiles at 512 and 85,488 at 256. The combined
1,900-photo source set has 76,144 / 285,864 single-grid tiles, or
152,288 / 571,728 clean-plus-stego outputs. RAISE legacy counts do not change.
DIV2K bags contain at most 16 / 64 real tiles at 512 / 256; the combined
maximum still comes from RAISE: 70 / 260.

## 3. Shared modules and file changes

Names below are proposed. No files in this table are implemented by this plan.
Keep NumPy image work separate from PyTorch training work. NumPy is the array
library. PyTorch is the training library.

| File | Planned change or purpose |
| --- | --- |
| Existing `LSB-Dataset-Generator/lsb_prepare.py` | Keep one RAW decode settings definition. Expose full-photo decode. Keep the existing crop operation and `prepare.py` results for legacy replay. Reserve a disabled pre-demosaic source hook for future RAW noise. |
| Existing `LSB-Dataset-Generator/lsb_engine.py` | Put embedding in one internal core. Keep `embed()` and its three return values compatible. Add a training entry point that can omit the selection mask and public metadata object. It can return actual changed indices when requested. |
| New `LSB-Dataset-Generator/lsb_sampling.py` | Share settings draws, size limits, and seed construction. Move, do not rewrite, the current logic from `lsb_batch.py`. Keep compatibility wrappers for its existing private helpers. |
| New `LSB-Dataset-Generator/lsb_source.py` | Dispatch explicit `nef` / `png8` records or their full cache entries. Validate RGB depth, source dimensions, and provenance. PNG masters never enter RAW/noise code. |
| New `LSB-Dataset-Generator/lsb_pipeline.py` | Common transforms and explicit grid policies. Keep the old crop only for mode 1 legacy replay. Cut overlapped tiles for new runs. Calculate actual-change labels for every containing tile. No file writing or training augmentation. |
| Existing `LSB-Dataset-Generator/lsb_batch.py` | Use the shared sampling and embedding core for the existing tile-input runner. Add a master-backed writer for both source types through the common pipeline. Keep v6 generation and replay valid. |
| Existing `LSB-Dataset-Generator/generate.py` | Keep the current configuration route. Add explicit mode and source options for the new master-backed route. Do not infer a mode from a file extension. |
| New `LSB-Dataset-Generator/lsb_loader.py` | PyTorch photo work items, tile output iterator, batch assembly, and separate augmentation. Use the common pipeline. |
| New `scripts/build_otf_cache.py` | Build and validate a session cache. Report progress and disk use. |
| New `scripts/benchmark_otf.py` | Check source speed, embedding speed, labels, output tiles/s, and worker memory. |
| New `configs/otf-default.yaml` | Record current sampling defaults, separate training/evaluation profiles, `overlap_last` for new runs, tile size, and training-photo fraction. Add an explicit legacy replay profile with `topleft_crop`. |
| Existing `scripts/build_labels.py` | Keep legacy mode 1 labels. Add an explicit schema route for mode 2 labels from common tile records, not from the word `stego` in a path. |
| Existing notebook 04 | Later add a loader choice. Keep the v3c model, settings, and existing loader available for comparison. |
| Tests and documentation | Add source, pipeline, loader, cache, and replay tests. Update generator instructions when implementation changes them. |

**Provenance** is the record of how an output was made. Writer and loader
records must include mode, `source`, source type, photo ID/index, noise
applicability, full dimensions, grid policy, actual
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
| Byte-identical random sort replacement | Old random, 0.5 bpp, one bit per value, and match took about 586 ms/D7000 photo; contiguous took about 126 ms. Improve ordering only; gain is Pending. Keep the same `rng.choice` subset. | A changed subset, RNG state, or index order changes stego pixels. A Boolean mark array can use more work on sparse selections. | Compare sorted indices, pixels, and RNG state exactly. A candidate marks the existing chosen indices in a Boolean array, then reads marked positions in increasing order. It must not draw another subset. |
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

Approved speed work must keep image pixels byte-identical to the reference.
This includes the mask-free path, changed-index labels, and an identical
sorted-index replacement. Do not remove ordered payload assignment. Do not
replace `rng.choice` with another random-selection algorithm. A different
selection algorithm is not planned for either mode. It needs new user approval
before design or implementation.

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
| `bpp` | Positive fixed value or uniform range. Lower to capacity when necessary and record effective bpp. | Uniform range `[0.05, 0.5]`, with v6 range clipping. |
| `capacity_policy` | Lower to fit in both modes. Not a reject-on-overflow option. | `lower_to_fit`; exact legacy replay uses unchanged v6 size draws. |
| `payload_kind` | Fixed kind, choice list, or weighted choices. | Uniform ordered list `[random_bytes, letters]`. |
| `start` | Fixed legal start or random legal start. | Random. |
| `tile_size` | Positive configured square size, initially 512 or 256; not greater than the smaller photo side. | 512. Legacy replay requires 512. |
| `grid_policy` | Explicit policy stored with output and evaluation records. | `overlap_last` for all new runs in both modes. `topleft_crop` only for mode 1 legacy replay. |
| `train_photo_fraction` | Fraction greater than 0 and at most 1. | 1: all train photos. |
| `train_subset_seed` | Separate fixed seed for selecting photos from the combined train pool. | 0; does not enter embedding streams. |
| Training sources | RAISE only, DIV2K only, or both; fixed evaluation does not change. | Both in one combined pool. |
| `source_mix_weights` | Optional training-only source selection weights, not class weights. Record values and selection unit if used. | None: no weighting/oversampling; every planned photo emits its real tiles. |
| DIV2K source read | Direct strict `png8`, or verified `.npy` if measured useful. | Direct PNG initially. |
| `train_variant_policy` | `fixed` or `per_epoch`; both are supported. | `fixed` preserves the current repeated-embedding behaviour. |
| Variant count | Positive run setting for variants per tile or photo. | 1. Legacy replay and the initial evaluation profile use variant 1. |
| Generation profile | Mode, seed scheme, generator seed, and variants. | New-run profile uses `overlap_last`. Separate mode 1 legacy profile uses original cover indices, generator seed 1, variant 1, and `topleft_crop`. |

The sampling defaults use the original settings draw functions and list
order. Exact old output also requires the explicit legacy profile. Do not
implement uniform choices with a new weighted-choice function.
It can consume different random values. Weighted choices are a new sampler
version. Validate weights and record that version. Fixed settings also need
a recorded configuration because they consume fewer random draws.

### Capacity: lower to fit

For both modes, keep the v6 capacity calculation and size-range clipping.
Limit the upper draw bound to the carrier capacity. For new runs, also lower
a fixed request or a range whose minimum is above capacity, rather than reject
it for overflow. Never add a capacity-rejection run option without a new user
decision. Record the requested bounds or fixed size, payload bytes, and
`bpp_actual` after byte rounding. If even one payload byte cannot fit, report
a clear error; do not make a zero-byte payload or invent carrier pixels.

Mode 1 legacy replay must keep the original v6 draw and minimum-size-failure
behaviour unchanged. The current original configuration fits its minimum.
The new overflow handling must not change any original setting, random draw,
or output. Test scalar requests, range clipping, and byte rounding separately.

Select a training fraction by photo ID, with its separate subset seed. Keep
all chosen photos in their frozen source train split. Record the selected IDs.
Prefer nested subsets: a smaller fraction uses the start of the same seeded
photo order. Do not reduce validation or test when the training fraction
changes. Keep the current evaluation profile at the full original mix; do
not add a hard or reduced evaluation set for these short training tests.

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
Record `source` for every photo/sample, the frozen source manifest, mode,
sampling values and weights, sampler version, tile size, grid policy, starts,
full dimensions and versions, fixed seed scheme, seeds, variants, and split. Compare runs only when these
profiles match. Run name, training fraction, training epoch, training shuffle,
worker count, and GPU count must
not change evaluation pixels. Write the resolved profile beside predictions.
The same profile must work for OTF evaluation and a pre-built evaluation set.

The approved evaluation mix is the original ordered uniform choices: k=1..8,
lowest/plane, replace/match, contiguous/stride/random, stride 2..16, bytes or
letters, and bpp 0.05..0.5 with clipping. Training may use harder settings or
fewer photos without changing evaluation.

| Profile | Status | Fixed configuration |
| --- | --- | --- |
| `overlap_full_mix` | Default and only required tile-CNN evaluation profile. | Mode 1, `overlap_last`, original full embedding mix, fixed extended seeds, recorded generator seed, variant 1, generation epoch 0, original RAISE plus frozen DIV2K validation/test splits, recorded `source`, no source weighting, and recorded tile size (initially 512). |
| `legacy_v6` | Generator byte-identity test and optional evaluation profile only. Never a standing run requirement. | Mode 1, `topleft_crop` 512, original full mix, original cover indices, generator seed 1, variant 1, original four-integer seeds. Pixels equal `raise512-run1`. |

Freeze the resolved `overlap_full_mix` seed and grid fields before the first
run. Changing training settings, training tile size, variant policy, or photo
fraction does not change this evaluation block. A future evaluation change
needs user approval and a separate profile ID. "Full legacy mix" means the
original embedding distribution, not a required legacy-grid score.

Do not require every run to report `legacy_v6`. If a v3c comparison is needed
later, run the v3c configuration through the new generator and score it on
the same `overlap_full_mix` profile. Do not compare new overlap scores directly
with old v1-v3c scores from the cropped test set.

### Tile-level metric counting

Count each extracted real tile as one sample. Overlapping tiles are separate
samples even when they share pixels. Mode 2 keeps clean and unchanged
embedded-photo tiles at label 0; mode 1 retains its generated-stego label
convention. Do not deduplicate overlap pixels or give them
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

### Source-specific and combined reporting

For each validation/test set, report RAISE, DIV2K, and combined metrics and
sample counts. Use `source` to group predictions, not to change embeddings or
feed a feature to the model. Combined metrics use all individual samples;
do not average the two source AUC or P_E values. Evaluation is unweighted
and retains every real tile. Training-only source choices/weights must not
silently remove either source's evaluation rows.

Record clean-tile false-alarm rates per source and combined at one threshold
selected on combined validation, then held fixed for test. A false alarm is
a clean tile scored as stego. Retain per-source predictions to inspect the
possible JPEG-history difference; classifier errors alone cannot prove it.
Compare training with/without DIV2K using the same evaluation profile.

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

Photo indices are globally unique across RAISE and DIV2K. RAISE indices and
legacy cover indices stay unchanged when DIV2K is added. Use its reserved
index block, not a local DIV2K number as a RAISE index. Assert no cross-source
seed identity collisions. Source weights/queue order do not enter entropy.

Record a `seed_scheme` name to distinguish legacy and extended seeds. Mode 2
always uses the full oriented photo as the carrier. Its new output therefore
differs from the earlier cropped-photo benchmark even with the same entropy.
Record the full-photo algorithm version. Tile size and extraction policy are
not embedding entropy. A numeric mode ID does not enter the legacy sequence.
Never append an epoch value to legacy entropy, even epoch 0. That would change
the output.

Support both training policies as a run setting:

| Policy | Training generation behaviour | Evaluation |
| --- | --- | --- |
| `fixed` | Reuse the recorded variants. In the extended scheme, hold the generation epoch at 0 even when the training epoch advances. Change only order and augmentation. | Fixed seeds and variants. |
| `per_epoch` | Use the training epoch in the extended streams to make new settings and payloads. Record the epoch for replay and resume. | Fixed seeds and variants; never use the training epoch. |

The exact legacy replay profile uses fixed variants and the original entropy.
Changing mode 1 variants each epoch uses the extended scheme, not legacy
replay. Both policies use the same pipeline and writer.
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

The tile CNN trains on mode 1 first. Mode 2 is second and supplies the future
photo-aggregation model. **Multiple instance learning (MIL)** uses a bag of
items to classify the bag. It is the likely future model, but the model and
fixed-size bag batching are not implementation choices yet. Here, one bag is
exactly one whole photo. Do not combine photos into one bag. Keep the mode 2
source, seeds, overlap geometry, labels, and record interface complete now.

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

Assign whole photos to workers. Do not decode either master once per tile.
Default: combine RAISE and DIV2K train photos and mix their tiles in the same
bounded cross-photo shuffle queue, without source balancing. Optional source
choices/weights affect training only. Validation/test combine both frozen
sets and retain every planned tile. Source and class records stay visible. Preserve
independent augmentation for clean and stego tiles. A pair-aware loading
option can share decode work, but it must not silently replace the notebook's
independent image ordering.

Use natural class frequency for both modes. Do not apply class loss weights
or balanced class sampling. With one clean output and one mode 1 variant,
the natural frequency is 50/50. Record actual output counts if variant count
changes; do not assume that every configuration is balanced.

In the old RAISE-only cropped mode 2 sample, 69.03% of embedded-photo tiles
had label 1.
Across clean and embedded outputs, about 34.5% had label 1. These are not
measurements of the new full-photo overlap route. Its frequency depends on
grid, placement, k, and bpp. Measure it without correcting the frequency.
Do not relabel unchanged tiles to make a 50/50 batch. Keep duplicate clean
pixels from unmodified embedded tiles visible in the records.

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
  bag. Use a small `B_bag` to fit device memory. RAISE counts are 70, 54,
  and 24 at 512; 260, 204, and 96 at 256. DIV2K counts vary by image,
  up to 16 / 64 at 512 / 256. Read each actual count from its record.
  A compiler can prepare fixed shapes per bucket. This remains a proposal.
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

### Active execution platform: team VM

All implementation tests, benchmarks, and training for this work run on the
team VM. The following values are user-provided, not new measurements:

| Item | Setting |
| --- | --- |
| Connection | `ssh gcp-aai3001` |
| Repository | `/srv/aai3001/AAI3001-CVDL` |
| GPU | One NVIDIA L4. |
| CPU / RAM | 8 vCPUs and 31 GB RAM. |
| Free disk | About 100 GB; check it again before cache creation. |
| Master sources | Selected NEFs under `ImageSets/RAISE`, plus `ImageSets/DIV2K_train_HR` and `ImageSets/DIV2K_valid_HR`, are already on the VM. Use recorded paths; no duplicate master downloads. |
| Normal reads | Full-photo `.npy` for RAISE; direct `png8` for DIV2K initially. Direct NEF remains a correctness/speed reference. |
| Initial loader workers | Test 4 and 8, with `OMP_NUM_THREADS=1` and PyTorch CPU threads 1 per worker. |
| Initial speed target | At least 215 output tiles/s for the measured 512-pixel single-L4 baseline. Measure actual consumer demand again for new settings or 256-pixel inputs. |

```sh
ssh gcp-aai3001
cd /srv/aai3001/AAI3001-CVDL
```

These commands identify the planned execution location. This document-only
change does not run training or change the remote checkout. Check the VM
branch, team use, available disk, and environment before implementation.

The user runs actual full training interactively in notebook 04 through
VS Code remote on the VM. Agents must not start actual full training. Agents
may run VM unit tests, benchmarks, and QUICK/short smoke runs, including the
real notebook or a copy/script. Use a separate test run name; never overwrite
user run folders. Remove test checkpoints, logs, and temporary notebooks
after checks. Retain a small verification report, not unwanted test artifacts.

PC measurements remain historical references. Do not schedule new PC,
Kaggle, or Colab execution in these work steps. Keep portability in the
interfaces. Kaggle two-GPU estimates above are future design notes only.

### Full-photo `.npy` session cache

For RAISE, use the selected `.npy` format, one full oriented RGB `uint8`
photo per file. Keep all real edge pixels. Both modes use the same entries before extraction or
the legacy-only crop. Tile size, grid policy, and embedding distribution do
not require a cache rebuild. Do not implement PNG-cache or cropped-cache formats
in this work. DIV2K PNGs are masters, not rejected RAISE PNG cache entries.

The full RAISE array set is about 42.11 GiB, or 45.22 GB, from audited source
dimensions and three bytes per pixel. This excludes small headers and build
records. It is disk storage, not a proposal to load all photos into RAM.
It fits the reported 100 GB free disk in principle. Check actual free space,
current team jobs, and temporary-file needs before building it.

NumPy can load `.npy` directly or use a read-only memory map. A memory map
reads file pages when needed instead of copying the whole file at open.
Benchmark both reads on the VM. Keep only a bounded number of photos in
worker memory.

On the old PC test, 4 processes decoded 42 NEFs to 8 bits in 10.25 seconds.
A linear extrapolation gave about 4.1 minutes for 1,000. It did not include
cache writes or validation. The full `.npy` build time on the VM is Pending.
Measure it; do not call the PC extrapolation a VM build result.

DIV2K PNG decode is expected to be cheap; `.npy` benefit is Pending VM
measurement. Start without a DIV2K cache. If direct decode limits throughput,
compare direct PNG with verified `.npy` reads using the same pixels/photos.
All 900 DIV2K arrays would use 7,541,773,920 bytes, about 7.02 GiB / 7.54 GB
plus headers, versus about 3.7 GB for compressed masters. Combined full-array
caches would use about 49.13 GiB / 52.76 GB plus headers. Spend that disk only
if measured speed justifies it; do not create another DIV2K master.

Use a caller-supplied VM-local `cache_root`, for example
`/srv/aai3001/otf-cache/raise8-full`. Check write permission and disk space.
Do not modify the NEF master path. Cache records must stay outside tracked
source files. The cache is not an uploaded second master.

Cache rules:

1. Record master-set version, source/type, photo ID/index, full dimensions,
   and source file size. RAISE: decode settings, rawpy/LibRaw versions, noise
   disabled. DIV2K: strict PNG depth/mode, Pillow version, stored orientation,
   noise not applicable. Validate the record, not just the directory name.
2. Write an entry to a temporary file, validate it, then rename it. Add it to
   the completed-entry record only after the rename.
3. Resume complete entries. Rebuild incomplete entries. Reject different
   provenance, shape, depth, or colour order. A legacy cropped entry cannot
   satisfy the new full-photo cache.
4. Only one builder writes each entry. Other team processes read completed
   entries. Do not replace a cache that an active team job is using.
5. Keep the cache until the user requests removal or confirms session cleanup.
   Clean temporary build files. Do not automatically remove a shared cache.
6. Future RAW noise changes source provenance. Do not reuse this noise-free
   cache for different RAW pixels. Epoch-varying RAW noise would need a
   different source/cache policy, which is not implemented now.

### Optional removal of pre-built dataset images

The user may remove pre-built image sets after the mode 1 OTF loader works.
This is not automatic and is not permission to delete files in this plan task.
All gates below must pass before an image-removal operation:

| Gate | Required evidence |
| --- | --- |
| Full mode 1 replay | All 49,796 original pairs are byte-identical before augmentation. Retain the test report, versions, and per-pair RGB pixel hashes so later checks do not depend on deleted PNGs. |
| Working training path | Notebook 04 trains from the mode 1 OTF loader on the VM. Validation/test use the fixed `overlap_full_mix` profile. Checkpoint resume and complete evaluation counts pass. |
| Small records retained | Preserve `labels.csv`, `metadata.csv`, all splits, manifests, selection records, generation/preparation settings, and completion records. Keep the original cover-index mapping and new frozen photo/tile manifests. |
| Independent rebuild | OTF/pre-built outputs reproduce from the NEF and DIV2K PNG masters plus retained records, without deleted dataset images. Keep both master sets. |
| Shared VM agreement | The user asks the team before the shared VM copy is deleted. Get agreement for the exact image paths. Do not delete files used by an active team job. |

Delete only the approved large image artifacts, not the dataset directory
with its records. Keep records in `data/raise/` and the proposed `data/otf/`.
Retain small records in `Datasets/raise512-run1/` and its `records/` tree. Historical
image paths in CSV rows remain replay identifiers; loaders must resolve them
through the NEF/photo mapping, not require those PNG paths to exist.
Do not remove model checkpoints, predictions, reports, NEFs, or DIV2K master
PNGs as part of dataset cleanup. Update the data-folder instructions after removal.

## 9. Tests and speed acceptance

Test image pixels before augmentation. PNG container bytes and compression
can differ even when image pixels match. For v6 metadata, compare the existing
public fields as well as pixels. A new pipeline version must not misrepresent
itself as the old v6 writer.

| Test | Pass condition |
| --- | --- |
| Frozen records | 1,000 RAISE plus 900 DIV2K photo IDs/unique indices; one split per photo. Legacy: 49,796 RAISE tiles. Overlap at 512 / 256: RAISE 65,184 / 243,120; DIV2K 10,960 / 42,744; combined 76,144 / 285,864. No missing/duplicate identities. |
| DIV2K source | Verify all 900 VM PNG headers/dimensions; reject non-RGB/non-8-bit. Preserve stored pixel values. No rawpy or RAW hook. |
| Combined pools | Confirm default natural-proportion unions/shared shuffle. Training source changes preserve full evaluation and unique seed identities. |
| Source metrics | Per-source/combined counts match prediction union. Clean false-alarm rates use the same validation-selected threshold. No source-weighted evaluation. |
| Source identity | Direct NEF full decode gives the same legacy top-left crop and stored clean tile pixels. First use the 42-photo sample; run all 1,000 before release. Repeat for each supported rawpy/LibRaw build. |
| Source/cache identity | Each full cache entry equals direct oriented decode. All sources give the same overlapped tiles at 512 and 256 and the same mode 2 embedded photo. Reject legacy cropped cache for new runs. |
| Overlap grid | Exact multiples, one-tile axes, partial edges, both orientations, and side smaller than T. Every real pixel is covered. Starts are ordered, legal, and end at the real edge. No padding or dropped pixels. |
| Mode 1 replay | With original config, seed, cover index, and variant, clean and embedded tile pixels equal `raise512-run1`. First check a representative sample, then all 49,796 pairs. Verify v6 public metadata and masks when requested. |
| Shared writer/loader | Both consume the same records and seeds. Before augmentation, compare pixels, labels, IDs, effective settings, and variant records. |
| Reduced embed path | Full and mask-free paths give identical stego pixels and identical random-stream state. Selection masks keep the existing selection meaning. |
| Mode 2 determinism | Repeat the same full-photo seed. Compare the embedded photo, cut tiles, labels, and records across worker counts, caches, and ranks. With equal generation inputs, full stego pixels must also match between 256 and 512. |
| Label rule | No actual change gives all labels 0, including selected but unchanged values. Test one changed channel, duplicate changed channels in one pixel, byte boundaries, and intervals. A changed overlap pixel labels both edge tiles or all four corner tiles. All clean-photo tiles remain 0. |
| Changed-index labels | For every tile rectangle, labels and distinct changed-pixel counts equal a full tile-comparison reference. Test overlap starts and exclusive ends, exact multiples, last row/column, last photo pixel, random placement, and high k. |
| Capacity | Both modes lower oversized fixed/range requests to fit and record requested/effective bpp. Legacy range draws and minimum-failure behaviour stay exact. A carrier that cannot hold one byte fails clearly. One-bit plane/stride 16 holds at most about 0.1875 bpp. |
| Worker/epoch rules | Persistent workers use the requested epoch. Fixed validation/test pixels do not change. Restart does not lose or repeat planned samples. Distributed workers do not produce duplicate work. |
| Fixed shapes | Test 512 and 256 grids and their new bag limits. A short evaluation batch retains every real tile. Future bag batches have only real tiles and use their actual count; no fake image padding. |
| Short-run configuration | Explicit legacy replay passes; new runs use overlap. Test fixed/per-epoch variants, settings, ranges, sizes, and photo fractions. All training changes leave `overlap_full_mix` pixels fixed. Legacy evaluation is optional, not required. |
| Natural frequency | No class weights or balanced class sampler. Record actual labels/counts in each mode. A bounded cross-photo shuffle changes order, not class frequency. |
| Dataset-removal gates | No removal before full replay, notebook 04 OTF training, record retention, and user/team agreement on the shared VM. Future loader tests work without stored PNGs. |
| Tile metrics | Count each extracted rectangle once, including each overlap tile. Shared pixels do not reduce sample weights. Profile records include policy and starts. Loader/sampler duplicate IDs do not count twice. |
| Future two-GPU route | When approved, both ranks receive different photos. Combined evaluation has each real tile once and global metrics. Step counts are equal. Total loader workers respect the four-CPU budget. Notebook strategy tests remain deferred. |
| Cache failures | Interrupted writes, missing files, changed configuration, wrong colour order, and insufficient disk fail safely or rebuild. |

The speed script must report:

- Machine, CPU affinity, worker count, library versions, source, mode, seed
  policy, grid policy, tile size, memory limits, batch size, and prefetch setting.
- Separate source, settings, embed, label, tile conversion, and total times.
  Report median and p90; p90 is the time below which 90% of measurements fall.
- At least 30 photos for photo timings, and at least 200 tiles for tile timings.
  Include RAISE's three cameras/orientations and representative DIV2K sizes.
  Report per-source timing as well as the natural combined-pool result.
- Fixed slow settings: random placement, 0.5 bpp, match, lowest k=1, and plane
  k=1/k=8. Also measure the original mixed settings.
- At least three throughput trials. Include source loading, changed-index
  labels, augmentation, float32 conversion, and queue transfer. Report both
  cold and warm local-file results. Count only real output tiles. Report
  full-photo pixels/s too; overlap changes tiles/photo and repeats pixels.
- Output tiles/s at 4 and 8 workers on the team VM. A clean tile and an
  embedded tile each count as one output image. State class fractions too.
- Peak worker and total resident memory, queue use, and cache disk bytes.
- A separate check during the user's interactive training or an authorised
  QUICK/short smoke run. Loader-only speed does not prove lack of contention.
  Do not start actual full training for this benchmark.

The current execution target is at least 215 tiles/s on the team L4 VM for
the measured 512-pixel baseline. Measure real training demand for each new
input size and model setting. Use RAISE `.npy` and initially direct DIV2K PNG. Retain the
old 340 tiles/s PC figure only as a historical comparison. Kaggle and Colab
acceptance tests are not current work.

## 10. Dependencies

Checked `pyproject.toml` for this plan. It already lists the following packages:

| Package | Planned use |
| --- | --- |
| rawpy | Exact NEF decode. The benchmark used rawpy 0.27.1 and LibRaw 0.22.1. |
| NumPy | Arrays, seeds, embedding, changed-index labels, `.npy` cache. |
| Pillow | Strict DIV2K `png8` decode, pre-built PNG output, and legacy identity tests. Not the selected RAISE cache format. |
| PyYAML | Configuration files. |
| PyTorch and torchvision | Tile loader and training boundary. |
| Lightning | Existing training, checkpoint, and logger integration. |
| pandas, matplotlib, TensorBoard | Existing notebook tables, plots, and monitoring. |

No additional package is required for NEF, strict `png8`, and `.npy` cache.
OpenCV and PyPNG were installed only in the local container for measurements.
Do not add them for this design. If a future 16-bit RGB PNG source is approved,
a teammate must add a tested 16-bit decoder, for example
`opencv-python-headless`, and update the lock through the team's CUDA workflow.
Pillow alone does not preserve a 16-bit RGB array.

Check package availability and recorded decoder versions in the team VM's
CUDA environment before implementation. Do not replace its GPU PyTorch build
only to add rawpy. This plan does not change `pyproject.toml` or `uv.lock`.
The local AMD/ROCm container was used for historical measurements; it is not
the execution location for this team's new runs.

## 11. Settled decisions and future open items

The user settled these choices on 2026-10-13. They are not open questions.
The added DIV2K split remains a proposal, listed separately below.

| Choice | Decision |
| --- | --- |
| Training variants | Implement both fixed variants and new settings/payloads each epoch, selected per run. Validation/test stay fixed. |
| Training purpose and order | Mode 1 OTF first for the tile CNN. Mode 2 second for likely future MIL aggregation; its full pipeline design stays in this plan. |
| Class frequency | Natural frequency in both modes. No class weighting or balanced class sampling. |
| Capacity | Lower to fit, with v6 clipping and effective bpp records in both modes. Exact legacy replay stays unchanged. |
| Sources | DIV2K `png8` masters join training/testing. Default combined natural-proportion pools; optional training source choices/weights. Keep per-source records/metrics. |
| Cache | Full oriented RAISE `.npy`. DIV2K direct PNG first; optional `.npy` if measured useful. No PNG-cache implementation. |
| Pre-built images | They may be removed only after full 49,796-pair replay and notebook 04 OTF training work. Retain small records. The user asks the team before shared VM image removal. |
| Tile order | Bounded cross-photo shuffle queue. |
| Speed changes | Byte-identical changes only. Keep the chosen subset and sorted-index order. A different random-selection algorithm is not planned and needs new approval. |
| Evaluation | Only `overlap_full_mix` is required: mode 1 overlap, full original mix, fixed extended seeds. No standing legacy evaluation. `legacy_v6` is a byte-identity test and optional profile only. |
| Baseline comparison | If needed later, re-run v3c through the new generator and compare on the same overlap profile. |
| Grid | Full coverage with `overlap_last` for new runs at both sizes. Only legacy mode 1 replay uses `topleft_crop`. Mode 2 embeds the same full photo at 256 and 512 for equal generation inputs. |
| Execution | All team implementation tests and training run on the team VM. Other platform execution is deferred. |

One new source choice needs approval before manifest freeze: the section 2
DIV2K split proposal (680 train / 120 validation / 100 test, seed 0).
DIV2K cache value is Pending measurement, not a master-format choice.

The remaining future model/device choices are:

1. Fixed-size MIL batches: use the proposed groups by actual tile count, or
   another method with real tiles only? Decide when MIL work starts. Keep
   the bucket proposal; do not add fake source pixels or fake bag tiles.
2. MIL model details: decide the aggregation rule, model, and bag-level loss
   when photo training starts. One bag remains one whole photo. These choices
   must not require changes to source pixels, seeds, tile geometry, or labels.

## 12. Work steps

Run implementation checks and training on the team VM. Each small step has
one check before the next starts. Mode 1 becomes usable before mode 2 work.
Keep the complete mode 2 interfaces in mind during shared-module design.

### Phase A: mode 1 OTF for the tile CNN

| Order | Small task | Check |
| ---: | --- | --- |
| 1 | Get DIV2K split approval; freeze combined records. Keep original RAISE records/images. | Inventories, counts, unique photo indices, source fields, and image-level splits pass. |
| 2 | Expose exact NEF decode, strict `png8`, and legacy-only crop. | Legacy RAISE source identity and DIV2K header/pixel checks pass; all edges remain. |
| 3 | Add explicit grid policies and overlap starts. | Counts, full pixel coverage, legal starts, and no padding pass at both sizes. |
| 4 | Move shared sampling/seed helpers and add lower-to-fit records. Keep legacy wrappers. | Existing v6 replay draws stay exact; new fixed/range overflow tests pass. |
| 5 | Refactor one embedding core with the mask-free return path. Reserve actual-change output for mode 2. | Pixels and RNG state match the full path. |
| 6 | Add mode 1 legacy replay and new overlap pipeline. | Sample replay, tile identities, geometry, and per-tile bpp pass. |
| 7 | Add run/source settings and freeze combined `overlap_full_mix`. | Training sources/weights, hard settings, sizes, and fractions leave fixed full evaluation unchanged. |
| 8 | Build full-photo `.npy` cache with safe resume. | Full source/cache identity, failed-write tests, disk bytes, and VM build time pass. |
| 9 | Connect the mode 1 pre-built writer to the common pipeline. | Writer and OTF pipeline agree before augmentation; legacy output stays exact. |
| 10 | Add mode 1 combined photo assignment and fixed tile batches. | Counts, natural source/class proportions, shape, and memory pass. No default source/class balancing. |
| 11 | Add bounded cross-photo shuffle, fixed/per-epoch policies, and resume. | Both policies pass; fixed evaluation and completed sample order survive restart. |
| 12 | Benchmark combined loading and identical-index sort on the VM; measure DIV2K cache value. | Pixels/RNG state stay exact; 4/8-worker demand, source timings, and memory are recorded. |
| 13 | Complete full legacy source/replay and freeze reference pixel hashes. | All 49,796 original pairs match; retain per-pair RGB hashes, records, and report for PNG-independent checks. |
| 14 | Verify notebook 04 OTF in a separate QUICK/short VM test run, keeping v3c model settings. The user runs full training interactively. | Fixed combined overlap evaluation, per-source metrics, counts, and resume pass. Clean test artifacts; preserve user runs. |
| Optional, after 14 | Review pre-built image removal with the user and team. | All section 8 gates pass. Delete only agreed artifacts; retain records, NEFs, and DIV2K master PNGs. No automatic cleanup. |

### Phase B: mode 2 for future photo aggregation

Start after the mode 1 loader and notebook path work. Pending team approval
for image removal does not block mode 2 development; keep the images meanwhile.

| Order | Small task | Check |
| ---: | --- | --- |
| 15 | Add full-photo mode 2 using the shared source and embedding core. | Bpp uses full photo area; fixed seeds give the same stego photo at 256/512. |
| 16 | Add changed-index labels for every containing overlap tile. | One/two/four-tile labels and distinct pixel counts equal the full comparison. |
| 17 | Connect mode 2 pre-built output and natural-frequency tile/photo records. | Common writer/loader pixels and labels agree, including unchanged tiles. |
| 18 | Add mode 2 loader output and the bag adapter interface, not an MIL model. | One photo per bag, correct counts, deterministic variants, and no fake images. |
| 19 | Benchmark mode 2 for both sources on the VM. | Record throughput, natural source/label frequencies, memory, and source/cache identity. Actual full model training remains user-run. |
| Later | Choose MIL batching/model details when approved. | Preserve the fixed source, full-photo seeds, geometry, and label interface. |
| Later | Add other platform or two-GPU execution only after approval. | Choose and test distributed assignment and equal steps at that time. |

Before mode 1 release or image deletion, complete its source identity,
49,796-pair replay, and notebook OTF checks. Before mode 2 release, also pass
full-photo overlap coverage and cross-size identity. Record real results in
the journal and update instructions only after implementation is verified.
The user pushes local commits; this plan does not authorise a push.
