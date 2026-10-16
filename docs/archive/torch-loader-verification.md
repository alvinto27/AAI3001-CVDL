> Archived 2026-10-16: done; verification remains for reference. See [findings](../findings.md).

# Mode 1 PyTorch loader

Status: plan step 7 complete. Parity/count checks, throughput checks, joint
GPU checks and all unit tests passed. Notebook 04 was read but not changed
in step 7. This is the step 7 record. [Step 8](notebook-otf-verification.md)
now connects the notebook, transfers uint8 and joins small worker groups in the
parent; the float32 and whole-worker-batch contract below is historical.

Current defaults: a 128-image lead per training worker, derived as 32 groups
of four images. Explicit `prefetch_factor` still overrides that derived count.
Notebook `V3_PREFETCH_IMAGES=128` controls the image budget; `V3_PREFETCH` is an
optional group-count override, not a default of four groups. Main torch threads
are 1 (`V3_MAIN_THREADS`); val/test/train-eval use four nonpersistent workers
(`V3_EVAL_WORKERS`), separate from training `V3_WORKERS`. See the
[notebook controls](notebook-otf-verification.md#controls) for the run record.

## Notebook data contract

Notebook cells 2, 8, 11, 12, 25 and the model/trainer cells were inspected.
The new `lsb_torch.py` is the torch boundary; NumPy source/settings/embedding
and cache modules still import no torch.

| Contract | Loader behaviour |
| --- | --- |
| Image | float32, CHW, original 0–255 values; no normalization or resize. |
| Default training | Iterable `(image, label, index)`, label 0/1, independently shuffled images. |
| Optional paired training | Iterable `(clean, stego, pair_index)`, two images per item. |
| Val/test | Map-style `(clean, stego, frame_index)`, deterministic table/index order. |
| Augmentation | Independent 0.5 horizontal/vertical flips and uniform 0..3 rot90, same operations/distributions as notebook `augment_image()`. Per-image key-derived torch generators make it independent of workers. |
| Pair frame | Unique cover_id/pair_id, source_photo, camera_model, tile_row/col, k, bit_mode, method, placement, payload_kind, bpp_actual; plus photo_index, source, grid, tile_size, y/x, requested/effective bpp, clipped, variant, split. |
| Model precision | Notebook Trainer has no precision override: **32-true**, no autocast or GradScaler. Orchestrator confirmed the joint check must match this. |

The following need notebook adapters in the **next** step, not changes here:

- `shuffle=True` cannot be used with this IterableDataset; its buffer replaces it.
- Call `set_epoch(epoch)` before each new training iterator. A shared CPU tensor
  carries the epoch to persistent workers; do not change it mid-iteration.
- QUICK currently samples old PNG pair rows. New limits/subsets must use stable
  new identities without deriving source indices from the subset.
- `make_pairs`, path-based `source_split`, frame clean/stego paths, train-eval
  subsets, and later PNG-based diagnostics need new adapters. The frozen source
  splits replace creation of a new file split. They must not write into checkout.
- Iterable worker tails are retained. PyTorch's `len(DataLoader)` is an estimate
  when each worker has its own partial last batch. Use `batch_count(workers,
  batch_size)` for the exact expected batches, including checkpoint-step checks.
- Cell 25's bpp bins cover only 0.05..0.5. Clipped or rounded OTF payloads can
  fall outside them; report those instead of silently losing their metric rows.
- DIV2K camera is `not_applicable`; its source is still explicit in metrics.

## API and settings

```python
settings = RunSettings()  # source="decode", both training sources, overlap512
corpus = Corpus(repo_root, settings)
train = TrainingImages(corpus)
train.set_epoch(0)
loader = training_loader(train, batch_size=32, num_workers=8, pin_memory=True)
val = FixedPairs(corpus, "val")
frame = val.frame
```

`RunSettings` holds sources, tile size (512/256), grid, training YAML (default
run1 mix), variants/tile, fixed variants, run seed, train photo fraction,
optional source multipliers, source mode, cache root, buffer image capacity,
augmentation and paired mode. Top-left training is RAISE-only 512.

Training fraction selects a stable seeded subset **once**, per source/camera.
Selection is recorded in `corpus.run_record`, including selected photo IDs.
Rounded counts retain at least one photo per nonempty stratum. Epochs reorder
that same subset. Default fraction 1 uses all 1,381 train photos.

Optional source weights are **photo-count multipliers**, not class weights or
new evaluation proportions. A multiplier below one chooses an epoch subset;
above one repeats full-source permutations plus a prefix. Tile counts follow
those photos. Repeated sample keys retain identical bytes and augmentations.
Default None uses every selected photo once at natural tile/class proportions.

Workers deterministically shard a run-seed/epoch photo permutation by stride.
Each loads one whole photo, makes each clean/stego tile/variant with Mode 1,
and streams owned uint8 images through a bounded replacement shuffle buffer.
Queue ordering can change with workers, but pixel generation and augmentation
keys contain no worker/order state. No balanced class sampling is used.

Default buffer is 1,024 **images**, stored as uint8 until emitted. Its payload
is 768 MiB/worker at 512 and 192 MiB/worker at 256, plus one source photo,
metadata, decoder/torch working memory and batch/pin queues. Paired mode stores
half as many pair items for the same image capacity. The historical step 7
prefetch was four whole batches per worker (128 images at batch32); returned
tensors were float32. The current uint8 small-group loader derives tasks from
128 images instead, so reducing group size does not reduce that lead.
Default in_order=True keeps the prescribed worker batch order.

`source="decode"` always reads masters, with OMP=1. `source="cache"` requires
cache_root. Corpus audits header/all records, source versions/master-stat and
NPY shape/header once per session, without full array/master hashes. The
separate [cache checks](cache-verification.md) own hash validation. Each actual
photo load uses **normal np.load**, cheap checks remain on, hashes off. There
is no mmap training source and no decode fallback.

Evaluation profile `overlap_full_mix` always uses both frozen sources, all
photos/tiles in the split, the run1 YAML mix and fixed val/test keys. Training
sources/fraction/weights/custom YAML cannot change it. Variants/tile and tile
size remain explicit. `legacy_v6` is exactly RAISE top-left512, one variant,
original cover indices/seed/settings; purpose `all` exposes all 49,796 pairs.
Map-style eval keeps a last-photo working array per worker, not a disk cache.
Preview arrays are cleared when the dataset is copied to a new worker.

## Correctness and counts

The [performance-default checks](notebook-otf-verification.md#performance-default-checks-2026-10-13)
record the later 85-test suite, ordered old/new prefetch parity, four-worker full
validation, asynchronous PC/L4 windows, and early-stop worker cleanup.
The tables below retain the original step 7 verification results.

| Check | Result |
| --- | --- |
| Decode/cache, all tiles | Eight train photos: each RAISE camera/orientation plus DIV2K 0001/0002. At 512: 640 train images, 320 val pairs. At 256: 2,432 train images, 1,216 val pairs. Zero pixel/frame differences; train and val keys. |
| Workers | Real sample sets match at 0 and 4 workers at both sizes. Unit tests also compare keyed augmentation at 0 and 2 workers. |
| Persistent epochs | Unit test with two persistent workers: set_epoch(1) changes every stego sample; resulting full sample set equals single-process epoch 1. |
| Fixed variants/fraction | Unit tests prove epoch-fixed bytes, decode/cache identity, seeded stable recorded subset and unchanged subset across epochs. |
| Legacy loader | 2,000 uniformly spread pairs from the 49,796-entry profile match original clean/stego PNG pixels; zero differences. 197.099656 s. |
| Parity time | 103.066557 s for both sizes, source modes/purposes and worker comparison. |
| Counts time | 14.161691 s. Train variants=2 doubles count; default selected train photos=1,381. |
| Step 7 tests | 79 passed (45 source/manifest/settings/core/cache/loader, plus 34 engine/batch/prepare). Five new loader tests also prove unordered scheduling preserves the keyed sample set. |

| Tile size | Train images/epoch | Val images | Test images | Total images |
| --- | ---: | ---: | ---: | ---: |
| 512 | 107,900 | 22,476 | 21,912 | 152,288 |
| 256 | 405,144 | 84,408 | 82,176 | 571,728 |

Each image count is manifest tile totals x 2 x variants. Eval dataset length
is pairs (half the image count). Paired training length is likewise pairs.

## Dummy consumer and joint check

Dummy consumer uses batch32, pin_memory=True, default
buffer/augmentation, retains all samples and consumes at least 120 seconds after
16 initial batches (the first worker buffer fill has already completed). It
samples whole process-tree CPU, storage read_bytes and peak summed RSS/PSS.
RSS can count shared libraries repeatedly; PSS divides shared memory.
CPU percentages use 100% per busy CPU, up to 800% on the eight-vCPU VM.

Joint check uses only notebook KV/XuNet definitions (cells 10/14), clip3,
GroupNorm, AdamW lr1e-3/weight_decay1e-4, cross entropy, 32-true. It first requires
nvidia-smi to show L4 utilization zero and no other compute process. It performs
300 optimizer steps, reports the last 280 after warm-up, and writes no model or
checkpoint. A separate authorized four-worker check uses 100 steps / 80 steady
steps. All run logs stay in scratch and are deleted after reporting.

### Baseline dummy consumer

Initial baseline used prefetch_factor=1, in_order=True, to limit batch RAM.
All windows below are at least 120 s after buffer fill; measurements include
augmentation, float conversion, inter-process transfer and host pinning.
RSS/PSS are whole process-tree peaks in decimal GB. Disk values are observed
kernel read_bytes and are not cold-cache comparisons.

| Source | Tile | Workers | Images/s | CPU % | RSS GB | PSS GB | Disk MB/s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Cache | 512 | 4 | 303.478 | 363.7 | 7.844 | 6.886 | 71.156 |
| Cache | 512 | 6 | 358.075 | 521.0 | 11.102 | 9.703 | 79.733 |
| Cache | 512 | 8 | 417.583 | 688.7 | 14.277 | 12.427 | 48.333 |
| Cache | 256 | 4 | 1067.742 | 345.0 | 4.301 | 3.345 | 48.533 |
| Cache | 256 | 6 | 1403.448 | 538.7 | 6.051 | 4.652 | 0.432 |
| Cache | 256 | 8 | 1325.526 | 563.8 | 7.812 | 5.924 | 106.830 |
| Decode | 512 | 8 | 129.374 | 404.3 | 14.484 | 12.560 | 17.602 |
| Decode | 256 | 8 | 309.038 | 242.6 | 8.274 | 6.355 | 0.000 |

Decode256 above is the bounded rerun. The first attempt was stopped by the
Orchestrator during an unnecessary whole-epoch drain; its timing was not
logged, so no result is claimed from that attempt. Initial cache512/8 early
forced worker shutdown emitted two C++ termination messages after the timing
window. No iteration/sample error occurred. The helper was changed to natural
producer completion with asserted worker exit codes; the cache rerun below has
all-zero exit codes and no termination messages.

### Decode scheduling and short benchmark cleanup

A benchmark-only subclass caps the photo set at 750 and has a shared producer
stop flag. After the timed window it stops new production, naturally drains
only the bounded queue, and checks every worker exit code. This does not change
production epochs or sample keys. The old whole-epoch drain proved nothing
extra and made decode512 wall time 860.687 s for a 121.446 s measurement.
Bounded decode512 cleanup is only 18.7–19.3 s; decode256 cleanup is 4.9 s.

| 512 source | Workers | Prefetch | Ordered | Images/s | CPU % | RSS GB | PSS GB | Disk MB/s |
| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| Decode | 8 | 4 | True | 193.539 | 774.1 | 17.679 | 15.766 | 4.001 |
| Decode | 8 | 8 | True | 183.468 | 771.9 | 19.995 | 18.109 | 0.000 |
| Decode | 8 | 4 | False | 201.561 | 797.5 | 13.990 | 12.122 | 0.359 |
| Cache | 8 | 4 | False | 421.007 | 742.7 | 13.631 | 11.810 | 137.161 |

All have 120 s windows and all worker exit codes zero. Larger prefetch fixes
much of the decode head-of-line idle time. Prefetch8 uses more RAM and did not
win. Unordered delivery adds 4.1% over ordered prefetch4 for decode here, but
batch arrival order is not reproducible. **Ordered remains the default.**
Cache gain over its prefetch1 baseline is only 0.82%, with different page
residency; no robust large cache gain is claimed. Prefetch4 is the chosen
step 7 whole-batch default, not the current small-group count. The buffer
payload remains unchanged; extra batches add
host memory. In particular, the 256/cache6 low disk reading reflects warm pages.

### L4 joint checks

Each run first showed `NVIDIA L4, 0` and an empty compute-process list. Precision
32-true, clip3/GroupNorm, batch32, cache512, prefetch4. No autocast/GradScaler,
checkpoint, repository log, or model file. Warm-up is the first 20 steps.

| Workers | Ordered | Total steps | Steady images/s | GPU util % | Loader wait ms/batch | Wait fraction | Peak RSS GB | Peak PSS GB | CPU % | Disk MB/s |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 8 | False | 300 | 268.282 | 100.000 | 0.483 | 0.405% | 18.923 | 17.045 | 540.3 | 87.583 |
| 8 | True | 300 | 266.727 | 99.833 | 0.672 | 0.561% | 19.237 | 17.298 | 508.0 | 1.578 |
| 4 | True | 100 | 266.734 | 100.000 | 0.280 | 0.234% | 10.343 | 9.437 | 500.9 | 53.771 |

Steady wall times are 33.397635 s, 33.592407 s and 9.597581 s respectively.
All worker exit codes zero; GPU process list empty after completion. The loader
keeps up above the roughly 215 images/s target in these checks. For cache mode,
four ordered workers are a useful low-RAM starting point: about 8 GB less peak
PSS than eight, with the same short-run model throughput. The four-worker joint
window is short and mostly buffered, so it is not proof of long-run stability;
the 120 s dummy rate of 303 images/s supports trying it. Decode mode needs eight
workers/prefetch4 for the measured improvement and is still slower than the
cached joint model rate. Notebook integration/default choice remains next work.

## Re-run

Copy code, YAML and the unchanged notebook to VM scratch. Use the existing VM
Python and `scripts/archive/check_torch_loader.py --repo-root ... --cache-root ...` with
`--gate counts`, `parity`, `legacy`, `speed`, or (only if idle) `gpu`. Speed gate
requires `--source`, `--tile-size`, `--workers`, `--seconds 120`; GPU gate requires
an unchanged scratch `--notebook` path. Do not run from a writable VM checkout.

DataLoader sharding/persistent-worker/pinning guidance was checked against
current PyTorch docs via Context7 `/pytorch/pytorch` (official data.md and
DataLoader source). No dependency files or notebook cells were changed.
