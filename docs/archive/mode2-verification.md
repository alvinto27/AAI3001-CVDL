> Archived 2026-10-16: done; verification remains for reference. See [findings](../findings.md).

# Mode 2 full-photo verification

Status: Phase B steps 15–16 complete (2026-10-13); on-the-fly steps 18–19
complete (2026-10-14). Step 17 (pre-built writer) is deferred. The notebook,
writer and MIL model are not connected. No training was run.

## Library contract

[`lsb_mode2.py`](../../LSB-Dataset-Generator/lsb_mode2.py) has three functions:

- `embed_photo(photo, record, config, OTFSeed(...))` returns an owned stego
  RGB8 photo, a photo record, and sorted, read-only changed RGB-value indices.
  Read the full oriented NEF or PNG8 with the shared source/cache API first.
  The clean input is not changed. No noise is added here.
- `tile_change_counts(indices, record, tile_size)` returns two row/column
  grids: changed RGB-value counts and distinct changed-pixel counts.
- `iter_photo_tiles(clean, stego, record, metadata, indices, tile_size)`
  returns owned clean/stego tiles and side records. Keep unchanged stego-photo
  tiles. Their label is 0, not 1.

A RGB value is one channel of a pixel. Three changed channels in one pixel
count as three values but one pixel. A clean tile always has label 0. A
stego-photo tile has label 1 only if its changed-value count is positive.
`photo_embedded` records the output role, independently of that tile label.
`photo_actual_changed` records whether its source output photo changed.
Each tile also records `actual_changed`, both counts, source, photo identity,
variant, grid, row/column, actual start and a separate clean/stego sample ID.
These are ordinary tile outputs, not photo bags.

The photo record uses algorithm version `mode2_full_v1` and seed scheme
`otf_v1`. It records NumPy version, full dimensions/pixel count, source/noise,
master path, camera, variant policy, effective generation seed/epoch and both
12-integer seed keys. Its settings record includes requested/effective bpp,
requested/effective payload bytes, capacity and clipping. Its embedding record
includes payload bytes and actual changed/used RGB-value counts.

Bpp and capacity use the **full photo**, never one tile or the sum of overlap
areas. Generation takes no tile-size argument. Shared `otf_seed_sequence`
uses carrier `photo`, with grid, size, row and column all neutral 0.
Validation/test use evaluation seed 1 and epoch 0. Training supports fixed
and per-epoch variants. The older four-integer Mode 2 proposal is not used;
this implementation follows the approved shared OTF seed route.

Mapping uses blocks of at most 65,536 changed indices. For each coordinate,
check the last axis start at/before it and that start's predecessor. Test both
axis intervals with inclusive starts and exclusive ends. A change can count
in one, two or four tiles. Distinct-pixel counting removes repeated channels
across block boundaries too. There is no production full-photo comparison
or RGB difference mask. The comparison below is a verification reference only.

## Correctness checks

PC execution used container `aai3001`. Both machines used NumPy 2.5.3,
rawpy 0.27.1 / LibRaw 0.22.1 and Pillow 12.3.0. Source decoding used
`OMP_NUM_THREADS=1`. No GPU or backend setting was changed.

- **93 tests passed:** 59 shared tests and 34 generator tests. The eight new
  Mode 2 tests also passed CPU-only in VM scratch.
- Unit coverage: all **384** combinations of lowest/plane, k=1–8,
  replace/match, contiguous/stride/random, random-byte/letter payloads and
  low/high bpp. Both 256 and 512 grids match direct comparison.
- Real-photo coverage: **32 photos**, eight each from D7000, D90, D40 and
  DIV2K; **1,536 settings cases** and **296,448 tile rectangles** compared at
  both sizes. Every group covers all 384 settings combinations across its
  eight photos. Both RAISE orientations are included. Zero label, RGB-value
  count or distinct-pixel count mismatches.
- Full-photo requested payload rounding/capacity was checked in every real
  case. Fixed/ranged overflow clipping and the one-byte minimum are tested.
- All 32 photos: repeatability, evaluation run-seed/epoch invariance,
  independent 256/512 full-stego SHA-256 equality, emitted clean labels and
  direct-decode/cache embedding/metadata/index equality. Temporary verified
  full-photo caches were removed. Unit tests cover owned and read-only mmap
  cache inputs. The retained VM cache was not changed.
- Explicit one/two/four-tile edge/corner cases at both sizes; three changed
  channels in one pixel, one-index blocks, empty actual changes, exact axis
  multiples, exclusive ends, last row/column and last photo pixel.
- Eight real photos: complete full-photo, tile-pixel, tile-record and index
  fingerprints match serial forward/reverse calls and spawn worker counts
  1/4, with simulated rank environment values 0/3. This is not a multi-GPU test.
- Mode 1 code is unchanged. Its existing 500-pair frozen PNG replay passed
  for all six camera/orientation groups, zero mismatches. Its eight-photo OTF
  gate passed serial/reversed/parallel generation, both tile sizes, all
  purposes and fixed variants, zero mismatches. The earlier 49,796-pair full
  replay was not rerun for this change.

SHA-256 of the JSON list of sorted `(photo_id, full_stego_hash)` pairs for the
32 real repeat/cross-size cases:

```text
8ee1887296e0189c063de69471001ab2462dd1545b6ee59dac3a19738b256828
```

## Natural validation label frequencies

All **270 frozen validation photos**: RAISE 150 (D7000 107, D90 42, D40 1),
DIV2K 120. One clean and one generated photo output per photo, variant 1,
fixed evaluation seed 1/epoch 0. Settings use the normalized choices in
[`configs/raise512-run1.yaml`](../../configs/raise512-run1.yaml), through the new
OTF lower-to-fit draw, not legacy replay. Choices are lowest/plane, k=1–8,
replace/match, the three placements, stride 2–16, random bytes/letters,
random start and bpp 0.05–0.5. Configuration-file SHA-256:

```text
29b1e7baaee60e44782791cb5016562a311b930fc15bc75750c4ac253b1c9de7
```

Each real overlap tile is one sample. No pixel deduplication, fractional
weight, source weighting, relabeling or class correction is applied. The
positive count is among stego-photo tiles; all-tile percentage also includes
an equal number of clean tiles. Combined results pool individual tiles,
not source percentages. Placement is the drawn full-photo placement.

| Source | Placement | Size | Photos | Positive / stego tiles | Stego positive | All-tile positive |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| RAISE | contiguous | 512 | 48 | 659/3154 | 20.8941% | 10.4471% |
| RAISE | contiguous | 256 | 48 | 1649/11756 | 14.0269% | 7.0134% |
| RAISE | stride | 512 | 49 | 1885/3222 | 58.5040% | 29.2520% |
| RAISE | stride | 256 | 49 | 6469/12012 | 53.8545% | 26.9272% |
| RAISE | random | 512 | 53 | 3406/3406 | 100.0000% | 50.0000% |
| RAISE | random | 256 | 53 | 12716/12716 | 100.0000% | 50.0000% |
| RAISE | all | 512 | 150 | 5950/9782 | 60.8260% | 30.4130% |
| RAISE | all | 256 | 150 | 20834/36484 | 57.1045% | 28.5522% |
| DIV2K | contiguous | 512 | 39 | 231/472 | 48.9407% | 24.4703% |
| DIV2K | contiguous | 256 | 39 | 494/1864 | 26.5021% | 13.2511% |
| DIV2K | stride | 512 | 46 | 419/564 | 74.2908% | 37.1454% |
| DIV2K | stride | 256 | 46 | 1322/2200 | 60.0909% | 30.0455% |
| DIV2K | random | 512 | 35 | 420/420 | 100.0000% | 50.0000% |
| DIV2K | random | 256 | 35 | 1656/1656 | 100.0000% | 50.0000% |
| DIV2K | all | 512 | 120 | 1070/1456 | 73.4890% | 36.7445% |
| DIV2K | all | 256 | 120 | 3472/5720 | 60.6993% | 30.3497% |
| Combined | contiguous | 512 | 87 | 890/3626 | 24.5450% | 12.2725% |
| Combined | contiguous | 256 | 87 | 2143/13620 | 15.7342% | 7.8671% |
| Combined | stride | 512 | 95 | 2304/3786 | 60.8558% | 30.4279% |
| Combined | stride | 256 | 95 | 7791/14212 | 54.8199% | 27.4099% |
| Combined | random | 512 | 88 | 3826/3826 | 100.0000% | 50.0000% |
| Combined | random | 256 | 88 | 14372/14372 | 100.0000% | 50.0000% |
| Combined | all | 512 | 270 | 7020/11238 | 62.4666% | 31.2333% |
| Combined | all | 256 | 270 | 24306/42204 | 57.5917% | 28.7958% |

Total samples are **22,476 at 512** and **84,408 at 256**. These are different
sample sets despite identical full stego photos. D40 has only one validation
photo; its stego-positive fraction is 25% at both sizes. This small count
must not be treated as a population estimate. The old cropped-photo 69.03%
figure is not the result of this full-photo overlap route.

## CPU timing and peak memory

PC: AMD Ryzen 7 7700, eight cores/16 threads. Eight photos per group, selected
in manifest order with both orientations first where available. VM: eight
Xeon vCPUs at 2.20 GHz; one corresponding photo per group. Before VM work,
compute-process and notebook/training-kernel checks were empty. Execution
used only mirrored scratch code and read-only live source files. VM timing
was 57.191 seconds; no GPU computation was used. PC timing was 224.126 seconds.

Each photo/profile uses a fresh subprocess and three trials. Times below
are trial medians: PC over 24 trials/group/profile, VM over three. Embed time
measures `embed_fast` itself. Settings time is the remaining `embed_photo`
wall time (draws, seeds and record). Label time measures both value and pixel
count grids, separately for 512 and 256; it excludes tile copies and decode.
Peak RSS is the highest Linux process resident-memory peak, in MiB,
**including decode, source/stego arrays, core work and labels**. It is not an
isolated counter allocation. No full-difference reference ran in timed cases.

`mix` is the fixed evaluation mix above. Other profiles use random placement,
match, 0.5 full-photo bpp and random-byte payload: lowest k=1, plane k=1, plane
k=8. A plane setting still uses one bit per RGB value; k chooses the bit
position. These profiles therefore have similar selection sizes. They are
CPU microbenchmarks, not tile-loader or training throughput claims.

### PC

| Group | Profile | Embed ms | Settings ms | Label 512 ms | Label 256 ms | Peak RSS MiB |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| D7000 | mix | 50.609 | 0.183 | 36.080 | 38.433 | 639.94 |
| D7000 | lowest1 | 699.013 | 0.171 | 88.753 | 93.124 | 658.31 |
| D7000 | plane1 | 695.558 | 0.164 | 85.870 | 90.944 | 650.78 |
| D7000 | plane8 | 690.291 | 0.168 | 85.436 | 90.352 | 650.70 |
| D90 | mix | 28.342 | 0.174 | 15.453 | 16.157 | 389.78 |
| D90 | lowest1 | 524.703 | 0.167 | 67.748 | 71.280 | 469.16 |
| D90 | plane1 | 520.430 | 0.165 | 65.814 | 69.386 | 507.36 |
| D90 | plane8 | 515.677 | 0.165 | 65.791 | 69.462 | 507.55 |
| D40 | mix | 17.752 | 0.168 | 7.177 | 7.753 | 236.72 |
| D40 | lowest1 | 254.225 | 0.164 | 30.697 | 32.858 | 274.91 |
| D40 | plane1 | 250.589 | 0.156 | 29.472 | 31.522 | 273.31 |
| D40 | plane8 | 249.099 | 0.162 | 29.430 | 31.340 | 273.37 |
| DIV2K | mix | 9.004 | 0.156 | 5.604 | 5.466 | 94.16 |
| DIV2K | lowest1 | 110.866 | 0.146 | 13.206 | 14.096 | 213.59 |
| DIV2K | plane1 | 111.562 | 0.151 | 13.244 | 14.190 | 209.64 |
| DIV2K | plane8 | 110.390 | 0.150 | 13.391 | 14.144 | 209.18 |

### VM, CPU only

| Group | Profile | Embed ms | Settings ms | Label 512 ms | Label 256 ms | Peak RSS MiB |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| D7000 | mix | 122.991 | 0.470 | 183.612 | 188.779 | 248.79 |
| D7000 | lowest1 | 883.364 | 0.452 | 312.407 | 316.529 | 599.41 |
| D7000 | plane1 | 872.974 | 0.420 | 310.595 | 320.208 | 649.50 |
| D7000 | plane8 | 860.454 | 0.421 | 314.185 | 317.750 | 649.66 |
| D90 | mix | 25.229 | 0.452 | 20.778 | 21.157 | 201.10 |
| D90 | lowest1 | 679.220 | 0.408 | 240.926 | 243.818 | 468.22 |
| D90 | plane1 | 654.628 | 0.416 | 238.980 | 245.824 | 506.38 |
| D90 | plane8 | 640.769 | 0.408 | 237.751 | 242.041 | 506.52 |
| D40 | mix | 42.590 | 0.460 | 21.749 | 21.969 | 123.40 |
| D40 | lowest1 | 343.972 | 0.411 | 112.974 | 116.412 | 271.00 |
| D40 | plane1 | 326.714 | 0.384 | 111.913 | 115.473 | 272.44 |
| D40 | plane8 | 327.298 | 0.416 | 113.304 | 117.201 | 272.42 |
| DIV2K | mix | 3.919 | 0.389 | 4.148 | 3.795 | 71.87 |
| DIV2K | lowest1 | 146.899 | 0.402 | 52.571 | 53.687 | 172.22 |
| DIV2K | plane1 | 145.502 | 0.358 | 52.562 | 54.177 | 169.62 |
| DIV2K | plane8 | 141.039 | 0.373 | 52.911 | 54.425 | 169.47 |

VM photo IDs/shapes: `r00617aa1t` 3280×4948, `r01058910t` 2868×4310,
`r221dfc60t` 2014×3039, `div2k_0001` 1404×2040. The VM sample is smaller
than the PC sample; mixed-profile rows are not matched-group speed ratios.

## Checks and remaining scope

Run Python inside the configured container, not on the PC host:

```sh
podman exec -w /work aai3001 python -m unittest discover -s tests
podman exec -w /work/LSB-Dataset-Generator aai3001 python -m unittest discover -s tests
podman exec -w /work aai3001 python scripts/check_mode1.py --repo-root /work --gate sample
podman exec -w /work aai3001 python scripts/check_mode1.py --repo-root /work --gate otf
```

Documentation links/anchors and `git diff --check` passed for steps 15–16.
That work did not change the notebook, checkpoints, datasets, journal or
backend. Step 17 remains deferred; steps 18–19 follow below. See the
[design plan](plan-otf-generator.md).

## On-the-fly loader and photo bags (steps 18–19)

[`lsb_mode2_torch.py`](../../LSB-Dataset-Generator/lsb_mode2_torch.py)
reuses the Mode 1 `Corpus` inventory, selected train schedule, source/cache
readers, configuration, and worker setup. Its map-style `Mode2Photos` dataset
assigns **one whole photo to one worker task**. Each task reads its source
once, calls `embed_photo` once, and cuts all clean and embedded tiles via
`iter_photo_tiles`. Training uses variant 1, per-epoch seeds unless
`fixed_variants=True`; validation/test always use their frozen seed and epoch.
Only `overlap_last`, one photo variant, uint8 and independent image output
are accepted. More than one photo variant per epoch remains a later policy.

`TileLoader` emits `(images, actual_change_labels, stable_keys, side_records)`.
Images are `B x 3 x T x T` uint8; the consumer converts them to float32.
The side record is the complete Mode 2 tile record: photo/source/camera,
original grid coordinates/start, clean or embedded role, both change counts,
actual label and full-photo settings. No unchanged embedded tile is removed
or relabeled. The keyed training augmentation is optional: it uses separate
clean/embedded roles and tile coordinates in a stream **separate from the
photo embedding seed**. Reference comparisons and CPU measurements below set
augmentation off to compare original pixels. Photo tasks arrive in fixed
index order across worker counts; the training output can use a bounded
cross-photo image shuffle (default 1,024 images). Prefetch is specified in
images and rounded up against the largest whole-photo task (at least one
task/worker). At the 128-image default this is one task per worker; a smaller
photo can therefore prepare fewer than 128 images. The last tile batch is
short, never padded or dropped. `set_epoch` uses shared
memory so persistent workers see the next epoch.

`BagLoader` emits one `PhotoBag` with exactly one real photo output: clean or
embedded, not both mixed together. Its `tiles` are `N x 3 x T x T` uint8 in
row-major grid order. `tile_labels` has length `N`. Separate fields record
`photo_embedded`, `photo_actual_changed`, `photo_label`, and the
`photo_label_rule="any_changed_tile"`; the label is 1 only when at least
one tile changed. A clean bag always has both flags false and label 0.
Side records retain the clean/embedded sample IDs. `batch_size=1` yields a
bag; larger batch sizes yield a list of bags without padding. This is an
interface, **not** an MIL model or a fixed-`N` batching scheme. The grid
maximum is 70 tiles at 512 and 260 at 256 for RAISE; DIV2K maxima are
16 and 64. `N` is the actual count for each bag, not this limit.

### Pixel, label, seed and regression checks

Three synthetic-loader tests use exact partial-overlap geometry, 256 tiles,
source decode and cache, worker counts 0/2/4, train/val/test, and the direct
`embed_photo` + `iter_photo_tiles` reference. All pixel bytes, labels and
complete JSON side records match, including unchanged embedded tiles. An
independent work-item assertion counts **one source read and one embed** for
a photo. The train epoch changes embedded bytes; `fixed_variants=True`
keeps them fixed. Val/test remain fixed even after `set_epoch`. Bag tests
confirm no mixed photo, grid count/order, exact tile labels, photo-label
rule, no padding and identical photo order/labels across workers. Keyed
training augmentation yields identical output across 0/2/4 workers.

A second read-only check used **six real photos** (one RAISE and one DIV2K
in each train/val/test split). It compared loader output against the direct
reference for both decode and cache at 0/2/4 workers: **18 cases**, zero
pixel, label, side-record or bag failures. It checked full bag identity and
real tile counts. The cache was read only.

Mode 1 source was unchanged. To guard against accidental changes in the
200-batch stream, the first **6,400 images** (keys, labels and uint8 bytes)
from the 6-worker cache loader produced the same SHA-256 for the frozen
`HEAD` Mode 1 module and the current module:

```text
32bd10352c3e2c3d6a5c4600632bd0f962c9528975572e162bb973f7acc99075
```

The final PC suites passed **72 shared tests** (including four new Mode 2
loader tests) and 34 generator tests. The four new tests also passed in VM
CPU-only scratch. No training, GPU job or notebook edit was made for these
checks.

### CPU throughput and proportional RAM

Both machines used OMP=1 and no GPU work. The VM had no GPU process or
notebook kernel before its cases; the PC had no user notebook kernel. Each
case used the same first 128 seed-0 training photo tasks at 512: **9,808
images**, or **256 separate clean/embedded bags**, from both sources.
The timer includes source read, full-photo embedding, labels, tile creation,
transfer from worker processes, shuffle for tiles, and Python iteration;
it excludes inventory/cache-session setup. PSS is the largest observed sum
of parent plus child proportional memory, sampled every eight outputs.
It is not a kernel-enforced maximum. Rates are single-run observations,
not full-epoch or GPU training rates.

| Platform / source / workers | Tile images/s | Tile peak PSS GB | Bags/s | Bag peak PSS GB |
| --- | ---: | ---: | ---: | ---: |
| PC / cache / 4 | 512.1 | 4.41 | 14.88 | 3.04 |
| PC / cache / 6 | 505.7 | 5.30 | 14.94 | 4.04 |
| PC / decode / 8 | 326.0 | 6.00 | 8.96 | 4.41 |
| VM / cache / 4 | 268.5 | 4.16 | 8.59 | 2.61 |
| VM / cache / 6 | 286.9 | 4.97 | 8.22 | 3.70 |
| VM / decode / 8 | 136.1 | 6.04 | 3.77 | 4.61 |

A separate full 512 training epoch through cache6 yielded **107,900 tile
images from 1,381 selected train photos** on both machines: PC 127.477 s
(846.43 images/s), VM 252.010 s (428.16 images/s). The difference from
the first-128-photo rates includes startup, page-cache state, photo mix and
large-stream amortization; do not infer a controlled cache speedup from
these unlike windows. Peak sampled PSS was 5.73 GB on PC and 5.27 GB on VM.

### Natural training labels

The full cache6 epoch on both machines produced the **same** per-source
counts. Each embedded photo has an equally sized clean output, but unchanged
embedded tiles stay label 0. No class balancing, source weights or loss
weights were applied. Percentages below are positive labels divided by all
real output tiles, with the stego-only percentage given separately.

| Source | Clean + embedded tiles | Actual label 1 | All-tile label 1 | Embedded-tile label 1 |
| --- | ---: | ---: | ---: | ---: |
| RAISE | 91,348 | 27,008 | 29.5661% | 59.1321% |
| DIV2K | 16,552 | 5,870 | 35.4640% | 70.9280% |
| Combined | 107,900 | 32,878 | 30.4708% | 60.9416% |

The table is the natural **training** mix for seed 0 at 512; it differs from
the fixed validation profile above. No source weighting or class-frequency
correction is applied.
