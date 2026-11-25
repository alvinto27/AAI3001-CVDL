## 2026-10-13 — Late 10-12 commits, doc corrections, and on-the-fly generation

### Did

- Recorded six commits from 2026-10-12 that had no journal entry:
  - `267f167`: added README files for the ignored data folders
    (`ImageSets/`, `Datasets/`, `checkpoints/`, `logs/`) and listed them in
    `docs/README.md`.
  - `84b20b8`: added the RAISE citation where the data is used: the root
    README, `Datasets/README.md`, the rebuild guide, and the v1, v3b, and v3
    reports.
  - `7c0984e`: notebook 04 now runs without `notebooks/rocm_conv1_wgrad.py`.
    If the module is missing, it prints a message and uses the standard conv1.
  - `cc2f87a`: notebook 04 now runs the best-checkpoint callback before the
    every-epoch callback, so `last.ckpt` stores the correct best path. If that
    path is missing, the notebook selects the `best-*.ckpt` file with the
    lowest `val_pe`.
  - `e5df2eb`: added the executed v3c notebook
    `notebooks/runs/04_v3c_clip3_group.ipynb` and listed it in the README.
  - `3787b14`: git now tracks the small v3a, v3b, and v3c run outputs (test
    CSVs, heatmaps, `metrics.csv`, `hparams.yaml`). The `.ckpt` model files
    stay ignored and live on the VM under
    `/srv/aai3001/AAI3001-CVDL/checkpoints/`.
- Corrected stale doc statements in `43f7163` and in this journal:
  - Replaced "local only" and "not committed" notes for run outputs.
  - Corrected the "logs/ cleared" notes.
  - Marked the v3b report as superseded by v3c and closed its done items.
  - Corrected the v3b all-stego claim in `results-v3.md`: that check was run
    only for v3c.
  - Replaced "observed run-to-run noise" with "not measured".
  - Used 4 decimal places in the V3c row of the V3 comparison table.
- Counted confident false alarms from each run's `test_predictions.csv`
  (clean rows, `label` 0) to give a source for the noisy-cover plan.
- Ran a CPU-only benchmark of tile -> embed on the fly. "On the fly" means
  the data loader makes each stego tile during training, instead of reading
  a stored stego PNG.
  - The Worker used a temporary script and deleted it. No repository code
    changed.
  - PC: AMD Ryzen 7 7700, `nproc` 16.
  - Input: 4,096 real train pairs. DataLoader batch of 32 images, with the
    notebook flips and 90-degree rotations. Warm file cache. 3 trials.
- Ran a CPU-only benchmark of embed -> tile on the fly (plan B). The loader
  embeds into the whole cropped photo, then cuts the tiles.
  - The Worker deleted its temporary files. No repository code changed.
  - PC: AMD Ryzen 7 7700, `nproc` 16. Latency was measured on one CPU with
    threads=1.
  - Input: 36 train photos (32 D7000, 3 D90, 1 D40).
  - Settings came from the original `lsb_batch._draw_settings`, applied to
    the whole cropped photo, so bpp applies to the photo. Seeds were
    `SeedSequence([1, photo index, 1, 0])` for the settings and
    `[..., 1]` for the embedding.
  - The embed step was a copy of the engine without the mask and
    `EmbedResult` tail. Only that tail was removed.
- Compared NEF and 16-bit RGB PNG as the single master source.
  - CPU only, AMD Ryzen 7 7700. Input: 42 train photos (32 D7000, 5 D90,
    5 D40). One-core timings were pinned to one CPU with threads=1.
  - No repository code changed. The Worker deleted the temporary files
    (about 12 GB).
  - The Worker installed two packages in the container only:
    opencv-python-headless 5.0.0.93 and pypng 0.20220715.0. They are lost
    if the container is recreated.
- Context from the user:
  - The Kaggle TPU v5e-8 had a queue (position #525). The TPU is not the plan.
  - The user reviewed the paid Colab tiers. The user already has 400 GB of
    Google storage (Google AI Plus, student plan). Storage is not a limit,
    and no paid plan is necessary now.
- The Worker wrote the on-the-fly generator design plan:
  [docs/archive/plan-otf-generator.md](../../archive/plan-otf-generator.md), commit `4d1036b`. It is
  a design only. It has no code. It covers:
  - shared modules for the pre-built dataset writer and the training loader;
  - exact replay in mode 1 (tile -> embed);
  - deterministic mode 2 (embed -> tile), with labels from actual pixel
    changes;
  - a RAW-noise hook, disabled;
  - speed work, with exactness checks;
  - seeds and variants;
  - settings that each short run can change: k, bit_mode, method,
    placement, stride, bpp, payload, tile size (512 or 256), and the
    fraction of train photos;
  - a session cache, tests, and 17 work steps.
  - No new package is necessary. rawpy, NumPy, Pillow, PyYAML, PyTorch, and
    Lightning are already in `pyproject.toml`.
- The Worker updated the plan for overlapped tiles in commit `18ef1fa`.
  - Tile grid per landscape photo (rows x columns = tiles). Portrait photos
    swap rows and columns.

    | Camera | 512 grid | 256 grid |
    | --- | --- | --- |
    | D7000 | 7 x 10 = 70 | 13 x 20 = 260 |
    | D90 | 6 x 9 = 54 | 12 x 17 = 204 |
    | D40 | 4 x 6 = 24 | 8 x 12 = 96 |

  - Totals for one grid over all 1,000 photos: 65,184 tiles at 512 and
    243,120 tiles at 256.
  - Mode 2 embeds into the full photo. The seeds do not depend on the tile
    size, so 256 and 512 runs see the same stego photo.
  - A changed pixel in an overlap marks each tile that contains it as
    stego: 2 tiles at an edge overlap, up to 4 at a corner.
  - In new mode 1 runs, the generator embeds each overlapping tile
    independently.
  - Memory for one D7000 clean + stego float32 bag at 512: about 420 MiB,
    30% more than with the crop.
  - The earlier speed, memory, and 34.5% label-1 results are historical crop
    results. The values for overlapped tiles are Pending.
- Set up SSH to the team VM with one shared connection (OpenSSH
  ControlMaster). Thus Tailscale SSH check mode asks for a login only one
  time in each session.
  - VM: NVIDIA L4, 8 vCPUs, 31 GB RAM, about 100 GB of free disk.
  - The repository is at `/srv/aai3001/AAI3001-CVDL`, on branch `main`.
  - The RAISE NEFs are already on the VM.
- The ROCm-Works team (a separate orchestrator) is standing down. Agreed
  rules: announce PC GPU jobs in the shared orchestrator room. ROCm-Works
  writes only in `rocm-perf/` and makes no commits.
- The Worker settled the plan policies and added DIV2K as a second image
  source in commit `e96ad9f`.
  - DIV2K header audit (read-only): all 900 PNG files are 8-bit RGB. The
    shortest side is 648 pixels.
  - Overlapped-grid tiles, one grid:

    | Set | Tiles at 512 | Tiles at 256 |
    | --- | ---: | ---: |
    | DIV2K official train | 9,732 | 37,936 |
    | DIV2K official valid | 1,228 | 4,808 |
    | DIV2K total | 10,960 | 42,744 |
    | RAISE + DIV2K | 76,144 | 285,864 |

  - DIV2K is about 14% of all tiles.
- Moved the VM checkout from `main` to `yx` (fast-forward to `18ef1fa`).
  The teammate's untracked `modelling/` folder was not touched.
- The Worker did plan step 1: froze the photo and tile records. Commit
  `63e4f59`.
  - Builder: `scripts/build_manifests.py`, with tests in
    `tests/test_build_manifests.py`.
  - Record and runbook:
    [docs/active/frozen-source-records.md](../../active/frozen-source-records.md).
  - The build ran on the VM from a scratch copy. Two builds gave
    byte-identical CSV files. The PC copies match the VM copies by SHA-256.
  - Input: `ImageSets/RAISE/tiles512-records/manifest.csv` (SHA-256
    `400f0b3e823f088183316c4948e983fb74b6808acb4059cf9eb8506fcb35f6fb`).
    The VM dataset copy has no `records/` folder. Thus this preparation
    record is the canonical input.
  - README fixes: the root README and `ImageSets/README.md` no longer say
    that DIV2K is unused. `ImageSets/README.md` now allows the reviewed
    `data/raise` and `data/div2k` CSV files in git.
- The Worker did plan step 2, the source step. Commit `a7feb63`. The full
  record is [docs/archive/source-verification.md](../../archive/source-verification.md).
  - `LSB-Dataset-Generator/lsb_raw.py`: shared RAW settings and decode path.
    `lsb_prepare.py` uses it. The preparation settings did not change.
  - `LSB-Dataset-Generator/lsb_source.py`: full-photo source API for `nef`
    and `png8` sources. It checks sizes against the manifest, has an
    `OMP_NUM_THREADS=1` guard for loader workers, and has a pre-demosaic
    RAW-noise hook that is disabled.
  - Grids from the manifests: `topleft_crop` (legacy, 512) and
    `overlap_last` (512 and 256).
  - Check script: `scripts/check_sources.py`. Tests:
    `tests/test_lsb_source.py`.
- The Worker did plan step 3. Commit `967cfd4`. The full record is
  [docs/active/settings-and-seeds.md](../../active/settings-and-seeds.md).
  - `LSB-Dataset-Generator/lsb_settings.py` holds the settings draws moved
    from `lsb_batch.py`. The logic is unchanged, and `lsb_batch.py` imports
    it.
  - It keeps the legacy seed key
    `SeedSequence([seed, cover_index, variant, stream])`: stream 0 is for
    settings and stream 1 is for embedding. It adds a versioned on-the-fly
    seed key.
  - Check script: `scripts/check_settings.py`. Tests:
    `tests/test_lsb_settings.py`.
- The Worker did plan step 4. Commit `c2bdb53`. The full record is
  [docs/archive/embed-core-verification.md](../../archive/embed-core-verification.md).
  - `lsb_engine.py` now has one internal embed core. `embed()` keeps its
    signature and returns `(stego, mask, EmbedResult)` unchanged.
  - New `embed_fast()` returns `(stego, metadata dict, changed unit indices
    or None)`. It has no mask. It computes changed indices only when asked,
    and keeps the `units_changed` count. It always copies the clean carrier;
    there is no in-place option.
  - Check script: `scripts/check_embed_fast.py`. Tests:
    `tests/test_embed_fast.py`.
- The Worker did plan step 5. Commit `fc87b9c`. The full record is
  [docs/archive/mode1-verification.md](../../archive/mode1-verification.md).
  - `LSB-Dataset-Generator/lsb_mode1.py` provides `make_sample()` and
    `replay_legacy()` for clean or stego samples. It uses `lsb_source`,
    `lsb_settings`, and `lsb_engine.embed_fast`; it does not copy their
    logic.
  - `lsb_source.py` gained `tile_at()` for direct tile lookup, sharing grid
    checks with `iter_tiles`.
  - Check script: `scripts/check_mode1.py`. Tests:
    `tests/test_lsb_mode1.py`.
- The Worker did plan step 6. Commit `7e89fec`. The full record is
  [docs/archive/cache-verification.md](../../archive/cache-verification.md).
  - Cache code: `LSB-Dataset-Generator/lsb_cache.py`,
    `scripts/build_cache.py`, `scripts/check_cache.py`, and
    `tests/test_lsb_cache.py`.
  - The VM cache is `/srv/aai3001/otf-cache/v1/`. It has 1,900 uint8
    `(H, W, 3)` `.npy` files (one full clean photo each), 1,900 JSON records,
    and one format header. Writes are atomic and resumable, with per-entry
    locks. Group `aai3001` can write the folder.
  - The strict loader checks the header, identity, decoder fingerprint, shape,
    and master and array SHA-256. The fingerprint includes RAW settings,
    rawpy 0.27.1, LibRaw 0.22.1, NumPy 2.5.3, Pillow 12.3.0, and the
    disabled noise hook. It never silently decodes a bad entry. Hash checks
    can be disabled only after a session check.
- The Worker did plan step 7. Commit `970750a`. The full record is
  [docs/archive/torch-loader-verification.md](../../archive/torch-loader-verification.md).
  - `LSB-Dataset-Generator/lsb_torch.py` provides `RunSettings`, `Corpus`,
    `TrainingImages` with `set_epoch`, `FixedPairs` for validation/test, and
    `training_loader`. It has tests in `tests/test_lsb_torch.py` and a check
    script at `scripts/check_torch_loader.py`.
  - The default source is `decode`; `cache` uses a cache root. Each worker
    loads one whole photo, makes all clean and stego tiles with keyed OTF
    seeds, and shuffles them in a bounded uint8 buffer. The default buffer
    is 1,024 images per worker.
  - Images are float32 CHW values in the 0-255 range, with keyed flips and
    90-degree rotations. Validation and test use fixed `overlap_full_mix`
    with both sources and the run1 mix, independent of training settings.
    `legacy_v6` remains for a v3c comparison re-run.
  - Loader defaults are `prefetch_factor=4` and `in_order=True`.
- The Worker committed the run1 records in commit `8c4e549`.
  - `data/raise/run1_labels.csv.gz` is 3,307,725 bytes, and
    `data/raise/run1_metadata.csv.gz` is 3,180,260 bytes. Both use `gzip -n`
    for deterministic output.
  - The originals are `labels.csv` (19,806,781 bytes, SHA-256
    `3195008b...b511b`) and `metadata.csv` (15,485,145 bytes, SHA-256
    `fdc6d413...f0be3`). PC and VM copies are identical.
  - Decompressed hashes equal the originals. A second compression produces
    the same gzip bytes. The record is also in
    `docs/active/frozen-source-records.md` and the root README.
  - `masks/` and `records/` are not kept in git. Masks can be regenerated.
- The Worker did plan step 8. Commit `f3dca17`. The full record is
  [docs/archive/notebook-otf-verification.md](../../archive/notebook-otf-verification.md).
  - Notebook 04 now uses `V3_DATA=otf` by default, with `png` retaining the
    `raise512-run1` path. OTF settings include `V3_SOURCE`,
    `V3_CACHE_ROOT`, `V3_WORKERS`, `V3_PREFETCH`, `V3_TILE_SIZE`,
    `V3_RUN_SEED`, `V3_EVAL_PROFILE`, `V3_TRAIN_PHOTO_FRACTION`, and
    `V3_SOURCES`. The defaults are decode, 6 workers, prefetch 4, tile 512,
    run seed 0, and `overlap_full_mix`; `legacy_v6` is available.
  - The default OTF run name ends in `_otf`, so it cannot resume a PNG
    checkpoint. Settings are saved in `data_run_settings.json`.
  - New module: `LSB-Dataset-Generator/lsb_notebook.py`. Notebook 02 cells 2
    and 6 read self-contained OTF metadata. The PNG path is unchanged. Saved
    outputs and the executed record notebooks are unchanged.
  - Workers yield small groups, and the main process joins groups from all
    workers into each batch. The loader yields uint8; the notebook converts
    it to float32 on the device. A callback calls `set_epoch` at each epoch.
- The user ran the PC full OTF v3c run. The executed record is
  `notebooks/runs/04_v3c_otf_pc_2026-10-13.ipynb` (commit `728db2b`).
  Training used the RX 9070 XT, decode, 8 workers, and fixes 1-3.
  Training epochs took 5:02-5:05, and whole epochs with validation took
  7:19-7:24. The user stopped the diagnostics cell after more than 45 min.
- The VM was preempted once during a Worker test on 2026-10-13 and was off
  overnight. The user restarted it. No user data was lost.
- On the user's decision, the Orchestrator deleted the image folders of
  `Datasets/raise512-run1` on the VM: `clean/`, `stego/`, and `masks/`, with
  49,796 PNG files in each folder. It kept 99 MB: `labels.csv`,
  `metadata.csv`, `done.jsonl`, `run.log`, `run_config.yaml`, and
  `splits/split_seed0.csv`. The split hash equals
  `data/raise/split_seed0.csv`.

- The user ran the first full OTF v3c run on the VM L4. The executed record
  is `notebooks/runs/04_v3c_otf_2026-10-13.ipynb` (commit `973f010`, SHA-256
  `3f167964...4bc9`, equal to the VM file). Settings were `V3_DATA=otf`,
  `QUICK=False`, cache `/srv/aai3001/otf-cache/v1`, 6 workers, prefetch 4
  groups, tile 512, run seed 0, `overlap_full_mix`, 10 epochs, batch 32,
  and v3c (clip 3, GroupNorm). Run name:
  `baseline_kv_xunet_v3_clip3_group_otf`.
- The user ran a PC slow-run diagnosis and stopped the run. The Worker made
  no code change. The GPU-only result was normal. The diagnosis found small
  groups of 4 with prefetch 4, a main-process join and pin cost, and 16 main
  torch threads as causes. Eight idle persistent validation workers also held
  memory.
- The Worker applied fixes 1-3 in commit `9c3bf0b`: one main torch thread;
  four non-persistent evaluation workers; and prefetch counted in images,
  with 128 images per worker. `V3_PREFETCH` remains a group-count override.

### Found

- Clean test tiles with high `prob_stego`, out of 7,433 clean tiles:

  | Model | >= 0.5 | >= 0.9 | >= 0.99 | >= 0.999 |
  | --- | ---: | ---: | ---: | ---: |
  | V3a | 63 | 57 | 33 | 18 |
  | V3b | 78 | 58 | 54 | 47 |
  | V3c | 57 | 12 | 5 | 2 |

- The old "about 55" v3b figure in the noisy-cover plan matches 54 at the
  0.99 threshold. The >= 0.5 counts match the false alarms at 0.5 in
  `results-v3.md`.
- One run and one seed per model, so these counts are not a measured
  difference between models.

#### On-the-fly generation benchmark

Latency per output image (median / p90; p90 means 90% of images were
faster):

| Path | Median (ms) | p90 (ms) |
| --- | ---: | ---: |
| Stored pair (read clean and stego PNGs) | 5.835 | 6.204 |
| On-the-fly pair (read clean PNG, embed) | 3.279 | 4.337 |
| Embed only, per stego tile | 0.668 | 2.658 |
| Full generator API, per stego tile | 0.736 | 2.770 |

- PNG decode is the main cost. Embedding is not.
- DataLoader throughput, images/s, median of 3 trials:

  | Workers | Stored | On the fly |
  | ---: | ---: | ---: |
  | 4 | 405.9 | 559.7 |
  | 8 | 733.7 | 854.8 |
  | 16 | 962.2 | 1022.4 |

- Training needs about 340 images/s on the PC (10.7 it/s x 32) and about
  215 images/s on the VM (6.7 it/s x 32).
- The VM estimate is about 511 images/s on 8 vCPUs. This value is scaled
  from the PC. It is an estimate, not a VM measurement.
- The slowest settings store one bit per carrier value. Random placement,
  0.5 bpp (bits per pixel), and match, with one bit per carrier value (k=1,
  or plane mode; plane k=8 was measured): about 5.8 ms median. At k=1,
  lowest and plane are the same embedding. Random placement, 0.5 bpp, and
  replace take about 4.3 ms.
- Correctness: 32 of 32 sampled tiles were byte-identical to the stored stego
  PNGs. The test used the original cover index, `SeedSequence([seed, index,
  variant, 0])` for the settings, and `[..., 1]` for the embedding.
- The embed engine uses vectorized NumPy. It always builds the mask and the
  `EmbedResult` object. Training does not need them.
- Caveats: the on-the-fly test reused one decode for the clean and stego
  twins in one batch. The notebook shuffles images independently, so it does
  not get this saving. The cache was warm, and no GPU training ran at the
  same time.

#### Embed -> tile benchmark on whole photos

Tile grid audit over all 1,000 photos and 49,796 tiles:

- Each photo has a complete, non-overlapping grid of 512 x 512 tiles from the
  top-left corner. Edge pixels outside the grid are not in any tile.
- Stitching the stored clean tiles rebuilds each cropped photo.

  | Camera | Grid, rows x cols (landscape / portrait) | Photos (landscape + portrait) | Tiles |
  | --- | --- | ---: | ---: |
  | D7000 | 6 x 9 / 9 x 6 | 570 + 144 | 38,556 |
  | D90 | 5 x 8 / 8 x 5 | 194 + 84 | 11,120 |
  | D40 | 3 x 5 / 5 x 3 | 6 + 2 | 120 |

- One NEF decode (D90, 4310 x 2868, project settings) took 683.8 ms. This is
  one measurement.

Per-photo latency, median / p90 in ms, n=36 photos. The three load sources
are: stitch the stored clean tile PNGs, read one `.npy` array file per photo,
or read one full-photo PNG (compression level 1).

| Step | Stitch | `.npy` | Full PNG |
| --- | ---: | ---: | ---: |
| Load | 292.73 / 304.37 | 2.66 / 2.78 | 327.58 / 344.52 |
| Embed | about 50-54 / 196-200 | about 50-54 / 196-200 | about 50-54 / 196-200 |
| Cut tiles and labels | about 147 / 155 | about 147 / 155 | about 147 / 155 |
| Total | 489.51 / 644.11 | 205.05 / 348.89 | 523.04 / 673.87 |
| Total per output tile (median) | 4.619 | 1.899 | 4.940 |

- File size per photo (median): `.npy` 40.50 MiB, full PNG 19.86 MiB (p90
  22.89 MiB). A full `.npy` cache would be about 36.5 GiB.
- Slowest embed (D7000, match), median per photo: random placement and
  0.5 bpp with one bit per carrier value took about 586 ms (lowest k=1) and
  578 ms (plane k=8). Lowest k=8 took about 76 ms. Contiguous and stride
  placement at 0.5 bpp took about 118-126 ms, and about 14 ms at 0.05 bpp.
- Placement on whole photos: contiguous placement makes horizontal bands;
  about 6-18 of 54 tiles get payload. Stride placement is also localized.
  Random placement reaches all 54 tiles.
- DataLoader output, tiles/s. One item is one photo with its clean and
  stego tiles. Prefetch factor 1.

  | Workers | Stitch | `.npy` |
  | ---: | ---: | ---: |
  | 4 | 437.95 | 609.46 |
  | 8 | 738.65 | 999.91 |
  | 16 | 977.10 | 1265.04 |

- Peak resident memory (RSS) was about 1.1 GiB per worker. This value
  includes shared pages.
- All rows are above the PC target of 340 tiles/s (images/s), also at 4 workers. The
  VM estimate is about 489 tiles/s (stitch) and about 633 tiles/s (`.npy`).
  These values are scaled from the PC, not measured on the VM.
- Label statistics, 36 photos with mixed settings:
  - 1,286 of 1,863 tiles from stego photos (69.03%) have label 1. The
    other 577 tiles correctly have label 0: no pixel value changed.
  - Random placement: 100% label 1. Stride: 42-83%. Contiguous: 14-24%.
  - Over all clean and stego outputs, about 34.5% of tiles have label 1.
  - The smallest changed-pixel count in a label-1 tile was 413. No label-1
    tile had 10 or fewer changes. This is a sample result, not a guarantee.
- Correctness:
  - 1,863 of 1,863 stitched and recut clean tiles were byte-identical to the
    stored tiles.
  - Two whole-photo embeds with the same seed were byte-identical for 36 of
    36 photos.
  - The stored stego tiles were embedded per tile. They are not expected to
    match whole-photo embedding.
- Capacity: with one bit per carrier value and stride 16, the maximum is
  about 0.1875 bpp. The original `_draw_settings` limits the bpp draw to the
  capacity.

#### Master-source benchmark: NEF versus 16-bit PNG

Exactness against the stitched stored clean tiles, 42 photos:

| Conversion to 8 bits | Byte-identical photos |
| --- | ---: |
| NEF -> 8-bit crop | 42 / 42 |
| 16-bit PNG -> high byte (`x >> 8`) | 42 / 42 |
| 16-bit PNG -> `round(x/257)` | 0 / 42 |
| 16-bit PNG -> `floor(x/257)` | 0 / 42 |
| 16-bit PNG -> `round(x/256)` | 0 / 42 |

- Each mismatch was +/-1 in one channel value. For example, `round(x/257)`
  changed 390,661,093 of 1,575,223,296 values.

Load time per photo, median / p90 in ms, on one CPU:

| Camera | NEF -> 8-bit crop | 16-bit PNG decode + shift + crop (OpenCV default) |
| --- | ---: | ---: |
| D7000 | 827.96 / 871.48 | 388.34 / 396.40 |
| D90 | 593.33 / 622.39 | 298.07 / 304.61 |
| D40 | 286.29 / 308.90 | 144.97 / 146.92 |

- rawpy 0.27.1 with LibRaw 0.22.1 uses OpenMP threads. Without
  `OMP_NUM_THREADS=1`, one decode uses about 3 CPUs. One D7000 probe took
  422 ms wall time this way. Set `OMP_NUM_THREADS=1` in the loader workers.
- The OpenCV default PNG encoder was the fastest that was tested. OpenCV 5
  writes its default PNG at level 1 with the RLE strategy. An explicit
  level 1 uses a different strategy and decoded slower (D7000 607.86 ms).
- Pillow reads 16-bit RGB PNG only as 8-bit. PyPNG keeps 16 bits, but it
  took about 14 s per D7000 photo.

File size per photo, median MiB:

| Camera | NEF | 16-bit PNG (OpenCV default) |
| --- | ---: | ---: |
| D7000 | 20.28 | 76.25 |
| D90 | 10.27 | 56.19 |
| D40 | 5.34 | 28.89 |

- Extrapolated to 1,000 photos, the 16-bit PNG set is about 68.65 GiB
  (73.71 GB). The NEF set is about 17 GB. An 8-bit cropped PNG cache would
  be about 19.39 GiB. This is an estimate.

DataLoader, 4 workers, full embed -> tile pipeline, output tiles/s (median
of 2 trials), against the PC target of 340 tiles/s:

| Source | Tiles/s | Result |
| --- | ---: | --- |
| NEF, direct decode | 269.58 | Fail (79% of target) |
| 16-bit PNG, explicit level 1 | 332.53 | Fail |
| 16-bit PNG, OpenCV default | 393.02 | Pass |
| Stitch stored 8-bit tiles | 432.87 | Pass |

- Peak RSS was about 1.5 GiB per worker.
- These are PC measurements, not Kaggle measurements.
- One-time conversion with 4 processes, extrapolated from 42 photos to
  1,000: NEF -> 8-bit decode only takes about 4.1 min. NEF -> 16-bit plus PNG
  encode (OpenCV default) takes about 6.9 min. The encode time of the 8-bit
  cache was not measured.

#### ROCm caution, cause found

Status: cause found on 2026-10-13. The first report is kept below as history.
The result of the float64 check follows it.

First report:

- The ROCm-Works team reported this on 2026-10-13. In a fixed-seed loss
  check, the setup with `MIOPEN_GEMM_ENFORCE_BACKEND=5` gave a different
  loss path from all earlier runs:

  | Step | Loss with the variable | Loss in earlier runs |
  | ---: | ---: | ---: |
  | 50 | 0.367 | about 0.11 |
  | 200 | 0.0038 | 0.0027 |

- The ROCm switch in notebook 04 sets this variable by default.
- `MIOPEN_GEMM_ENFORCE_BACKEND=5` only selects the library that MIOpen uses
  for its matrix multiplications: hipBLASLt instead of rocBLAS.
- The conv1 weight-gradient fix (unfold plus batched matrix multiplication)
  runs in PyTorch, not in MIOpen. Thus the fix itself cannot change the
  forward pass. It was open at this point whether backend 5 changes any
  result.
- ROCm-Works also added an end-to-end inference check. It uses the same
  weights and inputs in three setups: the variable unset, the variable set
  to 5, and float64 on the CPU. It reports the maximum difference in
  `prob_stego`.
- At this point the cause was not known. It could be normal run-to-run
  variation, or a wrong gradient in a layer.
- The ROCm-Works worker checked each convolution layer against a float64
  reference, with and without the variable.
- Until that check was done, PC results from notebook 04 with the ROCm
  switch on were not final. This applies to the v3b and v3c PC runs. V3a
  ran on the VM L4 with CUDA, so it is not affected.
- New runs of this team run on the VM only.

Result of the ROCm-Works float64 check (N=8). Values are the maximum
relative error against float64:

| Quantity | Layers | Variable unset | Variable = 5 | Result |
| --- | --- | --- | --- | --- |
| Forward output | KV filter, conv1-conv5 | <= 8.4e-7 | <= 8.4e-7 | Correct |
| Data gradient and bias gradient | All layers | <= 8e-7 | <= 8e-7 | Correct |
| Weight gradient | conv1 / conv2 | 1.6e-5 / 9.4e-6 | 1.6e-5 / 9.4e-6 | Correct |
| Weight gradient | conv3 / conv4 / conv5 (1x1) | 4e-6 / 2e-6 / 1e-6 | 0.81 / 0.74 / 0.94 | Wrong with the variable |
| Weight gradient, conv1 unfold fix (both versions) | conv1 | 1.1e-6 / 5.3e-7 | 1.1e-6 / 5.3e-7 | Correct |

- Cause: with the variable unset, MIOpen selects Winograd for the 1x1
  weight gradients. With the variable set to 5, MIOpen's GEMM-based 1x1
  weight-gradient solver `GemmWrw1x1_stride1` becomes the fastest choice.
  On hipBLASLt, this solver returns wrong values.
- The conv1 unfold fix does not use MIOpen for the weight gradient, so it
  is correct in both settings.
- The loss check agrees. With the variable, the loss at step 50 was
  0.15-0.52 and changed between runs. Without it, the loss was about 0.11 in
  every setting.
- Forward pass and inference are not affected. With the same weights and 64
  images in evaluation mode, the maximum absolute logit difference against
  float64 was 1.2e-8 (variable unset) and 1.8e-8 (variable = 5). The
  `prob_stego` difference between the two settings was at most 3.0e-8.
- Thus the v3b and v3c test metrics are correct measurements of the trained
  models. Only the training can be affected.
- Likely extent (ROCm-Works analysis; confirmation is Pending):
  - The MIOpen find-db (the record of tuned solver choices) in the container
    already had records for conv3-conv5. At batch 32, the recorded fastest
    solver is Winograd, which is correct.
  - The last partial batch of each epoch has 20 images
    (69,780 = 2,180 x 32 + 20). For 20 images, the recorded fastest solver
    for conv3 and conv4 is `GemmWrw1x1_stride1`.
  - Thus most likely about 1 step per epoch, about 10 of about 21,800
    steps, had wrong conv3 and conv4 weight gradients.
  - Limit: the find-db replay shows which solver the stored records select.
    It cannot prove which solver ran in v3b and v3c, because the training
    logs did not record the MIOpen solver choices.
  - The effect is probably small, but v3b and v3c are not clean baselines.

Confirmation by ROCm-Works (2026-10-13):

- Find-db replay. ROCm-Works used a read-only copy of the `aai3001` user
  database, with `MIOPEN_GEMM_ENFORCE_BACKEND=5` set as in v3b and v3c.
  Weight-gradient solver and relative error:

  | Batch | Layer | Selected solver | Relative error, variable = 5 | Relative error, variable unset | Result |
  | ---: | --- | --- | --- | --- | --- |
  | 32 | conv3, conv4, conv5 | Winograd | about 1e-6 to 1e-5 | - | Correct |
  | 20 | conv3 | `GemmWrw1x1_stride1` | 0.93 | 2.9e-6 | Wrong with the variable |
  | 20 | conv4 | `GemmWrw1x1_stride1` | 0.33 | 9.0e-7 | Wrong with the variable |
  | 20 | conv5 | Winograd | - | - | Correct |

  Batch 20 is the last partial batch of each epoch
  (69,780 = 2,180 x 32 + 20).
- Most likely, the last step of each epoch had wrong conv3 and conv4 weight
  gradients: 10 of 21,810 training steps in each of v3b and v3c.
  - The forward pass, all data gradients, and all other weight gradients
    were correct on every step.
  - Validation and test use only the forward pass. Thus their metrics are
    correct measurements.
  - Limit: the replay shows which solver the records select. The training
    logs did not record the MIOpen choices.
  - The database entries for batch 20 did not change after v3b and v3c ran.
    A backup from the previous morning has the same entries.
- Cause in the MIOpen 3.6.1 source (`rocm-libraries` commit `39d8d603`):
  - `gemm_v2.cpp` maps backend 5 to hipBLASLt.
  - With rocBLAS, `CallGemmStridedBatchedSequential` loops over the images
    and adds each result into the same weight gradient.
  - With hipBLASLt, it makes one batched call with output stride 0. Thus the
    results of the images are not added correctly. The code has a "todo"
    comment at this point.
  - The result is wrong for batch >= 2: measured relative error 0.83-1.06 at
    batch 2, 8, 20, and 32. It is correct for batch 1 (2.5e-6).
- Step time without the variable, on the PC at batch 32, median:

  | Data | No fix | Current conv1 fix | v2 conv1 fix |
  | --- | ---: | ---: | ---: |
  | Synthetic, 3 fresh runs | 188 ms | 102.8 ms | 87.8 ms |
  | Real, 200 steps | - | 99.3 ms (322 images/s) | 85.4 ms (375 images/s) |

  With the current fix, the removal of the variable costs almost no speed.

ROCm-Works v2 conv1 fix proposal (2026-10-13):

- The proposal is a drop-in replacement for
  `notebooks/rocm_conv1_wgrad.py`. It keeps class `UnfoldWgradConv2d`, the
  same `from_conv` call, and the same `state_dict` keys. No notebook change
  is needed.
- The weight-gradient matrix multiplication reads strided views of the
  unfold buffer. This removes a copy of about 10 ms per step. Defaults
  change from 4 images x piece 4096 to 8 images x piece 1024.
- The docstring says not to set
  `MIOPEN_GEMM_ENFORCE_BACKEND=5`. The constructor rejects `groups != 1` and
  padding other than `"same"`.
- The self-check compares with a float64 CPU reference at batch sizes 1, 20,
  and 32. Forward and data gradients are unchanged.
- Maximum relative error against float64 for weight / bias gradients:

  | Batch | Weight | Bias |
  | ---: | ---: | ---: |
  | 1 | 4.8e-7 | 1.6e-7 |
  | 20 | 6.1e-7 | 1.6e-7 |
  | 32 | 5.8e-7 | 1.5e-7 |

- Fixed-seed loss at steps 1 / 50 / 100 / 200:
  - No fix: 0.7302 / 0.1119 / 0.0111 / 0.00272.
  - Proposal run 1: 0.7302 / 0.1140 / 0.0115 / 0.00280.
  - Proposal run 2: 0.7302 / 0.1138 / 0.0113 / 0.00273.
  - The difference comes from GPU summation order and is the same size as
    the current version.
- PC speed at batch 32 with the variable unset:

  | Data | No fix | Current fix | Proposal |
  | --- | ---: | ---: | ---: |
  | Synthetic, 3 fresh runs | 188 ms median | 102.8 ms median | 91.6 / 87.1 / 87.1 ms (an earlier set of 3 runs: median 87.8 ms) |
  | Real data, 200 steps | - | 99.3 ms (322 images/s) | 85.6 ms (374 images/s) |

  At batch 20, the proposal took 56.9 ms. Peak GPU memory for one batch-32
  step was 2.91 GB, the same as the current version. The fix has no effect
  on the VM because it is active only on ROCm.
- User decision (2026-10-13): approved. The Worker implemented the proposal
  after step 3 and ran its self-check on the PC GPU.
- ROCm-Works tried a Triton kernel for the conv1 and conv2 weight gradients
  on the PC GPU in container `aai3001-perf`. It was correct, but gave no clear
  gain over committed fix `60f5c17`. An early version was about 4 ms per step
  faster. There is no repository proposal. The committed fix stays, and the
  PC GPU window is closed.
- Worker commit `60f5c17` replaced `notebooks/rocm_conv1_wgrad.py` with the
  proposal file, unchanged. Its SHA-256 is
  `afe4a475322d0660a1877c08d1d6398d6ffa22eb55133c7339982c23b11f2301`, the
  same as the proposal. The class, `from_conv` parameters, and `state_dict`
  keys are unchanged. Notebook 04 and other docs are unchanged.
- The PC GPU self-check ran once in the `aai3001` container with no MIOpen
  variables set. Maximum relative error against float64, weight / bias:

  | Batch | Weight | Bias | Result |
  | ---: | ---: | ---: | --- |
  | 1 | 5.561e-7 | 1.550e-7 | PASS |
  | 20 | 6.125e-7 | 1.619e-7 | PASS |
  | 32 | 6.430e-7 | 1.534e-7 | PASS |

  All pass the rule `<= 1e-5`. The self-check covers only weight and bias
  gradients. Forward and data gradients still use MIOpen and were checked
  earlier by ROCm-Works.

#### Frozen photo and tile records (step 1)

| File | Data rows |
| --- | ---: |
| `data/raise/photo_manifest.csv` | 1,000 |
| `data/raise/legacy_tile_manifest.csv` | 49,796 |
| `data/div2k/photo_manifest.csv` | 900 |
| `data/div2k/split_seed0.csv` | 900 |

- There are 1,900 unique photo IDs and indices: RAISE 0-999 (the full sorted
  list) and DIV2K 1000-1899. All 1,900 master paths exist on the VM. The
  original cover IDs and indices are kept.
- RAISE split, in photos:

  | Camera | Train | Val | Test |
  | --- | ---: | ---: | ---: |
  | D7000 | 500 | 107 | 107 |
  | D90 | 195 | 42 | 41 |
  | D40 | 6 | 1 | 1 |
  | Total | 701 | 150 | 149 |

  The join to `data/raise/split_seed0.csv` gave no mismatch. The
  Orchestrator also checked it: 0 mismatches over 49,796 tiles.
- DIV2K split, seed 0, PCG64 permutation:

  | Split | Photos | Tiles at 512 | Tiles at 256 |
  | --- | ---: | ---: | ---: |
  | Train | 680 | 8,276 | 32,216 |
  | Val | 120 | 1,456 | 5,720 |
  | Test | 100 | 1,228 | 4,808 |

- Overlap totals at 512 / 256: RAISE 65,184 / 243,120; DIV2K 10,960 /
  42,744. These match the plan values.
- Nine unit tests pass on the VM.

#### Source verification (step 2)

- All 1,000 NEFs decoded on the VM. All 49,796 stored clean tiles are
  byte-identical to the legacy crop (0 mismatches).
- All 900 DIV2K decodes equal the PNG pixels and the manifest sizes.
- Overlap tile counts at 512 / 256: RAISE 65,184 / 243,120; DIV2K 10,960 /
  42,744.
- Six camera and orientation samples have full pixel coverage and no
  padding at both tile sizes.
- 9 source tests pass, and 18 of 18 pass together with the manifest tests.
- Versions: rawpy 0.27.1, LibRaw 0.22.1, NumPy 2.5.3, Pillow 12.3.0.
- Wall time was 611.7 s with 8 processes. This includes reading each stored
  clean PNG.
- NEF decode per photo on the VM, median / p90 in s:

  | Camera | Median | p90 |
  | --- | ---: | ---: |
  | D7000 | 3.775 | 3.981 |
  | D90 | 2.884 | 3.091 |
  | D40 | 1.403 | 1.470 |

- This is about 4.5-5 times slower than the earlier one-core PC measurement
  (D7000 0.83 s). The VM CPU is an Intel Xeon at 2.20 GHz with 4 cores x 2
  threads (8 vCPUs). The check also read PNG files at the same time. Thus
  this is not a clean speed measurement.

#### Settings and seeds (step 3)

- OTF key v1 fields, in order: version, purpose (train 1 / val 2 / test 3),
  seed, epoch, carrier (tile 1 / photo 2), photo index, grid ID (none 0 /
  `topleft_crop` 1 / `overlap_last` 2), tile size, tile row, tile column,
  variant (from 1), and stream (settings 0 / embedding 1).
- Training uses `run_seed` and epoch. Epoch 0 is used for fixed variants.
  Validation and test always use seed 1 and epoch 0, so their embedding does
  not change between runs. A photo carrier in mode 2 sets grid, size, row,
  and column to 0.
- Lower-to-fit applies only to the new OTF path. It draws the requested size,
  lowers it to the largest payload that fits, and records requested and
  effective bpp and bytes, capacity, and a clipped flag. A carrier that
  cannot hold 1 byte fails.
- The legacy path is unchanged. It clips range upper bounds and fails on
  oversized fixed values.
- Legacy regression on the VM: all 49,796 stego rows were redrawn. There
  were 0 mismatches in all 11 fields (bit mode, k, method, placement, start,
  start_random, stride, payload kind, payload bytes, size kind, and size
  value). Placement counts were random 16,564, stride 16,623, and contiguous
  16,609. Redraw time was 13.2 s.
- For the 33,232 non-random rows, the start check used the engine's own
  payload and unit-selection functions. It did not embed images.
- Real `generate()` on covers 000003 (random), 000004 (stride), and 000005
  (contiguous) produced stego PNG files byte-identical to the stored files.
- 60 tests pass on the VM: 26 new source/manifest/settings tests and 34
  existing batch/engine/prepare tests.

#### Embed core verification (step 4)

- 20,000 random cases covered all 96 bit-mode / k / method / placement
  combinations (at least 208 cases each), shapes 256 x 256, 512 x 512,
  odd shapes, and one full 3280 x 4948 photo. Pixels, masks, metadata, RNG
  state after the call, and changed indices matched between an old-engine
  copy, the new `embed()`, and `embed_fast()`.
- 503 stored legacy stego rows from all 12 mode/method/placement groups had
  0 mismatches.
- 63 unit tests pass.
- One-core median time per call, in ms:

  | Input and placement | `embed` | `embed_fast` | `embed_fast` with changed indices |
  | --- | ---: | ---: | ---: |
  | 512 tile, contiguous | 5.45 | 4.93 | 5.83 |
  | 512 tile, random | 8.68 | 8.13 | 9.08 |
  | 256 tile, contiguous | 1.54 | 1.41 | 1.67 |
  | Full D7000 photo, contiguous | 424 | 377 | 433 |
  | Full D7000 photo, random | 880 | 833 | 890 |

  Without changed indices, `embed_fast` is about 6-11% faster. With indices,
  it is slower than `embed()`. At 512, it gives about 203 contiguous tiles/s
  per core and 123 random tiles/s per core.
- A boolean mark array was rejected as a sort replacement. It gave the same
  output but was slower in all 6 tested cases. For example, full photo,
  dense: sort 92.5 ms versus mark 183.6 ms. The sort is unchanged.

#### Mode 1 verification (step 5)

- First gate: 500 legacy pairs from 12 NEFs, covering all 6 camera and
  orientation groups. Placement counts were contiguous 166, stride 163, and
  random 171. There were 0 clean or stego pixel mismatches. Wall time was
  10.1 s.
- Full gate: all 49,796 clean and stego pairs from the 1,000 NEFs, each
  decoded once, with 8 processes and `OMP_NUM_THREADS=1`. There were 0
  mismatches. Wall time was 724.3 s.
  - Summed worker time: NEF decode 3,534.8 s (about 3.5 s per photo), cut
    plus embed 209.1 s (about 4.2 ms per pair), and PNG read plus compare
    2,025.6 s. These phases overlap, so they do not add to wall time.
- OTF checks on 8 photos (each RAISE camera/orientation group and DIV2K
  0001/0002), with `overlap_last` at 512 and 256, including edge tiles:
  - Repeat calls were identical in bytes and metadata.
  - 32 validation/test cases were unchanged when `run_seed` and epoch
    changed. 16 train cases changed with both.
  - 16 fixed-variant cases were identical at epochs 0, 1, 5, and 100.
  - 16 forced lower-to-fit cases with request 10 bpp recorded
    `clipped=True`; for example, a 512 tile had effective 0.00299 bpp and
    98 bytes.
  - Two serial orders and 4 processes gave identical output: 816 sample
    calls and 0 mismatches.
- 68 unit tests pass.

#### Full-photo cache verification (step 6)

- Clean decode speed on the VM with no other work running. Values are
  photos/s at 1 / 4 / 8 processes, with 8 photos per camera:

  | Camera | 1 process | 4 processes | 8 processes |
  | --- | ---: | ---: | ---: |
  | D7000 | 0.44 | 1.58 | 1.79 |
  | D90 | 0.56 | 2.05 | 2.31 |
  | D40 | 1.15 | 3.86 | 4.14 |

  One process took 2.26 s per D7000 photo, compared with 0.83 s on the PC.
  Eight processes give only about 13% more than four (4 cores x 2 threads).
- Full build: 1,900 photos, 8 processes, 554.7 s. Size is 52.76 GB
  (49.15 GiB). Free disk after the build was about 50.1 GiB.
- All 1,900 array hashes and master hashes match their records (364.8 s).
- All 49,796 clean and stego pairs from the cache, without NEF decoding,
  match the stored PNGs (382.6 s).
- Cache read plus cut plus embed for all 49,796 pairs, without reference PNG
  reads, took 154.8 s with 8 processes: about 322 pairs/s (about 643
  images/s). Summed time was 917 s for cache reads and 316 s for cut plus
  embed. Cache reads are the larger part.
- One-CPU random reads of all 1,900 photos: normal `np.load` gave 14.45 and
  14.42 photos/s in two passes. mmap plus copy gave 12.75 and 10.97. Disk
  reads were about 204-206 MB/s. RAM is 31 GiB, less than the 52.8 GB cache.
  About 33-49% of reads likely came from the page cache. This is an
  estimate because there was no `drop_caches` permission.
- One random 512 tile per random photo through mmap gave 79.9 tiles/s and
  read 3.11 times the tile bytes from disk.
- VM monitoring showed about 200 MiB/s disk reads. A 200 GiB balanced
  persistent disk gives about 196 MiB/s by Google's formula
  (`140 + 0.28 x 200`). A same-size standard persistent disk gives about
  24 MiB/s (`0.12 MiB/s per GiB`) and is not suitable.

#### Torch loader verification (step 7)

- Decode and cache produced 0 mismatches for all tiles of 8 photos (6 RAISE
  camera/orientation groups and 2 DIV2K), with train and validation keys at
  both 512 and 256. The result was unchanged with 0 or 4 workers.
- The legacy profile checked 2,000 pairs across all 49,796 and had 0
  mismatches against the stored PNGs. 79 tests pass.
- Images per epoch (clean and stego), train / validation / test:

  | Tile size | Train | Validation | Test |
  | ---: | ---: | ---: | ---: |
  | 512 | 107,900 | 22,476 | 21,912 |
  | 256 | 405,144 | 84,408 | 82,176 |

  The default has 1,381 train photos: 701 RAISE and 680 DIV2K. The RAISE
  parts add up to 65,184 and 243,120 tiles at 512 and 256.
- Loader-only speed, batch 32, after buffer fill, in images/s:

  | Source and tile size | 4 workers | 6 workers | 8 workers |
  | --- | ---: | ---: | ---: |
  | Cache, 512, prefetch 1 | 303 | 358 | 418 |
  | Cache, 256, prefetch 1 | 1,068 | 1,403 | 1,326 |

  Decode at 512 with 8 workers gave 129 images/s at prefetch 1, 194 at
  ordered prefetch 4, 183 at ordered prefetch 8, and 202 at unordered
  prefetch 4. Decode at 256 with 8 workers and prefetch 1 gave 309.
  Unordered loading keeps sample bytes but not batch order, so it is not the
  default.
- Joint L4 check with the v3c model, batch 32, 512 cache, and prefetch 4:
  unordered 8 workers gave 268 images/s; ordered 8 workers gave 267; ordered
  4 workers gave 267. GPU use was 100%, 99.8%, and 100%; loader wait was
  0.4%, 0.6%, and 0.2%; RAM was 17.1, 17.3, and 9.4 GB. The GPU is now the
  limit. A 512 epoch takes about 6.7 min.
- Correction: notebook 04 uses 32-true precision with no autocast. The
  earlier AMP note was wrong.
- The benchmark now uses a bounded photo list and ends naturally. All worker
  exit codes were 0. The earlier `terminate called without an active
  exception` was benchmark shutdown behavior, not a loader fault.

#### End-of-day OTF and PC diagnosis

- PC validation P_E for epochs 0-9 was 0.0374, 0.0294, 0.0272, 0.0210,
  0.0254, 0.0161, 0.0160, 0.0149, 0.0154, and 0.0158. The best epoch was
  7. Test on the same 21,912 images and order as the VM run: accuracy 0.9813,
  AUC 0.9974, P_E 0.0173. RAISE P_E was 0.0115, DIV2K 0.0619, and k=1
  0.0555.
- VM comparison: accuracy 0.9818, AUC 0.9975, P_E 0.0170; RAISE 0.0112,
  DIV2K 0.0574, and k=1 0.0559.
- PC diagnostics did not finish: decode ran at 0 workers and reached about
  14,000 photo decodes before the user stopped the kernel at 23:44. The
  breakdown cell did not run in the record. Eight idle persistent training
  workers were swapped out during testing; swap reached 7.9 of 8.2 GB.
- Native TF32 flags are matmul off and cuDNN on. On the L4, 3 runs each gave
  GPU-only / cache-loader images/s: native 285.9 / 277.9, both on 280.4 /
  270.2, and both off 271.0 / 265.3. TF32 changes logits by about 5e-4
  relative to float64; with TF32 off the difference is 2.5e-7.
- The PC diagnosis found small groups of 4 with prefetch 4, giving only 16
  images ahead per worker instead of 128. Decode bursts stalled ordered
  batches (p95 555 ms). The main join and pin copy cost about 5.7 ms; 16
  main torch threads made the join 40 ms instead of 2.8 ms. Eight idle
  persistent validation workers held 4.24 GB PSS.
- On 8,640 images, the fixes gave:

  | Case | Images/s |
  | --- | ---: |
  | PC GPU-only, fixed real batch | 366.0 |
  | PC decode, old prefetch 4 groups | 289.2 |
  | PC decode, new 128 images | 384.5 |
  | L4 GPU-only, fixed real batch | 284.6 |
  | L4 cache, old | 277.3 |
  | L4 cache, new | 276.2 |

- PC memory was 4.68 GB main plus 11.15 GB for 8 train workers. L4 memory
  was 2.75 GB main plus 7.47 GB for 6 workers. Full PC validation took
  103.6 s with 8 persistent workers and 136.0 s with 4 non-persistent
  workers; metrics were identical. Keys, labels, and bytes matched in 200
  batches at both tile sizes. QUICK passed all 15 cells. 85 tests pass.
- In decode mode, validation workers decode the same NEF several times when
  one photo's batches go to different workers: 51 decodes with 2 workers and
  101 with 4 for 1,600 pairs. Decoding used 96% of worker time.

#### Notebook OTF verification (step 8)

- Distinct photos per batch, 200 cache batches with 6 workers, mean / minimum:

  | Tile size | Before | After |
  | ---: | ---: | ---: |
  | 512 | 13.77 / 6 | 26.66 / 20 |
  | 256 | 5.89 / 2 | 18.90 / 12 |

  Workers per batch rose from 1 to 6. Across 6,360 shared keys per size,
  there were 0 byte changes.
- The uint8 path equals the old float32 path in a unit test and a real device
  batch.
- Loader-only cache speed at 512, 6 workers, 120 s: 596.7 images/s,
  452% CPU, 8.2 GB PSS RAM, and 154.6 MB/s disk.
- L4 check with v3c, 32-true precision, batch 32, and 300 steps: 278.7
  images/s with the loader versus 284.7 images/s for a fixed device batch.
  The gap is 2.1%. GPU use was 100%, RAM 9.4 GB, and CPU 292%. The L4 is
  the limit. The 17.2 ms loader wait mostly overlaps queued GPU work. A
  512 epoch takes about 6.4 min.
- QUICK smoke runs completed all 15 cells in order on the VM: OTF 100.3 s
  for 2 epochs and PNG 24.3 s. The OTF test used 32 images and had P_E 0.1875;
  QUICK scores have no meaning on tiny data. The checks showed uint8 equality,
  changed epoch-1 training stego, unchanged validation/test, and GroupNorm
  train/eval difference 0. All 16 test pairs stayed in the bpp bins.
- Notebook 02's existing v3c PNG predictions gave a byte-identical
  per-setting table before and after (SHA-256 `26fe36f6...a467`).
- 82 tests pass.

#### First full OTF run and PC diagnosis

- The VM run trained from 09:07 to 10:21, about 74 min. Validation P_E
  for epochs 0-9 was 0.0357, 0.0356, 0.0247, 0.0219, 0.0187, 0.0158,
  0.0177, 0.0146, 0.0147, and 0.0148. The best epoch was 7.
- Test used 21,912 images: accuracy 0.9818, AUC 0.9975, P_E 0.0170.
  Per-source P_E was RAISE 0.011153 (9,728 pairs) and DIV2K 0.057410
  (1,228 pairs). Per-camera P_E was D7000 0.011482 (7,490), D90 0.009937
  (2,214), D40 0.000000 (24), and not_applicable / DIV2K 0.057410 (1,228).
- Weak settings were lowest k=1 at 0.060117, plane k=1 at 0.052000,
  bpp 0.05-0.15 at 0.029448 (2,428 pairs), random placement at 0.021721,
  and stride at 0.020380. Contiguous placement was 0.007815. Bpp bins had
  0, 2,428, 4,141, 4,387, and 0 pairs for <=0.05, 0.05-0.15, 0.15-0.30,
  0.30-0.50, and >0.50.
- These results are not directly comparable with old v3c P_E 0.0139: the
  test set, grid, source mix, and stego generation changed. `legacy_v6`
  gives the like-for-like test. A legacy test of this checkpoint is Pending.
- The PC slow run reached 8.85 it/s, with 24.7/33 GB RAM, 6.3 GB swap,
  and 16 workers, so the user stopped it. GPU-only was 85.6 ms/step. The
  uint8 conversion was not the cause (85.66 versus 85.65 ms).
- The old prefetch gave 289.2 images/s. The new 128-image prefetch gave
  384.5 images/s. The main-process join and pin cost about 5.7 ms; 16 main
  torch threads made the join 40 ms instead of 2.8 ms. Decode bursts had a
  p95 of 555 ms. Eight idle validation workers held 4.24 GB PSS.
- Throughput after fixes, on 8,640 images:

  | Case | Images/s |
  | --- | ---: |
  | PC GPU-only, fixed real batch | 366.0 |
  | PC decode, old prefetch | 289.2 |
  | PC decode, 128 images | 384.5 |
  | L4 GPU-only, fixed real batch | 284.6 |
  | L4 cache, old | 277.3 |
  | L4 cache, new | 276.2 |

- PC RAM was 4.68 GB for the main process plus 11.15 GB for 8 train
  workers. L4 RAM was 2.75 GB for the main process plus 7.47 GB for 6
  workers. Full PC validation took 103.6 s with 8 persistent workers and
  136.0 s with 4 non-persistent workers; metrics were identical.
- Data keys, labels, and bytes were identical in 200 batches at both tile
  sizes. QUICK ran all 15 cells, and early stopping left no workers. 85
  tests pass.
- In decode mode, validation workers decode the same NEF several times when
  one photo's batches go to different workers: 51 decodes with 2 workers and
  101 with 4 workers for 1,600 pairs. Decoding used 96% of worker time.

#### VM disk and retained records

- VM disk on 2026-10-13: 193 GB total, 144 GB used (75%), and 49 GB free.
  Largest items were the `.npy` cache (50 GB), `Datasets/raise512-run1`
  (38 GB), RAISE NEFs (17 GB), DIV2K (3.7 GB), the system (about 13 GB),
  `.venv` (5.6 GB), and the uv download cache (5.6 GB).
- Run1 records not in git: `labels.csv` (19 MB; 3.3 MB compressed),
  `metadata.csv` (15 MB; 3.2 MB compressed), and `records/` plus
  `dataset_info.json` (81 MB; 5.1 MB compressed).
- Already in git: `data/raise/split_seed0.csv`, `selection.csv`, and
  `legacy_tile_manifest.csv`. The `masks/` folder is 507 MB, can be
  regenerated, and need not be kept.

#### Run1 image deletion effect

- VM disk changed from 144 GB used / 49 GB free to 107 GB used / 87 GB
  free (56%). No process used the folder. The teammate's `modelling/`
  notebook does not refer to it.
- `V3_DATA=png` no longer works on the VM because its images are gone.
  `V3_EVAL_PROFILE=legacy_v6` recreates the same 49,796 pairs from the NEFs;
  step 5 verified byte identity. Notebook 02 still works on old PNG-path
  predictions because `labels.csv` remains. The rebuild guide can recreate
  the images. Teammates' PC copies are not affected.

### Decided

- The user confirmed the reason for the v3c baseline. The goal of v3c is not
  to beat v3a. V3c is the baseline because it is batch-independent
  (GroupNorm) and adds clipping with no clear loss in detection. Its P_E is
  0.0019 higher than v3a (0.0139 versus 0.0120). This difference is small,
  and run-to-run noise is not measured.
- The user decided these points for the generator and the project:
  - The project will become multiple instance learning (MIL). In MIL, the
    model gets a group of items (a bag) with one label for the group. Here,
    one bag is one whole photo, and its tiles are the items. The final goal
    is a stego decision for each photo. This is future work.
  - The generator must support two modes: tile -> embed (current, plan A)
    and embed -> tile (plan B). Plan B is now reopened.
  - The bpp value applies to the whole photo, not to each tile.
  - A tile is stego only if at least one pixel value changed. Selected
    pixels that did not change do not count, because the model cannot see
    them.
  - Use on-the-fly generation if it does not slow down training.
  - Add no noise for now. The noisy-cover plan stays parked.
- The user decided the training platform and the data upload:
  - Kaggle notebooks will run the training. (Superseded: later on
    2026-10-13 the user said Kaggle is not a priority now. See the
    run-plan decision below.)
  - Upload only one master image set to Kaggle, one time. Do not upload the
    NEFs or the other image sets. (Superseded: the user withdrew "do not
    upload the NEFs" later on 2026-10-13. The NEFs are now the master set.
    See the master decision below.)
- Proposed, Pending user approval: the master set is 8-bit RGB full-photo
  PNG files, one file per photo, cropped to the tile grid. It goes in a
  private Kaggle dataset. The RAISE terms allow only non-commercial research
  and education. (Superseded: the user chose the NEFs as the master set
  later on 2026-10-13. See the master decision below.)
- For each 16-bit to 8-bit conversion, use the high byte (`x >> 8`). Never
  use division by 257 with rounding.
- The master source was open at this point. The user decided it next (see
  the following item). The recorded options were:
  - NEF master (about 17 GB, keeps the raw data). Make a one-time 8-bit
    cache in each session, or decode directly when enough CPUs are
    available.
  - 16-bit PNG master (about 74 GB). Direct load is faster. It does not keep
    the raw data.
- Master decision (user, 2026-10-13): the 1,000 RAISE NEFs (about 17 GB)
  are the single master set. Upload them one time, to Google Drive, Kaggle,
  or both.
  - This decision supersedes the earlier "do not upload the NEFs" line and
    the 8-bit PNG master proposal. Those lines stay above as history.
  - Reasons: the NEF set is the smallest upload. It keeps the raw sensor
    data for a possible later raw-domain noise model. It reproduces the
    current tiles exactly (42 of 42 photos byte-identical).
  - On machines with few CPUs, make a one-time 8-bit cache in each session.
    On machines with enough CPUs, decode the NEFs directly.
- Run-plan decision (user, 2026-10-13):
  - Kaggle execution is not a priority now. The two-GPU notebook setup
    (`ddp_notebook`) is deferred. This supersedes the earlier "Kaggle
    notebooks will run the training" line. The NEF master decision stays.
  - Short test runs are the main need. Each full run takes about 40-60
    minutes, and many configurations must be tested. The generator must
    accept settings for each run without a dataset rebuild. A screening
    protocol is a later step.
  - The user reviewed free Colab and the paid Google AI Pro plan. The user
    already has 400 GB of storage, so no paid plan is necessary now.
- Tile-grid decision (user, 2026-10-13):
  - The current pipeline crops each photo from the top-left corner to whole
    512 x 512 tiles. It drops the edge strips. With the 512 grid, this drops
    about 13% of the D7000 pixels, 15% of the D90 pixels, and 36% of the D40
    pixels.
  - New runs will use overlapped tiles. Each axis of length L gets
    n = ceil(L / T) tiles of size T. The last tile ends at the photo edge and
    overlaps its neighbour.
  - Do not use padding. Padded pixels are not real photo data.
  - Legacy mode 1 replay keeps the top-left crop, so that it reproduces
    `raise512-run1` exactly.
  - In mode 2, the generator embeds into the whole photo. Thus bpp applies
    to all photo pixels, and runs with 256 and 512 tiles see the same stego
    photo.
- Generator decisions (user, 2026-10-13), from the open decisions in the
  plan:
  1. Training variants: support both fixed variants and new settings in each
     epoch. Validation and test stay fixed.
  2. Class balance: keep the natural balance. The tile CNN trains on single
     tiles in mode 1. Mode 2 is for a future aggregation (MIL) model. Build
     mode 1 on the fly first.
  3. Capacity: if the payload does not fit, lower the bpp to fit. Record
     the effective bpp.
  4. Session cache: one full-photo `.npy` file per photo. The pre-built
     datasets may be deleted after the on-the-fly loader works, but only if
     all of these are true:
     - mode 1 replay is byte-identical on all 49,796 pairs;
     - notebook 04 trains from the loader;
     - the small records are kept;
     - the team agrees for the shared VM copy.
  5. Mix tiles from many photos with a bounded shuffle queue.
  6. Speed work must give byte-identical results only: a mask-free path,
     labels from changed indices, and a sort replacement with an identical
     result. Do not use a new random-selection algorithm without a new
     approval.
  7. Evaluation uses the full legacy settings mix for now. Harder training
     comes only from the training settings. (Refined later on 2026-10-13:
     the full mix is evaluated on the overlap grid as `overlap_full_mix`.)
  8. Fixed-size MIL batches: decide later.
- All execution for this team runs on the team VM. The VM checkout will
  move to branch `yx` after the user pushes it. (Done: see Did.)
- DIV2K decisions (user, 2026-10-13):
  - Use DIV2K HR for extra training and testing. Mix it into one pool with
    RAISE, in natural proportions, with one shared shuffle.
  - Validation and test are combined sets. Report the metrics combined and
    for each source. The per-source values check for false alarms from a
    possible JPEG history in DIV2K.
  - Source weights are an optional training setting.
  - DIV2K split:
    - Test: images 0801-0900 (100 images, the official validation folder).
    - Train and validation: images 0001-0800, split 680 / 120. The split is
      drawn one time with fixed seed 0 and frozen in a committed CSV file.
    - The CSV file, not the seed, is the source of truth. The split never
      changes between runs.
- Per-run seed decision (user, 2026-10-13): a `run_seed` setting can change
  the model initialisation, the shuffle order, the augmentation, and the
  training embedding settings for each run. The split and the validation
  and test embedding stay fixed. Cross-validation is possible later for
  final comparisons, but not for screening.
- Evaluation decision (user, 2026-10-13):
  - `overlap_full_mix` is the only required fixed evaluation profile.
  - There is no standing legacy evaluation. Legacy mode 1 replay stays as
    the generator exactness test.
  - If a v3c comparison is necessary, re-run the v3c configuration through
    the new generator.
- Run rules (user, 2026-10-13):
  - The user runs the full training runs interactively in the notebook
    (VS Code remote on the VM) to watch performance.
  - Agents may run the notebook only for QUICK or smoke tests, with
    separate run names. Agents must clean up after these tests.
- The plan stays as written (user choice). The orchestrator runs the work
  from a short step checklist, one step at a time.
- Code workflow (user, 2026-10-13):
  - The Worker commits locally on the PC.
  - The Orchestrator checks the commits, approves them, and pushes them.
  - Execution runs on the VM from a scratch copy.
  - The VM checkout changes only by fast-forward after a push.
- ROCm caution: the status changes from "open" to "cause found". See
  "ROCm caution, cause found" under Found.
- Done: removed the `MIOPEN_GEMM_ENFORCE_BACKEND=5` default from notebook
  04 in commit `5252943`.
  - The setup cell now says not to force MIOpen GEMM backend 5 because
    hipBLASLt `GemmWrw1x1_stride1` gives wrong conv3/conv4/conv5 1x1 weight
    gradients. It says that the forward pass and conv1 unfold fix are
    correct, and to restart the kernel if the override is already set.
  - The markdown cell no longer says that the switch requests backend 5.
    The conv1 unfold fix and `V3_ROCM_CONV1_FIX` switch are unchanged.
  - Only 2 cell sources changed. Outputs and metadata are unchanged. The
    executed record `notebooks/runs/04_v3c_clip3_group.ipynb` is unchanged.
  - The user's approval is complete. A find-db cleanup is not necessary for
    correctness.
- Re-runs of v3b and v3c (user, 2026-10-13): on hold for now. V3c stays the
  baseline, but it is marked as not a clean baseline.
- Session cache decision (user, 2026-10-13): the full-photo `.npy` cache
  is optional. The step 7 loader gets a source setting. By default, it
  decodes directly from the master NEF or PNG files, with no disk cache.
  The cache is used only when a cache path is set.
  - On the VM, the notebook points to `/srv/aai3001/otf-cache/v1/`, so VM
    users do not pay extra disk for the shared cache.
  - Teammates with strong CPUs do not need to use about 53 GB of disk.
  - Both sources must produce byte-identical samples. Step 7 checks this.
  - The cache holds the full clean 8-bit RGB photo after RAW conversion. It
    is not cut into tiles and is not embedded. Cutting and embedding happen
    during training.
- Loader decisions (user, 2026-10-13): use main torch threads 1, four
  non-persistent evaluation workers, and prefetch counted in images at 128
  per worker. `V3_PREFETCH` remains a group-count override. These fixes are
  in `9c3bf0b`.
- The user agreed to queue fix 4 (background join and pinned copy) and fix 5
  (one photo per evaluation worker, decode once, same order) after Mode 2.
- Reporting rule (user): workers report only at task end, when blocked, or
  when a decision is needed.
- TF32 decision (user, 2026-10-13): keep the notebook's native flags
  (matmul TF32 off, cuDNN TF32 on). Do not add a switch. The PC run had no
  TF32 and the VM run had TF32, but their test results were the same.

### Next

- Pending: the open items from the 2026-10-12 V3c entry are unchanged.
- Done: the Worker measured embed -> tile on the fly on whole photos. See
  "Embed -> tile benchmark on whole photos" above.
- Done: user decision on the master source. The NEFs are the master set.
  See the master decision above.
- Pending, deferred: a short real speed test on Kaggle (4 CPU cores).
  Kaggle is not a priority now.
- Done: loss or sampling balance for the label-1 tiles. The user chose the
  natural class balance (generator decision 2).
- Done: the Worker wrote the on-the-fly generator design plan for both
  modes: [docs/archive/plan-otf-generator.md](../../archive/plan-otf-generator.md) (`4d1036b`).
- Done: the Worker updated the plan for overlapped tiles (`18ef1fa`).
- Done: the user answered the open decisions in the plan. See the generator
  decisions above.
- Done: the Worker recorded these decisions in the plan (`e96ad9f`).
- Done: moved the VM checkout to `yx`.
- Done: plan step 1. Froze the RAISE and DIV2K photo and tile records, and
  the DIV2K split CSV (`63e4f59`).
- Done: step 2, the source step (`a7feb63`). See
  [docs/archive/source-verification.md](../../archive/source-verification.md).
- Done: step 3, the shared settings and seed helpers (`967cfd4`). See
  [docs/active/settings-and-seeds.md](../../active/settings-and-seeds.md).
- Done: v2 conv1 fix implementation (`60f5c17`).
- Done: step 4, one embed core plus a mask-free fast path (`c2bdb53`). See
  [docs/archive/embed-core-verification.md](../../archive/embed-core-verification.md).
- Done: step 5, mode 1 in the shared pipeline (`fc87b9c`). See
  [docs/archive/mode1-verification.md](../../archive/mode1-verification.md).
- Done: step 6, the full-photo `.npy` cache (`7e89fec`). See
  [docs/archive/cache-verification.md](../../archive/cache-verification.md).
- Done: step 7, the mode 1 loader (`970750a`). See
  [docs/archive/torch-loader-verification.md](../../archive/torch-loader-verification.md).
- Done: committed the compressed run1 labels and metadata to `data/raise/`
  (`8c4e549`).
- Done: step 8, notebook 04 connected to the OTF loader (`f3dca17`). See
  [docs/archive/notebook-otf-verification.md](../../archive/notebook-otf-verification.md).
- Done: the user's first full OTF v3c runs on the VM and PC. The VM
  checkpoint is `baseline_kv_xunet_v3_clip3_group_otf`; the PC record is
  `notebooks/runs/04_v3c_otf_pc_2026-10-13.ipynb` (`728db2b`). A `legacy_v6`
  test is Pending if a like-for-like comparison is needed.
- Done: after step 8, team agreement was recorded and
  `Datasets/raise512-run1` image folders were deleted on the VM. The kept
  records are described above.
- Pending: tell teammates that the VM checkout is on branch `yx`, and that
  the `raise512-run1` images are gone on the VM. `V3_DATA=png` there needs a
  rebuild.
- Pending: user decision on whether to run the VM checkpoint with
  `legacy_v6`.
- Later: Mode 2. Fixes 4 and 5 are queued after Mode 2.
- Done: the VM machine type decision after step 7: keep `g2-standard-8`.
- VM machine decision (user, 2026-10-13): keep `g2-standard-8`:
  one L4, 8 vCPUs (4 cores x 2 threads), and 32 GB RAM. After step 7, the
  loader keeps the L4 fully busy, so there is no upgrade.
  - If a change is needed, the candidate is `g2-standard-16`: one L4,
    16 vCPUs, and 64 GB RAM.
  - `g2-standard-24` is not suitable. It has 2 L4 GPUs, while notebook 04
    trains on 1 GPU. `ddp_notebook` is deferred.
  - The full-photo `.npy` cache is about 50 GB (RAISE about 45 GB and DIV2K
    about 5 GB), larger than 32 GB of RAM. Cache reads therefore come from
    disk.
- VM disk decision (user, 2026-10-13): wait. Take no disk action now. Do
  not write or build more cache. Steps 7 and 8 only read the cache, and mode
  2 uses the same cache.
  - After step 8, ask the team to delete `Datasets/raise512-run1` on the VM.
    This frees 38 GB, as allowed by plan decision 4.
  - Before that deletion, commit `labels.csv.gz` and `metadata.csv.gz` to
    `data/raise/` (option A). This is Pending.
- Done: ROCm-Works float64 layer check. The cause is found: wrong 1x1
  weight gradients with the variable. The v3b and v3c test metrics are
  correct; their training most likely had about 1 wrong step per epoch
  (not provable from the logs).
- Done: ROCm-Works confirmed the affected step count (most likely 10 of
  21,810 steps in each of v3b and v3c) and measured the step time without
  the variable. See "ROCm caution, cause found".
- Done: v2 conv1 fix implementation and PC self-check (`60f5c17`).
- Done: VM machine type decision after step 7: keep `g2-standard-8`.
