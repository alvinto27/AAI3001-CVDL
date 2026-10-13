# Full-photo source API and verification

This implements plan step 2 only. No cache, embedding, training, or GPU work
is part of the source step. Frozen input records are described in
[the source record guide](frozen-source-records.md).

## Shared RAW definition

`LSB-Dataset-Generator/lsb_raw.py` defines the settings once. Both
`lsb_prepare.py` and `lsb_source.py` use its `decode_raw_pixels()` function.
Preparation also uses `raw_settings_record()` for its unchanged v6 record.

| Setting | Value |
| --- | --- |
| `use_camera_wb` | `True` |
| `no_auto_bright` | `True` |
| `output_color` | `rawpy.ColorSpace.sRGB` |
| `output_bps` | 8 |
| `demosaic_algorithm` | `rawpy.DemosaicAlgorithm.AHD` |
| Orientation | rawpy default; no explicit flip argument. |

The settings record is a copy of the immutable definition. The disabled RAW
noise hook runs before demosaic and does nothing. No configuration can enable
noise yet. The NEF file is never changed. The preparation CLI keeps its
original crop, raster-source handling, PNG encoding, and public record fields.

## Source and tile API

Use `lsb_source` from the `LSB-Dataset-Generator` module directory.

| Function | Result / rule |
| --- | --- |
| `configure_source_worker()` | Set `OMP_NUM_THREADS=1` in a fresh worker. Refuse an already-loaded rawpy with a conflicting current setting. |
| `load_photo(row, repo_root)` | Full `uint8` RGB array, shape `(source_height, source_width, 3)`. Paths are relative to the data checkout. No resize, padding, or float conversion. |
| `overlap_starts(length, tile_size)` | Strictly ordered starts; final start is `length - tile_size`. Reject a side smaller than the tile. |
| `grid_axes(row, policy, tile_size)` | Validate and use the frozen starts/counts. Overlap supports 512/256; legacy supports RAISE 512 only. |
| `topleft_crop(pixels, row)` | Legacy cropped view, using the recorded grid/crop dimensions. |
| `iter_tiles(pixels, row, policy, tile_size)` | Row-major tile views of the full input array with stable geometry. Do not mutate these views. |

Every `SourceTile` exposes `pixels` and `identity`:

```text
(photo_id, grid_policy, tile_size, row, column, y, x)
```

Call `iter_tiles()` with the full photo even for `topleft_crop`; do not pass
its cropped result as though it were a full source. Overlap starts must match
both the frozen manifest and the defined edge-overlap rule. Stale or altered
grid/shape records fail rather than produce a different grid silently.

`nef` uses the shared RAW decoder and requires noise `disabled`.
`png8` uses Pillow with an 8-bit RGB PNG header and mode `RGB`; it rejects
palette, grayscale, alpha, transparency, higher depth, and animation. There
is no mode conversion, EXIF transpose, gamma change, or colour-profile
conversion. PNG pixels keep their stored orientation. Noise must be
`not_applicable`; PNG never enters rawpy or the RAW hook.

### Worker launch rule

Export `OMP_NUM_THREADS=1` **before rawpy loads**, preferably before Python
starts. An environment change after LibRaw/OpenMP loads cannot reliably
change its thread count. Use spawned workers and call the worker setup in
each process. The standalone preparation CLI's thread policy is unchanged.

## Re-run checks on the VM

Copy code only to a VM scratch folder. Use the existing virtual environment
and read-only checkout data. Never run this against output paths in the VM
checkout; the verifier writes only standard output.

```sh
cd ~/otf-dev/step2-source
OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 \
  /srv/aai3001/AAI3001-CVDL/.venv/bin/python \
  -m unittest discover -s tests -p test_lsb_source.py -v

OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 \
  /srv/aai3001/AAI3001-CVDL/.venv/bin/python scripts/check_sources.py \
  --repo-root /srv/aai3001/AAI3001-CVDL --workers 8
```

The verifier decodes every source, reads every original clean tile, checks
both overlap grids, and checks full pixel coverage on one RAISE photo from
each camera/orientation group. It also checks preparation/source pixels and
camera metadata on those six photos. DIV2K pixels are compared with a
separate direct Pillow read. No source or image output file is written.

## VM results, 2026-10-13

All checks passed with eight spawned CPU workers and `OMP_NUM_THREADS=1`.
Total wall time: **611.690 seconds**, or **10 minutes 11.7 seconds**. This
includes source decoding, all reference PNG reads/comparisons, overlap
checks, six extra preparation comparisons, and worker startup/shutdown.

| Check | Result |
| --- | --- |
| RAISE direct NEF decode / manifest shape | 1,000 / 1,000 passed. |
| Legacy clean tile byte identity | 49,796 / 49,796 passed; zero mismatches. |
| RAISE overlap grids, 512 / 256 | 65,184 / 243,120 tiles; exact manifest totals. |
| RAISE full coverage | Six photos: every camera and both orientations, both tile sizes. Every real pixel covered; no padding; last tiles end at real edges. |
| Preparation/source parity | Six full photos and their camera metadata matched. Existing legacy positions and v6 RAW settings record passed regression tests. |
| DIV2K source pixels / manifest shape | 900 / 900 passed against independent direct PNG reads. |
| DIV2K overlap grids, 512 / 256 | 10,960 / 42,744 tiles; exact manifest totals. |
| Source unit tests | 9 / 9 passed. |

Times below are seconds per **direct full NEF decode**, not per tile and not
per complete reference-file verification. They include source read, rawpy
processing, array validation, and any first-use import overhead. Measurements
were taken during the parallel full check, not an isolated speed benchmark.

| Camera | Photos | Median seconds/photo | p90 seconds/photo |
| --- | ---: | ---: | ---: |
| D7000 | 714 | 3.775373 | 3.981206 |
| D90 | 278 | 2.883643 | 3.091062 |
| D40 | 8 | 1.402633 | 1.469743 |

Coverage/preparation samples: D7000 `r00617aa1t` / `r043951a3t`, D90
`r01058910t` / `r006b0e4bt`, D40 `r221dfc60t` / `r684e92a7t` (landscape /
portrait). Environment: rawpy 0.27.1, LibRaw 0.22.1, NumPy 2.5.3, Pillow
12.3.0. No dependency files changed. No source files or VM checkout files
were written. Scratch logs, code copies, and test outputs are removed after
verification; this record retains the measured result.

Pending: shared sampling and embedding steps. Cache construction is not part
of this check.
