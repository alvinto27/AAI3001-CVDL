# Datasets

Local data only. Git tracks only this README. The small records that describe
the data are in [`data/`](../data/) and are tracked.

## What is where

| Path | Contents | Size | Record in `data/` | Re-fetch |
| --- | --- | ---: | --- | --- |
| `originals/raise/` | `NEF/` (1,000 RAISE RAW photos), `RAISE_all.csv`, `selection.csv`, `tiles512-records/`, logs | 17 GB | `data/raise/` | [RAISE site](https://loki.disi.unitn.it/RAISE): `uv run python scripts/download_raise.py --selection data/raise/selection.csv --out Datasets/originals/raise/NEF` |
| `originals/div2k/` | `DIV2K_train_HR/` (800 PNG), `DIV2K_valid_HR/` (100 PNG) | 3.8 GB | `data/div2k/` | [DIV2K site](https://data.vision.ee.ethz.ch/cvl/DIV2K/): `DIV2K_train_HR.zip`, `DIV2K_valid_HR.zip` |
| `originals/alaska/tiff_various_color/` | 22,096 ALASKA2 TIFF and `SHA256SUMS`. 18,200 are selected | 33 GB | `data/alaska/` | Pi SSD `rpi400@rpi400-server:/srv/datasets/alaska/tiff_various_color/` |
| `originals/lsdir/hr/` | 11,000 selected LSDIR PNG in 84 shard folders and `SHA256SUMS` | 19 GB | `data/lsdir/` | Pi SSD `rpi400@rpi400-server:/srv/datasets/lsdir/hr/` (full copy and `inventory.csv`) |
| `generated/raise512-run1-records/` | `labels.csv`, `metadata.csv`, `splits/`, `dataset_info.json`, `records/`. The PNG images are not kept | 0.12 GB | `data/raise/run1_*.csv.gz` | see "Rebuild" below |
| `generated/otf-cache/` | Optional decoded-photo cache. Not present by default | 0 | none | see "Cache" below |

`master_path` in the `data/*/photo_manifest.csv` files points into `originals/`.
Frozen source records: [frozen-source-records.md](../docs/active/frozen-source-records.md),
[alaska-selection-v2.md](../docs/active/alaska-selection-v2.md),
[lsdir-selection-v1.md](../docs/active/lsdir-selection-v1.md).

## Cache (optional, `V3_SOURCE=cache`)

Training reads the photos directly (`V3_SOURCE=decode`, the default). The cache holds
decoded RAISE and DIV2K photos (1,900 photos, about 53 GB). To build it in container `aai3001`:

```sh
podman exec -w /work aai3001 python scripts/build_cache.py \
  --repo-root /work --cache-root /work/Datasets/generated/otf-cache/v1 --workers 8
podman exec -w /work aai3001 python scripts/check_cache.py \
  --repo-root /work --cache-root /work/Datasets/generated/otf-cache/v1 --gate verify
```

Then set `V3_SOURCE=cache` and `V3_CACHE_ROOT=<that path>`. The cache must sit in a
Git-ignored path. See the [cache record](../docs/archive/cache-verification.md).

## Rebuild the old PNG dataset (notebooks 01-03, `V3_DATA=png`, `scripts/check_*.py`)

The `clean/`, `stego/`, and `masks/` images (about 38 GB) were deleted. Notebook 04 and 05
OTF runs do not need them. To make them again, follow the
[rebuild guide](../docs/archive/rebuild-raise512-run1.md) from the repository root:
`bash scripts/rebuild_raise512_run1.sh`. It writes into
`generated/raise512-run1-records/` and needs the RAISE photos in `originals/raise/NEF/`.

## Citations and terms

- RAISE: D.-T. Dang-Nguyen, C. Pasquini, V. Conotter, G. Boato, “RAISE – A Raw Images
  Dataset for Digital Image Forensics,” ACM Multimedia Systems, Portland, Oregon,
  March 18-20, 2015. For **non-commercial research and educational purposes**
  ([guide](https://loki.disi.unitn.it/RAISE/guide.html)).
- DIV2K: E. Agustsson and R. Timofte, “NTIRE 2017 Challenge on Single Image
  Super-Resolution: Dataset and Study,” CVPR Workshops, July 2017. For **academic research
  only**; copyright belongs to the original owners.
- ALASKA2 and LSDIR: follow the terms of their source sites.

Never commit images, caches, or generated files. Keep only this README here.
