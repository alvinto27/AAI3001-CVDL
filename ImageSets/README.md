# ImageSets

This folder stores source image sets and intermediate image records. The
image data is local only. Only this README is tracked.

## Current contents

| Path | Contents | Local size |
| --- | --- | ---: |
| `RAISE/NEF/` | 1,000 RAISE RAW photos, selected from `data/raise/selection.csv` | about 17 GB |
| `RAISE/tiles512-records/` | Tile manifest and preparation records. The tile PNGs were deleted after the 2026-10-12 audit because every tile has a pixel-identical clean copy in the dataset. | about 5 MB |
| `RAISE/RAISE_all.csv` | RAISE catalogue | about 4.1 MB |
| `DIV2K_train_HR/` | 800 PNGs, IDs 0001-0800 | about 3.3 GB |
| `DIV2K_valid_HR/` | 100 PNGs, IDs 0801-0900 | about 429 MB |

The RAISE source area is about 17 GB. The current ImageSets folder is about
21 GB. RAISE and DIV2K now have [frozen source records](../docs/active/frozen-source-records.md).
DIV2K training integration is Pending; its PNG headers are used by the record
builder. Both sources are planned for combined training and evaluation.

## Create or download

Download the selected RAISE photos with:

```sh
uv run python scripts/download_raise.py \
  --selection data/raise/selection.csv \
  --out ImageSets/RAISE/NEF
```

Prepare 512 x 512 tiles with:

```sh
uv run python LSB-Dataset-Generator/prepare.py \
  --input ImageSets/RAISE/NEF \
  --output ImageSets/RAISE/tiles512 \
  --tile 512 --seed 0 --png-compress-level 1
```

The complete rebuild is in the [RAISE 512 rebuild guide](../docs/archive/rebuild-raise512-run1.md).
The committed selection is `../data/raise/selection.csv`.

Download DIV2K from the [official DIV2K page](https://data.vision.ee.ethz.ch/cvl/DIV2K/):

- `DIV2K_train_HR.zip` produces `DIV2K_train_HR/`.
- `DIV2K_valid_HR.zip` produces `DIV2K_valid_HR/`.

## Citations and terms

RAISE's [official page](https://loki.disi.unitn.it/RAISE) asks users to cite:

> D.-T. Dang-Nguyen, C. Pasquini, V. Conotter, G. Boato, “RAISE – A Raw
> Images Dataset for Digital Image Forensics,” ACM Multimedia Systems,
> Portland, Oregon, March 18-20, 2015.

The [official RAISE guide](https://loki.disi.unitn.it/RAISE/guide.html) says
that RAISE is for **non-commercial research and educational purposes**.
Follow the source site's terms before sharing or reusing the photos.

The [official DIV2K page](https://data.vision.ee.ethz.ch/cvl/DIV2K/) asks users
to cite:

> E. Agustsson and R. Timofte, “NTIRE 2017 Challenge on Single Image
> Super-Resolution: Dataset and Study,” IEEE Conference on Computer Vision
> and Pattern Recognition (CVPR) Workshops, July 2017.

The same page says that DIV2K is for **academic research purposes only**,
that the images were collected from the Internet, and that copyright belongs
to the original owners. It also gives a removal contact for copyright owners.

## Do not commit

Never commit NEF files, DIV2K PNG files, tile files, or downloaded catalogues.
Keep only this README in git under `ImageSets/`. The reviewed small CSV records
in `data/raise/` and `data/div2k/` are tracked; source images stay local.
