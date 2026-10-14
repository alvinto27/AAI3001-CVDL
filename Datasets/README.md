# Datasets

This folder stores generated datasets. The data is local only. Only this
README is tracked.

## Current dataset

`raise512-run1/` contains paired 512 x 512 RGB tiles:

| Path or file | Contents | Local size |
| --- | --- | ---: |
| `clean/` | 49,796 clean PNG tiles | about 18 GB |
| `stego/` | 49,796 stego PNG tiles | about 19 GB |
| `masks/` | 49,796 mask PNG files | about 507 MB |
| `labels.csv` | 99,592 data rows | about 19 MB |
| `metadata.csv` | generator metadata | about 15 MB |
| `records/` | rebuild and verification records | small |
| `splits/split_seed0.csv` | source-photo train/validation/test split | about 2.24 MB |

The complete local dataset is about 38 GB. Clean and stego images stay in
their own folders. A source photo must stay in one split.

Cite [D.-T. Dang-Nguyen, C. Pasquini, V. Conotter, G. Boato, “RAISE – A Raw Images Dataset for Digital Image Forensics,” ACM Multimedia Systems, Portland, Oregon, March 18-20, 2015.](../ImageSets/README.md#citations-and-terms). RAISE is for non-commercial research and educational purposes.

## Full-photo OTF cache

`otf-cache/v1/` stores decoded full-photo RGB arrays for the on-the-fly (OTF)
loader. A complete cache has 1,900 photos and uses about 52.8 GB (49.2 GiB).
It is local generated data and Git does not track it.

Build and verify the PC cache in container `aai3001`:

```sh
podman exec -w /work aai3001 python scripts/build_cache.py \
  --repo-root /work --cache-root /work/Datasets/otf-cache/v1 --workers 8
podman exec -w /work aai3001 python scripts/check_cache.py \
  --repo-root /work --cache-root /work/Datasets/otf-cache/v1 --gate verify
```

An in-checkout cache is accepted only when `git check-ignore` confirms that
Git ignores its path. See the [cache verification record](../docs/cache-verification.md).

## Rebuild

Run the [RAISE 512 rebuild guide](../docs/rebuild-raise512-run1.md) from the
repository root. The unattended command is:

```sh
bash scripts/rebuild_raise512_run1.sh --shutdown 2>&1 | tee -a rebuild-raise512-run1.log
```

The guide uses `scripts/`, `configs/raise512-run1.yaml`, the committed
`data/raise/selection.csv`, and the RAISE source files in `ImageSets/RAISE/`.
It checks for 49,796 clean, stego, and mask files and 99,592 label rows.

## Do not commit

Never commit `clean/`, `stego/`, `masks/`, `otf-cache/`, CSV records, split
copies, checkpoints, or other generated dataset files. Keep only this README
in git.
