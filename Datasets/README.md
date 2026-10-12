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

Never commit `clean/`, `stego/`, `masks/`, CSV records, split copies,
checkpoints, or other generated dataset files. Keep only this README in git.
