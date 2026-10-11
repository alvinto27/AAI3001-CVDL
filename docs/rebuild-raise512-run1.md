# Rebuild the RAISE 512 dataset

This guide rebuilds `Datasets/raise512-run1` from the RAISE source files.
Run every command from the repository root.

## Before you start

| Item | Requirement |
| --- | --- |
| Repository | Use the commit that contains this guide. It must include the v6 generator, `configs/raise512-run1.yaml`, and the files in `data/raise/`. |
| Python | Use Python 3.12 with `uv sync`, or install the generator requirements with pip. |
| RAW decoder | Use `rawpy==0.27.1`. The lock file pins this version. A different rawpy or LibRaw version can produce different pixels. |
| Disk | Keep about 80 GB free. |
| RAISE source | The committed selection contains the source URLs. No RAISE catalog file is needed. |

The small committed RAISE files are:

| File | Size |
| --- | ---: |
| `data/raise/selection.csv` | 97,228 bytes (about 95 KiB) |
| `data/raise/split_seed0.csv` | 2,345,461 bytes (about 2.24 MiB) |

Set up the team environment:

```sh
uv sync
```

For a separate environment, install the generator requirements instead:

```sh
python -m pip install -r LSB-Dataset-Generator/requirements.txt
```

The download step needs network access to the RAISE NEF URLs.

## Run the unattended rebuild

The script runs the seven steps below in order. It is safe to run the same command again after a Spot preemption. The download, tile preparation, and generator resume from their recorded work. Label and split files are replaced with the same deterministic files.

Start a tmux session on the VM:

```sh
tmux new -s rebuild
```

Run this command inside tmux. The command keeps a copy of all output in a log:

```sh
bash scripts/rebuild_raise512_run1.sh --shutdown 2>&1 | tee -a rebuild-raise512-run1.log
```

Use `--delete-intermediate` too if you want to remove the NEF and tile folders after all checks pass:

```sh
bash scripts/rebuild_raise512_run1.sh --delete-intermediate --shutdown 2>&1 | tee -a rebuild-raise512-run1.log
```

Detach with `Ctrl-b`, then `d`. Check the result later:

```sh
tail -n 3 rebuild-raise512-run1.log
```

The last line must start with `REBUILD DONE:` and show `clean=49796 stego=49796 masks=49796 labels=99592`. A failed run ends with `REBUILD FAILED at step n name`. Run the same command again to resume.

## Rebuild steps

1. Download the 1,000 photos in the committed selection. The `--selection` option reads the selection file directly. It does not read a full RAISE catalog.

   ```sh
   uv run python scripts/download_raise.py \
     --selection data/raise/selection.csv \
     --out ImageSets/RAISE/NEF
   ```

   The committed manifest has 1,000 rows and the header `seed=0 count=1000`.

2. Decode all selected photos into 512 x 512 RGB PNG tiles. Do not set `--max-tiles-per-image`.

   ```sh
   uv run python LSB-Dataset-Generator/prepare.py \
     --input ImageSets/RAISE/NEF \
     --output ImageSets/RAISE/tiles512 \
     --tile 512 \
     --seed 0 \
     --png-compress-level 1
   ```

3. Generate the clean images, stego images, and masks. `read_config` resolves a relative path from the YAML file's folder, not from the current folder. Therefore `../ImageSets/RAISE/tiles512` and `../Datasets/raise512-run1` in `configs/raise512-run1.yaml` resolve to the two paths below when the command runs from the repository root.

   ```sh
   uv run python LSB-Dataset-Generator/generate.py \
     --config configs/raise512-run1.yaml
   ```

4. Verify the generated images by deterministic replay.

   ```sh
   uv run python LSB-Dataset-Generator/generate.py \
     --verify Datasets/raise512-run1
   ```

5. Build the label file from the generator metadata and tile manifest.

   ```sh
   uv run python scripts/build_labels.py \
     --dataset Datasets/raise512-run1 \
     --tiles-manifest ImageSets/RAISE/tiles512/manifest.csv
   ```

6. Install the committed train/test split.

   ```sh
   mkdir -p Datasets/raise512-run1/splits
   cp data/raise/split_seed0.csv Datasets/raise512-run1/splits/split_seed0.csv
   ```

7. Check the file counts.

   ```sh
   test "$(find Datasets/raise512-run1/clean -type f -name '*.png' | wc -l)" -eq 49796
   test "$(find Datasets/raise512-run1/stego -type f -name '*.png' | wc -l)" -eq 49796
   test "$(find Datasets/raise512-run1/masks -type f -name '*.png' | wc -l)" -eq 49796
   test "$(awk 'NR > 1 { count++ } END { print count + 0 }' Datasets/raise512-run1/labels.csv)" -eq 99592
   ```

## Expected result

| Item | Count |
| --- | ---: |
| Clean PNG files | 49,796 |
| Stego PNG files | 49,796 |
| Mask PNG files | 49,796 |
| Data rows in `labels.csv` | 99,592 |

The clean, stego, and mask files must remain in their `clean/`, `stego/`, and `masks/` folders. The split file must be in `splits/`.

## Free intermediate space

Keep `ImageSets/RAISE/NEF` and `ImageSets/RAISE/tiles512` until the label and count checks finish. Then they are no longer needed for this dataset:

```sh
rm -rf ImageSets/RAISE/NEF ImageSets/RAISE/tiles512
```

On the reference run, the NEF files used about 17 GB and the tile PNG files used about 18 GB. Removing both freed about 35 GB. Check the size on your machine with `du -sh` before removal.
