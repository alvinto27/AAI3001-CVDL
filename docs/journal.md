# Steganalysis project journal

This journal records the stego work in this repository. Entries are in date order.
Numbers are kept when they are known. Pending work is marked.

## 2026-10-11 — Dataset and first baseline

### Did

- Replaced the old v5 generator with v6. The work is in `LSB-Dataset-Generator/`.
  - The embed engine supports `lowest` and `plane` bit modes, `k=1..8`,
    `replace` and `match`, and contiguous, stride, or random placement.
  - Payloads are uniform random bytes or letters. The bulk runner uses separate
    random-number streams, resumes work, replays output, and writes PNG text
    metadata and `metadata.csv`.
  - The cover tool decodes RAW files with rawpy 0.27.1, camera white balance,
    sRGB, and 8-bit output. It cuts 512 x 512 tiles. It does not resize.
  - Main commits: `b208258`, `8f1c52d`, `2255168`, and `a95cf61`.
- Chose plan A: cut tiles first, then embed. Plan B (embed full photos, then
  tile) is parked. Plane matching means `v +/- 2^(k-1)`. Random bytes are the
  encrypted-like payload.
- Downloaded 1,000 RAISE NEFs with seed 0. Selection was stratified. The
  download was about 18 GB. The downloader is in `scripts/download_raise.py`.
- Built `Datasets/raise512-run1`.
  - 49,796 clean tiles, 49,796 stego tiles, and 49,796 masks.
  - `labels.csv` has 99,592 data rows.
  - Generator seed 1 and one variant per cover.
  - Random settings per tile, with bpp 0.05 to 0.5 and stride 2 or 16.
  - Tiles by camera: D7000 38,556; D90 11,120; D40 120.
  - Replay verification passed 49,796 of 49,796 tiles.
- Split by source photo, not by tile. The split is 70/15/15 and is stratified
  by camera. The committed source split is `data/raise/split_seed0.csv`.
- Built baseline notebook `notebooks/01_baseline_kv_xunet.ipynb` in `d2fddcc`.
  - It uses a fixed KV filter. A KV filter is a high-pass filter that exposes
    small pixel changes. It is followed by an Xu-Net style CNN with 19.5 K
    parameters.
  - Clean and stego twins are in the same batch. Training uses flips and
    90-degree rotations. Inputs are raw 0-255 float32 values.
  - It uses Adam at 1e-3 and StepLR, with a tenfold drop at epoch 5.
  - P_E means the minimum average of false-alarm and miss rates over thresholds.
    P_E=0 is perfect detection and P_E=0.5 is guessing. AUC is the area under
    the receiver-operating-characteristic curve.
- Added the rebuild record in `docs/rebuild-raise512-run1.md` and the VM files
  in `configs/raise512-run1.yaml`, `data/raise/selection.csv`, and
  `scripts/rebuild_raise512_run1.sh` (`9435257`, `672c627`).

### Found

- The RAISE subset is large but repeatable. The clean and stego counts match.
- Plan A keeps each stego tile tied to a known clean tile.
- The team Spot VM has an NVIDIA L4, 8 CPUs, and 32 GB RAM in Seoul. Its disk
  was grown from 100 GB to 200 GB with Balanced PD.
- The first VM rebuild had four of 1,000 downloads fail during a short network
  problem. The script stopped at step 1 and shut the VM down as designed.

### Decided

- Rebuild the dataset on the VM from committed files instead of uploading
  about 38 GB.
- Use `--verify` and file counts as checks. Do not add SHA-256 fingerprints.
- Keep the baseline in float32. Small KV residuals must not be rounded away.

### Next

- Pending: rerun the VM rebuild after the network failure.
- Pending: finish a baseline run and record held-out results.

## 2026-10-12 — VM rebuild, evaluation, and baseline diagnosis

### Did

- Completed the VM rebuild. The rebuild guide is
  `docs/rebuild-raise512-run1.md`.
- Made `--verify` use all available CPU cores in `e27a21d`.
- Added notebook 02 for per-setting breakdowns, global-threshold detection
  rates, and k-by-bpp heatmaps (`307420f`, `3f31122`).
- Ran the baseline on the PC and VM. The baseline uses BatchNorm. BatchNorm
  normalizes activations with statistics from the current batch.

### Found

- VM verification passed 49,796 of 49,796 tiles with zero failures.
- VM `labels.csv` was byte-identical to the PC copy.
- VM times were about 12 minutes for tiles, 14 minutes for stego generation,
  and 35 minutes for verification on one core.
- The PC RX 9070 XT reached about 5.2 iterations/second, or about 7 minutes
  per epoch, with batch size 16 pairs. The VM L4 reached about 6.4 iterations
  per second.
- The likely speed cause is library tuning: NVIDIA cuDNN is better tuned than
  AMD MIOpen for these small convolutions on gfx1201. Deterministic mode also
  limits MIOpen choices. An earlier PC estimate of 13 minutes per epoch came
  from cold-cache tests. The GPU, not the CPU, was the limit on the PC.
- At PC epoch 6 of 10, the baseline showed train accuracy 0.980 and train loss
  0.060, but validation accuracy 0.611 and validation loss 2.28. At epoch 3,
  validation accuracy was 0.598 and validation loss was above 3.
- The VM v1 checkpoint `checkpoints/baseline_kv_xunet/last.ckpt` was at epoch 1,
  global step 2,181. Its file name records `val_loss=13.05`.
- A seed-0 sample of 1,000 train pairs and 1,000 validation pairs gave these
  results. Each pair made 32 images in a batch of 16 pairs.

  | set | eval mode (running BatchNorm statistics) | train-mode BatchNorm with twins together |
  | --- | ---: | ---: |
  | train | accuracy 0.5125, loss 13.08 | accuracy 0.969, loss 0.104 |
  | val | accuracy 0.506, loss 12.92 | accuracy 0.963, loss 0.113 |

- Train and validation behaved the same. This rules out simple content
  memorisation. The model learned to compare each image with its twin through
  BatchNorm batch statistics. When scored alone, it was at chance and very
  overconfident. The hypothesis is confirmed.
- The effect was likely easy to find because inputs were raw 0-255 values and
  tile photo textures varied widely.

### Decided

- Build baseline v2 in notebook 03. The work is in progress.
- For v2, put twins in different batches, select the best checkpoint by
  validation accuracy, use AdamW with weight decay 1e-4, set deterministic mode
  off, set `MIOPEN_FIND_MODE=1`, and detect the repository root automatically.
- Run v2 on the VM while the PC finishes v1.
- Delay larger models such as SRNet until the heatmap shows which settings are
  hard. Possible later tests are 256-pixel crops, dropping trivial settings,
  and fp16.

### Next

- Pending: finish the PC v1 run and record its final test result.
- Pending: run notebook 03 on the VM and compare it with v1.
- Pending: use notebook 02 heatmaps to choose settings for larger models.
