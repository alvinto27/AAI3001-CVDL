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
- A plain v2 PC speed loop used 32 images per step and 300 timed steps. It
  measured 5.68 steps/s (182 images/s) with deterministic mode, 5.76 steps/s
  (184 images/s) without it, and about 5.59 steps/s (179 images/s) with
  `MIOPEN_FIND_MODE=1`. Warm `cudnn.benchmark=True` reached 5.77 steps/s
  (185 images/s), but a fresh process paid about 145 seconds for one-time
  search. The GPU was 100% busy and CPU load was low, so the GPU was the limit.
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
- The speed result does not support deterministic mode as the cause of the PC
  versus L4 gap. A separate ROCm performance investigation will continue.
- The effect was likely easy to find because inputs were raw 0-255 values and
  tile photo textures varied widely.
- The PC v1 run stopped during epoch 9. It used the epoch-5 checkpoint,
  `epoch04-val_loss0.3209.ckpt`, for test evaluation. That checkpoint was
  selected by validation loss. The run had val accuracy 0.598 at epoch 3,
  val loss 0.3209 at epoch 5, val loss 2.28 and val accuracy 0.611 at epoch 6,
  val loss 29.6 and val accuracy 0.501 at epoch 8, and val loss 33.7 and val
  accuracy 0.500 at epoch 9. Train accuracy stayed near 0.98.
- The test set had 14,866 images (7,433 pairs). The result was P_E 0.0276 and
  AUC 0.989. At the best global threshold 0.981, P_FA was 0.010 and P_MD was
  0.045. Accuracy at threshold 0.5 was 0.839. Clean-image `prob_stego`
  quantiles at 50%, 90%, and 99% were 0.277, 0.813, and 0.982. Scores ranked
  well but were badly calibrated.
- The generator code makes `lowest, k=1` and `plane, k=1` the same embedding:
  both change bit 0, and both match by choosing `v +/- 1`. Edge values 0 and
  255 use the same valid-direction rule in both paths. They differ at k=2:
  `lowest` changes bits 0 and 1, while `plane` changes only bit 1. The combined
  k=1 result was P_E 0.111 for 956 pairs; by bpp it was 0.188 (287 pairs),
  0.086 (338), and 0.045 (331). The old label-specific values were 0.103 and
  0.115, within sampling noise.
- Per-setting P_E was about 0.03 for k2 and at most 0.02 for k>=3. Plane k6-k8
  had P_E 0.000. For bpp bins 0.05-0.15, 0.15-0.30, and 0.30-0.50, P_E was
  0.055, 0.023, and 0.008. The hardest heatmap cells were k1 at 0.05-0.15
  bpp: lowest 0.174 (n=149) and plane 0.199 (n=138). At 0.15-0.30 bpp they
  were 0.080 and 0.085; these labels describe the same k=1 embedding.
- Replace and match gave 0.028 and 0.027. Random bytes and letters gave 0.026
  and 0.029. Contiguous, random, and stride gave 0.018, 0.026, and 0.037;
  stride values varied from 0.00 to 0.07 with about 150 pairs each, so they
  are noisy.
- Local PC records are in `checkpoints/baseline_kv_xunet/`: the predictions,
  per-setting CSV, and four heatmap PNG files. They are not committed.
- About 80% of settings were near-solved by the 19.5 K-parameter model. Only
  k1-k2 with small payloads clearly separate models. One run and one seed make
  these numbers approximate.

### Decided

- Build baseline v2 in notebook 03. The work is in progress.
- For v2, put twins in different batches, select the best checkpoint by
  validation accuracy, use AdamW with weight decay 1e-4, set deterministic mode
  off, and detect the repository root automatically.
- Use the B PC speed setting: deterministic mode off, default MIOpen find mode,
  and no cuDNN benchmark. `MIOPEN_FIND_MODE=1` gave no gain and was removed
  from notebook 03.
- Run v2 on the VM while the PC finishes v1.
- Treat `lowest, k=1` and `plane, k=1` as one setting in future datasets, or
  report them as one combined group.
- Focus future datasets and larger models on k<=2 and bpp<=0.3.
- Compare v2 with v1 on the same test split. Include accuracy at threshold 0.5
  because the v1 score ranking was good but its calibration was poor.
- Delay larger models such as SRNet until the hard settings are confirmed.
  Possible later tests are 256-pixel crops, dropping trivial settings, and fp16.

### Next

- Pending: run notebook 03 on the VM and compare it with v1 on the same test split.
