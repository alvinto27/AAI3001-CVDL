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
- Finished v2 on the VM L4: 10 epochs with independently shuffled images,
  AdamW weight decay 1e-4, and checkpoint selection by validation accuracy.
- Completed the running-statistics and batch-composition diagnoses on the VM.
- Completed the L4 convolution benchmark for the ROCm investigation. Details
  stay in `rocm-perf/`, local only.

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

- Train and validation behaved the same. The first diagnosis suggested a
  twin-in-batch shortcut, but it did not separate twin pairing from the use of
  batch statistics. Diagnosis 2 below corrects that conclusion: twins were
  not the cause.
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

- V2 did not close the train/validation gap. Train accuracy was 0.956-0.980.
  Validation accuracy/loss were 0.533/4.31 at epoch 2, 0.513/10.4 at epoch 4,
  0.522/13.8 at epoch 7, and 0.510/24.6 at epoch 9. The kept checkpoint was
  epoch 1, with best validation accuracy 0.5448. Epoch numbers here start at
  1; the checkpoint field `epoch=0` means epoch 1.
- V2 test accuracy at 0.5 was 0.5352, AUC 0.915, and P_E 0.1148. V1 epoch 5
  gave 0.839, 0.989, and 0.0276 on the same test split. Separating twins did
  not fix the problem.
- Selecting by validation accuracy failed: it uses the broken 0.5 threshold.
  Select by validation P_E or AUC instead.
- In this Lightning version, `save_last=True` wrote `last.ckpt` only when a
  top-k checkpoint was saved. On both PC v1 and VM v2, last was best, not the
  latest epoch. A resume after preemption would restart from the best epoch.
  Save a checkpoint every epoch next time.

#### Diagnosis 2 — Running statistics versus batch statistics

Both models used seed-0 samples of 1,000 train pairs and 1,000 validation
pairs. Each entry is **accuracy / cross-entropy loss**. Modes are:

- **a:** evaluation mode with stored running statistics.
- **b:** evaluation mode after resetting BatchNorm statistics and recalibrating
  with 100 random unpaired train batches of 32 images, cumulative averaging.
- **c:** train-mode BatchNorm on a model copy, 63 random unpaired batches of
  32 images (2,016 images; covers can repeat across batches, never within one).
- **d:** train-mode BatchNorm on a copy, with twins together (2,000 images).

No gradients or original checkpoint statistics were changed.

| Checkpoint | Set | a | b | c | d |
| --- | --- | --- | --- | --- | --- |
| V2 epoch 1 | Train | 0.5465 / 4.0007 | 0.7515 / 0.5356 | 0.9524 / 0.1252 | 0.9565 / 0.1217 |
| V2 epoch 1 | Val | 0.5410 / 3.9859 | 0.7495 / 0.5609 | 0.9608 / 0.1165 | 0.9585 / 0.1266 |
| V1 PC epoch 5 | Train | 0.8365 / 0.3318 | 0.5400 / 7.0538 | 0.9722 / 0.0754 | 0.9750 / 0.0708 |
| V1 PC epoch 5 | Val | 0.8585 / 0.3009 | 0.5435 / 6.9305 | 0.9782 / 0.0726 | 0.9775 / 0.0697 |

V2 last and best files have identical model tensors, both from epoch 1, so
there is only one v2 model here. Modes c and d are approximately equal: twins
were not the real cause. Recalibration helps v2 (validation 0.54 to 0.75) but
harms v1 (0.86 to 0.54). It is not a reliable fix. V1's good epoch 5 was
partly luck in its stored statistics.

The first BatchNorm layer (`bn1`) had these statistics. Mean magnitude means
mean absolute value across channels; variance means mean across channels.
The change ratio is mean absolute change divided by the stored magnitude.
Batch variance spread uses the first 50 unpaired calibration batches.

| Model | Mean magnitude stored / recalibrated | Mean change ratio | Variance stored / recalibrated | Variance change ratio | Batch variance min / median / max |
| --- | --- | --- | --- | --- | --- |
| V2 epoch 1 | 3.3043 / 2.9098 | 0.1194 | 147.533 / 103.051 | 0.3015 | 9.0 / 75.7 / 412 |
| V1 PC epoch 5 | 4.7433 / 4.3120 | 0.0909 | 281.616 / 237.977 | 0.1550 | 21.2 / 170 / 1027 |

#### Diagnosis 3 — Batch composition

The same seed-0 pair samples were used. Each mode used 63 batches of 32
images, with the same randomly selected covers across modes. Covers can
repeat across batches; each batch has distinct covers. Model copies used
train-mode BatchNorm without gradients:

- **e:** all clean; accuracy is the fraction predicted clean.
- **f:** all stego; accuracy is the fraction predicted stego.
- **g:** exactly 16 clean and 16 stego, without twins.

Each entry is **accuracy / mean prob_stego / cross-entropy loss**.

| Checkpoint | Set | e: all clean | f: all stego | g: mixed 50/50 |
| --- | --- | --- | --- | --- |
| V2 epoch 1 | Train | 0.0124 / 0.9729 / 9.6020 | 0.7887 / 0.7801 / 0.6030 | 0.9568 / 0.4881 / 0.1216 |
| V2 epoch 1 | Val | 0.0060 / 0.9790 / 9.6525 | 0.7728 / 0.7706 / 0.6923 | 0.9613 / 0.4899 / 0.1265 |
| V1 PC epoch 5 | Train | 0.0094 / 0.9875 / 28.7793 | 0.8323 / 0.8337 / 0.6749 | 0.9792 / 0.4946 / 0.0626 |
| V1 PC epoch 5 | Val | 0.0035 / 0.9946 / 29.4181 | 0.8170 / 0.8175 / 0.8469 | 0.9782 / 0.4987 / 0.0679 |

Both models call almost every clean tile stego in all-clean batches, yet score
0.96-0.98 on mixed batches. They judge images relative to their batch.
The current root-cause view is BatchNorm with raw 0-255 KV residuals whose
scale varies widely with texture and high-k stego. The measured batch
variance range was about 45-50 times; this is a variance ratio, not a measured
45-50-fold change in standard deviation.

### Decided

- V2 in notebook 03 is complete; its unpaired batches did not close the gap.
- V2 used unpaired batches, validation-accuracy checkpoint selection, AdamW
  with weight decay 1e-4, deterministic mode off, and automatic root detection.
  Validation-accuracy selection is superseded by validation P_E for v3.
- Use the B PC speed setting: deterministic mode off, default MIOpen find mode,
  and no cuDNN benchmark. `MIOPEN_FIND_MODE=1` gave no gain and was removed
  from notebook 03.
- All stego training now runs on the VM. The PC GPU belongs to the ROCm
  investigation.
- Treat `lowest, k=1` and `plane, k=1` as one setting in future datasets, or
  report them as one combined group.
- Focus future datasets and larger models on k<=2 and bpp<=0.3.
- V3 tests one change at a time on the VM: v3a clips the KV residual to
  `[-T, T]`, with T=3, and keeps BatchNorm; v3b replaces BatchNorm with
  GroupNorm, with no clipping; v3c combines both only if needed. GroupNorm
  computes statistics within each image rather than across the batch.
- Every v3 run selects by validation P_E and saves a checkpoint every epoch.
  Report test P_E, AUC, and accuracy at 0.5, the a-versus-c gap, the all-clean
  batch check, and k-by-bpp heatmaps.
- Delay larger models such as SRNet until the hard settings are confirmed.
  Possible later tests are 256-pixel crops, dropping trivial settings, and fp16.

### Next

- V3a and v3b are now complete; see the continuation below. V3b used an
  approved PC GPU window rather than the VM.

## 2026-10-12 — V3 runs and ROCm convolution fix (continuation)

### Did

- Added notebook 04 in `ef2a2e3`, with `CLIP_T` and `NORM` switches. V3a
  clips the KV residual to [-3, 3] and keeps BatchNorm. V3b uses no clipping
  and GroupNorm with 8 groups. Other training and split settings stay the same.
- Selected checkpoints by validation P_E. Saved one checkpoint every epoch
  plus a separate best checkpoint. Both full runs verified `last.ckpt` had
  `epoch=9`, the final epoch of 10. CUDA alone uses `cudnn.benchmark=True`.
- Added the ROCm conv1 weight-gradient fix in `ee5813f`.
  - With MIOpen 3.6.1, Winograd weight gradient is not applicable to this
    512 x 512 convolution at batch size >=15. The fallback makes one rocBLAS
    matrix multiplication per image. See `rocm-libraries#12222`.
  - The replacement uses unfold plus batched matrix multiplication. At batch
    32, measured training step time fell from 174-189 ms to 103.5 ms, or
    309 images/s. Weight-gradient relative difference was 2.3e-6.
  - The user chose ON by default on ROCm. Set `V3_ROCM_CONV1_FIX=False` to
    disable it. It has no effect on CUDA. Restart the kernel after changing
    this setting so the MIOpen environment setting takes effect.
- Finished both ten-epoch runs. V3a ran on the VM L4; v3b ran in the PC ROCm
  container during an approved GPU window. Kept logs and run folders, and
  removed both temporary training scripts after completion.
- Records: VM `logs/v3a_vm.log` and
  `checkpoints/baseline_kv_xunet_v3_clip3_batch/`; PC `logs/v3b_pc.log` and
  `checkpoints/baseline_kv_xunet_v3_noclip_group/`. At that time these were
  not committed. Since `3787b14`, git tracks the small outputs (test CSVs,
  heatmaps, `metrics.csv`), but not the `.ckpt` files. The v3a log is now at
  `logs/v3a_legacy_from_vm/v3a_vm.log`.

### Found

#### Validation trend and selected checkpoints

Epoch numbers here start at 1; checkpoint filenames start at 0. Values below
are the per-epoch validation P_E printed in the logs.

| Epoch | V3a: clip 3 + BatchNorm | V3b: no clip + GroupNorm |
| ---: | ---: | ---: |
| 1 | 0.0239 | 0.0268 |
| 2 | 0.0182 | 0.0165 |
| 3 | 0.0146 | 0.0189 |
| 4 | 0.0203 | 0.0174 |
| 5 | 0.00963 | 0.0163 |
| 6 | 0.00970 | 0.00957 |
| 7 | 0.00870 | 0.0102 |
| 8 | 0.00883 | 0.00997 |
| 9 | 0.00830 | 0.00943 |
| 10 | 0.00790 | 0.0102 |

- V3a kept epoch 10: `best-epoch09-val_pe0.0079-val_auc0.9988.ckpt`.
  Test accuracy at 0.5 was 0.9843, AUC 0.9982, and P_E 0.0120. The best
  global threshold was 0.1646 (about 0.165), with P_FA 0.0112 and P_MD 0.0129.
  K=1 P_E was 0.0277 for lowest and 0.0503 for plane.
- V3b kept epoch 9: `best-epoch08-val_pe0.0094-val_auc0.9984.ckpt`.
  Test accuracy at 0.5 was 0.9831, AUC 0.9971, and P_E 0.0157. The best
  global threshold was 0.2833, with P_FA 0.0128 and P_MD 0.0187. K=1 P_E was
  0.0437 for lowest and 0.0585 for plane.
- V3a training was about 6.7 iterations/s and 6.5 minutes per epoch (total
  training time 63.76 minutes). V3b was about 10.5 iterations/s, or 95 ms per
  training step, and 4 minutes per epoch (total 39.27 minutes). The PC power
  cap was about 255 W; this is a cap, not a measured mean power draw.

#### Batch dependence

The checks use the same seed-0 samples of 1,000 train pairs and 1,000
validation pairs as the earlier diagnoses. Mode a scores all sample images
in evaluation mode. Mode c uses 63 mixed 50/50 batches, with distinct covers
within each batch. Mode e uses all-clean batches with the same cover sampler.
The sampled aggregates need not match exactly: modes c and e can repeat
covers across batches.

| Model | Set | a: eval accuracy / loss | c: train-mode accuracy | e: all-clean accuracy / mean prob_stego |
| --- | --- | --- | --- | --- |
| V3a | Train | 0.9890 / 0.03486 | 0.9916 | 0.0169 / 0.9835 |
| V3a | Val | 0.9880 / 0.04339 | 0.9896 | 0.0218 / 0.9754 |
| V3b | Train | 0.9890 / 0.03476 | 0.9906 | 0.9955 / 0.0164 |
| V3b | Val | 0.9885 / 0.04053 | 0.9911 | 0.9965 / 0.0154 |

- Clipping alone closed the evaluation gap. V3a evaluation uses stored
  statistics, is per-image, and works well. But its train-mode BatchNorm still
  calls almost every tile in an all-clean batch stego.
- GroupNorm removes batch dependence. V3b's train/eval maximum logit
  difference on identical images was exactly 0 in all four checks. Its small
  detection cost appears mostly at k=1.

#### Test comparison

All rows use the same test split. K=1 columns retain the historical label
split: lowest and plane are the same k=1 embedding, not different methods.
Batch independence refers to normalization in both train and evaluation mode.
V2 k=1 values come from the VM's `logs/baseline_v2_vm.log`.

| Model | Accuracy at 0.5 | AUC | P_E | K=1 P_E lowest / plane | Batch independent? |
| --- | ---: | ---: | ---: | --- | --- |
| V1 epoch 5 | 0.839 | 0.989 | 0.0276 | 0.103 / 0.115 | No: train-mode BatchNorm |
| V2 epoch 1 | 0.5352 | 0.915 | 0.1148 | 0.3017 / 0.3101 | No: train-mode BatchNorm |
| V3a epoch 10 | 0.9843 | 0.9982 | 0.0120 | 0.0277 / 0.0503 | No in train mode; yes in eval mode |
| V3b epoch 9 | 0.9831 | 0.9971 | 0.0157 | 0.0437 / 0.0585 | Yes: GroupNorm |

V3a and v3b ran on different GPUs: L4 versus ROCm with the conv1 fix. One
seed does not separate hardware or run variation from the model changes.
Small differences (roughly +/-0.002 P_E) may not be real. Run-to-run noise
is not measured; this is a caution, not a confidence interval.

### Decided

- V3b is the current baseline candidate because it is batch-independent.
  (Later superseded: v3c became the baseline in the next entry.)
- V3c will test clip 3 plus GroupNorm: one change from v3b.

### Next

- Done: ran v3c and compared it with v3b on detection and batch checks. See
  the next entry.
- Done: wrote the [v3b results report](results-v3b.md), with training curves,
  test plots, setting breakdowns, and batch-independence checks.

## 2026-10-12 — V3c run and notebook monitoring (continuation)

### Did

- Updated notebook 04 in commit `4a890d5`.
  - Added TensorBoard 2.21.0 to `pyproject.toml` and `uv.lock`. The lock was
    made on the VM in a temporary folder. Only TensorBoard and its helper
    packages were added. After the push, teammates and the VM must run
    `uv sync`.
  - Added CSV and TensorBoard loggers to each run folder.
  - Kept existing CSV rows when resuming. Lightning deletes `metrics.csv` on
    logger re-initialisation unless the notebook preserves it.
  - Logged `val_fa` and `val_md` at each epoch.
  - Redrew `training_curves.png` after each epoch.
  - Added the all-stego check (f).
  - The resume check accepts epoch 9 or epoch 10 and checks that
    `global_step` is unchanged.
- V3c ran on the PC AMD RX 9070 XT with clip 3, GroupNorm, and the ROCm conv1
  fix enabled. It ran at about 10.7 iterations/s (about 93 ms/step) and took
  39.7 minutes for 10 epochs.
- Deleted the v3b and v3c logs on 2026-10-12 after recording results. (Later,
  `logs/` again held local task logs and `logs/v3a_legacy_from_vm/`.) Numbers from
  `logs/v3b_pc.log` and `logs/v3c_pc.log` remain in this journal,
  `docs/results-v3b.md`, and `docs/results-v3.md`, run outputs stay in
  `checkpoints/`, and notebook 04 now defaults to v3c.

### Found

#### V3c validation trend

The values below come from
`checkpoints/baseline_kv_xunet_v3_clip3_group/logs/csv/run/metrics.csv`.
Epoch numbers start at 1. Checkpoint filenames start at 0.

| Epoch | Validation loss | Validation P_E |
| ---: | ---: | ---: |
| 1 | 0.0818977430 | 0.0219456702 |
| 2 | 0.0696384236 | 0.0165261608 |
| 3 | 0.1285918653 | 0.0175297745 |
| 4 | 0.0673173890 | 0.0176635887 |
| 5 | 0.0529728606 | 0.0142513048 |
| 6 | 0.0452369750 | 0.0103037599 |
| 7 | 0.0399012156 | 0.0101030376 |
| 8 | 0.0411364138 | 0.0097015928 |
| 9 | 0.0397889987 | 0.0098354071 |
| 10 | 0.0385187827 | 0.0095677776 |

- The selected V3c checkpoint was
  `best-epoch09-val_pe0.0096-val_auc0.9985.ckpt`, human epoch 10. Test
  accuracy at 0.5 was 0.9858, AUC was 0.9982, and P_E was 0.0139. The best
  global threshold was 0.458815, with P_FA 0.008476 and P_MD 0.019239.
- V3c per-k P_E was:

  | k | Lowest | Plane |
  | ---: | ---: | ---: |
  | 1 | 0.031983 | 0.062628 |
  | 2 | 0.007642 | 0.007883 |
  | 3 | 0.003198 | 0.005123 |
  | 4 | 0.003394 | 0.000000 |
  | 5 | 0.002070 | 0.001012 |
  | 6 | 0.001147 | 0.000000 |
  | 7 | 0.004505 | 0.000000 |
  | 8 | 0.004751 | 0.000000 |

#### V3c normalization checks

The checks used the same labels as the earlier entries: a is evaluation
mode, c is mixed train-mode normalization, e is an all-clean batch, and f is
an all-stego batch. The values for c, e, and f are accuracy / mean
`prob_stego`; a is accuracy / loss.

| Set | a | c | e | f |
| --- | --- | --- | --- | --- |
| Train | 0.987000 / 0.035752 | 0.990079 / 0.502371 | 0.995040 / 0.019812 | 0.984127 / 0.982507 |
| Validation | 0.990500 / 0.041141 | 0.992560 / 0.502398 | 0.998016 / 0.018719 | 0.987599 / 0.985346 |

GroupNorm's maximum train/evaluation logit difference on identical images was
0 in every check.

#### V3 comparison

All rows use the same held-out test split and one seed. The all-clean values
are clean accuracy in train-mode all-clean batches, shown as train / validation.

| Model | Accuracy at 0.5 | AUC | P_E | Best threshold | K=1 P_E lowest / plane | All-clean accuracy train / val |
| --- | ---: | ---: | ---: | ---: | --- | --- |
| V3a clip 3 + BatchNorm | 0.9843 | 0.9982 | 0.0120 | 0.1646 | 0.0277 / 0.0503 | 0.0169 / 0.0218 |
| V3b no clip + GroupNorm | 0.9831 | 0.9971 | 0.0157 | 0.2833 | 0.0437 / 0.0585 | 0.9955 / 0.9965 |
| V3c clip 3 + GroupNorm | 0.9858 | 0.9982 | 0.0139 | 0.4588 | 0.0320 / 0.0626 | 0.9950 / 0.9980 |

Clipping with GroupNorm improved overall P_E from 0.0157 to 0.0139 and
lowest k=1 P_E from about 0.044 to about 0.032. It moved the best threshold
near 0.5. Plane k=1 P_E became about 0.063 instead of about 0.059. Each
version has one run and one seed. Run-to-run noise is not measured, so
differences of about 0.002 P_E may not be real.

#### Validation-loss spike

V3b and V3c both had a validation-loss spike at human epoch 3. The V3b CSV
reported val_loss 0.1210675091, and the V3c CSV reported 0.1285918653. The
V3a VM CSV at `/srv/aai3001/AAI3001-CVDL/checkpoints/lightning_logs/version_5/metrics.csv`
also had one spike, but at human epoch 4: val_loss rose from 0.0642 to 0.1146
and val_P_E rose from 0.0146 to 0.0203. All three runs recovered before or at
the learning-rate drop. A possible cause is that learning rate 1e-3 is too
high at that point. This was not tested.

### Decided

- V3c is the current baseline. It is batch-independent, has the best accuracy
  at 0.5, and has the best-calibrated threshold. Its P_E is 0.0019 higher
  than V3a; run-to-run noise is not measured. See the 2026-10-13 entry for the
  user's reason.

### Next

- Done: wrote the [V3 comparison report](results-v3.md).
- Pending: repeat the runs with other seeds to measure noise.
- Pending: test learning-rate warm-up for the epoch-3 spike.
- Pending: focus on plane k=1 at low bpp.
- Pending: inspect confidently wrong images, including clean images with
  `prob_stego` near 1.0.
- Pending: finish the TensorBoard chart layout and train/validation colour
  runs in notebook 04.
- Pending: review the [noisy-cover dataset plan](plan-noisy-covers.md).

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
- The slowest setting is random placement, 0.5 bpp (bits per pixel), match,
  and k=1 or plane mode: about 5.8 ms median. Plane k=8 is similar. Random
  placement, 0.5 bpp, and replace take about 4.3 ms.
- Correctness: 32 of 32 sampled tiles were byte-identical to the stored stego
  PNGs. The test used the original cover index, `SeedSequence([seed, index,
  variant, 0])` for the settings, and `[..., 1]` for the embedding.
- The embed engine uses vectorized NumPy. It always builds the mask and the
  `EmbedResult` object. Training does not need them.
- Caveats: the on-the-fly test reused one decode for the clean and stego
  twins in one batch. The notebook shuffles images independently, so it does
  not get this saving. The cache was warm, and no GPU training ran at the
  same time.

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

### Next

- Pending: the open items from the 2026-10-12 V3c entry are unchanged.
- Pending: the Worker is measuring embed -> tile on the fly on whole photos:
  speed, memory, and per-tile label statistics.
- Pending: write a generator design plan for both modes.
