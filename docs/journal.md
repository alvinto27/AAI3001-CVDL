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
  [docs/plan-otf-generator.md](plan-otf-generator.md), commit `4d1036b`. It is
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
    [docs/frozen-source-records.md](frozen-source-records.md).
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
  record is [docs/source-verification.md](source-verification.md).
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
  [docs/settings-and-seeds.md](settings-and-seeds.md).
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
  [docs/embed-core-verification.md](embed-core-verification.md).
  - `lsb_engine.py` now has one internal embed core. `embed()` keeps its
    signature and returns `(stego, mask, EmbedResult)` unchanged.
  - New `embed_fast()` returns `(stego, metadata dict, changed unit indices
    or None)`. It has no mask. It computes changed indices only when asked,
    and keeps the `units_changed` count. It always copies the clean carrier;
    there is no in-place option.
  - Check script: `scripts/check_embed_fast.py`. Tests:
    `tests/test_embed_fast.py`.
- The Worker did plan step 5. Commit `fc87b9c`. The full record is
  [docs/mode1-verification.md](mode1-verification.md).
  - `LSB-Dataset-Generator/lsb_mode1.py` provides `make_sample()` and
    `replay_legacy()` for clean or stego samples. It uses `lsb_source`,
    `lsb_settings`, and `lsb_engine.embed_fast`; it does not copy their
    logic.
  - `lsb_source.py` gained `tile_at()` for direct tile lookup, sharing grid
    checks with `iter_tiles`.
  - Check script: `scripts/check_mode1.py`. Tests:
    `tests/test_lsb_mode1.py`.
- The Worker did plan step 6. Commit `7e89fec`. The full record is
  [docs/cache-verification.md](cache-verification.md).
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
  [docs/torch-loader-verification.md](torch-loader-verification.md).
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
    `docs/frozen-source-records.md` and the root README.
  - `masks/` and `records/` are not kept in git. Masks can be regenerated.
- The Worker did plan step 8. Commit `f3dca17`. The full record is
  [docs/notebook-otf-verification.md](notebook-otf-verification.md).
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
  modes: [docs/plan-otf-generator.md](plan-otf-generator.md) (`4d1036b`).
- Done: the Worker updated the plan for overlapped tiles (`18ef1fa`).
- Done: the user answered the open decisions in the plan. See the generator
  decisions above.
- Done: the Worker recorded these decisions in the plan (`e96ad9f`).
- Done: moved the VM checkout to `yx`.
- Done: plan step 1. Froze the RAISE and DIV2K photo and tile records, and
  the DIV2K split CSV (`63e4f59`).
- Done: step 2, the source step (`a7feb63`). See
  [docs/source-verification.md](source-verification.md).
- Done: step 3, the shared settings and seed helpers (`967cfd4`). See
  [docs/settings-and-seeds.md](settings-and-seeds.md).
- Done: v2 conv1 fix implementation (`60f5c17`).
- Done: step 4, one embed core plus a mask-free fast path (`c2bdb53`). See
  [docs/embed-core-verification.md](embed-core-verification.md).
- Done: step 5, mode 1 in the shared pipeline (`fc87b9c`). See
  [docs/mode1-verification.md](mode1-verification.md).
- Done: step 6, the full-photo `.npy` cache (`7e89fec`). See
  [docs/cache-verification.md](cache-verification.md).
- Done: step 7, the mode 1 loader (`970750a`). See
  [docs/torch-loader-verification.md](torch-loader-verification.md).
- Done: committed the compressed run1 labels and metadata to `data/raise/`
  (`8c4e549`).
- Done: step 8, notebook 04 connected to the OTF loader (`f3dca17`). See
  [docs/notebook-otf-verification.md](notebook-otf-verification.md).
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

## 2026-10-14 — PC cache, diagnostics, and compile benchmark

### Did

- The Worker built a PC cache after the user's decision that a cache inside
  the checkout is allowed only when git ignores it. Commit `e78a1a0`.
  - Path: `Datasets/otf-cache/v1` in the container, or
    `/work/Datasets/otf-cache/v1`.
  - Eight workers built 1,900 photos in 170.4 s. Size was 52.76 GB.
  - Verification passed with 0 mismatches. The 1,900 JSON records equal the
    VM cache except for `master_mtime_ns`; aggregate SHA-256 was
    `bde842a0...e482`.
  - Git was installed in container `aai3001`. This is lost if the container
    is recreated.
- The Worker fixed the OTF diagnostic in commit `ff06623`. It reads each
  selected photo once and keeps only sampled tiles. The printed CHECK lines
  stayed identical.
- The Worker benchmarked `torch.compile` in commit `dd8cb60`, with advice
  from ROCm-Works in `rocm-perf/notes/compile_advice_nb04.txt`.
  - `V3_COMPILE=off` is the default; `default` enables compile. Compile is
    in place just before fit. Checkpoints keep the same keys and resume in
    both directions.
- Worker_2 completed fix 5 in commit `cc3aeed`, after the original Worker
  session ended during the task. Details are in
  [docs/notebook-otf-verification.md](notebook-otf-verification.md).
  - Evaluation gives each worker whole photos with `PhotoPairTasks` and
    `OrderedPairLoader`. The parent restores the original pair order and
    batch boundaries. It uses 4 non-persistent workers; 0 workers also works.
    Early close drains only in-flight tasks.
- Training workers now shut down after fit. This freed about 10.0 GB of PC
  RAM for 8 workers.
- The user ran short PC notebook 04 tests with commit `dd8cb60`,
  `V3_COMPILE=default`, `QUICK=False`, and the RX 9070 XT. The user stopped
  them by hand and deleted the outputs afterward.
- The Worker limited compile helper threads in commit `22e61a9`. With
  `V3_COMPILE=default`, the notebook sets
  `TORCHINDUCTOR_COMPILE_THREADS=4`; an explicit value overrides it.
- The Worker completed Mode 2 steps 18 and 19 in commit `e84df32`. The full
  record is [docs/mode2-verification.md](mode2-verification.md).
  - New module: `LSB-Dataset-Generator/lsb_mode2_torch.py`. Each worker owns
    whole photos, reads and embeds each photo once, and yields uint8 tiles
    with actual-change labels and side records through a bounded 1,024-image
    shuffle.
  - Fixed evaluation keeps every validation/test tile in fixed order.
    `BagLoader` gives one bag per photo, the real tile count, no padding,
    per-tile labels, and photo-level fields. A photo is label 1 when any tile
    changes.
  - Manifest maxima are 70 / 260 for RAISE and 16 / 64 for DIV2K at 512 /
    256. Step 17, the pre-built Mode 2 writer, is deferred. No MIL model or
    notebook change was made.

### Found

- The diagnostic fix reduced times as follows:

  | Run | Before | After |
  | --- | ---: | ---: |
  | QUICK PC | 194.1 s | 17.3 s |
  | Full VM cache | 371.6 s | 100.5 s |
  | Full PC decode | >45 min, incomplete | 464.6 s |

- PC breakdown from Notebook 02 on the PC predictions (scratch output):
  overall P_E 0.017296, AUC 0.997441; RAISE 9,728 pairs P_E 0.011513;
  DIV2K 1,228 pairs P_E 0.061889.
- Compile correctness against eager gradients was 2.4e-7 on the PC and
  2.4e-7 on the L4 with TF32 off in the check. With native TF32 on, the L4
  difference was 4.9e-4, a TF32 effect. QUICK compile smoke runs passed on
  both PC and VM.
- Steady images/s, eager -> compiled:

  | Case | Eager | Compiled |
  | --- | ---: | ---: |
  | PC decode, 8 workers | 380.3 | 474.9 |
  | PC cache, 6 workers | 379.5 | 462.2 |
  | PC GPU-only | 369.2 | 457.8 |
  | L4 cache, 6 workers | 279.6 | 332.4 |
  | L4 GPU-only | 287.5 | 348.8 |

- The first compiled step took 14.9 s on the PC and 30-36 s on the L4.
- Steady compiled PC tests, as shown by the progress bar:

  | PC test | Training speed | Validation per epoch |
  | --- | ---: | ---: |
  | Decode, 6 workers, epoch 1 | 13.54 it/s (about 433 images/s) | about 2 min |
  | Cache, 6 workers, epoch 0 | over 13 it/s (partly warm) | about 20 s |
  | Cache, 6 workers, epochs 1 / 2 | 13.85 / 13.87 it/s (about 443 images/s) | about 20 s |

- During the cache run, the GPU was 100% busy at 251-255 W, 85-86 C,
  and 2,940-3,120 MHz. Each of 6 cache workers used about 21% CPU. The GPU
  limits the long run at about 13.86 it/s, below the short GPU-only result
  of 14.3 it/s (about 458 images/s).
- Cache-test validation P_E for epochs 0-2 was 0.0353, 0.0332, and 0.0247.
- Estimated full PC time with cache and compile is about 4:03 training plus
  0:20 validation per epoch, or about 44 min for 10 epochs. Decode without
  compile is about 7:20 per epoch, or about 73 min for 10 epochs.
- Sixteen idle `torch.compile` helper processes stayed alive. Each used about
  410 MB RSS, and available PC RAM was about 6.3 GB.
- Compile-helper limit results, PC cache, 6 workers, 300 steps:

  | Measure | Before | After |
  | --- | ---: | ---: |
  | Compiler processes | 17 | 5 |
  | Cold summed RSS | 9.44 GB | 3.03 GB |
  | Cold summed PSS | 1.30 GB | 0.80 GB |
  | Warm summed PSS | 0.53 GB | 0.49 GB |
  | Warm images/s | 471.1 | 473.9 |
  | Cold images/s | 409.9 | 417.9 |
  | Warm first step | 5.4 s | 5.1 s |

  Most RSS was shared pages. The real cold saving is about 0.5 GB.
- Mode 2 parity used 6 real photos, decode and cache sources, train/val/test,
  and 0, 2, and 4 workers. There were 0 pixel, record, label, or bag
  mismatches against `lsb_mode2.iter_photo_tiles`. Mode 1 was unchanged:
  200 batches and 6,400 images had fingerprint `32bd1035...9075` before and
  after.
- Full train epoch at 512, cache, 6 workers, 1,381 photos and 107,900 tiles:
  PC 127.5 s (846 images/s, peak PSS 5.73 GB); VM 252.0 s (428 images/s,
  peak PSS 5.27 GB).
- Natural label-1 share at 512 was 27,008/91,348 (29.57%) for RAISE,
  5,870/16,552 (35.46%) for DIV2K, and 32,878/107,900 (30.47%) combined.
  No rebalancing or class weights were used.
- Sample benchmark, 128 photos and 9,808 tiles in 256 bags, tile images/s /
  bags/s:

  | Source | 4 workers | 6 workers | 8 workers |
  | --- | ---: | ---: | ---: |
  | PC cache | 512.1 / 14.9 | 505.7 / 14.9 | - |
  | PC decode | - | - | 326.0 / 9.0 |
  | VM cache | 268.5 / 8.6 | 286.9 / 8.2 | - |
  | VM decode | - | - | 136.1 / 3.8 |

#### Evaluation worker fix (fix 5)

- At 512 with 4 workers, photo reads and time changed as follows:

  | Case | Photo reads before -> after | Time before -> after | Peak worker PSS before -> after |
  | --- | ---: | ---: | ---: |
  | PC decode val | 810 -> 270 | 135.6 -> 47.2 s | 2.41 -> 2.81 GB |
  | PC decode test | 745 -> 249 | 133.8 -> 46.6 s | 2.33 -> 2.78 GB |
  | PC cache val | 810 -> 270 | 19.9 -> 23.6 s | 2.08 -> 2.46 GB |
  | VM cache val | 810 -> 270 | 68.6 -> 74.9 s | 1.88 -> 2.27 GB |

- The cache slowdown is from parent re-stacking (5.4 of 17.0 s). It was
  not changed because safe copies are required.
- Validation (11,238 pairs) and test (10,956 pairs) were byte-identical
  before and after. Decode/cache, 512/256, and 0/2/4 workers were identical.
  Train-eval 1,000 pairs were identical, with 554 reads instead of 1,000.
  `test_predictions.csv` from best epoch 7 (21,912 rows) was byte-identical.
  The Mode 1 fingerprint `32bd1035...9075` was unchanged.
- QUICK PC cache passed all 15 cells in 55.4 s. Evaluation shutdown took
  0.47-0.65 s, with no leftover workers. The Seoul Spot VM was stopped twice
  on 2026-10-14 and restarted once in between, up at 16:23 +08. The first
  restart failed because the zone had no L4 capacity. Spot availability was
  Limited in Seoul (11-20% preemption, $0.39/h), High in Taiwan ($0.60/h),
  and Medium in Singapore (20%+, $0.63/h). Nothing was lost.

### Decided

- The cache inside the PC checkout is allowed only when git ignores it.
- PC run default (user, 2026-10-14): use the cache at
  `/work/Datasets/otf-cache/v1`, 4 training workers, and
  `V3_COMPILE=default`. Four workers are not measured yet; confirm them in
  the next full PC run. The expected speed is about 13.86 it/s.
- Plan steps 15, 16, 18, and 19 are Done. Step 17 is deferred. Mode 2 is
  on-the-fly only for now.
- Fix 5 is Done for evaluation (`cc3aeed`).

### Next

- Pending: confirm the 4-worker PC setting in the next full PC run.
- Done: limited compile helper processes with `TORCHINDUCTOR_COMPILE_THREADS=4`
  (`22e61a9`).
- Pending: MIL batching and model design, when photo-level work starts; the
  user will decide then.
- Pending: remove VM scratch `~/otf-dev/eval-yx` when the VM is back.
- Pending: user decision on moving the VM to Taiwan if Seoul stays Limited.
- Pending: user decision on the cache-mode slowdown: accept it, or use the
  old evaluation loader when a cache is set.
- Pending: fix 4 (background join and pinned copy).

- Done (user, 2026-10-14): told teammates that the VM is on branch `yx`,
  the `raise512-run1` images are gone on the VM, and `V3_DATA=png` there
  needs a rebuild.
