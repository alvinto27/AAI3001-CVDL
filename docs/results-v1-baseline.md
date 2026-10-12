# V1 baseline steganalysis results

Date: 2026-10-12

This report describes the first steganalysis baseline. It records one PC run
with one seed. The numbers are useful for diagnosis. They are not a final
claim about all images or all embedding methods.

## 1. Summary

- The test set had 7,433 clean/stego pairs, or 14,866 images.
- The checkpoint selected by validation loss gave P_E 0.0276 and AUC 0.989.
- The score ranking was strong, but the scores were badly calibrated at the
  default 0.5 threshold. Accuracy there was 0.839.
- Most settings were easy for this small 19.5 K-parameter CNN. The hard region
  was k=1-2 with bpp at or below 0.3.
- The run exposed a BatchNorm problem. The model could use the clean/stego twin
  in the same batch. V2 must test unpaired batches and fix this issue.

## 2. Setup

### Data

The data is in [`Datasets/raise512-run1`](../Datasets/raise512-run1/). It has
49,796 clean 512 x 512 RGB tiles and 49,796 stego tiles. The tiles came from
1,000 RAISE RAW photos from Nikon D7000, D90, and D40 cameras.

Each stego tile has one randomly selected setting:

| Setting | Values in this run |
| --- | --- |
| `bit_mode` and `k` | `lowest` or `plane`; `k=1..8` |
| `method` | `replace` or `match` |
| `placement` | `contiguous`, `stride` (2 to 16), or `random` |
| `payload_kind` | `random_bytes` or `letters` |
| `bpp` | 0.05 to 0.5 |

The split is 70/15/15 by **source photo**, stratified by camera. A source
photo never crosses a split. The test set is 7,433 pairs.

### Model and training

The model starts with a fixed **KV filter**. This is a high-pass filter that
keeps image changes and suppresses smooth content. It runs separately on each
colour channel. The result is called a residual. An Xu-Net-style CNN then
classifies the residual and the image features. The model has 19.5 K
parameters.

The input is raw 0-255 float32 data. Training used Adam with learning rate
1e-3, StepLR with a tenfold drop after epoch 5, batch size 16 twins (32
images), and flips or 90-degree rotations applied in the same way to both
twins. A **twin** is a clean tile and its stego version. They show the same
picture.

The PC run used an AMD RX 9070 XT. It stopped during epoch 9 of 10. Test
predictions used `epoch04-val_loss0.3209.ckpt`, the epoch-5 checkpoint with the
lowest validation loss. The checkpoint was chosen from validation data only.
The test score was therefore not used to choose the checkpoint.

## 3. Key terms for this experiment

- **Steganography** hides data in an image. **Steganalysis** looks for the
  hidden data without needing to read the message.
- A **cover** or **clean** image is the original. A **stego** image is the same
  image after data was embedded.
- **LSB** means least significant bit. It is the bit that changes the numeric
  pixel value by the smallest amount.
- `k` controls which bits are used. In `lowest` mode, `k` low bits are used. In
  `plane` mode, one bit plane is used: `k=1` is the LSB and larger `k` values
  select higher planes. Larger changes are easier to see.
- `replace` writes the target bits. `match` chooses a nearby valid pixel value
  that matches the target, so it usually changes the cover less.
- **Placement** controls where payload bits go. `contiguous` uses nearby carrier
  values, `stride` skips by a fixed step, and `random` chooses positions.
- **Payload kind** is either uniform random bytes or letters. It changes the
  bit pattern, not the image size.
- **bpp** means bits per pixel. A 512 x 512 tile has 262,144 pixels. In this
  dataset, 0.05 bpp is about 1.6 KB and 0.5 bpp is about 16 KB. Bpp counts
  pixels; each pixel has three colour values.
- **P_E** is the minimum, over thresholds, of `(P_FA + P_MD) / 2`. `P_FA` is a
  clean image called stego. `P_MD` is a stego image missed as clean. P_E=0 is
  perfect and P_E=0.5 is guessing. P_E is useful here because accuracy depends
  on one chosen threshold, while P_E shows the best balance of the two errors.
- A **global-threshold detection rate** uses one threshold for all groups. A
  per-group P_E may choose a different best threshold for each group. They are
  not the same measure and should not be compared as if they were.
- **Calibration** asks whether a score behaves like a probability. For example,
  scores near 0.8 should be stego about 80% of the time. A model can rank
  images well but still have poor calibration.

## 4. Overall result

| Measure | Result |
| --- | ---: |
| Test images | 14,866 |
| Test pairs | 7,433 |
| P_E | 0.0276 |
| AUC | 0.989 |
| Best global threshold | 0.981 |
| P_FA at that threshold | 0.010 |
| P_MD at that threshold | 0.045 |
| Accuracy at threshold 0.5 | 0.839 |

The clean-image `prob_stego` score quantiles were 0.277 at the median, 0.813
at the 90th percentile, and 0.982 at the 99th percentile. Thus the model
ranked the test images well, but many clean images received high scores.

At the global threshold, detection rates were 0.768 for lowest/k1 and 0.735
for plane/k1. The overall bit-mode rates were about 0.95 to 0.96. These rates
use one threshold. The per-setting P_E values below select a threshold for each
setting, so they answer a different question.

## 5. Where it fails: k and payload size

The aggregate P_E by bit mode and k was:

| k | lowest | plane |
| ---: | ---: | ---: |
| 1 | 0.103 | 0.115 |
| 2 | 0.031 | 0.027 |
| 3 | 0.020 | 0.008 |
| 4 | 0.009 | 0.004 |
| 5 | 0.007 | 0.002 |
| 6 | 0.005 | 0.000 |
| 7 | 0.010 | 0.000 |
| 8 | 0.007 | 0.000 |

P_E by bpp bin was 0.055 for 0.05-0.15, 0.023 for 0.15-0.30, and 0.008
for 0.30-0.50. The hardest heatmap cells were k=1 at low payload:

| Setting | 0.05-0.15 bpp | 0.15-0.30 bpp | 0.30-0.50 bpp |
| --- | ---: | ---: | ---: |
| lowest, k=1 | 0.174 (n=149) | 0.080 (n=144) | 0.043 (n=176) |
| plane, k=1 | 0.199 (n=138) | 0.085 (n=194) | 0.048 (n=155) |
| lowest, k=2 | 0.061 | 0.021 | 0.010 |
| plane, k=2 | 0.039 | 0.019 | 0.009 |

![P_E heatmap for lowest mode](figures/v1-baseline/pe_lowest.png)

*Figure 1. P_E heatmap for the `lowest` bit mode.*

![P_E heatmap for plane mode](figures/v1-baseline/pe_plane.png)

*Figure 2. P_E heatmap for the `plane` bit mode.*

The hard cases are also the more useful cases for a careful hider: low k and
small payloads. The model solves the easier settings, but that does not show
that it can detect a careful embedding.

## 6. What did not matter much

- `replace` and `match` gave P_E 0.028 and 0.027.
- `random_bytes` and `letters` gave P_E 0.026 and 0.029.
- `contiguous` and `random` gave P_E 0.018 and 0.026.
- Stride values ranged from 0.00 to 0.07. Each value had only about 150 pairs,
  so this result is noisy.

About 80% of settings were near-solved by this 19.5 K-parameter model. The
main separation was k and bpp, not payload text or method.

## 7. Calibration quirk

The best global threshold was 0.981, not 0.5. Accuracy at 0.5 was 0.839,
while AUC was 0.989. This means the ranking was useful, but the score values
were not reliable probabilities. The high clean-image quantiles show the same
problem.

For this reason, future runs must report both a threshold-free ranking measure
such as AUC and a threshold metric such as accuracy or P_E. They should also
report calibration when a probability threshold is used.

## 8. Training behaviour and the BatchNorm shortcut

The training log showed this pattern:

| Point | Validation result | Train accuracy |
| --- | --- | ---: |
| Epoch 3 | val_acc 0.598 | about 0.98 |
| Epoch 5 | val_loss 0.3209, the best checkpoint | about 0.98 |
| Epoch 6 | val_loss 2.28, val_acc 0.611 | about 0.98 |
| Epoch 8 | val_loss 29.6, val_acc 0.501 | about 0.98 |
| Epoch 9 | val_loss 33.7, val_acc 0.500 | about 0.98 |

BatchNorm uses statistics from a training batch, but running statistics in eval
mode. A VM check used 1,000 train pairs and 1,000 validation pairs:

| Set | Eval mode | Train-mode BatchNorm with twins together |
| --- | ---: | ---: |
| Train accuracy | 0.5125 | 0.969 |
| Validation accuracy | 0.506 | 0.963 |
| Loss | about 13 | 0.104 train, 0.113 validation |

Train and validation behaved the same in this check. That argues against simple
content memorisation. The model learned a shortcut: BatchNorm statistics let it
compare a twin with its partner. When scored alone, it was at chance and very
overconfident.

V2 is now at epoch 2 with twins in different batches, AdamW weight decay
1e-4, and checkpoint selection by validation accuracy. It has train accuracy
0.956, validation accuracy 0.533, and validation loss 4.31. This shows that
twin pairing is not the whole cause. The current suspect is the mismatch
between training-batch and running BatchNorm statistics, made worse by raw
0-255 input and varied tile textures. Diagnosis 2 is **Pending**.

## 9. What this means for next steps

- Focus future data and larger models on k<=2 and bpp<=0.3.
- Compare v2 with v1 on the same test split.
- Check accuracy at threshold 0.5 and calibration in v2.
- Fix or replace the BatchNorm behaviour before trusting a larger model.
- Use more seeds before drawing a final conclusion.

## 10. Limits and records

- This is one run and one seed. Treat the numbers as approximate.
- The test set comes from the same generated dataset family and has one random
  setting per stego tile.
- The v1 run stopped early in epoch 9. Test evaluation used the fair epoch-5
  validation-selected checkpoint, but later training behaviour is still useful
  only as a diagnosis.
- Each stride value has only about 150 pairs. Stride comparisons are therefore
  uncertain.
- Local prediction records are in
  `../checkpoints/baseline_kv_xunet/test_predictions.csv` and
  `../checkpoints/baseline_kv_xunet/test_predictions_per_setting.csv`. The
  checkpoint folder is ignored and is not part of the report commit.

Related files:

- [Project journal](journal.md)
- [Baseline notebook](../notebooks/01_baseline_kv_xunet.ipynb)
- [Dataset rebuild guide](rebuild-raise512-run1.md)
- [V1 PC notebook record](../checkpoints/baseline_kv_xunet/baseline_pc_9070xt.ipynb)
