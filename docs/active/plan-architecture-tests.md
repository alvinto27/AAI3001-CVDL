# Plan: architecture tests — delayed pooling and SRNet (ideas board #10)

Status: **draft, waiting for the user's choices** (2026-10-19). No code or runs yet.

## Questions

- **A.** Does our SRM model lose the weak signal because it pools right after the first learned layer?
- **B.** Does a published deep weak-signal model (SRNet) do better on our data than our SRM model?

## Today's model (SRM-30, `srm-20261017T052857Z-4521`)

SRM front end (30 fixed filters per colour channel) -> conv1 5x5 -> **pool** -> conv2 5x5 -> pool -> conv3 1x1 -> pool -> conv4 1x1 -> pool -> conv5 1x1 -> pool -> mean -> classifier. Five 2x2 average pools (256 -> 8). Only conv1 and conv2 look at neighbours.

## SRNet facts (Boroumand, Chen, Fridrich, "Deep Residual Network for Steganalysis of Digital Images", IEEE TIFS 2019)

- **No fixed filters.** SRNet learns its front end from random weights, end-to-end. It does not use SRM filters. (Other networks do: Yedroudj-Net and YeNet start from SRM filters.)
- 12 layers. The first 7 layers have **no pooling** (full resolution); pooling starts at layer 8; global average at the end.
- The paper is grayscale. On RGB the first layer takes 3 channels; the RGB benefit is not proven.
- The worker builds it from the paper (and the authors' public code, if it is available), with a citation. No guessed layers.

## Arms

| Arm | Model | Change vs SRM-30 |
| --- | --- | --- |
| A | SRM-30 with the first two pools removed (conv1 and conv2 at full resolution; three pools later, 256 -> 32, then mean) | Only the pooling position |
| B | SRNet as published (learned front end) | Whole architecture |
| Base 2 | SRM-30 again, seed 1 | Only the seed |

All arms use the SRM-30 recipe: 30 epochs, AdamW LR 3e-4, warm-up 1, 180,000 tiles per epoch, RAISE 0.4 / ALASKA 0.4 / LSDIR 0.2, 256 overlap-last grid, `train-k1to4.yaml`, OTF, checkpoint validation `train_matched` with `60da...`.

Why Base 2: two separately trained models differ by chance (random start). Without one repeat we cannot tell a small gain from chance. Base 2 also matches the new checkpoint-validation setting (SRM-30 used `capped`).

## Choices for the user

| # | Choice | Recommendation | Why |
| --- | --- | --- | --- |
| D1 | Arms | A, B and Base 2 | A tests the pooling idea; B tests a proven design; Base 2 sizes chance |
| D2 | SRNet front end | As published (learned). Hybrid "SRNet + our SRM filters" only later, if B is promising | One change at a time; the published form is the reference |
| D3 | Batch size for SRNet | Decide from the PC timing pilot (memory); if it must be smaller than 128, scale the LR by the batch ratio | SRNet keeps 7 full-resolution layers, so memory is much larger |
| D4 | Epochs and boot | 30 epochs if the PC pilot predicts the run fits the 180-min boot; if not, the user decides (fewer epochs or a longer boot limit) | No resume after G4 preemption yet |
| D5 | Checkpoint used | Best on validation (as SRM-30) | Same as the baseline |

## Decision rule (to confirm before any result)

Lessons from 2026-10-19: use paired differences on the same tiles, and compare with the size of chance.

Main measure: pooled AUC over the hard-256 k=1 `low` cells of ALASKA seen + ALASKA unseen + LSDIR (as in the curriculum test). Guard: standard-256 AUC. Each with 2,000 photo-cluster bootstrap replicates, seed 20261019, the same draws for all models.

| Result | Meaning |
| --- | --- |
| The arm's paired AUC difference vs **both** SRM-30 and Base 2 has a 95% interval above 0, **and** is larger than the SRM-30 vs Base 2 gap, **and** its standard-256 AUC difference interval is not wholly below 0 | **The arm helps** |
| Above one baseline only, or not larger than the seed gap | Inside chance: no claim |
| Otherwise | No gain |

Also reported (not deciding): AUC per panel and placement (scattered vs contiguous is the key context for A); P_E / FA / miss at each model's own threshold; training time.

## Work needed (after approval)

1. Code (worker): arm A as an opt-in model option; SRNet as an opt-in model with a citation; Base 2 needs only a seed setting. Default models unchanged; golden ALL EQUAL. Evaluator support for the new models (including cell maps if possible; arm A has a 32 x 32 grid).
2. PC pilot per arm: memory, steps per second, and a few hundred steps to show the loss falls. Predict the G4 time per epoch.
3. User decides D3/D4 from the pilot, then G4 runs (one per boot), evaluation, and the decision table.

Cost (estimates, before the pilot): A about $3-4; B about $5-10; Base 2 about $2.5.
