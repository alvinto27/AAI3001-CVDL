# Plan: architecture tests — delayed pooling and GBRAS-Net (ideas board #10)

Status: **approved by the user 2026-10-19**: arms A and G, choices D3-D6 as recommended, decision rule below (confirmed before any result). A 2-GPU VM and the seed repeat were considered and declined (user). Code: Worker_4. **Arm A done: no gain (see Results). ~~Arm G waits for the user's adapter choice.~~ Arm G resumed 2026-10-21 (user): adapter option 1 (learned 1x1, 90 -> 30); code next.**

## Questions

- **A.** Does our SRM model lose the weak signal because it pools right after the first learned layer?
- **G.** Does a newer network body behind the **same fixed 30-filter SRM bank** (GBRAS-Net) do better than our XuNet body?

User decisions 2026-10-19: keep the **fixed SRM front end** (no learned first layer; a ladder of later steps is below); test delayed pooling and GBRAS-Net first; ~~SRNet~~ dropped (learned front end); ~~Base 2 (seed repeat)~~ dropped by the user.

## Today's model (SRM-30, `srm-20261017T052857Z-4521`)

~~SRM front end (30 fixed filters per colour channel, 90 channels) -> conv1 5x5 -> **pool** -> conv2 5x5 -> pool -> conv3 1x1 -> pool -> conv4 1x1 -> pool -> conv5 1x1 -> pool -> mean -> classifier. Five 2x2 average pools (256 -> 8). Only conv1 and conv2 look at neighbours.~~ Corrected 2026-10-19: SRM-30 is `XuNetWide` (width 4, no normalisation): conv1 5x5 -> pool -> conv2 5x5 -> pool -> conv3 1x1 -> pool -> conv4 1x1 -> pool -> conv5 1x1 -> pool -> conv6 1x1 -> conv7 1x1 -> mean -> classifier. Arm A keeps conv6 and conv7.

## GBRAS-Net (Tabares-Soto et al., "GBRAS-Net: A Convolutional Neural Network Architecture for Spatial Image Steganalysis", IEEE Access 2021; code: github.com/BioAITeam/Steganalysis)

- Preprocessing: the 30 SRM filters, fixed. Body: depthwise and separable convolutions with skip connections; about 166k parameters.
~~- Grayscale in the paper. Our adaptation: our existing SRM front end (30 filters per colour channel, 90 channels) feeds the GBRAS-Net body; only the body's first layer input width changes. Recorded as an adaptation.~~ Not possible as written (Worker_4, from the authors' code at `0b40ab0`): the first GBRAS block adds a 30-channel skip from its input; 90 channels do not match. ~~**User choice pending:**~~ **User chose (1) on 2026-10-21:** (1) a learned 1x1 adapter 90 -> 30, then GBRAS as published (Orchestrator recommends); (2) widen the early GBRAS blocks to 90; (3) average R/G/B responses into 30 fixed channels. Our SRM front end and clamp stay unchanged (no GBRAS 3*tanh). Built from the authors' code only; the paper PDF was not reachable.
- The worker builds the body from the paper and the public code, with a citation and a layer-by-layer check against the code. No guessed layers.

## Arms

| Arm | Model | Change vs SRM-30 |
| --- | --- | --- |
| A | SRM-30 with the first two pools removed (conv1 and conv2 at full resolution; three pools later, 256 -> 32, then mean) | Only the pooling position |
| G | Our SRM front end + GBRAS-Net body | Only the body |

Both use the SRM-30 recipe: 30 epochs, AdamW LR 3e-4, warm-up 1, 180,000 tiles per epoch, RAISE 0.4 / ALASKA 0.4 / LSDIR 0.2, 256 overlap-last grid, `train-k1to4.yaml`, OTF. Known small difference: checkpoint validation `train_matched` with `60da...` (needed for the deployment threshold of new runs); SRM-30 selected its checkpoint on `capped`.

## Choices for the user

| # | Choice | Recommendation | Why |
| --- | --- | --- | --- |
| D3 | Batch size | 128 if it fits; if not, decide from the PC pilot and scale the LR by the batch ratio | Arm A keeps two layers at full resolution, so memory rises |
| D4 | Epochs and boot | 30 epochs if the PC pilot predicts the run fits the 180-min boot; if not, the user decides | No resume after G4 preemption yet |
| D5 | Checkpoint used | Best on validation (as SRM-30) | Same as the baseline |
| D6 | Minimum gain (no seed repeat) | +0.03 pooled AUC | Without a seed repeat we do not know the size of chance; demand a gain bigger than the curriculum's +0.02 |

## Decision rule (confirmed by the user 2026-10-19, before any result)

Main measure: pooled AUC over the hard-256 k=1 `low` cells of ALASKA seen + ALASKA unseen + LSDIR (as in the curriculum test). Guard: standard-256 AUC. Paired differences vs SRM-30 (eval `11a82bf812e573c7c62d`) on the same tiles: 2,000 photo-cluster bootstrap replicates, seed 20261019, the same draws for all models.

| Result | Meaning |
| --- | --- |
| Paired pooled AUC difference vs SRM-30 has a 95% interval above 0 **and** a point gain >= D6, **and** the standard-256 AUC difference interval is not wholly below 0 | **The arm helps** |
| Interval above 0 but gain < D6 | Possible small gain; inside likely chance (no seed repeat): no claim |
| Otherwise | No gain from this change |

Also reported (not deciding): AUC per panel and placement (scattered vs contiguous is the key context for A); P_E / FA / miss at each model's own threshold; training time.

## Results

Arm A result (2026-10-19, eval `9915836bfe1b229e5259`, evidence `9dfaa27`, `runs/verification/architecture-decision-2026-10-19/`): **no gain; clearly worse.** Pooled hard k=1 low AUC SRM-30 .739 -> delayed .679, paired difference -.060 [-.067, -.054]; standard AUC guard -.0023 [-.0030, -.0016] (wholly below 0). All three literal checks false. Worse in every panel (ALASKA seen -.055, unseen -.046, LSDIR -.067) and every placement (contiguous -.074, random -.047, stride -.057). P_E standard / ALASKA seen / unseen / LSDIR: SRM-30 3.07 / 20.03 / 17.31 / 9.63%, delayed 3.43 / 23.12 / 19.90 / 11.91%. Training 216 min vs 93 min. Caveat (not able to explain a drop this size): one seed; SRM-30 selected its checkpoint on `capped`, arm A on `60da...`. Conclusion: early pooling in this XuNet does not hide the weak signal; it helps.

## Later steps (only with evidence; user 2026-10-19)

1. Fixed SRM, better body (this plan).
2. SRM filters as a starting point, allowed to adjust (Zhu-Net style), only if step 1 stalls.
3. Learned filters constrained to high-pass (WSERNet style).
4. Fully learned front end (SRNet).

## Work needed (after approval)

1. Code (worker): arm A and the GBRAS-Net body as opt-in model options, with citations. Default models unchanged; golden ALL EQUAL. Evaluator support for both (cell maps if possible). **Arm A done (`1bb7e20`). Arm G waits for the adapter choice.**
2. PC pilot per arm: memory, steps per second, and a few hundred steps to show the loss falls. Predict the G4 time per epoch. **Arm A done** (the pilot over-predicted the time, 267 vs 216 min, because its driver synchronised every step; use `pilot=true`, `aec69d0`, for arm G).
3. User decides D3/D4 from the pilot, then G4 runs (one per boot), evaluation, and the decision table. **Arm A done** (batch 128, 30 epochs, 300-min boot; run 12; `scripts/architecture_decision.py`, `9dfaa27`).

Cost (estimates, before the pilot): A about $3-4 (actual about $6, 216 min); G about $2-4 (small model).
