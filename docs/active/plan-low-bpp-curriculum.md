# Plan: low-bpp curriculum from SRM-e50 (ideas board #9)

Status: **draft, waiting for the user's choices** (2026-10-19). No code or runs yet.

## Question

Can SRM-e50 learn the weak k=1, low-bpp signal better if it starts from what it already knows and the payload is made smaller step by step?

## Why this is different from `srm-hard`

| | `srm-hard` | This plan |
| --- | --- | --- |
| Start | From scratch | From the SRM-e50 best checkpoint (epoch 43) |
| Order | Hard mix from epoch 1 | Easy to hard, in stages |
| Photo mix | Changed (RAISE / ALASKA / LSDIR 1/3 each) | Unchanged: SRM-e50's 40 / 40 / 20 |
| Result | Mostly moved the threshold; small k=1 AUC gain | To measure |

Reference method: SRNet trained low payloads by starting from a higher-payload model (Boroumand, Chen, Fridrich, "Deep Residual Network for Steganalysis of Digital Images", 2019).

## SRM-e50 starting point (from its saved run settings)

- 50 epochs, AdamW, LR 3e-4, warm-up 1 epoch, LR drops at epochs 8 and 33; batch 128.
- 180,000 tiles per epoch, 256 training grid, mix RAISE 0.4 / ALASKA 0.4 / LSDIR 0.2.
- Embedding: `configs/train-k1to4.yaml` (k 1-4 equally, bpp uniform 0.05-0.5, replace and match, three placements).
- G4: about 3 min per epoch.

## Choices for the user

| # | Choice | Recommendation | Why |
| --- | --- | --- | --- |
| C1 | **Stages** | 3 stages of 5 epochs: bpp 0.05-0.35, then 0.05-0.25, then 0.05-0.15; k=1 weight 1, then 2, then 3 (k 2-4 weight 1 each) | Gradual steps, like SRNet; ends at the hard target |
| C2 | **Keep some of the original mix** | 25% of each batch from SRM-e50's original embedding mix in every stage | Stops the model forgetting high bpp and k 2-4 (the standard panel must not get worse) |
| C3 | **Learning rate** | 3e-5 (one tenth of the original), warm-up 1 epoch, no drops | Fine-tuning needs a small LR; a large one erases what it knows |
| C4 | **Control run** | Yes: the same 15 epochs and LR, but with the original embedding mix only ("plain continue") | Without it, a gain could come from 15 more epochs, not from the curriculum. This is the `srm-hard` lesson: change one thing at a time |
| C5 | **Checkpoint choice** | Use the last epoch, not the best one | The best-on-validation epoch would favour the early, easy stages; validation (`60da...`) has the standard mix, not the low-bpp target |
| C6 | **Deployment threshold** | Min P_E on validation `60da...` (same as SRM-e50); frozen, as for all models | Same rule as every model |

Cost with C4: 2 G4 runs of about 50 min each, about $1.5 each, about $3 total. Without C4: about $1.5.

## Success rule (to confirm before any result)

Lesson from check 2b: use AUC (threshold-free), not misses at one cut-off.

| Result | Meaning |
| --- | --- |
| Curriculum AUC on the hard k=1 `low` cells, pooled over ALASKA seen + unseen + LSDIR, is above BOTH SRM-e50 and the control (photo-bootstrap 95% intervals do not overlap), **and** standard-256 P_E is no more than 0.5 points worse than SRM-e50 | **Curriculum works** |
| Above SRM-e50 but not above the control | The gain comes from extra training, not from the curriculum |
| Otherwise | No gain; the k=1 low-bpp limit is not a training-order problem |

Also reported (not deciding): AUC per panel and placement; ALASKA FA and misses at the frozen threshold; the noisy cells.

## Work needed (after approval)

1. Code (worker): start from a checkpoint's weights with a new optimiser; change the embedding settings per stage; a mix of "original" and "curriculum" embeddings per batch. Default training unchanged; golden ALL EQUAL.
2. One short PC smoke run (1 epoch per stage, small tiles count).
3. G4 runs (curriculum, then control), evaluation on standard-256 + hard-256, and the decision table.
