# Plan: low-bpp curriculum from SRM-e50 (ideas board #9)

Status: **done 2026-10-19: rule not met; small real gain recorded; #9 closed (see Result).** Approved by the user 2026-10-19: choices C1-C6 as recommended (including the C4 control run), and the success rule below, confirmed before any result. Code: Worker_3.

## Result (2026-10-19; user chose option B)

Runs: curriculum `srm-low-bpp-curriculum-20261019T051356Z-9566` (eval `e9ef059bb12c9076a226`), control `srm-low-bpp-control-20261019T060708Z-9736` (eval `af6e9879061fd441462f`). Decision evidence: `04415c0` (`scripts/curriculum_decision.py`), outputs `runs/verification/curriculum-decision-2026-10-19/`.

| Measure | SRM-e50 | Curriculum | Control |
| --- | ---: | ---: | ---: |
| Pooled hard k=1 low AUC [95%] | .745 [.735, .755] | .763 [.753, .773] | .744 [.733, .754] |
| Standard-256 P_E at own cut-off | 2.54% | 3.10% | 2.65% |
| Standard-256 AUC | .9964 | .9964 | .9964 |

Paired differences (same photo draws): curriculum - SRM-e50 +.018 [+.013, +.023]; curriculum - control +.020 [+.015, +.024]; control - SRM-e50 -.001 [-.005, +.002]. The gain is present in every placement and panel. Standard guard: +0.56 points [+0.28, +0.84] (limit +0.5).

**Literal rule: not met** (model intervals overlap; standard guard exceeded). The curriculum model is not adopted. **Recorded finding (user, option B; the rule's "Otherwise" wording is not accurate here):** the curriculum gives a small real gain on k=1 low bpp (+0.02 AUC, caused by the curriculum: the control gained nothing) at +0.56 points standard P_E. Training order helps a little but is not the main limit (ALASKA seen k=1 low stays at .70). #9 closed. No follow-up run.

Lesson: to compare models on the same test, use paired-difference intervals, not "model intervals do not overlap" (shared photo sampling makes non-overlap far too strict).

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
| C2 | **Keep some of the original mix** | ~~25% of each batch~~ exactly 25% of the stego tiles **per epoch** (seeded positions, so about 25% per batch on average) from SRM-e50's original embedding mix in every stage. Changed by the Orchestrator 2026-10-19 during the build, before any result: the existing loader does not build class-balanced batches, so an exact per-batch quota would need changes to the default loader path | Stops the model forgetting high bpp and k 2-4 (the standard panel must not get worse) |
| C3 | **Learning rate** | 3e-5 (one tenth of the original), warm-up 1 epoch, no drops | Fine-tuning needs a small LR; a large one erases what it knows |
| C4 | **Control run** | Yes: the same 15 epochs and LR, but with the original embedding mix only ("plain continue") | Without it, a gain could come from 15 more epochs, not from the curriculum. This is the `srm-hard` lesson: change one thing at a time |
| C5 | **Checkpoint choice** | Use the last epoch, not the best one | The best-on-validation epoch would favour the early, easy stages; validation (`60da...`) has the standard mix, not the low-bpp target |
| C6 | **Deployment threshold** | Min P_E on validation `60da...` (same as SRM-e50); frozen, as for all models | Same rule as every model |

Cost with C4: 2 G4 runs of about 50 min each, about $1.5 each, about $3 total. Without C4: about $1.5.

## Success rule (confirmed by the user 2026-10-19, before any result)

Lesson from check 2b: use AUC (threshold-free), not misses at one cut-off.

| Result | Meaning |
| --- | --- |
| Curriculum AUC on the hard k=1 `low` cells, pooled over ALASKA seen + unseen + LSDIR, is above BOTH SRM-e50 and the control (photo-bootstrap 95% intervals do not overlap), **and** standard-256 P_E is no more than 0.5 points worse than SRM-e50 | **Curriculum works** |
| Above SRM-e50 but not above the control | The gain comes from extra training, not from the curriculum |
| Otherwise | No gain; the k=1 low-bpp limit is not a training-order problem |

Measurement details fixed before the decision table (Orchestrator, 2026-10-19, after the per-panel evaluations were seen, before any pooled number or interval):
- Pooled AUC = one AUC over all pairs of the hard-256 k=1 `low` cells of ALASKA seen, ALASKA unseen and LSDIR (replacement and matching, all placements), from each evaluation's saved `hard_predictions.csv` (no new inference).
- 95% intervals: 2,000 photo-cluster bootstrap replicates, seed 20261019, the same photo draws for all three models. "Above" = intervals do not overlap (the rule). Paired differences with intervals are reported as context.
- Standard guard: the point difference in standard-256 P_E at each model's own validation cut-off (curriculum minus SRM-e50); its interval is context. Already known from the evaluations: 3.10 - 2.54 = +0.56 points, so the guard fails literally. The Orchestrator flags this to the user and does not override it.

Also reported (not deciding): AUC per panel and placement; ALASKA FA and misses at the frozen threshold; the noisy cells.

## Work needed (after approval)

1. Code (worker): start from a checkpoint's weights with a new optimiser; change the embedding settings per stage; a mix of "original" and "curriculum" embeddings per batch. Default training unchanged; golden ALL EQUAL.
2. One short PC smoke run (1 epoch per stage, small tiles count).
3. G4 runs (curriculum, then control), evaluation on standard-256 + hard-256, and the decision table.
