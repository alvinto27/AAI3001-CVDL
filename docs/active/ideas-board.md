# Ideas board

Status: **list to review with the user.** No item is approved. Nothing here starts without a user decision. Created 2026-10-18.

Evidence for each item: [2026-10-18 journal](../journal/2026-10-18/README.md) (numbers on the new 256 suite, each model at its own validation cut-off).

## Checks (cheap, no training)

| # | Idea | Question it answers | Evidence so far | Cost | State |
| ---: | --- | --- | --- | --- | --- |
| 1 | **DONE 2026-10-19 (`ed52419`): signal found; decision row "raise #12; keep #9".** **Classic detectors (WS + SPA) on the hard-test tiles**. Revised 2026-10-18 after external review: (a) same paired clean/stego tiles as the SRM-e50 hard test; (b) Weighted Stego and Sample Pair Analysis, each on R, G and B separately plus one combined statistic fixed in advance; (c) positive control first: k=1 replacement at higher bpp on easy tiles, to prove the detectors work on our data; (d) main cells: ALASKA (and LSDIR) k=1 replacement, 0.05-0.15 bpp, noisy; (e) matching samples as negative control; (f) report AUC with 95% intervals, ROC low-false-alarm region, and miss at a validation-selected 5% false-alarm threshold; compare with SRM-e50 on the same tiles | Does a known LSB-replacement statistic find signal where SRM-e50 fails? | All 11 models at chance on ALASKA k1 low+noisy (AUC .51-.52). SRM-e50 misses replacement and matching at similar rates (53-60% vs 58-60%); **this does not prove the CNN ignores the replacement fingerprint** (both change similar numbers of pixels; miss rates at one threshold can hide differences; ROC curves are needed). Note: at 0.05 bpp the payload is spread over R, G, B, so each channel carries only ~1.7% of values | CPU only, ~1-2 h | User: yes, **after the user returns** |
| 2 | **DONE 2026-10-19 (`9563540`): no gain; mean pooling best (AUC); decision "lower #3".** **Max / top-k cell pooling**: re-score existing checkpoints by the highest cell score (or mean of the top few) instead of the mean of all 8 x 8 cells; threshold on validation | Does averaging dilute packed (contiguous) payloads? | Contiguous k=1 at k=1 low bpp is a full-width band of ~11 rows (~4% of a 256 tile). SRM-e50 contiguous low-bpp misses: standard 12.2%, ALASKA seen 50.6%, unseen 45.5%, LSDIR 43.5%. Risk: one noisy clean cell can flag a clean tile (noisy ALASKA FA already 35-38%) | GPU inference only, ~10-15 min per model | To review |

| 2b | **DONE 2026-10-19 (`d3ae6b8`): tanh fix struck.** **Tanh saturation vs noise**: measure SRM tanh1 saturation per tile on the hard test and relate it to tile noise and to misses | Does SRM's first tanh flatten the weak LSB signal on noisy tiles? | Measured 2026-10-17 on **SRM-30** (not SRM-e50; old 512 validation, whole set only; `runs/verification/post-training-model-comparison-20261017.csv`): SRM tanh1 inputs beyond +/-3: 20.8% clean / 26.4% stego; KV models 0.5-0.7%. Not yet split by noise. Counter-evidence: KV models do not saturate and are equally at chance on ALASKA k1 low+noisy (overlap-256 AUC .512 vs SRM-e50 .522), so saturation is unlikely to be the main cause of the noisy failure; it may still cap SRM elsewhere. Existing tool: `evaluate_run.py --tanh-saturation` (needs a per-tile / hard-test extension). A fix idea exists (handoff 2026-10-17, "SRM tanh1 saturation fix (option C)", idea only) | GPU inference, ~15 min + small code change | To review |

**User 2026-10-19: run checks 1, 2 and 2b first** ([plan with decision rules](plan-diagnostic-checks.md)); conclusive results may strike experiments. Order: 1, then 2, then 2b. Item 2 decides whether cell-level training (item 3) is worth a G4 run.

**How to read item 1** (wording agreed 2026-10-18): if WS/SPA detect low-bpp replacement where SRM-e50 fails, that is an exploitable signal the CNN does not capture: a concrete feature target (for example, add the statistic as an input or combine the two detectors). If they fail too, that shows only that these detectors and our CNNs fail; it does **not** show that the information is absent (higher-order, cross-channel or camera-noise features might still work). Defensible wording: "Both the CNN and conventional LSB-replacement detectors failed to reliably detect weak embeddings in high-noise ALASKA images under the evaluated conditions." Small AUC differences (for example .53 vs .52) need intervals before they count.

## Experiments (need training)

| # | Idea | Why | Evidence / risk | Cost | State |
| ---: | --- | --- | --- | --- | --- |
| 3 | **Cell-level training** (**lowered** by check 2, 2026-10-19; external review agrees: lower priority; when done, also add a "where the model looks" cell-score picture to the report data figure, user 2026-10-19): supervise the 32 px cells with the known changed-pixel map | We generate the stego, so we know which cells changed. Related published work exists (selection-channel-aware steganalysis) | Helps packed payloads on ALASKA/LSDIR (many clean cells in a stego tile). Little new information for scattered low-bpp payloads (almost every cell holds a change). Does not address the noisy case. [Plan](plan-per-cell-training.md) is out of date (512 tiles, old tests, old pass marks) and needs a rewrite | Rewrite + code + 1 G4 run | On hold; do after item 2 |
| 4 | **Untangle `srm-hard`**: one run that changes only the k/bpp mix (keep 40/40/20 photos), or only the photo shares | `srm-hard` changed both; we cannot tell which caused the small k=1 gain and the higher ALASKA false alarms | vs SRM-30: hard k1 low+noisy AUC standard .905 vs .882, LSDIR .671 vs .651; ALASKA seen P_E 21.95 vs 20.03 | 1 G4 run, ~$3 | To review |
| 5 | **Seed repeats** (e.g. SRM-e50 x 2 extra seeds) | Several model gaps are under 1 point; one seed cannot show whether they are real | Pending since 2026-10-17 | 2 G4 runs, ~$6-9 | To review |
| 9 | **Low-bpp curriculum**: fine-tune from a strong checkpoint (e.g. SRM-e50) in steps toward k=1 and 0.05-0.15 bpp; compare with `srm-hard` (weighted sampling from scratch) | Established method: the SRNet paper trained low payloads by seeding from a higher-payload model (Boroumand, Chen, Fridrich, SRNet). Different from `srm-hard`, which only re-weighted sampling and mostly moved the threshold | Fine-tune run is shorter than a full run; needs a curriculum schedule and a resume-from-checkpoint path | 1 G4 run, ~$1.5-3 | **Chosen by the user 2026-10-19**; [plan](plan-low-bpp-curriculum.md) draft |
| 10 | **Weak-signal architecture ideas**: information-preserving pooling, learnable residual filters, delayed downsampling, attention (e.g. WSERNet, Liang et al. 2026; SRNet avoids pooling in its first layers) | Our XuNet pools 5 times; early pooling can lose a weak signal | Large change; most published results are grayscale, so RGB benefit is unproven | Design + code + several G4 runs | Later (external suggestion) |
| 11 | **RGB channel analysis**: embed in R only, G only, B only, and spread over RGB at the same total bpp; test whether explicit cross-channel features help | Our payload is spread over 3 channels (~1.7% of values per channel at 0.05 bpp); unknown whether the model uses cross-channel relations | Needs a new frozen test panel (generator channel option to check) and inference; pairs well with item 1's per-channel statistics | Build + GPU inference ~15 min per model | To review (external suggestion) |
| 12 | **CNN + statistical features** (**raised** by check 1, 2026-10-19): add classic replacement statistics as inputs, or a separate fusion classifier | Only if item 1 finds signal the CNN misses | Depends on item 1 | Code + 1 G4 run | Waits for item 1 |
| 6 | Front-end upgrade / BF16 / longer training | Older pending items | Hold until items 1 and 5 are settled | - | On hold |

## Evaluation fixes

| # | Idea | Why | State |
| ---: | --- | --- | --- |
| 13 | **Save the logit margin in the evaluation** (next to `prob_stego`) and use it for AUC and thresholds | float32 softmax saturates: 26% of SRM-e50 validation scores are exactly 1.0. With mean pooling these are almost all stego (52.6% of stego, 0.0% of clean exactly 1.0; clean >= 0.999: 0.1%), so clean-vs-stego ties are rare and AUC and the 0.45 threshold are barely affected. It matters for max/top-k pooling, where clean tiles can saturate too. Low priority for the main evaluation | To review (found in check 2, 2026-10-19) |

## Report topics (no work)

| # | Topic | Evidence |
| ---: | --- | --- |
| 7 | One deployment threshold costs ALASKA | SRM-e50 at its validation cut-off: ALASKA seen FA 16.3%, noisy ALASKA hard cells FA 35-38% |
| 8 | Training sources matter more than architecture for unseen data | wide-R2 (RAISE+DIV2K only) ALASKA seen P_E 32.65%, FA 44.4% |
