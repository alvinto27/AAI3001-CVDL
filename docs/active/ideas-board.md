# Ideas board

Status: **list to review with the user.** No item is approved. Nothing here starts without a user decision. Created 2026-10-18.

Evidence for each item: [2026-10-18 journal](../journal/2026-10-18/README.md) (numbers on the new 256 suite, each model at its own validation cut-off).

## Checks (cheap, no training)

| # | Idea | Question it answers | Evidence so far | Cost | State |
| ---: | --- | --- | --- | --- | --- |
| 1 | **Classic detector on noisy ALASKA**: run Weighted Stego (WS) on the noisy ALASKA k=1 hard tiles, split by LSB replacement vs +/-1 matching | Is the stego signal there at all, or are these tiles near the limit? | All 11 models at chance on ALASKA k1 low+noisy (AUC .51-.52). Replacement and matching are missed equally (SRM-e50: 53-60% vs 58-60%), although replacement has a known fingerprint that WS uses | CPU only, ~1 h | User: yes, **after the user returns** |
| 2 | **Max / top-k cell pooling**: re-score existing checkpoints by the highest cell score (or mean of the top few) instead of the mean of all 8 x 8 cells; threshold on validation | Does averaging dilute packed (contiguous) payloads? | Contiguous k=1 at k=1 low bpp is a full-width band of ~11 rows (~4% of a 256 tile). SRM-e50 contiguous low-bpp misses: standard 12.2%, ALASKA seen 50.6%, unseen 45.5%, LSDIR 43.5%. Risk: one noisy clean cell can flag a clean tile (noisy ALASKA FA already 35-38%) | GPU inference only, ~10-15 min per model | To review |

Order if both are approved: 1 then 2. Item 1 decides whether noisy ALASKA is worth more work; item 2 decides whether cell-level training (item 3) is worth a G4 run.

## Experiments (need training)

| # | Idea | Why | Evidence / risk | Cost | State |
| ---: | --- | --- | --- | --- | --- |
| 3 | **Cell-level training**: supervise the 32 px cells with the known changed-pixel map | We generate the stego, so we know which cells changed. Related published work exists (selection-channel-aware steganalysis) | Helps packed payloads on ALASKA/LSDIR (many clean cells in a stego tile). Little new information for scattered low-bpp payloads (almost every cell holds a change). Does not address the noisy case. [Plan](plan-per-cell-training.md) is out of date (512 tiles, old tests, old pass marks) and needs a rewrite | Rewrite + code + 1 G4 run | On hold; do after item 2 |
| 4 | **Untangle `srm-hard`**: one run that changes only the k/bpp mix (keep 40/40/20 photos), or only the photo shares | `srm-hard` changed both; we cannot tell which caused the small k=1 gain and the higher ALASKA false alarms | vs SRM-30: hard k1 low+noisy AUC standard .905 vs .882, LSDIR .671 vs .651; ALASKA seen P_E 21.95 vs 20.03 | 1 G4 run, ~$3 | To review |
| 5 | **Seed repeats** (e.g. SRM-e50 x 2 extra seeds) | Several model gaps are under 1 point; one seed cannot show whether they are real | Pending since 2026-10-17 | 2 G4 runs, ~$6-9 | To review |
| 6 | Front-end upgrade / BF16 / longer training | Older pending items | Hold until items 1 and 5 are settled | - | On hold |

## Report topics (no work)

| # | Topic | Evidence |
| ---: | --- | --- |
| 7 | One deployment threshold costs ALASKA | SRM-e50 at its validation cut-off: ALASKA seen FA 16.3%, noisy ALASKA hard cells FA 35-38% |
| 8 | Training sources matter more than architecture for unseen data | wide-R2 (RAISE+DIV2K only) ALASKA seen P_E 32.65%, FA 44.4% |
