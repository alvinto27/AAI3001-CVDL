# Dated project journal

The dated entries were moved from the former single journal file without deleting their history. Each day keeps its original entries in order. Open [status](../status.md) for the current work hold; old “Next” items below are historical, not permission to run them.

- [2026-10-11](2026-10-11/README.md) — Built the RAISE v6 dataset and the first XuNet baseline.
  Recorded tile and replay counts, camera splits, and the initial VM rebuild failure.
- [2026-10-12](2026-10-12/README.md) — Rebuilt the dataset and diagnosed the V1/V2 BatchNorm issue.
  Ran V3a, V3b, and V3c. **All three** dated continuation entries remain in this day's file, in order.
- [2026-10-13](2026-10-13/README.md) — Recorded OTF generation, source records, cache and loader contracts.
  The long entry also keeps its ROCm corrections, measurements, and decisions.
- [2026-10-14](2026-10-14/README.md) — Checked PC cache, loader diagnostics, compile behavior, and G4 planning.
  The measured rates and open hypotheses remain in the original entry.
- [2026-10-15](2026-10-15/README.md) — Recorded G4 priming, R1/R2 and wide XuNet results, and source downloads.
  Speed work and the late-output fix remain here; see also the [wide result](../results/result-wide-xunet.md).
- [2026-10-16](2026-10-16/README.md) — Recorded final source data, the mixed G4 run, and Pi swap changes.
  The corrected old-model ALASKA 5%-FA finding is in the [check B result](../results/result-check-b.md).
  Afternoon: per-cell design written (not approved), all research on hold, docs reorganised.
  Then bookkeeping: one `runs/` folder (best + latest checkpoints only), `Datasets/originals` +
  `generated` on PC and G4 (137 GB freed), code tidy-up (122 tests), model-copy parity proven,
  notebook refactor started.
- [2026-10-17](2026-10-17/README.md) — Training tiling complete (6 arms; [result](../results/result-training-tiling.md)): overlap best, 256 tiles better.
  One evaluation tool (`evaluate_run.py`), temperature and padding checks, loader profile, SRM run on the G4.
- [2026-10-18](2026-10-18/README.md) — Project survey; train-matched validation built (`131492b`; 10% of training pairs per epoch).
  Leftover cleanup and the threshold-label check approved. Then: standard and hard 256 tests, one validation threshold, `srm-hard` run, all 11 models rescored, report notebook, ideas board.
- [2026-10-19](2026-10-19/README.md) — Report review: remove the "Look at" lines (revision 2).
