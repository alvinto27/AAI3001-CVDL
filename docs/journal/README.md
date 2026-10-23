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
- [2026-10-19](2026-10-19/README.md) — Report revisions 2 and 3; diagnostic check 1 (classic detectors find a signal SRM-e50 misses: raise #12) check 2 (no pooling gain: lower #3) check 2b (tanh fix struck); low-bpp curriculum runs 10-11 (small real gain, rule not met, #9 closed).
  Architecture arm A, delayed pooling (run 12): no gain, clearly worse (-0.060 pooled k=1 low AUC). PC pilot mode. Arm G (GBRAS-Net) waits for the user.
- [2026-10-23](2026-10-23/README.md) — Paired training: R1 PASS (hard k=1 low AUC +.012, standard same); R2 paired cell model FAIL on its heatmap rule but standard P_E 2.43% (best so far); 50-epoch paired curriculum FAIL (no gain over R1; one spot preemption). Slim G4 resume built and installed (`58aa081`).
- [2026-10-22](2026-10-22/README.md) — Arm G (GBRAS-Net) code (`f86dec1`, ROCm BatchNorm fix `741d2ef`), PC pilot (learns; batch 32 only on the PC), run 14 stopped on the G4 by the speed rule: about 7x SRM-30 compute (30 epochs about 9.5 h). Speed-ups measured on the G4: BF16 only 1.10x, channels-last none; 30 epochs still about 9-9.5 h (about $15-16). **Arm G closed (user).** Paired training (#16) queued for the next working day.
- [2026-10-21](2026-10-21/README.md) — Legacy-512 PC evaluation retired to runnable archives (`3c3dc39`, 407 to 370 tests); 512 end test off in the 3 baseline notebooks (`7921d58`); golden retired, comparison constants list added (`72f51b5`, 360 tests); laptop works by SSH into this PC.
- [2026-10-20](2026-10-20/README.md) — Review of pre-run checks: suite profiled (one loader test = 42% of 8.2 min); fast/full test levels and a changed-path helper built (`891aefe`); golden compare only when shared code changes, always before a G4 launch. Arm G paused. Then C (`a1ae22c`) and `TESTS: NONE` for docs-only changes (`c7b5176`); B not done (loader workers start/stop on purpose for RAM). Test trim `4959099` (445 to 407 tests; one-off check code archived). Paused; next: retire legacy-512.
