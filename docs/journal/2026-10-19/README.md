## 2026-10-19 — Report revisions 2 and 3; diagnostic checks 1 and 2

Recorded by the Orchestrator.

### Did

- User reviewed the readable demo reports (`fa17f25`): much clearer.
- User question: train and validation curves almost overlap; is validation unseen? Orchestrator check on saved records: `srm-hard` training photos (`data_run_settings.json` `train_photo_selection`, same as `train_grid_plan.jsonl`: RAISE 701 / ALASKA 13,500 / LSDIR 7,000) share **0** photos with its validation list (150 / 1,000 / 1,485). All standard-256 and hard-256 pair lists (RAISE 149, DIV2K 100, ALASKA 3,700, LSDIR 2,500 photos) share 0 photos with training and validation. SRM-30 training vs list `60da...`: 0 shared. (`data/hard-test/256-v1/noise-val.csv` holds val tiles by design: noise edges only.) Close curves are expected: stego is re-embedded every epoch (no memorisation of stego), ~216,000 pairs per epoch, validation matches the training mix. srm-hard epoch 29: train loss 0.319 / acc 83.8%, val 0.330 / 83.2%. Not checked: near-duplicate photos across splits (for example RAISE bursts).
- Check 1 (WS + SPA) done by Worker_3: `ed52419` (`cvdl/classic_lsb.py`, `scripts/check1_classic.py`, tests; translations of the Binghamton `ws.m` / `sp.m` reference code). Positive control WS 0.994 / SPA 0.984 (SRM-e50 0.997, same 416 pairs). WS interval wholly above SRM-e50 in 10 of 12 noisy ALASKA k=1 replacement cells (SPA 9); largest gap on scattered low-bpp ALASKA (random: WS 0.814 vs SRM-e50 0.608). Matching controls at chance for WS/SPA; SRM-e50 equal on replacement and matching, so it does not use the replacement fingerprint. Validation thresholds do not transfer to noisy tiles (WS test FA up to 38.7%). 368 tests pass; golden ALL EQUAL. Details: [plan, check 1 result](../../active/plan-diagnostic-checks.md).
- Check 2 first pass (Worker_3): mean-pooling parity exact (max abs 7.2e-7 over 144,672 images). Raw result mean / max / top-4: contiguous target misses 46.5 / 28.9 / 41.9%; pooled noisy-clean FA 22.5 / 19.7 / 6.2%; validation P_E 10.3 / 32.3 / 24.5%. Not decided: float32 softmax saturation (validation exactly 1.0: 26 / 70 / 58%) leaves the threshold at 1.0. User: re-score all three on the float64 logit margin (option A). Board item 13 added: save the logit margin in the evaluation.
- Check 2 done (Worker_3, `9563540`, float64 margins): mean / max / top-4 validation P_E 10.29 / 16.09 / 14.68%; target contiguous low k=1 misses 46.5 / 59.6 / 62.2%; noisy-clean FA 22.5 / 0.8 / 0.4%; AUC mean best on every group (target contiguous .839 vs .771 / .781). Decision row: lower #3 (not strike). The earlier Orchestrator idea that mean pooling dilutes packed payloads is not supported. 379 tests; golden ALL EQUAL.

### Found

- **Check 1:** LSB replacement leaves a detectable fingerprint that SRM-e50 does not use. WS separates replacement stego from clean on noisy ALASKA (AUC 0.66-0.69) where SRM-e50 is at 0.54-0.60. SRM-e50 scores replacement and matching about the same. The earlier phrase "noisy ALASKA is at chance" holds for SRM-e50, not for the replacement signal itself.
- **Check 1:** the WS advantage is largest on scattered payloads (random / stride). On contiguous bands it is a tie on ALASKA, and SRM-e50 is better on LSDIR (0.894 vs 0.753).
- **Check 1:** a better AUC is not yet a better detector. The WS score drifts with noise: thresholds set on validation give up to 38.7% false alarms on noisy test tiles. Any gain needs a combined model (#12).
- **Check 2:** float32 softmax saturates. 26% of SRM-e50 validation scores are exactly 1.0 (mean pooling), and 70% under max pooling. For mean pooling the 1.0 scores are almost all stego (52.6% of stego, 0.0% of clean; clean >= 0.999: 0.1%), so the main AUC and the 0.45 threshold are barely affected. Ties between clean and stego break thresholds only under max / top-k pooling.
- **Check 2:** mean pooling does not dilute packed payloads. Mean is best on every group, AUC with separate intervals (target contiguous .839 vs max .771 / top-4 .781). The single cell scores cannot tell noisy clean cells from stego cells: max pooling needs a very high threshold, so noisy-clean FA falls (22.5% to 0.8%) but misses rise. The Orchestrator's earlier "dilution" explanation (B1 diagnosis) is not supported.
- **Check 2:** with mean pooling and the one threshold, 30-38% of clean noisy ALASKA tiles are false alarms (hard `noisy` / `low_noisy` cells), against 6-8% on LSDIR and 3-4% on the standard panel. This is the cost of one threshold (report topic #7).
- Overconfidence question (user): a threshold depends only on score order, so extreme probabilities matter only when the model is confidently wrong. Mean pooling is not. Check 2 shows single cells are confidently wrong on noisy clean cells. Cell-level training (#3) is the fix if that becomes the goal.

### Decided

- User: remove the "Look at: ..." line from every report section (obnoxious; most repeat what the chart shows). Keep the question headings; move useful hints into the charts (legend entry or short note). Recorded in the [readability plan](../../active/plan-report-readability.md), revision 2. No other report changes for now.
- Check 2 pooling definition (Orchestrator, before results): rank cells by the stego-minus-clean logit margin; max = top cell, top-4 = top 4 cells; average their paired logits. Not per-class max (it can mix cells).
- Check 2 rule scope (Orchestrator, before results): misses pooled over ALASKA seen + unseen + LSDIR, and no single panel worse; noisy-clean FA pooled over the same panels; validation P_E at the pooling's own threshold.
- User (option A, after the float32 first pass, before any decision): re-score all three poolings on the float64 logit margin. First pass kept as superseded.
- Check 2b scope (Orchestrator, before results): SRM-e50 decides, overlap-256 is contrast. Spearman of the saturated share vs `tile_kv_std` with interval above 0, and the noisiest quarter more saturated. Within noisy target cells, the most-saturated quarter is missed >= 10 points more. Equal-count rank quarters, tie-break by id, quarters fixed in the bootstrap.
- Board: #12 raised (check 1); #3 lowered, not struck (check 2); #13 added (save the logit margin in the evaluation; low priority).
- User: add a data sample figure as the first report section (revision 3): built once from real frozen test tiles (photos with the tile grid, clean vs stego tile, amplified change maps for contiguous / random / stride at k=1 low bpp, one noisy ALASKA tile). "Where the model looks" waits for cell-level training.

### Next

- ~~Worker_3: build revision 2~~ done: `c1ca714` (Look-at lines removed; hints in charts; 358 tests).
- ~~Then revision 3~~ done: `675c712` (data sample figure `docs/figures/report/data-sample.png`, built once by `scripts/build_data_figure.py` from 7 frozen test pairs; clean and stego pixels equal the evaluator's for all 7; `--check` byte-identical; 360 tests). Report work complete.
- Check 2b (tanh saturation vs noise): Pending, with Worker_3 (GPU granted).
- User: run the three checks (1 WS+SPA, 2 highest-cell pooling, 2b tanh saturation) before any experiment; [plan](../../active/plan-diagnostic-checks.md); decision rules confirmed by the user before any result. Order (user): finish the report work (revisions 2 and 3) first, then the checks.
- Ideas board items: waiting for the user's choice ([ideas board](../../active/ideas-board.md)).
