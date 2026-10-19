## 2026-10-19 — Report review, revision 2

Recorded by the Orchestrator.

### Did

- User reviewed the readable demo reports (`fa17f25`): much clearer.
- User question: train and validation curves almost overlap; is validation unseen? Orchestrator check on saved records: `srm-hard` training photos (`data_run_settings.json` `train_photo_selection`, same as `train_grid_plan.jsonl`: RAISE 701 / ALASKA 13,500 / LSDIR 7,000) share **0** photos with its validation list (150 / 1,000 / 1,485). All standard-256 and hard-256 pair lists (RAISE 149, DIV2K 100, ALASKA 3,700, LSDIR 2,500 photos) share 0 photos with training and validation. SRM-30 training vs list `60da...`: 0 shared. (`data/hard-test/256-v1/noise-val.csv` holds val tiles by design: noise edges only.) Close curves are expected: stego is re-embedded every epoch (no memorisation of stego), ~216,000 pairs per epoch, validation matches the training mix. srm-hard epoch 29: train loss 0.319 / acc 83.8%, val 0.330 / 83.2%. Not checked: near-duplicate photos across splits (for example RAISE bursts).

### Decided

- User: remove the "Look at: ..." line from every report section (obnoxious; most repeat what the chart shows). Keep the question headings; move useful hints into the charts (legend entry or short note). Recorded in the [readability plan](../../active/plan-report-readability.md), revision 2. No other report changes for now.
- User: add a data sample figure as the first report section (revision 3): built once from real frozen test tiles (photos with the tile grid, clean vs stego tile, amplified change maps for contiguous / random / stride at k=1 low bpp, one noisy ALASKA tile). "Where the model looks" waits for cell-level training.

### Next

- Worker_3: build revision 2 and regenerate both demo reports (CPU only). Pending.
- Then revision 3 (data sample figure). Pending.
- User: run the three checks (1 WS+SPA, 2 highest-cell pooling, 2b tanh saturation) before any experiment; [plan](../../active/plan-diagnostic-checks.md); decision rules proposed, to confirm. Order (user): finish the report work (revisions 2 and 3) first, then the checks.
- Ideas board items: waiting for the user's choice ([ideas board](../../active/ideas-board.md)).
