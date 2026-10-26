# 2026-10-22

## Architecture arm G (GBRAS-Net): code, PC pilot, G4 speed stop

### Did

- Worker_2: `f86dec1` opt-in `model='gbras'` (`cvdl/gbras.py`; body from the authors' code `0b40ab0`, layer-by-layer audit; 167,464 learned weights incl. the 2,730-weight adapter); native 16 px class-score maps, averaged to 32 px for diagnostics; notebook `notebooks/experiments/srm-gbras.ipynb` (SRM-30 recipe, train-matched `60da`, best checkpoint). Eval-mode BatchNorm test: a tile scores the same alone or in a mixed batch. FULL 395 pass.
- Worker_2: `741d2ef` ROCm fix: BatchNorm with `weight=None` fails backward on ROCm ("tensor does not have a device"); gamma is now a fixed non-saved ones buffer (same math, same learned count).
- Worker_2: PC pilot. Batch 128 and 64 ran out of GPU memory on the PC; batch 32 trained 3 pilot epochs: loss .624 / .543 / .501, pilot validation P_E 31.1 / 25.8 / 27.9%, BN statistics finite, 7.1 GiB (about 29 GiB projected at 128). The matched SRM-30 timing at batch 32 ran out of memory in the PC conv1 workaround, so no PC speed ratio.
- Run_Supervisor: run 14 `srm-gbras-20261022T093648Z-9731` launched on the G4 (user: automatic launch; speed checked on the G4 with a stop rule).

### Found

- G4 at batch 128: **2.48 steps/s** after compile (SRM-30: 18.4), about 19 min per epoch, 30 epochs about **9.5 h**; GPU memory 37-42 GB of 98. Epoch 0 (with compile) 20.7 min.
- Stopped by the Orchestrator before epoch 1 ended (stop requested 10:15:40Z, TERMINATED 10:22:23Z; about 45 VM min, about $1.2). No delivery.
- Cause: GBRAS keeps full 256 px resolution through many depthwise layers; few weights but about 7x the compute of SRM-30.

### Decided

- User: resume arm G (GBRAS-Net, [architecture plan](../../active/plan-architecture-tests.md)) with adapter option 1: a learned 1x1 layer 90 -> 30 channels after the fixed SRM bank, then the GBRAS body as published. Idea #15 (routing by processing history) dropped for now (recorded on the `pipeline` board). Pipeline work stays on branch `pipeline`; `yx` continues the model work. (Moved here from the 2026-10-21 entry, where it was filed by mistake.)
- Run 14 stopped; 30 epochs do not fit the 165/180-min limits. Next step: user decision (full 30 epochs on a long boot about $16; shorter run (not a fair comparison); speed-ups first; or stop arm G). ~~Pending.~~
- **User: option C** (speed-ups first: BF16 mixed precision, channels-last, compile options; measured on the G4), then, if still above 180 min, one long unsupervised run. Worker_2: read-only speed plan next.
- User: speed plan approved ([plan](../../archive/plan-gbras-speed.md)): caps 3 preemption relaunches / 15 stockout attempts / $20; BF16 accepted as a recorded difference; **watchdog on the Pi** (the PC is not on overnight).
- User raised paired training (ideas #16): partners in the same batch with the same flip/rotation. Found: the current paired mode does not tie the augmentation (the seed includes the label). Order chosen by the user: GBRAS timing boot -> GBRAS long overnight run -> paired work next working day (code fix, SRM-e50 paired, cell model paired). Pending: run-13 style or 50 epochs from scratch for the cell model.
- User agreed: the GBRAS long run starts unattended only when resume and the Pi watchdog are ready and tested; otherwise it moves to the next night and the paired work (#16) fills the day.
- User: tell the user before every G4 launch. The GCE max run duration (now 300 min, termination action STOP, spot) can be changed by the Orchestrator with `gcloud compute instances set-scheduling rtx6000pro --zone=us-central1-c --max-run-duration=<N>m` only while the VM is TERMINATED; for a long run, propose the value, set it after the user's OK, and set it back to 300 min afterwards.

## GBRAS speed-ups measured on the G4

### Did

- Worker_2: step 1 `3c86e52` (SRM + clamp kept FP32 under AMP for every SRM model; opt-in `channels_last`; FP32 path bit-identical for KV, SRM-wide, delayed, run 13, GBRAS; FAST 406 pass). Step 2a `1f3efe0` (timing and numerics benchmark on the shared runtime; FULL 416 pass).
- Run_Supervisor: timing boot `gbras-speed-bench-20261022T120338Z-3374` (GCE limit set to 60 min for the boot, restored to 300 min after). 12:04:50Z-12:22:38Z, about $0.5. The four configurations completed; the last part (validation and checkpoint overhead of the winner) failed with a code error (`cannot prime while the current epoch is running`, `cvdl/gbras_bench.py` line 341), so the tool wrote no projection. No retry.

### Found

| Configuration (batch 128) | Steps/s | Peak GPU memory | Numerics gates |
| --- | ---: | ---: | --- |
| FP32, NCHW (run 14 setting) | 2.48 | 37.1 GB | reference |
| BF16, NCHW | 2.73 | 18.8 GB | pass (probability p99 gap .0029) |
| BF16, channels-last | 2.73 | 18.8 GB | pass (.0022) |
| FP32, channels-last | 1.81 | 37.3 GB | pass |

- BF16 gives only 1.10x; channels-last gives nothing (slower in FP32). Memory halves. The model is limited by memory traffic of many small full-resolution passes, not by the arithmetic precision.
- Projection (Orchestrator arithmetic; run 14 validation about 1 min per epoch): 2,813 steps x 0.367 s = 17.2 min per epoch; 30 epochs about **9-9.5 h, about $15-16** on spot.

### Decided

- ~~Pending (user): long run at about $15-16 (needs resume and the Pi watchdog first; the $20 cap leaves little room for preemptions) or stop arm G. The overhead bug is not worth a fix: the run 14 validation time is enough for the projection.~~
- **User: stop arm G (option B).** Finding for the report: GBRAS-Net costs about 7x the compute of SRM-30 at 256 px colour tiles; BF16 gives 1.10x; too costly for this project. Code stays (opt-in). Resume and the Pi watchdog are not built (design kept in the speed plan). Spend on arm G: run 14 about $1.2, timing boot about $0.5.

### Next

- Next working day: ideas #16 paired training (code fix for tied augmentation, SRM-e50 paired, then the cell model paired). Pending (user): cell model run-13 style (15 epochs from the new e50) or 50 epochs from scratch.
- User: paired training plan approved ([plan](../../active/plan-paired-training.md)) with the Orchestrator's changes: R2 = run 13's 15-epoch fine-tune, paired, from R1 (fallback SRM-e50); R2 start checkpoint set by a notebook commit (no runner change); evaluation tools built while R1 trains; R1 rule has a second route (hard low AUC +.01). Decision rules frozen before any result. Code and PC pilot tonight; R1 and R2 on 2026-10-23.
- Worker_2: paired implementation `d28e2bb` (augmentation tie with key 1 for both partners; paired cell items, counts and loss flattening; low-bpp curriculum and augmented PNG pairing refused; notebooks `srm-e50-paired` and `srm-cell-aux-paired` differ from their originals only by name and `paired_batches=True`). Unpaired tensors and RNG draws unchanged. FULL 425 pass; both G4 preflights pass. PC probes (8 pairs, 32 updates, correctness only): R1 CE .6934 -> .6932; R2 total .422 -> .374 (CE .271 -> .223, cell BCE .755 -> .753). Run queue rows 15 (R1, ready) and 16 (R2, waits for R1).
- User raised ideas #17: paired low-bpp curriculum (#9 + #16). Recorded; decide after the R1/R2 results.
- User: #17 becomes a full 50-epoch curriculum from scratch (SRM-e50 recipe); the paired variant if R1 passes its rule, else unpaired. Worker_2: read-only plan.
- User: the #17 curriculum run stays at a fixed 50 epochs; more epochs / early stopping shelved as ideas #18 until evidence shows they are needed (SRM-e50 curve: flat after the last LR drop; run 11 gave nothing).
- User: [curriculum50 plan](../../active/plan-curriculum50.md) approved: schedule A (100% normal mix epochs 1-7, then 50% normal + 50% staged .05-.35 / .05-.25 / .05-.15 bpp with k=1 weight 1/2/3; LR drops at epochs 8 and 33); rule: hard k=1 low AUC +.01 with the interval above 0, standard P_E guard +0.25 points, selected epoch >= 8. Worker_2 builds both notebooks.
- Worker_2: curriculum50 implementation `77f0ecd` (`low_bpp50_v1`; notebooks differ from `srm-e50.ipynb` only by name + schedule (+ paired)). Run 10/11 data, RNG and records reproduce exactly; epochs 1-7 equal the normal recipe; resume across the stage change restores the optimizer and LR. FULL 434 pass (a first FULL had one PC-safety semaphore warning after the GPU probes; the safety tests and a clean FULL alone passed). PC probes (correctness only) CE: unpaired .690 -> .663, paired .693 -> .679. Both G4 preflights pass. Run queue row 17.

### Next

- 2026-10-23: tell the user, launch R1 (row 15). Build the evaluation tools while R1 trains. R1 verdict -> R2 (row 16) and the curriculum variant (row 17); order to choose.
