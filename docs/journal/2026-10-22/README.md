# 2026-10-22

## Architecture arm G (GBRAS-Net): code, PC pilot, G4 speed stop

### Did

- User (2026-10-21): resume arm G with adapter option 1 (fixed SRM bank -> learned 1x1 conv 90 -> 30 -> GBRAS-Net body as published). [Architecture plan](../../active/plan-architecture-tests.md).
- Worker_2: `f86dec1` opt-in `model='gbras'` (`cvdl/gbras.py`; body from the authors' code `0b40ab0`, layer-by-layer audit; 167,464 learned weights incl. the 2,730-weight adapter); native 16 px class-score maps, averaged to 32 px for diagnostics; notebook `notebooks/experiments/srm-gbras.ipynb` (SRM-30 recipe, train-matched `60da`, best checkpoint). Eval-mode BatchNorm test: a tile scores the same alone or in a mixed batch. FULL 395 pass.
- Worker_2: `741d2ef` ROCm fix: BatchNorm with `weight=None` fails backward on ROCm ("tensor does not have a device"); gamma is now a fixed non-saved ones buffer (same math, same learned count).
- Worker_2: PC pilot. Batch 128 and 64 ran out of GPU memory on the PC; batch 32 trained 3 pilot epochs: loss .624 / .543 / .501, pilot validation P_E 31.1 / 25.8 / 27.9%, BN statistics finite, 7.1 GiB (about 29 GiB projected at 128). The matched SRM-30 timing at batch 32 ran out of memory in the PC conv1 workaround, so no PC speed ratio.
- Run_Supervisor: run 14 `srm-gbras-20261022T093648Z-9731` launched on the G4 (user: automatic launch; speed checked on the G4 with a stop rule).

### Found

- G4 at batch 128: **2.48 steps/s** after compile (SRM-30: 18.4), about 19 min per epoch, 30 epochs about **9.5 h**; GPU memory 37-42 GB of 98. Epoch 0 (with compile) 20.7 min.
- Stopped by the Orchestrator before epoch 1 ended (stop requested 10:15:40Z, TERMINATED 10:22:23Z; about 45 VM min, about $1.2). No delivery.
- Cause: GBRAS keeps full 256 px resolution through many depthwise layers; few weights but about 7x the compute of SRM-30.

### Decided

- Run 14 stopped; 30 epochs do not fit the 165/180-min limits. Next step: user decision (full 30 epochs on a long boot about $16; shorter run (not a fair comparison); speed-ups first; or stop arm G). Pending.
