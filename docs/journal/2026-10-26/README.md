# 2026-10-26

## #20 arm A: learnable SRM

### Did

- Worker_4 plan (`runs/.scratch/learn-srm-plan-2026-10-26/PLAN.md`, sha `e3f962f6…9452`): `xunet_wide_learn_srm` = R1's model with the 90 SRM kernels (2,250 weights) trainable from their exact SRM values; clamp ±3 and the FP32 guard kept; same LR, decay and clip. Notebook diff vs R1: name + model + `val_batch_images=64`.
- Found: the start-up model check asserts a frozen SRM and must become model-aware; a state dict does not record `requires_grad`, so the model tag and resume checks must carry it.
- Cost estimate (no measurement): training arithmetic 6.5 G -> 9.0 G MAC per image (+39%; conv1 now needs its input gradient); about 13 steps/s on the G4, about 200 min (budget 200-250 min), about $5.5-7.

### Decided

- **User approved** (2026-10-26) all five points:
  1. the model as planned;
  2. the frozen speed gate after epoch 3: train speed (epochs 2+3) >= 10.5 steps/s and projected <= 250 min, else stop with no verdict; limits `--timeout-min 270 --max-run-minutes 285`;
  3. rule `learn_srm` = both `paired_r1` routes vs R1 (eval `6ef349895655136bef90`); the kernel movement report and the #11 re-score are context only;
  4. a PC correctness and timing probe first;
  5. the first real use of the Pi watchdog (disarm it first if the run is stopped on purpose).

### Next

- Worker_4: implementation, PC probes, FULL. Pending.
