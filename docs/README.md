# Documentation

- [Project journal](journal.md)
- [Research checklist](research-checklist.md)
- [Session handoff 2026-10-15](handoff-2026-10-15.md)
- [V1 baseline results](results-v1-baseline.md)
- [V3b baseline results](results-v3b.md)
- [V3 comparison results](results-v3.md)
- [Noisy-cover dataset plan](plan-noisy-covers.md)
- [ALASKA#2 and LSDIR source integration plan](plan-new-sources.md)
- [ALASKA/LSDIR source code, manifest contract, and verification](plan-new-data-code.md)
- [ALASKA selection v2 (splits, leakage rule, counts)](alaska-selection-v2.md)
- [LSDIR selection v1 (splits, leakage rule, counts)](lsdir-selection-v1.md)
- [On-the-fly generator design plan](plan-otf-generator.md)
- [Optional G4 GPU LSB generator plan](plan-gpu-generator.md)
- [G4 epoch overhead analysis and options](plan-epoch-overhead.md)
- [Wide XuNet model-size comparison plan](plan-wide-xunet.md)
- [Wide XuNet G4 result against R2](result-wide-xunet.md)
- [Frozen RAISE and DIV2K source records](frozen-source-records.md)
- [Full-photo source API and verification](source-verification.md)
- [Shared settings draws and seed keys](settings-and-seeds.md)
- [Shared embed core and mask-free verification](embed-core-verification.md)
- [Mode 1 sample API and legacy replay](mode1-verification.md)
- [Mode 2 full-photo embedding and actual-change labels](mode2-verification.md)
- [Full-photo session cache and read checks](cache-verification.md)
- [Mode 1 PyTorch loader and contract checks](torch-loader-verification.md)
- [Notebook OTF integration and QUICK checks](notebook-otf-verification.md)
- [Notebook diagnostic speed and parity checks](notebook-diagnostics-verification.md)
- [Notebook optional compile switch and verification](notebook-compile-verification.md)
- [G4 batch runner and first-boot steps](g4-batch-runner.md)
- [Rebuild the RAISE 512 dataset](rebuild-raise512-run1.md)

## Local data folders

| Folder | Guide |
| --- | --- |
| [`ImageSets/`](../ImageSets/README.md) | Source images and intermediate image records |
| [`Datasets/`](../Datasets/README.md) | Generated training and test datasets |
| [`checkpoints/`](../checkpoints/README.md) | Local checkpoints and run outputs |
| [`logs/`](../logs/README.md) | Detached-run console logs |
