# Notebook refactor: before-values and checks

The [golden JSON](../../tests/golden/refactor-golden.json) is the phase-0 record;
do not overwrite it during the refactor. The [checker](../../scripts/refactor_golden.py)
records effective settings/RUN_NAMEs, train/val/test identities, decoded batches,
four real checkpoints on 64 reproducible tiles, and a deterministic CPU run.
Library versions, checkpoint/input hashes and tile coordinates are recorded.
Notebook 04 is frozen; phase 1 leaves notebook 05 unchanged as a reference.

## Run

```sh
# Default: settings, data, four-checkpoint CPU outputs, CPU 20-step run (~3 minutes).
podman exec -w /work aai3001 python scripts/refactor_golden.py compare
# Same-process GPU package-vs-notebook proof (check GPU availability first).
podman exec -w /work -e CVDL_PARITY_DEVICE=cuda aai3001 python -m unittest discover -s tests -p test_model_copy_parity.py
# GPU QUICK smoke only: run once at the end of the refactor, not every phase.
podman exec -w /work aai3001 python scripts/refactor_golden.py compare --only gpu
```

Check `pgrep -af ipykernel_launcher` and `podman exec aai3001 amd-smi process`
before GPU work. Photos and the four local checkpoints are required. The test
suite skips real-checkpoint tests when checkpoints are absent. Capture/compare
uses `runs/.scratch/` and removes its temporary run outputs.

| Check | Rule |
| --- | --- |
| Settings: G4 newdata-1/wide-1, speed-1 tool settings, CPU/GPU QUICK | Exact. Actual setup/parser outputs, effective settings and run names |
| Standard test: 10,956 pairs | Exact settings SHA `fdd989f7bdab0ec6f9fd6f4c57bdde981d16b54aa8c37e5da8fd3dfcc8b06837` |
| Capped validation: 8,035 pairs (5,000 RAISE / 1,535 ALASKA / 1,500 LSDIR) | Exact pair SHA `6bb3c8efadad06af50ba014a95c98b4fc57008119bdd55b794ec2a1729e4d5ca` |
| Data: decode, first 6 train batches at epochs 0/1; first 2 val/test batches; full frames; workers 0 and 4 | Exact within each worker count (train interleaves differ across counts). Uses newdata-1, batch 8, 1% train photos, priming off |
| Four checkpoints: wide r2, wide mixr2, g4-v4-r2, g4-prime-1 | CPU logits/maps hashes exact across runs. GPU package vs frozen notebook models exact within one process; no historical GPU byte-hash check |
| CPU: 20 train steps + one end-of-epoch validation; fixed seed, 8 threads, workers 0, no norm | Exact losses, final weights, metrics and checkpoint keys. Ordinary sanity validation unchanged; downstream test/panel cells not run |
| GPU QUICK: 6/4 workers, wide/mixed model, two epochs | Smoke only: finishes with same files, counts, identities and key lists. Loss/score/metric deltas printed but never fail |

The 64 model tiles are 8 clean + 8 stego from each of RAISE/DIV2K/ALASKA/LSDIR.
RNG seed 7 selects crops and random LSB replacement (k=1…4, 50% channel values).
These are parity fixtures, not production accuracy results. The GPU parity test
runs these tiles against both models with identical checkpoint weights, using
notebook 04 for prime-1 and notebook 05 for the other three.

## Why GPU cross-run checks are smoke checks

GPU kernels are not bit-exact across processes, including on unchanged code.
Five additional QUICK runs below showed discrete mode flips; both the phase-0
control and phase-1 code produced the same AUC flip (.625 → .62890625). A
one-off module-file/backend audit at standard prediction save found identical
loaded code and flags, and no `cvdl` import before that point. Later the wide
fixed-checkpoint GPU hashes also flipped (about 6e-5 on a 929 logit). Therefore
CPU historical hashes remain the cross-run proof, and exact same-process GPU
notebook/package comparisons prove the GPU implementation. The audit hook and
control worktree are not retained; there are no extra GPU array fixtures.

| Run vs phase-0 golden | Max score delta | Max loss delta | Epoch 1 validation AUC |
| --- | --- | --- | --- |
| Phase-0 control 1 | 0 | 0 | .625 |
| Phase-0 control 2 | 0 | 0 | .625 |
| Phase-0 control 3 | 2.7061e-5 | 1.1086e-5 | .62890625 |
| Phase-1 repeat 1 | 1.8865e-5 | 1.0014e-5 | .625 |
| Phase-1 repeat 2 | 2.9173e-4 | 1.8716e-5 | .62890625 |

Initial phase-0 self-reproduction is recorded in
[self-reproduction.json](../../tests/golden/self-reproduction.json). Its original
GPU tolerance fields are historical, not the current comparison rules above.
CPU final weights SHA:
`5525eb379fe37a74b9b1d28bc3caee453cf7835edfb548385ab1b04d28812edb`.

## Known defect and later change

**QUICK does not cap extra panels:** the bpp-1 panel evaluates all 10,956 pairs.
The GPU smoke [settings](../../tests/golden/quick-gpu.env) retain that behaviour;
the CPU [settings](../../tests/golden/cpu-det.env) disable extras. Fix QUICK's
extra-panel cap deliberately in the settings phase and review its expected
pair-count/settings/output differences; never silently replace the before JSON.

The package/API design is in the [cvdl plan](plan-cvdl-package.md). Keep old
checkpoint parameter and hyperparameter names. Update capture adapters explicitly
when later phases remove notebook globals; do not weaken exact CPU/data checks.
