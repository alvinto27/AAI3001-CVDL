# Notebook 05 refactor: golden before-values

Phase 0 only. No training or model code has been moved. Notebook 04 is the frozen
v3 baseline; its only change is a short banner in the first markdown cell.

## Files and commands

- [Capture/check script](../../scripts/refactor_golden.py)
- [Golden JSON](../../tests/golden/refactor-golden.json)
- [CPU settings](../../tests/golden/cpu-det.env)
- [GPU QUICK settings](../../tests/golden/quick-gpu.env)

Run Python in the ROCm container. The original photos and the four local
checkpoints are required. Check that no notebook or other GPU job is active
before the GPU checks (`pgrep -af ipykernel_launcher`; `podman exec aai3001 amd-smi process`).

```sh
podman exec -w /work aai3001 python scripts/refactor_golden.py capture
podman exec -w /work aai3001 python scripts/refactor_golden.py compare
# A shorter check during a phase:
podman exec -w /work aai3001 python scripts/refactor_golden.py compare --only settings,data,models,cpu
```

`capture` writes the before-values. Do not overwrite the tracked JSON during the
refactor. `compare` only reads that file; it reports differences and returns a
nonzero exit code if a check fails. Temporary outputs are under `runs/.scratch/`
and are removed after each check. Use `capture --out runs/.scratch/second.json`
for an independent second capture. `--only` selects parts.

The JSON is specific to this PC/container: library versions, device and resolved
`/work` paths are recorded. A version or machine change needs review, not a new
baseline made silently.

## Coverage

| Part | Recorded | Compare rule |
| --- | --- | --- |
| settings | Actual notebook setup cell, uppercase settings, RUN_NAME, RunSettings, train/val/test frame hashes, epoch 0/1 schedule hashes; G4 newdata-1/wide-1 and CPU/GPU QUICK env files | Exact |
| speed-1 settings | `run_notebook.settings_file` and `speed_matrix.settings()`; this job bypasses the notebook | Exact; no speed test is started |
| standard test | 10,956 pairs; the same settings-field CSV/hash as `eval_checkpoint.py` | `fdd989f7bdab0ec6f9fd6f4c57bdde981d16b54aa8c37e5da8fd3dfcc8b06837` |
| capped validation | Full newdata-1 validation pair CSV, 8,035 pairs (5,000 RAISE / 1,535 ALASKA / 1,500 LSDIR) | `6bb3c8efadad06af50ba014a95c98b4fc57008119bdd55b794ec2a1729e4d5ca` |
| data | Newdata-1 with 1% train photos, batch 8, decode, priming off, nonpersistent eval; 6 train batches in epochs 0 and 1, first 2 val/test batches, full frame hashes, workers 0 and 4 | Exact **within each worker count**; training interleave differs between worker counts, so their hashes are not expected to equal each other |
| models | Four best checkpoints: wide-w4 r2, wide-w4 mixr2, g4-v4-r2, g4-prime-1; strict state load; 64 real tiles (8 clean + 8 stego per source); logits and 16×16 margin maps; CPU and ROCm GPU | CPU and GPU hashes each repeat exactly and are mandatory. CPU vs GPU rounding deltas are recorded, not expected to be zero. Tile list, seed and input-pixel hash recorded |
| CPU | Notebook 05: fixed seed, plain XuNet/no norm, eager, decode, batch 8, 8 fixed threads, workers 0, 20 train steps and one end-of-epoch validation pass (QUICK: 16 pairs); sanity validation also runs as usual | Per-step losses, final weight hash, metrics.csv values and checkpoint key lists exact |
| GPU | Notebook 05 via unmodified `scripts/run_notebook.py`: mixed sources, wide model, eager, 2 epochs, train/eval workers 6/4, QUICK 16-pair validation/test, full extra bpp-1 panel | Counts, file list, checkpoint keys, epoch AUC/P_E/accuracy and identities exact. Losses/scores: absolute tolerance 1e-4. Panel P_E/AUC: report-only, warn above 0.01. Timings and TensorBoard event names excluded |

For the model input tiles, the script uses one photo from each source, crops
512×512 windows with RNG seed 7, and makes a stego counterpart by random LSB
replacement (k=1…4, 50% of channel values). These are parity fixtures, not a
production benchmark. The JSON stores the photo path, crop coordinates, kind,
k, input-pixel hash and checkpoint file SHA. In phase 0, both the trained
notebook class (04 for g4-prime-1, 05 otherwise) and `map_demo` are evaluated
with the same state dict; any output difference stops capture.

The CPU driver executes notebook cells in their original order and wraps
`Trainer.fit` to record losses and final weights, then stops after fit. It does
not run the later test, extra-panel or diagnostic cells. No notebook-only
settings or alternate model implementation are introduced. Hyperparameter
names/values, state-dict keys/shapes/dtypes, callback-state keys and optimizer
state key lists come from this fresh run's `last.ckpt`.

## Known defect: QUICK extra panels are not capped

**QUICK does not cap extra panels.** The bpp-1 panel still evaluates all
10,956 pairs; source panels would also run at full size. The GPU reference
keeps this behaviour and runs only `V3_EXTRA_TESTS=test-bpp1`. The CPU reference
sets extras to `none` (the full CPU panel previously took more than 23 minutes
and was stopped after training and standard test had finished).

Fixing QUICK to cap extra panels is an **intentional settings-phase change**.
Its smaller pair count, settings hash, metrics and output list must be reviewed
as expected differences then. Do not silently update the golden to conceal it.

## Refactor use

Keep the before JSON unchanged. After a code move, adapt the capture entry point
only when required (for example, `cvdl.models.model_from_hparams` replaces the
current model copy automatically). Never weaken the comparison to obtain a
pass. Module source names can change; model class names, parameter keys, input
bytes and outputs cannot. The current notebook setup/loader globals form the
capture interface; replace the adapter explicitly if a phase removes them.

## Self-reproduction result

Two independent captures passed `compare --candidate runs/.scratch/second.json`.
The [self-reproduction summary](../../tests/golden/self-reproduction.json) records the observed deltas.

| Check | Before versus before |
| --- | --- |
| Settings, data, four-checkpoint outputs (CPU and GPU) | Exact |
| CPU: 20 per-step losses, metrics and weights | Exact; final weights SHA `5525eb379fe37a74b9b1d28bc3caee453cf7835edfb548385ab1b04d28812edb` |
| GPU QUICK loss maximum absolute difference | 4.6492e-6 (enforced limit: 1e-4) |
| GPU standard-test probability maximum difference | 3.0190e-5 (enforced limit: 1e-4); raw score CSV hashes differ as expected |
| GPU epoch AUC/P_E/accuracy, checkpoint keys, counts and file list | Exact |
| Full bpp-1 P_E: overall / worst bin difference | 0.00018255 / 0.00335570 (report-only) |
| Full bpp-1 AUC: overall / worst bin difference | 0.00009053 / 0.00043403 (report-only) |
| Runtime per complete capture | About 4.8 minutes: settings 23 s, data 56 s, models 58 s, CPU 26 s, GPU 122 s |

The QUICK model is near chance, so the full panel's small-bin rank/tie metrics
are noisy. They are printed but not enforced; a delta above 0.01 prints a
warning. Counts, panel identities and schemas still fail on a change. The exact
CPU and four-checkpoint checks carry the refactor-equivalence proof; the GPU
QUICK run checks the live training/output path. Five comparator tests ensure
that exact fields cannot accidentally enter the tolerance/report-only paths.
The full repository suite passes **127 tests** (122 existing + 5 comparator tests).
