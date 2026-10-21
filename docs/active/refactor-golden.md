# Notebook refactor: before-values and checks

The [golden JSON](../../tests/golden/refactor-golden.json) is the phase-0 record;
do not overwrite it during the refactor. The [checker](../../scripts/refactor_golden.py)
records effective settings/RUN_NAMEs, train/val/test identities, decoded batches,
four real checkpoints on 64 reproducible tiles, and a deterministic CPU run.
Library versions, checkpoint/input hashes and tile coordinates are recorded.
Notebook 04 is frozen. The pre-package notebook 05 is frozen, with outputs
cleared, in [the parity archive](../../notebooks/archive/05_baseline_kv_xunet_v4_pre-cvdl.ipynb).
The current notebook 05 is the thin package caller.

## Policy (2026-10-21, user)

The golden proves that a code move or cleanup changed nothing. It is not absolute. ~~Never recapture or weaken the golden.~~ Replaced by two cases:

| Case | Rule |
| --- | --- |
| Code move, cleanup, refactor (no intended behaviour change) | The unchanged compare must print **ALL EQUAL**. No recapture, no allow-list change, no partial compare. |
| **Intended behaviour change** (for example a changed default) | **Controlled re-baseline**, all steps required: (1) the unchanged compare prints ALL EQUAL on the code before the change; (2) before the change, write down exactly which golden entries will change and why; (3) make the change; (4) the compare differs **only** in those entries; (5) the user approves the list and the result; (6) recapture, in a separate commit, with a record here: date, reason, the changed entries, and the old and new SHA-256 of `tests/golden/refactor-golden.json`. |

Any difference that is not on the list stops the work: find the cause, do not recapture. The earlier "intended difference" sections below (phase 4a, 4c) are the same idea as an allow-list.

Known limit: the golden covers mostly the old 512 path (notebook 05, the old standard-test identity, the `newdata-1` / `wide-1` settings). The current paths (256 evaluation, experiment notebooks, SRM models, deployment threshold) are covered by unit tests, not by the golden.

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
notebook 04 for prime-1 and the frozen pre-package notebook 05 archive for the
other three.

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

## Intended difference: QUICK extra-panel cap (phase 4a)

The phase-0 GPU QUICK record evaluated all **10,956 bpp-1 pairs**. The shared
extra-panel evaluator now selects at most **16 pairs** when QUICK is true,
with the existing fixed, camera/source-stratified subset and seed 1. This also
caps source panels. Extra-panel pair counts, settings hashes and predictions
are intended differences; extra `run.json` records `quick_pair_cap`.

The standard notebook test, its full 10,956-pair identity/hash, validation and
training are unchanged. The default comparator still returns ALL EQUAL: CPU
extras are disabled, and settings/data checks do not evaluate extra panels.
The historical golden JSON is unchanged. At the final GPU smoke, retain the
standard-test file/count/identity/key checks; do not require full-size extra
panel predictions. No GPU run was needed for phase 4a's cap test.

Standalone `eval_checkpoint.py` (archived 2026-10-21 to `scripts/archive/`; the golden does not import it) remains full when QUICK is absent. `--quick`
requests the cap; `--no-quick` disables an inherited QUICK value. Future thin
notebooks must pass their resolved `settings.quick` to `evaluate()`.

## Thin-notebook drivers and allow-list (phase 4c)

The settings/data/CPU drivers still execute the **current** notebook 05's code
cells in order. They do not execute the archive. The settings driver reads the
resolved `Settings.legacy_globals()` view and `to_run_settings()` to compare the
same historical fields. Data still hashes real loaders at epochs 0/1. CPU still
records all 20 losses, final weights, metrics and checkpoint structure. The
archive is used only by independent same-process parity tests and the model
fallback reference. Behaviour tests use the package callbacks/Lightning module.

The historical golden JSON is not changed. The comparator has an explicit,
named allow-list; it prints every use. All other differences still fail.

| Allowed name | Exact scope |
| --- | --- |
| `run-settings-file` | One new `run_settings.json` in the run folder; all other filenames remain checked |
| `thin-cell-layout` | 16 completed old code cells become exactly 10 completed new code cells; failed/missing cells fail |
| `quick-extra-panel-cap` | Only the bpp-1 extra panel uses 16 instead of 10,956 pairs; rebuild its fixed seed-1 subset, settings SHA, categories, values and every group/image/pair count independently |
| `gpu-setting-table-scores` | Only the GPU per-setting CSV score-byte hash may change; the actual saved table is separately rebuilt from the saved standard predictions and checked byte-for-byte |

The standard test's pair identity hash and checkpoint/data/CPU checks are not
relaxed. GPU scalar/probability deltas remain informational as before. The GPU
runner also passes `--set lr=0.0003` (the unchanged LR) and checks that the final
`run_settings.json` records `lr` with source `override` and in override history.

The package/API design is in the [cvdl plan](plan-cvdl-package.md). Keep old
checkpoint parameter and hyperparameter names. Update capture adapters explicitly
when later phases remove notebook globals; do not weaken exact CPU/data checks.
