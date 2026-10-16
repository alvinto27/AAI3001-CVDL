# Logs

This folder holds temporary command logs and the `alaska-*` source-record logs. Only this README is tracked. Keep finished PC and G4 run results under `runs-out/<run>/`, not here or in `checkpoints/<run>/`. For notebook 05 headless runs, `runs-out/<run>/run.log` is the one full copy of notebook output; `executed.ipynb` has figures and short pointers to that log. The job watcher gets short step/rate reports instead of a second full log.

## Create and use

For a detached training run, write output to a named log:

```sh
podman exec -d -w /work aai3001 bash -c \
  "python train.py --run-name my-run --epochs 10 > logs/my-run.log 2>&1"
```

Keep the log until the run has been checked and its results are recorded in
the journal or a results report. The v3b and v3c logs were deleted after
their numbers were recorded in the reports. The V3a VM log is kept under `runs-out/early-pc/baseline_kv_xunet_v3_clip3_batch/logs/legacy_from_vm/`. The `alaska-*` logs here are evidence for the source records. The `task31_*` and `task33_*` scratch logs were removed.

## Do not commit

Never commit console logs, TensorBoard server logs, temporary logs, or other
run output. Delete them after the results are recorded. Keep only this README
in git.

See the [project journal](../docs/journal/README.md), [V3b report](../docs/results/results-v3b.md),
and [V3 report](../docs/results/results-v3.md) for recorded results.
