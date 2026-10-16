# Logs

This folder stores console output from detached or long-running commands. The
folder is local only. Only this README is tracked.

## Create and use

For a detached training run, write output to a named log:

```sh
podman exec -d -w /work aai3001 bash -c \
  "python train.py --run-name my-run --epochs 10 > logs/my-run.log 2>&1"
```

Keep the log until the run has been checked and its results are recorded in
the journal or a results report. The v3b and v3c logs were deleted after
their numbers were recorded in the reports. The v3a VM log is kept in
`logs/v3a_legacy_from_vm/`. Other files here are local task logs.

## Do not commit

Never commit console logs, TensorBoard server logs, temporary logs, or other
run output. Delete them after the results are recorded. Keep only this README
in git.

See the [project journal](../docs/journal/README.md), [V3b report](../docs/results/results-v3b.md),
and [V3 report](../docs/results/results-v3.md) for recorded results.
