# 2026-10-21

## Legacy-512 evaluation retired; remote access from the laptop

### Did

- User resumed the work and approved the legacy-512 retirement plan with the Orchestrator's recommendations (archived code stays runnable by hand; `eval_checkpoint.py`, the check-B texture script and the check-B threshold CLI archived too; tanh kernels archived; old runs rescored on 512 only on a copy of the run folder; `--only main` and `--hard` kept).
- Worker: read-only plan (29 grouped items: 12 archive, 15 keep, 2 unsure, answered by the user), then implementation `3c3dc39` (33 files). Record: [legacy-512 retirement](../../active/legacy-512-retirement.md).
- Orchestrator: review (370 tests discovered; no active import of archive code; `--test-suite legacy-512` refused by argparse), push, docs.
- Laptop access (no repository change): the laptop (`fedora-laptop`, RTX 4060 8 GB) can SSH into this PC (`ssh fedora-desk`, key-only). Work from the laptop uses this PC through SSH and herdr; nothing is copied.

### Found

- Golden compare ALL EQUAL (141.1 s). Fast level 341 run + 29 skipped, 142.4 s (was 147.5 s). Full level 370 run, 473.8 s (was 490.4 s).
- Notebook 05, `newdata-1` and `wide-1` still run the 512 test at the end of training by default (`post_fit_eval=full`); all 12 experiment notebooks set `none`. Not changed.
- Two shared parts had to stay active: `cvdl.panels.code_commit` and the hard-test score, Wilson interval and design constants (the 256 path uses them).
- RAISE, DIV2K, the PC evaluations and `runs/verification/` exist only on this PC (the Pi has ALASKA, LSDIR and the G4 outputs).

### Decided

- User: retire legacy-512 as planned, with the Orchestrator's recommendations.
- User: the golden record is not absolute. New [golden policy](../../active/refactor-golden.md#policy-2026-10-21-user): ALL EQUAL for code moves; an intended behaviour change uses a controlled re-baseline (changed entries listed first, user approval, old/new hashes recorded).
- User: switch off the 512 end-of-training test of notebook 05, `newdata-1` and `wide-1` for future G4 runs. Worker plan first (options: launch guard, override, notebook/default change with re-baseline, docs only).
- User: work away from home by SSH from the laptop into this PC, not by copying the environment.

### Next

- Pending: the user decides when to resume arm G (G1).
- Pending (optional): back up RAISE, DIV2K and the PC evaluations to the Pi; `loginctl enable-linger` so a logout does not stop herdr and the container.
- Pending: switching off the 512 end-of-training evaluation in notebook 05 / `newdata-1` / `wide-1` is a separate decision.
