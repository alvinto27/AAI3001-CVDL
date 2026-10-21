# Legacy-512 evaluation retirement (2026-10-21)

Commit `3c3dc39` (Worker, user approved 2026-10-21). The PC command `scripts/evaluate_run.py --test-suite legacy-512` and its optional stages are retired. The code is archived, not deleted, and stays runnable by hand. Git-ignored evidence: `runs/.scratch/legacy-512-retirement-plan-2026-10-21/PLAN.md` (plan) and `runs/.scratch/legacy-512-retirement-implementation-2026-10-21/REPORT.md` (implementation).

## What changed

| Item | Now |
| --- | --- |
| `scripts/evaluate_run.py` | Runs only the standard-256-v1 suite and the hard-256 test. `--test-suite legacy-512` and the old options (`--tables`, `--label`, `--skip-main`, `--padding-check`, `--padding-pairs`, `--temperature`, `--tanh-saturation`, `--tanh-pairs`, `--tanh-train-batches`, `--only tables/padding/temperature/tanh/hard`) are refused before any checkpoint, GPU or file work. `--only main`, `--hard`, `--no-hard`, `--higher-k`, `--settings-from`, `--set`, `--output` and `--overwrite` stay. |
| Shared parts kept active | `cvdl.panels.code_commit`; `cvdl.hardtest.score`, `wilson` and the design constants (`COMBINATIONS`, `CONDITIONS`, `KS`, `PAIRS_PER_CELL`, `PER_COMBINATION`). The 256 path uses them. |
| Not changed | Notebook 05, `newdata-1`, `wide-1` and the experiment notebooks (their end-of-training 512 evaluation, controlled by `post_fit_eval`); `cvdl/evaluation.py`; the golden compare; configs; data (including `data/hard-test/v1/`); `runs/`. |
| Tests | 407 to 370 (fast 341 run + 29 skipped). Archived tests are in `tests/archive/` and are not discovered. |

## Old path to archive path

| Old path | Archive path |
| --- | --- |
| `scripts/evaluate_run.py` legacy branch (old `evaluate`, stage dispatch, writers) | `scripts/archive/evaluate_legacy_512.py` |
| `cvdl/hardtest.py` v1 parts (sampling, build, load, run) | `cvdl/archive/hardtest_v1.py` |
| `cvdl/panels.py` old panel tables and validation writers | `cvdl/archive/panels.py` |
| `cvdl/calibration.py`, `cvdl/padding.py`, `cvdl/saturation.py`, `cvdl/post_fit.py` | `cvdl/archive/` (same names) |
| `scripts/eval_checkpoint.py`, `scripts/check_b_tiles.py`, `scripts/check_b_thresholds.py`, `scripts/build_hard_test.py`, `scripts/padding_check.py` | `scripts/archive/` (same names) |

Older docs, journals and result records keep the old paths as written. Use this table to find the code.

## Rules

- Active code must not import from `cvdl/archive/` or `scripts/archive/`.
- **Rescore an old run on 512 only on a copy of its run folder.** The archived main stage writes its predictions into the run folder; `--output` does not redirect them. Never change existing outputs under `runs/`.
- Before an idea reuses archived code, restore it with its tests to the active folders (same rule as the [test trim](test-levels.md)).
- Frozen notebook 04 imports `eval_checkpoint` by name. Its bytes stay unchanged; running it in full is not supported (the golden compare runs only its model definitions).

## Checks (Worker, 2026-10-21)

Helper: FAST, GOLDEN REQUIRED. Golden compare ALL EQUAL (141.1 s). Fast level 341 run + 29 skipped, 142.4 s. Full level 370 run, 473.8 s. Orchestrator review: 370 discovered; no active import of archive code; `--test-suite legacy-512` gives an argparse error.

Not part of this step: switching off the 512 end-of-training evaluation in notebook 05, `newdata-1` and `wide-1` (a separate decision).
