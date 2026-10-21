# Test levels and the changed-path rule

Status: built `891aefe` (2026-10-20), user approved. This record replaces the old standing rule "full CPU suite and golden compare before every push".

## Why

On 2026-10-20 the full CPU suite took 8.2 min (424 tests) and the golden compare took 2.5 min. Both ran before every push, even for changes that could not affect them. Profile (Worker, 2026-10-20; raw logs in the git-ignored `runs/.scratch/test-profile-2026-10-20/`):

| Part | Time | Share of the full suite |
| --- | --- | --- |
| `tests/test_lsb_torch.py` (28 tests, loader worker processes) | 330 s | 67% |
| └ one test: packed-loader tail order, 36 cases | 206 s | 42% |
| `test_cvdl_training` (20-step CPU train + resume; near-duplicate of the golden `cpu` stage) | 34 s | 7% |
| Rebuilds of frozen lists (`hardtest256`, `train_matched_validation`, `standard_test`) | about 48 s | about 10% |
| Tests tied to one notebook (frozen 04, archived 05, finished experiments; 17 files) | about 86 s run alone | about 16% |

## Commands

```sh
podman exec -w /work aai3001 python -B scripts/required_test_level.py           # which checks this change needs
podman exec -w /work aai3001 python -B scripts/test_cpu.py --level fast         # about 2.7 min
podman exec -w /work aai3001 python -B scripts/test_cpu.py                      # full (default), about 8.3 min
podman exec -w /work aai3001 python -B scripts/refactor_golden.py compare       # golden, GPU visible, about 2.5 min
```

- `test_cpu.py` sets the CPU-only environment (no GPU, 1 thread) itself. The fast level skips the whole `test_lsb_torch` module with the visible reason `full level only: ...`. The full level runs every test.
- A plain `python -B -m unittest discover -s tests` is unchanged and runs every test.
- `required_test_level.py` is read-only (no fetch). It compares `merge-base(origin/yx, HEAD)` to HEAD, plus staged, unstaged and untracked files, and prints `TESTS: FAST|FULL` and `GOLDEN: REQUIRED|NOT REQUIRED` with the matched paths. If it cannot read the changes, it prints FULL and REQUIRED.

## Rule

| Check | Required when a changed path matches |
| --- | --- |
| Full tests | `LSB-Dataset-Generator/**`; `tests/test_lsb_torch.py`; `tests/test_lsb_source.py`; `configs/raise512-run1.yaml`; `configs/test-bpp1.yaml`; the data-figure replay inputs (added `4959099`: `tests/test_data_figure.py`, `scripts/build_data_figure.py`, `docs/figures/report/data-sample.png`, `cvdl/standardtest.py`, `cvdl/validation.py`, `cvdl/__init__.py`, the four `data/<source>/photo_manifest.csv`, `data/test/standard-256-v1/`, `data/hard-test/256-v1/`); `pyproject.toml`; `uv.lock`; `Containerfile*`; `Dockerfile*`; any path outside the known areas |
| Golden compare | `cvdl/**`; `LSB-Dataset-Generator/**`; `tests/golden/**`; the golden script and the files it reads or runs (list in `GOLDEN_FILES` in the helper); dependency and container files; any path outside the known areas |
| Fast tests | Every code change that needs neither of the above |
| No tests (`TESTS: NONE`, added `c7b5176`) | Only Markdown docs changed (`docs/**/*.md`, root `README.md`, `AGENTS.md`, `CHANGELOG.md`), or nothing changed. Exceptions (code reads them, so FAST): `docs/active/plan-architecture-tests.md` (`scripts/architecture_decision.py` hashes it) and `docs/figures/report/data-sample.png` (figure and report tests) |

**Always, regardless of the helper:** full tests and the golden compare once before every G4 launch, on the launch commit.

When the golden may change (intended behaviour change only): see the [golden policy](refactor-golden.md#policy-2026-10-21-user).

Evidence for the golden list: an audited golden compare on 2026-10-20 (ALL EQUAL, 143 s) recorded every file it opened. 37 tracked files; all are matched by the helper's rule. It did not open `data/train-grids/`, `split.csv`, `selection*.csv`, `provenance.csv`, `legacy_tile_manifest.csv` or `run1_*.csv.gz`. The helper also keeps GPU-only and fallback paths found by reading the code.

Replay on past commits: `9dfaa27` (decision script) gives FAST + golden NOT REQUIRED; `1bb7e20` (arm A, `cvdl/`) gives FAST + golden REQUIRED; `863d9fb` (packed loader) gives FULL + golden REQUIRED.

## Measured (2026-10-20, `891aefe`)

| Level | Run | Skipped | Wall |
| --- | --- | --- | --- |
| Fast | 413 | 28 | 160.6 s |
| Full | 441 | 0 | 495.4 s |

## Follow-up changes (2026-10-20, user approved)

| Item | State |
| --- | --- |
| **C:** `test_cvdl_training` without the repeated 20-step golden comparison | Done `a1ae22c`. It now trains 2 steps (with validation and a real best checkpoint), then resumes 2 steps and checks step 4 and the kept metrics rows. The exact 20-step check stays in the golden `cpu` stage. Gain only about 5 s (33.5 s to 28.7 s): most of the time is real-photo setup and validation, not the 20 steps. Loader shutdown moved into `finally` |
| **B (shrink the 36-case packed-loader test)** | **Not done (B3, user).** A already skips it in the fast level. The workers start and stop on purpose: training workers shut down after fit (freed about 10 GB of PC RAM, 2026-10-14), evaluation workers are released after each pass, and they start with `spawn` (each worker loads PyTorch again). Each case needs its own worker count, so workers cannot be shared across cases, and `fork` would test a path that real runs do not use |
| Helper `TESTS: NONE` for Markdown-only changes | Done `c7b5176` (21 helper tests) |

Fast level after these changes: 417 run, 28 skipped, 155.9 s wall.

## Test trim (2026-10-20, `4959099`, user approved)

A read-only audit of all 55 test modules (Worker; git-ignored evidence in `runs/.scratch/test-necessity-audit-2026-10-20/`) found that most tests protect current or planned code, often error paths that the golden compare does not run. Changes:

| Change | Tests |
| --- | --- |
| Duplicates deleted: 4 inherited re-runs in `test_tanh_saturation`; `test_cvdl_training.test_capped_validation_runtime_dependency` (covered by `test_notebook_05_capped_sanity`) | -5 |
| Archived with their code: checks 1, 2, 2b (`cvdl/archive/{classic_lsb,cell_pooling,tile_saturation}.py`, `scripts/archive/check1_classic.py`, `check2_pooling.py`, `check2b_saturation.py`) and `scripts/archive/curriculum_decision.py`; their tests in `tests/archive/` | -33 |
| Moved to full: `test_data_figure.DataFigureReplayTests` (frozen 7-pair figure rebuild, 7.5 s) | 0 |

Rules: `tests/archive/` has no `__init__.py`, so neither level discovers it. Active code must not import from `cvdl/archive/`. Before an idea reuses archived code (for example #12 with WS/SPA, or #3 with cell pooling), restore the code **with its tests** to the active folders.

Measured after the trim: full 407 run, 0 skipped, 490.4 s; fast 378 run, 29 skipped, 147.5 s; golden ALL EQUAL.

Not done (user): 9 small frozen-notebook and old-metric tests stay in fast (moving them saves about 0.1 s). ~~Next (paused): retire the legacy-512 evaluation (`--test-suite legacy-512`), which keeps about 30 more tests alive; design first.~~ Done 2026-10-21 (`3c3dc39`, [legacy-512 retirement](legacy-512-retirement.md)): 407 to 370 tests; fast 341 run + 29 skipped, 142.4 s; full 473.8 s. Test gap: ideas board #14.

