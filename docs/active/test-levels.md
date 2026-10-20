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
| Full tests | `LSB-Dataset-Generator/**`; `tests/test_lsb_torch.py`; `tests/test_lsb_source.py`; `configs/raise512-run1.yaml`; `configs/test-bpp1.yaml`; `pyproject.toml`; `uv.lock`; `Containerfile*`; `Dockerfile*`; any path outside the known areas |
| Golden compare | `cvdl/**`; `LSB-Dataset-Generator/**`; `tests/golden/**`; the golden script and the files it reads or runs (list in `GOLDEN_FILES` in the helper); dependency and container files; any path outside the known areas |
| Fast tests | Every code change that needs neither of the above |
| No tests (`TESTS: NONE`, added `c7b5176`) | Only Markdown docs changed (`docs/**/*.md`, root `README.md`, `AGENTS.md`, `CHANGELOG.md`), or nothing changed. Exceptions (code reads them, so FAST): `docs/active/plan-architecture-tests.md` (`scripts/architecture_decision.py` hashes it) and `docs/figures/report/data-sample.png` (figure and report tests) |

**Always, regardless of the helper:** full tests and the golden compare once before every G4 launch, on the launch commit.

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
