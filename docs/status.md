# Project status — 2026-10-20

**Start here:** Read the [2026-10-19 handoff](active/handoff-2026-10-19.md) and the [2026-10-19 journal](journal/2026-10-19/README.md). Evaluation redesign is done (standard-256 and hard-256 tests, one frozen deployment threshold from validation, report notebook); all 11 documented models are rescored. Diagnostic checks 1, 2 and 2b are done ([plan](active/plan-diagnostic-checks.md)): #12 raised, #3 lowered, tanh fix struck. Low-bpp curriculum (#9) done: rule not met; small real gain (+0.02 k=1 low AUC) at +0.56 points standard P_E; not adopted ([plan, result](active/plan-low-bpp-curriculum.md)). Architecture arm A (delayed pooling) done: **no gain, clearly worse** (pooled hard k=1 low AUC -.060 vs SRM-30; 2.3x compute; [plan, result](active/plan-architecture-tests.md)). Arm G (GBRAS-Net) waits for the user's 90-to-30 adapter choice. PC pilot mode `pilot=true` added (`aec69d0`). **2026-10-20: arm G (G1) paused by the user; review of pre-run checks first. Test levels built (`891aefe`, [record](active/test-levels.md)): fast level 2.7 min vs full 8.3 min; the helper `scripts/required_test_level.py` decides ~~full tests and golden~~ the test level per change; ~~full + golden~~ full tests always before a G4 launch (golden retired 2026-10-21). Follow-ups done: C (`a1ae22c`), `TESTS: NONE` for Markdown-only changes (`c7b5176`); B not done (user, B3). Test trim `4959099`: 445 to 407 tests (5 duplicates deleted, 33 archived with the code of checks 1, 2, 2b and the curriculum decision); fast level 2.5 min. ~~**Paused by the user.** Next, when the user resumes: retire the legacy-512 evaluation (design first); arm G (G1) stays paused.~~ Golden retired; comparison constants list added (`72f51b5`, 360 tests). 512 end-of-training test off in notebook 05 / `newdata-1` / `wide-1` (`7921d58`). Legacy-512 evaluation retired `3c3dc39` (407 to 370 tests; [record](active/legacy-512-retirement.md), [2026-10-21 journal](journal/2026-10-21/README.md)). ~~Next: the user decides when to resume arm G (G1).~~ **Focus (user, 2026-10-21): finish the basic deliverables. Model tests (arm G) on hold; next work is cell-level training (#3): [approved plan](active/plan-cell-training-256.md) (one 15-epoch run matched to `srm-low-bpp-control`, 2.65%; about $1.4-1.8); code done `731a27c` (368 tests); next: PC pilot, then one G4 run.** ~~Next: the user decides arm G (GBRAS-Net) or stops the architecture tests; then the [ideas board](active/ideas-board.md).~~ Best model: SRM-e50 (standard-256 P_E 2.54% at its validation cut-off). Ideas: [ideas board](active/ideas-board.md).

## Current state

| Area | State (2026-10-19, end of session) |
| --- | --- |
| Best model | SRM-e50 (`srm-e50-20261017T111310Z-0214`, eval `35afd1c7938bae78968c`): standard-256 P_E 2.54%; ALASKA seen / unseen / LSDIR 19.42 / 16.30 / 9.39% at its validation cut-off |
| Evaluation | `scripts/evaluate_run.py`; default = standard-256-v1 suite + hard-256 test (`data/test/standard-256-v1/`, `data/hard-test/256-v1/`); one frozen threshold at min P_E on the run's validation; option C mapping for old models (`configs/evaluation-calibration-v1.json`). ~~`--test-suite legacy-512` keeps the old suite.~~ Retired 2026-10-21; code archived ([record](active/legacy-512-retirement.md)). Report notebook: `scripts/build_report.py` ([guide](active/report-notebook.md)). All 11 models plus runs 10-12 are scored on it. |
| Validation | New runs use train-matched validation (10% of training pairs; frozen recipe, for example `60da...`) ([plan](active/plan-train-matched-validation.md)) |
| Experiments done 2026-10-19 | Checks 1, 2, 2b ([plan](active/plan-diagnostic-checks.md)); #9 curriculum, closed ([plan](active/plan-low-bpp-curriculum.md)); #10 arm A delayed pooling, no gain ([plan](active/plan-architecture-tests.md)) |
| Open user decision | #10 arm G (GBRAS-Net adapter G1 / G2 / G3, or stop); see the [handoff](active/handoff-2026-10-19.md) |
| Ideas | [Ideas board](active/ideas-board.md); nothing starts without a user decision |
| G4 runs | [Run queue](active/run-queue.md): 12 runs, all evaluated. One run per boot; normal limit `--timeout-min 165 --max-run-minutes 180`. **The GCE max run duration is still 300 min** (raised for run 12; the user may set it back to 180). G4 resume after preemption is not built: a preempted run restarts with a new job ID ([run guide](active/g4-batch-runner.md)). |
| PC pilots | `--set pilot=true` for PC smoke runs (4,096 draws per epoch, fixed 256-pair slice of the real validation list); refused on the G4 and by the evaluator ([how-to](active/g4-batch-runner.md)) |
| Hardware | G4 `rtx6000pro` TERMINATED (last boot: run 12, 10:07-13:45Z 2026-10-19). Pi online, `/srv` 473 GB free; 2 GiB zram, no swap file ([Pi record](active/pi-host-changes.md)). PC container `aai3001` idle. |
| Branches | `yx` pushed. `main` stays at PR #2 until `yx` is merged as a whole (pending, C1; expect add/add conflicts from PR #1). |
| Layout | Run outputs in git-ignored `runs/<run>/` (best + `last.ckpt`); photos in `Datasets/originals/{raise,div2k,alaska,lsdir}` on the PC and the G4; shared code in `cvdl/` ([plan](active/plan-cvdl-package.md)); experiment notebooks in `notebooks/experiments/`; tests in `tests/` (360 pass; fast level 331 + 29 skipped; archived tests in `tests/archive/` are not discovered, see [test levels](active/test-levels.md)). ~~Golden check: `python -B scripts/refactor_golden.py compare`~~ Golden retired 2026-10-21 ([history](archive/refactor-golden.md)); comparison constants test instead ([list](active/settings-and-seeds.md#comparison-constants-list)). |
| PC rules | At most 12 real data workers; total RAM under ~20 GB; kill whole process groups; after each run `pgrep -af spawn_main` empty and `/dev/shm` normal; workers ask before GPU or heavy CPU work. |
| Deferred | Per-cell training: design only ([plan](active/plan-per-cell-training.md), out of date; board #3). Prebuilt writer ([design](archive/plan-otf-generator.md)); noisy covers ([parked](archive/plan-noisy-covers.md)); G4 GPU generator ([shelved](archive/plan-gpu-generator.md)). |

## Pending items (waiting for the user)

| Item | Record |
| --- | --- |
| #10 arm G or stop | [Architecture plan](active/plan-architecture-tests.md), [handoff](active/handoff-2026-10-19.md) |
| Seed repeats | Board #5. Declined for the architecture tests (2026-10-19); still open in general. |
| BF16 / front-end upgrade / longer training | Board #6, on hold. Front-end ladder (fixed SRM first, learned filters only with evidence): [architecture plan](active/plan-architecture-tests.md). |
| A2 report notebooks for all models; A3 results write-up; A4 `findings.md` rewrite; C1 merge `yx` into `main`; C2 G4 spend record | [Handoff](active/handoff-2026-10-19.md), section 4 |
| ~~ALASKA train-only models: six or two~~ | Settled: option B used. ~~Fair control R0~~: dropped 2026-10-16. |

## Key commits and records

| Commit | Record |
| --- | --- |
| `e592cce`, `1b54858` | Opt-in source readers/mixer and capped checkpoint validation. |
| `daf893c`, `2659b3d` | Final ALASKA and LSDIR records. |
| `1a6a6bf`, `3a6b1fd` | Sanity-validation fix and successful G4 checkout. |
| `ab9d3f1` | Mixed-source [result](results/result-newdata-1.md). |
| `042903d` | PC [check B](results/result-check-b.md). |
| `a6170ed` | Per-cell [design only](active/plan-per-cell-training.md). |
| `465de12` | Latest dated journal update before this documentation reorganization. |
| `45ec074` | Documentation reorganization (status, findings, journal by day). |
| `99ca3a1` | Keep only last + best checkpoints and one run log; old per-epoch checkpoints removed. |
| `5881bbd`..`7d56dbe` | One `runs/` folder for all runs. PR #2 (merged, `0981ce7`) brought the same move to `main`. |
| `76f6257`, `9798b56` | Data layout `Datasets/originals` + `generated` (PC), G4 scripts and one-time G4 reorganisation. |
| `ac8810a`, `d6a3875` | Model-copy parity test; code tidy-up (122 tests). |
| `632dd22`, `0d851b6`, `2841fdf` | SRM front end, SRM notebook, SRM-e50 notebook. |
| `863d9fb`, `efde4df` | Packed loader (`pack_blocks=16`, default) and PC loader scripts. |
| `7567501`, `d721e06` | Evaluator: no training-record writes; restricted `--set`. |
| `795fd93` | Opt-in `k_weights` / `bpp_mixture`; `srm-hard` notebook. |
| `d74d716` | Hard test v1 (270 tests). |
| `131492b`, `bd36403`, `a56c275`, `68e4f0b` | Train-matched validation; standard-256 suite; hard-256 test; one deployment threshold. |
| `716b2cd`, `fa17f25`, `c1ca714`, `675c712` | Report notebook and revisions 1-3. |
| `ed52419`, `9563540`, `d3ae6b8` | Diagnostic checks 1, 2, 2b. |
| `c501603`, `04415c0` | Low-bpp curriculum code and decision evidence (runs 10-11). |
| `1bb7e20`, `aec69d0`, `9dfaa27` | Arm A delayed pooling; PC pilot mode; arm A decision evidence (run 12). |
| `891aefe`, `a1ae22c`, `c7b5176`, `4959099`, `3c3dc39`, `7921d58`, `72f51b5` | Fast/full test levels and the changed-path helper; resume test without the duplicate golden check; `TESTS: NONE` for Markdown-only changes; test trim ([record](active/test-levels.md)). |

The [documentation index](README.md) lists each current, result, and historical record. The [run guide](active/g4-batch-runner.md) has the G4 launch routine.
