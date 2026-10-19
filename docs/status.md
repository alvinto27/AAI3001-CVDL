# Project status — 2026-10-19

**Start here:** Read the [2026-10-19 journal](journal/2026-10-19/README.md). Evaluation redesign is done (standard-256 and hard-256 tests, one frozen deployment threshold from validation, report notebook); all 11 documented models are rescored. Diagnostic checks 1, 2 and 2b are done ([plan](active/plan-diagnostic-checks.md)): #12 raised, #3 lowered, tanh fix struck. Low-bpp curriculum (#9) done: rule not met; small real gain (+0.02 k=1 low AUC) at +0.56 points standard P_E; not adopted ([plan, result](active/plan-low-bpp-curriculum.md)). **Next: the user picks from the [ideas board](active/ideas-board.md).** Best model: SRM-e50 (standard-256 P_E 2.54% at its validation cut-off). Ideas: [ideas board](active/ideas-board.md). The items below date from 2026-10-17 and are partly stale (cleanup pending).

## Current state and items

- **G4 disk uses the new data layout since 2026-10-16 06:04Z** (checkout `9798b56`, free disk 116 GB). Commits before `76f6257` no longer match the G4 disk.

- **Layout (2026-10-16):** all run outputs in git-ignored `runs/<run>/` (best + `last.ckpt` only, one `run.log`); photos in `Datasets/originals/{raise,div2k,alaska,lsdir}` on the PC and the G4; `V3_SOURCE` defaults to `decode` (the cache is deleted; rebuild it to use `cache`). See `runs/README.md` and `Datasets/README.md`.

- **G4 run queue:** [`active/run-queue.md`](active/run-queue.md) lists every approved run and its state. Check it before and after each launch.

- **Evaluation (2026-10-17, to be replaced by the redesign below):** one tool, `scripts/evaluate_run.py`. **Default = standard suite (main stage) + hard test v1.** `--padding-check`, `--temperature`, `--tanh-saturation` run only on request. `--set` accepts machine settings only. The evaluator never writes into a run's training record. See the [2026-10-17 journal](journal/2026-10-17/README.md).

- **Hard test v1 (`d74d716`):** `data/hard-test/v1/`, frozen. 4 panels (RAISE+DIV2K, ALASKA seen, ALASKA unseen, LSDIR) x k 1 and 2 x {all bpp, low bpp 0.05-0.15, noisiest quarter, low bpp and noisy} = 32 cells x 1,008 pairs. Test-split photos only; noise edges from validation only. Reported next to the unchanged standard test. Rebuild check: `scripts/build_hard_test.py --run runs/srm-20261017T052857Z-4521 --noise runs/srm-20261017T052857Z-4521/post-training/breakdown/clean_tile_noise.csv --check`.

- **Next G4 run:** `srm-hard` (`795fd93`, `configs/train-hard-v1.yaml`): RAISE/ALASKA/LSDIR 72,000 pairs each per epoch, k=1 50%, half of bpp draws from 0.05-0.15; 30 epochs; ~112 min. **On hold** until the train-matched validation exists (user decision 2026-10-17 late: validation follows training).

- **Evaluation redesign (user, 2026-10-17 late):** standard test 256 tiles, k 1-4, bpp 0.05-0.5, overlap-last for all sources; k 5-8 opt-in; hard test rebuilt at 256 (32 x 1,008); one threshold at min P_E on validation as the main report (changed 2026-10-18; per-source thresholds reference only); one report notebook per run. Details: [2026-10-17 handoff](active/handoff-2026-10-17.md), section 2.

- **PC rules:** at most 12 real data workers; keep total RAM use under ~20 GB; kill whole process groups; after each run `pgrep -af spawn_main` empty and `/dev/shm` normal; workers ask before GPU or heavy CPU work.

- **Notebook refactor (done):** shared code moves into the `cvdl/` package (`active/plan-cvdl-package.md`). Notebook 04 is frozen as the v3 baseline; new work starts from 05. Phases 0-5 done (`bc7eba2`..`8c1c0dc`): models, metrics, data, training, settings, evaluation; notebook 05 is 10 short cells with experiment settings in its top cell (`settings = load(...)`). New experiment = copy notebook 05, fill the top cell. G4: `scripts/g4/submit.sh run --notebook PATH --timeout-min N [--dry-run]` (pre-flight on the PC; see `active/g4-batch-runner.md`). G4 smoke passed (tiling arm 1). Check after each phase: `python scripts/refactor_golden.py compare` (CPU, exact; `active/refactor-golden.md`).

- **Code tidy-up (2026-10-16):** one-off checks moved to `scripts/archive/`, Mode 2 (full-photo/MIL) dropped and archived in `LSB-Dataset-Generator/archive/`, notebooks 01 and 03 in `notebooks/archive/`, all tests in `tests/`. See `scripts/archive/README.md`.

- **Completed:** ALASKA/LSDIR source records and mixed sampler; final verified masters; G4 mixed run and [result](results/result-newdata-1.md); PC threshold/noise check B and [result](results/result-check-b.md). The frozen standard test and RAISE/DIV2K records remain in use.
- **On hold:** A, per-cell training (design only, no code or model run); C, capacity/longer training/BF16; seed repeats, front-end changes, and other free checks. No follow-up is authorized by the plans.
- **Deferred records:** prebuilt writer, OTF plan step 17: [historical design](archive/plan-otf-generator.md), OTF implemented but writer deferred. Noisy covers: [parked by the user](archive/plan-noisy-covers.md). G4 GPU LSB generator: [shelved design](archive/plan-gpu-generator.md), not built. G4 resume after Spot preemption: postponed until C (longer training); today a preempted G4 run restarts with a fresh job ID (see [run guide](active/g4-batch-runner.md)). One boot is limited to 180 min since 2026-10-17 (`--timeout-min 165 --max-run-minutes 180` for long runs).
- **Hardware and delivery:** the G4 is **stopped** (last boot: SRM-e50, 11:13-13:48Z on 2026-10-17). `main` stays at PR #2 until `yx` is merged as a whole (expect add/add conflicts from PR #1). The Pi is online. Its USB-SSD swap file was removed; **2 GiB zram only** remains. See the [Pi record](active/pi-host-changes.md) and [2026-10-16 journal](journal/2026-10-16/README.md). Do not start the VM to resend a result.

## User decisions still pending

These rows carry the pending decisions from the archived [research checklist](archive/research-checklist.md). They are **not** authorizations to act. The checklist's ALASKA option-B row is historical: the later source record and mixed run already use the selected option B; confirm any further selection request with the user before changing records.

| Decision | Owner | Current record |
| --- | --- | --- |
| ALASKA train-only models: six or two (option B) | User | Checklist marked Pending; Orchestrator recommended option B. Final selection used option B; do not reopen it without a new decision. |
| BF16 for G4 | User | Pending; speed matrix complete, batch 32 retained. |
| Fair control R0 | User | **Dropped** (2026-10-16): no normalisation is kept for detection and maps; the GroupNorm cost is not needed. |
| Seed repeats | User / Orchestrator | Pending; one seed is not enough for small differences. |
| Front-end upgrade | User / Orchestrator | Pending after the wide and new-source results. |

The old checkpoint-selection validation decision is **done**: the mixed run used the fixed 8,035-pair capped list, not the earlier checklist estimate. Free checks still listed as pending in that checklist are noise-versus-miss, denoise-and-compare, tanh saturation, and a filter linear probe. Check B completed a noise-versus-error analysis; do not relabel the other checks as done or start them while work is on hold.

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

The [documentation index](README.md) lists each current, result, and historical record. The [run guide](active/g4-batch-runner.md) is reference material only while work is on hold.
