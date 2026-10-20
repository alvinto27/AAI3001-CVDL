## 2026-10-20 — Review of pre-run checks; test levels

Recorded by the Orchestrator. Team: Orchestrator, Worker, Run_Supervisor (new sessions, intercom room `CVDL_Work`).

### Did

- User paused arm G (G1) to review the checks before a run (tests, regression tests, smoke tests, speed tests): they take too long.
- Orchestrator read the 10-18/19 worker session records and logs. Measured gate costs: full CPU suite 490-517 s (406-424 tests), golden compare about 148 s (settings 24 s, data 59 s, models 38 s, cpu 27 s), PC pilot 208 s, old-style smoke notebooks 264 s and 269 s. Arm A example (10-19 UTC): code 2 min, then full suite 9 min, GPU pilots 21 min (about 5 attempts: batch 128 out of memory on the 16 GB PC GPU, 512 validation out of memory, a settings guard refusal; speed estimate later 2.5x wrong), golden 3 min; then pilot mode built with another suite + golden (21 min).
- Worker profiled the full suite (read-only): 424 tests, 0 skipped, 490.9 s wall. `test_lsb_torch` 330 s of test time (67%); one test (packed-loader tail order, 36 worker/pack/tail cases, each starting loader workers twice) 205.7 s (42%). `test_cvdl_training` 33.5 s, near-duplicate of the golden `cpu` stage (it adds a 2-step resume). Frozen-list rebuilds about 48 s. Tests tied to one notebook (frozen 04, archived 05, finished experiments; 17 files): about 86 s run alone. `test_tanh_saturation` runs 4 inherited tests twice.
- User question (why the golden compare checks notebook 05): all 13 run notebooks have code cells 2-10 identical to notebook 05; only the settings cell differs. The golden record (10-16) proves the move of notebook 05's code into `cvdl/` changed nothing. It covers the old default path (512 tiles, newdata-1, 4 old checkpoints), not the newer opt-in features (SRM, 256, train-matched validation, curriculum, delayed pooling, pilot).
- Worker built test levels: `891aefe` (`scripts/test_cpu.py`, `scripts/required_test_level.py`, `tests/test_cpu_levels.py`, 17 new tests). Fast 160.6 s (413 run, 28 skipped); full 495.4 s (441 run, 0 skipped). Audited golden compare ALL EQUAL (143 s): 37 tracked files opened, all matched by the helper rule. Orchestrator check: helper replay on past commits gives the expected decisions (`9dfaa27` FAST/no golden; `1bb7e20` FAST/golden; `863d9fb` FULL/golden); unreadable base gives FULL/REQUIRED; 17 new tests pass. Pushed.
- Worker: C done `a1ae22c` (resume test without the repeated 20-step golden comparison; 33.5 s to 28.7 s). Orchestrator review: only the test file changed; resume, best checkpoint and metrics-prefix checks kept; loader shutdown in `finally`. User: keep. Pushed.
- Worker: helper answer `TESTS: NONE` for Markdown-only changes, `c7b5176`. Code reads two doc files (`plan-architecture-tests.md` is hashed by `architecture_decision.py`; `data-sample.png` by figure and report tests); both stay FAST. 21 helper tests; fast level 417 run, 28 skipped, 155.9 s. Pushed.

### Found

- The checks are slow mainly because the same full checks run after every change, whatever the change touches. The biggest single cost is one exhaustive loader test (process start-up, not data), not the notebook-tied tests (suspected by the user; real but about 16%).
- The Orchestrator's first path rule claim was wrong in part: `test_lsb_torch` also reads `configs/raise512-run1.yaml` and `configs/test-bpp1.yaml` (Worker check before coding). Both are on the FULL list.

### Decided

- User: build A (fast/full levels), B (shrink the 36-case loader test in the fast level, all cases kept in full) and C (drop the repeated 20-step check from `test_cvdl_training`, keep its resume check), in the order A, B, C.
- User: path rule narrowed by the Orchestrator (full only for loader code and its test inputs, dependency files and unknown paths; fast otherwise).
- User (after C): C kept, though the gain is only about 5 s. B3: leave the loader tests as they are. Add `TESTS: NONE` for Markdown-only changes.
- User: golden compare only when the helper says REQUIRED (shared code or its exact inputs), plus always before a G4 launch. Old rule "full suite + golden before every push" replaced ([test levels](../../active/test-levels.md)).

### Next

- ~~B, then C (Worker, one task at a time).~~ C done; B not done (B3, user): the loader workers start and stop on purpose (RAM fix 2026-10-14, `spawn` workers), so the cases cannot share workers.
- Pending: arm G (G1) stays paused until the user resumes it.
