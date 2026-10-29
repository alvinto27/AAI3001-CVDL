# 2026-10-29 — run 21 trained heatmap and local correctness pilot

Entries are chronological. Earlier no-GPU statements describe their bounded implementation tasks; the separately authorized local probe is recorded at the end.

## Did

- Recovered and reviewed the interrupted worker's uncommitted edits in eight `cvdl` modules and its new experiment notebook. No existing edits were discarded.
- Completed the narrow run 21 changed-cell path: 257-parameter spatial head, one shared feature pass, strict weights-only source checks, fresh optimizer/scheduler, weights-only start records, and resume refusal before record overwrite.
- Exposed auxiliary logits beside signed covariance attribution. Head-only photo inference skips gradient attribution; explicit dual inference uses one feature pass. Kept old inference defaults and unchanged covariance math/`SPEC`.
- Added [the plan](../../active/plan-run21-cell-heatmap.md), notebook guidance, seven focused synthetic/read-only tests, and one existing test expectation update for newly supported cell supervision.
- Checked the host date (`2026-10-29`). Checked local SHA-256 values for run 21 BEST, run 21 last and run 18 last; all match the plan. The read-only run 21 BEST load preserves its file hash, all copied base tensors and exact detector outputs after adding the head.
- Ran only focused container CPU checks through the shared runner. No training fit, GPU job, heavy CPU/full suite, remote operation, commit, push, or dependency change.

Final focused command:

```sh
podman exec -w /work aai3001 python -B scripts/test_cpu.py --tests tests.test_gapcov_cell_training tests.test_cov_pool tests.test_photo_heatmap tests.test_cell_training.CellTrainingTests.test_cell_loss_missing_classes_sparse_weight_and_finite_gradients tests.test_cell_training.CellTrainingTests.test_warm_start_only_head_missing_and_strict_checkpoint_reconstruction tests.test_cell_training.CellTrainingTests.test_feature_off_existing_training_path_pixels_order_rng_and_recipe
```

**20 tests PASS, 3.147 s, no skips.** This includes 7 new tests, 7 covariance tests, 3 photo-stitching tests, and 3 existing cell-training tests. Before final hardening, the same command passed 20 tests in 3.482 s. Initial development runs of the new module had two test-fixture override errors (not product failures), then 6 PASS in 2.065 s; the seventh test adds the local BEST read-only check.

Historical gate command:

```sh
podman exec -w /work aai3001 python -B scripts/test_cpu.py --tests tests.test_gapcov_context.IdentityTests.test_same_literal_routes_and_frozen_source
```

**1 test FAIL, 0.000 s.** Current `models.py` hash `403023c538d217f243b7f5f6c1de8f567928e7d2ab572ac105429315acd63c5d` differs from historical approved `09d3e41599a7cd28c167dacf95a67ce91ec2bca95b1ecfb5e73a491289a432bb`. The historical approval constants remain unchanged. This is a known launch blocker, not a waived check. FAST/FULL are not claimed green and were not run.

## Found

- The spatial head needs 256 input channels, not the classifier's 784. The trained head sees 48×48 patches at stride 32; covariance attribution uses tile-global 256×256 context. Neither is a calibrated pixel probability.
- The old early class-score return hid auxiliary maps. Computing the auxiliary output in the same attribution feature pass fixes it without changing feature-off attribution math.
- A generic scratch initializer would overwrite the continuation start record as random. The weights-only path now excludes that call; tests check both fresh start and resume.
- `runs-out/` is absent on the PC. The notebook's staging path needs a native `runs/` initializer override for any separately approved local run. No staging was done.
- `scripts/photo_heatmap.py` still assumes trained head + CAM with CAM cutoffs and labels. Shared inference supports covariance dual/head-only maps, but that CLI is a **Pending** follow-up.
- Historical `cov_pool` decision source approval rejects the extended model file (`cov_pool source drift`). Source-approval versioning and a two-reference report/decision workflow need separate review; no CLI or frozen hash changes were made here.

## Decided

- User-approved recipe: 15 paired epochs from **run 21 BEST**, weights only, LR 0.00003, warmup 1, no drops, original embedding schedule, CE + 0.2 changed-cell BCE. Same source mixture .4/.2/.4, 180,000 draws, 128 images, 256 overlap-last tiles, train-matched recipe `60da4391043e8ce48ff2`, validation cap 64, evaluate new last checkpoint.
- Detector reference: **run 21**. Trained heatmap reference: **run 18 last**.
- Evaluation gates are **PROPOSED / Pending (user)**: standard P_E upper 95% difference ≤ +0.0025 versus run 21; hard k=1 low AUC lower 95% difference ≥ −0.01. Compare trained heatmaps against run 18 and require candidate FA ≤7% at its own validation 5%-FA cutoff. No +0.02 gain requirement, test-fitted cutoff, or automatic adoption.
- Keep run 18 as submission/`pipeline` default. Implementation approval does not authorize launch.

## Next

- **Pending (user):** review proposed gates, source-approval blocker, next launch/pilot decision and later complete test requirements.
- **Pending:** photo-heatmap CLI labels/map choices, separate detector/cell reference report/decision support, and native initializer override or approved staging.
- **Pending:** all training results, GPU stability, detector and heatmap evaluation, and adoption. Worker stopped after the core report, then was reactivated for the bounded source-lineage task below. No launch in either task.

## Source-lineage follow-up

### Did

- Confirmed the source-version policy with the orchestrator before edits. Preserved original `COV_APPROVAL`, `MODEL_SHA`, math/notebook hashes, verdict rules and saved run 21 evidence.
- Added exact `run21-original-v1` and `cell-aux-v2` source scopes in `cvdl/gapcov_provenance.py`. Pinned the actual run 21 launch revision from its saved settings: `9425d07cbf474ae20a8f9bf0bddb6eb14b9c23cc`, not a moving branch or short revision.
- Bound v2 provenance into new training-start/model records and checkpoint hyperparameters. New start and resume check the executing source and trusted lineage before records change; resume also checks all three saved identities.
- Kept original decision/context/usage commands pinned-only. Added explicit live refusal tests. Historical metadata tests now use actual pinned Git source files, including the old cell-diagnostics file; they do not claim that historical hashes approve current inference.
- Added the source-fixture helper and provenance validator to the changed-path helper's FULL dependencies, with a regression assertion. No FULL suite was run.

Final focused command:

```sh
podman exec -w /work aai3001 python -B scripts/test_cpu.py --tests tests.test_gapcov_provenance tests.test_gapcov_cell_training tests.test_gapcov_context tests.test_cov_pool tests.test_learn_srm tests.test_wiser_b.ArchitectureTests tests.test_wiser_b.RuleTests tests.test_wiser_b.ContextTests.test_model_only_declared_addition_and_original_blob tests.test_wiser_b.ContextTests.test_original_validation_and_code_drift tests.test_wiser_b.ContextTests.test_context_manifest_drift_and_sibling_guards tests.test_full_evidence.PolicyTests
```

**52 PASS, no skips, 4.844 s.** This includes 6 new provenance tests; existing core, historical source/rule and test-policy checks also pass. The FULL-only 29,280-image panel replay was not run.

Earlier diagnostic commands:

```sh
podman exec -w /work aai3001 python -B scripts/test_cpu.py --tests tests.test_gapcov_provenance tests.test_gapcov_cell_training tests.test_gapcov_context tests.test_learn_srm.IdentityTests tests.test_learn_srm.ContextAndMovementTests tests.test_wiser_b.ContextTests.test_model_only_declared_addition_and_original_blob tests.test_wiser_b.ContextTests.test_original_validation_and_code_drift
```

24 tests, 1 error, 3.649 s: the usage-report test attempted its historical inference command against extended live source. Its gate correctly refused (`usage head source drift`). The test now checks that live refusal is read-only instead of lending old hashes to current inference. A subsequent command equal to the final command without `tests.test_full_evidence.PolicyTests` passed **48 tests in 4.846 s**.

### Found

- Exact historical launch blobs are available locally: model `09d3e41599a7cd28c167dacf95a67ce91ec2bca95b1ecfb5e73a491289a432bb`; math `31cf8ecb4eb7f36c12b789f669f7c13f23ff51f0fdb8be46d6e33ecdaeb6ea28`; original notebook `bb67c346c1886a94af2e8dee43c953747a25fc1c0ce8d772182c56253e825fcd`.
- V2 separately pins actual executing model `403023c538d217f243b7f5f6c1de8f567928e7d2ab572ac105429315acd63c5d` and new notebook `7200edda277584b2a734a40640eb445b209d4233a456428e59281e9c5f180429`, plus unchanged math/original notebook. Canonical `SPEC` hash is `a16afe93634a5acf98b1d06353418e85ae4220c427ec7b4038e48dccd76de02f`.
- Tests refuse changed model/math/both notebook bytes, changed live `SPEC`, wrong full historical revision, wrong historical blob bytes, missing local Git object, a substitute source directory, and changed/missing saved provenance. New-start and resume refusal leaves records unchanged.
- Existing learn-SRM source fixtures had pinned only models.py; the new shared fixture pins all historical `cvdl` source, so the changed diagnostics cannot leak into an old byte approval. No historical model inference is run under these fixtures.

### Decided

- Original commands remain pinned-only: `training_decision.py --rule cov_pool`, GAP/covariance context v1 and `report_gapcov_usage.py` are deliberately refused in the extended live checkout. Reproduce them in an actual trusted pinned checkout when separately needed; do not execute extended runtime under old approval.
- New cell start/resume uses v2. It verifies BOTH the trusted original Git blobs and exact current executing bytes/paths/live `SPEC`; old blobs alone are insufficient. Missing Git objects refuse, without fetch. Full policy and hashes are in the [plan](../../active/plan-run21-cell-heatmap.md).
- The original source-assumption test failure is resolved by correct versioned testing, not by changing its historical hash. Future source changes need explicit reviewed versioning. Full-suite success and launch readiness are not claimed.

### Next

- **Pending:** review this bounded provenance change, then separately address the remaining photo CLI/two-reference reporting, complete test requirements, native initializer/staging and launch decisions.
- No GPU, full suite, remote/cloud operation, launch, commit, push or dependency changes in this follow-up.

## Map evaluation and photo CLI follow-up

### Did

- Completed map-specific cell reporting and rendering. Covariance attribution is signed/context-only with global RF256/stride32; auxiliary logits are scored normally with RF48/stride32. Covariance differences use `aux_minus_class_score`; old CAM reports keep `aux_minus_cam` and their schema. Removed the legacy absolute auxiliary target mark for the covariance candidate.
- Added checkpoint/source/manifest binding for new saved row CSVs, score arrays and summary, with independent file hashes in a sidecar. Existing run 18 output was not edited or assigned invented metadata.
- Added explicit photo CLI `head-only` and `head-attribution` choices, own calibration/summary identity checks, validation-only cutoff/limit checks, and per-map captions/geometry. Head-only actual synthetic CLI execution avoids `autograd.grad`; covariance CAM refuses. Original run 13 defaults were verified read-only against pinned inputs.
- Added `scripts/compare_run21_cells.py`: a narrow saved-output numeric report against detector run 21 BEST/eval `cdc91a4b6d63a1517064` and run 18 auxiliary maps. The reference pin record covers nine detector evaluation artifacts and all seven files in the existing run 18 cells folder (six non-PNG artifacts plus `cells.png`). The pin-record SHA-256 is `8f7deb06dcab576001ef69055cc840054c329e2592e2db265242cc7aadd606f4`.
- Reused strict row joins, own validation fits and existing paired photo bootstrap/cell comparison code. The wrapper snapshots inputs before/after, checks complete counts/targets/identities and refuses quick/pilot/stale/overlapping/mismatched inputs. Its three proposed guard booleans are explicitly Pending (user); no frozen PASS/adoption field or reused `cov_pool`/`paired_r2` approval.
- Added 11 focused workflow tests. Updated one existing test import to package-qualified form so shared-runner focused imports can reuse its synthetic fixtures. No model, covariance-math or notebook edits in this follow-up.

Final command:

```sh
podman exec -w /work aai3001 python -B scripts/test_cpu.py --tests tests.test_run21_map_workflow tests.test_gapcov_cell_training tests.test_gapcov_provenance tests.test_cov_pool tests.test_photo_heatmap tests.test_cell_metrics tests.test_training_decision
```

**50 PASS, no skips, 8.897 s** on the final rerun (earlier run: 8.976 s). The final rerun redirected runner output to `/tmp/run21-map-tests.log`, displayed its result and removed that temporary file. Includes new cell-eval binding/CLI checks, actual synthetic photo head-only/dual CLI invocation, cutoff/identity/metadata/target/row/quick/pilot refusals, changed-during-processing refusal, strict calibration compatibility, reference-pin checks, legacy run 13 and existing covariance/source/metric/decision checks. Synthetic paired photo bootstrap uses the required 2,000 draws; no full real-data comparison was run.

Intermediate checks: first focused workflow import failed because an existing test used an unqualified sibling import; changed it to `tests.test_architecture_decision`. Seven workflow tests then passed in 1.050 s. The intermediate expanded command (without `tests.test_gapcov_provenance`, before two additional workflow tests) passed 42 tests in 8.161 s. No product-source hashes were changed to make tests pass. After adding a read-only check that the pinned run 18 auxiliary cutoff/limits exactly match its own validation arrays, the final workflow-only rerun was:

```sh
podman exec -w /work aai3001 python -B scripts/test_cpu.py --tests tests.test_run21_map_workflow
```

**11 PASS, no skips, 2.706 s.** Output used the same temporary-log wrapper and the log was removed. `git diff --check` passed. Model/math/new-notebook SHA-256 values remain exactly the v2 pins.

### Found

- The existing run 18 cell folder has seven files, not six when the PNG is included. Exact artifact pins allow comparison without pretending its old summary contains a new checkpoint binding.
- Run 21 calibration origin is `post_hoc_256`; a new candidate's own calibration is `train_matched`. The wrapper permits this origin difference only when recipe/list/geometry/policy all match exactly. Both numeric cutoffs are checked from their respective validation predictions, never copied or fit on test rows.
- Shared source-photo bootstrap remains 2,000 draws, detector seed20261019 and cell seed20261021. These intervals do not measure training-seed uncertainty.
- Numeric cell cutoffs are logits, not pixel probabilities. Photo any-tile detection is not a calibrated photo false-alarm rule. Captions and output metadata now make those limits explicit.

### Decided

- Preserve the original trained-head/CAM defaults for run 13. Non-default photo runs must name their own evaluation and cell summary; covariance requires explicit head-only or head+attribution selection.
- Comparison guards remain PROPOSED / Pending (user): standard error difference upper95 ≤ +0.0025, hard AUC difference lower95 ≥ −0.01, candidate cell FA ≤ .07 at its own validation5% cutoff. No +.02 requirement, new absolute AUC threshold, automatic adoption, submission or pipeline change.
- [Plan](../../active/plan-run21-cell-heatmap.md) contains the future command recipe and exact scope. It is not permission for resource-intensive evaluation.

### Next

- **Pending:** orchestrator review, later full-suite evidence and launch/staging decisions. Real candidate training, inference, metric comparison and photo quality are unmeasured.
- No training, GPU, full suite, cloud/remote operation, commit, push or dependency changes. Stop after this bounded evaluation/inference task report.

## Separately approved local GPU correctness pilot

### Did

- Added dedicated `scripts/pilot_run21_cells.py` and focused CPU tests; did not reuse the scratch pilot. Verified host date2026-10-29 and native BEST path/SHA. Exact base copying, only257 new head parameters, fresh Adam/scheduler, paired pixel/tag alignment, finite gradients and frozen SRM were checked on real GPU data.
- Preflight: GPU idle/no processes,428MB VRAM used,27GiB RAM available, `/dev/shm`92KiB. Used four data workers, zero eval workers, own process group, test-suite kernel lock,12GiB RAM floor,600-second watchdog and private disposable caches/checkpoints. Preserved FP32 and the ROCm unfold-wgrad workaround.
- Used `pilot=true`, `quick=true`, training-photo fraction.005, batch16 images/8 pairs,4,096-draw cap and256-pair validation slice. Verified full18,000-pair train-matched recipe first. Kept declared15 epochs, original embedding schedule and LR3e-5/warmup1; post-fit none and prime off. This is selected-data correctness evidence, not accuracy or full-source/G4 speed evidence.
- First CPU tests: **4 PASS,1.408 s**. First GPU attempt: **40.737 s,exit1**, because the final evidence comparison mixed saved CUDA SRM and Lightning-teardown CPU tensors. Substantive GPU checks completed, but this was not a clean pass. Preserved its blocked evidence and exception unchanged.
- Reported the error immediately. Orchestrator authorized detached-CPU comparison plus regression and exactly one identical-profile rerun. Updated focused command below: **5 PASS,1.426 s**. Authorized GPU rerun: **39.577 s,exit0**, including final frozen-SRM equality.

```sh
podman exec -w /work aai3001 python -B scripts/test_cpu.py --tests tests.test_run21_cell_pilot
```

### Found

Rerun32-update replay of one real batch (48,621 changed pixels), through shared `training_step`:

| Loss | First8 mean | Last8 mean |
| --- | --- | --- |
| CE | .153356623 | .120829844 |
| BCE | .596875519 | .598153323 |
| CE + .2 BCE | .272731729 | .240460513 |

- Replay6.322 s. Total falls; BCE rises slightly. No auxiliary-learning/accuracy claim and no64-step extension.
- Compiled actual `return_aux` forward/backward15.710 s. Tolerance1e-4 on per-tensor maximum absolute error divided by eager maximum absolute value; tile1.884e-7,aux3.227e-7,max gradient2.249e-6. Eager reconstruction matches exactly; no `_orig_mod` state keys. Finite head-only/dual maps `[2,8,8]`.
- Fresh8 updates took.749188 s (10.6782 steps/s), PC-only. GPU peak allocated11,452,424,192/reserved15,378,415,616 bytes. Minimum available RAM21.961GiB.
- Actual Lightning checkpoint at epoch0/step1, uninterrupted step2, shared-runtime rebuild and `ckpt_path` restored step2: weights, Adam moments, scheduler and epoch/global step **bit exact**. Both returned at epoch1/step2. Source provenance retained and pilot checkpoint evaluator-refused. Cached same-next-batch check does not prove arbitrary mid-epoch data ordering.
- Final external cleanup: no GPU/Python/worker/compiler processes;428MB VRAM used/15,876MB free,27GiB RAM available,shm92KiB. No disposable directories/checkpoints remain. BEST/last SHA and pinned model/math/notebook bytes unchanged.

### Decided

- Retain original `runs/verification/run21-cell-pc-pilot-2026-10-29/` blocked evidence and clean `...-r2/` evidence. Each has small `evidence.json`, `pilot.log`, `exit-status`, external `cleanup.txt`. Do not relabel the first run successful.
- Full measurements, safety scope and limitations are in the [plan](../../active/plan-run21-cell-heatmap.md#bounded-local-correctness-pilot--2026-10-29). Correctness evidence only; no detector or heatmap quality result. Run18 remains default.

### Next

- **Pending:** full-suite evidence on intended launch revision, paid/cloud launch/resource decision, real training stability, evaluation guards and adoption.
- No additional GPU probe, full training, paid/cloud/remote operation, dependency change, commit or push in the pilot task. Report completion and stop.

## Local launch-preparation revision — FULL check scheduled

### Did

- The next separately authorized task reviewed status/diff and confirmed all current changes belong to the bounded implementation tasks. `git diff --check` is clean. Rechecked all four source pins and original run21 BEST/last plus run18 last SHA values; unchanged.
- Committed implementation, tests, config and notebook with explicit file paths on child-repository branch `yx`: `3d35a02`. Documentation is committed separately. No main, pipeline or superproject change; no push or history rewrite.

### Found

- Shared CPU runner defaults to FULL, holds the container-wide kernel lock and records local evidence only for a clean, unchanged passing checkout and fixture/runtime key. Focused and FAST checks cannot substitute for this check.

### Decided

- After this final documentation commit, run the container `scripts/required_test_level.py`, then exactly one serial default `scripts/test_cpu.py`. Start from clean HEAD and keep the actual results in ignored `runs/verification/run21-cell-launch-readiness-2026-10-29/` first. Do not change the tested tree to record the result immediately afterward.

### Next

- **Pending at this documentation snapshot:** FULL result, actual counts/skips/duration, clean start/end and local evidence-key checks, process/shm cleanup. Report any failure before implementation edits; preserve its log.
- Paid/cloud launch, remote operations/push, additional GPU probes and adoption are not authorized.
