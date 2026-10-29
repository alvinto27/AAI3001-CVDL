# Run 21 with a trained changed-cell head

## State — 2026-10-29

The user approved the implementation and the recipe below. **No paid/cloud launch is authorized by this record.** The separately approved, bounded local correctness pilot completed below. Full training, full tests, remote staging, proposed evaluation guards and adoption remain **Pending (user)**. No candidate accuracy result exists. Run 18 remains the submission and `pipeline` default.

The detector reference is run 21. The trained heatmap reference is run 18. These are different roles; do not substitute one for the other.

## Approved recipe

| Setting | Value |
| --- | --- |
| Notebook | [srm-cell-aux-paired-gapcov32.ipynb](../../notebooks/experiments/srm-cell-aux-paired-gapcov32.ipynb), run 16/18 fine-tune template |
| Model | Existing `xunet_wide_gapcov32`; SRM, width 4, no normalization, clip 3 |
| Start | **Run 21 BEST**, epoch 43; weights only; fresh AdamW and scheduler |
| Training | 15 epochs, paired batches, 128 images (64 pairs), 180,000 draws per epoch |
| Loss | Tile cross-entropy + 0.2 changed-cell BCE; existing `changed_cells_v1` target/loss rules |
| Learning rate | 0.00003; 1 warmup epoch; no drops; weight decay 0.0001; gradient clip 1 |
| Embedding | Original schedule, `configs/train-k1to4.yaml`; not the 50-epoch curriculum |
| Sources | RAISE .4 / LSDIR .2 / ALASKA .4 |
| Tiles | 256, `overlap_last`; source tile setting 512 |
| Validation | `train_matched`, recipe `60da4391043e8ce48ff2`; cap 64 images per batch |
| Evaluation checkpoint | New run's `last.ckpt`; no automatic post-fit evaluation |

## Read-only checkpoint identities

All three local SHA-256 values were checked on 2026-10-29. No original checkpoint was modified.

| Role | Run / checkpoint | SHA-256 |
| --- | --- | --- |
| Fine-tune initializer | `srm-e50-paired-gapcov32-20261026T115421Z-1917`, `best-epoch43-val_pe0.0329-val_auc0.9946.ckpt` | `4ba2a4b59a535b53ab9c0235c54231476b45e2193982bf28f7b7f2171d9898cf` |
| Original run 21 last (not the initializer) | Same run, `last.ckpt` | `110e64f5d4e92de7419606b98dcf320d52a1fc00ab203b3a18a7e91d5781f946` |
| Heatmap reference | `srm-cell-aux-curriculum50-paired-20261023T133537Z-1366`, `last.ckpt` | `eee9c63a0bd9a91b63758dfc4ddfcca37e47a2a621fbb663557c37bec4b00b80` |

The local initializer is:

```text
runs/srm-e50-paired-gapcov32-20261026T115421Z-1917/checkpoints/srm-e50-paired-gapcov32-20261026T115421Z-1917/best-epoch43-val_pe0.0329-val_auc0.9946.ckpt
```

The notebook uses the same suffix under `runs-out/` for G4 staging. `runs-out/` is absent on this PC. Before a separately approved PC run, override `init_checkpoint` to the native path above and keep the pinned hash. Do not copy or edit an original checkpoint. No remote files were staged.

## Implementation contract

- The spatial features have 256 channels. Add `Conv2d(256, 1, 1)`: **257 parameters**, output `[N,1,8,8]` on 256 tiles. The classifier's 784 inputs are not spatial channels.
- One feature pass feeds the original GAP + covariance detector and the auxiliary head. Covariance math and `SPEC` are unchanged. Feature-off state names, RNG, outputs and signed attribution stay unchanged.
- The start loader checks source SHA-256, exact `SPEC`, frozen published SRM bank, exact source keys and shapes before copying. Every base tensor is copied exactly; only the new head keeps its initialization. No optimizer, scheduler, epoch or step is restored from run 21.
- `training_start.json` records a weights-only start, not a random scratch start. Base and head fingerprints are also saved in model records and checkpoint hyperparameters.
- Resume checks recipe/start identity, loss, head keys/shapes, `SPEC`, frozen SRM, initialization fingerprints, and head Adam moments before run records are overwritten. Historical scratch runs keep their existing path.
- `cell_maps` exposes both trained auxiliary logits and signed covariance attribution. A covariance model never uses a linear CAM. Head-only photo inference skips gradient attribution; explicit dual maps use `('trained_head','attribution')`. Old inference defaults are unchanged.
- Trained-head geometry: receptive field **48×48**, stride **32**. Attribution geometry: **256×256 tile-global context**, stride 32. Neither map is a causal pixel mask or a calibrated pixel probability.

## Evaluation gates — PROPOSED, Pending (user)

These are not frozen decision rules and do not authorize adoption:

1. Detector versus run 21: standard-test P_E difference upper 95% bound ≤ **+0.0025** (0.25 percentage points); hard k=1 low AUC difference lower 95% bound ≥ **−0.01**.
2. Heatmap versus run 18: compare the trained-head changed-cell results on the same frozen examples. Candidate false alarms ≤ **7%**, using its **own validation 5%-FA cutoff**. Keep the reference's own validation cutoff. No new +0.02 AUC gain requirement is imposed.
3. Use validation only to fit cutoffs. Do not fit them on test results. Report paired uncertainty and both reference identities. Evaluation and adoption decisions remain Pending (user).

No submission bundle, photo-verdict policy, or `pipeline` setting changes in this task.

## Focused verification and remaining blockers

2026-10-29: [new CPU tests](../../tests/test_gapcov_cell_training.py), covariance math/model tests, photo stitching tests, and three existing cell-training tests: **20 tests PASS, 3.147 s** (final rerun; exact command in the journal). Synthetic tensors only, plus a read-only run 21 BEST load. The actual run 21 BEST base tensors and detector outputs match exactly after head attachment. All Python commands used the container shared CPU runner; no fit loop or GPU work ran.

**Initial core milestone (before the versioned policy below):** the historical source-approval test failed. It was not bypassed:

```sh
podman exec -w /work aai3001 python -B scripts/test_cpu.py --tests tests.test_gapcov_context.IdentityTests.test_same_literal_routes_and_frozen_source
```

Result: **1 test FAIL**. Current `cvdl/models.py` SHA-256 is `403023c538d217f243b7f5f6c1de8f567928e7d2ab572ac105429315acd63c5d`; historical approved hash is `09d3e41599a7cd28c167dacf95a67ce91ec2bca95b1ecfb5e73a491289a432bb`. The old `cov_pool` decision route also refused changed model source. Historical approval constants are unchanged. The source-versioning blocker was then resolved by the explicit policy below, not by refreshing old hashes. The historical test now checks actual pinned bytes; separate tests require live historical-route refusal. FAST/FULL have not been run and are not claimed green.

Pending follow-ups:

- User review of the proposed gates and launch decision; later full CPU tests on the launch revision.
- Historical source-approval handling is resolved below. The narrow two-reference numeric report is implemented in the evaluation follow-up below; it is not a frozen decision/adoption route.
- Photo CLI map choices, labels and identity checks were completed in the evaluation follow-up below. Real candidate photo scoring and its quality remain Pending.
- The bounded local pilot below verified the native initializer override. G4 staging, full training stability, accuracy, heatmap quality and adoption remain Pending.

## Source lineage policy — 2026-10-29 follow-up

The orchestrator confirmed this bounded policy before implementation. [Source validator](../../cvdl/gapcov_provenance.py) defines two exact versions, not a broad list of acceptable hashes:

| Version | Scope | Current checkout accepted? |
| --- | --- | --- |
| `run21-original-v1` | Original model, math and experiment notebook at full launch revision `9425d07cbf474ae20a8f9bf0bddb6eb14b9c23cc` | **No**: the extended model is deliberately refused |
| `cell-aux-v2` | Exact extended executing model, unchanged math, original notebook, new cell notebook and unchanged live `SPEC`; also verifies the original revision and its three exact Git blobs | **Yes**, only while every pinned byte and `SPEC` matches |

The original launch revision comes from run 21's saved `run_settings.json`. Its source hashes match the original approvals. Both scopes read and hash the trusted Git blobs; v2 additionally checks the actual executing checkout paths and bytes. A directory of old source files cannot be supplied instead of the executing v2 checkout. Historical `SPEC` is parsed as literal data from the trusted math blob, never executed.

| Pinned input | Historical SHA-256 | Extended SHA-256 |
| --- | --- | --- |
| `cvdl/models.py` | `09d3e41599a7cd28c167dacf95a67ce91ec2bca95b1ecfb5e73a491289a432bb` | `403023c538d217f243b7f5f6c1de8f567928e7d2ab572ac105429315acd63c5d` |
| `cvdl/cov_pool.py` | `31cf8ecb4eb7f36c12b789f669f7c13f23ff51f0fdb8be46d6e33ecdaeb6ea28` | Unchanged |
| Original GAP/covariance notebook | `bb67c346c1886a94af2e8dee43c953747a25fc1c0ce8d772182c56253e825fcd` | Unchanged |
| New cell notebook | Not part of v1 | `7200edda277584b2a734a40640eb445b209d4233a456428e59281e9c5f180429` |

Canonical `SPEC` SHA-256: `a16afe93634a5acf98b1d06353418e85ae4220c427ec7b4038e48dccd76de02f` (sorted compact JSON).

New cell starts validate v2 before writing run records. `training_start.json`, `model_run_settings.json` and checkpoint hyperparameters carry the full `source_provenance` record. Resume revalidates live v2 and requires the same provenance in all three saved identities before overwriting records. Missing/changed provenance refuses. Future intentional changes to pinned files require a reviewed source-version update; do not silently refresh old v1 approvals. The historical Git objects must be available locally; missing objects fail closed, with no fetch.

Supported/refused commands:

- Original `training_decision.py --rule cov_pool`, GAP/covariance context v1 freeze/score/report, and `report_gapcov_usage.py` remain **pinned-only**. Run their historical implementation in an actual checkout of the launch revision. The extended live checkout refuses original approval, even though its trusted historical blobs are available. Existing original results/manifests and decision thresholds are unchanged.
- The new cell notebook's training start/resume uses **v2 only**, not `cov_pool` approval. There is no new frozen cell decision route. A later narrow saved-score numeric report is described below.
- Normal historical metadata tests use real pinned-source trees, not fake hash results. These fixtures do not execute historical inference. The inference-capable usage-report test requires live refusal on the extended checkout rather than using a pinned-source fixture to approve current execution. The full-only panel replay uses the pinned source fixture for data replay; no model inference runs there.

Final follow-up verification: **52 focused CPU tests PASS, no skips, 4.844 s**. Exact command and earlier diagnostic run are in the [journal](../journal/2026-10-29/README.md#source-lineage-follow-up). Tests cover changed current model/math/both notebooks/live `SPEC`, wrong historical revision/blob, missing Git object, live v1 refusal, fake executing-root refusal, and new-start/resume refusal before record overwrite. Historical learn-SRM/B36 tests and verdict boundaries also pass. `tests/frozen_sources.py` and the provenance validator now require FULL under the changed-path helper because the full-only replay uses them. No full suite, GPU work, remote staging or launch ran.

## Evaluation and photo compatibility — 2026-10-29 follow-up

[Cell scorer](../../scripts/cell_eval.py) keeps the historical CAM schema for old models. For covariance, `class_score` is explicitly signed attribution, context only, global receptive field 256, stride 32. `aux` is a normal trained-head diagnostic, receptive field 48, stride 32; it is not marked context-only. The two-map difference is named `aux_minus_class_score`, never `aux_minus_cam`. Figures record separate map geometry. No legacy absolute auxiliary target mark is applied to the covariance candidate.

New saved cell output includes `manifest.json` and `artifact_identity.json`. The summary and sidecar bind the checkpoint SHA-256, source provenance and manifest hash to both row CSVs, both score arrays and the summary hash. Consumers verify these bytes and the actual selected checkpoint. Validation 5%-FA cell cutoffs and 1%/99% color limits are recomputed from that model's own saved validation scores; borrowed or test-fitted values refuse.

[Photo CLI](../../scripts/photo_heatmap.py) supports explicit `--maps head-only` or `--maps head-attribution`. Covariance CAM is refused. Head-only skips gradient attribution. Non-default runs must supply their own evaluation ID and cell summary; checkpoint identity must match deployment calibration and cell artifacts. Original run 13 defaults retain their exact pinned summary/checkpoint. Captions and JSON state per-map geometry and that the any-tile verdict is **not a calibrated photo false-alarm guarantee**. Photo figures are demonstrations, not new calibration evidence.

[Two-reference wrapper](../../scripts/compare_run21_cells.py) reads saved scores only. Its [reference pins](../../configs/run21-cell-comparison-v1.json) contain the run 21 BEST checkpoint and nine consumed evaluation artifacts for `cdc91a4b6d63a1517064`. The run 18 folder actually contains **seven files**: six non-PNG files plus `cells.png`; all seven original hashes are pinned. No metadata is retroactively added to run 18.

The wrapper uses strict detector row joins and the existing auxiliary-head comparison/target checks. It refuses quick/pilot, missing/stale identity, mismatched targets/rows, duplicate keys, validation/test overlap, or changed inputs. Candidate and reference inputs are hashed before and after processing. Detector reference `post_hoc_256` and candidate `train_matched` are accepted only with identical recipe ID/hash, full/selected list hashes, geometry and policy. Each numeric cutoff is independently checked against its own validation min-P_E fit; no cutoff is copied. Cell cutoffs likewise use own validation 5%-FA fits.

The output is only `comparison.json`, `bootstrap-replicates.npz` and a short `REPORT.md`. Shared paired source-photo draws: **2,000**, detector seed **20261019**, cell seed **20261021**, percentile 95% intervals. Detector reference is run 21; auxiliary-map reference is run 18. Existing `cov_pool` and `paired_r2` approval paths are not reused. The three guard checks are labeled **PROPOSED / Pending (user)**, with no `passed` verdict, +0.02 gain rule, new absolute AUC target, automatic adoption or pipeline change.

### Future command recipe (not permission to run)

Replace `CANDIDATE_RUN`, `CANDIDATE_EID`, and checkpoint path after a separately approved training/evaluation run. Outputs must be new folders. Full-data inference and the full saved-score bootstrap require their own resource approval; none ran in this implementation task.

```sh
# Own last checkpoint; source/manifest/score binding is written with cell output.
podman exec -w /work aai3001 python -B scripts/cell_eval.py CANDIDATE_LAST.ckpt --device cpu --workers 0 --output runs/verification/CANDIDATE_RUN-cells --figure

# Saved outputs only; detector run21 and heatmap run18 are pinned internally.
podman exec -w /work aai3001 python -B scripts/compare_run21_cells.py runs/CANDIDATE_RUN/evaluation/standard-256-v1/CANDIDATE_EID --candidate-cells runs/verification/CANDIDATE_RUN-cells --output runs/verification/CANDIDATE_RUN-comparison

# Selected model's own calibration and cell summary; use head-attribution for both maps.
podman exec -w /work aai3001 python -B scripts/photo_heatmap.py --run CANDIDATE_RUN --evaluation-id CANDIDATE_EID --cell-summary runs/verification/CANDIDATE_RUN-cells/summary.json --maps head-only --device cpu --batch-size 1 --output runs/verification/CANDIDATE_RUN-photos
```

Focused verification: **50 PASS, no skips, 8.897 s**, including 11 new workflow tests and read-only checks of the real reference pins and legacy run 13 inputs. Exact command and intermediate test results are in the journal. Model/math/notebook source bytes and v2 source hashes are unchanged. All real candidate metrics, full-data inference, full-suite evidence, staging and launch remain Pending.

## Bounded local correctness pilot — 2026-10-29

The user separately authorized this local GPU probe, not full training or a cloud launch. [Dedicated script](../../scripts/pilot_run21_cells.py) and [focused tests](../../tests/test_run21_cell_pilot.py) leave the pinned model/math/notebook bytes unchanged. Native run 21 BEST SHA and every copied base tensor match exactly; only the 257-parameter head is new. AdamW and scheduler start fresh.

Scope: `pilot=true`, `quick=true`, training-photo fraction **.005**, **16 images / 8 pairs**, four training workers, zero evaluation workers, FP32, ROCm unfold-wgrad workaround on, `prime_next_epoch=false`, `post_fit_eval=none`. The recipe still declares 15 epochs, LR3e-5, warmup1, original embedding schedule, .4/.2/.4 source mix and 180,000 draws. Pilot data caps this at 4,096 draws and a fixed 256-pair validation slice, after verifying the full 18,000-pair `60da4391043e8ce48ff2` recipe. No validation accuracy was measured. The first real batch has 48,621 changed pixels; tags match actual clean/stego differences.

Safety: own process group, shared test-runner kernel lock, available-RAM watchdog with 12 GiB floor, 600-second execution bound, and private temporary checkpoint/compiler-cache directories. Compilation is included in that bound and timed separately; one compiler thread is used. All temporary files and loggers are closed in cleanup. The settings record says `compile=off` for initial eager construction; the script explicitly compiles the actual `return_aux` GPU forward/backward later.

Evidence folders under `runs/verification/` each retain `evidence.json`, `pilot.log`, `exit-status` and external `cleanup.txt`:

| Attempt | Folder suffix | Result | Elapsed |
| --- | --- | --- | --- |
| Original | `run21-cell-pc-pilot-2026-10-29` | **Blocked / exit1**: final SRM comparison mixed saved CUDA and Lightning-teardown CPU tensors; substantive checks had completed | 40.737 s |
| Authorized identical-profile rerun | `run21-cell-pc-pilot-2026-10-29-r2` | **Complete / exit0**, including final frozen-SRM equality | 39.577 s |

The original failure remains intact. Only the final comparison was corrected to detached CPU copies, with a CPU teardown regression; no precision, batch, source or recipe fallback. Before GPU work, initial focused tests passed **4/4 in1.408 s**; after the fix, **5/5 in1.426 s** through the shared container runner. No further GPU probe ran.

Rerun measurements:

| Same real batch, 32 shared `training_step` updates | First8 mean | Last8 mean |
| --- | --- | --- |
| Tile CE | .153356623 | .120829844 |
| Changed-cell BCE | .596875519 | .598153323 |
| CE + .2 BCE | .272731729 | .240460513 |

Replay took **6.322 s**. Total loss falls; auxiliary BCE rises slightly. This does **not** establish auxiliary learning or accuracy. No extension to64 updates was needed or run. Gradients remain finite through conv1, body, covariance reduction, classifier and head; required weights have nonzero gradients and SRM stays frozen.

- Actual compiled forward/backward: **15.710 s** first-call window. Per-tensor error is `max(abs(compiled-eager)) / max(abs(eager))`, with exact equality required when the eager scale is zero; tolerance **1e-4**. Tile error **1.884e-7**, auxiliary error **3.227e-7**, largest gradient error **2.249e-6**. Reconstructed eager checkpoint outputs match eager outputs exactly; no `_orig_mod` keys.
- Fresh-batch timing: **8 updates / .749188 s =10.6782 steps/s**. This small PC-only window is not a full-source benchmark or a G4 forecast.
- Reconstructed head-only and dual-map outputs are finite, both `[2,8,8]`; head-only returns no attribution.
- Real Lightning continuation: save at epoch0/global step1, continue to step2, then rebuild through the shared runtime and restore with `ckpt_path` on the same cached next batch. Weights, Adam moments, scheduler and trainer epoch/global step match **bit exactly**. Returned trainer epoch1/global step2 agrees in both paths. Source provenance is retained; pilot checkpoint is evaluator-refused. This does not claim arbitrary mid-epoch data-order restoration; Lightning warns about the deliberately cached non-resumable loader.
- Peak GPU allocated **11,452,424,192 bytes**, reserved **15,378,415,616 bytes**. Minimum available RAM **21.961 GiB**, above12GiB throughout.
- External cleanup: no GPU/Python/worker/compiler processes, GPU VRAM back to **428MB used /15,876MB free**, RAM **27GiB available**, `/dev/shm` **92KiB**. No temporary directories or checkpoints remain. Original BEST and last SHA values and v2 source pins remain unchanged.

**Pending:** full-suite evidence on the intended launch revision, paid/cloud launch/resource decision, real training stability and all detector/heatmap quality/adoption decisions. The pilot supplies correctness evidence only.

## Local launch-preparation freeze — next authorized check

Implementation is committed locally on `yx` as `3d35a02`. This documentation commit freezes the intended clean revision for exactly one serial FULL CPU suite. Before the suite, run `python -B scripts/required_test_level.py`; then run `python -B scripts/test_cpu.py` with its default FULL selection and shared lock, both inside container `aai3001`. No FAST/focused check runs concurrently. Full results are **Pending at this documentation snapshot**.

Keep start/end Git status, commit/tree IDs, complete helper/suite logs, exit status and local FULL evidence key under ignored `runs/verification/run21-cell-launch-readiness-2026-10-29/`. Record actual results there first; do not edit this launch-tested tree immediately after the check. A failure must be reported before any implementation edit. This authorizes no GPU, remote/push, paid/cloud launch, adoption or superproject change.
