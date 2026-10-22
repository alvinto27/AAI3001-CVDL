# Paired training (ideas #16): R1 SRM-e50 paired, R2 cell model paired

**Status: approved by the user 2026-10-22 with these changes to Worker_2's plan below** (they win where the text differs):

1. **No runner change for R2's start checkpoint.** After R1, a one-line notebook commit sets R2's `init_checkpoint` path and SHA-256 (as for run 13). Both possible start checkpoints are on the VM disk; no Pi staging. ~~Allowlisted init override through submit/preflight/job_run and Pi staging~~.
2. **Evaluation tools are built while R1 trains**, not before. The decision rules are frozen now (below).
3. **Augmented PNG pairing is refused** (legacy `PairDataset`), not fixed.
4. **R1 rule, second route added.** R1 passes if EITHER (a) standard ΔP_E <= -.002 with the 95% upper bound < 0 AND pooled hard k=1 low ΔAUC lower bound >= -.01, OR (b) pooled hard k=1 low ΔAUC >= +.01 with the 95% lower bound > 0 AND standard ΔP_E upper bound <= +.0025. Otherwise R2 starts from SRM-e50 best (`bf64848a…f076`).
5. R2 rule as in section 4 (head-vs-head mean per-eligible-stego-tile AUC +.02, lower bound > 0; standard guard +.0025; hard guard -.01; cell false alarm <= 7%).
6. The 50-epoch from-scratch cell model is "possible later" only.

Limits: R1 `--timeout-min 200 --max-run-minutes 240`, R2 `--timeout-min 90 --max-run-minutes 120`; GCE max run 300 min. The user is told before each launch.

---

**Read-only investigation; no code changes, training, GPU commands or launch.** Main R2 follows the revised assignment: 15-epoch warm-start continuation, not 50 epochs from scratch. Proposed rules below require approval before any results.

## 1. Fix and ownership

`LSB-Dataset-Generator` is an ordinary tracked directory **in this AAI3001-CVDL repository**, not a submodule/separate checkout (`git submodule status` empty; its `git rev-parse --show-toplevel` is this repository; files have mode100644, not160000). Commit its changes here on `yx`, with explicit paths.

`lsb_torch.image_tensor` seeds a private generator with `entropy + [AUG,label]`. In the **paired branch only**, use augmentation label/key **1 for both partners**: the stego keeps its historical orientation; the clean follows it. Leave the function's default seed construction, its three draws (flip/flip/rotation), and every unpaired call unchanged. Two independently constructed private generators with the same key give the same operations without touching global RNG state. Do not change embedding keys or k/bpp sampling.

For cells, carry clean+stego as one shuffle item, with targets attached. Apply the same key to both images and to the stego changed-count/valid-negative 8x8 maps through the existing `cell_targets` transform. Clean counts are zero and clean validity all true. Proposed paired tag layout: `[pairs,2,130]`; IDs stay even(clean)/odd(stego). Flatten images **all clean then all stego**, and flatten tags in that identical order before CE +0.2×cell BCE. Keep the BCE's existing per-image reduction and sparse-positive weighting; no extra factor2.

## 2. Equal training budget; compatibility audit

Keep `batch_size=128` **images**, hence64 pairs per optimizer step. `lsb_notebook.loaders` already halves the training loader size for pairs. `TrainingImages.__len__` returns `count` pairs instead of `count*2` images: keep `tiles_per_epoch=180000` tile/embedding draws, thus360000 images and **2813 steps/epoch** in both modes. MixedLoader joins worker tails globally; assert actual totals including the final32 pairs/64 images. Do not halve the draw budget or double the configured batch.

Keep source shares RAISE.4/LSDIR.2/ALASKA.4, selected photos, seed0, original `train-k1to4.yaml`, variants, 256px `overlap_last`, buffers/prefetch/packing/workers unchanged. Source quotas and tile selections occur before pairing; embedding draws remain one per stego tile. Shuffle capacity is measured in images: default1024 becomes512 intact pairs; block grouping similarly halves item counts, retaining the same image lead. Pair order and the number of shuffle RNG draws necessarily change; **unpaired output/draws must not**. Each paired CE averages128 outputs, with exactly balanced labels, not64 losses.

| Path | Current behavior / required work |
|---|---|
| Ordinary OTF pairing | Works except independent augmentation. Source mix and packing support it. |
| `settings.py` embedding schedule | Explicitly rejects paired mode. Permit **original** paired continuation; retain low-bpp curriculum refusal (not this task). Keep15=3×5 epochs and initialization checks. |
| `CurriculumImages` | Cell constructor rejects pairing; `_items_mixed` always emits individual images even without cells, so merely removing guards is unsafe. Implement intact paired original items and aligned tags; preserve all unpaired branches. `_output` must support the paired tag layout. |
| `CurriculumCounts` | Assumes `batch[1]` is labels; paired input is the stego image tensor. Adapt to count one stego per pair from its original/staged tag, not pixels or both partners. |
| `SteganalysisModule` | Rejects paired cells; cell training_step assumes unpaired images/labels/tags. Add the paired flattening above; ordinary pair_logits_targets already concatenates2B correctly. |
| Validation / compile / ROCm | `60da` validation is already paired, unaugmented and independent of the training flag; preserve its frozen rows/keys. In-place net compile and PC conv1 workaround are not pairing-restricted: net still receives128 images (PC16). Verify compiled forward/backward and state reload; do not bypass workaround. |
| PC pilot | `pilot=true` supports pairing but requires OTF + train_matched + post_fit_eval=none. R1's PC-only pilot therefore needs a recorded60da validation override; do not copy that override into R1 cloud recipe. G4 preflight intentionally refuses pilots. |
| Other paths | PNG PairDataset calls augment_image twice independently: tie transforms in its paired branch too, or explicitly refuse augmented PNG pairing until fixed; leave ImageDataset unchanged. Archived Mode2 rejects pairing (retain). G4 speed_test/step_profile and gbras_bench hard-code/refuse paired recipes: not launch paths for these runs; do not use their numbers as paired evidence. |

## 3. Notebooks and initialization handoff

| | R1: `srm-e50-paired.ipynb` | R2: `srm-cell-aux-paired.ipynb` |
|---|---|---|
| Base | Exact `srm-e50.ipynb` + paired flag/new name | Exact run13 cell recipe + paired flag/new name + selected initialization |
| Training | Scratch,50epochs, LR3e-4, warmup1, drops(8,.3)/(33,.1) | Weights-only continuation,15epochs, LR3e-5, warmup1, no drops, original schedule, cell0.2 |
| Checkpoint validation | Historical **capped** RAISE5000/LSDIR1500 policy; not60da | train_matched **60da4391043e8ce48ff2**, full18000pairs |
| Evaluate/use | **Best** by existing validation-P_E selection, like SRM-e50 | **Last**, like run13; new optimizer/scheduler/head, no optimizer/epoch resume |

Both retain FP32, NCHW/default layout, G4 default compile, source recipe, batch128,180000draws and post_fit_eval=none. Run standard-256 and hard evaluation separately with each model's own frozen deployment-validation threshold; checkpoint-selection validation is not the deployment calibration set.

Freeze R2 source selection now: if R1 passes §4, initialize from **that selected R1 best**; otherwise use SRM-e50's historical best (`bf64848a49bdc6e20b7fcdba47cb8eee0ed247b909a928cc45e4a4102db9f076`). Never choose best/last after inspecting test scores. Existing `init_checkpoint` + required `init_checkpoint_sha256` and `weights_only` already support a paired source and leave only the new257-weight head initialized.

**Missing launch capability:** local run_notebook accepts native `--set init_checkpoint=...`/SHA, but canonical G4 submission currently allows only notebook+job ID metadata and passes no native overrides. Add a narrowly allowlisted initialization path+SHA handoff through submit → pinned preflight → job validation → run_notebook; no arbitrary experiment/shell metadata. Record effective overrides and refuse missing/mismatched files before training. Notebook may retain the known fallback defaults; R1 selection must use the explicit override, not a hard-coded unknown run path. R1's checkpoint remains on the VM persistent disk under `runs-out/<R1>/checkpoints/<R1>/`; pass its actual relative path and verified delivery SHA. Stage the fallback tonight in a content-addressed initializer location on the trusted Pi; support a bounded authenticated startup fetch on R2 boot if absent locally, verify SHA then atomically install. No checkpoint in Git, no extra transfer-only GPU boot, no assumption that PC `runs/` paths exist on the VM.

R2 vs run13 changes pairing during both body training and fine-tuning **and which trained body initializes it**; it is a pipeline comparison, not an isolated fine-tune effect. Fallback isolates paired fine-tuning more closely. One seed and conditional source selection remain limitations. A50-epoch from-scratch cell model is **possible later**.

## 4. Proposed decision rules — freeze before R1

Use identical frozen score rows, exact pair/label/photo joins,2000 paired **photo-cluster** bootstrap draws; retain repeated clean rows and share draws between models. Tile seed20261019; cell seed20261021. Intervals quantify test-photo uncertainty, not training-seed uncertainty. Let Δ=candidate−reference; P_E is a fraction (.002=0.20percentage points).

- **R1 vs `srm-e50-20261017T111310Z-0214` (standard P_E2.54%):** adopt/pass only if standard ΔP_E≤−.002 and its95% upper bound<0, AND pooled hard k=1 low ΔAUC95% lower bound≥−.01. Hard scope: ALASKA seen/unseen + LSDIR,0.05–0.15bpp, both methods/all three placements. Publish per-panel/noisy breakdowns as context, not alternative winning routes. Otherwise use fallback for R2.
- **R2 vs run13 last (`srm-cell-aux-20261021T054704Z-6997`):** primary is contiguous TEST pooled **mean per-eligible-stego-tile AUC**, aux head vs aux head, not pooled-cell AUC/CAM. Require point Δ≥+.02 and95% lower bound>0; standard ΔP_E95% upper bound≤+.0025 (existing0.25-point guard), and pooled hard low ΔAUC95% lower bound≥−.01. Also retain cell false-alarm≤7% at each model's validation-frozen5% cell cutoff. Otherwise no automatic replacement of run13.

Reuse architecture_decision joins/shared-photo AUC bootstrap and cell_eval frozen manifest/scoring/localization. **Neither existing CLI is a drop-in new-run decision:** architecture_decision hard-codes arm-A runs/output and lacks a P_E interval; parameterize safely and add paired fixed-threshold P_E draws while preserving old behavior. cell_eval's normal comparison is aux−CAM, not two heads: compare the two saved aux score arrays on identical eligible rows with shared photo draws. No test-fitted thresholds or rule changes after results.

## 5. Time, safety and smallest PC check

Unmeasured forecasts, not speed claims: SRM-e50153.1min leaves only11.9min under165. Same images/steps suggest **R1~153–180min** (allow0–15% loader/composition overhead plus small startup margin), not half the time. Run13 took47.5min including the existing cell overhead; **R2~48–60min**, allow~70 with staging. Propose R1 `--timeout-min 200 --max-run-minutes 240`; R2 `--timeout-min 90 --max-run-minutes 120`. Current actual GCE300min is sufficient for either separate boot; metadata flags do not set scheduling. Verify actual300/STOP before launch; any reduction to240/120 needs approval while TERMINATED. Tell the user before each launch. No resume/watchdog promise; stop on OOM/nonfinite, no batch/precision fallback.

PC batch16images=8pairs, one GPU job at a time, ≤12 workers. First CPU tests: asymmetric image/mask over many keys; paired difference equals the transformed raw LSB difference; recompute reduced cell targets from transformed masks; unpaired before/after tensors and RNG draws exactly equal across uint8/float32, packed/unpacked and worker settings. Then minimal GPU correctness pilot: real pilot=true data (verify full60da first), capture one8-pair batch and replay32updates per model through shared Lightning; extend only to64 if loss trend is pending. Require finite values, nonzero body/head gradients, exact orientation/target alignment and lower last8 vs first8 mean loss; record CE/BCE separately. Warm-start CE previously rose in both control and cell pilots, so a non-falling short run is **Pending/investigate**, not invented success. Disable priming for capped GPU probes; test counts/priming/tails on two **fully consumed tiny CPU fixture epochs** (never prime a partially consumed real epoch), plus a tiny GPU compile/reload check. This is not a speed or accuracy comparison; remove disposable artifacts/workers afterwards.

## 6. Ordered tasks

1. Approve/freeze this revised recipe, source-selection rule, numeric gates and caps.
2. Implement paired-only augmentation + intact original/cell items, guards, flattening and counts; CPU regression/parity tests; commit explicit source paths here.
3. Add the two named notebooks and narrow initialization handoff/staging; test preflight, SHA failures, effective settings, weights-only/head initialization and selection policies.
4. Add parameterized saved-score paired P_E/AUC and head-vs-head evidence; freeze baseline files, row identities and hashes tonight.
5. Run the minimal sequential PC pilots; investigate any pending gate; run required test-level helper and FULL before cloud launch. Verify exact G4 profiles without launching tonight.
6. Tomorrow: notify user → R1 → delivery/SHA + standard/hard evaluation → frozen verdict → select/hash initializer → notify user → R2 → delivery + standard/hard/cell comparison. Record real timing/results and leave ambiguous outcomes as such.
