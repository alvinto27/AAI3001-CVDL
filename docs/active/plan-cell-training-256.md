# Plan: cell-level training at 256 (changed-cell targets, one run)

**Status: approved by the user 2026-10-21 (decisions A, B and C as recommended in section 12).** The pass marks and decision rules in section 7 are frozen before any training. This plan replaces the [older 512 plan](plan-per-cell-training.md). Written by the Worker (read-only plan at `0bcedbb`); copied here unchanged from section 1 on, apart from this header.

## 1. Goal and recommended decision

**Train an auxiliary cell head using the embedding change mask reduced to cell labels.** The mask is a training TARGET, never a model input. Show the current tile-head CAM (`cell_logits`) and the trained auxiliary head **on exactly the same report tiles**. Finish one controlled experiment, its evaluation, and the report picture; do not reopen a model search.

Recommendation: one **15-epoch continuation from the SRM-e50 starting checkpoint**, matched to the completed `srm-low-bpp-control` run. Keep its original payload distribution. Add only the target path, auxiliary head, and auxiliary loss. No region-random payload, curriculum change, inference pooling, or extra baseline G4 run.

A null result still completes the experiment if the implementation and measurements are sound. It does not authorize another model or another training run. Arm G and other model experiments remain ON HOLD.

## 2. Existing matched control — resolved

The completed control is **`srm-low-bpp-control-20261019T060708Z-9736`**. This is verified from its saved `run_settings.json`, `model_run_settings.json`, calibration, and current-suite summaries, not inferred from a notebook name.

| Item | Completed control | Proposed cell-target run |
|---|---|---|
| Backbone | `xunet_wide`, width 4, SRM, norm none, clip 3 | Identical |
| Starting weights | SRM-e50 `best-epoch43-val_pe0.0432-val_auc0.9917.ckpt` | Identical checkpoint, not the control's final weights |
| Starting checkpoint SHA-256 | `bf64848a49bdc6e20b7fcdba47cb8eee0ed247b909a928cc45e4a4102db9f076` | Identical |
| Optimizer/start | Weights only; new AdamW, weight decay 1e-4; no optimizer/scheduler/epoch restore | Identical; only aux head starts new |
| Budget | 15 epochs, 180,000 tile pairs/epoch, batch 128 images; 2,813 steps/epoch | Identical |
| LR | 3e-5, warm-up 1, no LR drops; grad clip 1 | Identical |
| Source mix | RAISE .4 / LSDIR .2 / ALASKA .4 | Identical, including declared source order |
| Payload | `configs/train-k1to4.yaml`, `embedding_schedule='original'`, stage epochs 5 | Identical; all 15 epochs use original embedding settings |
| Original payload details | k 1–4 equally; bpp .05–.5; lowest-k/k-plane, replace/match, random/contiguous/stride, existing payload kinds | Identical |
| Training geometry | 256, `overlap_last`; unpaired; original seed/run seed 0 and augmentation | Identical |
| Checkpoint validation | `train_matched`, recipe `60da4391043e8ce48ff2`, 18,000 pairs | Identical |
| Evaluated checkpoint | Explicit `last` after epoch 14 | Identical; do not switch to best validation epoch |
| Tile score | Ordinary two-class tile softmax | Identical |
| Intentional changes | None | Changed-cell targets; 1×1 cell head; CE + .2 × cell BCE |

Control standard-256 evaluation: **`af6e9879061fd441462f`**, 10,944 pairs, P_E **2.6498538012%**, AUC **.9964441934**. Its saved deployment threshold is **.4874875545501709**, origin `train_matched`, recipe `60da...`, mapping null. This proves the existing run already meets the requested 256 validation/calibration condition. The new run gets its **own** cutoff by the same rule on the same validation recipe; it does not copy this numeric cutoff.

### Other existing SRM runs

| Candidate | Match / reason not to use as the controlled baseline |
|---|---|
| SRM30 | Same SRM-wide family and original mix, but 30 epochs from scratch, LR 3e-4, capped 512 checkpoint validation. Current standard256 P_E **3.0656067251%** (eval `11a82bf812e573c7c62d`); benchmark only, not the old legacy/test-optimized 1.67% figure. |
| SRM-e50 | Same family/mix, 50 epochs from scratch, LR 3e-4 with drops at 8/33, capped 512 checkpoint validation. Its best model supplies the common starting weights. Best standard P_E **2.54%** remains the performance benchmark. |
| `srm-delayed` | Uses the same `60da...` train-matched 256 recipe and original source/payload mix, but model `xunet_wide_delayed` removes the first two pools, uses a scratch start and 30 epochs/LR 3e-4 with drops. Not aux-only matched. |
| `srm-hard` | Uses train-matched 256, but changes source quotas, payload distribution/recipe and training schedule. Not aux-only matched. |
| Low-bpp curriculum | Same 15-epoch start/LR/validation as the control, but changes payload mix/stages. Use the original control, not this run. |

These notebook recipes were resolved cheaply and saved in `cpu-shape-and-recipes.json`. Saved original control/e50 settings were checked separately. The control's primary result is P_E 2.65%, not the e50 best result 2.54%; do not silently substitute one for the other.

**Scope of inference:** this is one cell-target continuation versus an existing same-recipe continuation, not an across-seed study or an auxiliary-only comparison against e50's original 50-epoch training. The historical G4 software version/precision/compiler settings must be recorded and matched where available. Any unavoidable software difference is disclosed. No new baseline run is hidden in this plan.

## 3. Real 256 geometry and attachment point

`cvdl/models.py::XuNetWide` already exposes `features`, `cell_logits`, and a linear tile classifier.

```text
RGB [N,3,256,256]
   -> fixed SRM 5×5 -> learned feature stack (five 2×2 pools)
   -> feature grid F [N,256,8,8]
        ├─ global mean -> existing Linear(256,2) -> tile CE / unchanged tile score
        ├─ existing classifier as 1×1 projection -> CAM [N,2,8,8] (diagnostic)
        └─ NEW Conv2d(256,1,1) -> aux logits [N,1,8,8] -> cell BCE
                                               ↑
                                   changed-mask TARGET only
```

The new head has **257 parameters**. Compute the backbone once during training; do not run a second image pass or replace the tile classifier. Default inference must still return only `[N,2]` logits.

Each output position has stride **32 pixels**, so its nominal target cell is a **32×32** block. The actual receptive field is **48×48**, not 32×32. The SRM and first learned 5×5 convolutions plus the second 5×5 and five pools give receptive fields 5 → 9 → 10 → 18 → 20 → 24 → 32 → 48, ending at stride 32. For cell `(r,c)`, the nominal block is `[32r,32r+32) × [32c,32c+32)`; the receptive-field support is `[32r-8,32r+40) × [32c-8,32c+40)`, clipped at image edges.

The CPU probe confirmed features `[1,256,8,8]`, CAM `[1,2,8,8]`, aux example `[1,1,8,8]`. Gradient support of interior cell `(3,3)` was exactly rows/columns **88–135**, 2,304 pixels. Mean CAM logits equalled the ordinary tile logits in this probe. This verifies geometry, not learned localization or performance. Evidence: `cpu-shape-and-recipes.json`.

Attach the new head to this **pre-global-pooling feature map**, after the current feature stack. Do not attach after global pooling (spatial information is gone), change the stride, or copy the old 512/16×16 plan.

## 4. Labels through the actual OTF loader

### What the mask means

Use `embed_fast(..., return_changed_units=True)` at the existing embedding call. Its sorted indices describe **actually changed RGB units**. Do not use the slow/public selected-position mask: selected units can retain the same value, and multiple bit selections can refer to one RGB unit.

For a changed unit `u` in a 256-wide tile:

```text
pixel = u // 3
row = pixel // 256
col = pixel % 256
cell = (row // 32, col // 32)
```

Count **unique changed spatial pixels**, not channels, per nominal cell. Let `n[r,c]` be that count (0–1024); binary target `y[r,c] = (n[r,c] > 0)`. Clean samples have all-zero targets. Any exceptionally unchanged stego has an all-zero auxiliary target while retaining its original stego tile label; log the count rather than inventing positive cells.

### Sparse labels and receptive-field boundaries

- Keep every positive nominal cell; do **not** discard a whole-cell positive boundary ring. Low-bpp contiguous payloads may occupy only one cell row, so that rule can remove the entire signal.
- Positive reliability weight: `min(n/16, 1)`. This downweights a cell with one changed pixel without pretending it is as well supported as a densely changed cell.
- A zero-target cell whose **48×48 receptive field contains a changed pixel outside its nominal cell** is ambiguous: exclude it from the negative cell loss and interior cell metrics.
- A zero-target cell with no changes in its receptive field is a valid negative. All clean-image cells are valid negatives.
- Derive this small negative-validity mask before dropping the transient changed indices. No full image mask or changed-index array is retained in JSON metadata or the shuffle buffer.

The overlap rule is deliberately based on the measured 8-pixel halo, not an arbitrary one-cell ring. It is applied equally when comparing maps. Record how many positives/negatives remain. If a subset has no valid negatives, report it as ineligible for within-stego localization AUC, not as a perfect result.

### Transport with minimal disruption

Current control training uses `cvdl/curriculum.py::CurriculumImages(mode='original')`, not just base `TrainingImages`. Its three-item batch and count callback matter. Proposed path:

1. Extend Mode1 `make_sample` with an explicit opt-in target return; default `(image, metadata)` and default metadata remain unchanged. Training-grid crop/padding already happens **before embedding**, so indices are in 256 tile coordinates. Do not remap 512 parent-photo coordinates.
2. Generate image and targets at that one embedding call. Do not regenerate a second stego, change draw/embedding keys, or spend RNG draws choosing a new payload.
3. In the opt-in original-schedule dataset, carry counts/validity with each sample through its existing bounded shuffle. Keep their identity tied to the same pair/sample tag.
4. Apply **the exact same flips and rot90** to image, cell counts, and negative-validity mask. Refactor/reuse the current private keyed augmentation decisions; preserve the existing three random decisions and their order. Never independently draw target augmentation.
5. Preserve the outer `(images, labels, tags)` batch. In the enabled path only, extend the existing integer tag from `[sample_id, original_flag]` to `[sample_id, original_flag, 64 counts, 64 negative-validity bits]` (130 columns). Keep original flag in column 1, so `CurriculumCounts` remains correct. Compute floating weights at loss time.
6. This avoids a fourth tuple component: `MixedLoader._join` currently handles three components, and `_joined_batches` counts `group[-1]`. The extended tag remains `[B,130]`, so its length is still B. Verify this with real packed replay tests, not shape reasoning alone.
7. `PackedCollate`, `pack_blocks=16`, mixed-source joins, short tails, worker IDs and next-epoch priming must transport tags intact. Default paths retain their existing three components and original tag shapes. Validation/test loaders retain their existing ordinary indices and do not use training sidecars.

The tag costs about 1 KiB/image as int64; small count/validity arrays are sufficient in the pre-collation buffer. Never retain per-sample 256×256 arrays there. Reject unsupported geometry/paired combinations explicitly for this narrowly scoped opt-in instead of partially supporting them. This experiment remains unpaired, 256, original schedule; no new paired cell-training support is required.

## 5. Payload choice — keep the existing distribution

**Do not add region-random now.** The user goal is mask-derived target supervision, not a new placement generator. Changing placement frequency or region size would break the unusually good existing control.

Existing low-bpp k=1 contiguous payloads already provide localized examples: at .05–.15 bpp they use about 3,277–9,831 RGB units, roughly **4.3–12.8 full-width pixel rows** before alignment effects. They typically leave most of an 8×8 cell grid clean. Existing random/stride payloads are not assumed to be localized: measure occupancy from actual targets. Random low-bpp can touch almost all 64 cells, and a stride may span much of the tile. Such samples can still supply cell positives, but are not evidence that the head learned location.

Keep all original examples and apply the auxiliary loss to them, including clean negatives. Report auxiliary losses/occupancy separately for contiguous, random and stride. Do not retune a placement mixture or drop scattered examples after seeing results.

Why cautious: historical **256** check 2 found inference pooling hurt overall and worsened rather than improved target misses. On its 1,008 target pairs, mean miss was **46.53%**, max **59.62%**, and top4 **62.20%**; validation P_E rose by **5.80 / 4.39 percentage points** for max/top4. Its predeclared ALASKA-seen/unseen/LSDIR k=1-low-contiguous target remains a sensible diagnostic slice, but its old pooling marks are not the new success rule. Scattered low-bpp has very little spatial-label contrast, and this proposal does not promise to solve noisy payloads or the general k=1 limit.

## 6. Loss, warm start, checkpoints and compatibility

Use the original tile cross-entropy plus **0.2 × auxiliary BCE-with-logits**. No deployment-score blend and no learned mixing coefficient.

Per image, compute positive BCE as `sum(min(n/16,1) × BCE) / max(1,number_of_positive_cells)` and negative BCE as the mean over valid negative cells. If both classes exist, average these two terms; if only one exists, use that term. Then average over images. A missing class contributes no artificial zero half-loss. This avoids many negative cells overwhelming a thin payload while preserving sparse-positive downweighting; normalizing positive weights by their own sum would cancel that downweighting in sparse-only cases. No additional positive multiplier or loss search.

Log tile CE, cell loss, positive/negative cell counts and occupancy; total loss alone cannot diagnose whether the auxiliary task learned anything. Shared features and tile classifier continue training as in the control; SRM remains fixed. Validation uses the existing tile task and full frozen matched recipe, not this auxiliary loss.

Implementation recommendations:

- Default feature **off**, with no head allocation, new RNG consumption, state keys, altered forward return, or target computation. Add only explicitly enabled experiment settings/hparams (proposed `cell_supervision='changed_cells_v1'`, `cell_loss_weight=.2`; default supervision off).
- Keep the head inside `net` and have `model_from_hparams` reconstruct it only from an enabled checkpoint's explicit hparams. Ordinary `forward` still returns tile logits; an explicit training/diagnostic option may return the aux map too. This must remain compatible with the current compiled-net entry point; bypassing the compiled `forward` through a separate `features()` training call is not a verified solution.
- Initialize the opt-in head without advancing the original training RNG stream (scoped CPU RNG preservation). Reuse the control's seed and all existing schedule/augmentation keys. Verify enabled-target pixels/order against disabled-target pixels/order directly.
- **Warm-start blocker to implement:** current `curriculum.weights_only` requires exactly equal `net` state keys. Extend it narrowly for an explicitly enabled cell head: require every original backbone/classifier key and shape, allow only the declared new head keys to be absent from the original e50 checkpoint, reject every other difference. Keep strict old behavior for feature off. Save original-weight identity equal to the control's `initial_weights_sha256 = 54dedf6ac7ae10149d11e308073fcaca1198ca24b674af23c773eb96a614fe6c`; record new head identity separately.
- Resume restores the entire enabled run, including head/optimizer/scheduler/RNG/count state; it does not warm-start again. Extend the existing start identity checks to cover target version and loss weight. No permissive `strict=False` loading of unexplained keys.
- **Evaluation blocker to test:** `standardtest.evaluate` reconstructs `model_from_hparams` and strictly loads *all* net state keys. Therefore the enabled model factory must reconstruct the saved head even though the ordinary tile scorer does not execute it. Keep strict loading. Do not discard arbitrary unknown keys to make evaluation pass.
- Explicit `evaluation_checkpoint='last'` is intentional and matches this control; shared default `best` and its constants guard stay unchanged. `post_fit_eval='none'` remains: current standard256/hard256 evaluation is a separate PC step, never retired 512 post-fit orchestration.

The input-constants file should need **no change**: original training YAML, seeds, main/hard suites and validation recipe are unchanged. New opt-in settings are not a reason to recapture anything. If an existing guard fails, diagnose the unexpected original-input change; do not update expected constants simply to pass. Golden capture/compare and legacy512 remain retired.

## 7. Predeclared evaluation and honest result labels

Freeze selection manifests, rules and thresholds before G4 training. The numeric rules below are **recommendations for user approval**, not already approved marks or measured results. The completed deliverable is an implemented, verified one-run experiment plus a report, even if the scientific marks fail.

### Tile evaluation — frozen current suites

Evaluate the new **last** checkpoint once with the active standard256 + hard256 PC evaluator. The new run calibrates its own deployment threshold on the complete 18,000-pair `60da...` recipe by the existing min-P_E/tie policy, then freezes it. Do not optimize on test or copy the control's cutoff. Baseline comparison uses the existing saved control predictions/evaluation, not new baseline inference.

Report:

- Standard256 P_E/FA/miss at each model's own frozen deployment cutoff and standard AUC. Controlled guard: new P_E no more than **+0.25 percentage points** versus control (about **2.90%** total; calculate from the unrounded 2.6498538012%). Compare e50 **2.54%** separately.
- Predeclared target: pooled hard256 **k=1, `low`, `contiguous`**, ALASKA seen + ALASKA unseen + LSDIR, both embedding methods/bit modes/payload kinds. Record exact matched pair/photo counts from saved rows; do not pool all low placements and call it localization.
- Additional detector-benefit mark: target AUC at least **+.02** versus control, with the **paired photo-bootstrap 95% difference interval above zero**. Report per-panel values; no model-interval-nonoverlap rule.
- Random and stride versions of that target are context, not alternative targets if contiguous fails. Report hard noisy/low-noisy false alarms at the deployment threshold; a gain with a noisy-clean regression is a trade-off, not an unqualified improvement. Use +2 percentage points in pooled distinct noisy-clean FA versus control as a proposed detector-benefit guard.
- Use 2,000 photo-cluster bootstrap replicates, seed 20261021, the same photo resamples for both models; retain all selected rows belonging to a sampled photo. Thresholds stay frozen during test bootstrap. Report intervals as test-sampling uncertainty, not training-seed uncertainty.

### Small cell diagnostic, not a new benchmark suite

Freeze a small manifest from **existing** validation/test pair rows, before training; do not alter their source recipes or frozen test identities.

Recommended cap: validation 32 k=1-low-contiguous and 32 k=1-low-random pairs per training source (RAISE/ALASKA/LSDIR: at most **192 pairs**); test the same counts per RAISE standard / ALASKA seen / ALASKA unseen / LSDIR (**256 pairs**). Select by a declared seeded hash order of stable pair IDs, with distinct source photos within each source/split selection. Use existing actual bpp/bin fields and original embedding keys. Record all method/bit/payload counts. If a selection cannot meet its cap, stop and disclose before G4 rather than silently drawing new embeddings or changing the condition. Val/test photo separation follows and verifies the existing frozen splits.

A dedicated small replay helper can request actual changed units from the same generator/key and must verify its generated pixels equal the ordinary frozen pair loader. Target generation is for labels only. The model sees RGB alone.

On the **same tiles**, score: (a) candidate tile-head CAM margin `cell_logits[:,1]-cell_logits[:,0]`; (b) candidate trained auxiliary logit; optionally (c) the existing control's CAM as before-training context. Use raw scores for AUC so softmax saturation does not erase rank. Do not restore inference max/top4 pooling or change the ordinary tile softmax score.

Cell measurements:

1. Within-stego localized cell AUC (positives and valid negatives in the same contiguous stego), reported per source/panel and pooled with per-photo uncertainty. Never claim localization from an AUC formed only by clean-image cells versus all stego-image cells.
2. Valid-negative cell FA and positive-cell recall at a **cell-only** cutoff chosen on validation for at most 5% valid-negative FA (include clean partners and valid unchanged stego cells; use an explicit deterministic tie rule). Freeze separately for each map type. This cutoff is diagnostic, not the deployment tile cutoff.
3. Occupancy/sparsity, eligible cells/photos, and scattered-random results. Nearly all-positive random maps are explicitly not a spatial-localization test.

Proposed evidence that the auxiliary target was learned: held-out localized auxiliary AUC **≥.75**, its photo-bootstrap lower bound **>.5**, and test valid-negative FA **≤7%** at that validation cell cutoff. Report auxiliary-minus-CAM AUC on the same cells with a paired interval; a better auxiliary map is an observation, not assumed in advance.

Result labels:

| Evidence | Conclusion |
|---|---|
| Alignment/tests valid; one run/evaluation/figure complete | Required experiment deliverable complete, including a negative result |
| Cell marks pass + standard guard | Cell-target learning demonstrated without material standard regression |
| Above + target hard AUC benefit and noisy-FA guard | Evidence this continuation also helps tile detection |
| Cell marks pass, no tile gain | Learned localization; no demonstrated detector improvement |
| Cell marks fail, or a tile guard fails | Negative result / trade-off; state it plainly; no automatic follow-up run |

Do **not** replace SRM-e50 as the standard best merely because the looser controlled guard passes. A new default would need to meet the existing 2.54% benchmark or explicit user acceptance of a trade-off.

## 8. Required report figure — CAM versus trained head

Use fixed tile IDs chosen before training, not the most attractive outputs afterwards. Recommend three rows: a low-bpp contiguous RAISE tile, an ALASKA contiguous tile, and a scattered-random counterexample, with clean partners available. Prefer reuse of already frozen report examples where their IDs/settings match; otherwise save the new fixed IDs in the diagnostic manifest.

Required columns, same RGB tile and same 8×8 registration:

1. Stego tile (and clean thumbnail/partner).
2. Actual changed-pixel map, amplified for visibility, with the 32-pixel grid.
3. Ground-truth per-cell target / sparse weight and ignored-negative indication.
4. **Current tile-head CAM** from the candidate's `cell_logits`, using its stego-minus-clean margin.
5. **Trained auxiliary cell-head map** from the same candidate feature map.

Optional sixth column: matched control CAM on the same tile. It is useful but droppable if producing it threatens the time budget; the candidate CAM-versus-trained-head comparison is **not droppable**.

Use nearest-neighbor enlargement, not smoothing that suggests pixel accuracy. Keep color scales fixed across tiles for each map type, based only on validation; label separate CAM-margin and auxiliary-logit scales. No per-image auto-normalization hiding failure. Save tile scores, labels, bpp/k/method/placement, pair/photo IDs, target counts and checkpoint identity alongside the PNG.

Caption: the CAM projection receives tile-label supervision, whereas the auxiliary projection receives direct changed-cell targets; both candidate maps share a trunk affected by the new loss. They are not independent models, causal saliency, attention, or probabilities that an individual pixel changed. The feature sees 48×48 despite its 32-pixel cell spacing. Do not claim that comparing these two maps isolates head versus shared-feature learning.

Add a new cell-comparison figure beside the existing report data figure; do not overwrite its frozen input asset. Brief report text must state original control, unchanged tile score, localization result, standard/hard result and limitations.

## 9. Archive reuse decision

**Do not restore archived pooling/check2 for this experiment.** `cvdl/archive/cell_pooling.py` is inference pooling, not target supervision. `scripts/archive/check2_pooling.py` is historical large-suite/checkpoint orchestration with saved legacy attachments; it is not a small changed-target evaluator. Read its old result as context only.

The current model already exposes the required CAM, and the actual embedding engine already exposes changed units. Implement a small active target reducer/diagnostic path with its own tests. No active import from archive, compatibility forwarding, retired 512 scoring or golden enforcement.

If implementation review later proves an archived function necessary, **stop and propose restoring that implementation with its tests**. Do not copy the implementation into active code while leaving its tests archived. That is a scope decision, not permission granted by this plan.

## 10. Minimal implementation → tests → pilot → ONE G4 → report

1. **Approval:** confirm matched-control continuation, original payload mix, target/weight/loss and scientific marks. Record exact baseline checkpoint/evaluation/recipe IDs and freeze diagnostics/figure selection. Orchestrator updates the active plan/status/journal; this scratch plan makes no tracked doc change.
2. **Implement one bounded feature:** opt-in Mode1 target return; original-schedule target transport/augmentation; head/model-factory/compiled-forward wiring; narrow weights-only initialization; settings/checkpoint identity; small diagnostic/report helper. New notebook/job is a copy of control settings with a new name and explicit cell flags; retain `last`, original schedule, recipe and full budget.
3. **Focused CPU tests, serial and bounded:**
   - Changed units really differ between clean/stego; selected-but-unchanged is excluded. Verify RGB-channel dedup, image/cell edges, 256 crop/padding, clean/empty/sparse/dense targets and measured 8-pixel halo handling.
   - Every flip/rotation keeps image and counts/validity registered. Target on/off yields identical generated images, metadata, keys and sample/source order. No stored large mask.
   - Three-item batches / tag columns survive worker sharding, mixed-source joins, default and packed collate, short tails, bounded shuffle, priming and packed replay. Feature off retains existing tags and consumes no extra RNG.
   - New head dimensions/finite BCE/missing-class reductions/gradients; ordinary tile output unchanged before shared-weight updates; target never enters the network input. Freeze status of SRM unchanged.
   - Exact base-weight warm start with head-only missing keys; fail wrong source hash, wrong architecture or other missing/extra keys. Enabled checkpoint strict reconstruction and CPU inference; enabled resume preserves new head/optimizer/count/identity. Disabled checkpoint compatibility and existing live model-copy/SRM/resume tests remain intact.
4. **FULL CPU suite before launch**, using the supported runner, serially, hidden GPU visibility, `-B`, no bytecode and thread limits. The current clean base has 360 FULL methods / 331 FAST +29 skips; feature tests may raise these totals. Do not freeze that count as an expected result. Constants are input guards, not output recapture.
5. **One short PC pilot after approval**, in the ROCm container, using the existing `pilot` safety mode and the real validation recipe. Keep declared identities verified before pilot slicing. Small batch if needed on the PC; 3 short epochs with original schedule and matching stage-epoch override exercises boundaries and resume, not a new model experiment. Check loss components, actual consumed counts, target occupancy, gradient flow, head checkpoint reload and evaluation-interface reconstruction. Pilot artifacts remain ineligible for deployment/report scoring. Time warmed steps/epochs without a GPU synchronization per training step; record memory and loader stalls. Delete only disposable pilot artifacts after evidence is retained.
6. **Review evidence; FULL again after any code change.** Check G4 train pixels/source quotas/seeds/YAML/start SHA against the control; check explicit last/60da/post-fit-none. Never reduce G4 batch/epochs or alter mixes silently after a slow pilot; stop for review if production equivalence or budget cannot be met.
7. **One G4 boot/run only**, 15 full epochs. No G4 evaluation job or second baseline. Normal sync/delivery and safe shutdown remain. Proposed timeout 90 min / max-run 105 min, subject to the existing queue's approval and explicit reconciliation of the previously extended 300-minute GCE setting. Do not rely on a stale long auto-stop. A timed-out incomplete run is a blocker, not permission for a second experiment.
8. **PC evaluation/report:** existing full standard256+hard256 evaluator once for the new last checkpoint; existing control/e50 saved predictions for comparison; small paired cell diagnostic; required CAM-versus-aux picture and brief decision table. Record real timing/results in the journal/status. No retired post-fit512 evaluation, golden work, Arm G or model search.

**Before a G4 launch: always FULL tests.**

## 11. Cost and droppable work

Measured SRM-e50 notebook wall was **153.1 min / 50 epochs** (~3.06 min/epoch); scaling gives **45.93 min** for 15 epochs. The more relevant existing matched-control notebook wall (2,831 sec, recorded in the run queue) is **47.18 min**; the journal rounds that to 47.2 min. This includes its real train-matched validation cadence. Do not price a 50-epoch new model when a matched 15-epoch control already exists.

Budget **50–65 VM minutes** including modest head/target overhead and startup/delivery, pending pilot evidence. At the active handoff's historical **$1.63/hour**, that is approximately **$1.36–$1.77**; the proposed 105-minute stop is about **$2.85** maximum runtime at that rate. This is a planning estimate, not a current cloud quote. Target reduction/loader overhead and compile behavior remain unmeasured for the new feature. No new package is planned.

Keep: alignment/packing/resume tests, FULL suite, safe pilot, one production run, current-suite evaluation, small cell diagnostic, candidate CAM-versus-trained-head figure. Drop first if time tight: control-CAM picture column, extra stride cell panels, animation, broad heatmap gallery, top-k localization metrics, additional bpp/region sweeps. Never drop safety or silently turn one controlled run into a search.

## 12. Short user decision list

- **A — Recommended:** one 15-epoch cell-supervised continuation matched to `srm-low-bpp-control`, same e50 initialization and original payload mix; no extra control run or region-random.
- **B — Recommended target/loss:** actual changed pixels → 32-pixel nominal cells; sparse weight `min(n/16,1)`, ambiguous negative halo excluded; class-balanced per-image BCE, auxiliary weight **.2**; tile score unchanged.
- **C — Approve or amend before launch:** proposed cell marks (.75 AUC / 7% cell FA), standard guard +.25 points versus control, optional detector-benefit rule (+.02 target AUC with positive paired interval and noisy-FA guard), and 50–65-minute expected / 105-minute maximum VM budget.

The auxiliary-target goal and required same-tile CAM-versus-trained-head picture are already confirmed by the user. These choices do not reopen whether the mask should be used as a training target.

## Evidence and source inventory

- `cpu-shape-and-recipes.json`: actual random-input feature/CAM/head shape and receptive-field probe; resolved six notebook recipes; saved control/e50 settings and model records.
- `baseline-records.json`: saved-run comparison and the control's current-suite calibration/summary; cost arithmetic.
- `source-anchors.txt`: current definitions for model/head, Mode1 embedding, augmentation, joins, curriculum and strict warm start.
- `cleanup.txt`: clean tracked worktree, container process/bytecode/resource check after inspection.
- Baseline artifacts: `runs/srm-low-bpp-control-20261019T060708Z-9736/checkpoints/srm-low-bpp-control-20261019T060708Z-9736/{run_settings,model_run_settings}.json`; its standard256 eval `af6e9879061fd441462f/{main_summary.csv,validation/calibration.json}`.
- Active recipes: `notebooks/experiments/{srm,srm-e50,srm-delayed,srm-hard,srm-low-bpp-control,srm-low-bpp-curriculum}.ipynb`; `configs/train-k1to4.yaml`.
- Existing findings: `docs/active/plan-low-bpp-curriculum.md` (completed control/result); `docs/active/plan-diagnostic-checks.md`; `docs/journal/2026-10-19/README.md`; `runs/verification/diagnostics-2026-10-19/check2/decision-evidence.json` and timing/input audit records; `docs/active/run-queue.md` (measured run timings/limits); `docs/active/handoff-2026-10-19.md` (historical spot rate).
- Main implementation touchpoints: `cvdl/models.py`, `cvdl/lit.py`, `cvdl/curriculum.py`, `cvdl/runtime.py`, `cvdl/settings.py`, `cvdl/standardtest.py`, `cvdl/pilot.py`, `LSB-Dataset-Generator/lsb_mode1.py`, `LSB-Dataset-Generator/lsb_torch.py`.

Pending: user choices B/C and launch approval; implementation; new pilot overhead; all new training/evaluation results. No result is predicted by the successful geometry probe.
