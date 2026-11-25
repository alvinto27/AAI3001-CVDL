# 50-epoch low-bpp curriculum from scratch (ideas #17)

**Status: approved by the user 2026-10-22: schedule A (50% original mix kept from epoch 8; stages 1-7 / 8-20 / 21-32 / 33-50, bpp .05-.35 / .05-.25 / .05-.15, k=1 weight 1 / 2 / 3) and the frozen rule in section 3.** R1's verdict picks the variant (paired vs R1, else unpaired vs SRM-e50); only one is run. Fixed 50 epochs; 80 epochs and early stopping shelved (ideas #18). Written by Worker_2; copied unchanged below apart from this header.

---

**2026-10-22; read-only investigation. No code, Python, training or GPU commands run.** Prepare two notebooks before tomorrow's R1 verdict; run **one**, paired if R1 passes either frozen route, otherwise unpaired. Both start randomly: **R1 selects batch composition, NOT initialization weights.** Schedule/rules require approval before implementation/results.

## 1. Recommended schedule and evidence

Keep SRM-e50's50epochs, AdamW3e-4, warmup1 and LR drops `((8,.3),(33,.1))`. The table uses **one-based** epoch numbers. Important: `v4_lr_factor` drops at `(epoch-1)*steps_per_epoch`, so API drops8/33 mean zero-based training epochs7/32; keep this conversion explicit. All bpp intervals start at.05. Original means the complete unchanged `train-k1to4.yaml` distribution (equal k1–4, bpp.05–.5, original methods/placements/payload types).

| Epochs | Original share of stego draws | Remaining share: bpp range; k weights(1,2,3,4) | Reason |
|---|---:|---|---|
|1–7|100%|None; normal recipe|Learn features from scratch before narrowing the task. Normal mix, not high-payload-only training.|
|8–20|50%|.05–.35; (1,1,1,1)|First narrowing at the first LR drop.|
|21–32|50%|.05–.25; (2,1,1,1)|Increase low-payload/k1 exposure before the final LR drop.|
|33–50|50%|.05–.15; (3,1,1,1)|18 epochs of hard-target refinement at the lower LR.|

**Recommend this conservative50% retention (A).** Exact per-epoch quota:180000 original draws initially; subsequently90000 original +90000 staged, before worker sharding/shuffling, not per-batch. Final overall k1 probability is37.5%, not50% (the original half also contains k1). Retaining more original data may protect standard performance; that is a hypothesis, not a guarantee. **Alternative B:** same boundaries/weights,25% original from one-based epoch8, closer to run10's stronger final43.75% k1 share. It risks repeating the standard trade-off; do not prepare/run both schedules or choose between them after test results. Freeze A or B first; both pairing variants use that one schedule.

**What run10 supports:** the15-epoch warm-start curriculum gained hard low-k1 AUC +.018 [+.013,+.023] vs SRM-e50 and +.020 [+.015,+.024] vs its plain-continuation control. Standard P_E rose2.54→3.10%, +.56points [+.28,+.84], while standard AUC stayed.9964. Thus extra plain training did not explain the hard gain, and the operating-point trade-off matters. It does **not** establish that50epochs, scratch training,50% retention or pairing will work, nor isolate ordering from changed mixture exposure.

**Verified literature:** Boroumand, Chen & Fridrich, *Deep Residual Network for Steganalysis of Digital Images*, TIFS2019, §III.B, pp5–6 ([authors' PDF](https://ws2.binghamton.edu/fridrich/Research/SRNet.pdf), fetched/read for this plan). They seed other payload detectors from0.4bpp/bpnzac, or progressively seed0.3 from0.4,0.2 from0.3, then0.1 from0.2. Both approaches give similar spatial-domain results; progressive training is slightly better in JPEG. This supports easier-to-harder initialization as a precedent, **not our exact50-epoch mixed-RGB-LSB schedule or replay fraction**. Their late-training validation-snapshot policy also differs from the all-epoch best requested here. Do not present this proposal as an SRNet replication.

## 2. Isolated implementation, no legacy drift

Add an explicit **versioned opt-in**, e.g. `embedding_schedule='low_bpp50_v1'`, with the fixed table above. Keep legacy `low_bpp`/`original`, their three-stage resolver, quarter-assignment RNG domain/draws, initialization requirements and defaults untouched. The new resolver accepts zero-based epochs0–49 with boundaries0/7/20/32/50, returns stage/original fraction, and drives sampling and `CurriculumCounts`; no fake `stage_epochs=50/3`.

Settings need a **separate new-mode guard**:50epochs, original SRM-e50 recipe/LR, mixed OTF256 overlap_last, cell supervision off, **no init checkpoint/SHA**, either batch mode. Do not require warm start or3×5 epochs for this mode, and do not silently reinterpret the legacy stage-length field. Record table/version/hash and the effective quota in data/run/checkpoint identity; update runtime's schedule record so it does not falsely say three5-epoch stages/25% replay. Preserve historical metadata when the new mode is off. Add a new-mode scratch identity/check path: today's start helpers are weight-init-gated and cannot be called with a null checkpoint. Local resume must restore stage/optimizer/LR and reject a changed table; no G4 resume feature is proposed.

Use the existing occurrence-position assignment before worker sharding: seeded half-quota for the new mode only, all-true flags for its first7epochs. Keep source/tile selections, embedding seed construction and original-config sample generation unchanged. `d28e2bb` already renders intact paired items/tags, key1 augmentation and one-stego-per-pair counts: allow **the new mode** through its paired guards and reuse that path. Keep legacy15-epoch low-bpp pairing refused unless separately requested. Unpaired output branches must remain identical for every existing mode/notebook.

## 3. Comparisons, selection and proposed frozen rule

Prepare `srm-e50-curriculum50.ipynb` and `srm-e50-curriculum50-paired.ipynb`: exact SRM-e50 recipe plus the new schedule/name, and paired flag only in the second. Keep seed0, width4/SRM/norm-none, FP32/NCHW/default compile,128images/step,180000tile draws=360000images/2813steps each epoch; paired128images=64pairs. Keep RAISE.4/LSDIR.2/ALASKA.4, packing, buffer image capacity, augmentation policy,50epochs and no post-fit inference.

Compare paired candidate to **R1 best**, unpaired candidate to **SRM-e50 best** (`srm-e50-20261017T111310Z-0214`). Only the training mixture/schedule differs within each comparison. No weights are loaded from either baseline. Freeze the conditional variant choice now; do not run both and select the better test result.

Keep historical **capped validation** (RAISE5000/LSDIR1500 policy), select the **best validation-P_E checkpoint over all50epochs**, as SRM-e50; no60da substitution. Fixed validation contains no test leakage, but its normal mix can favor earlier/general-purpose snapshots over late low-bpp specialists. This is a selection preference, not proof that late training failed. Record selected epoch/stage; never switch to last or choose epochs on hard-test scores. If the selected snapshot is in one-based epochs1–7, report **no selected curriculum exposure** and do not credit the curriculum for any gain. Last-checkpoint results, if reported, are explicitly secondary.

**Proposed pass rule:** pooled hard k1 low ΔAUC≥+.01 **and** paired95% lower bound>0; standard ΔP_E95% upper bound≤+.0025 (0.25percentage points). Also require selected checkpoint one-based epoch≥8 (saved zero-based epoch≥7) to claim curriculum benefit. Δ=candidate−its matched baseline. Hard scope: ALASKA seen/unseen +LSDIR,0.05–0.15bpp, both methods/all placements. Use2000 shared photo-cluster bootstrap draws, seed20261019, exact paired-row/metadata joins and retained repeated clean rows. Standard P_E uses each model's own frozen **deployment** validation cutoff, distinct from capped checkpoint-selection validation. Publish per-panel/noisy FA/AUC context; no alternate winning route. Test-photo intervals do not measure seed uncertainty, and this compares complete training policies, not ordering alone.

## 4. Time, pilot and ordered work

**Fixed 50 epochs.** The 80-epoch and early-stopping options are shelved until evidence shows more epochs are needed.

Forecast **~155–180min**, not measured: same network/images/steps as153.1min SRM-e50, plus schedule/flags/counting overhead. Run10/11 took47.3/47.2min, but both used the curriculum wrapper and a continuation, so that is not a plain-e50 overhead bound. Propose timeout200/max240 for either variant; actual GCE300min/STOP is enough. Metadata does not change scheduling. No timing boot or PC-to-G4 speed extrapolation; notify the user before the selected launch.

Before launch: CPU tests all50stage outputs/boundaries, exact quotas and consumed counts, source/draw budgets, first7epochs matching normal-recipe images/labels/order/RNG, paired/raw-difference alignment, workers/packing/tails, resume stage identity, and historical run10/11 tensor/RNG/record parity. Small sequential PC checks for **both variants**, batch16images (8pairs when paired), actual ROCm conv1 workaround: CPU-sample every stage, then pass stage-sampled batches through shared Lightning/default compile; check finite loss/gradients, fresh optimizer/random initialization and strict reload. A small fixed-batch scratch replay can check loss falls; do not compare loss numerically across changing distributions. Use pilot markers/recorded60da override only for PC, priming off for capped probes, no accuracy/speed claim. No GPU run is part of this plan.

**Order:** approve schedule/rule → isolated code + legacy parity tests → both notebooks (only pairing/name differ) and effective-profile checks → minimal pilots + required-level/FULL tests → pinned G4 preflight for both, before R1 verdict → verdict selects one prebuilt notebook → notify user/launch → deliver/check SHA → standard/hard evaluation and frozen matched-baseline decision. Evaluation analysis should reuse the paired evidence tools prepared for R1, not revive run10's model-interval non-overlap rule.
