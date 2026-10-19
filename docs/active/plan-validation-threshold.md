# Plan: one deployment threshold from validation (work list #4)

Status: **built** in `68e4f0b` (2026-10-18, Worker_2): `cvdl/deployment.py`, `scripts/build_calibration.py`, mapping `configs/evaluation-calibration-v1.json` (10 checkpoints), wide-R2 list `data/val/8328cc6b0673cade0312/` (RAISE 15,239 / DIV2K 2,761 from its 45,674 / 8,276 training pairs per epoch). No threshold values yet: they come from rescoring (#6). Design by Worker_2 (2026-10-18).

Source decisions: [2026-10-18 journal](../journal/2026-10-18/README.md) (one threshold; per-source thresholds reference only; no DIV2K reference) and the [2026-10-17 handoff](../archive/handoff-2026-10-17.md) section 2 row "Thresholds" (superseded text).

## Rule

- **One threshold per evaluated checkpoint:** min P_E on validation, pooled over all validation pairs (not a mean of per-source P_E). FA = clean with score >= t / N clean; miss = stego with score < t / N stego; P_E = (FA + miss) / 2.
- Reuse `cvdl/metrics.py` `threshold(frame, 'pe')` (line 140) and `rates` (line 171). Candidates: every distinct validation score plus `nextafter(max, +inf)`. Ties within 1e-12: smaller FA wins; then the first candidate in ascending order. Keep scores and cut-off in float64; no rounding or clamping. Reject non-finite scores and bad labels.
- The threshold is fixed **before** any test result is computed and is applied unchanged to every standard-256 panel, source, k and source x k row, every hard-256 cell, and the opt-in k 5-8 panels. Never refit on a test.

## Where validation comes from (user decisions, 2026-10-18)

| Runs | Validation for the threshold | Label |
| --- | --- | --- |
| New runs with a saved `validation_recipe` (e.g. `srm-hard`, recipe `bf7ede374cab01f73f5d`) | The run's own frozen train-matched list | `train_matched` |
| Old runs 2-10 (below) | **Shared frozen list `60da4391043e8ce48ff2`** (18,000 pairs, 256 overlap-last, `configs/train-k1to4.yaml`, mix RAISE/ALASKA/LSDIR 40/40/20) | `post_hoc_256` (not the run's original checkpoint-selection validation) |
| Old run 1 (wide-R2, RAISE+DIV2K) | A new frozen RAISE+DIV2K list: 18,000 pairs, 256 overlap-last, `configs/train-k1to4.yaml`, RAISE/DIV2K shares derived from its saved training records (not invented) | `post_hoc_256` |

Option C (post-hoc calibration at 256 overlap-last for all old checkpoints) was chosen over A (transfer a 512 threshold) and B (rebuild per-geometry validation; fails the unique-tile rule for ALASKA trim).

Old runs to rescore (user: only these 10; the 7 earlier models are out of scope):

| # | Run | Trained |
| ---: | --- | --- |
| 1 | `xunet_v4_..._wide-w4_r2-20261015T121753Z-29499` (wide-R2) | 512, RAISE+DIV2K |
| 2 | `xunet_v4_..._wide-w4_mixr2-20261016T024859Z-9032` (newdata-1) | 512, 40/20/40 |
| 3-5 | `tiling-{overlap,trim,pad}` (512) | 512 |
| 6-8 | `tiling-{overlap,trim,pad}-256` | 256 |
| 9 | `srm-20261017T052857Z-4521` | 256 overlap |
| 10 | `srm-e50-20261017T111310Z-0214` | 256 overlap |

Roster source: `runs/verification/post-training-model-comparison-20261017.csv` rows 2-10 plus SRM-e50.

**Mapping:** a tracked frozen mapping (checkpoint SHA-256 -> validation recipe id and label) lets `evaluate_run.py runs/<run>` resolve the list with no flag. A run with a saved recipe uses it first. An unmapped old run **fails with a clear message**; no hidden fallback. Checkpoints and original run records are never changed.

## Outputs

- `evaluate_run.py runs/<run>`: no flag. It scores the validation list once with the selected checkpoint, then the tests.
- `<evaluation>/validation/{predictions.csv, calibration.json, source_reference_thresholds.csv}`. `calibration.json` pins checkpoint SHA, recipe id and label, list SHA, score-CSV SHA, rule version, exact cut-off, validation FA/miss/P_E/AUC/counts. Calibration inputs and policy are part of the evaluation id.
- Main columns (standard `main_summary.csv` and hard `hard_cells.csv`): `threshold_origin`, `cutoff`, `FA`, `miss`, `P_E_at_validation_cutoff`, AUC, counts; hard cells keep the distinct-clean FA and all-stego miss denominators, plus Wilson 95% bounds.
- **Reference:** `source_reference_thresholds.csv` and a separate reference summary labelled `if_source_known_reference`: one min-P_E threshold per source present in the validation list, **except DIV2K**. ALASKA's reference is shared by seen and unseen. No mixed RAISE+DIV2K reference.
- **Diagnostics** keep their names: `test_min_P_E`, `test_min_cutoff`, `*_at_0_5`. Never promoted to main.
- QUICK uses a pinned subset with its own identity, marked non-deployment; never cached as the full calibration.
- `--test-suite legacy-512` unchanged.
- The report notebook (#5) reads only these saved files.

## Out of scope

Report notebook (#5), the rescoring runs themselves (#6), the 7 earlier models, any G4 run. GPU use only on approval.

## Verification

- Analytical tests: cut-off, tie rule, saturated scores, `nextafter`, float64 round-trip; pooled vs source-averaged P_E differ on synthetic data.
- Every main, hard and higher-k row carries the same threshold id and value; changing test scores cannot change the calibration.
- Validation and test pairs are disjoint; frozen list, mapping and input tampering are rejected; unmapped old run fails.
- Checkpoints and original run records unchanged (SHA before and after).
- Reference thresholds per source, no DIV2K; QUICK identity separate.
- New wide-R2 list: builder `--check` byte-identical twice; shares match its training records.
- Full `unittest discover` passes; `refactor_golden.py compare` ALL EQUAL (GPU visible, on approval); legacy-512 unchanged.
- Timed inference pilot only with separate approval.
