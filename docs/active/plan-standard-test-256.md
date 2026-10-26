# Plan: standard test at 256 (work list #2)

Status: **built** in `bd36403` (2026-10-18, Worker_2): suite `standard-256-v1`, recipe `5045c1076ab3f540a61c`, 8 frozen lists in `data/test/standard-256-v1/`. Default of `scripts/evaluate_run.py`. Pending: timed GPU pilot, hard test at 256 (#3), validation-threshold columns (#4). Design by Worker_2 (2026-10-18); the user decisions below are final.

Source decisions: [2026-10-17 handoff](../archive/handoff-2026-10-17.md) section 2 (standard test, grid, sources, k 5-8 opt-in), and the [2026-10-18 journal](../journal/2026-10-18/README.md) (one threshold from validation).

## Problem

Today's standard test and source panels use 512 tiles and k 1-8 (`configs/raise512-run1.yaml:1-18`). ALASKA and LSDIR panels use the source-record trim grid (`LSB-Dataset-Generator/lsb_torch.py:200-205`). `FixedPairs` takes every tile of the split (`lsb_torch.py:436-444`), and no full frozen test list with a content hash exists. Training uses 256 overlap-last tiles and k 1-4. The published P_E is the test-panel minimum (`cvdl/metrics.py:84-96`), mixed with fixed-0.5 FA and miss.

## User decisions

| # | Decision |
| ---: | --- |
| 1 | 256 x 256 tiles, **overlap-last grid for all sources**, k 1-4, bpp U(0.05, 0.5), stego types equal. bpp-1 panel dropped. |
| 2 | **Same frozen test for every model.** Never derived from a training recipe or notebook. |
| 3 | **New suite is the default** of `scripts/evaluate_run.py`: no flag needed. ~~The old suite stays only behind an explicit `--test-suite legacy-512`.~~ Retired 2026-10-21 ([legacy-512 retirement](legacy-512-retirement.md)). |
| 4 | **Pair counts are multiples of 96** (4 k x 24 stego-type combinations), exactly balanced within each source and k. |
| 5 | **Photo-balanced tile selection** within each source (one unused tile per photo per round), so large photos do not dominate. |
| 6 | **k 5-8 check is opt-in** (`--higher-k`): the same tiles and slot settings with k mapped 1->5, 2->6, 3->7, 4->8; a separate report, not in the main score. |
| 7 | **One deployment threshold** (min P_E on the run's validation) is the main result (task #4). Test-panel minimum P_E is a diagnostic only and must be named as such. |

## Panels

Test-split pools at 256 overlap-last (measured by Worker_2 from the manifests; no decoding):

| Panel | Photos | Unique tiles | Tiles per photo | Pairs (new) | Pairs (old 512) |
| --- | ---: | ---: | --- | ---: | ---: |
| Standard: RAISE + DIV2K | 149 + 100 | 36,280 + 4,808 | 96-260 / 32-64 | **10,944** (RAISE 9,696 + DIV2K 1,248) | 10,956 |
| ALASKA seen (28 cameras) | 1,700 | 15,313 | 4-16 | **2,688** | 2,648 |
| ALASKA unseen (8 cameras) | 2,000 | 18,070 | 4-16 | **3,168** | 3,153 |
| LSDIR | 2,500 | 50,586 | 9-130 | **5,280** | 5,245 |
| Total | | | | **22,080** | 22,002 |

Per panel: k 1-4 each one quarter. Within each source and k, the 24 combinations (bit mode 2 x method 2 x placement 3 x payload 2) are exactly equal (RAISE 101 and DIV2K 13 per k x combination cell). The RAISE/DIV2K share follows today's standard panel. Stride 2-16, start random, one variant, no augmentation. Effective bpp and clipping are recorded; clipped pairs are never redrawn or dropped.

## Design

- **Files:** tracked `configs/test-256-v1.yaml` (test-only semantic settings) and `data/test/standard-256-v1/{recipe.json, summary.json, pairs-standard.csv, pairs-alaska-seen.csv, pairs-alaska-unseen.csv, pairs-lsdir.csv}`, plus `pairs-k5to8-<panel>.csv` for the opt-in check.
- **Builder:** an explicit offline builder with `--check` (rebuild must be byte-identical; refuse to overwrite). The runtime only reads and verifies; it refuses missing or changed lists and never builds.
- **Identity:** canonical recipe JSON SHA-256 (schema, sampler and geometry version; seed 1 and domain `standard-test-256-v1`; balance rules; semantic config; quotas; manifest and split hashes of every source). Per-CSV SHA-256 and the recipe SHA in `summary.json`. No paths, training settings, model or hardware in the hash.
- **Rows:** pair id, cover id, panel, source, photo, camera, split, photo index, tile size, grid, row/col/y/x, slot, k, bit mode, method, placement, stride, payload, requested bpp, payload bytes, effective bpp, clipped, exact draw kwargs, settings and embedding seed keys. Round-trip floats.
- **Selection:** independent keyed streams for photo/tile order, slot shuffle and draws. No dependence on run seed, model, epoch or workers. Record camera counts and realised per-photo quotas.
- **Geometry:** explicit 256 overlap-last replay adapter, patterned on `MatchedValidationPairs` (`cvdl/validation.py`). Do not loosen the legacy `RunSettings` 256 restriction (`lsb_torch.py:62-63`) and do not reuse `HardPairs` unchanged (it hardcodes 512).
- **Old models:** scored on the same 256 tiles with their own weights (the model pools spatially before the classifier, `cvdl/models.py:131-134`). Report the train/test geometry for each model.
- **Evaluator:** `evaluate_run.py runs/<run>` runs the new suite by default; `--higher-k` adds the k 5-8 panels; ~~`--test-suite legacy-512` runs today's suite unchanged.~~ (retired 2026-10-21, [legacy-512 retirement](legacy-512-retirement.md)) The new branch is separate from the legacy `post_fit.run`. Output goes to `runs/<run>/evaluation/standard-256-v1/<evaluation-id>/` (settings with checkpoint SHA and list hashes, per-panel predictions with full rows, summaries, timings; higher-k in its own subfolder). Refuse to overwrite without `--overwrite`. Never write training records.
- **Hard test:** until task #3 rebuilds it at 256, the default suite reports the hard test as pending. It never mixes the 256 standard test with the 512 hard test silently. ~~`--test-suite legacy-512` keeps hard v1.~~ (retired 2026-10-21)
- **Metric names:** no bare `P_E` in new outputs. Diagnostics: `test_min_P_E` (and its cut-off), `FA_at_0_5`, `miss_at_0_5`, `P_E_at_0_5`, AUC, counts, per source and per k. Main columns (`threshold_origin=validation_min_pe`, cut-off, FA, miss, `P_E_at_validation_cutoff`) come with task #4; leave a clear place for them.

## Out of scope

- Hard test at 256 (#3), the validation threshold code (#4), report notebook (#5), rescoring (#6).
- Any G4 run. Any change to `notebooks/experiments/srm-hard.ipynb` (Run_Supervisor owns it).

## Verification

- Exact counts and quotas; k and the 24 combinations exactly equal within each source and k; test-split photos only, no train/val overlap; unique coordinates; every eligible photo represented; coordinates match overlap-last axes with no padding.
- Builder `--check` twice byte-identical; lists independent of seed, model, epoch, workers and hardware; changed inputs or lists are rejected.
- Higher-k maps 1:1 to the same tiles and slots with k + 4; main results unchanged whether it is on or off.
- No overwrite; training-record SHA unchanged; the three metric families are distinct on synthetic data.
- Full `scripts/test_cpu.py` passes; `refactor_golden.py compare` ALL EQUAL without recapture; old capped validation hash unchanged; `--test-suite legacy-512` output unchanged.
- CPU replay of a few pairs at 0 and 2 workers byte-identical; a CPU 256 forward pass with a 512-trained checkpoint.
- Only after GPU approval: one timed full evaluation of one model (per-panel and total wall time), then process and memory cleanup checks.
