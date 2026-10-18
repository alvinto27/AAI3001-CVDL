# Plan: hard test at 256 (work list #3)

Status: **built** in `a56c275` (2026-10-18, Worker_2): recipe `d6d91a5a98dae856d1ae`, `data/hard-test/256-v1/`. Runs by default in the standard-256 suite. Noise edges (validation Q3): RAISE 1.691, DIV2K 2.442, ALASKA 2.613, LSDIR 2.516. Noisy test pools: RAISE+DIV2K 10,507, ALASKA seen 3,475, unseen 3,278, LSDIR 12,980; every cell uses 1,008 unique tiles (no reuse). Pending: validation-threshold columns (#4). Design by Worker_2 (2026-10-18).

Source decision: [2026-10-17 handoff](handoff-2026-10-17.md) section 2 ("Hard test") and section 6 (v1 details). Hard test v1 (`data/hard-test/v1/`, `cvdl/hardtest.py`) stays unchanged for `--test-suite legacy-512`.

## Kept from v1

- 4 panels (RAISE+DIV2K, ALASKA seen, ALASKA unseen, LSDIR) x k {1, 2} x {all bpp 0.05-0.5, low bpp 0.05-0.15, noisy, low+noisy} = **32 cells x 1,008 = 32,256 pairs**.
- 24 stego-type combinations x 42 per cell; keyed balanced tile assignment.
- Miss counts all stego pairs; FA counts **distinct** clean tiles within each cell.
- Noise measure: clean-tile KV noise (fixed definition as `scripts/check_b_tiles.py:139-142`, `cvdl/data.py:94-100`): KV residual clipped to [-3, 3], population std over RGB interior pixels, 2-pixel border excluded. Clip, filter, dtype and version are fixed in the recipe; never read from a model.
- Noisy = score above the source's validation Q3 (linear 75th percentile).
- Test-split photos only.

## Changed (user decisions, 2026-10-18)

| # | Decision |
| ---: | --- |
| 1 | 256 tiles, **overlap-last grid for all sources**. |
| 2 | **Noise edges are model-independent:** KV noise over **all** val-split 256 overlap-last tiles per source, incl. DIV2K (noise edge only; no DIV2K threshold). Values, edges and input hashes are frozen. |
| 3 | **Candidates are all test-split 256 tiles**, not only the standard-256 list. Hard tiles **may overlap** standard-256 tiles; overlaps are reported. |
| 4 | **CPU noise scan approved:** 8 workers (ceiling 12), one torch thread each, photos streamed, RAM under ~20 GB; start with a small timing pilot. |

Pools (measured, no decoding): validation RAISE 150 photos / 36,484 tiles, DIV2K 120 / 5,720, ALASKA 1,000 / 8,966, LSDIR 1,500 / 30,584 (2,770 photos, 81,754 tiles). Test: RAISE+DIV2K 249 / 41,088, ALASKA seen 1,700 / 15,313, unseen 2,000 / 18,070, LSDIR 2,500 / 50,586 (6,449 photos, 125,057 tiles). Estimate 10-30 min CPU in total; to be measured.

## Pool rule for noisy cells

| Noisy test tiles in a panel | Rule |
| --- | --- |
| >= 1,008 | Unique tiles in each cell |
| 504-1,007 (ALASKA only, as v1) | Balanced reuse, at most 2 uses per tile in a cell |
| < 504, or any shortage outside ALASKA | Build fails and reports. Never lower Q3 or the cell size. |

Reuse across different cells is allowed (as v1).

## Design

- **Files:** `data/hard-test/256-v1/{recipe.json, summary.json, noise-val.csv, noise-test.csv, pairs-<panel>.csv}`; domain `hard-test-256-v1`.
- **Identity:** canonical recipe SHA-256 over manifest and split hashes, geometry, KV definition and dtype, quantile method and edges, noise-file SHAs, seed, cells, types, bpp, stride and the reuse rule. Each list SHA in `summary.json`.
- **Two steps:** offline noise extraction (decode each photo once; save noise per tile coordinate), separate from a light builder whose `--check` rebuilds the lists from the frozen noise files byte-identically. Optional noise-recompute check. No runtime writes; no overwrite.
- **Replay:** the explicit 256 replay adapter, not the 512 `HardPairs`.
- **Evaluator:** the default suite runs hard-256 (replaces "pending"); `--no-hard` turns it off. Hard enabled and its recipe SHA are part of the evaluation id and provenance. Output `<evaluation>/hard-test/{hard_predictions.csv, hard_cells.csv, hard_summary.json}`. Do not inherit v1's per-source FA5 cut-offs. Until task #4: diagnostics (`*_at_0_5`, AUC, `test_min_P_E`) and clearly pending main columns; task #4 applies the same validation cut-off to every standard and hard cell.
- **Legacy:** `--test-suite legacy-512` still runs hard v1 unchanged.

## Out of scope

Threshold code (#4), report notebook (#5), rescoring (#6), any G4 run, any GPU use without approval.

## Verification

- 32 cells x 1,008; 24 x 42 per cell; k and bpp ranges per cell; noisy cells only above the frozen edge; test-split photos only; reuse within the rule; overlaps with standard-256 reported.
- Noise extraction: pilot timing, then full scan; recompute check on a sample matches the frozen values.
- Builder `--check` twice byte-identical; tampered inputs rejected.
- CPU replay of a few pairs per panel at 0 and 2 workers byte-identical.
- Full `unittest discover` passes; `refactor_golden.py compare` ALL EQUAL (GPU visible, on approval); hard v1 and `legacy-512` unchanged.
