# Plan: three diagnostic checks (ideas board 1, 2, 2b)

Status: **approved to run** (user, 2026-10-19: "do the 3 tests first"). The decision rules below were **confirmed by the user on 2026-10-19, before any result**; do not change them after seeing results. Run order: 1, then 2, then 2b, **after the report work** (revisions 2 and 3; user, 2026-10-19). Board: [ideas board](ideas-board.md).

Purpose: get answers that can **strike experiments off the board** before any G4 spend.

## Shared rules

- Read-only on every run record, checkpoint and frozen list (SHA-256 before and after).
- Use the frozen pair lists and their recorded embedding settings: standard-256 `data/test/standard-256-v1/`, hard-256 `data/hard-test/256-v1/`, validation list `data/val/60da4391043e8ce48ff2/`. Rebuild clean and stego pixels through the existing pair code, never with a new random draw. Prove byte equality with the evaluator's pixels on a sample.
- Main model: **SRM-e50** (`runs/srm-e50-20261017T111310Z-0214`, evaluation `35afd1c7938bae78968c`). Compare on the **same pairs**.
- Any threshold is set on validation, never on a test.
- Report AUC with 95% intervals (bootstrap over photos, fixed seed). Small differences only count if the intervals do not overlap.
- Outputs go under `runs/verification/diagnostics-2026-10-19/<check>/` (git-ignored). Code changes are opt-in, tested, with the golden compare ALL EQUAL.
- CPU-only checks need no grant; GPU work needs an Orchestrator grant. PC safety rules apply.

## Check 1: classic detectors (WS + SPA)

**Result (2026-10-19, Worker_3, `ed52419`; outputs `runs/verification/diagnostics-2026-10-19/check1/`):** positive control passed (416 pairs: WS AUC 0.994, SPA 0.984; SRM-e50 0.997 on the same pairs). In 10 of 12 noisy / low+noisy ALASKA k=1 replacement cells the WS 95% interval lies wholly above SRM-e50 (SPA: 9). Examples, ALASKA seen, random placement: low 0.814 [0.774, 0.855] vs SRM-e50 0.608 [0.583, 0.637]; noisy 0.680 [0.646, 0.721] vs 0.563 [0.538, 0.589]; low+noisy 0.575 [0.559, 0.597] vs 0.510 [0.498, 0.525]. Contiguous is a tie on ALASKA low (0.769 vs 0.777) and SRM-e50 is better on LSDIR low contiguous (0.894 vs 0.753). Matching controls: WS 0.489-0.519, SPA 0.479-0.513 (no signal, as expected); SRM-e50 scores replacement and matching about equally. **Decision row applied: "raise #12; keep #9".** Caveat: validation 5%-FA thresholds do not transfer to noisy test tiles (actual test FA WS up to 38.7%, SRM-e50 up to 19.6%), so a better AUC is not yet a better operating point.

Question: does a known LSB-replacement statistic find signal on noisy ALASKA where SRM-e50 fails?

1. Detectors: Weighted Stego (WS) and Sample Pair Analysis (SPA), each on R, G and B separately, plus one combined score fixed before any result (mean of the three channel estimates).
2. **Positive control first:** standard panel (RAISE+DIV2K), k=1, **replacement**, requested bpp >= 0.35, from the hard `standard/k1/all` cell (or the standard-256 list). **Pass mark: WS combined AUC >= 0.90.** If it fails, stop: the implementation or the data handling is wrong.
3. Main cells: k=1 replacement in ALASKA seen and unseen and LSDIR, conditions `low`, `noisy`, `low_noisy`; split by placement (random / contiguous / stride).
4. Negative control: the same cells with +/-1 matching. Expect weak or no WS/SPA signal.
5. Operating point: false-alarm 5% threshold set on the replacement subset of the validation list; report miss at that threshold.
6. Compare with SRM-e50's saved scores on the same replacement pairs (AUC with intervals, ROC low-false-alarm region).

Cost: CPU, ~1-2 h.

## Check 2: highest-cell pooling

Question: does averaging the 8 x 8 cell evidence dilute packed (contiguous) payloads?

1. For SRM-e50, compute the per-cell logits that the tile score already averages (as `scripts/map_demo.py` does). Tile scores: **mean** (must reproduce the saved scores within GPU noise; this is the parity check), **max**, **top-4 mean**.
2. For each pooling: one threshold at min P_E on the validation list (as today), then the standard-256 and hard-256 tests.
3. Report contiguous / random / stride misses at k=1 low bpp per panel, noisy-clean false alarms, and overall P_E.

Cost: GPU inference, ~15 min per model; small code change. Optional second model: SRM-30.

## Check 2b: tanh saturation vs noise

Question: does SRM's first tanh flatten the weak signal on noisy tiles?

1. For SRM-e50 and one KV model (overlap-256, for contrast): per tile, the share of tanh1 inputs beyond +/-3, on the hard-256 tiles (clean and stego).
2. Relate it to the tile noise value (`tile_kv_std`) and, within each noisy cell, to misses (saturated-share quartiles vs miss rate).

Cost: GPU inference, ~15 min; extend `cvdl/saturation.py` to per-tile output.

## Decision rules (confirmed by the user, 2026-10-19)

| Check | Result | Effect on the board |
| --- | --- | --- |
| 1 | Positive control fails | Check invalid; fix and rerun. No board change |
| 1 | WS or SPA AUC on noisy ALASKA k=1 replacement clearly above SRM-e50 (intervals do not overlap) | Signal exists that the CNN misses: **raise #12** (classic statistics + CNN); keep #9 |
| 1 | WS/SPA at or below SRM-e50 | **Strike #12.** Report wording: both the CNN and classic replacement detectors fail on noisy ALASKA under these conditions (not "undetectable") |
| 2 | Top-4 or max cuts ALASKA/LSDIR contiguous low-bpp misses by >= 10 points, at equal or better validation P_E and no worse noisy-clean false alarms | Dilution confirmed: **raise #3** (cell-level training) |
| 2 | No such gain | **Lower #3** (not strike: the cells were trained for mean pooling; cell training could still change them) |
| 2b | Saturation rises with noise **and**, within noisy cells, the most saturated quarter is missed >= 10 points more than the least saturated | **Add an SRM tanh fix experiment** (handoff 2026-10-17 "option C") |
| 2b | Otherwise | **Strike the tanh fix** idea |

None of the checks changes #5 (seed repeats), #9 (curriculum) or #4 (untangle `srm-hard`).
