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
   Definition fixed before any result (Orchestrator, 2026-10-19): rank cells by the stego-minus-clean logit margin, take the top cell (max) or top 4 cells, average their paired logits, then softmax. Mean pooling is unchanged.
   Decision-rule scope fixed before any result (Orchestrator, 2026-10-19):
   - "Cuts contiguous low-bpp misses by >= 10 points" = the **pooled** miss over the three target panels (ALASKA seen, ALASKA unseen, LSDIR; hard `low` cells, k=1, contiguous, replacement and matching together) falls by >= 10 points vs mean pooling, **and** no single target panel gets worse.
   - "No worse noisy-clean FA" = the **pooled** clean FA over the hard `noisy` and `low_noisy` cells of the three target panels (k=1 and k=2, distinct clean tiles) is not higher than mean pooling's. Report every cell; flag any single cell that rises by more than 5 points.
   - "Validation P_E equal or better" = the pooling's validation P_E at its own validation threshold is <= mean pooling's.
   **Amendment (user, 2026-10-19, after the first pass and before any decision):** float32 softmax saturated (validation scores exactly 1.0: mean 26%, max 70%, top-4 58%), so the min-P_E threshold could not separate tied scores. All three poolings are re-scored on the float64 stego-minus-clean **logit margin** (same order as the softmax, no saturation ties). Rules unchanged. The first-pass numbers stay recorded as superseded.
2. For each pooling: one threshold at min P_E on the validation list (as today), then the standard-256 and hard-256 tests.
3. Report contiguous / random / stride misses at k=1 low bpp per panel, noisy-clean false alarms, and overall P_E.

Cost: GPU inference, ~15 min per model; small code change. Optional second model: SRM-30.

**Check 2 result (2026-10-19, Worker_3, `9563540`; outputs `runs/verification/diagnostics-2026-10-19/check2/`):** float64 logit-margin scores (amended). Mean pooling parity with the evaluator: max abs 3.5e-7. Mean / max / top-4: validation P_E 10.29 / 16.09 / 14.68%; target contiguous k=1 low misses 46.5 / 59.6 / 62.2% (max and top-4 worse in all three target panels); pooled noisy-clean FA 22.5 / 0.8 / 0.4%. Threshold-free AUC: target contiguous .839 / .771 / .781; target random .682 / .638 / .660; standard .996 / .973 / .979; ALASKA seen .903 / .810 / .821 (mean best everywhere, intervals separate). **Decision row applied: "No such gain" -> lower #3 (not strike).** Reading: mean pooling is not what holds back packed payloads; single untrained cell scores are unreliable on noisy clean cells. Superseded float32 first pass kept in `check2/superseded-float32/`.

## Check 2b: tanh saturation vs noise

Question: does SRM's first tanh flatten the weak signal on noisy tiles?

1. For SRM-e50 and one KV model (overlap-256, for contrast): per tile, the share of tanh1 inputs beyond +/-3, on the hard-256 tiles (clean and stego).
2. Relate it to the tile noise value (`tile_kv_std`) and, within each noisy cell, to misses (saturated-share quartiles vs miss rate).

Cost: GPU inference, ~15 min; extend `cvdl/saturation.py` to per-tile output.

Decision scope fixed before any result (Orchestrator, 2026-10-19). The decision uses SRM-e50 only; overlap-256 is contrast.
- Saturated share = per tile, the fraction of first-tanh inputs with |x| > 3 (the existing `cvdl/saturation.py` definition).
- "Rises with noise" = over all distinct hard-256 clean tiles of the three target panels (ALASKA seen, ALASKA unseen, LSDIR), Spearman correlation of saturated share with `tile_kv_std` is positive with a photo-bootstrap 95% interval above 0, **and** the noisiest quarter of these tiles (by `tile_kv_std`) has a higher mean saturated share than the least noisy quarter.
- "Most saturated quarter missed >= 10 points more" = within each hard `noisy` and `low_noisy` cell of the three target panels (k=1 and k=2, stego tiles), split stego tiles into quarters by their own saturated share; pool over cells; miss at SRM-e50's frozen validation cut-off (saved `prob_stego`, cut-off 0.45057889819145203). Top quarter miss minus bottom quarter miss >= 10 points. Report the paired photo-bootstrap 95% interval; flag it as borderline if the interval contains 10.
- Both parts must hold.

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
