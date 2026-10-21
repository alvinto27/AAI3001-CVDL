# Findings across runs and checks

Updated 2026-10-21 for the standard-256 / hard-256 tests. Read [status](status.md) first. Each row cites its record; the [run queue](active/run-queue.md) has every run's job ID and evaluation ID.

**Terms.** *P_E* = overall error, the mean of the false-alarm rate (FA, clean tiles called stego) and the miss rate (stego tiles called clean). Each model uses **its own cut-off, chosen on validation** before the test. *AUC* = how well the scores rank stego above clean, independent of the cut-off (1.0 = perfect, 0.5 = guessing). *Standard* = RAISE + DIV2K, 10,944 pairs. *Hard target* = k=1, low bpp, contiguous payloads on ALASKA seen/unseen + LSDIR. *Interval* = 95% photo-bootstrap range.

## Submission model (user, 2026-10-21)

**Run 13, `srm-cell-aux-20261021T054704Z-6997`, `last.ckpt`** (SHA-256 `4dc9ab8e...78b9`), evaluation `ea32a4be0e2ce3597fbd`. One checkpoint gives both the tile score and the cell heatmap (a trained 1x1 cell head on the 8x8 feature map; one cell = 32 px step, 48x48 px view). The user chose it over the frozen standard guard, which it misses by 0.07 points. SRM-e50 stays the best score-only model.

## Standard-256 ledger

P_E % at each model's own validation cut-off (standard / ALASKA seen / ALASKA unseen / LSDIR). Rescore source: [2026-10-18 journal](journal/2026-10-18/README.md); runs 10-13 from their evaluations.

| Run | What changed | Standard | ALASKA seen | ALASKA unseen | LSDIR | Standard AUC | Record |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| overlap-512 | KV front end, 512 tiles | 5.61 | 23.94 | 21.43 | 12.62 | — | [10-18](journal/2026-10-18/README.md) |
| trim-512 / pad-512 | other tilings, 512 | 6.48 / 6.53 | 23.49 / 24.20 | 21.31 / 20.93 | 12.95 / 12.88 | — | [10-18](journal/2026-10-18/README.md) |
| newdata-1 | mixed sources, wide KV, 512 | 5.13 | 24.29 | 21.21 | 12.64 | — | [10-18](journal/2026-10-18/README.md) |
| overlap-256 | 256 training tiles | 3.43 | 21.21 | 18.75 | 11.27 | — | [10-18](journal/2026-10-18/README.md) |
| trim-256 / pad-256 | other tilings, 256 | 4.82 / 5.07 | 21.89 / 22.19 | 19.11 / 19.27 | 10.86 / 11.27 | — | [10-18](journal/2026-10-18/README.md) |
| SRM-30 | SRM front end, 30 epochs | 3.07 | 20.03 | 17.31 | 9.63 | — | [10-18](journal/2026-10-18/README.md) |
| **SRM-e50** | SRM, 50 epochs | **2.54** | 19.42 | 16.30 | 9.39 | .9964 | [10-18](journal/2026-10-18/README.md) |
| srm-hard | hard data mix | 3.10 | 21.95 | 19.16 | 9.77 | — | [10-18](journal/2026-10-18/README.md) |
| run 10 curriculum | low-bpp curriculum from SRM-e50 | 3.10 | — | — | — | .9964 | [plan, result](active/plan-low-bpp-curriculum.md) |
| run 11 control | SRM-e50 + 15 epochs, same recipe | 2.65 | 18.81 | 16.52 | 9.31 | .9964 | [plan, result](active/plan-low-bpp-curriculum.md) |
| run 12 delayed pooling | arm A architecture | 3.43 | — | — | — | .9930 | [plan, result](active/plan-architecture-tests.md) |
| **run 13 cell head** | run 11 + cell head and cell loss | 2.97 | 18.95 | 16.21 | 9.26 | .9962 | [plan, result](active/plan-cell-training-256.md) |

## Cell heatmap (run 13)

The model gives an 8x8 grid of cell scores per 256 tile. Frozen diagnostic set: 192 validation / 256 test pairs, chosen before training (`configs/cell-diagnostic-v1.json`). Primary measure: inside each contiguous stego tile, do changed cells score above unchanged cells (mean per-tile AUC, 128 test tiles).

| Map | All | RAISE | ALASKA seen | ALASKA unseen | LSDIR |
| --- | ---: | ---: | ---: | ---: | ---: |
| **Run 13 trained cell head** | **.863** [.836, .890] | .972 | .817 | .822 | .840 |
| Run 13 CAM (tile-head weights on the cells) | .806 [.770, .840] | .968 | .720 | .747 | .788 |
| Control CAM (no cell training) | .793 [.757, .828] | .957 | .711 | .723 | .783 |

- Trained head minus control CAM: **+.070 [+.053, +.088]**; minus its own CAM: +.057 [+.042, +.075].
- At a strict cell cut-off (5% false cells on validation; 3.8% on test) every map finds about **16%** of the changed cells. Training improved the ranking inside a tile, not the count above a strict cut-off.
- Figure: `runs/verification/cell-run13-2026-10-21/cells/cells.png` (git-ignored run output).

## Photo verdict (rule c, accepted by the user 2026-10-21)

A whole photo is cut into 256 tiles; a small logistic rule reads 8 summary numbers (4 from the cell-head map, such as the largest connected group of flagged cells, and 4 from the tile scores) and gives one yes/no. Fitted and calibrated on 448 validation photos; frozen before the test. Record: `runs/verification/photo-verdict-run13-2026-10-21/REPORT.md`; [2026-10-21 journal](journal/2026-10-21/README.md).

| TEST photos (576; 1 clean + 12 stego versions each) | Clean photos flagged | Messages missed | P_E | AUC |
| --- | ---: | ---: | ---: | ---: |
| All | 8.5% | 9.3% | 8.9% | .960 |
| RAISE / DIV2K / LSDIR | 2.1 / 9.4 / 3.1% | 1.7 / 2.3 / 4.2% | 1.9 / 5.9 / 3.6% | — |
| ALASKA seen / unseen | 9.4 / **17.2%** | 20.8 / 14.0% | 15.1 / 15.6% | — |
| Hard: k=1, 0.05-0.15 bpp | 7.8% | 38.1% | 23.0% | .872 |

- Frozen mark: FA <= 7% **failed** (8.5%; validation 4.9%); hard P_E <= 25% passed. Accepted with this limit.
- Tile scores add value over cell features alone: P_E -2.0 pts [-2.7, -1.3]; -6.8 pts on hard k=1. "Any tile above the tile cut-off" flags 5 of 6 clean photos and must not be used.

## What we learned (256 era)

- **256 training tiles beat 512**, and overlap-last tiling beats trim and pad at both sizes (overlap-256 3.43% vs overlap-512 5.61%).
- **The SRM front end is the largest single gain** (SRM-30 3.07% vs KV overlap-256 3.43%; better on every panel). 50 epochs improve it further (2.54%).
- **Low-bpp k=1 is the hard core.** Every model is near chance on ALASKA k1 low+noisy (AUC .51-.52 at the 10-18 rescore). Training tricks aimed at it move the threshold more than the ranking: srm-hard (misses down, false alarms up), the curriculum (+.019 hard AUC, +0.56 pts standard P_E), the cell head (+.005 hard AUC, +0.32 pts).
- **Bigger receptive fields did not help.** Delayed pooling (arm A) was worse on every panel at 2.3x compute.
- **Cell supervision works for the heatmap.** A target taken from the embedding mask raised within-tile localization by +.07 AUC at the same compute, with the largest gain on the photo sources that are not RAISE.
- **The noisy-clean false alarms stay high** (about 19-21% on the hard noisy-clean union for runs 11 and 13). This is the main limit for real photos.

## Open questions

| Question | Status |
| --- | --- |
| Can a model separate k=1 low-bpp payloads in noisy ALASKA photos? | Open; all tests so far near chance |
| Does a GBRAS-Net style front end (arm G) help? | On hold (user, 2026-10-21) |
| Does a heavier cell-loss weight or a cell-only cut-off improve recall at a strict cut-off? | Not tested |
| Do the gains hold across seeds? | Not tested; every run is one seed |

## Earlier findings (512 tests, up to 2026-10-17)

> Superseded by the sections above. Kept unchanged as the record of the 512 era.


> ~~**Note (2026-10-19):** this ledger covers runs up to 2026-10-17 on the **old 512 tests** (threshold 0.5 or best threshold). It is not updated for the standard-256 / hard-256 suite. Current numbers: [2026-10-18 journal](journal/2026-10-18/README.md) (all 11 models rescored), [2026-10-19 journal](journal/2026-10-19/README.md), the [run queue](active/run-queue.md) and the result sections of the active plans. A rewrite is pending (A4, with the results write-up A3).~~ Rewritten 2026-10-21: see the 256 sections above.

Read [status](status.md) before doing more work. This is a cross-run index, not a new experiment. `P_E` below is the best-threshold error on the named test; clean false alarm (FA) and k1 miss use the **fixed 0.5 threshold** unless stated otherwise. `—` means the cited record did not give a comparable value or cost. The early prebuilt RAISE test has **7,433 pairs**; the later standard OTF RAISE/DIV2K test has **10,956 pairs**. Do not compare their P_E as if they were the same panel. Small one-seed differences are not proof of a general gain. Rates and costs are cited in the linked result or [dated journal](journal/README.md).

### Training-run ledger (512 tests)

| Run | Data | Model and recipe | Test P_E | AUC | Lowest k1 miss at 0.5 | Clean FA at 0.5 | Special panel or outcome | VM cost estimate | Result record |
| --- | --- | --- | ---: | ---: | ---: | ---: | --- | ---: | --- |
| V1 baseline | Prebuilt RAISE, 7,433 test pairs | XuNet KV + BatchNorm, Adam 1e-3, StepLR; stopped during epoch 9 | .0276 | .989 | — | —; accuracy .839 | Combined k1 P_E .111; train-mode BatchNorm depends on batch | — (PC run) | [V1](results/results-v1-baseline.md), [journal day 12](journal/2026-10-12/README.md) |
| V2 baseline | Prebuilt RAISE, same test | XuNet + BatchNorm, unpaired shuffled images, AdamW, 10 epochs | .1148 | .915 | — | —; accuracy .5352 | Batch issue remained; superseded | — (L4 run) | [V3 comparison](results/results-v3.md), [journal day 12](journal/2026-10-12/README.md) |
| V3a | Prebuilt RAISE, same test | Clip KV to ±3; keep BatchNorm, 10 epochs | .0120 | .9982 | — | 63 / 7,433 clean tiles | Plane k1 P_E .0503; train-mode all-clean failure | — (L4 run) | [V3 comparison](results/results-v3.md) |
| V3b | Prebuilt RAISE, same test | No clip; GroupNorm, 10 epochs | .0157 | .9971 | — | .01049 (78 / 7,433) | Plane k1 P_E .0585; batch-independent, later superseded | — (PC run) | [V3b](results/results-v3b.md), [V3 comparison](results/results-v3.md) |
| V3c, prebuilt | Prebuilt RAISE, same test | Clip 3 + GroupNorm, 10 epochs | .0139 | .9982 | — | 57 / 7,433 clean tiles | Plane k1 low-BPP P_E .1123; batch-independent | — (PC run) | [V3 comparison](results/results-v3.md) |
| V3c / `g4-prime-1` | OTF RAISE + DIV2K, 10,956 standard test pairs | Clip 3 + GroupNorm, full-source OTF | .016475 | .997728 | .1584 | .0071 | RAISE / DIV2K P_E .011410 / .054560; bpp-1 P_E .0125 | — | [Checklist record](archive/research-checklist.md#1-tried--experiments-and-runs), [journal day 15](journal/2026-10-15/README.md) |
| R1 / `g4-v4-r1b` | Same standard OTF test | No norm, clip 3, train k1–8, LR 3e-4, 20 epochs | .022864 | .989898 | .1804 | .0172 | Bpp-1 P_E .0187; map has no GroupNorm spatial leak | — | [Journal day 15](journal/2026-10-15/README.md), [checklist](archive/research-checklist.md#1-tried--experiments-and-runs) |
| R2 / `g4-v4-r2` | Same standard OTF test | No norm, clip 3, train k1–4, LR drops at 5/20, 30 epochs | .025009 | .989593 | .10411 | .02327 | Bpp-1 P_E .019898; LR and k-range both changed | — | [Wide comparison](results/result-wide-xunet.md), [journal day 15](journal/2026-10-15/README.md) |
| Wide-R2 | Same standard OTF test | Wide XuNet, same R2 recipe, 30 epochs | .019533 | .996092 | .10264 | .01396 | Bpp-1 P_E .014878; held-out ALASKA clean scores saturate at 1.0 | ~$1.66 | [Wide result](results/result-wide-xunet.md), [check B](results/result-check-b.md) |
| Mixed-source wide | Same **unchanged** standard test | Wide XuNet, RAISE/LSDIR/ALASKA 40/20/40, 45,000 tiles/epoch, capped mixed validation, 30 epochs | .025370 | .994210 | .11437 | .01406 | ALASKA unseen P_E .23359 → .13876 and LSDIR test .08151 → .06635 vs wide-R2; standard test regressed | ~$1.28 (plus failed attempt ~$0.20) | [Mixed result](results/result-newdata-1.md), [check B](results/result-check-b.md) |

The V3a/V3c FA entries use the **published counts**, not an invented rate. G4 dollar figures are VM-time estimates, not billing records; no comparable cost was published for the other run rows. The old wide-R2 model has **no useful strict ≤5%-FA ALASKA validation cutoff**: 418 of 1,535 clean validation scores are exactly 1.0. Including them breaks the FA limit; excluding them predicts everything clean. Earlier old-model ALASKA “miss at 5% FA” values **21.25% / 22.24% were wrong** because ties at 1.0 made the actual FA about 25.6% / 28.3%. Use [the correction and validated thresholds](results/result-check-b.md), not those old values.

### What we learned and what was checked (512 era)

- **Normalization matters for maps.** V1/V2 BatchNorm depended on batch composition. V3a clipping improved evaluation but its train-mode all-clean check still failed. GroupNorm removed batch dependence in V3b/V3c, yet a map probe found a *spatial* leak within a chunk. No-normalization R1 had no shift outside changed chunks in 36/36 cases, but weak k1 detection remained. [V3 comparison](results/results-v3.md); [archived checklist, Tried/Tested](archive/research-checklist.md).
- **Wider is not a universal fix.** Wide-R2 improved R2 standard P_E .025009 → .019533 but barely moved lowest-k1 miss .10411 → .10264. The later mixed model improved unseen-camera ALASKA AUC .79818 → .93403 and fixed-0.5 FA 47.48% → 13.80%, while its standard P_E worsened .019533 → .025370 and ALASKA unseen k1 miss at 0.5 rose 31.30% → 49.11%. [Wide](results/result-wide-xunet.md); [mixed](results/result-newdata-1.md).
- **Calibration and weak-signal detection are separate.** The mixed ALASKA validation-picked FA5 cutoff gave 5.97% seen / 5.93% unseen test FA but missed 29.72% / 24.20% of stego images and 68.14% / 66.16% of k1. One shared cutoff gave 29.53% / 24.36% ALASKA FA. KV-noise quartiles and within-camera checks link both clean false alarms and k1 misses to texture; these are associations, not causal proof. [Check B](results/result-check-b.md).
- **Data and run contracts were tested.** Decode/cache 4/8-worker train streams matched 12/12 per-epoch hashes; resume matched; persistent-worker on/off maintained data order, and PC validation first-batch wait fell 4.780 → .870 s. The standard-test settings hash stayed unchanged. ALASKA and LSDIR files and manifests were verified. The mixed G4 warm rate reached 54.5 steps/s, with no old late-epoch slowdown. [Archived checklist, Tested](archive/research-checklist.md#2-tested--checks-and-verifications); [mixed result](results/result-newdata-1.md).
- **Other tried directions:** MIL/photo bags (Mode 2) were dropped after overlap skewed source balance; the code is archived in `LSB-Dataset-Generator/archive/`. Fixed-setting map demos and a no-normalization map probe were diagnostic, not training results. A G4 speed matrix kept batch 32; a GPU LSB generator was shelved after a CPU profile found only about 8% embedding work. Source-reader and manifest work completed. [Archived checklist, Tried](archive/research-checklist.md#1-tried--experiments-and-runs); [GPU plan](archive/plan-gpu-generator.md).

### Hypotheses and their status (2026-10-17)

This table carries the hypotheses from the [archived checklist](archive/research-checklist.md#3-hypotheses). It updates only statuses supported by the later mixed result and check B. **Untested** does not authorize work while the project is on hold.

| Hypothesis | Status | Evidence / limit | Test still needed |
| --- | --- | --- | --- |
| No-norm tanh input saturation causes R1/R2 underfitting and spikes | Untested | R1/R2 showed spikes; the wide model changed more than tanh behavior | Instrument tanh ranges with a controlled comparison |
| KV front end limits k1 detection | Supported, not isolated | R2 → wide lowest-k1 miss .10411 → .10264 despite wider layers | Same data/seed/schedule, controlled front-end change |
| Scattered low-BPP k1 is noise-limited | Supported, not causal | About 25 changes per 32 px cell at .05 BPP in a prior map; check B found noise linked to both errors | Match tile noise and changed-pixel counts across repeated seeds |
| Model ignores periodic stride and payload structure | Supported | Similar or worse stride miss than random at equal BPP; small subgroup counts | Equal-photo, matched effective-BPP stride/payload checks |
| `map(raw) − map(denoised)` cancels clean hot spots | Untested | A denoised reference may also remove real cover noise; about two inference passes | Compare clean FA, cell AUC and k1 P_E with validation calibration |
| Per-cell training removes clean hot spots | ~~Untested; design on hold~~ Tested at 256 (run 13, see above) | Current cell map is not supervised; [plan written, not approved](active/plan-per-cell-training.md) | Only after approval: opt-in cell labels, unchanged tile tests and cell validation |
| LSDIR shrink can teach an easy-source shortcut | Weakened, not isolated | Mixed source uses LSDIR 20%; its separate test P_E is .06635, not near zero | Controlled with/without-LSDIR runs, not approved |
| R2 k1–4 improvement came from k-range rather than LR | Untested | R2 changed both factors | Factorial controls with the same data and seed |
| Source diversity helps unseen cameras | Partly supported; one seed | Unseen ALASKA AUC .79818 → .93403 and corrected FA improvement; standard test and k1 misses regress | Separate held-out cameras and seed repeats with fair controls |
