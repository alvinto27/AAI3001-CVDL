> Archived 2026-10-16: superseded as the current task list; the original record stays here. See [status](../status.md) and [findings](../findings.md).

# Research checklist

Status date: 2026-10-16

This file records what the project tried, checked, and still needs. It uses
`P_E` for the best-threshold error probability unless it says **at threshold
0.5**. Best-threshold `P_E` is the minimum of `(false alarm + miss) / 2`
over the score thresholds. False alarm and miss values at threshold 0.5 use
the fixed probability threshold 0.5. Small differences from one seed are not
proof of a model improvement.

Journal links use the 2026-10-15 entry unless a different date is shown:
[2026-10-15 journal entry](../journal/2026-10-15/README.md#2026-10-15--g4-priming-run-and-pc-observation).
The commit codes identify the source or record commit. The journal entry is
the project record for each result.

## 1. Tried — experiments and runs

| Experiment or run | Change from baseline | Key result | Verdict | Record |
| --- | --- | --- | --- | --- |
| V1 baseline | BatchNorm with raw KV residuals | Test accuracy 0.839, AUC 0.989, `P_E` 0.0276. At the best threshold, false alarm / miss was 0.103 / 0.115. | Dropped. BatchNorm depended on the batch. | [Journal](../journal/2026-10-12/README.md#2026-10-12--vm-rebuild-evaluation-and-baseline-diagnosis); `results-v3.md` |
| V2 baseline | BatchNorm and the earlier training setup | Test accuracy 0.5352, AUC 0.915, `P_E` 0.1148. At the best threshold, false alarm / miss was 0.3017 / 0.3101. | Dropped. The batch problem remained. | [Journal](../journal/2026-10-12/README.md#2026-10-12--vm-rebuild-evaluation-and-baseline-diagnosis); `results-v3.md` |
| V3a | Clip the KV residual to `[-3, 3]`; keep BatchNorm | Test accuracy 0.9843, AUC 0.9982, `P_E` 0.0120. Train-mode all-clean accuracy was 0.0169 / 0.0218 for train / validation. | Superseded. Evaluation improved, but train-mode BatchNorm still had a batch leak. | [Journal](../journal/2026-10-12/README.md#2026-10-12--v3-runs-and-rocm-convolution-fix-continuation); `results-v3.md` |
| V3b | GroupNorm; no residual clipping | Test accuracy 0.9831, AUC 0.9971, `P_E` 0.0157. Lowest / plane k=1 `P_E` was 0.0437 / 0.0585. | Superseded by V3c as the historical baseline. Kept as a comparison. | [Journal](../journal/2026-10-12/README.md#2026-10-12--v3-runs-and-rocm-convolution-fix-continuation); `results-v3b.md` |
| V3c / `g4-prime-1` | Clip 3 plus GroupNorm on the full RAISE + DIV2K OTF set | Test `P_E` 0.016475, AUC 0.997728. At threshold 0.5, clean false alarm was 0.0071 and lowest k=1 miss was 0.1584. RAISE / DIV2K `P_E` was 0.011410 / 0.054560. | Kept as the main comparison baseline. It is not the map base candidate because GroupNorm has a spatial leak. | [Journal](../journal/2026-10-15/README.md#2026-10-15--g4-priming-run-and-pc-observation); commit `ad8c754` |
| MIL / Mode 2 step 17 | Photo bags and a photo-level MIL model | No final MIL run or metric. Overlap duplicated changed pixels in bags and skewed source balance. | Dropped. Use a spatial map direction instead. | [Journal](../journal/2026-10-15/README.md#2026-10-15--g4-priming-run-and-pc-observation); commit `bde38ef` |
| Map demo v1 | Current checkpoint; full-photo and chunked map; mixed settings per photo | Contiguous cell AUC was 0.94–1.00 in 18 cases. The photo mean diluted small regions; top-1% and max were positive on all 6 clean photos. Random placement changed nearly every cell. No training. | Superseded as a comparison because settings were random per photo. | [Journal](../journal/2026-10-15/README.md#2026-10-15--g4-priming-run-and-pc-observation); commits `0a0e623`, `d5ccde4` |
| Map demo v2 | Fixed Group A (`lowest`, k=1) and Group B (`lowest`, k=4), `match`, `random_bytes` | Clean ALASKA Canon EOS700D `P(stego)` was 0.554315. Group A random k=1 missed all 6 photos at 0.05 bpp. At 0.275 bpp, RAISE was 0.999995 / 0.998437, while DIV2K was 0.154259 / 0.266355. | Superseded by the no-normalisation R1 map check. It showed source and clean-hot-spot problems. | [Journal](../journal/2026-10-15/README.md#2026-10-15--g4-priming-run-and-pc-observation); commit `da97765` |
| No-normalisation R1 map | R1 checkpoint; no GroupNorm; chunked map | No shift outside changed chunks in 36/36 cases. Chunked vs whole-photo max absolute difference was 0.00064. Clean ALASKA EOS700D false alarm fell to 0.037. k=1 random score was about 0.03 even at 0.5 bpp. | Kept as evidence for the map design. Per-cell training is still needed. | [Journal](../journal/2026-10-15/README.md#2026-10-15--g4-priming-run-and-pc-observation); commit `00ce10c` |
| XuNet v4 R1 / `g4-v4-r1b` | New notebook 05, no normalisation, clip 3, LR `3e-4`, 20 epochs, k=1–8 training | Best validation `P_E` / AUC 0.0202 / 0.9949. Test `P_E` / AUC 0.022864 / 0.989898. At threshold 0.5, clean false alarm was 0.0172 and lowest k=1 miss was 0.1804. RAISE / DIV2K `P_E` was 0.0179 / 0.0615. | Dropped as the base candidate. The run underfit and had a validation spike. | [Journal](../journal/2026-10-15/README.md#2026-10-15--g4-priming-run-and-pc-observation); commit `ad8c754` |
| XuNet v4 R2 / `g4-v4-r2` | No normalisation; 30 epochs; train k=1–4; LR drops at epochs 5 and 20 | Best validation `P_E` / AUC 0.0227 / 0.9939. Test `P_E` 0.0250; bpp-1 `P_E` 0.0199. At threshold 0.5, clean false alarm was 0.0233 and lowest k=1 miss was 0.1041. | Superseded as the base candidate by wide XuNet. The LR and k-range changes were confounded. | [Journal](../journal/2026-10-15/README.md#2026-10-15--g4-priming-run-and-pc-observation); commit `563ae3c` |
| Wide XuNet / `wide-w4_r2` | Wider layers after the same front end; R2 recipe | Test `P_E` / AUC 0.019533 / 0.996092. Bpp-1 `P_E` / AUC 0.014878 / 0.997560. At threshold 0.5, clean false alarm was 1.396%, lowest k=1 miss was 10.264%, and k=1–4 `P_E` was 0.02373. | Pass. Base-model candidate. V3c still has lower test `P_E` (0.0165), so seed repeats are needed. | [Wide result](../results/result-wide-xunet.md); [Journal](../journal/2026-10-15/README.md#2026-10-15--g4-priming-run-and-pc-observation); commit `5720c87` |
| Mixed-source wide / `wide-w4_mixr2` (`...024859Z-9032`) | Wide XuNet, R2 recipe; data RAISE 40% / LSDIR 20% / ALASKA 40%, 45,000 tiles per epoch; capped validation 8,035 pairs | ALASKA unseen AUC 0.798 -> 0.934, FA at 0.5 47.5% -> 13.8%, old model saturates (25.6% of clean unseen tiles score exactly 1.0, so 5% FA is not reachable); LSDIR test P_E 0.0815 -> 0.0664; standard test P_E 0.0195 -> 0.0254. Best val epoch 25 (RAISE 0.021, ALASKA 0.156, LSDIR 0.076). | Kept for new-source research; wide-R2 kept for the standard test. Still underfits; ALASKA FA high on 21/28 seen cameras. | [Result](../results/result-newdata-1.md); [Journal](../journal/2026-10-16/README.md#2026-10-16--new-source-data-pi-freeze-first-mixed-source-g4-run); commit `ab9d3f1` |
| G4 speed matrix | Batch 32/64/128; fp32 and BF16; 600 timed steps after 50 warm-up steps | Images/s: 2,615.72 / 2,710.68 / 2,573.36 / 2,542.93 / 2,671.68 / 2,696.26 for 32-fp32, 32-BF16, 64-fp32, 64-BF16, 128-fp32, 128-BF16. | Keep batch 32. BF16 and validation/data-supply work are Pending. | [Journal](../journal/2026-10-15/README.md#2026-10-15--g4-priming-run-and-pc-observation); commit `ad33aa8` |
| GPU LSB generator design | Proposed train-only GPU embedding path | PC probe found decode 83% of CPU work and embedding about 8%. Estimated gain was 0–25%; no G4 generator run exists. | Shelved. | [GPU plan](plan-gpu-generator.md); [Journal](../journal/2026-10-15/README.md#2026-10-15--g4-priming-run-and-pc-observation); commit `a30748d` |
| DIV2K training plan | Earlier RAISE + DIV2K training mix | No new metric for removal. DIV2K files remain on disk and are kept for separate evaluation. | Dropped from the training plan. The team must keep the source records and report DIV2K separately. | [Journal](../journal/2026-10-15/README.md#2026-10-15--g4-priming-run-and-pc-observation) |

## 2. Tested — checks and verifications

| Check | Result and evidence | Status / record |
| --- | --- | --- |
| BatchNorm all-clean and all-stego checks | V3a train-mode all-clean accuracy was 0.0169 / 0.0218. V3c train/eval maximum logit difference was 0; all-clean / all-stego checks were 0.9950 / 0.9980 on validation. | Done. [Journal](../journal/2026-10-12/README.md#2026-10-12--v3c-run-and-notebook-monitoring-continuation); `results-v3.md` |
| GroupNorm locality and spatial leak | GroupNorm changed unchanged cells in a changed chunk by mean absolute 0.63–11.70 and max 140.3. Other chunks stayed exactly unchanged in 16/16 cases. Frozen clean statistics gave cell AUC 0.934–1.000, but required the clean cover. | Done; supports no normalisation for maps. [Journal](../journal/2026-10-15/README.md#2026-10-15--g4-priming-run-and-pc-observation); commits `d5ccde4`, `00ce10c` |
| Stream exactness and checkpoint resume | Decode/cache 4/8 worker streams had identical per-epoch SHA-256 for epochs 0–2 (12/12). A resume after epoch 0 matched an uninterrupted stream. | Done. [Journal](../journal/2026-10-15/README.md#2026-10-15--g4-priming-run-and-pc-observation); commit `d796971` |
| Standard test hash | The standard test hash `fdd989f7...837` stayed unchanged after optional bpp-1 and speed-fix work. | Done. [Journal](../journal/2026-10-15/README.md#2026-10-15--g4-priming-run-and-pc-observation); commits `fb3dfb7`, `0d6d157` |
| Persistent validation-worker parity | First 8 train-batch hashes matched with workers on/off for epochs 0 and 1. All 32 QUICK validation pairs matched on/off and across epochs. PC first-batch wait fell 4.780 → 0.870 s. | Done. [Journal](../journal/2026-10-15/README.md#2026-10-15--g4-priming-run-and-pc-observation); commit `0d6d157` |
| Notebook logging-byte parity | Old and new CellStream output bytes matched. A 32 MiB append+join took 0.0048 s + 0.0126 s. | Done. [Journal](../journal/2026-10-15/README.md#2026-10-15--g4-priming-run-and-pc-observation); commit `0d6d157` |
| LSDIR inventory | 84,991 HR PNG files, 155,204,828,212 B, all RGB 8-bit PNG, no duplicate paths. One byte-identical pair was found. | Done. [Journal](../journal/2026-10-15/README.md#2026-10-15--g4-priming-run-and-pc-observation); commit `3b2e8a4` |
| ALASKA v1 decode and SHA check | PC copy: 9,723 files, TIFF, RGB, 8-bit, both sides at least 512, 9,723 SHA entries match. | Done. [Journal](../journal/2026-10-15/README.md#2026-10-15--g4-priming-run-and-pc-observation); commit `0b43910` |
| G4 wide host and epoch timers | Host showed RAM 124.6 → 123.3 GiB, RSS 10.84 → 10.93 GiB, swap 0, I/O wait 0%; timers located the first validation wait at 18.773 s and late validation compute at 29.236 s. | Done. [Wide result](../results/result-wide-xunet.md); commit `0334e11` |
| Logging-bug data-only probe | PC epochs 20–27 had no slowdown; epoch 24 train was 1.117 s and validation 3.917 s. | Done. [Journal](../journal/2026-10-15/README.md#2026-10-15--g4-priming-run-and-pc-observation); commit `98a9b83` |
| New-data code parity | With the mix unset, the train stream (epochs 0/1) and val/test frames are byte-identical to before; standard test hash unchanged. Mix-on PC QUICK: 24/12/24 tiles, 0 repeats. Capped QUICK pair-list SHA `2f1424db...0ce8a`. 89 tests pass on a clean worktree. | Done. [Journal](../journal/2026-10-15/README.md#2026-10-15--g4-priming-run-and-pc-observation); commits `e592cce`, `1b54858` |
| ALASKA full download and v2 selection | 70,282/70,282 files, 0 failures, all verified; Pi re-hash and PC v1 re-hash 0 bad; 0 eligible IDs missing. `selection_v2.csv` SHA-256 `437fcfef...ff85` reproduces; 0 split-gap fallbacks; unseen models absent from other splits; no Leica_Mono; no repeated SHA-256. | Done. [Journal](../journal/2026-10-15/README.md#2026-10-15--g4-priming-run-and-pc-observation); commit `5ca650b` |

## 3. Hypotheses

| Statement | Status | Evidence so far | Test that would settle it | Cost |
| --- | --- | --- | --- | --- |
| No-normalisation tanh inputs in layers 1–2 saturate and cause R1/R2 underfitting and spikes. | Untested | R1/R2 underfit or spike; wide improves, but the cause is not isolated. | Log tanh ranges and saturation fractions by layer during training; compare with a controlled normalisation or input-scale run. | Low to medium; one instrumented short run. |
| The KV front end limits k=1 detection. | Supported, not isolated | Wide layers after the same front end changed lowest k=1 miss only 10.411% → 10.264%, while overall and DIV2K scores improved. | Keep the wide back end and test a front-end change with the same data, schedule and seed. | High; a new training run. |
| Scattered low-bpp k=1 is noise-limited by the square-root law. | Supported | At 0.05 bpp, about 1 in 120 values changed by ±1 and about 25 changes occurred per 32 px cell; clean/stego maps looked the same and the photo score changed 0.0015 → 0.0030. | Compare miss rate with measured tile noise and repeated seeded embeds at matched changed-pixel counts. | Low to medium; analysis plus a small map run. |
| The model ignores periodic stride structure and payload structure. | Supported | At equal bpp, stride misses were similar to or worse than random; contiguous k=1 letters / random bytes misses were 1.2% / 1.9%. | Use matched effective-bpp, equal-photo tests with more samples per stride and payload type. | Medium. |
| `map(raw) - map(denoised)` cancels clean hot spots. | Untested | The calibration idea may remove clean scene responses, but denoising may also remove natural noise and costs about 2x compute. | Run the denoise-and-compare demo on clean/stego pairs and measure clean false alarm, cell AUC, and k=1 P_E. | Medium to high. |
| Per-cell training fixes clean hot spots and the tile-label/map objective mismatch. | Untested | The current tile model produces clean hot spots; the generator can provide changed-pixel masks. | Train with dense or soft per-cell labels and compare clean hot spots, chunk/whole equality, cell AUC, and k=1 P_E. | High; new data code and training. |
| LSDIR, already shrunk 4x, is easier and may teach an easy-data shortcut. | Weakened | LSDIR HR is already shrunk 4x and is smoother than camera output. The source mix was changed to RAISE 40%, LSDIR 20%, ALASKA 40% to reduce this risk. | Train with and without LSDIR; report per-source and unseen-camera results at matched source weights. | High; multiple runs. |
| The R2 k=1–4 improvement is caused by the k-range change rather than the LR plan. | Untested | R2 changed both the k range and LR drops. Wide and R2 comparisons do not isolate either change. | Run a factorial control: old k range with new LR, and new k range with old LR, with the same seed and data. | High; at least two full runs. |
| Source diversity improves generalisation to unseen cameras. | Partly supported | Mixed run: unseen ALASKA AUC 0.798 -> 0.934 and FA at 0.5 47.5% -> 13.8%, mixed model reaches 5% FA on unseen ALASKA (miss 25.3%), the old model cannot (score saturation). One seed. | Use held-out ALASKA camera models and separate LSDIR test data; compare to the RAISE/DIV2K control across seed repeats. | High; new-source runs and repeats. |

## 4. To do — ordered work

| Order | Work | Owner | Status | Blocked by | User decision? |
| ---: | --- | --- | --- | --- | --- |
| 1 | Finish ALASKA/LSDIR readers, records, per-source sampler, extra validation/test sets, capped checkpoint validation. | Worker_2 | Done (`e592cce`, `1b54858`) | — | No |
| 2 | Run the final ALASKA selection v2 after the download. Keep the 8 unseen models and the 11-ID gap rule. | Worker_3 | Done (`5ca650b`); option B open | — | Yes: option B (only G11 and D90 train-only) |
| 3 | LSDIR subset selection v1 (7,000 / 1,500 / 2,500) and records | Worker_3 | Done (`4d6f27b`, `2659b3d`) | — | — |
| 4 | Copy data to PC and G4; build records | Workers | Done (`daf893c`; G4 verified 29,200 files) | — | — |
| 5 | First new-data G4 run (also measures post-fix timing) | Orchestrator + Worker_2 | Done (`ab9d3f1`) | — | — |
| 6 | B: ALASKA false-alarm review; per-source thresholds chosen on validation | Worker_2 | Done (`042903d`, [result](../results/result-check-b.md)) | — | No |
| 7 | A: per-cell (dense) training | Worker_2 | Pending | B | Design review |
| 8 | C: more capacity / longer training / bf16 | — | Pending | A | Yes |

### Decisions still pending

| Decision | Owner | Status | Record |
| --- | --- | --- | --- |
| Checkpoint-selection validation set | User | Done: capped (~8,052 pairs) | `1b54858` |
| ALASKA train-only models: six (committed) or two (option B) | User | Pending | Orchestrator recommends option B. |
| BF16 for G4 | User | Pending | Speed matrix is complete; batch 32 is kept. |
| Fair control R0 | User / Orchestrator | Pending | Needed before claiming a source or model gain. |
| Seed repeats | User / Orchestrator | Pending | One-seed results are not enough for small differences. |
| Front-end upgrade | User / Orchestrator | Pending | Do after wide result and source-data run. |

### Free checks offered

| Check | Owner | Status | Cost or note |
| --- | --- | --- | --- |
| Miss rate versus tile noise | Worker_2 | Pending | Small analysis plus a map run. |
| Denoise-and-compare demo | Worker_2 | Pending | About 2x inference work. |
| Tanh saturation | Worker_2 | Pending | Instrumented short run. |
| Filter linear probe | Worker_2 | Pending | Optional diagnostic; no training result yet. |

Update this checklist in the same commit as each journal update. Keep values
that are estimates or reports marked as such. Do not treat one-seed results as
final evidence.
