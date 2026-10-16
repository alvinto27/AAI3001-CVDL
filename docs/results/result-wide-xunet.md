# Wide XuNet against R2: G4 result

The G4 ran job `xunet_v4_clip3_nonorm_lr0.0003_warm1_drops5x0.3-20x0.1_k1to4_e30_wide-w4_r2-20261015T121753Z-29499` at commit `4f5fa5f`. It completed all 16 notebook 05 cells, delivered to the Pi, and stopped. The PC copy is in `runs/<job-id>/`; the private `job.env` was not copied. The VM still uses the old installed one-file `job.sh` (SHA-256 `5677bff8272dff40ed1ccc7208ea4f99021a7b13fac1f596824706dc3e1269a0`). It ran the checked-out notebook runner, which wrote `host.csv` and `epoch_timing.csv`. There was no retry or preemption. The fit cell took **3,235.450 s (53.92 min)**; the complete notebook took **3,491.374 s (58.19 min)**. The GCE start operation began 12:18:03 UTC; guest shutdown was 13:19:01 UTC. This is about **60.97 VM min, or $1.66 at $1.63/h**, as an operation-time estimate, not a billing record. The pre-run 67-minute notebook estimate was conservative.

## Standard test: 10,956 pairs

`P_E` below is the notebook's **best threshold over all scores** within the named group. `FA` and `miss` use the **fixed 0.5 threshold** and count clean and stego images, respectively. AUC is threshold-free. The best checkpoint is chosen by validation P_E, not test P_E. R2 is `runs/g4-v4-r2/` under its own best checkpoint; both runs use the same standard test.

| Group | Pairs | R2 P_E | Wide P_E | R2 AUC | Wide AUC | R2 FA / miss at 0.5 | Wide FA / miss at 0.5 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| All | 10,956 | .025009 | **.019533** | .989593 | **.996092** | 2.327% / 2.793% | **1.396% / 2.629%** |
| k 1–4 | 5,522 | .030786 | **.022275** | .988156 | **.995696** | 2.572% / 3.731% | **1.684% / 3.350%** |
| RAISE | 9,728 | .018503 | **.015111** | .991916 | **.997097** | 1.830% / 1.984% | **1.141% / 2.046%** |
| DIV2K | 1,228 | .074104 | **.052932** | .959482 | **.982787** | 6.270% / 9.202% | **3.420% / 7.248%** |
| k1 lowest bit | 682 | .053519 | **.049120** | .972672 | **.987164** | 3.226% / **10.411%** | **2.053% / 10.264%** |
| k1 bit plane | 750 | .066000 | **.042000** | .975273 | **.988994** | 2.533% / 12.400% | **1.333% / 11.067%** |
| k8 lowest bit | 713 | .039271 | **.035764** | .973350 | **.985557** | 3.366% / **4.628%** | **2.104% / 5.189%** |

**Pass marks fixed before training:**

| Mark | Actual wide value | Verdict |
| --- | --- | --- |
| k 1–4 P_E at least about .002 below the stated R2 .0303 | **.022275**, lower by .008025 against the stated .0303 (or .008511 against the direct R2 regrouped .030786) | **PASS** |
| k1 lowest-bit miss below 10.4% at threshold 0.5 | **10.264%** (R2 10.411%) | **PASS**, but only 0.147 percentage point below R2 |
| All-clean FA at threshold 0.5 no higher than 2.3% | **1.396%** (R2 2.327%) | **PASS** |

The stated R2 k 1–4 benchmark was `.0303`; direct recomputation on the saved R2 prediction rows with the same tied-score P_E method yields `.030786`. Both comparisons pass by much more than .002. Do not confuse the optimized P_E with the fixed-threshold FA and miss.

At **equal effective BPP bins**, k1 stego miss rates at threshold 0.5 are:

| Placement | Effective BPP | Pairs | R2 miss | Wide miss |
| --- | --- | ---: | ---: | ---: |
| Contiguous | .05–.15 | 115 | 6.96% | **6.09%** |
| Contiguous | .15–.30 | 169 | 0.00% | 0.00% |
| Contiguous | .30–.50 | 234 | 0.85% | 0.85% |
| Random | .05–.15 | 99 | 56.57% | **48.48%** |
| Random | .15–.30 | 140 | 14.29% | 14.29% |
| Random | .30–.50 | 186 | 5.38% | **4.30%** |
| Stride (all stride values) | .05–.15 | 115 | **26.09%** | 27.83% |
| Stride (all stride values) | .15–.30 | 279 | 10.75% | **10.39%** |
| Stride (all stride values) | .30–.50 | 95 | 8.42% | **7.37%** |

These are the same photo/pair counts in both runs. Wide improves most bins; it is worse at low-BPP stride and at k8 lowest-bit fixed-threshold miss. Do not claim a universal improvement.

The **extra bpp-1 test** (separate evaluation settings, 10,956 pairs) gave R2 P_E **.019898**, AUC **.991922**; wide P_E **.014878**, AUC **.997560**. Both used the saved best checkpoint and the same extra test setup. This is not the standard test score.

## Validation, fit, and interpretation

Wide selected **epoch 24** (one-based) with validation P_E **.017663**. R2 selected **epoch 20**, validation P_E **.022735**. Wide validation P_E fell from .0464 at epoch 1 to about .018–.020 in epochs 11–30. It spiked to .0485 at epoch 5, then recovered to .0240 at epoch 6; smaller bumps reached .0274 at epoch 9 and .0236 at epoch 18. R2 spiked to .0493 at epoch 5, .0407 at epoch 9, and .0369 at epoch 15. The saved `logs/csv/run/metrics.csv` contains each epoch of both curves.

Wide train loss fell from .4632 to .1196 and validation loss from .2260 to .0886. R2 fell from .4834 to .1459 and .2398 to .1091. Both losses fell, and wide improved validation P_E and AUC: the evidence does **not** show wide underfitting. Train and validation distributions differ, so their loss values alone do not measure a generalization gap. The wide best epoch is late and the final validation P_E (.0183) is close; this run alone does not prove the optimum model size.

## Time and host state

The new timers isolate the stable transition more closely. The table excludes the first epoch compile and uses early epochs 2–24 (zero-based 1–23) and late epochs 25–30 (zero-based 24–29). Seconds are mean values from `epoch_timing.csv`. The last epoch has no next-epoch wait or prime.

| Timed phase | Early mean, s | Late mean, s |
| --- | ---: | ---: |
| Train first-batch wait | .010 | .039 |
| Train first batch to last batch | 61.888 | **101.890** |
| Last train batch → validation start | .003 | .020 |
| Validation start → validation epoch start | .017 | .041 |
| Validation epoch start → first validation batch | **18.773** | **17.999** |
| First → last validation batch | 13.163 | **29.236** |
| Last validation batch → validation-end hook | .009 | .035 |
| Validation end → next train first batch | .457 | .503 (five epochs) |
| Curve redraws, scalar work, prime, checkpoint saves | .407, .235, .015, .032 | .388, .472, .014, .026 |

The former **~20 s transition** is mainly the **first validation batch wait (~19 s)**, not checkpoint writing or priming. The eval loader has 16 non-persistent workers; its first-batch wait includes worker creation and initial data. The timers cannot split these two causes, but they locate the work. A persistent eval worker trial can target this delay without changing pair order; verify data and memory. The measured `prime_s` is only the callback submission, not the background work. The last epoch's first validation wait was 14.47 s, lower than the other late epochs.

The late slowdown **happened again** at epoch 25 (zero-based 24), in both training (61.9 → 101.9 s) and validation compute (13.2 → 29.2 s); the first-batch wait stayed near 19 s. Minute-sampled `host.csv` and `nvidia.csv` show the following periods (UTC). Samples include both train and validation, so they cannot identify a batch wait directly.

| Mean sample | 12:35–12:58 (23) | 13:01–13:13 (12) |
| --- | ---: | ---: |
| Host CPU busy / I/O wait | 27.8% / **0%** | 16.8% / **0%** |
| RAM available | 124,588 MiB | 123,315 MiB |
| Python process RSS sum | 10,842 MiB | 10,933 MiB |
| Page cache | 26,194 MiB | 27,739 MiB |
| Swap used / disk read | **0 MiB / ~0 MiB/s** | **0 MiB / ~0 MiB/s** |
| NVIDIA GPU use / power | 66.5% / 298 W | 44.0% / 218 W |
| NVIDIA reported GPU memory | 4,517 MiB | 4,517 MiB |

RAM, swap, disk I/O wait, RSS growth, and GPU-memory growth **do not explain** the slowdown. CPU use and GPU use both fell, so there was less work completed, but these minute samples cannot show whether the cause was a data pipeline delay, CPU scheduling, or GPU scheduling. Temperature also fell (54.6→48.8 °C), which does not support heat throttling. The same late epoch boundary in R2 suggests a common phase-dependent cause, **not proof of a wide-model fault**. Add per-batch loader-wait and GPU-clock samples before changing the data stream.
