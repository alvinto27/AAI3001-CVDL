# Mixed ALASKA/LSDIR wide XuNet: first G4 result

The successful G4 job is `xunet_v4_clip3_nonorm_lr0.0003_warm1_drops5x0.3-20x0.1_k1to4_e30_wide-w4_mixr2-20261016T024859Z-9032`, from checkout `3a6b1fd`. It ran all 16 notebook 05 code cells, exited 0, delivered to the Pi at 03:37:26 UTC on 2026-10-16, and shut down. Its PC copy is `runs-out/<job-id>/`; the private `job.env` was not copied. The earlier attempt, `...20261016T023422Z-16940`, stopped during Lightning sanity checking because the partial sanity batch lacked ALASKA. Commit `1a6a6bf` fixed that hook without weakening full-epoch checks. The earlier failed attempt was delivered when this VM booted; **do not combine its incomplete results with this run**. Neither failure nor success was a Spot preemption or a capacity retry.

**Result:** The mixed model improves detection on ALASKA and LSDIR against the earlier wide-R2 checkpoint when both checkpoints score the *same* five new-source panels. The gain includes eight ALASKA camera models withheld from mixed training. But the standard RAISE/DIV2K test regresses, and weak k1 stego images are more often missed at threshold 0.5. Keep wide-R2 as the better standard-test model; use the mixed checkpoint only if new-source coverage is worth that trade-off. Do not claim that the mixed model is better in every setting.

## Identity and evaluation rules

The mixed run used 30 epochs, wide XuNet width 4, batch 32, FP32, 32 train / 16 persistent evaluation workers, `V3_COMPILE=default`, 45,000 seeded tiles per epoch, and train source weights RAISE/LSDIR/ALASKA **40/20/40**. Source logs report zero repeated tiles. Learning rate was 3e-4 with one warmup epoch and drops at epochs 5 and 20. Source mode was full-photo decode, not an image cache. The standard test still has **10,956 pairs** and the unchanged settings SHA-256 `fdd989f7bdab0ec6f9fd6f4c57bdde981d16b54aa8c37e5da8fd3dfcc8b06837`; RAISE and DIV2K records were not changed.

Checkpoint selection used a fixed, ordered **8,035-pair** validation list: 5,000 RAISE, all 1,535 ALASKA validation pairs, and 1,500 LSDIR validation pairs. The saved `validation_pairs.csv` SHA-256 is `6bb3c8efadad06af50ba014a95c98b4fc57008119bdd55b794ec2a1729e4d5ca`. It excludes DIV2K. The selected checkpoint is **epoch 26 (zero-based 25)**, `best-epoch25-val_pe0.0574-val_auc0.9841.ckpt`, SHA-256 `a3866b7c645ef1f05314ecb5a4df0cfaff1d45c9e51c488dd81995abbbe45ae4`. The wide-R2 comparison checkpoint is `best-epoch23-val_pe0.0177-val_auc0.9957.ckpt`, SHA-256 `9ba63252e8eecbf398f646f2d69641b70ad37c914ed8b7d6b4665dda54a961e1`. **The two models use different validation sets to select their checkpoints.** Score comparisons below use identical test records, but they do not isolate checkpoint selection from training-source effects.

`P_E` is the best error across score thresholds *within each reported group*; it is not an operational threshold chosen on an independent calibration set. AUC does not need a threshold. False alarm (FA) counts clean images with score ≥0.5; miss counts stego images with score <0.5. These two columns show the fixed-threshold trade-off. The five new panels use the same pair IDs and evaluation-settings hash for both checkpoints; the wide-R2 panel predictions were produced on the PC with `scripts/eval_checkpoint.py` and 4 workers, while the mixed checkpoint's panels were produced on the G4 with 16 workers. Their saved `run.json` files contain the individual settings hashes and checkpoint hashes.

## Standard held-out test: wide-R2 versus mixed

| Group | Pairs | Wide-R2 P_E | Mixed P_E | Wide-R2 AUC | Mixed AUC | Wide-R2 FA / miss % | Mixed FA / miss % |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| All | 10,956 | **.019533** | .025370 | **.996092** | .994210 | 1.396 / **2.629** | **1.406** / 3.760 |
| RAISE | 9,728 | **.015111** | .020302 | **.997097** | .995350 | 1.141 / **2.046** | **.761** / 3.464 |
| DIV2K | 1,228 | **.052932** | .059039 | **.982787** | .981410 | **3.420** / 7.248 | 6.515 / **6.107** |
| k1 | 1,432 | **.046439** | .059008 | **.988149** | .982990 | **1.676** / **10.684** | 1.746 / 12.011 |
| k1 lowest bit | 682 | **.049120** | .056452 | **.987164** | .984370 | 2.053 / **10.264** | **1.906** / 11.437 |
| k1 bit plane | 750 | **.042000** | .060667 | **.988994** | .981850 | **1.333** / **11.067** | 1.600 / 12.533 |

The all-test P_E increases by **.005837** (+29.9% relative) and k1 P_E by **.012569**. This is a measurable regression on the unchanged standard test, not evidence that the source records moved. The separate **bpp-1** test, also 10,956 pairs, changes from wide-R2 P_E `.014878`, AUC `.997560` to mixed P_E `.020947`, AUC `.996207`. Do not merge the bpp-1 test with the standard test.

## Five new-source panels on identical pairs

| Panel | Pairs | Wide-R2 P_E / AUC | Mixed P_E / AUC | Wide-R2 FA / miss % | Mixed FA / miss % | k1 miss %: R2 → mixed (k1 pairs) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| ALASKA val | 1,535 | .24560 / .78433 | **.15635 / .92128** | 50.55 / **9.84** | **14.27** / 18.18 | 32.76 → 55.75 (174) |
| ALASKA test, seen cameras | 2,648 | .25057 / .78424 | **.16616 / .91587** | 49.74 / **8.31** | **15.03** / 18.47 | 28.32 → 52.21 (339) |
| ALASKA test, unseen cameras | 3,153 | .23359 / .79818 | **.13876 / .93403** | 47.48 / **9.10** | **13.80** / 14.91 | 31.30 → 49.11 (393) |
| LSDIR val | 3,164 | .08834 / .95975 | **.07253 / .97208** | 9.01 / **9.20** | **4.74** / 10.05 | 38.17 → 44.27 (393) |
| LSDIR test | 5,245 | .08151 / .96280 | **.06635 / .97422** | 7.91 / **8.52** | **3.55** / 10.22 | 38.80 → 47.48 (634) |

The unseen-camera ALASKA P_E improves by `.09483` (~40.6% relative), AUC rises `.79818` → `.93403`, and fixed-threshold FA falls **47.48% → 13.80%**. LSDIR test P_E improves `.08151` → `.06635` (~18.6% relative). This supports improvement on new-source held-out photos, including unseen ALASKA camera models; wide-R2 had **not** trained on either new source. The improvement is **not free**: at threshold 0.5, ALASKA unseen k1 miss rises by **17.81 percentage points**, and LSDIR test k1 miss by **8.68 points**. The mixed model shifts toward fewer false alarms and more missed weak stego images. A later threshold change must be calibrated on separate validation data, not tuned on these test panels.

### ALASKA camera checks

These are the **test** splits, not the training photos. Seen cameras occur in mixed training, but test photos are held out. Unseen cameras are excluded from mixed training. Each camera has the same pair count in both predictions. `!` marks mixed-model FA **above 10%** at threshold 0.5; this is a diagnostic flag, not a predeclared pass criterion. Per-camera P_E is optimized on that camera alone, so use it for diagnosis, not for choosing a deployment threshold.

| Seen camera | Pairs | Wide-R2 FA % | Mixed FA % | Wide-R2 P_E | Mixed P_E |
| --- | ---: | ---: | ---: | ---: | ---: |
| Sigma_SD1Merrill | 86 | 97.7 | **31.4 !** | .477 | .384 |
| Sony_Alpha6000 | 97 | 49.5 | **24.7 !** | .242 | .155 |
| LG_G7fit-LMQ850 | 94 | 66.0 | **24.5 !** | .282 | .149 |
| Sigma_SD10 | 96 | 87.5 | **24.0 !** | .359 | .312 |
| Canon_EOS-100D | 87 | 62.1 | **23.0 !** | .236 | .144 |
| HTC_OneA9 | 92 | 55.4 | **22.8 !** | .223 | .174 |
| Kodak_Z990 | 93 | 51.6 | **21.5 !** | .269 | .226 |
| HTC_OneM9-OPJA100 | 99 | 52.5 | **20.2 !** | .263 | .182 |
| Nikon_D610 | 90 | 65.6 | **17.8 !** | .267 | .117 |
| Panasonic_FZ28 | 90 | 43.3 | **17.8 !** | .211 | .139 |
| Nikon_D810 | 96 | 52.1 | **17.7 !** | .250 | .151 |
| Apple_IpadPro-2ndGen | 94 | 42.6 | **17.0 !** | .266 | .229 |
| Leica_M9 | 83 | 47.0 | **16.9 !** | .199 | .127 |
| Huawei_Honor8-FRDL09 | 98 | 45.9 | **13.3 !** | .204 | .128 |
| Nikon_D5200 | 91 | 48.4 | **12.1 !** | .220 | .082 |
| Canon_EOS20D | 105 | 56.2 | **11.4 !** | .267 | .129 |
| Nikon_D5600 | 97 | 35.1 | **11.3 !** | .165 | .149 |
| Nikon_D3000 | 95 | 33.7 | **10.5 !** | .195 | .126 |
| Nikon_D7100 | 95 | 49.5 | **10.5 !** | .158 | .089 |
| LG_G5-H840 | 97 | 49.5 | **10.3 !** | .247 | .144 |
| Canon_EOS-500D | 99 | 42.4 | **10.1 !** | .273 | .202 |
| Samsung_GalaxyS8-SM-G950F | 109 | 45.0 | 9.2 | .216 | .119 |
| Apple_IpadPro-1stGen | 88 | 35.2 | 9.1 | .188 | .097 |
| Pentax_K10D | 91 | 33.0 | 8.8 | .192 | .110 |
| Panasonic_DMC-GM1 | 98 | 29.6 | 8.2 | .179 | .107 |
| Nikon_1-AW1 | 99 | 38.4 | 8.1 | .232 | .121 |
| Canon_EOS700D | 96 | 58.3 | 7.3 | .198 | .089 |
| Pentax_ist-DS | 93 | 24.7 | 5.4 | .161 | .081 |

| Unseen camera | Pairs | Wide-R2 FA % | Mixed FA % | Wide-R2 P_E | Mixed P_E |
| --- | ---: | ---: | ---: | ---: | ---: |
| Panasonic_DMC-TZ60 | 376 | 46.0 | **17.6 !** | .253 | .181 |
| Huawei_P20lite-ANE-LX1 | 413 | 45.8 | **16.9 !** | .219 | .150 |
| Samsung_GalaxyS9plus-SM-G965F | 410 | 43.7 | **16.6 !** | .263 | .178 |
| Sony_A7R | 402 | 53.7 | **16.4 !** | .245 | .150 |
| Canon_EOS-60D | 396 | 52.0 | **15.4 !** | .245 | .150 |
| Ricoh_GRDigital-III | 407 | 44.7 | 9.3 | .205 | .107 |
| Google_Pixel-XL | 390 | 44.4 | 9.0 | .188 | .078 |
| Pentax_K50 | 359 | 49.9 | 8.6 | .214 | .091 |

Even after the large aggregate improvement, **21/28 seen** and **5/8 unseen** cameras exceed 10% FA at threshold 0.5. Sigma_SD1Merrill also retains P_E `.384` and Sigma_SD10 P_E `.312` on seen tests. These cameras merit a targeted error review before calling the detector camera-robust. Small per-camera k1 samples make camera-level k1 claims unstable; panel-wide k1 counts are in the table above.

## Validation behavior

The following are selected rows from `logs/csv/run/metrics.csv`; the file has all 30 epochs. The source columns are diagnostics. **Only combined P_E selected the checkpoint.** Epoch numbers here are zero-based.

| Epoch | Combined P_E | RAISE P_E | ALASKA P_E | LSDIR P_E | Note |
| ---: | ---: | ---: | ---: | ---: | --- |
| 0 | .2115 | .1274 | .2912 | .2320 | Cold first epoch |
| 3 | .0763 | .0306 | .1899 | .1050 | Initial improvement |
| 7 | .0717 | .0286 | .1814 | .0880 | Before spike |
| 8 | .1124 | .0517 | .1805 | .1047 | Combined spike |
| 9 | .0698 | .0285 | .1772 | .0907 | Recovery |
| 10 | .0956 | .0365 | .1932 | .1017 | Spike |
| 11 | .0622 | .0208 | .1638 | .0833 | Recovery |
| 12 | .0977 | .0478 | .1844 | .0997 | Spike |
| 14 | .0832 | .0390 | .1938 | .0873 | Spike |
| 15 | .0658 | .0252 | .1596 | .0773 | Recovery |
| 20 | .0616 | .0218 | .1606 | .0763 | Late plateau |
| 24 | .0599 | .0216 | .1544 | .0773 | Close to best |
| **25** | **.0574** | .0207 | .1564 | .0757 | **Saved best** |
| 26 | .0624 | .0224 | .1684 | .0793 | Rebound |
| 27 | .0587 | .0214 | .1531 | .0760 | Better ALASKA; worse combined |
| 29 | .0660 | .0246 | .1619 | .0787 | Last checkpoint, not best |

Train loss fell `.6455` → `.2736` and combined validation loss `.4733` → `.1746`. At the best epoch, validation losses were RAISE `.06956`, ALASKA `.45884`, LSDIR `.22151`, and combined `.17230`. ALASKA is the hardest validation source throughout. Combined spikes occur with both RAISE and new-source changes; they do not imply training stopped. The best checkpoint is a *combined* compromise: epoch 27 had lower ALASKA P_E (.1531 versus .1564) but higher combined P_E (.0587 versus .0574). The final full validation did not silently omit any source.

## Runtime, cold reads, and cost

The 30-epoch fit took **2,373.352 s (39.56 min)**; the whole notebook took **2,719.598 s (45.33 min)**. Runner exit was 03:36:23 UTC and delivery ended 03:37:26 UTC. The GCE start operation began 02:50:18 UTC; guest shutdown was 03:37:26 UTC, about **47.13 VM min or $1.28 at $1.63/h** as an operation-time estimate, not a billing record. The failed earlier attempt cost about seven additional VM minutes (~$0.20, also an estimate). These figures do not include dataset preparation or the separate PC baseline evaluation.

The table shows mean seconds from each job's `epoch_timing.csv`. Early warm means zero-based epochs 1–23, late means 24–29. The old wide-R2 job used **3,372** training steps/epoch and standard-only validation. The mixed job used **2,813** steps/epoch and capped mixed validation, so compare *rates and phases*, not raw train time as if work were equal.

| Phase | Wide-R2 early | Wide-R2 late | Mixed early | Mixed late |
| --- | ---: | ---: | ---: | ---: |
| Train first-batch wait | .010 | .039 | .260 | .324 |
| Train first → last batch | 61.888 | **101.890** | **51.602** | **51.576** |
| Train steps/s (steps / span) | 54.5 | 33.1 | 54.5 | 54.5 |
| Validation first-batch wait | **18.773** | **17.999** | **1.392** | **1.313** |
| Validation first → last batch | 13.163 | **29.236** | 19.370 | 18.231 |

The mixed job's epoch 0 was an outlier: first train-batch wait **83.88 s**, train steps **109.44 s**, first validation wait **21.74 s**, validation compute **22.26 s**. While it warmed, the first displayed epoch-0 rate was only **7.4 steps/s**, with host I/O wait **41.9%**, physical reads **275.7 MiB/s**, and page cache still growing. By epoch 3 it reached **55.4 steps/s**, GPU utilization **100%**, I/O wait **0.04%**, reads **7.2 MiB/s**, and page cache ~58 GiB. Later disk reads and I/O wait were near zero. The sampler groups selected tiles by photo and decodes each source photo once per group, not once per tile. The cold disk plus first compile explains the *temporary* slowdown; the timing data cannot separate those two costs exactly. Mixed warm training kept the old early **~54.5 steps/s**, not a claimed model-throughput gain.

Persistent validation workers cut warm first-batch wait from the old **~19 s** to **~1.4 s**, even with mixed source decoding. Unlike wide-R2, the mixed job had **no late slowdown** in epochs 24–29: train spans stayed ~51.6 s and validation compute ~18–19 s. This is observed, but it does not establish which code or host change removed the earlier slowdown. The mixed validation compute spans are longer than old early standard-only validation, and the two evaluation sets differ. The `host.csv`, `nvidia.csv`, and `epoch_timing.csv` files are saved under each job for a closer phase review.

## Decision and follow-up

- Retain wide-R2 for the unchanged standard test. Keep this mixed checkpoint for research on ALASKA/LSDIR and unseen ALASKA cameras, with a clear fixed-threshold miss warning for k1.
- Review mixed-model false alarms on Sigma_SD1Merrill, Sigma_SD10, and the five flagged unseen cameras. Use held-out validation to study any threshold change; **do not tune against these tests**.
- If standard-source quality and k1 sensitivity must be kept, test a new mix, loss, or calibration proposal as a separate experiment. This report does **not** authorize another G4 run.
