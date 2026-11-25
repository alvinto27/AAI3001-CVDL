## 2026-10-15 — G4 priming run and PC observation

### Did

- The user started the Pi before G4 run `g4-prime-1`. It used decode,
  32 train workers, 16 evaluation workers, compile default, and
  `V3_PRIME_NEXT_EPOCH=on`. It ran commit `ad8c754` through `origin/yx`.
  The user's live `notebooks/04` run and `checkpoints/pc-prime-test` were
  not changed.
- Worker_2 ran a PC slowdown diagnosis after restarting the container. It
  profiled 600 steps from committed notebook classes in a scratch copy from
  HEAD. The user's shifted top cell made the fixed cell 14/19 extraction in
  `step_profile.py` fail, so the scratch copy was used.
- Worker_2 also ran the committed notebook headlessly with
  `scripts/run_notebook.py`: decode, 6 workers, compile, priming off,
  `V3_DIAG_WORKERS=0`, and 20% train-photo fraction. It stopped by SIGTERM
  after epoch 2 on purpose.
- The user ran full-data two-epoch headless PC run `pc-headless-2ep` from a
  scratch copy of the committed notebook. Only `EPOCHS` changed to 2. It
  used decode, 6 workers, compile, priming off, `V3_DIAG_WORKERS=0`, and
  `scripts/run_notebook.py`. Output was `runs-out/pc-headless-2ep/`, 14 MB,
  git-ignored.
- The PC ALASKA subset download completed with 9,723 of 9,723 files,
  15,311,632,398 B, 4,220 s, 2.304 files/s, 3.628 MB/s, and 0 failures.
  The validation logs are `logs/alaska-download-v1.log` and
  `logs/alaska-validation-v1.log`.
- Worker_2 ran the suspicion-map demo from `g4-prime-1` best epoch 8 with
  no training. It used `scripts/map_demo.py` (`0a0e623`) and
  `scripts/map_demo_gn_check.py` (`d5ccde4`). Outputs are in the
  git-ignored `runs-out/map-demo/` folder. The checkpoint SHA-256 is
  `741c9db0871b74acf6da4ad2d43f2ab0326b8acc531259d268f9371821f0848d`.
- The first demo used random settings per photo, so its photos were not
  comparable. Settings were lowest/plane bit mode, k 1-8, replace/match
  method, contiguous/stride/random placement, and random_bytes/letters
  payload. Draws were RAISE D7000 lowest k=7 match, RAISE D90 lowest k=5
  match, DIV2K 0801 plane k=1 match, DIV2K 0802 plane k=6 match, ALASKA
  Pixel XL lowest k=3 replace, and ALASKA Canon EOS700D plane k=4 replace.
- Worker_2 ran map demo v2 with fixed settings in `scripts/map_demo.py`
  (`da97765`). Outputs are in `runs-out/map-demo-v2/`.
- Commits `e68b958`, `57c24a1`, `e1186fd`, `00ce10c`, `fb3dfb7`, `f9ab8b3`,
  `a30748d`, and `16c49d8` added notebook 05, no-clip support, the G4
  bootstrap split, map checks, bpp-1 scoring, the R2 schedule, GPU-generator
  design, and batch/precision speed-matrix support.
- The user ran the no-normalisation G4 R1 job `g4-v4-r1b`; R2 was later
  launched as `g4-v4-r2` and is still Pending.
- Worker_2 new-data code `e592cce`: a strict ALASKA uncompressed RGB8 TIFF
  reader, the LSDIR PNG path, a per-record `trim512` grid (RAISE/DIV2K keep
  `overlap_last`; the grid policy is recorded per photo), and
  `scripts/build_new_source_manifests.py`. The builder writes
  `photo_manifest.csv`, `split.csv`, `provenance.csv` and `build_record.json`,
  checks size, SHA-256 and duplicates, and refuses to overwrite. Also added:
  opt-in `V3_SOURCE_MIX` and `V3_TILES_PER_EPOCH` per-epoch sampling,
  separate extra panels (ALASKA val/test_seen/test_unseen, LSDIR val/test),
  and `docs/archive/plan-new-data-code.md` with the LSDIR selection contract
  (`lsdir_id, shard, path, split, width, height, tiles512, bytes, sha256`).
- Worker_2 capped checkpoint validation `1b54858`: `V3_CHECKPOINT_VAL=capped`
  is the default only when `V3_SOURCE_MIX` is set. It uses a fixed seeded
  sample of up to 5,000 RAISE validation pairs (no DIV2K), all ALASKA val
  pairs and up to 1,500 LSDIR val pairs. The ordered pair CSV and its SHA-256
  go to the run outputs; a resume with changed pair IDs is refused.
  Per-source validation P_E and loss are logged each epoch (CSV and
  TensorBoard).
- The ALASKA download finished at 23:26 +08: 70,282/70,282 files,
  110,550,230,800 B, 0 failures, all files verified.
- Worker_3 final ALASKA selection v2 `5ca650b`: `data/alaska/selection_v2.csv`
  (18,200 rows, SHA-256
  `437fcfef817689cd1af5347f08f948cfddfad6523667afd9dace37eb5ef1ff85`, the
  same bytes on a second run), `scripts/alaska_select_v2.py` and
  `docs/active/alaska-selection-v2.md`. Pi scan with `--verify`: 70,282 files
  re-hashed equal to `SHA256SUMS`, 0 bad. PC v1 copy: 9,723 re-hashed, 0 bad.
  All 80,005 IDs are covered; 0 eligible IDs are missing. All files are
  uncompressed RGB 8/8/8.
- Worker_3 commit `90425cb` (train-only list, run-based blocks) reached
  `origin` through a later push without a separate review. The Orchestrator
  reviewed it afterwards; it agrees with the decisions.
- The Orchestrator wrote the session handoff `docs/archive/handoff-2026-10-15.md`
  (`f679b44`). Scribe stopped before it recorded the last two updates; the
  Orchestrator recorded them here and committed the first version of
  `docs/archive/research-checklist.md` (written by Scribe).

### Found

- G4 booted at 00:59:42Z. The runner ended at 01:18:40Z, Pi delivery at
  01:19:46Z, and the VM was offline at 01:21:01Z. Status was done, exit 0,
  all 15 cells passed, and cleanup was clean.
- Runner time fell from 1,602.266 s in `g4-full-1` to 1,137.973 s:
  26:42 to 18:58, a saving of 464.3 s.
  - Cell 29 fell from 499.524 s to 87.480 s, saving 412.0 s.
  - Fit fell from 967.311 s to 914.918 s, saving 52.4 s.
- Epochs 1-9 reached step 75 in 1 s instead of 11-12 s. Train-only epoch
  time fell from 60-62 s to 49-51 s. Steady speed from step 150 to 3,300
  was 67.84 to 67.19 it/s, so steady speed did not change.
- Validation wall time per epoch rose from 27-28 s to 32-33 s because priming
  adds about 4-5 s. The final epoch took 27 s.
- Service peak memory was 79,963,168,768 B (74.47 GiB) of 176 GiB. There
  was no out-of-memory event.
- Test accuracy was 0.9828, AUC 0.997728, and P_E 0.016475. RAISE P_E was
  0.011410 and DIV2K P_E was 0.054560. The earlier `g4-full-1` values were
  0.9833 / 0.997603 / 0.016247, with RAISE 0.010948 and DIV2K 0.052524.
  The best epoch was 8, with validation P_E 0.0147 and AUC 0.9973.
- Small metric differences are expected from non-deterministic compiled GPU
  math. The stream exactness proof is the PC hash test.
- In the user's live PC `pc-prime-test`, decode, 6 workers, compile, prime on:
  epoch 0 ran at 9.86 it/s; epoch 1 at about 10.67 it/s, versus the journal
  value of 13.54 it/s for decode, 6 workers, compiled. Workers used 40-50%
  CPU, CPU was about 70% idle, and the GPU reached 3,176 MHz and 100% GFX
  activity at about 235 W. Hotspot temperature was 79 C. Validation took
  about 1 min; decode had taken about 47 s before. Priming contention is
  expected. The cause is not known. Worker_2 will profile after the run.

#### PC slowdown diagnosis

- Profiled steady speed, in it/s, with cuDNN benchmark false / true:

  | Case | False | True |
  | --- | ---: | ---: |
  | Raw fixed | 13.36 | 13.29-13.30 |
  | Lightning fixed | 13.41 | 13.28 |
  | Lightning decode, 6 workers | 13.84 | 13.76 |
  | Lightning cache, 4 workers | - | 13.74 |
  | Raw cache, 4 workers | - | 13.79 |

  Progress bar and callback toggles at decode/6 gave 13.81-13.84 it/s.
  GPU power was 252-254 W at 3,100-3,145 MHz, with a 255 W cap, AUTO
  performance, and no throttling.
- The headless notebook reached 13.64 it/s: epochs 1 and 2 were at step 150
  in 14 s and step 600 in 47 s. This matches the journal value of 13.54.
- No system change was found: kernel 7.2.9 since October 11, the same
  container image, no ROCm/mesa/amdgpu package change, and no
  MIOPEN/HSA/HIP/ROCM overrides.
- The VS Code Python extension changed to 2026.10.0 and `python-envs` to
  1.40.0 at 09:03 +08 before the slow live runs. The Jupyter extension
  2025.9.1 did not change. This is a candidate, not proof. VS Code rendering
  on the same GPU is another candidate.

#### Full-data headless PC run

- The run completed with status done, exit 0, all 15 cells, and clean
  cleanup. It ran from 02:29:09Z to 02:50:30Z (1,280.949 s).
- Steady training from step 150 to 3,300 was 13.816 it/s in both epochs.
  Time to step 75 was 17 s in epoch 0 and 10 s in epoch 1. Train-only
  epochs took 259 s and 247 s. Validation took 43 s per epoch, or 49 s wall
  time with callbacks.
- Cell times: fit 612.928 s, test evaluation 188.401 s, serial diagnostics
  458.332 s, and Notebook 02 2.910 s. GPU in epoch 1 was 100% busy at
  252-254 W, 3,095-3,103 MHz, with hotspot 85-89 C. Container memory was
  about 15.3 GB and host available memory about 13 GiB.
- This speed-only run used StepLR step size 1, so its scores are not
  comparable with 10-epoch runs.

#### New-source checks

- ALASKA#2 has 80,005 images from about 50 camera models, according to
  `ALASKAv2Base_by_DSC.txt`. Twenty sampled various-size TIFF files were
  0.79-3.1 MB at 512 x 512 to 1024 x 1024, mean 1.69 MB. The full set is
  estimated at about 135 GB. The 512 set files are 786,572 B each, about
  63 GB total. ALASKA includes a Nikon D90, also present in RAISE.
- LSDIR's CVPR 2023 workshop paper reports 84,991 train, 1,000 validation,
  and 1,000 test images. It describes about 4,000 Flickr photos shrunk 4x
  with Lanczos and cropped to multiples of 48. Hugging Face access needs a
  login and accepted terms. Total size is not known.
- Current 512 data has 53,950 train tiles: RAISE 45,674 (85%) and DIV2K
  8,276 (15%), from 1,381 photos (701 and 680). Validation has 11,238 tiles
  and test has 10,956.

#### Source download and throughput checks

- ALASKA download speed, various-size TIFF: Pi Wi-Fi, one connection with a
  0.2 s pause 0.380 MB/s; 4 connections 1.191 MB/s (64 files, 95.449 s);
  8 connections 1.503 MB/s (64 files, 68.872 s). Sixteen stopped after one
  curl 28 timeout. PC Wi-Fi, 8 connections: 3.973 MB/s (25.699 s) and
  3.592 MB/s (27.517 s), 0 errors. The PC reaches 21.5 MB/s from Cloudflare
  on one connection; the 204 ms round trip to the ALASKA server is the limit.
- The Pi uses Wi-Fi with `eth0` down. Its `/srv` is a 1 TB Crucial MX500
  USB SSD with ext4.
- The ALASKA index SHA-256 is `cd4bf8b9...0e91a54`. Its count columns are
  wrong for Canon_EOS-60D (3,678 versus range 2,750) and Canon_EOS-500D
  (1,872 versus 2,800). The ranges total 80,005.
- LSDIR Hugging Face has shards 00-16 at 8.07-9.91 GB, about 155 GB total,
  with sizes matching the GitHub issue #5 SHA list. It also has train_x2/x3/x4
  (not used), `val1.tar.gz` at 765 MB, and no test archive. The released HR
  PNGs are already shrunk 4x by the authors and will be used unchanged
  (plan fix `ee300d0`).
- Container Pillow decoded all 9,723 ALASKA files successfully. They are
  TIFF, RGB, 8-bit, and both sides are at least 512. All 9,723 SHA-256SUMS
  entries match. The 16 shapes from 512, 640, 720, and 1024 on each side
  have 565-639 files each. Files are in the git-ignored
  `Datasets/alaska/tiff_various_color/`; `selection_v1.csv` has SHA-256
  `a6570d3191b074446fba77ee4e2dc62dd166a54d9cc0ae04735b02617cfbe96b`.
- Unattended Pi services `cvdl-lsdir` and `cvdl-alaska` are in `scripts/pi/`
  (commits `33e5f96`, `0e37f28`, `c3ca8be`, `0762cfc`) and deployed copies
  match the repository. Linger is on; services restart after 60 s on failure,
  guard 50 GiB free space, and write status to
  `/srv/datasets/{lsdir,alaska}/STATUS`. Restart tests resumed LSDIR after
  a mid-shard kill without a duplicate archive. ALASKA resumed after a kill
  with partial files, rechecked 51 SHA-256 values, and continued.
- The Pi moved from 2.4 GHz channel 6 to 5 GHz at 5,320 MHz. Its rate rose
  from 1.28-1.83 MB/s to 8.5-9.9 MB/s; the revert was cancelled and the
  setting persists. A backup is in `/srv/datasets/net-backup/`. At 13:19Z,
  LSDIR was 3/18 verified (27,934,654,536 B); ALASKA was 300/70,282 with
  0 failures. The morning status command reads both STATUS files over SSH.
- For tiled map inference, chunk starts use multiples of 32, with at least
  about 48 px margin; 64 px is used. Keep centre cells and drop duplicate
  inference cells. GroupNorm means per-chunk statistics at 512 match
  training, but whole-photo statistics do not. Keep a raw map for the
  verdict and a photo-normalised median/MAD map for location. Normalisation
  erases a photo-wide random-placement message.
- Orchestrator analysis: 256 has 3.75 times the train images per epoch
  (405,144 versus 107,900). Older PC numbers rose from 129 to 309 images/s
  for decode and 418 to 1,326 for cache when moving 512 to 256. Per-tile
  work, not decoding, limits 256. PyAV is not adopted.

#### Tile overlap and spatial map findings

- On 6,934 downloaded ALASKA TIFFs, with sides approximated as squares from
  file size, `overlap_last` at 512 gave 3.81 tiles per photo and each pixel
  appeared 2.07 times on average. A non-overlapping 512 grid gave 1.18
  tiles per photo and used 59% of pixels.
- At 256, overlap gave 10.92 tiles per photo (1.39 times the non-overlapping
  count); non-overlap gave 6.03 tiles per photo and used 75% of pixels.
  For a 573-pixel side, two 512 tiles share 88% of their pixels.
- Overlap counts the same changed pixels more than once in Mode 2 bags,
  makes tile tests look more certain than they are, and skews the even
  per-camera mix.
- Notebook 04's XuNet is fully convolutional: convolution layers, five
  average pools of 2, `x.mean(dim=(-2, -1))`, and `Linear(128, 2)`. It accepts
  any photo size. There is one 128-value cell per 32 x 32 pixels, and each
  cell sees about 48 x 48 pixels.
- Because `Linear(mean) = mean(Linear)`, the current checkpoint already
  gives a per-cell suspicion map and averages it. Maps can be drawn without
  retraining. The generator supports contiguous, stride, and random
  placement; a clear localized region appears only for localized placement.

#### Suspicion-map demo

- Photos were RAISE `r00679daet` (D7000) and `r0667a51ft` (D90), DIV2K
  test `0801` and `0802`, and ALASKA `00001` (Pixel XL) and `30827`
  (Canon EOS700D). Full-photo embedding used `lsb_engine`, clean images,
  contiguous and random placement, and 0.05, 0.275, and 0.5 bpp. Chunked
  inference used 512 chunks, starts on multiples of 32, a 64 px margin, and
  kept centre cells.
- The rebuilt model was exact: forward versus classifier(mean(features))
  had max absolute difference 0; versus mean per-cell logits it was
  1.19e-6.
- Localization worked without retraining: contiguous cell AUC was 0.94-1.00
  in all 18 cases. For D7000 at 0.05 bpp, 154 changed cells had mean +173.05,
  15,554 unchanged cells had mean -5.13, and AUC was 0.99995.
- The photo verdict failed. Cell means dilute small regions: D7000 contiguous
  0.05 bpp had photo mean -3.38. Top-1% and max were positive on all 6 clean
  photos; clean D7000 max was +156.06. The tile-trained model does not
  calibrate single-cell scores.
- Random placement at 0.05 bpp or more changed almost every cell; cell AUC
  was undefined in 17 of 18 cases. Mean verdicts were about +20 on RAISE,
  but negative for DIV2K 0801 at 0.05 and 0.275 and ALASKA Pixel XL at 0.05.
- Chunked and whole-photo maps differed; DIV2K 0801 contiguous 0.5 had max
  absolute difference 148.05 and correlation 0.909.
- GroupNorm changed unchanged cells in chunks containing changed pixels by
  mean absolute 0.63-11.70, max 140.3, while other chunks changed by exactly
  0 in all 16 of 16 cases with such chunks. Replacing each GroupNorm layer's
  statistics with clean-chunk statistics removed the band in 14 of 18 cases;
  the others were cells next to the strip within the receptive field. Frozen
  clean statistics gave cell AUC 0.934-1.000, but need the clean cover and
  are diagnostic only.

#### Suspicion-map demo v2

- The v2 run used the same six photos, checkpoint, and chunked inference.
  It produced 12 figures with fixed P(stego) 0-1 and -5 to +5 MAD colour
  bars, plus `metrics-v2.csv` with 84 rows. Group A was lowest k=1,
  match, random_bytes. Group B was lowest k=4, match, random_bytes.
  P(stego) is sigmoid(mean cell logit), the current photo verdict.
- Clean P(stego): RAISE 0.003268 / 0.000451, DIV2K 0.011232 / 0.011580,
  ALASKA Pixel XL 0.009859, and ALASKA Canon EOS700D 0.554315 (false
  positive).
- Group A random k=1 at 0.05 bpp missed all six (P 0.003-0.07, Canon 0.55).
  At 0.275, RAISE was 0.999995 / 0.998437, DIV2K 0.154259 / 0.266355,
  and Pixel XL 0.032288. At 0.5, RAISE was 1.0, DIV2K 0.883698 / 0.948439,
  and Pixel XL 0.163922.
- Group A contiguous cell AUC was 0.80-0.996 on five photos; Canon was
  0.152-0.205 with changed cells scoring lower. Group B random detected at
  0.275 and 0.5 on all photos except Canon, at 0.84 / 0.97. Group B
  contiguous cell AUC was 0.96-1.0 except Canon at 0.20 / 0.86 / 0.71.
- Group B RAISE contiguous verdicts were lower than Group A despite higher
  cell AUC; for r00679daet at 0.5 bpp they were 0.149816 versus 0.816317.
  The hypothesis that louder strips shift GroupNorm statistics and dim the
  strip is not proven.
- The k=1 random result repeats the source weakness: RAISE is detected at
  0.275 bpp, while DIV2K and unseen ALASKA Pixel XL are not. Clean Canon
  EOS700D is a smooth out-of-distribution image, with a false positive and
  inverted localization.

#### XuNet v4 no-normalisation results

- PC sanity run, reported by Worker_2: two full-data epochs with no
  normalisation had validation P_E 0.0758 / 0.0438, versus v3c 0.0374 /
  0.0294. There was no NaN. The pre-clip gradient norm was above 1 on
  83-88% of logged steps, and loss spikes recovered within about 50 steps.
- G4 R1 `g4-v4-r1b` completed from the local status record at 06:59:46Z to
  07:30:35Z, exit 0, last cell 31. It used notebook 05, no normalisation,
  LR 3e-4, one warm-up epoch, clip 1, 20 epochs, 32 workers, 16 evaluation
  workers, and priming on. Fit took 1,627.427 s and the full elapsed time
  was 1,848.801 s.
- R1 best epoch 19 had validation P_E 0.0202 and AUC 0.9949. Validation P_E
  spiked to 0.0590 at epoch 5. Test P_E was 0.022864 and AUC 0.989898,
  versus `g4-prime-1` v3c 0.0165 and 0.9977. R1 breakdowns were k=1 0.0718,
  RAISE 0.0179, and DIV2K 0.0615; v3c values were k=1 0.0579, RAISE 0.0114,
  and DIV2K 0.0546. The best checkpoint filename starts `b9af8dba`.
  Training loss stayed above validation loss and plateaued near 0.13,
  indicating underfitting.
- R1 map checks found no shift outside changed chunks in 36 of 36 cases.
  Chunked versus whole-photo maps differed by at most 0.00064. The clean
  ALASKA EOS700D false alarm fell from 0.554 to 0.037. However, k=1 random
  detection was about 0.03 even at 0.5 bpp on ALASKA; detection was mainly
  in smooth regions, so clean hot spots could hurt a top-k verdict.
- The bpp-1 extra test used `configs/test-bpp1.yaml` from commit `fb3dfb7`.
  The standard test hash stayed `fdd989f7...837`, and the scorer reproduced
  P_E 0.0165 for v3c and 0.0229 for R1. At bpp 1, v3c P_E was 0.0125 and
  R1 was 0.0187. k=1 missed at bpp 0.75-1.0: contiguous 1%, random 3%,
  and stride 12-19%; 37% of stride stego pairs were clipped by capacity.
- R2 `g4-v4-r2` started at 09:40Z and ended at 10:32:03Z, with exit 0.
  It used LR drops 5:0.3 and 20:0.1, training k=1-4, 30 epochs, and the
  bpp-1 extra test. Fit took 2,868.338 s. Best epoch 19 (1-based epoch 20)
  had validation P_E 0.0227 and AUC 0.9939. Validation P_E spikes were
  0.0391 at epoch 4, 0.0407 at epoch 9, and 0.0369 at epoch 15, so spikes
  continued at LR 9e-5. Validation still includes k=5-8, which R2 does not
  train on.
- Standard test P_E was R2 0.0250, R1 0.0229, and v3c 0.0165. The bpp-1
  extra-test P_E was R2 0.0199, R1 0.0187, and v3c 0.0125.
- At threshold 0.5, standard-test clean false-alarm rates were v3c 0.0071,
  R1 0.0172, and R2 0.0233. P_E over k=1-4 stego only was v3c 0.0287,
  R1 0.0367, and R2 0.0303. Miss rates were:

  | Setting | v3c | R1 | R2 |
  | --- | ---: | ---: | ---: |
  | Lowest k=1 | 0.1584 | 0.1804 | 0.1041 |
  | Plane k=1 | 0.1467 | 0.1867 | 0.1240 |
  | Lowest k=8 | 0.0112 | 0.0309 | 0.0463 |
  | Plane k=8 | 0.0015 | 0.0015 | 0.0183 |

- R2 bpp-1 k=1 random miss rates were 0.604, 0.125, 0.032, 0.043, and
  0.009 for the five increasing bpp bins from 0.05-0.15 through 0.75-1.0.
- Training k=1-4 gave R2 the best k=1 detection of the three runs, but it
  caused more clean false alarms and more misses at untrained k=5-8. R2
  changed both the LR plan and the k range, so the cause is not separated.
- R1 `nvidia.csv` had mean GPU utilisation 37.4% over 31 samples, with
  12 samples at 0%. The GPU-generator PC profile found photo decode at 83%
  of CPU work and embedding at about 8%; estimated gain was 0-25%.
- Reported G4 failure `g4-v4-r1` at 06:42Z: the old installed `job.sh`
  ignored `V3_NOTEBOOK` and ran notebook 04. It failed at cell 2 after about
  7 s because `NORM` was not in `{batch, group}`. Later boots skipped the
  recorded job ID. New job ID `g4-v4-r1b` fixed the run. Commit `e1186fd`
  split `job.sh` into a stable bootstrap and checked-out `job_run.sh`.
  Pending: one maintenance boot must install the bootstrap.
- Reported Pi download state: LSDIR completed 18/18 files,
  155,804,436,196 B, 0 failures, at 17:43 local. ALASKA reached
  30,800/70,282 at 18:15 local with 0 failures.
- The user installed gcloud CLI 589.0.0 in the PC user account. The
  unversioned official archive matched SHA-256
  `817026aec0adea0e2ee8ed3a1ba3eabfe9e5177581fbe16595a8fe495a601300`;
  the versioned URL did not match its published digest.

#### G4 batch and precision speed matrix

- Six hundred timed steps followed 50 warm-up steps for each configuration.
  The run used decode, 32 workers, compile default, and no validation or
  checkpoints. The complete attempt had six of six configurations. A first
  attempt started at 11:02:01Z and was Spot-preempted at 11:08:08Z; the
  second ran from 11:14:57Z to 11:23:28Z and shut down normally. Total VM
  time was about 14 min 38 s, about $0.40 at the $1.63/hour estimate.
- Results:

  | Batch / precision | Steps/s | Images/s | Loader wait ms/step | GPU+transfer ms/step | Peak torch MiB |
  | --- | ---: | ---: | ---: | ---: | ---: |
  | 32 / fp32 | 81.74 | 2,615.72 | 0.797 | 11.437 | 1,925 |
  | 32 / bf16 | 84.71 | 2,710.68 | 2.696 | 9.109 | 1,072 |
  | 64 / fp32 | 40.21 | 2,573.36 | 3.566 | 21.304 | 3,781 |
  | 64 / bf16 | 39.73 | 2,542.93 | 8.539 | 16.629 | 2,080 |
  | 128 / fp32 | 20.87 | 2,671.68 | 0.865 | 47.045 | 7,493 |
  | 128 / bf16 | 21.07 | 2,696.26 | 16.227 | 31.246 | 4,096 |

  GPU-utilisation samples were few: 3-14 per configuration.
- Analysis: images/s was about 2,540-2,710 for every configuration, so a
  larger batch gave no gain. At fp32, GPU step time limited the run; at batch
  128 it was 47.0 ms versus 0.9 ms loader wait. BF16 cut GPU time to 31.2 ms,
  but loader wait became 16.2 ms. CPU data supply and fp32 GPU speed were
  both near 2,700 images/s. Training steps are about half of a full-run
  epoch: R2 fit was 2,868 s / 30 = 95.6 s per epoch, while 3,372 steps at
  about 74 steps/s is about 46 s. Validation, checkpoints, and epoch
  transitions take the other half.

#### Epoch overhead and monitoring

- Worker_2 recorded the read-only epoch plan in `docs/archive/plan-epoch-overhead.md`
  (`679597f`). Commit `f777be2` added `scripts/host_monitor.py` and notebook
  05 epoch timers. The standard test hash remains
  `fdd989f7...837`; training code was not changed.
- Normal G4 epochs in R1 and early R2 were about 79 s: train about 47 s,
  validation 703 batches about 12 s, an unexplained train-to-validation or
  epoch-end transition about 20 s, and checkpoint about 0.2 s (0.25 MiB).
  The first epoch was about 121 s, including 43 s compile warm-up.
- The earlier R2 overhead estimate was corrected. R2 epochs 24-29 had train
  101.3 s versus 47.1 s, validation 27.2 s versus 12.3 s, and GPU samples
  about 20% / 163 W versus 38% / 218 W. The cause is unknown. QUICK PC
  timers show about 5 s waiting for the first validation batch each epoch;
  this is a possible part of the 20 s gap, not yet confirmed on G4.

#### Wide XuNet

- Worker_2 added the opt-in wide model in `docs/archive/plan-wide-xunet.md`
  (`4f5fa5f`). `V3_MODEL=xunet|xunet_wide` and `V3_WIDTH=4` keep the same
  front end and 32 px cell grid. No-norm parameters are 18,941 versus
  594,381; counted work is 1.56 versus 5.03 GFLOP per 512 tile.
- PC synthetic batch 32 was 6.706 versus 5.115 fp32 steps/s and 1,808 versus
  2,762 MiB; BF16 was 16.134 versus 13.259 steps/s. The PC QUICK run used
  batch 8, 6/4 workers, passed in 367.313 s, and all values were finite.
  A PC QUICK run with G4 settings, 32 workers and batch 32, was killed with
  exit 137 and `oom_kill=1`; G4 job files are not for the PC.
- G4 wide run `xunet_v4_clip3_nonorm_lr0.0003_warm1_drops5x0.3-20x0.1_k1to4_e30_wide-w4_r2-20261015T121753Z-29499`
  completed with exit 0 and no preemption. GCE started at 12:18:03Z and the
  guest stopped at 13:19:01Z. Notebook time was 3,491.374 s, fit was
  3,235.450 s, and VM time was about 60.97 min, or $1.66 at $1.63/hour
  (estimate). Results were delivered to the Pi.
- Standard test results, with 10,956 pairs, R2 -> wide: notebook P_E
  0.025009 -> 0.019533, AUC 0.989593 -> 0.996092; at threshold 0.5,
  clean false alarm 2.327% -> 1.396%, k=1-4 P_E 0.03029 -> 0.02373,
  k=1 lowest miss 10.411% -> 10.264%, k=1 plane 12.400% -> 11.067%,
  and k=8 lowest miss 4.628% -> 5.189%. RAISE P_E was 0.018503 ->
  0.015111 and DIV2K 0.074104 -> 0.052932. Bpp-1 P_E/AUC was
  0.019898/0.991922 -> 0.014878/0.997560. Low-bpp k=1 random miss was
  56.57% -> 48.48%; low-bpp stride miss was 26.09% -> 27.83%.
- Wide validation best epoch was 24 (1-based), P_E 0.017663. It had one
  spike at epoch 5, P_E 0.0485, and recovered at the next epoch. Final
  train loss was 0.1196 and validation loss 0.0886.
- Wide timing found the early train phase 61.9 s, validation 13.2 s, and
  the first validation batch wait 18.8 s; other gaps were below 0.5 s.
  Curves took 0.41 s, scalars 0.24 s, and checkpoint 0.03 s. From epoch
  24 zero-based, train time rose 61.9 -> 84.4, 99.6, 102.8, 105.3, 108.1,
  111.1 s and validation was 29.2 s. Host data did not show growth: RAM
  available 124.6 -> 123.3 GiB, RSS 10.84 -> 10.93 GiB, swap 0, and I/O
  wait 0%; CPU 27.8 -> 16.8%, GPU 66.5 -> 44%, and power 298 -> 218 W.
  This is an unknown epoch-dependent cause, not proof of a model fault.

#### Pi data downloads

- Worker_3 added the LSDIR extractor in commit `3b2e8a4`: two workers,
  Nice 19, idle I/O, and a 100 GiB free-space guard. It writes HR and val1
  under `/srv/datasets/lsdir`, `inventory.csv`, and extractor state.
  Measured extraction was 287.5 s per shard for two parallel shards, about
  61 MB/s; total time was estimated at about 40 min for 85,241 files.
- ALASKA transfer rate was 3.031 to 3.039 MB/s. At 20:08 +08, 47,800 of
  70,282 ALASKA files were downloaded, about 9,000 files/h after LSDIR
  finished; completion was estimated for 22:30-23:00. Pi `/srv` had 599 GB
  free.
- Worker_3 finished the LSDIR unpack and inventory in commit `3b2e8a4`.
  The HR set has 84,991 PNG files and 155,204,828,212 B; val1 has 1,000
  PNG files. `inventory.csv`, `inventory_val1.csv`, and `extract/REPORT.md`
  record shard, path, bytes, dimensions, colour type, bit depth, and SHA-256.
  The unpack took 43 min from 12:05 to 12:47 UTC, used two workers, and had
  0 failures. Pi `/srv` had 490 GB free after the unpack.

#### LSDIR and source-mix findings

- The HR count is 84,991. All files are RGB 8-bit PNG, with 0 duplicate
  paths. One byte-identical pair exists:
  `0022000/0021599.png` and `0022000/0021600.png`; drop one before splitting.
  Width is 540 / 1,164 / 4,464 and height is 540 / 864 / 3,876 for
  minimum / median / maximum. All photos have both sides at least 512.
- A trimmed 512 grid gives 1 tile for 25,946 photos, 2 for 50,811, 3-5
  for 1,823, and 6 or more for 6,411: 179,006 tiles total, mean 2.11.
  These counts are from the inventory analysis.
- Correction: val1 is the X2 low-resolution validation set. Its median is
  500 x 367, and only 290 photos have both sides at least 512. It is not an
  HR set. LSDIR validation and test will come from train shards.
- ALASKA subset v1 has 9,723 photos at 512-1,024, about 1.56 tiles/photo
  and about 15,200 tiles (estimate). RAISE train has 45,674 tiles from 701
  photos and 3 cameras, about 65 tiles/photo. LSDIR HR is already shrunk 4x
  by its authors and may be easier than camera output; report it separately.
- ALASKA transfer rate stayed 3.031-3.084 MB/s.

#### Source sampling decision

- Use per-epoch sampling by source, not data deletion: RAISE 40%, LSDIR 20%,
  and ALASKA 40%, about 45,000 tiles per epoch (estimate; current data has
  53,950). Draw a fresh random 18,000 of 45,674 RAISE tiles each epoch.
  Equal thirds was rejected because LSDIR is already shrunk and may create
  easy-data shortcuts.
- Keep RAISE validation and test unchanged. Give new sources extra
  validation and test sets.
- ALASKA 40% needs about 18,000 unique tiles per epoch, but subset v1 has
  about 10,900 train tiles. A larger full-download selection of about
  12,000-15,000 ALASKA training photos is proposed; user decision and the
  held-out camera list are Pending.

#### Map and data discussion

- At equal effective bpp and threshold 0.5, k=1 misses were contiguous
  0% at 0.2-0.4 bpp and 1% at 0.4-1.0; random 10% / 4%; stride 7-30% /
  4-18% with only 3-40 tiles per stride value. Contiguous k=1 letters /
  random-bytes misses were 1.2% / 1.9%. The model did not use periodic or
  payload structure; earlier high-bpp stride weakness was mostly capacity
  clipping.
- Scattered k=1 at 0.05 bpp changed about 1 in 120 values by +/-1, about 25
  per 32 px cell. Clean and stego maps looked the same; photo score changed
  from 0.0015 to 0.0030. This is below the photo's own noise per cell.
- Discussion only, no runs: map(raw)-map(denoised) may provide a per-photo
  reference but costs two passes and may remove natural noise. Physics/noise
  and model weakness are separate limits. Random placement with encrypted
  payload covers non-adaptive LSB tools, not adaptive WOW/S-UNIWARD/HILL or
  efficient coding. Inception-style branches may target structured hiding,
  but not random hiding. A wider model makes GPU work the limit, so BF16 is
  the main speed lever. Per-cell labels are available from change masks; the
  mask code change is not checked. Scattered low-bpp cells need soft or
  ignored labels. The tile label and map objective remain mismatched.

#### ALASKA selection v2 and epoch-24 slowdown

- Worker_3 added `scripts/alaska_scan.py` and `scripts/alaska_select_v2.py`
  in commit `91ada60`. The scanner checks TIFF headers and uses SHA values
  from `SHA256SUMS`; it scanned 57,439 Pi files in 46 s. The PC v1 copy has
  9,723 SHA-verified files, all uncompressed RGB 8/8/8. Selection v2 is
  seeded and order-independent, excludes flats, textures, RAW_wesaturate,
  and Leica_Mono, uses 8 unseen models up to 250 each, prefers v1 photos,
  and splits the other models into 13,500 train, 1,000 validation, and 1,700
  seen-camera test photos.
- The dry run is not final: 12,843 eligible IDs were missing and Pentax_K50
  was not downloaded. Present files gave train 13,446 photos / 21,057 tiles /
  21.19 GB; validation 987 / 1,525 / 1.54 GB; seen test 1,707 / 2,642 /
  2.67 GB; unseen test 2,000 / 3,153 / 3.17 GB; total 18,140 / 28,377 /
  28.57 GB. Of these, 6,624 photos were already on the PC from v1.
- The leakage rule uses ID-order blocks of 10 consecutive photos, with one
  split per block and an 11-ID minimum gap between splits. Block size 10 is
  a guess, not a measurement. The final run waits for the ALASKA download.
- Worker_2 found the epoch-24 slowdown in `scripts/run_notebook.py`: each
  progress chunk was appended to the growing cell output, copying all prior
  text. The training loop then waited and GPU/CPU use fell. A 3,000-write
  PC microbenchmark took 0.014 / 2.678 / 12.957 / 38.738 s for starting
  output 0 / 4 / 16 / 32 MiB. This is a logging bug only; the data stream and
  R2/wide results are unchanged.

#### Validation-worker and logging fixes

- Worker_2 committed speed fixes in `0d6d157`. `V3_PERSISTENT_WORKERS=on|off`
  defaults on for notebook 05 validation only. Train workers already persist;
  test and diagnostic workers remain short-lived. `OrderedPairLoader` keeps
  workers after a complete pass, resets after a partial pass, and shuts down
  explicitly during runner cleanup. `CellStream` now stores chunks in a list
  and joins once, with identical final bytes.
- Twenty-one tests passed, including persistent replay and partial-pass
  regression tests. The standard test hash remains `fdd989f7...837`.
- With priming on and PC 6/4 workers, the first 8 train-batch hashes matched
  with persistence on and off: epoch 0 `8d19c9a8...ba6aa`, epoch 1
  `8aab10a2...640a`. All 32 QUICK validation pairs matched on/off and across
  epochs: `e215a95c...05c`.
- PC validation first-batch wait fell from 4.780 s to 0.870 s in QUICK,
  saving 3.91 s; a data-only probe fell from 4.584 s to 0.867 s. PC
  process-tree PSS after validation rose from 2,880 to 4,573 MiB, an
  increase of 1.69 GiB for four retained workers. A 16-worker G4 estimate is
  +6.8 GiB, unverified; the wide run had more than 117 GiB available.
- The R2 and wide fit-cell outputs reached 31.2 MiB at epoch 24 and 37.9 MiB
  at epoch 29. Append-and-join for 32 MiB took 0.0048 s plus 0.0126 s with
  identical bytes. A PC data-only epochs 20-27 probe had no slowdown; epoch
  24 train was 1.117 s and validation 3.917 s. The data stream is not the
  cause. Post-fix G4 timing is Pending.

#### Pi network outage

- At about 22:05 +08, the Pi lost Internet access while the LAN worked.
  Gateway `192.168.18.1` replied in 3 ms, but 1.1.1.1, Google, and
  alaska.utt.fr timed out. The PC on the same router reached alaska.utt.fr
  with HTTP 206 in 2.7 s. Tailscale showed the Pi offline and no exit node
  was in use.
- ALASKA stopped at 64,097/70,282 with 16 curl 28 timeouts in retry backoff.
  With user approval, the Orchestrator reconnected Wi-Fi using detached
  `systemd-run` and `nmcli con down/up "netplan-wlan0-Lee Family"`.
  Internet returned: ping 6 ms and alaska.utt.fr HTTP 206 in 0.9 s.
  `cvdl-alaska` restarted at 22:24 +08 to skip the 1,200 s waits.
- The service re-verifies existing files with TIFF checks and SHA-256 first
  (about 100 GB, estimated 10-15 min), then downloads about 6,200 files
  (estimated about 40 min). Cause is unknown; router per-device limit or a
  stale Wi-Fi session is only a guess.
- New-data checks: old RAISE/DIV2K records are unchanged; the standard test
  hash `fdd989f7...b06837` is unchanged; with the mix unset, the train stream
  (epochs 0/1) and the val/test frames are byte-identical to before. PC QUICK
  with the mix on (30 LSDIR PNGs, 33 ALASKA TIFFs): 24/12/24 tiles per epoch,
  0 repeats, 16/16 cells, 422.4 s. Capped QUICK: 12/6/12 tiles, 16 val pairs
  (6/5/5), 116.8 s, pair-list SHA `2f1424db...0ce8a`.
- The full 89-test suite passes on a clean worktree. In the shared tree,
  `test_notebook_compile` and `test_notebook_diagnostics` fail only because
  they read the user's edited notebook 04 (the Orchestrator confirmed this on
  a clean worktree of `e592cce`).
- Today's split by tiles: train 53,950 (71%), validation 11,238 (15%), test
  10,956 (14%). Validation precision estimate (P_E about 0.02, binomial, two
  images per pair): +/-0.0008 at 16,000 pairs, +/-0.0009 at 11,238,
  +/-0.0011 at 8,000, +/-0.0016 at 4,000. Epochs are compared on the same
  fixed set, so differences resolve better than this.
- ALASKA v2 final splits (photos / tiles / bytes / on PC / to copy):

  | Split | Photos | Tiles | Bytes | On PC | To copy |
  | --- | ---: | ---: | ---: | ---: | ---: |
  | train | 13,500 | 21,141 | 21,301,332,312 | 3,285 | 10,215 |
  | val | 1,000 | 1,552 | 1,562,059,792 | 214 | 786 |
  | test_seen | 1,700 | 2,672 | 2,686,627,688 | 342 | 1,358 |
  | test_unseen | 2,000 | 3,153 | 3,173,550,368 | 2,000 | 0 |
  | all | 18,200 | 28,518 | 28,723,570,160 | 5,841 | 12,359 |

  Bytes to copy: 19,429,045,766. Train tiles 21,141 are above the 19,000
  target. 3,882 v1 files on the PC are not in v2.
- Per model: 24 mixed models have train 442-445 (Canon_EOS700D 356 because of
  the 75% cap), val 41-42, test_seen 70-71. Train-only models (even share):
  Canon_PowerShot-G11 333, Nikon_D90 395, Kodak_Z990 555, Pentax_K10D 556,
  Pentax_ist-DS 556, Sigma_SD10 556. The 8 unseen models have 250 each, all
  from v1.
- Leakage rule: blocks of 10 consecutive IDs per model (an assumed burst
  length, not measured); an empty block separates splits, so the minimum ID
  gap is 11 (11-21 per mixed model); val/test blocks come in runs of up to 3;
  a mixed model uses at most 75% of its photos; 0 fallbacks. No SHA-256 is
  repeated in the selection. The pool's one identical pair (58794 = 59282)
  is in the excluded RAW_wesaturate group.
- The builder accepts all 18,200 rows (`_manifest('alaska')`). Expected capped
  validation size: 5,000 RAISE + 1,552 ALASKA + up to 1,500 LSDIR = 8,052
  pairs. G4 validation estimate with the capped set: about 9-15 s (XuNet) and
  9-17 s (wide) per epoch (not measured).
- Correction: the train-only decision used partial-download counts. Full
  counts are G11 333, D90 395, Kodak_Z990 832, Pentax_ist-DS 1,076,
  Sigma_SD10 2,380, Pentax_K10D 5,461. Four of the six models are not small.

### Decided

- Pending user decision: G4 runs use `V3_PRIME_NEXT_EPOCH=on` as the
  recommendation.
- The PC default remains priming off pending the PC test.
- New-source plan (user, 2026-10-15):
  - ALASKA#2 uses a roughly 10,000-image subset of various-size uncompressed
    colour TIFF files, across cameras. Hold out about 8 camera models for
    testing only.
  - LSDIR uses about 3,000 train photos plus validation and test. Use the
    official split if HR images exist; otherwise use a seeded holdout.
  - DIV2K leaves the training plan. Its files stay on disk, and teammates
    must be told.
  - Download sources to the Pi first. Start with subsets; scale up only if
    diversity helps.
- Pending user approvals: accept the LSDIR terms on Hugging Face, read both
  licences, and tell teammates about DIV2K.
- ALASKA subset decision (user, 2026-10-15): select evenly by camera model,
  cap 250 per model, 39 models, 9,723 images, seed 0. Leica_Mono has 223.
  Exclude FLATS/TEXTURES (1,472) and RAW_wesaturate (4,681). No split yet.
  Commit `d664c01` contains `scripts/alaska_select.py` and
  `data/alaska/selection_v1.csv`; the script reproduces the CSV byte for
  byte.
- Download decision: the PC downloads the ALASKA subset (N=8, started at
  11:47:54). The Pi downloads full LSDIR (17 shards plus val1) and the other
  ALASKA IDs as unattended systemd services, with ALASKA starting at 13:15
  or later. The user later moves the Pi SSD to the PC to copy subsets.
- Wait on 256 throughput work. Do a G4 256 speed test only when 256 or
  Mode 2 step 17 enters the plan. Do not adopt PyAV now.
- Correction to the map-model design input (user, 2026-10-15): every
  normalisation compares against some group. Train-mode BatchNorm compares
  with other images in the batch and leaks the batch context: all-clean
  batches were called stego; V3a clean accuracy in all-clean train-mode
  batches was 0.0169 / 0.0218. GroupNorm compares with the rest of the same
  image or chunk and leaks spatial context: the map band and self-dimming
  were found above. GroupNorm removed batch dependence in V3b/V3c; train/eval
  maximum logit difference was 0.
- A map model needs a fixed reference or only the cell's own position.
  Candidate designs, not decisions, are: A) no normalisation with clip 3;
  B) freeze statistics after an early training phase; C) per-position
  channel normalisation across channels at each position only, which may
  remove the local noise-strength signal.
- Judge each candidate with the existing all-clean/all-stego batch checks
  (V3 tests e/f), chunked-versus-whole map equality, cell AUC, and photo P_E
  with k=1 reported. Train with per-cell labels and use a trained per-cell
  map for the photo verdict.
- Map direction decision (user): use no normalisation because GroupNorm has
  a spatial leak. Use new notebook `05_xunet_v4_nonorm.ipynb`, not notebook
  04. It supports `V3_NORM` batch/group/none, `V3_EPOCHS`, `V3_LR`,
  `V3_WARMUP_EPOCHS`, `V3_GRAD_CLIP` (0 disables clipping),
  `V3_LR_DROPS`, `V3_TRAIN_YAML`, `V3_EXTRA_TESTS`, `V3_BATCH_SIZE`, and
  `V3_PRECISION`. G4 selects it with `V3_NOTEBOOK`.
- The GPU LSB generator is shelved. Its PC profile estimated a 0-25% gain,
  not a measured result. The G4 speed matrix is Done.
- Keep batch size 32. Pending user decision: use BF16; validation and
  epoch-transition overhead; and data supply (preload or GPU generator) only
  pay together with BF16.
- Wide XuNet is tested with the R2 recipe: train k=1-4, standard validation
  and test including k=5-8. Pass criteria fixed before the run are k=1-4
  P_E at threshold 0.5 below R2's 0.0303 by more than about 0.002, lowest
  k=1 miss below 10.4%, clean false alarm no higher than 2.3%, and no
  training blow-up. If it passes, it becomes the base model; otherwise move
  to new data with XuNet.
- Work order: wide XuNet result, then ALASKA and LSDIR subsets, then dense
  per-cell training on the better base. Calibration, an inception front, and
  SRNet remain ideas.
- Use LSDIR and ALASKA subsets. Unpack all LSDIR on the Pi. Suggested LSDIR
  subset is 3,000 train, 500 validation, and 1,000 test images from train
  shards, seeded and spread across shards, colour, both sides at least 512;
  val1 is an extra check set. User decision is Pending.
- Source mix decision: sample each epoch at RAISE 40%, LSDIR 20%, and ALASKA
  40%. Keep RAISE validation/test fixed and add separate new-source sets.
  A per-source sampler is Pending.
- ALASKA selection decision (user): take more photos from the full download
  and hold out 8 models for an unseen-camera test: Google_Pixel-XL,
  Samsung_GalaxyS9plus-SM-G965F, Huawei_P20lite-ANE-LX1, Panasonic_DMC-TZ60,
  Ricoh_GRDigital-III, Canon_EOS-60D, Sony_A7R, and Pentax_K50. Exclude
  Leica_Mono because it is monochrome (`R=G=B`); keep Nikon_D90 in training.
  Use 30 models for about 13,500 train, 1,000 validation, and 1,700
  seen-camera test photos, split by contiguous ID blocks to reduce burst
  leakage. Six models with too few photos go to train only: Canon_PowerShot-G11,
  Kodak_Z990, Pentax_K10D, Pentax_ist-DS, Sigma_SD10 (250 each), and
  Nikon_D90 (395). This keeps the 11-ID gap. Selection v2 final run is
  Pending.
- Worker_2 does speed fixes first, then new-data code. Worker_3 handles Pi
  data jobs. The Orchestrator does not monitor G4 runs; a worker monitors them.
- User decision: ALASKA v2 final selection is tomorrow's work; Worker_3 was
  not triggered tonight. If the Pi loses Internet again, reconnect with the
  same Wi-Fi procedure and check the router for a per-device limit.
- Hypothesis, not tested: without normalisation, tanh inputs in layers 1-2
  are not scaled. They may saturate tanh and explain R1/R2 underfitting and
  validation spikes.
- The Orchestrator runs `scripts/g4/submit.sh` only after the user says
  "go". The user signed in to gcloud; `submit.sh status` works from the PC.
- MIL is dropped. Mode 2 step 17 and the bag design leave the plan.
- New direction (user, 2026-10-15): use a spatial suspicion map. Train with
  exact per-cell labels from the cover/stego difference: the share of changed
  pixels in each cell. Take the photo verdict from the map with top-k cells.
- Tiling strategy is open: non-overlapping centred grid, seeded random crops
  per epoch, whole-photo input, or train on crops and test on whole photos.
- Pending user decision: run full PC training headlessly with
  `scripts/run_notebook.py` at the proven 13.6 it/s, or test the VS Code
  candidates. The two-epoch run confirms the headless speed.
- User: checkpoint selection uses the capped combined validation set (about
  8,000 pairs, about 10% of the ~81,000-tile unique training pool), not the
  standard set only and not the full combined set (~16,000). Validation
  photos stay set apart; only fewer tiles are scored. DIV2K and the full
  standard set stay in the fixed tests.
- ALASKA selection v2 (`5ca650b`) is accepted as committed. Pending user
  decision: keep the six train-only models (as committed) or option B, with
  only G11 and D90 train-only (0 fallbacks, minimum gap 11, val/test about
  36/61 per model over 28 models; a one-line change). The Orchestrator
  recommends option B.
- User: the session ended after the ALASKA selection task.

### Next

- Pending: `step_profile.py` finds cells by content instead of fixed cell
  numbers. This is a separate approved change.
- Pending: VS Code slowdown cause, by extension rollback or hidden-tab test.
- Pending: prime default decision.
- Done: Worker_2 demoed maps from `g4-prime-1` best epoch 8.
- Done: selected no normalisation for the map direction.
- Pending user: per-cell loss, crop policy, and crop size for the map model.
- Pending: choose the train/validation/test split and 8 held-out ALASKA
  cameras.
- Done: R2 `g4-v4-r2` completed. Its validation still includes k=5-8,
  which were not in its training draw; interpret its comparison with care.
- Pending: one G4 maintenance boot installs the stable bootstrap from
  `e1186fd`.
- Done: G4 speed matrix for batch sizes 32, 64, 128 and precision 32/BF16.
- Done: user signed in to gcloud; `submit.sh status` works from the PC.
- Pending: Worker_2 sets up and proves the unattended Pi services, then
  checks the PC ALASKA subset.
- Pending: when the Pi SSD is unplugged, shut the Pi down first; G4 results
  cannot be delivered because `cvdl-results` is on that SSD.
- Pending: train/validation/test split and the 8 held-out cameras. D90 is
  also in RAISE; Leica_Mono is monochrome.
- Pending user: accept LSDIR terms on Hugging Face, read both licences, and
  tell teammates about DIV2K.
- Pending: leave `V3_DIAG_WORKERS` at its default parallel setting for
  decode; the serial diagnostic path took 458 s here versus about 88 s with
  8 workers.
- Done: wide XuNet result and timing findings from Worker_2.
- Done: LSDIR inventory from Worker_3. Pending: ALASKA download completion.
- Done: logging-bug fix and persistent validation-worker parity checks.
  Pending: measure the post-fix G4 timing. The expected saving is about 14 min
  and $0.40 per 30-epoch run (estimate: about 5 min late slowdown plus 9 min
  validation start-up).
- Pending: download completion check and ALASKA v2 final run tomorrow;
  LSDIR subset, per-source sampler, and new-data code remain Pending.
- Pending user: choose the checkpoint-selection validation set after
  Worker_2 lists the options.
- Pending: copy data to G4, estimated at 35-45 GB.
- Pending, optional: measure miss rate against tile noise, run the denoise-
  and-compare demo, test tanh saturation, and run a filter linear probe.
- Done: new-data code, capped validation, ALASKA download and ALASKA
  selection v2.
- Pending user: ALASKA option B (two train-only models instead of six).
- Pending: LSDIR subset selection to the contract (suggested 7,000 / 1,500 /
  2,500 photos), final manifests, copy of about 19.4 GB ALASKA plus the LSDIR
  subset to the G4 (maintenance boot), and the first new-data G4 run (it also
  measures the post-fix timing). See `docs/archive/handoff-2026-10-15.md`.
