## 2026-10-14 — PC cache, diagnostics, and compile benchmark

### Did

- The Worker built a PC cache after the user's decision that a cache inside
  the checkout is allowed only when git ignores it. Commit `e78a1a0`.
  - Path: `Datasets/otf-cache/v1` in the container, or
    `/work/Datasets/otf-cache/v1`.
  - Eight workers built 1,900 photos in 170.4 s. Size was 52.76 GB.
  - Verification passed with 0 mismatches. The 1,900 JSON records equal the
    VM cache except for `master_mtime_ns`; aggregate SHA-256 was
    `bde842a0...e482`.
  - Git was installed in container `aai3001`. This is lost if the container
    is recreated.
- The Worker fixed the OTF diagnostic in commit `ff06623`. It reads each
  selected photo once and keeps only sampled tiles. The printed CHECK lines
  stayed identical.
- The Worker benchmarked `torch.compile` in commit `dd8cb60`, with advice
  from ROCm-Works in `rocm-perf/notes/compile_advice_nb04.txt`.
  - `V3_COMPILE=off` is the default; `default` enables compile. Compile is
    in place just before fit. Checkpoints keep the same keys and resume in
    both directions.
- Worker_2 completed fix 5 in commit `cc3aeed`, after the original Worker
  session ended during the task. Details are in
  [docs/archive/notebook-otf-verification.md](../../archive/notebook-otf-verification.md).
  - Evaluation gives each worker whole photos with `PhotoPairTasks` and
    `OrderedPairLoader`. The parent restores the original pair order and
    batch boundaries. It uses 4 non-persistent workers; 0 workers also works.
    Early close drains only in-flight tasks.
- The Worker completed fix 4 with Worker_2 review fixes in commit `05ecedc`.
  Details are in [docs/archive/notebook-otf-verification.md](../../archive/notebook-otf-verification.md).
  `MixedLoader` now slices, joins, and pins batches in one background thread
  with a queue of 2 batches. There is no new setting or notebook change.
- Training workers now shut down after fit. This freed about 10.0 GB of PC
  RAM for 8 workers.
- The user ran short PC notebook 04 tests with commit `dd8cb60`,
  `V3_COMPILE=default`, `QUICK=False`, and the RX 9070 XT. The user stopped
  them by hand and deleted the outputs afterward.
- The headless runner `scripts/run_notebook.py` was added in commit
  `b237003`. It supports an env file and `--set` settings. `QUICK` must be
  explicit. It uses a git-ignored empty output folder, puts
  `V3_OUTPUT_ROOT` inside it, and saves `executed.ipynb` and `status.json`
  after every code cell. It also writes a 10 s heartbeat and live `run.log`.
- The G4 run-and-stop kit was added in commits `9919f86` and `eddf21a`.
  Its files are under `scripts/g4/`; the record is
  [docs/active/g4-batch-runner.md](../../active/g4-batch-runner.md). It includes setup, systemd
  job, submit, speed-test, and cache-verification scripts.
  - `build_cache.py` supports `--allow-high-workers`; the default limit of
    1-8 workers is unchanged. A builder uses about 1 GiB RAM.
  - A 12-photo subset built with 8 and 12 workers was byte-identical.

- The Worker limited compile helper threads in commit `22e61a9`. With
  `V3_COMPILE=default`, the notebook sets
  `TORCHINDUCTOR_COMPILE_THREADS=4`; an explicit value overrides it.
- The Worker completed Mode 2 steps 18 and 19 in commit `e84df32`. The full
  record is [docs/archive/mode2-verification.md](../../archive/mode2-verification.md).
  - New module: `LSB-Dataset-Generator/lsb_mode2_torch.py`. Each worker owns
    whole photos, reads and embeds each photo once, and yields uint8 tiles
    with actual-change labels and side records through a bounded 1,024-image
    shuffle.
  - Fixed evaluation keeps every validation/test tile in fixed order.
    `BagLoader` gives one bag per photo, the real tile count, no padding,
    per-tile labels, and photo-level fields. A photo is label 1 when any tile
    changes.
  - Manifest maxima are 70 / 260 for RAISE and 16 / 64 for DIV2K at 512 /
    256. Step 17, the pre-built Mode 2 writer, is deferred. No MIL model or
    notebook change was made.
- The user set G4 disk throughput to 400 MB/s with 3,000 IOPS. The PC
  `~/.ssh/config` now has a Tailscale host entry for `rtx6000pro`.
- Commits `5e9d8c3`, `0541ae0`, `d5352ac`, `088a747`, and `33e5e31` updated
  the parallel resumable master upload, speed-test file limit and epoch
  handling, setup build tools, job open-file limit, and runner soft-limit
  logging.
- The first G4 job proof was `g4-quick-proof-1`: QUICK, decode, 32 train
  workers, 16 evaluation workers, compile default.
- The second G4 proof used job ID `g4-quick-proof-2` with the same settings
  and no SSH login to G4 during the run.
- The Worker applied the RemoveIPC fix in a maintenance boot, using checkout
  `f9fb3fd`. The boot ran from 13:54:26Z to 13:58Z. It changed
  `busctl RemoveIPC` from true to false through
  `/etc/systemd/logind.conf.d/cvdl-no-remove-ipc.conf`.
- The first full G4 run was `g4-full-1`: `QUICK=False`, decode, 32 train
  workers, 16 evaluation workers, compile default, run name `g4-otf-full-1`,
  and a 60 min timeout.
- The Worker parallelized notebook 04 diagnostic pair preload in commit
  `57274c7`. `lsb_torch.preload_pairs` and `PreloadedPairs` are used by
  notebook 04 cells 2 and 29. `V3_DIAG_WORKERS` is optional and defaults to
  `EVAL_WORKERS`. README and `docs/archive/notebook-otf-verification.md` were updated.
- The Worker added `scripts/g4/step_profile.py` in commit `e2bb160`. A G4
  maintenance boot from 15:16Z to 15:32Z ran 13 decode variants at 400
  steps with 32 workers, cache at 24 workers for 3 variants, and cache at
  32 workers for 1 variant. Results are in the git-ignored
  `runs-out/g4-step-profile/` folder. The Worker also diagnosed the
  epoch-start stall on the PC; that diagnosis made no code change.
- The Worker added persistent-worker priming in commit `d796971`.
  `PrimedTrainingLoader` is in `lsb_torch`; notebook 04 cells 2, 12, and 19
  use it. `V3_PRIME_NEXT_EPOCH` controls it and defaults off.

### Found

- The diagnostic fix reduced times as follows:

  | Run | Before | After |
  | --- | ---: | ---: |
  | QUICK PC | 194.1 s | 17.3 s |
  | Full VM cache | 371.6 s | 100.5 s |
  | Full PC decode | >45 min, incomplete | 464.6 s |

- PC breakdown from Notebook 02 on the PC predictions (scratch output):
  overall P_E 0.017296, AUC 0.997441; RAISE 9,728 pairs P_E 0.011513;
  DIV2K 1,228 pairs P_E 0.061889.
- Compile correctness against eager gradients was 2.4e-7 on the PC and
  2.4e-7 on the L4 with TF32 off in the check. With native TF32 on, the L4
  difference was 4.9e-4, a TF32 effect. QUICK compile smoke runs passed on
  both PC and VM.
- Steady images/s, eager -> compiled:

  | Case | Eager | Compiled |
  | --- | ---: | ---: |
  | PC decode, 8 workers | 380.3 | 474.9 |
  | PC cache, 6 workers | 379.5 | 462.2 |
  | PC GPU-only | 369.2 | 457.8 |
  | L4 cache, 6 workers | 279.6 | 332.4 |
  | L4 GPU-only | 287.5 | 348.8 |

- The first compiled step took 14.9 s on the PC and 30-36 s on the L4.
- Steady compiled PC tests, as shown by the progress bar:

  | PC test | Training speed | Validation per epoch |
  | --- | ---: | ---: |
  | Decode, 6 workers, epoch 1 | 13.54 it/s (about 433 images/s) | about 2 min |
  | Cache, 6 workers, epoch 0 | over 13 it/s (partly warm) | about 20 s |
  | Cache, 6 workers, epochs 1 / 2 | 13.85 / 13.87 it/s (about 443 images/s) | about 20 s |

- During the cache run, the GPU was 100% busy at 251-255 W, 85-86 C,
  and 2,940-3,120 MHz. Each of 6 cache workers used about 21% CPU. The GPU
  limits the long run at about 13.86 it/s, below the short GPU-only result
  of 14.3 it/s (about 458 images/s).
- Cache-test validation P_E for epochs 0-2 was 0.0353, 0.0332, and 0.0247.
- Estimated full PC time with cache and compile is about 4:03 training plus
  0:20 validation per epoch, or about 44 min for 10 epochs. Decode without
  compile is about 7:20 per epoch, or about 73 min for 10 epochs.
- Sixteen idle `torch.compile` helper processes stayed alive. Each used about
  410 MB RSS, and available PC RAM was about 6.3 GB.
- Compile-helper limit results, PC cache, 6 workers, 300 steps:

  | Measure | Before | After |
  | --- | ---: | ---: |
  | Compiler processes | 17 | 5 |
  | Cold summed RSS | 9.44 GB | 3.03 GB |
  | Cold summed PSS | 1.30 GB | 0.80 GB |
  | Warm summed PSS | 0.53 GB | 0.49 GB |
  | Warm images/s | 471.1 | 473.9 |
  | Cold images/s | 409.9 | 417.9 |
  | Warm first step | 5.4 s | 5.1 s |

  Most RSS was shared pages. The real cold saving is about 0.5 GB.
- Mode 2 parity used 6 real photos, decode and cache sources, train/val/test,
  and 0, 2, and 4 workers. There were 0 pixel, record, label, or bag
  mismatches against `lsb_mode2.iter_photo_tiles`. Mode 1 was unchanged:
  200 batches and 6,400 images had fingerprint `32bd1035...9075` before and
  after.
- Full train epoch at 512, cache, 6 workers, 1,381 photos and 107,900 tiles:
  PC 127.5 s (846 images/s, peak PSS 5.73 GB); VM 252.0 s (428 images/s,
  peak PSS 5.27 GB).
- Natural label-1 share at 512 was 27,008/91,348 (29.57%) for RAISE,
  5,870/16,552 (35.46%) for DIV2K, and 32,878/107,900 (30.47%) combined.
  No rebalancing or class weights were used.
- Sample benchmark, 128 photos and 9,808 tiles in 256 bags, tile images/s /
  bags/s:

  | Source | 4 workers | 6 workers | 8 workers |
  | --- | ---: | ---: | ---: |
  | PC cache | 512.1 / 14.9 | 505.7 / 14.9 | - |
  | PC decode | - | - | 326.0 / 9.0 |
  | VM cache | 268.5 / 8.6 | 286.9 / 8.2 | - |
  | VM decode | - | - | 136.1 / 3.8 |

#### Evaluation worker fix (fix 5)

- At 512 with 4 workers, photo reads and time changed as follows:

  | Case | Photo reads before -> after | Time before -> after | Peak worker PSS before -> after |
  | --- | ---: | ---: | ---: |
  | PC decode val | 810 -> 270 | 135.6 -> 47.2 s | 2.41 -> 2.81 GB |
  | PC decode test | 745 -> 249 | 133.8 -> 46.6 s | 2.33 -> 2.78 GB |
  | PC cache val | 810 -> 270 | 19.9 -> 23.6 s | 2.08 -> 2.46 GB |
  | VM cache val | 810 -> 270 | 68.6 -> 74.9 s | 1.88 -> 2.27 GB |

- The cache slowdown is from parent re-stacking (5.4 of 17.0 s). It was
  not changed because safe copies are required.
- Validation (11,238 pairs) and test (10,956 pairs) were byte-identical
  before and after. Decode/cache, 512/256, and 0/2/4 workers were identical.
  Train-eval 1,000 pairs were identical, with 554 reads instead of 1,000.
  `test_predictions.csv` from best epoch 7 (21,912 rows) was byte-identical.
  The Mode 1 fingerprint `32bd1035...9075` was unchanged.
- QUICK PC cache passed all 15 cells in 55.4 s. Evaluation shutdown took
  0.47-0.65 s, with no leftover workers. The Seoul Spot VM was stopped twice
  on 2026-10-14 and restarted once in between, up at 16:23 +08. The first
  restart failed because the zone had no L4 capacity. Spot availability was
  Limited in Seoul (11-20% preemption, $0.39/h), High in Taiwan ($0.60/h),
  and Medium in Singapore (20%+, $0.63/h). Nothing was lost.

#### Background join and pinned-copy fix (fix 4)

- Before the change, main-thread join plus pin took 6.59 ms of 70.9 ms
  compiled per step (9.3%) and 5.16 ms of 88.6 ms eager (5.8%).
- PC throughput after 30 warmup steps, before -> after:
  compiled + cache, 4 workers 467.4 -> 472.0 and a repeated run 471.0 ->
  468.8; eager + cache 356.9 -> 379.8 (+6.4%); compiled + decode, 6 workers
  457.6 -> 470.4 (+2.8%). The compiled cache result has no reliable gain.
- Extra pinned memory was at most about 72 MiB. Identity stayed exact for
  200 batches at 0/2/4 workers; the fingerprint stayed
  `32bd1035...9075`; epochs 2 and 3 matched the previous run. First-batch
  times for epochs 1/2/3, 4 workers, were 5.20/0.86/0.87 s before and
  5.19/0.87/0.86 s after.
- Clean stop took about 0.47 s with no thread or worker left. QUICK PC cache
  passed all 15 cells in 48.8 s. 76 shared tests and 34 generator tests pass.
- VM fix 4 check: L4, cache, 6 workers, batch 32, 30 warmup plus 300
  steady steps, native TF32 flags. Compared old `fe23238` with new
  `cfbf71f`:

  | Mode | Old images/s | New images/s | Repeat old -> new |
  | --- | ---: | ---: | ---: |
  | Eager | 280.38 | 278.72 (-0.59%) | 274.92 -> 274.48 (-0.16%) |
  | Compiled | 336.60 | 334.60 (-0.60%) | 336.69 -> 336.51 (-0.05%) |

  Main-thread fetch time fell from about 11.7-12.2 ms to 0.07-0.09 ms
  per step, but throughput did not change. The L4 is the limit. Fix 4 has
  no reliable VM gain; it helps PC eager (+6.4%) and PC decode (+2.8%).
- The VM returned at about 17:23 +08 and fast-forwarded to `cfbf71f`. The
  scratch folder `~/otf-dev/eval-yx` was removed.

#### Headless runner and G4 kit checks

- In PC test mode, QUICK ran all 15 cells with exit 0 and delivered 26 files
  to the Pi inbox. A wrong Pi host created a pending record; it delivered on
  the next boot. Failing and timeout runs uploaded their results and reached
  shutdown. Repeated job IDs were skipped. Simulated kills resent starting,
  running, and partial-folder records on the next boot.
- Spot preemption gives about 30 s. Upload then usually fails, so the next
  boot resends it.
- The G4 kit was not contacted. The initial G4 plan was `rtx6000pro` in
  `europe-north1-a`, `g4-standard-48` Spot, about $2.03/h, with a 2 h maximum
  run duration and Stop action. It is separate from the Seoul L4 project and
  billing pool. Budget "Credit Spending" is $383.45, with alerts at
  26/52/78/90%.
- The first real G4 setup was attempted in `europe-north1`: zone a and then
  zone b could not restart because of Spot stockout. The user deleted that VM
  and created `rtx6000pro` in `us-central1-c`, project
  `personal-compute-511601`, `g4-standard-48` Spot at $1.62936/h, with a 2 h
  maximum and Stop action. It has a 200 GB Hyperdisk Balanced disk (3,000
  IOPS, 140 MB/s).
- Job timeout defaults to 100 min and refuses values above max-run minus
  15 min. Missing job mode is off: send pending results, then shut down.
  Maintenance runs only when explicitly selected. Each job checks out its
  commit and runs `uv sync --locked`.
- Upload retry budget is 300 s per boot, after waiting up to 30 s for Tailscale
  and Pi DNS. Results go to `rpi400-server` `cvdl-inbox` through write-only
  rrsync into `/srv/cvdl-results`. Masters upload from the PC; the cache is
  rebuilt on G4 with 40 workers.

#### First real G4 setup

- The VM booted at 11:25:03Z and Worker_2 shut it down at 12:03:30Z,
  about 38 min later. It had an NVIDIA RTX PRO 6000 Blackwell Server
  Edition, 97,887 MiB, driver 595.91.07, CUDA available in torch, 48 vCPUs,
  and 176 GiB RAM.
- `setup.sh base` completed. Tailscale was 100.70.184.36; Tailscale SSH
  was not used, and the user disabled key expiry. The inbox key was installed
  and the job service was enabled.
- One upload stream reached 3.5 MB/s; 10 parallel streams reached 20-21
  MB/s. The 22.15 GB upload completed at 11:50:16Z. Counts and bytes matched
  the PC: RAISE 1,011 files / 18,167,565,575 B; DIV2K train 800 /
  3,533,759,878 B; DIV2K valid 100 / 449,507,452 B.
- The 40-worker cache build took 363.047 s for 1,900 entries and
  52,761,268,224 array bytes, about 145.3 MB/s average. Strict verification
  passed with 0 hash mismatches, and the normalized record digest equals the
  PC/VM reference. Free disk after was 114 GB.
- G4-to-Pi inbox delivery passed and the test result was removed.

#### G4 setup findings

- Spot preemption graphs rose over the last about 3 days in several regions:
  us-central1 reached 25% and europe-west2 18%. No region was clearly safer.
  us-central1 had High availability and was the cheapest tested option.

#### G4 speed and first-job proof

- Speed test on 1,381 photos and 107,900 images per epoch:
  - Cold reads: cache 427.2 MB/s; masters 435.2 MB/s. Before the disk
    change, reads were about 145 MB/s.
  - GPU-only: compiled 2,243.5 images/s (about 70 it/s at batch 32);
    eager 1,635.4 images/s.
  - Cache loader plus GPU: compiled 2,140-2,147 images/s; eager
    1,570-1,584 images/s. These were short, partly warm windows.
  - Cache loader only at 4/8/16 workers: 1,208 / 1,889 / 2,212 images/s.
    Decode loader only at 16/24/32/40 workers: 1,883 / 2,702 / 3,003 /
    3,193 images/s, using 15-20 of 48 cores.
  - First-epoch lower bounds: cache 87.5 s, disk-bound at 37.4 GB; decode
    48.1 s, GPU-bound at 15.7 GB of masters. A compiled epoch is about 48 s
    on G4 versus about 4.1 min on the PC.
- First proof booted at 13:13:08Z, started the job at 13:13:20Z, uploaded to
  the Pi at 13:13:48Z, and shut down at about 13:14Z.
  - Notebook execution failed at cell 12 in the validation loader. Spawned
    workers raised `FileNotFoundError` in `SemLock._rebuild` after the watch
    SSH session closed. Ubuntu logind `RemoveIPC=yes` deleted the user's
    POSIX semaphores in `/dev/shm`. The open-file limit was 65,536.
  - Data source decode was confirmed. The fix is commit `6e0323a`, which
    makes `RemoveIPC=no` and restarts logind. It is not yet applied on G4.

#### G4 job proof 2

- The runner ran from 13:22:51.96Z to 13:26:11.93Z (199.96 s). Pi upload
  finished at about 13:26:14Z, and the VM was offline by 13:26:29Z, about
  4 min uptime.
- The Pi result was done, exit 0, with 15/15 code cells and 0 tracebacks.
  It contains the checkpoint, test predictions, heatmaps, and curves.
  The best checkpoint was
  `best-epoch01-val_pe0.1562-val_auc0.8789.ckpt`.
- Cell times: settings 8.5 s, loaders 17.6 s, fit 106.3 s (epoch 0
  includes about 65 s compile), evaluation 45.3 s, and Notebook 02 17.3 s.
- QUICK test results were accuracy 0.7812, AUC 0.8633, and P_E 0.1250.
  These are QUICK results only and are not real measurements. Peak sampled
  GPU memory was 1,521 MiB, but sampling ran only 60 s and was sparse.
- This confirms the `RemoveIPC` cause: with the same settings and no SSH
  session ending, the run did not fail.

#### RemoveIPC and first full G4 run

- The semaphore survived a full SSH logout for a `setsid` holder and for a
  `systemd-run --uid` service holder.
- G4 booted at 14:02:12Z. The runner ran from 14:02:26Z to 14:29:08Z
  (1,602.27 s), with 15/15 cells and exit 0. Pi upload finished at
  14:29:12Z with 50 MB. The VM was offline at about 14:30-14:31Z. Billed
  time was about 28.5 min, about $0.77.
- Fit took 967.3 s (16.1 min). Epoch 0 took 2:25 at 23.2 it/s, including
  compile. Epochs 1-9 took about 1:00 each at about 55 it/s (about 1,770
  images/s). Validation took 11-12 s per epoch.
- The GPU was 100% busy in every training sample, with peak memory 3,645 MiB
  and about 420 W. CPU use was about 36-40%. The GPU did not wait for the
  decode loader. LSB embedding runs on the CPU in the loader workers
  (`lsb_torch.py`). Steady training was about 55 it/s, below the 70 it/s
  GPU-only test and the 67 it/s short loader+GPU test. The cause is not
  measured. Candidates: Lightning per-step overhead (logging, metrics) and
  CPU contention for the main process.
- Cell 29 diagnostics took 499.5 s, 31% of runtime, with the GPU idle.

#### Parallel diagnostic preload

- On the PC in decode mode, 1,000 pairs per split, old serial preload took
  299.95 s for train (611 photos) plus 121.69 s for validation (261 photos),
  421.6 s total. GPU checks took seconds.
- New preload times, train / validation:

  | Workers | Train | Validation |
  | ---: | ---: | ---: |
  | 4 | 91.07 s | 41.98 s |
  | 8 | 57.63 s | 29.57 s |
  | 15 | 53.42 s | 30.83 s |

- SHA-256 values for all 2,000 pairs were identical before and after at 4,
  8, and 15 workers: train `35b79849...f2f0`, validation
  `17eda82c...2f16`. Cache output had the same bytes. QUICK CHECK lines and
  diagnostic pair IDs were identical. All 15 cells passed.
- With the cache, parallel preload was slower: at 8 workers, train /
  validation was 11.80 / 10.12 s versus serial 8.14 / 4.59 s. Use
  `V3_DIAG_WORKERS=0` for cache runs if needed.
- Tests: `test_lsb_torch` 16 passed and `test_notebook_diagnostics` 3 passed.

#### G4 epoch-start profile

- Steady G4 speed: raw fixed data 69.79 it/s (speed-test style 68.21);
  Lightning fixed data 69.43-69.51 with progress bar, callback, and logging
  on or off; Lightning plus decode with 32 workers 66.89-67.10; Lightning
  plus warm cache with 24 workers 67.04-67.14. The loader adds about 0.5 ms
  per step.
- In `g4-full-1`, epochs 1-9 took 11-12 s for the first 75 steps, then
  about 68.5 it/s steady (step 150 at 13 s and step 3,300 at 59 s). The
  reported 55 it/s is the epoch average, including this stall.
- At each epoch start, every `TrainingImages` worker fills its 1,024-image
  shuffle buffer before yielding. Persistent workers do not keep the buffer.
  The estimate of about 11 s from 1,024 images per worker at about 94 images/s
  per worker is arithmetic, not a measurement. Cache is not expected to
  remove this stall on G4.
- PC short-schedule, no-GPU time to first batch, decode 8 / 32 workers was
  6.1 / 4.8 s; cache 8 / 32 was 1.1 / 1.6 s. The buffer was not full, so
  these values are not projected to G4.

#### Persistent-worker priming

- Per-epoch SHA-256 of the full training stream, epochs 0-2, photo fraction
  0.005: old and primed streams were identical for decode/4, decode/8,
  cache/4, and cache/8 (12 of 12 comparisons). A last.ckpt resume after
  epoch 0 matched an uninterrupted run.
- QUICK notebook with priming on passed all 15 cells and cleaned up workers.
  Tests: 41 passed, including 19 `test_lsb_torch` tests.
- PC boundary timing at photo fraction 0.05, 4 workers, with validation held
  at 12 s:

  | Source | Old | Primed |
  | --- | ---: | ---: |
  | Decode, validation end to batch 75 | 11.594 s | 3.32 s |
  | Cache, validation end to batch 75 | 2.316 s | 1.154 s |

  Without forced 12 s validation, priming added about 0.41 s to validation.
  Peak G4 RAM was not measured.

- Best epoch was 8 (validation P_E 0.0142, AUC 0.9976). Test accuracy was
  0.9833, AUC 0.9976, and P_E 0.0162. Per-source P_E was RAISE 0.010948
  (9,728 pairs) and DIV2K 0.052524 (1,228 pairs).
- Comparison: L4 full OTF P_E 0.0170 (RAISE 0.011153, DIV2K 0.057410);
  PC P_E 0.0173 (RAISE 0.0115, DIV2K 0.0619). These are single-seed
  results, so small differences are within run-to-run noise. Training time
  was about 74 min on L4, about 7:20 per PC epoch with validation, and about
  1:12 on G4.

### Decided

- The cache inside the PC checkout is allowed only when git ignores it.
- PC run default (user, 2026-10-14): use the cache at
  `/work/Datasets/otf-cache/v1`, 4 training workers, and
  `V3_COMPILE=default`. Four workers are not measured yet; confirm them in
  the next full PC run. The expected speed is about 13.86 it/s.
- Plan steps 15, 16, 18, and 19 are Done. Step 17 is deferred. Mode 2 is
  on-the-fly only for now.
- Fix 5 is Done for evaluation (`cc3aeed`).
- G4 decision (user, 2026-10-14): use `rtx6000pro` in `us-central1-c`,
  project `personal-compute-511601`, `g4-standard-48` Spot at $1.62936/h,
  with a 2 h maximum run and Stop action. This replaces the refused
  `europe-north1` location. Keep it separate from the Seoul L4 project and
  billing pool. Commit `2b84127` points submit.sh and the docs to
  `us-central1-c` and $1.63/h.
- Disk decision (user, 2026-10-14): raise throughput to 400 MB/s with
  3,000 IOPS, about +$10.40 per month. The VM was stopped for this change.
- Start without metadata is now off: the service sends pending results and
  shuts down. The next boot must set `job-mode=maintenance` in the console.
- Do not SSH into G4 during a job. Watch through the Pi. Apply the
  `RemoveIPC=no` fix by hand in a later maintenance boot.
- G4 data source decision (user): use decode. Forty decode workers give 1.5
  times the GPU rate, and no new caches are needed for new datasets. Decode
  and training at the same time are not measured yet.
- Epoch-start stall decision (user approval pending): Option 1 is to prime
  the next epoch's loader during validation. It keeps the sequence
  byte-identical but needs more engineering and memory. Option 2 is a smaller
  `buffer_size`; it changes the shuffle and is rejected for now. Cache is
  doubtful as a fix because each worker fills its buffer at every epoch start.
- Source diversity is tentative, not final: RAISE plus ALASKA#2
  (uncompressed colour PPM from alaska.utt.fr, not the Kaggle JPEG set,
  CC BY-NC-ND) plus LSDIR (training subset and official validation/test;
  image licence to check). DIV2K leaves the plan, but its files stay on
  disk. Re-evaluate the current best model on the new test sets as the new
  baseline. Balance sampling by source and hold out some ALASKA cameras for
  testing.

- G4 policy: timeout defaults to 100 min and refuses values above max-run
  minus 15. Missing job mode is off. Maintenance runs only when explicitly
  set. Each job checks out the job commit and runs `uv sync --locked`.
- G4 decode decision (user, 2026-10-14): 32 decode workers are the default
  for G4 runs.

### Next

- Pending: confirm the 4-worker PC setting in the next full PC run.
- Done: limited compile helper processes with `TORCHINDUCTOR_COMPILE_THREADS=4`
  (`22e61a9`).
- Done: VM fix 4 and fix 5 measurements. Fix 4 has no reliable VM gain;
  fix 5 was already verified above.
- Done: removed VM scratch `~/otf-dev/eval-yx` after the VM returned.
- Done: proof 2 (`g4-quick-proof-2`) passed end to end without SSH.
- Done: applied `RemoveIPC=no` on G4 in maintenance mode.
- Done: first full 10-epoch G4 run using decode (`g4-full-1`).
- Pending, optional: sample `nvidia-smi` every 10 s and include `boot.log`
  in the uploaded folder.
- Pending: G4 measurement run `g4-prime-1` with decode, 32 workers, and
  `V3_PRIME_NEXT_EPOCH=on`. It also measures the new cell 29 time. Then decide
  the default.
- The user stopped for the day at 2026-10-15 00:18 +08.
- Pending: the DIV2K weakness remains (P_E about 5 times RAISE); continue
  the source-diversity plan.
- Pending: dataset download plan, per-source balanced sampler, and licence
  checks.
- Later: Part 3, gcloud on the Pi.
- Pending: user decision on moving the VM to Taiwan if Seoul stays Limited.
- Done (user, 2026-10-14): chose option A, keep the photo-owned evaluation
  loader for all sources as one code path. Accepted cache-mode cost: about
  +3.7 s per PC validation (about +45 s per 10-epoch PC run, about 1.5%),
  about +6.3 s per VM validation, and about +0.4 GB peak PSS per evaluation
  worker.
- Done: fix 4, background join and pinned copy (`05ecedc`).

- Done (user, 2026-10-14): told teammates that the VM is on branch `yx`,
  the `raise512-run1` images are gone on the VM, and `V3_DATA=png` there
  needs a rebuild.
