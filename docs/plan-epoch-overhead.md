# G4 epoch time: R2 and R1

## Scope and evidence

This is an analysis, not a speed test. No training, VM operation, or code change was done for this record. The sources are `runs-out/g4-v4-r2/` and `runs-out/g4-v4-r1b/`: `run.log`, `nvidia.csv`, `status.json`, `metrics.csv`, TensorBoard event files, checkpoint modification times and sizes, and `lr_trace.csv`. The notebook source is `notebooks/05_xunet_v4_nonorm.ipynb`. Both runs used 3,372 train batches and 703 validation batches per epoch, 32 train workers, and 16 **non-persistent** validation workers. R2 ran 30 epochs; R1 ran 20. The train workers are persistent. Both runs enable next-epoch priming.

**Time rules:** `M` means measured in a log or file time. `D` means calculated from those times. `E` means an estimate that needs a timer to confirm. The progress bars show whole seconds. The epoch bar starts before the first train batch. Its time at batch 3,372 is the train-bar time. The validation bar times 703 batches, but does not separately time worker creation or shutdown. The epoch bar at the next epoch boundary gives the full epoch-bar time. Thus **transition time = full bar − train bar − validation bar**. This is a measured residual (D), **not** a measured worker or priming time. `metrics.csv` has epoch and step, but no wall time. TensorBoard `val_pe` records one wall time per epoch, not per batch. The checkpoint times below come from the copied files and preserve the Pi file times.

## Where the time goes

| Run and epochs | Train bar, s/epoch (M) | Validation bar, s/epoch (M) | Transition, s/epoch (D) | Full bar, s/epoch (M) | Checkpoint interval, s/epoch (M) |
| --- | ---: | ---: | ---: | ---: | ---: |
| R1, epoch 0 | 87 | 14 | 20 | 121 | — |
| R1, epochs 1–19, mean | 46.05 | 12.21 | 19.42 | 77.68 | 78.11 |
| R2, epoch 0 | 87 | 14 | 20 | 121 | — |
| R2, epochs 1–19, mean | 46.32 | 12.21 | 20.05 | 78.58 | 79.01 |
| R2, epochs 1–23, mean | 47.09 | 12.30 | 20.00 | 79.39 | 79.78 |
| R2, epochs 24–29, mean | **101.33** | **27.17** | 19.17 | **147.67** | **148.19** |

The train-bar time includes the first train batch. In epoch 0 it includes **43 s** for the first batch and compile warm-up (M, but compile is not isolated from first-batch input). In epochs 1 and later, the first batch reads 0 s at bar resolution; the time from the previous checkpoint to the next first batch is less than about 1 s **by the bar and checkpoint clocks** (D, coarse). The duration from *completion* of the first batch to completion of the last is about train-bar time minus first-batch time: 44 s in epoch 0 and about 41–110 s in later R2 epochs (D, ±1 s). There are no per-batch wall times to measure a narrower first-to-last range. In particular, the roughly 20 s transition is **not** an epoch-start wait: it occurs after the train bar reaches 3,372 and before the full epoch bar ends.

R2 per-epoch detail (seconds; the first four columns use progress-bar whole seconds, so small differences from file times are normal):

| Epoch | Train M | Validation M | Transition D | Full M | Checkpoint-to-checkpoint M |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 87 | 14 | 20 | 121 | — |
| 1 | 41 | 12 | 20 | 73 | 73.75 |
| 2 | 43 | 12 | 20 | 75 | 75.08 |
| 3 | 43 | 12 | 20 | 75 | 75.50 |
| 4 | 44 | 12 | 21 | 77 | 76.71 |
| 5 | 44 | 12 | 20 | 76 | 76.54 |
| 6 | 44 | 12 | 20 | 76 | 76.42 |
| 7 | 45 | 12 | 20 | 77 | 77.68 |
| 8 | 45 | 12 | 21 | 78 | 78.13 |
| 9 | 46 | 13 | 20 | 79 | 79.15 |
| 10 | 47 | 12 | 19 | 78 | 78.18 |
| 11 | 48 | 12 | 20 | 80 | 80.72 |
| 12 | 47 | 13 | 20 | 80 | 80.31 |
| 13 | 48 | 12 | 19 | 79 | 79.25 |
| 14 | 48 | 12 | 20 | 80 | 80.88 |
| 15 | 49 | 12 | 20 | 81 | 81.86 |
| 16 | 49 | 12 | 20 | 81 | 81.91 |
| 17 | 50 | 13 | 21 | 84 | 84.17 |
| 18 | 50 | 12 | 20 | 82 | 82.79 |
| 19 | 49 | 13 | 20 | 82 | 82.08 |
| 20 | 50 | 12 | 20 | 82 | 82.35 |
| 21 | 51 | 13 | 20 | 84 | 84.18 |
| 22 | 51 | 13 | 19 | 83 | 83.64 |
| 23 | 51 | 13 | 20 | 84 | 83.65 |
| 24 | 86 | 25 | 20 | 131 | 132.23 |
| 25 | 100 | 26 | 20 | 146 | 146.33 |
| 26 | 102 | 27 | 20 | 149 | 149.26 |
| 27 | 104 | 28 | 19 | 151 | 151.30 |
| 28 | 106 | 28 | 20 | 154 | 154.67 |
| 29 | 110 | 29 | 16 | 155 | 155.37 |

**Whole fit cell:** R2 took 2,868.338 s (M), or 95.61 s for each of 30 epochs (D). The 30 epoch bars add to 2,833 s (M); the other 35.34 s (D) belongs to fit setup, the two-batch sanity check, final cleanup, and bar rounding. R1 took 1,627.427 s (M), or 81.37 s for each of 20 epochs (D). Its bars add to 1,597 s; the other 30.43 s has the same types of work. Do not divide the 35 s setup by each epoch and call it validation time. The nominal 3,372 / 74 = 45.57 s training estimate is close to **early** train bars, not to late R2 train bars.

R2 and R1 epochs 1–19 differ by less than 1 s in full-bar mean. The six **late R2** epochs add about 54 s to each train bar and 15 s to each validation bar compared with R2 epochs 1–23. This is why the 30-epoch R2 fit-cell average is about 14 s higher than the 20-epoch R1 average. The stable transition accounts for about 20 s per epoch in *both* runs, not for the difference. GPU samples (`nvidia.csv`, one per minute) support a late throughput fault: before the R2 epoch-23 checkpoint, sampled utilization averaged 38% and power 218 W (34 samples); from that point until 10:29 UTC, they averaged 20% and 163 W (15 samples). R1 samples averaged 37% and 224 W (31 samples, including start and end). Samples mix train and validation and cannot identify the cause. The change can be CPU, storage, input pipeline, worker scheduling, or a shared host issue; none is proved by these logs. The LR changes in `lr_trace.csv` and `metrics.csv` do not time the stall. Record CPU, I/O wait, loader wait, worker status, and per-phase GPU use before assigning a cause.

## What the residual contains

| Work | Evidence | Time inside the epoch | Confidence |
| --- | --- | --- | --- |
| Train batches, including mixed-loader join and input wait | Train progress bar; no batch wall clock | 41–51 s in early R2, 86–110 s in late R2 | M for total, no split between GPU and input |
| Epoch start wait | First step shows 0 s in epochs 1–29; checkpoint spacing tracks the full bar | Below about 1 s at bar resolution | D, coarse |
| Validation 703-batch loop | Validation progress bar | 12–13 s early; 25–29 s late | M for loop; startup is outside the bar or mixed with its first batch |
| Validation transition (worker creation and shutdown, `prime_next`, score sorting, logging, curve redraw, checkpoint hooks) | Full bar minus train and validation bars | About 20 s, stable; **no exact sub-phase split** | D for total, E for any named cause |
| Checkpoint writes | One `epoch*.ckpt` each epoch (about 242–246 KiB), `last.ckpt` (about 244–247 KiB), and one best file at a time (about 244 KiB). TensorBoard `val_pe` wall time to epoch checkpoint mtime is usually 0.18–0.25 s; two R2 gaps are 0.55–0.56 s. | At most a small part of the 20 s transition; write start is not logged | M for sizes and end times, **not** a direct save duration |
| Training curves, CSV, TensorBoard | `TrainingCurves` reads CSV and redraws a roughly 97–99 KiB PNG **twice** each epoch plus once at train end. `PerEpochScalars` writes and flushes 2 train and 6 validation scalars. `AppendCSVLogger` writes 2 metric rows per epoch. `V4LRTrace` writes only selected boundary steps (a few CSV lines). | Cannot isolate; likely small, but not measured | M for counts and file sizes; E for cost |
| Next-epoch prime | `OTF_Epoch.on_validation_start` calls `train_loader.prime_next`; `PrimedTrainingLoader` closes the prior producer and starts a background thread. Next epoch consumes its first ready batch. | May contribute to transition; cannot measure from these logs | Code fact for placement; time unknown |
| Unexplained transition work | No timestamps at validation start, worker setup/teardown, `prime_next`, or callback entry and exit | **Up to about 20 s each epoch** until measured | D for upper envelope, cause unknown |

`on_validation_start` runs before validation work. The code does not record how long closing the prior mixed-loader producer takes, nor how long creating and stopping 16 non-persistent eval workers takes. The current log cannot say which part of the 20 s belongs to either task. R2's last epoch has a 16 s transition and cannot prime a next epoch; the difference is consistent with some prime cost, but last-epoch cleanup also differs, so it does not measure prime time. The TensorBoard time is after validation metrics; it does not give a first-step or validation-start time. The 1-minute NVIDIA sample interval is too wide to resolve a 20 s transition.

## Fix order and test rules

First add timers around train-end, prior producer close, prime submission and completion, eval iterator creation, first and last eval batch, eval iterator shutdown, each curve redraw, each scalar flush, and each checkpoint save. Also record per-batch loader wait and GPU utilization at finer intervals. The timers must only observe; then compare the same run settings. Do **not** claim a 20 s saving from an unmeasured sub-phase. Ranked options below use an **estimated** possible saving, not a benchmark result.

| Rank / change | Potential saved time per epoch (E) | Result change | G4 96 GiB / PC 16 GiB |
| --- | --- | --- | --- |
| 1. Keep the **exact validation uint8 pairs** on GPU after the first build; use the same pair order, batch size, float conversion, and model operations. Test score equality before use. | 0–12 s in an early epoch; up to 0–27 s late **only if** validation input is the bottleneck. It cannot remove required forward compute or a separate 20 s transition by itself. First preload adds cost once. | Same source bytes and labels; bitwise scores must be checked. Model selection must not change. | 22,476 images × 3 × 256 × 256 uint8 ≈ **4.12 GiB** for validation pairs, plus indexing and batch memory. Fits G4 beside its ~3 GiB observed allocation. PC 16 GiB VRAM may fit GPU bytes, but PC host RAM is also 16 GiB and a second full host copy or worker caches may not fit. Measure peak host and GPU memory first. FP32 residence would need ≈16.48 GiB and cannot fit PC GPU. |
| 2. Keep eval workers alive, with the same dataset and fixed validation order (`persistent_workers=True`), and confirm their photo cache remains bounded. | 0–20 s **only if** worker setup and teardown dominate the transition. Likely less if producer close is the cause; time it first. | Same pairs and order if workers do not hold stale state. Compare scores and epoch order. | G4 has room for 16 workers but each worker's full-photo cache costs host RAM. PC 16 GiB RAM is a risk; use fewer workers and measure RSS before use. Persistent workers do not put all pairs on GPU. |
| 3. Fix any measured mixed-loader close or prime wait. Keep identical schedule and keys; close only after the last batch, as now. | 0–20 s **only if** the timed close or prime path accounts for the transition. | No result change if the photo schedule, seeds, ordered groups, and first batch stay identical. Verify resume and sample hashes. | Can apply to G4 and PC; memory must stay bounded (one active loader and one ready batch). Do not launch a second full worker set. |
| 4. Reduce curve redraws (one PNG per epoch or after fit) and keep both CSV and TensorBoard scalar data. | 0–2 s (not measured); checkpoint size is too small to promise more. | Scores and checkpoints unchanged; intermediate PNG history or plot freshness changes. | Both G4 and PC; negligible memory difference. |
| 5. Save small checkpoints asynchronously **after** an immutable epoch snapshot. | 0–1 s (not measured); files are only about 0.25 MiB. | Exact result only if snapshots, names, best selection, ordering, crash handling, and resume semantics are preserved. An incomplete save changes recovery. | Both; holds a second checkpoint snapshot briefly. Not a first fix. |
| 6. Validate every N epochs. | Could avoid most of a ~32 s normal validation-plus-transition epoch, **if** the transition is validation-related. | **Changes results and model selection:** no per-epoch `val_pe`, a different best checkpoint, fewer curve values, and potentially a different final selected model. Do not use for the comparable R2/R1 run. | Both; no large extra memory. |

## New read-only timers for the next run

Notebook 05 now writes `<run-folder>/epoch_timing.csv`. Each row uses the zero-based epoch. `train_first_wait_s` is from the train-epoch-start hook to the first train-batch-start hook. `train_steps_s` is from the first batch start to the last batch end. `train_end_to_val_start_s` is from the last train batch end to the validation-start hook. `val_loader_setup_s` is from validation start to validation-epoch start. `val_first_wait_s` is from validation-epoch start to the first validation batch start; it can include iterator setup. `val_compute_s` is from the first validation batch start to the last batch end; it **also includes input waits between batches**. `val_last_to_end_s` ends at the validation-end hook. `val_end_to_next_first_s` includes the callback tail, checkpoint writes, worker shutdown, and the next train loader wait. It is blank on the final epoch. `curves_s`, `scalars_s` (including its per-batch accumulator), `lr_trace_s`, `prime_s`, and `checkpoint_s` time only their named code paths. Blank means the path did not run; zero-like values mean it ran quickly. Times use `time.perf_counter`; CUDA work is **not synchronized** just to time a hook, so durations for asynchronous GPU work can appear elsewhere. The callback runs first to set the interval marks. It prints one short `EPOCH TIMING` line each epoch. Timers do not set seeds, change loader order, or call the model.

The headless G4 runner now writes `<job-id>/host.csv` every 60 s, from the runner process. It also works if the VM still runs the old installed one-file `job.sh`, because that script calls the checked-out `scripts/run_notebook.py`. `host.csv` records UTC time; total CPU busy and CPU I/O-wait percent; load averages; RAM used and available; page cache; swap used; physical disk read and write MiB/s; Python process count and total RSS; runner open-file count; and cgroup memory when available. Percentages and disk rates are blank in the first row: they need two samples. CPU use is total host busy, not one core. The process RSS is a sum and can double-count shared pages; it is **not** PSS. Disk totals include whole `nvme`, `sd`, `vd`, and `xvd` devices and exclude their partitions. The `nvidia.csv` file still samples once a minute separately. Together these files can test whether the R2 late slowdown tracks I/O wait, page cache loss, or worker memory; they cannot prove a cause by themselves. PC headless runs can opt in with `--host-monitor on`; the default `auto` turns it on only for Google Compute Engine. The standalone sampler `python scripts/host_monitor.py --output runs-out/probe.csv --interval 45 --samples 3` takes about 90 seconds and does not run a notebook.

## Wide R2 run: first measured phase split

The approved wide model run at commit `4f5fa5f` finished all 30 epochs. See [its full result](result-wide-xunet.md). The early 23 steady epochs (zero-based 1–23) had a **18.773 s mean wait from validation-epoch start to first batch**, a 13.163 s first-to-last validation batch span, and a 61.888 s first-to-last train batch span. Train end to validation start was .003 s; validation start to validation-epoch start was .017 s; validation end to the next first train batch was .457 s. Curves, per-epoch scalar work, prime submission, and checkpoint saves averaged .407, .235, .015, and .032 s. Thus most of the old ~20 s residual is **before the first validation batch**, not in checkpoint writing or prime submission. The 16 eval workers are non-persistent; their creation and first input work are still combined in this measured wait. A persistent-eval-worker test now has a precise target. It must keep validation samples in the same order.

Wide epochs 24–29 slowed too: the train span averaged 101.890 s and the validation compute span 29.236 s, while the first validation wait stayed near 18 s. In steady early versus late minute samples, host CPU busy fell 27.8→16.8%, GPU use 66.5→44.0%, and power 298→218 W. Available RAM remained above 117 GiB in those samples; swap and disk I/O wait were 0; Python RSS stayed near 10.9 GiB, GPU memory at 4,517 MiB, and page cache increased. The late slowdown is **not explained** by RAM exhaustion, swap, storage reads, or a growing worker RSS. These host samples alone cannot name the cause. The PC investigation below found a growing output-capture cost in the runner; this did not change data samples.

## PC investigation and two fixes (after the G4 runs)

**Late slowdown: output capture.** Both G4 fit logs reached **31.2 MiB at epoch index 24** and **37.9 MiB at index 29**. The old `scripts/run_notebook.py` kept all stdout from the fit cell in one string and used `text += chunk` for each progress update. Each addition copied the whole, growing string. A PC CPU-only check with 3,000 realistic progress chunks took .014 s from empty text, 2.678 s from 4 MiB, 12.957 s from 16 MiB, and **38.738 s from 32 MiB**. This accounts for the same late epoch and increasing slowdown in both G4 models; it does **not** change the data stream or checkpoint choice. The runner now holds chunks in a list and joins them once per completed cell. From 32 MiB, the same 3,000 updates took **.0048 s to append plus .0126 s to join**, with identical final text. Late writes are also normalized before saving `executed.ipynb`. This is a PC proof; no post-fix G4 speed has been measured.

A PC data-only probe used the R2 k1–4 config, QUICK's selected photos, 6 train and 4 eval workers, and **no model**. It set train epoch directly to 20–27 and read all 64 QUICK train batches and the validation batches. After the first worker spawn, train time was **1.12–1.49 s** and validation time **3.90–3.93 s**. Epoch 24 was **1.117 s train / 3.917 s validation**. The first-eight-train-batch hashes varied with the keyed epoch seed. There was no jump at 24 in this PC subset. This does not prove all full-data photos take the same time; the output-capture test and identical log-size boundary supply direct evidence for the G4 pattern.

**Persistent validation workers:** `V3_PERSISTENT_WORKERS=on|off` defaults to `on` in notebook 05. It affects validation only. Train workers already persist; test and train-eval workers still exit. An interrupted validation pass, including Lightning's two-batch sanity check, closes its workers and starts a clean full pass. In a PC parity probe with next-epoch priming, hashes of the first eight train batches matched **on versus off** at each of epochs 0 and 1. Hashes of all 32 selected validation pairs matched across **both epochs and both settings**. A partial pass restarted and yielded the same eight validation batches. The 21 loader tests, including a new persistent replay regression, passed. The unchanged standard-test settings hash was `fdd989f7bdab0ec6f9fd6f4c57bdde981d16b54aa8c37e5da8fd3dfcc8b06837` for 10,956 pairs.

| PC 6-train/4-eval-worker check | Persistent off | Persistent on | Difference |
| --- | ---: | ---: | ---: |
| Epoch-1 first validation wait, data-only probe | 4.584 s | .867 s | **3.717 s saved** |
| Epoch-1 first validation wait, full two-epoch QUICK notebook with prime | 4.780 s | .870 s | **3.910 s saved** |
| After-validation process-tree PSS | 2,880 MiB | 4,573 MiB | **1.69 GiB extra** |

Both QUICK runs passed all 16 cells. The final on run took 70.214 s and saved valid stream text in `executed.ipynb`. PSS counts shared pages once. Summed RSS rose by about 3.24 GiB, but counts shared pages more than once. A linear, **unverified** projection to 16 G4 eval workers is **~6.8 GiB extra**. Wide had over 117 GiB available late, so this appears to fit; check G4 peak PSS before making a firm claim. On PC, four retained eval workers fit in the measured memory. Do not use the G4's 32/16 worker setting on the PC.

Neither fix changes model steps, pair bytes, train order per epoch, validation order, or model selection. An approved G4 run is still needed to measure 16-worker time savings and post-fix late-epoch speed. The old G4 timing records remain valid observations of the old runner; do not treat them as post-fix measurements.
