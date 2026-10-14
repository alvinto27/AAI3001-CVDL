# Notebook 04 OTF integration and QUICK checks

Status: plan step 8 complete. Only QUICK training was run by the worker. Full
training remains a user action. Notebook 04 and notebook 02 saved outputs and
metadata are unchanged. No file under `notebooks/runs/` was edited.

## Controls

`V3_DATA=otf` is the new default. `V3_DATA=png` retains the old dataset, split,
augmentation and float32 loading path. Default OTF names end in `_otf` so they
cannot silently resume the PNG baseline. `V3_RUN_NAME` overrides the name.

| Environment variable | Default / meaning |
| --- | --- |
| `V3_SOURCE` | decode unless `V3_CACHE_ROOT` is supplied; explicit override wins. |
| `V3_CACHE_ROOT` | Empty; required for cache source. |
| `V3_MAIN_THREADS` | 1; explicit main-process torch CPU thread count, also in reused kernels. |
| `V3_WORKERS` | 6 training workers. |
| `V3_EVAL_WORKERS` | 4 for val, test and train-eval; nonpersistent, released after each pass. |
| `V3_DIAG_WORKERS` | `V3_EVAL_WORKERS`; reads the diagnostics cell's 1,000 train and 1,000 validation pairs, one photo per task, in the original order. |
| `V3_PREFETCH_IMAGES` | 128 images ahead per worker; rounded up to whole tasks. |
| `V3_PREFETCH` | Unset; optional old group-count override for training. Explicit 4 means 16 images ahead with groups4, not the new default. Ordered delivery remains enabled. |
| `V3_TILE_SIZE` | 512; 256 supported. |
| `V3_RUN_SEED` | 0; training sample/selection seed. |
| `V3_EVAL_PROFILE` | overlap_full_mix; legacy_v6 is an option. |
| `V3_TRAIN_PHOTO_FRACTION` | 1 normally, 0.005 in QUICK, seeded per source/camera. |
| `V3_SOURCES` | raise,div2k; affects training, not fixed evaluation. |
| `V3_OUTPUT_ROOT` | Repository checkpoints directory; scratch for agent checks. |
| `V3_RUN_NAME` | Distinct run name; agent uses dated smoke names. |

Recommended VM controls: `V3_SOURCE=cache`,
`V3_CACHE_ROOT=/srv/aai3001/otf-cache/v1`, `V3_WORKERS=6`,
`V3_EVAL_WORKERS=4`, `V3_MAIN_THREADS=1`, `V3_PREFETCH_IMAGES=128`.
Leave `V3_PREFETCH` unset to derive 32 four-image training groups. Evaluation
uses the same image budget rounded to whole-photo tasks. At 512, validation
uses one photo task (up to 140 images) per worker in flight. Its workers stop
after each pass, so their photo arrays do not occupy RAM during
training. The record includes both worker counts, thread count, requested image
budget, effective training lead, derived factors and any explicit override.
The notebook prints the full settings/selected-photo record and saves
`data_run_settings.json` beside checkpoints. QUICK fixed eval subsets have
16 pairs each, chosen with fixed seeds by source/camera. Their pair IDs and the
later diagnostic pair IDs are recorded. Source splits come from the frozen
manifests, never a new OTF PNG split.

## Cell changes

| Notebook / cells | Change |
| --- | --- |
| 04 / 2 | Data switch, environment controls, full record, isolated output/name controls and OTF name suffix. Existing ROCm conv1/CUDA model setup remains. |
| 04 / 5, 11 markdown | Explain OTF controls, QUICK selection, groups and dtype. |
| 04 / 6 | OTF label/settings counts from pair metadata; PNG labels read unchanged. |
| 04 / 8 | Frozen OTF photo split/overlap checks; old PNG make_pairs/source_split branch retained. |
| 04 / 10 | Pair inspection reads OTF pair pixels, or original PNGs. |
| 04 / 12 | OTF datasets/metadata subsets, group loader, uint8 device equality and epoch checks; original PNG loaders retained. |
| 04 / 19 | Float conversion on device, shared epoch callback, saved settings record. Set next epoch at epoch end before persistent-worker prefetch; set current epoch at fit/start too. |
| 04 / 21 | Float device conversion for predictions; self-contained OTF pair/source/camera/settings/bpp/clipped fields, symbolic image identity rather than PNG path. |
| 04 / 25 | Source/camera P_E and complete OTF bpp bins with counts; no OTF rows lost. |
| 04 / 29 | OTF fixed pair diagnostics instead of PNG paths; fixed diagnostic IDs saved; old PNG sampling retained. |
| 02 / 2 | OTF predictions use their own metadata, no raise512-run1 lookup. PNG join branch unchanged. |
| 02 / 6 | Complete OTF bins and source/camera P_E rows. PNG table/bins unchanged. |

02 changes are code cells only. All notebook metadata and saved output objects
compare identical against the preceding commit; only source text changed.
04 cells 14/16 (model/probe), metric implementation check, save, and invocation
of 02 retain their roles. New `lsb_notebook.py` contains adapters only; source
transforms, settings and embedding still use the shared modules.

## Cross-worker mixing and dtype

`training_loader` now gives each worker four images per group (two pairs in
paired mode). `MixedLoader` joins groups in the parent into the requested batch,
retains all worker tails, and reports exact ceil(total items / batch size).
Ordered groups rotate across workers. No photo/worker state enters pixel or
augmentation keys. `group_images=0` retains the old per-worker whole-batch path
for checks. Training/eval output uint8 CHW; `tensor_dtype=float32` is retained
as a reference. Notebook training, validation and prediction convert on the
device to float32, with original 0..255 values and no normalization/resize.

The uint8 shuffle buffer is unchanged: 768 MiB/worker at 512, 192 MiB at 256.
Prefetch queues contain small uint8 groups rather than whole float32
batches. The current default is 128 images per worker (32 groups of four), not
four groups. At 512 this is 96 MiB of queued image payload per worker, versus
12 MiB with the historical four-group lead; the shuffle buffer is unchanged. Joining holds at most one final batch plus groups; pinning can hold
one extra uint8 batch copy, far below the old batch/queue storage. CPU without
an accelerator does not attempt to pin unpinned groups. CUDA and ROCm use the
same data path; the existing ROCm-only conv1 model fix remains separate.

Historical step 8 check: cache source, six workers, four groups prefetched
(16 images per worker), ordered, batch32, 200 batches per case:

| Tile | Path | Mean distinct photos | Minimum photos | Mean / min workers |
| --- | --- | ---: | ---: | ---: |
| 512 | Old whole-worker batch | 13.77 | 6 | 1 / 1 |
| 512 | Four-image groups | 26.66 | 20 | 6 / 6 |
| 256 | Old whole-worker batch | 5.89 | 2 | 1 / 1 |
| 256 | Four-image groups | 18.90 | 12 | 6 / 6 |

The mixed-source mean includes smaller DIV2K photos, so it exceeds the rough
RAISE-only photo estimates. At each size 6,360 keys present in both prefixes
had identical byte hashes; zero differences. Eight loader/adapter unit tests
cover group mixing, all samples retained, paired groups, worker/epoch keys,
uint8 versus float32 equality, fixed subset and out-of-range bpp retention.
The smoke also checks device conversion exactly on a real batch.

## Speed and L4 reference

All GPU checks required idle L4 utilization zero and no compute processes.
Model: notebook v3c clip3, GroupNorm, 32-true, batch32, AdamW. No checkpoint or
model file is written by the performance helper. The first 20 steps warm up.

| Check | Result |
| --- | --- |
| Mixed cache512 dummy consumer, six workers | 71,616 images / 120.015137427 s = **596.724726 images/s**; CPU451.783%; disk154.581 MB/s; peak RSS9.643 GB / PSS8.194 GB. |
| Mixed cache512 joint, six workers, ordered/prefetch4 | 300 total steps; 280 steady / 32.144799901 s = **278.738708 images/s**; GPU100%; loader wait17.157620 ms/batch (14.945290% of elapsed time); CPU292.130%; peak RSS10.817 GB / PSS9.369 GB; disk10.442 MB/s. |
| GPU-only reference | Fixed random uint8 batch already on device, same 300-step model check; 280 steady / 31.474218303 s = **284.677443 images/s**; GPU100%; main CPU103.383%; peak PSS1.679 GB. No loader/host transfer in timed steps. |

The mixed rate is about 2.09% below the GPU-only reference, within the requested
3% threshold. The large measured host wait overlaps queued GPU work; it is not
evidence of a large starvation penalty. Each step still includes uint8-to-float
conversion on GPU. All consumer/joint worker exit codes were zero.

An initial 750-photo consumer ended naturally after 85.450320 s at 668.037290
images/s. It is not the requested 120 s result; the longer rerun above is used.
The benchmark shared stop flag ends production at the window boundary and
naturally drains only its bounded queue (6.892 s here), not the whole epoch.

## Real notebook QUICK smoke

Runner uses `TerminalInteractiveShell`, whose `run_cell` is inherited from
`InteractiveShell`, one namespace, inline matplotlib backend, scratch repo-root
working directory/path, and the same environment controls as a kernel. It
stops at the first error. All 15 code cells ran in order, including notebook02
breakdown. No cells were replaced or skipped, and no packages were installed.

Differences from VS Code: a fresh terminal IPython process, not a long-lived
Jupyter kernel; display MIME is emitted to the runner rather than a notebook UI;
there is no edited executed-notebook artifact. Saved curves/predictions/heatmaps
and checkpoints come from the actual notebook code. Checks used OMP_NUM_THREADS=1.
The user will run the real notebook in VS Code; long non-QUICK training remains
untested by design.

| Check | OTF | PNG |
| --- | ---: | ---: |
| Run name | otf-smoke-2026-10-13 | png-smoke-2026-10-13 |
| Recorded all-cell wall seconds | 100.300245 | 24.279938 |
| Printed all-cell seconds (includes record write) | 100.300541 | 24.280358 |
| Epochs / optimizer steps | 2 / 128 | 2 / 16 |
| Printed training minutes | 0.35 | 0.10 |
| Train eval images / accuracy | 508 / 0.7736 | 64 / 0.7500 |
| Val images / accuracy | 32 / 0.6875 | 32 / 0.6250 |
| Test images / accuracy | 32 / 0.7500 | 32 / 0.5625 |
| Test AUC | 0.816406 | 0.671875 |
| Test P_E | 0.187500 | 0.312500 |

OTF selected train photos: DIV2K 0032, 0133, 0282; D40 r684e92a7t;
D7000 r4cb981b2t, rc8713000t; D90 r006b0e4bt. Actual training images=508,
64 batches/epoch. Validation epoch0 loss0.638430/accuracy0.6250/AUC0.781250/
P_E0.28125; epoch1 loss0.621097/accuracy0.6875/AUC0.804688/P_E0.21875.
Train epoch0 loss0.691910/accuracy0.527559; epoch1 loss0.537328/accuracy0.771654.

| OTF test group | Pairs | P_E |
| --- | ---: | ---: |
| raise | 12 | 0.125000 |
| div2k | 4 | 0.250000 |
| NIKON D7000 | 4 | 0.125000 |
| NIKON D90 | 4 | 0.125000 |
| NIKON D40 | 4 | 0.125000 |
| not_applicable (DIV2K) | 4 | 0.250000 |

Bpp pair counts: <=0.05:0; (0.05,0.15]:1; (0.15,0.30]:10;
(0.30,0.50]:5; >0.50:0. All 16 pairs retained. Tests separately include clipped/
rounded low-end and above-0.5 examples. QUICK numbers are smoke results, not
research results or a new baseline.

The smoke printed: uint8 device equality True; same-key epoch0/1 train stego
different True; val/test unchanged True; epoch starts 0 and 1. All identical
image GroupNorm train/eval comparisons had maximum logit difference zero.
Train diagnostic a/c/e/f accuracies:0.734375/0.718750/0.937500/0.531250;
val:0.640625/0.625000/0.968750/0.312500. All checkpoint assertions, tie-metric
checks, final per-setting CSV and four heatmap checks passed.

## PNG backward compatibility and safety

Notebook02 ran before/after its adapter on a scratch copy of the existing v3c
PNG prediction file. The per-setting tables are byte-identical, SHA-256:
`26fe36f60fc5b28f48d76ad21c8875f642c8809e377ab9cf1edaa556a2b6a467`.
No original run folder was written. The PNG QUICK notebook also finished all
cells. No file under Datasets, dependency file, journal, executed record or
rocm-perf was changed. Approved cache is retained.

One initial runner preflight refused utilization4% despite no compute process;
no code cell ran. A second setup attempt used base InteractiveShell and stopped
before cell2 at its unimplemented GUI hook; using the terminal subclass fixed
it. Failed attempts have no claimed notebook results.

Step 8 verification: 82 tests passed (48 shared pipeline/loader/adapter tests,
34 engine/batch/prepare tests). Final suite emitted no termination warning.
The group-mixing unit test now drains its short iterator before shutdown.
PC/VM SHA-256 values match for every tested code/notebook file. Documentation
checks passed with 22 files and zero broken links/anchors. All step8 scratch
copies, smoke checkpoints/logs, figures and predictions were removed after
recording results; approved cache and original run folders were retained.
See [loader step 7](torch-loader-verification.md) for prior cache/session parity,
legacy profile and full split counts. Lightning arbitrary-iterable support was
checked with Context7 `/lightning-ai/pytorch-lightning/2.6.6`.

## Performance default checks (2026-10-13)

The main torch CPU thread default is now 1. Evaluation uses four nonpersistent
workers. OTF training derives its task lead from 128 images per worker:
32 tasks of four images. The explicit `V3_PREFETCH` group override still works.
These controls do not change samples, augmentation keys, split order or model
code. Both fresh before/after QUICK processes already had one main thread;
the explicit setting also prevents a reused kernel from keeping a larger count.
The real notebook control was checked with a simulated reused16-thread process:
it reset to1 by default, then respected an explicit3-thread override.

### Parity and full validation

The final suites passed 85 tests: 51 shared tests and 34 generator tests.
At both 512 and 256, 200 training batches (6,400 images) have identical keys,
labels and pixels in the same order at factors 4 and 32, with eight workers.
Evaluation at eight and four workers likewise matches for 200 batches
(3,200 pairs) at each size. All measured training worker exit codes were zero.
This does not claim training order is unchanged when training worker count
changes; photo sharding already depends on that count.

Full decode validation has 11,238 pairs. Eight persistent workers in a fresh
process took 103.56011892899551 s (217.03335446544187 images/s). Four
nonpersistent workers took 135.99441288999515 s (165.27149551489785 images/s).
Peak summed evaluation-worker PSS was 2,989,032,448 bytes. All six metrics are
identical: loss 0.7803089618682861, accuracy 0.5, P_E 0.47997865080833435,
AUC 0.513175904750824, false alarm 0.0, missed detection 1.0.
An interim two-worker pass took 137.55978315700486 s; it is not the default.

A frozen-weight Lightning fit ran two sanity validation batches, one zero-rate
optimizer step, then the full four-worker validation pass. Weights stayed
identical and all six metrics matched the values above.

### Real asynchronous Lightning windows

Every case uses the exact notebook v3c/clip3/GroupNorm model, float32
`32-true`, no autocast, batch32 and 300 optimizer steps. The first 30 warm up;
the measured 270 contain exactly 8,640 images. Synchronization occurs only at
window boundaries, not each step. Rates are total images divided by elapsed
completion time. GPU-completion interval p95 uses events; host callback p95
uses CPU timestamps. All cases start from the same model state SHA-256
`91bab9c3a59896faa7ba386c715d41892925ac19072cf21538ed24c70501e590`.
GPU-only freezes the first real mixed batch on the device. Its first batch
matches the old/new cases in keys and pixel hash within each platform.

| Platform/case | Steady seconds | Images/s | GPU-completion p95 ms | Host callback p95 ms |
| --- | ---: | ---: | ---: | ---: |
| PC, fixed real GPU batch | 23.60652385999856 | 366.0005196546768 | 89.82775573730468 | 163.42893945002288 |
| PC, decode8, old factor4 | 29.872453932999633 | 289.22967022992134 | 337.70809173583956 | 779.387333050181 |
| PC, decode8, derived factor32 | 22.471585875995515 | 384.4855475567204 | 82.8984806060791 | 151.87416515036603 |
| L4, fixed real GPU batch | 30.363586276005663 | 284.55136759743107 | 113.26376914978027 | 113.07677315271576 |
| L4, cache6, old factor4 | 31.15216202800366 | 277.3483263291078 | 117.03321914672851 | 116.93229040065489 |
| L4, cache6, derived factor32 | 31.283151044000988 | 276.1870115912396 | 118.00458030700683 | 117.45694034871121 |

The new L4 rate is 2.9394889495048764% below its GPU-only reference. The new
PC rate is 5.050546900721431% above its fixed-batch reference (signed gap
-5.050546900721431%). This negative gap is an observation, not proof of
negative loader cost: repeated fixed input and changing training input are not
identical workloads. A matched random fixed-batch PC control gave
344.7630100971867 images/s. The reason for the PC reference difference remains
open; a future strict comparison should replay the same device-resident batch
sequence. Do not compare these rates directly to the earlier per-step
synchronized 209/312/369 measurements.

New PC peak PSS: main 4,678,875,136 bytes; eight train workers 11,145,823,232;
four-worker preview 2,058,414,080. New L4 peak PSS: main 2,754,504,704 bytes;
six train workers 7,474,259,968; four-worker preview 1,590,490,112. Both have
zero evaluation-worker PSS during training. Full PC evaluation peak is recorded
separately above. All train worker exits were zero. VM checks required zero L4
utilization, no compute process and no notebook kernel before each test. They
used mirrored scratch code, the retained cache, and a read-only live checkout.

### Photo-owned fixed evaluation (2026-10-14)

Val, test, and the notebook's 1,000-pair train-eval subset now assign each
photo's pairs to one worker task. The parent restores the fixed pair-table
order and original batch boundaries, including the stratified subset order.
Each evaluation iterator uses four nonpersistent workers by default; zero
workers also works. `V3_PREFETCH_IMAGES` now applies to a *whole photo task*:
round 128 images up by the largest task size in that split. For 512 validation,
one task can contain 140 images, so its factor is one. Training is unchanged.

Full-pass read-only checks (512, batch 16, four workers, uint8, no pinning)
used the same SHA-256 over ordered clean/stego batch bytes and pair indices
before and after. All 11,238 validation pairs had SHA-256
`8646835299a9f6936acf7cd1fb4165bfa8b8ef043ffe76e7fea06fd3552053de`.

| Platform/source/split | Old reads / seconds / peak worker PSS | New reads / seconds / peak worker PSS |
| --- | --- | --- |
| PC decode val | 810 / 135.627 / 2.409 GB | 270 / 47.174 / 2.808 GB |
| PC cache val | 810 / 19.879 / 2.081 GB | 270 / 23.620 / 2.462 GB |
| VM cache val | 810 / 68.603 / 1.880 GB | 270 / 74.897 / 2.273 GB |
| PC decode test (10,956 pairs) | 745 / 133.784 / 2.329 GB | 249 / 46.561 / 2.783 GB |

The full test pass has identical before/after SHA-256
`455c0a2f3e5fc146e59988d75a7835d6b0c5cf834d8747463db99609e2732e77`.
Photo-owned decode cuts full-pass time by about 2.9x. Cache is slower by
18.8% on this PC pass and 9.2% on the VM pass. This is a measured trade-off,
not a claim that cache evaluation is faster. In a separate warm-cache pass
without hashing, parent-side `torch.stack` for 703 restored batches took
5.392 s of 17.001 s total. Old loading collated those batches in workers;
photo-sized transfers and parent regrouping add cost when source reads are
already cheap. The independent warm-cache pass is not a matched old/new speed
comparison. Retain the photo-owner policy because it removes duplicate decode
reads and restores exact outputs; a copy-free parent path needs a separate
worker-lifetime and memory test.

All times include startup and shutdown; PSS samples are summed worker peaks,
not parent or GPU memory.

The PC full-stream identity checks use batch16 and hash each ordered batch's
clean pixels, stego pixels and pair indices. The pair position fixes its
clean=0/stego=1 labels. At 512, all val, test and train-eval (1,000 pairs)
streams matched the old zero-worker loader in decode and cache with zero, two
and four new workers. New photo reads equalled distinct photos: val 270,
test 249, train-eval 611. The 512 train-eval digest was
`436ad58b9095198f9973611fa1b02ba02e70df3fd6ea266510e783356faf40c0`.

At 256, PC cache streams matched at zero, two and four new workers for
42,204 validation pairs (270 photos; SHA-256
`a054df0ff5cf3d1cc690da903560229bf464f85b624813389c59ca7d2def7c63`),
41,088 test pairs (249 photos; SHA-256
`547861c2c800c7ee4b4a1418d3914bc5f9d6a766f371d896c53acfc7a2f786e1`),
and 1,000 train-eval pairs (554 photos; SHA-256
`142ca70569066ce2987cf767c7bc920175fd0e0aed2b6e5bbe62515d0104c1f4`).
Old zero-worker decode and new two/four-worker decode train-eval matched
this last digest; new decode read exactly 554 photos instead of 1,000.
Old and new zero-worker decode validation also matched the cache digest.

After the Spot VM returned, all 16 saved VM val/test 256 identity files
were reviewed: old zero-worker and new zero/two/four-worker streams matched
in both decode and cache, with 270 val / 249 test reads for every new pass.
The remote benchmark helper had omitted `fixed_subset` for the train split:
its seven saved train files instead process all 202,572 train pairs and do
**not** verify notebook train-eval. The correct 1,000-pair train-eval checks
above were performed on the PC. One extra VM full-train case was interrupted
by preemption and is not needed for notebook train-eval. The VM 512 full-cache
validation benchmark above used code files whose SHA-256 values match the
final local code and was completed before the preemption. No VM measurement
needed for this task remains Pending.

A read-only fixed checkpoint (`best-epoch07-val_pe0.0149-val_auc0.9976.ckpt`)
produced 21,912 test prediction rows with identical before/after CSV bytes,
SHA-256 `9ab54c66e9e098126b76441e284294e61045d185043de19dd26efa138495b6f6`.
No original checkpoint or output was edited. A 200-batch, 6,400-image
six-worker Mode 1 training prefix retained SHA-256
`32bd10352c3e2c3d6a5c4600632bd0f962c9528975572e162bb973f7acc99075`
(batch image bytes, labels, then keys). Unit checks passed 13 loader tests,
74 shared tests, and 34 generator tests. A scratch PC cache QUICK run executed
all 15 code cells in 55.374 s, including both epochs, sanity validation,
test predictions and notebook 02's four heatmaps. It wrote outputs to scratch only.

A direct early-close check used eight validation photos and four workers:
two-batch sanity-style close 0.645 s, one-batch iterator drop 0.619 s,
and a caught interrupt after two batches 0.614 s. An instrumented second
all-cell QUICK run took 53.769 s; actual Lightning sanity-check worker
shutdown took 0.475 s. Its six later evaluation shutdowns took
0.474–0.496 s. Each had four workers before and zero alive after. No workers
remained after the direct cases either; the 13-test loader suite emitted no
abort/shutdown warning. The loader first stops scheduling new photos and
drains only tasks already in flight before worker shutdown, rather than
generating the rest of a split.

### Background training join and pin (2026-10-14)

`MixedLoader` now runs the worker-group iteration, `torch.cat` join and
`pin_memory()` copy in one producer thread. Its output queue holds at most two
batches. The consumer still gets the same ordered batches. The thread stops
at epoch end but leaves persistent workers alive for the next epoch. A drop,
interrupt, error or `shutdown()` stops both thread and workers. `set_epoch()`
updates the shared epoch value before the retained workers start a new iterator.
No new environment setting was added.

The first PC check used the notebook XuNet clip3/GroupNorm model, AdamW,
batch32, 4 cache workers, 30 warm-up and 300 measured optimizer steps. The
model compiled only its inner network in `default` mode, as notebook 04 does.
The old main-thread costs were measured without changing production code:

| Mode | Step mean | Wait for next group | Join including pin | Pin part | Main fetch total |
| --- | ---: | ---: | ---: | ---: | ---: |
| Compiled | 70.891 ms | 0.241 ms | 6.588 ms (9.3% of step) | 4.007 ms | 7.040 ms |
| Eager | 88.616 ms | 0.218 ms | 5.163 ms (5.8% of step) | 2.847 ms | 5.559 ms |

The join plus pin cost exceeded the 2% decision limit in both modes, so the
background thread was implemented. The matched speed windows used the same
model and 9,600 measured images. Only window boundaries synchronized the GPU.

| PC mode/source/workers | Old images/s | New images/s | Change | Old / new mean main fetch |
| --- | ---: | ---: | ---: | ---: |
| Compiled/cache/4 | 467.441 | 471.979 | +0.97% | 6.916 / 1.643 ms |
| Compiled/cache/4 repeat | 471.018 | 468.836 | -0.46% | 6.709 / 1.756 ms |
| Eager/cache/4 | 356.897 | 379.820 | +6.42% | 5.408 / 0.051 ms |
| Compiled/decode/6 | 457.638 | 470.392 | +2.79% | 14.759 / 9.527 ms |

The compiled/cache difference changes sign on repetition. It is **not** a
reliable model-speed gain; eager/cache improves in this window. The joined
batch remains pinned when requested; the unpinned and zero-worker routes also
pass. At 512, each joined batch contains 32 x 3 x 512 x 512 uint8 bytes
(24 MiB). Two queued batches can hold 48 MiB of extra pinned image payload;
a third prepared batch waiting for queue space raises the bound to 72 MiB.
Labels and keys add less than 2 KiB at this bound. These are capacity bounds,
not a measured increase in process PSS. No VM speed measurement was run while
the Spot VM was unavailable; VM speed is Pending.

The old/new 200-batch stream hashes match at 0, 2 and 4 workers, and the
six-worker stream retains SHA-256
`32bd10352c3e2c3d6a5c4600632bd0f962c9528975572e162bb973f7acc99075`
for 6,400 images. The producer transfers its original exception traceback to
the consumer; a unit test verifies it. In a four-worker PC cache check, cleanup
after an early iterator drop took 0.469 s, a caught interrupt 0.471 s, and
explicit `shutdown()` 0.474 s. Each left zero producer threads and workers.
A two-epoch drain with `set_epoch()` also left zero threads and workers in
the first implementation; the persistent-worker correction below supersedes
that worker lifetime only. A scratch PC QUICK run executed all 15 notebook
code cells in 53.240 s; Lightning fit end had zero producer threads and zero
workers. The first suite had 75 shared tests and 34 generator tests. Notebook
04, dependency files, checkpoint originals and the project journal were not
changed.

#### Epoch-end race and persistent workers (review correction)

If a queue read times out as the producer exits, the consumer checks the queue
once more before it reports a missing result. A unit test forces the producer
to put `done` between the timeout and the thread-state check. Normal epochs now
retain the same persistent worker PIDs, as the old loader did. Early drop,
interrupt, error, and `shutdown()` still stop them. The `fe23238` loader
and corrected loader each processed complete earlier epochs at train photo
fraction 0.07,
then hashed the first 200 batches of epoch 2 and epoch 3 (32 images per batch).
Each hash covers ordered image bytes, labels and keys. All six old/new hashes
match. Four workers retained the same four PIDs through three epochs.

| Workers | Epoch 2 SHA-256 | Epoch 3 SHA-256 |
| ---: | --- | --- |
| 0 | `1ac2e2b84bdb461ba742128bd6d7e54ab35b24ca374643669d72f77473973f18` | `ba83480f5a8cff191f034459b51436de1b928a8899c6b501072d2998263fff4d` |
| 2 | `ab9bc807c47bfcad281f3e370642b0d3af2369e8be5b3716fe46b0939dd0e62b` | `90edb86b77820c89217883622df94d10b20c3d645a024e281f7ff49146b38d1c` |
| 4 | `ef7739e6431a41bf0af6415f863b016a17e41de00d2e51e432f410bf2525c4ff` | `e19d5f6d178961c09a003d9a7132ac2d5c1f90aad352546bae25e1b3bcddea0e` |

Time from `iter(loader)` through the first batch, PC cache, seconds:

| Workers | Old epoch 1 / 2 / 3 | Corrected epoch 1 / 2 / 3 |
| ---: | --- | --- |
| 0 | 1.100 / 0.764 / 0.750 | 1.055 / 0.713 / 0.735 |
| 2 | 3.104 / 0.778 / 0.766 | 3.089 / 0.788 / 0.763 |
| 4 | 5.201 / 0.859 / 0.865 | 5.193 / 0.871 / 0.861 |

After a normal epoch four workers remained alive; a dropped next-epoch
iterator stopped all four in 0.466 s, a caught interrupt in 0.465 s and
`shutdown()` in 0.464 s. Each left zero producer threads and workers.
The corrected PC cache QUICK run passed all 15 cells in 48.796 s; fit end had
zero threads and workers. Final suites: 76 shared and 34 generator tests pass.
The VM remains unavailable for speed measurement.

### Why four-worker decode validation remains slow

A controlled first-1,600-pair check used 100 full32-image batches:

| Workers | Pass seconds | Full-photo loads | Summed photo-load seconds | Summed item-generation seconds |
| --- | ---: | ---: | ---: | ---: |
| 2, nonpersistent | 24.43056574500224 | 51 | 38.28914535402146 | 41.00799804187409 |
| 4, nonpersistent | 25.835603224004444 | 101 | 78.55817700300395 | 81.50152954602527 |
| 8, persistent, fresh | 24.734317536000162 | 122 | 101.94830253097462 | 105.23241107203648 |

Four workers repeat almost twice as many whole-photo loads as two: ordered map
batches distribute tiles from the same photo to different workers. Photo loading
and decoding account for about96.4% of four-worker item-generation time. Equal
2/4 pass time therefore does not show that decoding is unlimited.

At four workers, process-ready delays span1.0161152510045213–4.017964180005947 s.
First-photo loads take0.6674019240017515–0.6784606350047397 s; first fetch takes
0.7260237510054139 s. Remaining fetch waits sum18.00667567698838 s, with
795.1887818999239 ms p95. GPU forward totals2.4190234146118166 s (median
24.02033233642578 ms). Device transfer totals0.09216743254661561 s; host
transfer calls0.010391062052804045 s. Validation-step host spans total
2.6321538840056746 s, mainly including GPU completion. Epoch AUC/P_E work takes
0.002354937998461537 s. Main-process CPU totals3.52 s. In that earlier
check, evaluation prefetch was four full batches/128 images.

A two-epoch, one-step-per-epoch, zero-rate Lightning fit verified true retained
worker reuse: the same eight PIDs took24.04181039600371 s on the first prefix
pass and16.056130497003323 s on the next. Startup retention saved about8 s.
Both passes loaded122 photos; the last-photo arrays did not remove restarted
prefix decoding. Separate `Trainer.validate` calls tear workers down and are
not a valid warm-worker comparison. The full103.56 s before-pass above was
fresh, so earlier warm workers did not cause that particular advantage.

The preceding measurements and proposal describe the earlier pair-batch
loader. The photo-owned implementation and new measurements are in the section
above. More workers or deeper queues alone cannot remove repeated photo reads.
The four-worker nonpersistent policy is unchanged.

### Early stop, interruption and QUICK

Instrumentation wrapped PyTorch's own shutdown method without changing it.
The direct two-batch iterator drop took1.4509231980046025 s including garbage
collection; shutdown itself took1.3554129410040332 s. Lightning's two-batch sanity
cleanup took0.5337219729990466 s. A real `KeyboardInterrupt` after two validation
batches completed cleanup in0.8126337830035482 s (worker shutdown
0.8092460670013679 s). Lightning's caught process exit1 is the expected interrupt
path, not a worker exit: all four worker exits were0. No workers remained alive,
no hang occurred, and no shutdown/termination diagnostic appeared.

Both instrumented QUICK runs executed all15 code cells through terminal IPython:
OTF261.99676013000135 s; PNG12.426183101997594 s. OTF preview drop after one
batch took0.42437690000224393 s; its two-batch sanity cleanup took
0.8975795030055451 s. Other full evaluation cleanup took0.4154–0.8785 s.
PNG sanity cleanup took0.04676537499472033 s. Every observed worker exit was0;
no workers remained. Existing Lightning worker/persistence suggestions and a
pytree deprecation still print; they were not suppressed and are not shutdown
faults. The expected graceful-interruption notice also prints.

Fresh before/after OTF QUICK totals were262.75344457499887/263.0877361340026 s.
CPU tie metrics took0.0033397070001228712/0.003548035994754173 s; P_E/AUC tables
0.04228401499858592/0.04400327699841 s; diagnostics
191.73584875899542/192.34160630899714 s. These CPU cells are not clearly slower.
Saved notebook outputs and metadata are unchanged; source cells2,5,11,12 changed.
The user's executed run record, checkpoint folder, datasets, dependency files,
journal and backend settings were not edited. All scratch outputs are removed
after reporting; the approved cache is retained.

## Parallel diagnostics preload

The diagnostics cell used to decode every selected photo in the main process.
`lsb_torch.preload_pairs` now reads each selected photo once in worker
processes (`OrderedPairLoader`) and returns the pairs in the old order.
PC check with decode source and 1,000 pairs per split: tensors, dtypes, order and
`diagnostic_pair_ids` have identical SHA-256 digests to the old path (train
`35b79849...f2f0a`, val `17eda82c...2f16`; the cache source gives the same
bytes). Old serial times were 299.95 s (611 photos, train) and 121.69 s
(261 photos, val). New times: 4 workers 91.07 s and 41.98 s, 8 workers
57.63 s and 29.57 s, 15 workers 53.42 s and 30.83 s. With the cache source the
serial path takes 8.14 s and 4.59 s, so worker start-up makes the parallel path
slower there (8 workers: 11.80 s and 10.12 s).
