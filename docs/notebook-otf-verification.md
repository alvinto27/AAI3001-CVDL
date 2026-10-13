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
| `V3_WORKERS` | 6. |
| `V3_PREFETCH` | 4; ordered delivery always used by the notebook. |
| `V3_TILE_SIZE` | 512; 256 supported. |
| `V3_RUN_SEED` | 0; training sample/selection seed. |
| `V3_EVAL_PROFILE` | overlap_full_mix; legacy_v6 is an option. |
| `V3_TRAIN_PHOTO_FRACTION` | 1 normally, 0.005 in QUICK, seeded per source/camera. |
| `V3_SOURCES` | raise,div2k; affects training, not fixed evaluation. |
| `V3_OUTPUT_ROOT` | Repository checkpoints directory; scratch for agent checks. |
| `V3_RUN_NAME` | Distinct run name; agent uses dated smoke names. |

Recommended VM controls: `V3_SOURCE=cache`,
`V3_CACHE_ROOT=/srv/aai3001/otf-cache/v1`, `V3_WORKERS=6`, `V3_PREFETCH=4`.
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
Prefetch queues now contain small uint8 groups rather than whole float32
batches. Joining holds at most one final batch plus groups; pinning can hold
one extra uint8 batch copy, far below the old batch/queue storage. CPU without
an accelerator does not attempt to pin unpinned groups. CUDA and ROCm use the
same data path; the existing ROCm-only conv1 model fix remains separate.

Cache source, six workers, prefetch4, ordered, batch32, 200 batches per case:

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

Final verification: 82 tests passed (48 shared pipeline/loader/adapter tests,
34 engine/batch/prepare tests). Final suite emitted no termination warning.
The group-mixing unit test now drains its short iterator before shutdown.
PC/VM SHA-256 values match for every tested code/notebook file. Documentation
checks passed with 22 files and zero broken links/anchors. All step8 scratch
copies, smoke checkpoints/logs, figures and predictions were removed after
recording results; approved cache and original run folders were retained.
See [loader step 7](torch-loader-verification.md) for prior cache/session parity,
legacy profile and full split counts. Lightning arbitrary-iterable support was
checked with Context7 `/lightning-ai/pytorch-lightning/2.6.6`.
