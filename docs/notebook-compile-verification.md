# Notebook 04 optional network compilation

Status: optional switch verified on the PC (ROCm RX 9070 XT) and team VM
(CUDA L4), 2026-10-14. Full training remains a user action. Results here
use v3c, `CLIP_T=3`, GroupNorm, float32 `32-true`, no autocast.

## Switch and safety

`V3_COMPILE=off` is the default. `V3_COMPILE=default` calls
`lightning_model.net.compile(mode="default", dynamic=False, fullgraph=False)`
after module construction (including the ROCm conv1 replacement) and before
`trainer.fit`. Empty/false/zero also select off; all other modes raise an
error. In-place compilation keeps state-dict keys, parameter identity and the
optimizer intact; no `_orig_mod.` prefix is added. The module constructor
remains eager, so checkpoint reload, test, and diagnostic copies remain eager
and runs can resume across off/default in either direction. The unsupported
`reduce-overhead` and autotune modes are deliberately not exposed: the
[ROCm trial](../rocm-perf/notes/compile_v3.txt) found nondeterminism or an
accuracy failure, respectively.

Before compile, the notebook refuses `MIOPEN_GEMM_ENFORCE_BACKEND` (the forced
backend previously gave wrong gradients). When explicitly set,
`TORCHINDUCTOR_CACHE_DIR` and `TRITON_CACHE_DIR` must be absolute paths outside
the repository; their normal unset defaults are outside it too. Keep these
cache paths external before starting the kernel. `model_run_settings.json`
next to the checkpoint records mode, compile target/options, live TF32 flags
and torch version; the OTF data record also stores `compile_mode`.

After fit, the notebook shuts down persistent training DataLoader workers
before test and diagnostics. On the PC, an eight-worker cache-loader probe
(with the normal prefetch of 128 images per worker) measured worker summed
proportional set size (PSS) of 10,552,436,736 bytes while active and
8,690,688 bytes after shutdown (the remaining process was the Python resource
tracker): about **10.54 GB released**. Available system RAM rose from
13,880,606,720 to 23,890,235,392 bytes (about 10.01 GB); swap-free stayed
7,350,796,288 bytes. This was an isolated loader probe, not a full fit.
Recreate the loader if rerunning the fit cell in the same kernel.

## Accuracy and checkpoints

The fixed-batch compiled-vs-eager test used the same real clean/stego pixels,
initial weights and labels. It checked logits, loss, every trainable gradient,
state-dict keys and parameter identities. The strict 1e-5 comparison sets
**both TF32 flags false inside that test only**; it restores native flags
afterwards. Neither notebook nor normal speed/training runs change TF32.
Here global gradient relative error means the maximum absolute difference
across all gradients divided by the maximum absolute eager gradient across
all parameters.

| Device / flags for check | Logits max relative | Loss max relative | Global gradient relative | Pass 1e-5 |
| --- | ---: | ---: | ---: | --- |
| L4, both TF32 flags false | 2.56e-7 | 1.51e-7 | 2.42e-7 | yes |
| L4, native (matmul false, cuDNN true) | 7.00e-5 | 9.06e-7 | 4.87e-4 | no |
| PC ROCm, native | 2.25e-7 | 0 | 2.40e-7 | yes |
| PC ROCm, both false / both true | 2.25e-7 | 0 | 2.40e-7 | yes |

On the L4, forcing both TF32 flags true also failed (logits 2.46e-4,
global gradient 4.76e-4). The largest per-parameter relative gradient
with native L4 flags was 0.01634 (`conv1.bias`); per-parameter ratios use a
smaller denominator than the global ratio. A float64 CPU reference showed
L4 logits relative error 5.24e-4 with TF32 on against 2.51e-7 with both
off. The KV and first convolution outputs changed by at most about 4.33e-7.
On ROCm, changing these flags did not change the measured relative errors:
there is no cuDNN TF32 path on that device.

Saving at step 2 and resuming to step 4 succeeded both off-to-default and
default-to-off, including optimizer state. Both checkpoint key lists matched;
`load_from_checkpoint` and the diagnostic `deepcopy` were eager in both
cases. A 200-step fixed-batch L4 run at native flags ended with loss
0.0003592–0.0003650 (eager, two trials) and 0.0003663–0.0003722
(compiled, two trials). First-step loss was 0.783105 eager and 0.783102
compiled. Repeated trials agree at step 1 but not bit-for-bit at steps
50–200; do not claim run-to-run determinism. Fixed 128-image validation
accuracy was 0.796875, P_E 0.1953125 and AUC 0.83984–0.84033 across
all four trials.

## Graphs and batch sizes

On L4, the standard train batch compiled as one graph with zero breaks.
A separate probe accumulated five guarded graphs for train batch 32 (with
grad), train tail batch 28 (with grad), eval batch 32, eval batch 12, and
train batch 32 with grad disabled. The first call took 29.12, 24.62, 6.26,
13.08 and 0.26 seconds respectively; the 28-image tail and eval mode can
compile separately. The actual OTF training schedule has 3,371 batches of
32 plus one final batch of 28 per 107,900-image epoch, for both 6 and 8
workers and epochs 0 and 1. No last batch is dropped or padded.

## Speed

A fixed real batch on GPU (GPU-only) and live OTF loaders used 300 training
steps, with 30 warmup steps and a synchronized 270-step/8,640-image timed
window. All normal benchmarks kept native TF32 flags. These are single-run
observations, not claimed full-epoch throughput. Rates are image completions
per second; GPU-only repeats one batch, whereas loader cases change images.

| Device / loader | Eager images/s | Compiled images/s | Compiled first step (s) |
| --- | ---: | ---: | ---: |
| L4, GPU-only | 287.50 | 348.80 | 35.52 |
| L4, cache, 6 workers | 279.61 | 332.40 | 30.30 |
| PC ROCm, GPU-only with decode8 setup | 369.22 | 407.81 | 14.93 |
| PC ROCm, decode, 8 workers | 380.28 | 474.94 | 5.59 (compile cache warm) |
| PC ROCm, GPU-only with cache6 setup | 369.25 | 457.83 | 5.07 (compile cache warm) |
| PC ROCm, cache, 6 workers | 379.45 | 462.16 | 5.44 (compile cache warm) |

The PC first cold compiled step took 14.93 s versus eager 3.23 s (about
11.70 s added); later cases used the same external Inductor/Triton caches
and **are not cold-compile measurements**. L4 first compiled steps were
35.52 s for GPU-only and 30.30 s for the live cache loader; the latter ran
after the first compiled case and is not an independent cold-cache measure. ROCm compiled
peak GPU allocation was roughly 4.02–4.09 GB versus 2.98–3.05 GB eager;
L4 compiled peak allocation was roughly 1.99–2.01 GB versus 2.83–2.90 GB
eager. See the ROCm trial for earlier isolated measurements; do not equate
fixed-batch rates to loader cost because the input sequences differ.

## TF32 decision

Actual flags in both notebook-setup processes were matmul `False`, cuDNN
`True`. ROCm has no cuDNN TF32 implementation. On L4, three eager runs each
showed median images/s of 285.93 GPU-only / 277.91 cache6 with native flags;
270.98 / 265.31 with both TF32 flags false; and 280.40 / 270.23 with both
true. Native cuDNN TF32 therefore saved roughly 4–6% versus fully off.
On the PC, forcing both flags off or on produced the same compiled-vs-eager
error figures above; eager GPU-only throughput was 368.78 versus 370.12
images/s in one run each (a 0.36% gap, not evidence of a speed effect).

The user chose to **keep native flags** and add no `V3_TF32` switch. The PC
(no TF32) and VM (cuDNN TF32) full test runs on the same 21,912 images had
P_E 0.0173 versus 0.0170, AUC 0.9974 versus 0.9975, and RAISE-only P_E
0.0115 versus 0.0112. The strict test-only TF32-off override does not change
trained models or the notebook's default.

## QUICK smoke

All 15 code cells passed with `V3_COMPILE=default`, OTF cache, native TF32,
and scratch-only output roots: PC ROCm completed in 63.42 s, VM L4 in
182.53 s (includes first compile, training, test, GroupNorm diagnostics and
notebook 02 breakdown). The PC used cache6; the VM used cache6. Test runs
and compiled caches were outside the repository. They do not constitute a
full epoch or full held-out evaluation. No saved user checkpoint was modified.
