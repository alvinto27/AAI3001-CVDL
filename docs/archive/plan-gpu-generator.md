> Archived 2026-10-16: shelved design; no GPU generator was built. See [status](../status.md).

# Plan: optional G4 GPU LSB generator

**Status:** Design only. No GPU generator exists. Do not change the current
training, validation, or test path until the checks below pass. This option is
for the G4. The PC and old runs must keep the current CPU generator.

## Present data path and measured cost

`lsb_source.py` reads one full RAISE NEF or DIV2K PNG, checks RGB8 dimensions,
and returns a full photo. `lsb_torch.TrainingImages._items` shards photos by
worker, takes 512-pixel tile views, and calls `lsb_mode1.make_sample` twice:
one clean copy and one stego copy. A keyed `lsb_settings.draw_otf_settings`
selects bit mode, k, method, placement, stride, payload type, and size. A
separate keyed NumPy stream drives `lsb_engine.embed_fast`: payload bytes,
selected carrier units, field padding, and match tie choices. `_output` makes
uint8 CHW tensors and applies keyed flips and quarter-turns. A bounded buffer
shuffles images. `MixedLoader` joins four-image worker tasks into batches.
Notebook 05 moves uint8 batches to the GPU and converts them to float32 in
the model step. Validation and test use `FixedPairs` and `OrderedPairLoader`;
they must stay as they are.

A small **PC CPU-only** probe timed 256 training image outputs (128 pairs,
with one extra stego sample read ahead by the iterator). It read six full
photos through the decode source. The measured wall time was 1,772 ms.
The six photo reads took 1,470 ms. The probe ran in the ROCm container, but
used no GPU. It is a short mixed-source sample, not a full-epoch G4 profile.
OS cache state and photo format can change the decode time. Timers inside
`embed_fast` are nested; do **not** add their time to the total for embed.

| CPU work | Calls | Total (ms) | What the time includes |
| --- | ---: | ---: | --- |
| Read/decode a full photo | 6 | 1,470 | Source read, decode, RGB8 and shape checks |
| Get tile view | 258 | 3.0 | Grid check and view; no full-tile copy |
| Draw stego settings | 129 | 4.0 | Keyed CPU NumPy draw and capacity limit |
| Full stego embed | 129 | 149.6 | All payload, positions, field and pixel work below |
| └ payload and bit unpack | 129 | 3.1 | Two payload types |
| └ choose/sort positions | 129 | 77.0 | Random subset or contiguous/stride sequence |
| └ pack target fields | 129 | 13.6 | Includes random trailing bits |
| └ replace/match pixel values | 129 | 43.9 | Both methods combined |
| Convert uint8 CHW, no augment | 256 | 41.3 | Separate timing on the same image outputs |
| Convert and augment | 256 | 82.6 | Actual output path; augment cost estimate: 41.3 ms above conversion |

The G4 R1 `nvidia.csv` has 31 one-minute GPU samples: 12 at 0% and an
average of 37.4%. If the first and last samples are removed, its 29 interior
samples have 10 at 0% and an average of 40.0%. An earlier short summary said
"12 of 29 at 0%"; that mixes the two counts. A 0% sample may be validation,
first compile, wait for data, or post-fit work. It does **not** by itself prove
that embedding is the limit. R1 fit took 1,627 s for 20 epochs, including
validation and checkpoint work.

## Proposed G4-only switch and work split

Use `V3_GPU_LSB=off|on` with default `off`. `off` must call the present loader
without changes. For `on`, build a second train-only loader and record the
mode, seed-rule version, settings YAML digest, and GPU generator version in
the run settings and checkpoint. Do not apply it to `FixedPairs`, validation,
test, the extra bpp-1 test, or diagnostic evaluation.

| Stage | Place | Data passed to the next stage |
| --- | --- | --- |
| Photo schedule and keyed settings | CPU parent | Pair ID, epoch, photo/tile key, label, bit mode, k, method, placement, stride, payload kind, requested/effective bytes, clipping flag, and embedding key |
| Read and decode photo; get cover tile | CPU workers | Owned uint8 CHW cover tile and the compact settings record; emit one cover for each clean or stego training item |
| Batch transfer | Pinned host memory to GPU | Cover uint8 and integer metadata tensors; keep image order and pair IDs |
| Payload, positions, fields, match choices, embed | GPU batch operation | Original cover, stego uint8 and optional changed mask |
| Flip/rotate and float conversion | GPU after embed | Float32 image and unchanged label/index contract to the model |

Start with keyed CPU settings draws in the parent, not in workers. This costs
about 4 ms for 129 stego items in the probe. It keeps capacity and setting
records exact without a new RNG implementation. The GPU should take over the
large position and embed work. An optional later GPU settings draw needs its
own proof and measurements. A simple `LightningModule.on_after_batch_transfer`
hook can run the train transform after device transfer; make it a no-op for
validation and test. Check that it runs once per batch and before
`training_step`. Test a separate prefetch stream only after the simple path
works. Keep compile limited to the model first; test GPU generation under
`torch.compile` separately.

Refuse `on` unless the device is the intended G4 CUDA GPU, CUDA is available,
and at least 80 GiB of VRAM is reported. Refuse ROCm, CPU, PNG training,
paired batches, and QUICK by default. Permit QUICK only with a separate test
flag (for example `V3_GPU_LSB_TEST_QUICK=1`), never by silent fallback. Check
512-pixel RGB8 cover tiles, the supported training settings, and the selected
source types before making workers. If any check fails, stop with a clear
error; do not quietly run CPU embedding under an `on` label.

## Exactness and change masks

The CPU `make_sample` uses distinct, keyed NumPy settings and embedding RNG
streams. NumPy random placement selects a subset without replacement, sorts
it, and consumes RNG calls in a specific order. The GPU must not claim the
same random bytes or positions merely because it uses the same integer seed.
GPU random generation can be statistically equivalent but will not normally
be **bit-identical** to NumPy. A seed derived from a stable sample key must
select every GPU random stream; it must not depend on worker count or batch
order. Use separate keyed streams/counters for payload, random positions,
field padding, match ties, and augmentation. In the first implementation,
the CPU parent can also draw the current augmentation choices from the keyed
CPU `torch.Generator` and pass the flip/turn flags. Applying those flags on
the GPU can be byte-identical even though GPU RNG draws are not.

| Proof | Required result |
| --- | --- |
| Fixed cover, settings, payload bytes, sorted positions, trailing bits, and match tie flags | GPU uint8 stego equals `embed_fast` byte for byte. Cover every `lowest`/`plane`, k=1–8, replace/match, random/contiguous/stride, random_bytes/letters, clipped capacity, k=8, and 0/255 match edges. Check nonwrapping starts and MSB-first bit order. |
| Draw distribution | Test many keyed samples by mode, k, placement, stride, payload and bpp. Compare positions and change counts with CPU reference distributions. Keep the exact current settings table where CPU pre-draws it. |
| Resume and training | Same GPU-mode run resumed at an epoch boundary yields the same keyed batches and schedule. Compare validation P_E/AUC from several seeds against CPU-mode runs; decide a noise tolerance **before** viewing the results. Do not claim one-run metric equality. |

For a map mask, compare the original uint8 cover with the final embedded uint8
image **before** augment: `changed_pixel = (cover != stego).any(channel)`.
Apply the same spatial flip/turn to the cover, stego and mask if the training
path needs a mask. Use the actual changed pixels, not the selected positions:
replace and match can leave a selected carrier value unchanged. Do not keep
all masks for training if no consumer needs them. For a full-photo map, keep
the existing CPU source and exact-change path unless it is separately proved.

## Epochs, batches, and memory

| Concern | Rule and check |
| --- | --- |
| Unpaired shuffle | Preserve the present two training items per pair, stable item IDs, photo sharding, worker tails, and bounded shuffled output. A clean item gets no stego embed. The cover can be copied into two independent items; do not mutate a shared view. Test all IDs exactly once at worker counts 0, 4, and 32. |
| Augment | Use the current keyed flags per item and label. Apply after embedding on the GPU. Check that clean/stego operations do not share an unwanted RNG stream. |
| Prime-next-epoch | The current `PrimedTrainingLoader` can prefetch worker covers for epoch E+1 during validation. It must **not** run GPU embedding until those covers are consumed by the train hook. Check epoch keys, no GPU validation overlap, shutdown, and same item order with priming on/off. |
| Resume | At `last.ckpt` epoch boundary, restore epoch, schedule, mode and seed-rule version. Recompute random values from sample keys. A mid-epoch resume needs a saved loader cursor and shuffle-buffer state; refuse an unsupported mid-epoch resume. Old CPU checkpoints must not resume into GPU mode without an explicit, tested migration. |
| Workers and compile | Decode stays in CPU workers. Compare 4/8/16/32 workers and cold/warm source reads. Keep model compile scope unchanged. Check key order and data bytes with eager/compiled model and persistent workers. |
| VRAM | A batch of 32 RGB 512×512 uint8 covers is about 24 MiB; one stego copy is about 24 MiB more. Int64 position arrays can use about 192 MiB at one full carrier per item, before sort and temporary tensors. Set a measured 2–4 GiB generator scratch target and a hard test limit. Record peak reserved/allocated VRAM during the first full-data run. Do not retain the full source corpus on the 96 GiB G4 GPU. |
| More sources | ALASKA and LSDIR increase photo count and decoded bytes. Leave decode/validation and source-specific controls on the CPU. Batch transfer only current cover tiles; add each source after its frozen records and tile policy have tests. |

## Expected gain, risks, and staged work

**Estimate, not a result:** removing about 150 ms of embed work per 129
stego items in the short PC probe does not remove its 1,470 ms of photo read
and decode. If G4 workers are mainly waiting on decode or validation, gain
could be **zero** or negative. If GPU generation removes long CPU stalls,
a busy G4 rate near 67–75 steps/s suggests a possible **0–25% fit-time** gain
against R1's 1,627 s / 20 epochs; end-to-end gain should be less because
validation, test, diagnostics, and transfer stay. Measure cold and warm
runs; do not claim this range as observed speedup.

| Order | Work | Exit check |
| ---: | --- | --- |
| 1 | Freeze CPU fixtures for seeds, settings, payload, positions, pixels, masks, item IDs, and all boundary cases. Add mode and provenance fields with default `off`. | Old byte hashes, standard test hash `fdd989f7bdab0ec6f9fd6f4c57bdde981d16b54aa8c37e5da8fd3dfcc8b06837`, and existing tests unchanged. |
| 2 | Build a CPU-only cover reader and compact settings table; preserve the shuffle and epoch contract. | Same cover bytes and item order at 0/4/32 workers, priming on/off, and last-checkpoint resume. |
| 3 | Build GPU embed from supplied payload/positions/tie flags. Add actual-change mask for tests. | Bit-exact fixture matrix; no out-of-range units or byte overflow. |
| 4 | Add keyed GPU draws, payload/position generation, and augment. | Distribution, key independence, and repeated-run/resume checks; no global RNG consumption. |
| 5 | Add train-only transfer hook behind the switch. Keep eval CPU. | QUICK forced-test run; standard test settings hash and predictions unchanged; `off` path identical. |
| 6 | Profile G4 cold/warm runs, GPU time, CPU workers, IO, and peak VRAM. | Compare with same-commit CPU mode; keep GPU mode only if reliable and useful. |

The main risks are mismatched NumPy/GPU RNG semantics, random-position memory
and sort cost, GPU contention with model training, stale prefetched epoch
keys, and later source growth. Stop at any failed exactness, reproducibility,
or memory check. Do not make the switch the default based only on one run.
