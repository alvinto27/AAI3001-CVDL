# Training tiling: three-arm design

**Design only.** The user lifted the research hold only for this experiment.
No implementation, image generation or training is authorised by this document.
The counts below are a read-only calculation from tracked photo dimensions;
no originals were decoded. Other research remains on hold.

## Fixed recipe and evaluation

Copy [newdata-1](../../notebooks/newdata-1.ipynb): wide XuNet, width 4,
clip 3, no norm, LR 0.0003, warmup 1, drops 5×0.3/20×0.1, gradient clip 1,
batch 32, FP32, 30 epochs, framework seed 0 and data run_seed 0.
**Training remains k 1–4 only**, using `configs/train-k1to4.yaml` unchanged.
Keep its bit modes, methods, placements, payload and bpp draws unchanged.
RAISE/LSDIR/ALASKA weights remain **40/20/40**, 45,000 tile pairs/epoch,
one variant/tile. DIV2K is test-only in this recipe, never a training candidate.
Keep workers, compile, priming, augmentation and checkpoint selection identical.
All arms start fresh; the historical mixed model is context, not a fourth arm.

Do not change any evaluation records, grids, settings, ordering or seeds:

- Capped validation: 8,035 pairs; `validation_pairs.csv` SHA-256
  `6bb3c8efadad06af50ba014a95c98b4fc57008119bdd55b794ec2a1729e4d5ca`.
- Standard test: 10,956 pairs; settings SHA-256
  `fdd989f7bdab0ec6f9fd6f4c57bdde981d16b54aa8c37e5da8fd3dfcc8b06837`.
  These are different kinds of hash, not interchangeable pixel hashes.
- Existing bpp-1 and all ALASKA/LSDIR panels stay separate and unchanged,
  including their existing k strata. Training k 1–4 does not filter test k.

The [mixed-run report](../results/result-newdata-1.md) is the source of these
reference identities. Save and compare the existing evaluation fingerprints
before and after implementation, not just pair counts.

## 1. Exact geometry

Use stored decoded height H and width W, not EXIF orientation guesses.
Apply each rule independently to each side L; take the Cartesian product of
row/column starts, in row-major order. T=512. All current training photos have
both sides ≥512 (checked in the metadata calculation). Fail if this is false
for a future inventory; do not silently change the shared photo population.

| Arm | Number of tiles on a side | Starts on that side |
| --- | --- | --- |
| overlap-last | n=ceil(L/T) | `min(i*T, L-T)`, i=0…n−1 |
| centre trim | n=floor(L/T); r=L−n*T | `floor(r/2)+i*T`, i=0…n−1 |
| centre pad | n=ceil(L/T); p=n*T−L | `−floor(p/2)+i*T`, i=0…n−1 |

Trim discards floor(r/2) pixels at the leading edge and ceil(r/2) at the
trailing edge. Pad adds floor(p/2) at the leading edge and ceil(p/2) at the
trailing edge. Thus odd extras go to bottom/right. An exact multiple has no
trim, pad or overlap; do not append a duplicate last tile. Overlap-last moves
only the last start, with overlap n*T−L, however large. Example L=1025:
overlap `[0,512,513]`, trim `[0,512]`, pad `[-255,257,769]`.
The formulas also work at 256; this experiment is **512 only**. Existing
`trim512`/legacy readers do not thereby gain 256 support.

## 2. New grids, old records

Today's default is **mixed**: RAISE uses recorded overlap-last, whereas ALASKA
and LSDIR use recorded **top-left** `trim512`. It is not any one of these arms.
Add one experiment setting, `train_grid`, allowed values `recorded` (default),
`overlap_last`, `centre_trim`, `centre_pad`. `recorded` must take today's exact
code path, identities, RNG consumption and outputs. Do not rename its grids.

For explicit arms, compute training axes from frozen photo sizes, uniformly
for all three sources; do not ask the frozen record's grid-policy field to
approve a different grid. Use a training-only resolver for pool sizes,
sampling, extraction and training pair offsets. Evaluation and legacy replay
must continue through their existing resolver. Do not mutate source CSVs,
splits, photo indices, recorded axes or their hashes.

Version the new geometry as `train-grid-v1`. At implementation, produce an
approved, tracked `data/train-grids/v1/summary.json` with input CSV SHA-256s,
rule version, T, source/train selection, pool counts and expected plan digests.
The canonical plan is one UTF-8 JSON line per training photo, sorted by frozen
photo_index: source, photo_id/index, H/W, rule/version/T and signed y/x starts;
use sorted keys, compact separators and LF endings. Hash the entire byte stream
with SHA-256. Recompute without decoding at preflight/runtime; reject a digest
mismatch, and save the plan plus digest in the run folder. Input sizes come
from the tracked CSV, not a new runtime scan. Assert decoded sizes agree.
Track only the compact summary, not new photo records or image tiles.

Keep OTF seed version and old grid codes intact. Overlap retains code 2;
reserve new codes 4/5 for centre trim/pad (3 remains frozen top-left trim512).
Record geometry version separately. Keep purpose, epoch, photo_index, tile
row/column, variant and settings/embedding streams keyed as today. Same seed
does not mean identical payloads across changed grids. Never change eval keys.

## 3. Crop → embed → pad → augment

For a signed tile start (y,x), intersect its T×T box with the real photo:
`[max(0,y), min(H,y+T)) × [max(0,x), min(W,x+T))`.
Copy that RGB8 real crop. Draw k, placement, stride, start and payload using
the existing settings/engine code on **this crop**, not on a padded canvas.
Placement indices and any regions use crop-local real coordinates; translate
coordinates only when writing provenance. This recipe has no new region mask.
Capacity uses real RGB carrier units; requested bpp uses real crop area A.
Retain existing byte rounding, one-byte minimum and lower-to-capacity rules;
record requested/effective bytes and `effective_bpp=8*payload_bytes/A`.
Never include padded area in capacity, region selection or bpp denominators.

Keep an untouched crop as clean; embed into its owned stego copy. Then place
both in identically positioned T×T RGB uint8 zero canvases (all channels 0),
using the real intersection offsets. No payload or embedding RNG visits pad.
Record real rectangle, pad extents, requested/effective bpp and seed keys.
Use the same pipeline for all arms; arms 1/2 have zero pad extents.

Apply **existing label-keyed flips/rot90 after padding**, unchanged in all
arms. Pair masks are identical **before** augmentation; do not add joint
augmentation. Clean and stego use different keys but the same flip/rotation
distribution, so padding position does not predict the label by construction.
Also test empirical mask/label balance; this is not a claim that finite
training samples or black borders cannot affect model behaviour.

## 4. Pool audit (512; one pair per candidate tile)

Read-only metadata audit, 2026-10-16. Train photos: RAISE 701, ALASKA 13,500,
LSDIR 7,000. With `C_s` the train photos of source s, pool sizes are exactly
`sum(ceil(H/T)*ceil(W/T))` for overlap/pad and
`sum(floor(H/T)*floor(W/T))` for trim, over `C_s`.

| Source | overlap-last | centre trim | centre pad | Tiles with pad (arm 3) |
| --- | ---: | ---: | ---: | ---: |
| RAISE | 45,674 | 34,890 | 45,674 | 20,166 (44.152%) |
| ALASKA | 41,391 | 21,139 | 41,391 | 33,792 (81.641%) |
| LSDIR | 41,520 | 14,475 | 41,520 | 40,138 (96.671%) |
| Total | 128,585 | 70,504 | 128,585 | 94,096 |

Padding fraction per tile is `f=1−real_crop_area/T²`. Distribution is over
**candidate tiles**, not photos, sampled epochs or clean/stego duplicates:

| Source | f=0 | 0<f≤10% | 10<f≤25% | 25<f≤50% | 50<f≤75% | f>75% |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| RAISE | 25,508 | 96 | 7,730 | 12,340 | 0 | 0 |
| ALASKA | 7,599 | 0 | 0 | 20,368 | 13,424 | 0 |
| LSDIR | 1,382 | 6,348 | 8,028 | 18,774 | 6,988 | 0 |

| Source | Mean f (all tiles) | Padded-only f: median / p95 / max |
| --- | ---: | ---: |
| RAISE | 11.787% | 29.688 / 41.498 / 43.226% |
| ALASKA | 34.646% | 37.500 / 60.938 / 60.938% |
| LSDIR | 30.846% | 30.469 / 61.914 / 74.213% |

For overlap, define duplicate-occurrence share as
`sum(n_y*n_x*T²−H*W) / sum(n_y*n_x*T²)`: repeated presentations beyond the
first, divided by all pixel occurrences in the **full pool**. It equals the
mean pad fraction above, but measures a different phenomenon. The share of
unique photo pixels covered more than once is respectively **12.643%,
41.799%, 35.160%** (RAISE, ALASKA, LSDIR). With overlaps `o_y=n_y*T−H`,
`o_x=n_x*T−W`, that repeated area is `o_y*W+o_x*H−o_y*o_x` per photo.
Last-axis overlap distribution (two axes per photo, units: pixels):

| Source | 0 | 1–128 | 129–256 | 257–384 | 385–511 | median / p95 / max |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| RAISE | 0 | 12 | 695 | 695 | 0 | 204 / 304 / 304 |
| ALASKA | 13,499 | 0 | 0 | 13,501 | 0 | 304 / 384 / 384 |
| LSDIR | 24 | 4,601 | 3,937 | 3,303 | 2,135 | 208 / 468 / 508 |

Trim drops `H*W−floor(H/T)*floor(W/T)*T²` real pixels per photo:
RAISE **1,415,649,716 (13.403%)**; ALASKA **1,549,694,720 (21.854%)**;
LSDIR **3,732,308,160 (49.587%)**. Percentages divide total dropped pixels by
total source training pixels, not by padded canvas size. Trim has no overlap;
pad has no duplicated real pixels. Neither recovers dropped trim information.

The sampler stays unchanged: each epoch selects **18,000 RAISE, 9,000 LSDIR,
18,000 ALASKA** pairs (90,000 labelled images). It uses seeded permutations
without replacement within each source pool, cycling only if its quota exceeds
that pool. None of these nine pools needs cycling. Epoch permutations change
as today. Smaller trim pools mean greater reuse across epochs, not extra
within-epoch repetitions. For pad, expected sampled padded-tile share is
**69.651%**, mean canvas pad fraction **24.743%**, weighted 40/20/40.
Those weights count pairs, not real pixels; padding changes real-pixel/payload
exposure. Do not rebalance weights or increase 45,000 to compensate.
Full-pool overlap percentages are not actual sampled-epoch duplicate shares:
only part of each pool is drawn; report sampled coverage separately if needed.

Audit input SHA-256s (`data/<source>/photo_manifest.csv`, entire file):

- RAISE: `fedab1b67cda8e90013ad8fde0cc9b03f62afca3df36d6af1b7a1b069645e7b8`
- ALASKA: `daa16066debe1f918ef70a889889d5c69094725142d76b9da57795477b23a1bf`
- LSDIR: `0f0969b713916fe2ae61bac76796ee40d07ec7a825d0cfb951a29e8d52afe53f`

Quantiles use linear interpolation; calculations use source_height/width and
split=train only. They require no photo decode, embedding or new upload.

## 5. Notebook identity and execution

Create copies of newdata-1, changing **only their first code cell**:

| Notebook (under `notebooks/experiments/`) | train_grid | Automatic run_name |
| --- | --- | --- |
| `tiling-overlap.ipynb` | `overlap_last` | `tiling-overlap` |
| `tiling-trim.ipynb` | `centre_trim` | `tiling-trim` |
| `tiling-pad.ipynb` | `centre_pad` | `tiling-pad` |

Remove the copied explicit long run_name. Keep other literal recipe keywords
identical, add only train_grid. In this **new namespace only**, shared settings
must derive run_name from the verified current notebook path's stem, and reject
an explicit conflicting run_name. Preflight/CLI already know the selected path;
supply it through internal execution context, not another experiment setting.
For interactive use, a shared notebook-identity helper must resolve the actual
kernel's notebook path (frontend-provided path or notebook-server session),
and fail clearly if absent/ambiguous. Do not guess from cwd or hard-code the
stem in a cell: that would drift on rename. In these cells, explicit non-default
train_grid must require notebook identity even when context is missing; never
fall back to the old generated name. Interactive integration is a gate
before creating usable notebooks, not an assumed existing API.

Leave all existing notebooks, RUN_NAME rules and golden names unchanged.
For new notebooks, one launch allocates `<stem>-<UTC YYYYMMDDTHHMMSSZ>-<4 digits>`
once. G4 uses its submitted CVDL_JOB_ID; PC/interactive launches allocate the
same form once in shared execution context. Use this ID for the run folder,
logs and Pi display, but keep run_name equal to stem. Persist both in
run_settings.json; never allocate an ID during preflight or each load() call.
Checkpoint paths for this namespace must use the instance ID, not just the
stem, to avoid resume/collision with an earlier launch. Check collisions before
training. Keep model/grid/LR/k details in the top cell and run_settings.json,
not names. No timestamp or experiment setting should be typed by the user.

## 6. G4 schedule and comparison

No new original-image uploads: reuse the same decoded-source inputs already
on G4; frozen `data/` records and the later compact grid summary travel by git.
After all PC gates pass, run fresh **overlap → trim → pad**, sequentially, at
one reviewed/pushed commit. Use the [notebook submission workflow](g4-batch-runner.md#submit-a-run)
with `--timeout-min 105`, existing 120-minute safety stop, and a dry run first.
Expected budget is about **$1.30/arm, $3.90 total**, an estimate, not measured
new timing or a spending cap. The timeout is a safety bound, not expected duration. Stop on wrong validation hash, missing sources or timeout; do not
silently shorten epochs, change settings or resume a different arm.

Select each best checkpoint by the identical capped-validation metric/rule.
Report checkpoint SHA/epoch, seeds, grid/input/validation/test digests, pool and
sampling statistics, elapsed time/cost, train/validation curves, standard-test
and separate extra-panel AUC/P_E, FA/miss at 0.5 and a validation-calibrated
threshold, by source/camera/k/bpp. Test-optimized P_E is diagnostic, not a way
to select a checkpoint or tune the deployed threshold. Compare all three fresh
arms on identical rows; one seed cannot establish statistical superiority.

Add a **separate padding-sensitivity diagnostic**, using frozen standard-test
parent identities. p=0 is the original frozen-pair reference row. For each
p=10/25/50%, use both training-like geometries (round half up with floor(x+0.5)):

- **Side strip:** real rectangle 512×round(512*(1−p)); zero-pad on one side.
  Choose among four sides with fixed diagnostic seed 1 and parent identity;
  rotate the rectangle for top/bottom padding.
- **Corner:** real square of side round(512*sqrt(1−p)), anchored to one of four
  corners; pad the two adjacent opposite sides. Choose the corner by the same
  label-independent keyed policy. No centred frame: that is not arm-3 geometry.

Keep orientation fixed across levels and models for a parent; include pattern,
not label, in the orientation key. Record actual fraction `1−real_area/512²`.
Crop the clean parent first, embed on that real crop, then pad. Reuse the
parent's requested bpp and bit/k/method/placement/payload-kind settings, not its
old full-tile byte count or absolute embedding start. Keep real-crop byte
rounding/capacity rules and record effective bpp. Use separate versioned
diagnostic embedding keys derived from parent identity, pattern and level;
save pairs/settings and digest once for all three models. Do not crop an
existing stego: that could remove contiguous payload. Do not resize, augment,
or mutate canonical grids/keys. Never re-embed the p=0 reference.

For each model, pattern and level, score **both** the exact same real clean/
stego crop directly (smaller input, no pad), and its zero-padded 512 version.
The fully convolutional/global-pooling model permits variable input sizes.
Batch controls only with identical shapes; never add implicit collation pad.
Keep weights, eval mode, dtype and preprocessing identical. Compare per-image
scores `padded−unpadded` (both labels), AUC, fixed-threshold FA/miss and k1/source
breakdowns; keep each model's original validation threshold. This paired control
isolates the effect of adding zero area with identical content and payload.
Smaller crops change the global-pooling area: disclose that explicitly; the
control measures the padding effect including its pooling-area change, not
black colour independently of geometry. Differences from p=0 also involve
content loss and must not be called pure padding effects.

Verify variable-shape inference on PC first; account for shape warm-up and
extra inference separately from training cost. Prefer post-training diagnostics
on PC to avoid extra paid G4 runtime; score all arms/controls on one backend,
record it, and recheck p=0 against canonical predictions within the established
numerical tolerance. Preserve the canonical p=0 predictions as the reference.
Never replace canonical panels, tune thresholds or select checkpoints with
this diagnostic.

## 7. Gates, open work and estimate

Before any G4 boot:

1. Default `train_grid=recorded`: golden compare **ALL EQUAL**, full suite;
   unchanged eval fingerprints and existing RUN_NAMEs, including source panels.
   Do not regenerate the golden or weaken its checks to accept default changes.
2. Unit tests: exact multiples, 512/513/1023/1024/1025, odd margins, rectangular
   photos, optional 256 formula checks, no duplicate starts, complete overlap/
   pad coverage, trim dropped margins, all source tags, invalid dimensions.
3. Verify all nine pool counts above and plan digests. Test real-only capacity,
   bpp/placement/regions, both bit modes and k1–4, zero pad after embedding,
   source immutability, identical pre-augmentation pair masks and replay across
   workers/epochs. Preserve current augmentation draws/distribution.
4. Test namespace naming in CLI/preflight/interactive paths, rename/conflict/
   missing identity, collision and checkpoint directories; old names unchanged.
5. PC QUICK once per arm, with recorded overrides/caps, inspect pad masks and
   pairs, identical initial-model hashes, nonzero real-pixel embedding, finite
   loss/gradients and checkpoint/resume. Check strip/corner geometry, keyed
   label-independent orientations, no-payload pad, p=0 identity and variable-
   size unpadded controls without hidden batch padding. Remove QUICK artefacts;
   inspect three G4 dry runs before submission.

Pending implementation: training-only resolver/real-crop interface, versioned
summary/digest generation, new setting/provenance and automatic notebook/run
identity (especially interactive path discovery), three notebooks, tests and
report tool. Confirm live G4 inputs match tracked inventories before boot.
No statistical winner rule or further seeds are approved; report trade-offs,
then ask before extending the experiment. Estimate **2–3 working days** for
implementation, audit/review and PC checks, then three sequential G4 jobs
(each bounded at 105 minutes) plus report preparation. This is a planning
estimate; no training or QUICK result has been produced in this design task.
