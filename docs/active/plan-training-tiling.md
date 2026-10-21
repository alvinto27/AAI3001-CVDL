# Training tiling: three-arm design

**Approved design and PC implementation.** The user lifted the research hold
only for this experiment. This task does not launch G4 jobs. Pool statistics
below came from tracked dimensions without decoding originals; later PC
training/evaluation checks are listed at the end. Other research stays on hold.

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
The training setting `train_grid` has values `recorded` (default),
`overlap_last`, `centre_trim`, `centre_pad`. `recorded` must take today's exact
code path, identities, RNG consumption and outputs. Do not rename its grids.

For explicit arms, compute training axes from frozen photo sizes, uniformly
for all three sources; do not ask the frozen record's grid-policy field to
approve a different grid. Use a training-only resolver for pool sizes,
sampling, extraction and training pair offsets. Evaluation and legacy replay
must continue through their existing resolver. Do not mutate source CSVs,
splits, photo indices, recorded axes or their hashes.

The geometry version is `train-grid-v1`. The tracked
[summary](../../data/train-grids/v1/summary.json) holds input CSV SHA-256s,
rule version, T, source/train selection, pool counts and expected plan digests.
The canonical plan is one UTF-8 JSON line per training photo, sorted by frozen
photo_index: source, photo_id/index, H/W, rule/version/T and signed y/x starts;
use sorted keys, compact separators and LF endings. Hash the entire byte stream
with SHA-256. Recompute without decoding at preflight/runtime; reject a digest
mismatch, and save the plan plus digest in the run folder. Input sizes come
from the tracked CSV, not a new runtime scan. Assert decoded sizes agree.
Track only the compact summary, not new photo records or image tiles.

Keep OTF seed version and old grid codes intact. Overlap retains code 2;
use new codes 4/5 for centre trim/pad (3 remains frozen top-left trim512).
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

The three copies of newdata-1 differ **only in their first code cell**.
The common post-fit cells are now guarded by post_fit_eval; full remains the
unchanged default for old notebooks.

| Notebook (under `notebooks/experiments/`) | train_grid | name = run_name |
| --- | --- | --- |
| `tiling-overlap.ipynb` | `overlap_last` | `tiling-overlap` |
| `tiling-trim.ipynb` | `centre_trim` | `tiling-trim` |
| `tiling-pad.ipynb` | `centre_pad` | `tiling-pad` |

Approved simplification: each top cell states literal `name="tiling-pad"`
(or its own stem), train_grid and `post_fit_eval="none"`, replacing the copied
long run_name. All other recipe values stay identical. Preflight, runner and
a test over experiments notebooks reject name/stem mismatch; conflicting
run_name or name overrides also fail. No interactive path-discovery machinery.
Old notebook names and golden RUN_NAMEs remain unchanged.

One headless launch allocates `<name>-<UTC YYYYMMDDTHHMMSSZ>-<4 digits>` once:
G4 supplies CVDL_JOB_ID; PC run_notebook.py allocates it. Interactive load uses
`<name>-interactive-<UTC>` instead. Use the ID for checkpoint/run folders,
logs and Pi display, while run_name equals name. Save both in run_settings.json.
Preflight allocates no ID. Hyperparameters remain in cells/settings, not names.

`post_fit_eval` is `full` (default) or `none`. The three arms use none: G4 fits
with per-epoch capped validation, writes best/last checkpoints and settings,
then stops without post-fit predictions, panels, breakdown or diagnostics.
Default full preserves old behaviour. After download, evaluate on the PC:

```sh
podman exec -w /work aai3001 python scripts/evaluate_run.py runs/<downloaded-run>
podman exec -w /work aai3001 python scripts/evaluate_run.py runs/<downloaded-run> --padding-check   # retired 2026-10-21, see legacy-512-retirement.md
```

The evaluator accepts the outer job folder or checkpoint folder, selects its
best checkpoint, reuses saved experiment/QUICK values and PC evaluation workers,
and runs the notebook's standard/bpp-1/source panels, breakdown and diagnostics.
It rebases saved config paths, checks validation identity, preserves training
settings/checkpoints and writes separate evaluation provenance/results. Old
pre-package runs are reconstructed explicitly from model/data settings. Do not
run multiple models concurrently. Padding controls are optional and separate.

## 6. G4 schedule and comparison

No new original-image uploads: reuse the same decoded-source inputs already
on G4; frozen `data/` records and the later compact grid summary travel by git.
After all PC gates pass, run fresh **overlap → trim → pad**, sequentially, at
one reviewed/pushed commit. Use the [notebook submission workflow](g4-batch-runner.md#submit-a-run)
with `--timeout-min 105`, existing 120-minute safety stop, and a dry run first.
Expected budget is about **$1.30/arm, $3.90 total**, an estimate, not measured
new timing or a spending cap. The timeout is a safety bound, not expected
duration. Stop on wrong validation hash, missing sources or timeout; do not
silently shorten epochs, change settings or resume a different arm.

Select each best checkpoint by the identical capped-validation metric/rule.
Report checkpoint SHA/epoch, seeds, grid/input/validation/test digests, pool and
sampling statistics, elapsed time/cost, train/validation curves, PC standard-test
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

Implementation and PC checks are complete; no G4 launch is authorised by
this task. Confirm live G4 inputs match tracked inventories before boot.
No statistical winner rule or further seeds are approved; report trade-offs,
then ask before extending the experiment. Estimate **2–3 working days** for
implementation, audit/review and PC checks, then three sequential G4 jobs
(each bounded at 105 minutes) plus report preparation. This is a planning
estimate.

## 8. PC gate evidence (2026-10-16)

- Default golden compare: **ALL EQUAL**. Full suite: **179 tests pass**.
  The 12 new tests cover rules/rounding, full pool/digest checks, real-only
  embedding and pad zeros, source-independent sampling, frozen evaluation
  frames, name checks, post-fit skipping and offline/padding controls.
- Full-size metadata for all three arms retains validation `6bb3c8ef…` and
  standard test `fdd989f7…`, plus identical five ALASKA/LSDIR settings hashes.
  Bpp-1 settings also retain `606bd680…` in the full backend comparison.
- One successful QUICK fit per arm through run_notebook.py: 2 epochs,
  2 train batches/epoch, batch 8, 30 scheduled pairs/epoch, QUICK eval cap 16.
  These are recorded overrides, not G4 recipe edits. Fit/runner time was about
  **43.4 s/arm**. All stopped with post_fit_eval=none, one best plus last
  checkpoint, no predictions until the separate evaluator was invoked.
- Initial weight SHA-256 was identical in all arms:
  `b5c85aac1e76b89903fb1f334c4b0bf5e45c6a0ab46eae906981477280110966`.
  Each QUICK folder then completed the PC evaluator, including standard,
  bpp-1, five source panels, breakdown/heatmaps and batch diagnostics.
  Pad QUICK additionally completed both geometries at 10/25/50%, padded and
  unpadded controls on 16 parent pairs, plus the frozen p=0 reference. These
  are smoke checks, not research results or an arm ranking.

Full backend check used a scratch copy of the existing mixed-run folder
`runs/xunet_v4_clip3_nonorm_lr0.0003_warm1_drops5x0.3-20x0.1_k1to4_e30_wide-w4_mixr2-20261016T024859Z-9032`.
The original was not written; both checkpoint files remained byte-identical.
PC profile used 4 eval workers, Torch `2.14.0+rocm10.1.0`, NumPy 2.5.3,
Pillow 12.3.0. Compare identical ordered cover IDs/labels to saved G4 CSVs:

| Panel | Pairs | Absolute AUC difference | Absolute P_E difference |
| --- | ---: | ---: | ---: |
| Standard | 10,956 | 5.04024e-7 | 0 |
| Bpp-1 | 10,956 | 2.62426e-7 | 4.56371e-5 |
| ALASKA val | 1,535 | 3.60747e-6 | 0 |
| ALASKA seen | 2,648 | 9.26995e-7 | 0 |
| ALASKA unseen | 3,153 | 8.55010e-7 | 0 |
| LSDIR val | 3,164 | 2.14766e-6 | 0 |
| LSDIR test | 5,245 | 2.61723e-6 | 9.53289e-5 |

Maximum differences: **AUC 3.607465331345594e-6**,
**P_E 9.532888465203782e-5**, both below the 1e-3 gate.
All six extra-panel settings hashes match the original G4 files.
GPU process checks before PC work found no GPU owner; notebook 02 was never
changed or staged by this task. G4 dry runs at the pushed final commit are
Pending orchestration; no VM boot occurred in these checks.
