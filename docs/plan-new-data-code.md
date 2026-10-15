# ALASKA and LSDIR source code plan and checks

## Status and limits

The readers, manifest builder, mixed tile sampler, and separate evaluation panels are implemented. **The final ALASKA v2 and LSDIR selections are pending.** Do not train a full mixed run or interpret the small smoke-run scores as a model result. Do not commit a manifest built from `/tmp/selection_v2_DRAFT.csv`: that draft has 18,200 rows, omits about 12,800 eligible photos and Pentax_K50, and its split counts can change. The full manifests must be generated from the committed, final selections and verified local masters.

The old `data/raise/` and `data/div2k/` records are not rewritten. Their overlap-last tile coordinates, default train stream, standard validation, and standard test remain unchanged. The standard-test settings SHA-256 is `fdd989f7bdab0ec6f9fd6f4c57bdde981d16b54aa8c37e5da8fd3dfcc8b06837` (10,956 pairs). New files and their `trim512` grid are separate and opt-in.

## Selection contracts and provenance

| Source | Required selection CSV fields | Splits | Master file |
| --- | --- | --- | --- |
| ALASKA | `alaska_id,camera_group,camera_model,split,block,width,height,tiles512,bytes,sha256,in_v1,url` | `train,val,test_seen,test_unseen` | `Datasets/alaska/tiff_various_color/{alaska_id}.tif` |
| LSDIR | `lsdir_id,shard,path,split,width,height,tiles512,bytes,sha256` | `train,val,test` | `Datasets/lsdir/hr/{path}` |

LSDIR `path` is a relative PNG member such as `0001000/0000551.png`; `shard` matches `shard-NN`; `lsdir_id` is a safe selection identifier. `tiles512=floor(width/512)*floor(height/512)`. IDs and member paths must be unique. The builder rejects repeated SHA-256 values, including the known byte-identical LSDIR members `0022000/0021599.png` and `0022000/0021600.png` if both appear. Worker_3 owns selection and duplicate-photo policy; this check is a last safety gate. ALASKA `camera_group` is the raw index name, and `camera_model` is normalized in the selection.

Run these commands **inside the ROCm container**, after selections and source masters are final; do not run Python on the host:

```sh
podman exec -w /work aai3001 python scripts/build_new_source_manifests.py --source alaska --selection data/alaska/selection_v2.csv --output-root /work
podman exec -w /work aai3001 python scripts/build_new_source_manifests.py --source lsdir --selection data/lsdir/selection_v1.csv --output-root /work
```

The command checks each file's size and SHA-256. It refuses to overwrite any output. Each source receives `photo_manifest.csv`, `split.csv`, `provenance.csv`, and `build_record.json` (selection hash, counts, decoder/grid policy, and verification state). Use `--output-root runs-out/<scratch>` for a temporary selection. `--no-verify-files` is only for a metadata dry run; **never use it to make final records**. Selection, manifests, and masters must be transferred to the G4 before any G4 mixed run. Do not start a G4 or copy an incomplete ALASKA draft by inference from this plan.

## Decode and grid rules

- ALASKA TIFF: one uncompressed RGB8 frame, three samples, unsigned values, contiguous planes, orientation 1, no extra samples. Reject alpha, palette, compression, other orientation, wrong shape, and wrong type. Return the stored pixels; do not apply ICC, rotate, resize, or pad.
- LSDIR PNG: one RGB8 image, no palette, alpha, or transparency; return stored pixels without color or orientation conversion.
- New records explicitly state `grid_policy=trim512`: start positions `0,512,1024,...` within the photo. Discard the incomplete right/bottom edges. Only 512 tiles are supported. The old RAISE/DIV2K records continue to state `overlap_last`; their seeds and records do not change. New `photo_index` ranges start at 100,000 (ALASKA) and 300,000 (LSDIR), and seed codes for `trim512`, `test_seen`, and `test_unseen` are appended without renumbering old codes.

## Opt-in mix and evaluation

For notebook 05, select source weights with `V3_SOURCE_MIX=raise:0.4,lsdir:0.2,alaska:0.4`. This selects the source names; do not include DIV2K. Use `V3_TILES_PER_EPOCH=45000` for about 18,000 RAISE, 9,000 LSDIR, and 18,000 ALASKA tiles per epoch. `V3_SOURCE=decode` reads masters. The sampler selects tile identities without replacement within a source when enough tiles exist and draws a keyed new sample for each epoch. It groups chosen tiles by photo to avoid repeated full-photo reads. If a source has too few distinct selected tiles, it repeats them and prints `repeated_tiles` for each source at epoch start. Use `V3_TRAIN_PHOTO_FRACTION=1` for the full candidate pool. Keep `V3_SOURCE_MIX` unset to keep the exact default training schedule and its original RAISE/DIV2K coverage. `V3_NEW_DATA_ROOT` may point to a scratch root with `data/alaska/` and `data/lsdir/` manifests; masters stay below `/work/Datasets/`.

`V3_CHECKPOINT_VAL=standard` (default) selects the best checkpoint on the existing RAISE/DIV2K validation pairs. `V3_CHECKPOINT_VAL=combined` adds ALASKA val and LSDIR val pairs to the checkpoint metric. This choice changes the model-selection target: do not compare resulting best-checkpoint P_E values as if they used identical validation sets. The standard test always filters to the original RAISE/DIV2K records. `V3_EXTRA_TESTS=source-panels` evaluates the best checkpoint independently on ALASKA val, ALASKA test_seen, ALASKA test_unseen, LSDIR val, and LSDIR test. `V3_EXTRA_TESTS=test-bpp1,source-panels` also preserves the existing bpp-1 pass. Separate `extra-<source>-<split>/` directories hold predictions, setting hashes, and metrics; the default remains `test-bpp1`. The standalone `scripts/eval_checkpoint.py` accepts `--source alaska|lsdir --split val|test_seen|test_unseen|test --manifest-root <root>` (valid source/split combinations only). Neither extra panel selects a checkpoint.

**Validation cost:** the original full validation has 11,238 pairs and the standard test 10,956 pairs. The *temporary smoke selection* added 11 validation pairs (combined 11,249) and separate panels of 3 ALASKA seen-test pairs, 8 ALASKA unseen-test pairs, and 7 LSDIR test pairs. These are not final selection sizes. For a final set, compute exact counts from `FixedPairs` after the final manifests exist. Estimated added validation time is approximately the old validation per-pair cost times `(ALASKA val pairs + LSDIR val pairs)/11238`, plus possible file-read and worker-start overhead; it cannot be measured accurately on the G4 until full masters are present. The earlier R2 G4 normal-epoch validation compute took about 13.2 s for 11,238 pairs: a simple linear estimate adds **~1.2 s per 1,000 new validation pairs**, *not* including decoding, startup, or any effect from different image formats and workers. This was before the later output-capture/persistent-worker fixes and is not a post-fix benchmark. With 1,000 ALASKA val photos at about 1–2 tiles/photo and an as-yet unknown LSDIR val selection, expect at least ~9–18% more pairs even before LSDIR. **Decision pending:** choose standard-only or combined validation before a full training run; the code does not choose it for you.

## Verification evidence and next gates

- Four new tests check strict RGB8 reads/rejections, frozen manifest fields and hash provenance, epoch-keyed ratios, worker parity, repeated-source accounting, isolated eval panels, and trim grids. All 21 earlier loader tests still pass. A direct old-versus-new default-stream comparison on an unchanged PNG record found identical 12-image stream digests for epochs 0 (`84f27b907dd6fbdde9a154feaee05eedc0a37f37723983e635f35a74baa12063`) and 1 (`7c4724a397ee88a322e577296be0f71ccf8b8c8d1b4b73f43a14bd032a6de236`).
- The real-master PC QUICK smoke used **30 read-only-copied LSDIR PNGs** (53 MiB), 33 ALASKA TIFFs from the incomplete draft, and 60 mixed tiles per epoch for two epochs: 24 RAISE, 12 LSDIR, 24 ALASKA per epoch, with **zero repeats** in both epochs. The standard-only checkpoint validation and old test were used. All 16 notebook code cells completed in 422.446 s; the slow post-fit test/diagnostic cell took 347.849 s on the PC. This is a wiring check, not a training result. Its scratch manifests and outputs must not be committed.
- The standalone panel scorer completed ALASKA test_seen (3 pairs, 2.76 s) and LSDIR test (7 pairs, 3.05 s) on the PC with separate setting hashes and output folders. These few-pair timings do not predict full-panel cost or measure model quality.
- **Pending:** receive final selections, verify split and photo hash disjointness, build the final manifests after masters are available, calculate actual panel sizes/time, decide checkpoint-validation policy, and obtain approval before a metered G4 run. Do not edit Worker_3's selection or Pi scripts.
