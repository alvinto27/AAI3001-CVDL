# Frozen source records

Status: plan step 1 complete, 2026-10-13. The generator and loader are not
implemented. Records cover 1,000 RAISE photos and 900 DIV2K images.

## Files and identities

Row counts exclude the header. Sizes are bytes. All files use UTF-8 and LF.

| File | Rows | Bytes |
| --- | ---: | ---: |
| `data/raise/photo_manifest.csv` | 1,000 | 385,759 |
| `data/raise/legacy_tile_manifest.csv` | 49,796 | 4,191,526 |
| `data/div2k/photo_manifest.csv` | 900 | 183,549 |
| `data/div2k/split_seed0.csv` | 900 | 45,587 |

RAISE photo IDs keep their NEF stems. Indices 0..999 follow the full sorted
RAISE list, not subset or worker order. DIV2K IDs are `div2k_0001` through
`div2k_0900`; index = `999 + numeric_ID`, giving 1000..1899. All 1,900 IDs
and indices are unique. Source fields are `raise` and `div2k`.

Master paths are relative to the repository. Legacy `clean_path` and
`stego_path` are relative to `Datasets/raise512-run1`, exactly as recorded in
its metadata. Original cover IDs retain leading zeros. Original cover
indices are not renumbered. The builder does not require the old tile PNGs.

Photo manifests include source type, camera, full oriented dimensions,
512/256 overlap rows, columns, and starts. Starts use semicolon-separated
integers. RAISE dimensions come from its audited preparation record; no RAW
decode is performed. DIV2K dimensions come from RGB8 PNG headers, including
header checksum validation. DIV2K has blank camera and legacy-grid fields;
noise is `not_applicable`. RAISE noise is `disabled`.

## Split and count checks

RAISE split rows are joined through metadata, labels, and preparation tiles.
Every photo has one split; clean/stego twins agree. Existing split rows stay
unchanged: 701 train photos, 150 validation photos, and 149 test photos.
The CSV value `val` means validation.

| RAISE camera | Train photos | Validation photos | Test photos |
| --- | ---: | ---: | ---: |
| D7000 | 500 | 107 | 107 |
| D90 | 195 | 42 | 41 |
| D40 | 6 | 1 | 1 |
| Total | 701 | 150 | 149 |

DIV2K split is now frozen, not a proposal. Use sorted official train IDs
0001..0800 and `Generator(PCG64(0)).permutation(800)` with NumPy 2.5.3.
The first 680 positions are train; the remaining 120 are validation.
Official validation IDs 0801..0900 are test. This matches `default_rng(0)`
in this version. Do not redraw the split during training. Future NumPy
versions can change generator transformations; verify against the frozen CSV.

| Source / split | Photos | Overlap 512 tiles | Overlap 256 tiles |
| --- | ---: | ---: | ---: |
| RAISE, all | 1,000 | 65,184 | 243,120 |
| DIV2K, train | 680 | 8,276 | 32,216 |
| DIV2K, validation | 120 | 1,456 | 5,720 |
| DIV2K, test | 100 | 1,228 | 4,808 |
| DIV2K, all | 900 | 10,960 | 42,744 |
| Combined | 1,900 | 76,144 | 285,864 |

Counts are single grids, before clean/stego duplication. All master paths
exist on the VM. RAISE's 49,796 legacy tiles have complete top-left grids:
D7000: 714 photos, 54 tiles each; D90: 278, 40 each; D40: 8, 15 each.
The six camera/orientation groups match the [design audit](plan-otf-generator.md#grid-policies-and-checked-counts).

## Rebuild safely on the VM

Edit and commit code on the PC. Copy only code to VM scratch, not data or
`.git`. Execute with the existing VM environment. Do not write into the VM
checkout. Example, from the scratch copy:

```sh
cd ~/otf-dev/step1-manifests
PYTHONDONTWRITEBYTECODE=1 /srv/aai3001/AAI3001-CVDL/.venv/bin/python \
  scripts/build_manifests.py \
  --repo-root /srv/aai3001/AAI3001-CVDL \
  --tiles-manifest ImageSets/RAISE/tiles512-records/manifest.csv \
  --output-root "$PWD/build-new"
```

The preparation manifest is canonical under `ImageSets/RAISE/tiles512-records`.
The VM has no dataset `records/tiles` directory. Its input SHA-256 is:

```text
400f0b3e823f088183316c4948e983fb74b6808acb4059cf9eb8506fcb35f6fb
```

`--tiles-manifest` accepts a repository-relative or absolute input path.
The script reads existing selection, split, labels, metadata, preparation
records, and PNG headers. It prints input hash, checks, output hashes, row
counts, sizes, and NumPy version. It never overwrites an existing output.
It rejects output roots or output symlinks leading into the source checkout.

Run twice with different output roots and compare the CSV bytes. Copy the
new records to the PC and compare SHA-256 before committing explicit paths.
Do not push. Remove the scratch copies and test outputs after review.

## Verification result

- Nine VM unit tests passed, including bad PNG modes/depth/checksums, duplicate
  keys, malformed CSV, source-checkout protection, and no-overwrite checks.
- Two full VM builds produced byte-identical CSV files.
- Every CSV copied to the PC has the same SHA-256 as both VM builds.
- All identity, split, camera-grid, tile-count, and master-path checks passed.
- No image generation, training, RAW decode, or source-file changes occurred.

| Output | SHA-256 |
| --- | --- |
| RAISE photo manifest | `75e92882a8734cdde0b11bcc48871dfcd4f8168a9286c99cc30ce6c38f2f5a6d` |
| RAISE legacy tile manifest | `7e2bcb4986102a5cf39b7a20b5634292bb8dcc9a6c2d64b38317370aca0a0ee5` |
| DIV2K photo manifest | `83cdfd2fe5c1cd2eff731c47d5d5aad1751405f0ae723e0afde8a02231e25640` |
| DIV2K split | `7d43a8275c8bbeb9b85161ee97a18f65dddda68682be2d14f3a4bf65018bd3e3` |

Done: full source decode and grid cutting in
[plan step 2 verification](source-verification.md) and
[step 3 settings/seed helpers](settings-and-seeds.md). The
[step 4 embed core](embed-core-verification.md) is also verified.
Pending: tile labels, pipeline, and loader work.
Source terms and citations remain in the [ImageSets guide](../ImageSets/README.md#citations-and-terms).
