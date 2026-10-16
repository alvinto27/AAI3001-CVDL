> Archived 2026-10-16: done; verification remains for reference. See [findings](../findings.md).

# Mode 1 sample API and legacy replay

Status: plan step 5 complete. The sample gate, full legacy replay, OTF
identity checks, and all unit tests passed on the team VM.
No loader, torch Dataset, cache, training, or GPU code is part of this step.

## API

`lsb_mode1.py` imports NumPy, standard Python types, and only the existing
source/settings/engine modules. It owns no crop, capacity, bit-write, settings
draw, or seed-key formula.

```python
pixels, metadata = make_sample(full_photo, photo_manifest_row,
    TileSpec("overlap_last", 512, row=0, col=0), config,
    OTFSeed(run_seed=7, purpose="train", epoch=0, variant=1,
            fixed_variants=False), kind="stego")
```

| Input or output | Meaning |
| --- | --- |
| Full photo | Oriented RGB8 array from `lsb_source.load_photo()`. |
| TileSpec | Grid policy, real tile size, row, column. Source rules validate it. |
| LegacySeed | Seed, original cover discovery index, variant; top-left 512 only. |
| OTFSeed | Run seed, purpose, epoch, variant, fixed-variants flag; shared v1 key. |
| kind | `clean` or `stego`; label is 0 or 1. |
| pixels | Owned RGB8 tile. The clean route copies its source view; stego uses `embed_fast()`. Caller edits cannot change the source photo. |
| geometry metadata | Photo ID/index, source, split, grid policy, tile size, row/column, y/x, kind/label. |
| seed metadata | Version plus effective settings and embedding entropy lists. Their part order is defined in [settings and seeds](../active/settings-and-seeds.md). |
| settings | Every drawn engine argument except its RNG, plus shared draw metadata. OTF includes requested/effective bpp, requested/effective bytes, capacity, and clipped flag. |
| embedding | All 12 public engine result fields, including actual start/stride. No mask or selected-unit cache is retained. |

Clean samples draw no settings: `settings` and `embedding` are None. Their
seed metadata describes the paired sample identity; they do not embed a
payload. For stego, requested start in `settings` and actual start in
`embedding` remain separate. OTF records effective key parts, not ignored
evaluation run seed/epoch. Thus evaluation metadata is stable as well as pixels.

The source module has `tile_at()` for direct access. It shares manifest-axis
validation and `_tile_view()` with `iter_tiles()`, without duplicating geometry
rules. It still returns a view; only the Mode 1 sample API returns owned pixels.

```python
pixels, metadata = replay_legacy(full_photo, photo_manifest_row,
    legacy_tile_manifest_row, run1_config, "stego")
```

This adapter checks photo/source/split and stored coordinates, then calls the
same sample function with the frozen cover index and run1 config seed.
Variant defaults to 1. No source or dataset files are read by the sample API.

## Verification workflow

Use `scripts/archive/check_mode1.py` from VM scratch with the existing VM Python.
The repository root is a read-only source of manifests, masters, YAML, and
stored PNG pixels. No generated dataset is written.

```sh
OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 \
  /srv/aai3001/AAI3001-CVDL/.venv/bin/python scripts/archive/check_mode1.py \
  --repo-root /srv/aai3001/AAI3001-CVDL --gate sample
# Run full only after sample passes; stop/report any mismatch.
# Repeat command with --gate full, then --gate otf.
```

The sample gate includes 500 clean/stego pairs over all cameras/orientations
and all placements. Full replay uses eight spawned workers, OMP=1, and decodes
each NEF once. Decode, cut+embed, and PNG read+comparison seconds are summed
worker time, not additive wall-clock stages. Workers execute these phases
concurrently; wall time also includes process/dispatch overhead.

OTF checks use six RAISE camera/orientation examples and two DIV2K photos,
512/256 edge tiles, all three purposes, four fixed-variant epochs, and an
intentional lower-to-fit case. They repeat in two serial call orders and in
four processes; compare byte hashes AND metadata.

## Results

| Check | Result |
| --- | --- |
| First sample | 500 pairs, 12 NEFs, all six camera/orientation groups, zero clean/stego pixel mismatches. |
| Sample placements | Contiguous 166, stride 163, random 171. |
| Sample time | Wall 10.071320 s; summed worker decode 36.133047 s, cut+embed 2.068902 s, PNG read+compare 19.072245 s. |
| Full replay | All 49,796 pairs from 1,000 NEFs, decoded once each; zero clean/stego pixel mismatches. Eight workers, OMP=1. |
| Full placements | Contiguous 16,609; stride 16,623; random 16,564; all six camera/orientation groups. |
| Full time | Wall 724.325993 s; summed worker decode 3,534.838352 s, cut+embed 209.146641 s, PNG read+compare 2,025.623093 s. |
| OTF cases | Eight photos (six RAISE camera/orientation groups, DIV2K 0001/0002), both overlap sizes, 48 purpose cases per execution layout. Repeat calls have identical bytes and metadata. |
| OTF seed/epoch | All 32 val/test cases keep bytes and metadata when run seed or epoch changes; all 16 train cases change bytes with each change. |
| Fixed variants | All 16 photo/size cases keep bytes and metadata at epochs 0, 1, 5, 100. |
| Lower-to-fit | All 16 cases record clipped=True for requested 10 bpp: 512 effective 0.00299072265625 bpp / 98 bytes; 256 effective 0.0029296875 bpp / 24 bytes. |
| Order/process count | Two serial photo orders and four spawned processes give identical sample hashes and metadata; zero differences. 816 total sample calls over those three layouts. |
| OTF time | Total 25.779619 s, including source decode and all three execution layouts. |
| Tests | 34 source/manifest/settings/fast/Mode 1 tests plus 34 engine/batch/prepare tests: 68 passed. |

Remove scratch code and logs after review. Do not remove masters, stored
images, manifests, or dataset records. This sample/replay module is the
Mode 1 component of the shared pipeline. A loader is not implemented here.
