# LSDIR selection v1

`data/lsdir/selection_v1.csv` lists the LSDIR HR photos for training and evaluation. The photos come from the 17 HR train shards.

- **Created by:** `scripts/lsdir_select.py`, from the Pi inventory.
- **Decision:** user, 2026-10-16.
- **SHA-256:** `c7e1b3209e96f1eda6ba59c4896ddbd639fffb7797e63048b5c4d674d65c9dbf`
- **Reproducible:** two runs gave identical bytes.

## Input

| input | rows | SHA-256 |
|---|---|---|
| `/srv/datasets/lsdir/inventory.csv` (Pi, from `scripts/pi/lsdir_extract.py`) | 84,991 | `31371bb2f01512139a70724794049ca9574274c1b95d20d50790e7825f12b0dc` |

- **Not committed:** the inventory is 9.8 MB. The script refuses any other SHA-256.
- **Photos:** all are RGB 8-bit PNGs. Every side is at least 540 px.
- **IDs:** 1–84,991 with no gap. Shard `shard-NN` holds IDs `5000·NN+1` to `5000·NN+5000`.

## Rules

1. **Shard balance:** each split's total is shared over the 17 shards in proportion to shard size (largest remainder, seeded tie-break). Each shard gives 411–412 train, 88–89 val and 147–148 test photos.
2. **Leakage rule:**
   - **Blocks:** in each shard, photos in ID order are cut into blocks of 10 consecutive IDs. A whole block goes to one split.
   - **Gaps:** blocks of different splits are never next to each other, so the smallest ID gap between splits is 11.
   - **Fallbacks:** 0.
   - **Seed:** blocks are taken in seeded-hash order (test, then val, then train). The last block of a split is cut to the quota by seeded hash.
3. **Why blocks:** LSDIR photos come from web sources, and neighbouring IDs can share a source or a scene. Two measurements support this:
   - Neighbouring IDs (distance 1) have the exact same width × height in **3.79 %** of pairs. Random pairs match in **1.14 %**.
   - The match rate falls with distance: 3.06 % at 2, 2.15 % at 5, 1.66 % at 10, 1.24 % at 50, and 1.18 % at 1,000.
   - The only byte-identical pair (`0022000/0021599.png` = `0022000/0021600.png`) has consecutive IDs.

   The blocks cost almost nothing here, because only 11,000 of 84,991 photos are used.
4. **Duplicate pair:**
   - **`0021600`:** never used.
   - **`0021599`:** may only be in train. In this seed its block was not chosen, so neither file is in the selection.
5. **Selection seed:** version `LSDIR_HR_train/selection_v1`, seed 0, hashed with shard and block.

## Totals

| split | photos | tiles512 | bytes |
|---|---|---|---|
| train | 7,000 | 14,475 | 12,634,438,119 |
| val | 1,500 | 3,164 | 2,746,328,895 |
| test | 2,500 | 5,245 | 4,561,215,999 |
| **all** | **11,000** | **22,884** | **19,941,983,013** |

- **Tiles per photo:** 1–35; the median is 2 in every split.
- **Val pairs:** 3,164 pairs (one per tile). The capped validation (`V3_VAL_CAP_LSDIR=1500`) takes a seeded sample of 1,500 of them.
- **Copies:** none yet. All 11,000 files (19.94 GB) must come from the Pi.

## Per shard

All shards have a smallest ID gap of 11 and 0 fallbacks.

| shard | train | val | test |
|---|---|---|---|
| shard-00 | 412 | 88 | 148 |
| shard-01 | 412 | 88 | 147 |
| shard-02 | 411 | 88 | 147 |
| shard-03 | 412 | 88 | 147 |
| shard-04 | 412 | 88 | 147 |
| shard-05 | 412 | 88 | 147 |
| shard-06 | 412 | 89 | 147 |
| shard-07 | 411 | 88 | 147 |
| shard-08 | 412 | 88 | 147 |
| shard-09 | 412 | 88 | 147 |
| shard-10 | 412 | 88 | 147 |
| shard-11 | 411 | 89 | 147 |
| shard-12 | 412 | 88 | 147 |
| shard-13 | 412 | 88 | 147 |
| shard-14 | 412 | 89 | 147 |
| shard-15 | 412 | 89 | 147 |
| shard-16 | 411 | 88 | 147 |
