# ALASKA selection v2

`data/alaska/selection_v2.csv` lists the ALASKA#2 various-size colour TIFFs for training and evaluation.

- **Created by:** `scripts/alaska_select_v2.py`, using header scans from `scripts/alaska_scan.py`.
- **Status:** final, made after the complete Pi download (70,282/70,282 files, 0 failures).
- **SHA-256:** `d3f122d637b09d9c65e7cafd8592f211c8b8381a3b3fa80303363bde5255206d` (regenerated 2026-10-16 for option B; supersedes `437fcfef…ff85`)
- **Reproducible:** two runs with the same inputs gave identical bytes.

## Inputs

| input | rows | SHA-256 | check |
|---|---|---|---|
| Pi scan of `/srv/datasets/alaska/tiff_various_color` | 70,282 | `d76537a3…aae8f0` | every file re-hashed and equal to the Pi `SHA256SUMS` (0 bad) |
| PC scan of v1 copy `Datasets/alaska/tiff_various_color` | 9,723 | `c4d178fe…afc67` | every file re-hashed and equal to its `SHA256SUMS` (0 bad) |

- Together the two scans cover all 80,005 IDs, with no ID missing.
- Every file is an uncompressed RGB TIFF with 8 bits per channel and sides of 512, 640, 720 or 1024.
- **Not committed:** the scan CSVs. Regenerate them with `alaska_scan.py DIR SHA256SUMS OUT.csv --verify`.

## Rules

1. **Excluded:**
   - `*_FLATS`, `*_TEXTURES` and `RAW_wesaturate`, as in v1.
   - `Leica_Mono`: monochrome sensor, so R = G = B.
2. **Unseen-camera test (`test_unseen`):**
   - 8 models: Google_Pixel-XL, Samsung_GalaxyS9plus-SM-G965F, Huawei_P20lite-ANE-LX1, Panasonic_DMC-TZ60, Ricoh_GRDigital-III, Canon_EOS-60D, Sony_A7R, Pentax_K50.
   - 250 photos each, all taken from v1.
   - These models have no photos in any other split.
3. **Train-only models (`TRAIN_ONLY`):**
   - 2 models: Canon_PowerShot-G11 (333 photos) and Nikon_D90 (395 photos; Nikon_D90 also occurs in RAISE).
   - They have no val or test_seen photos. They are too small for a block split with gaps (user decision 2026-10-16, option B, from full-download counts).
4. **Photo budget:**
   - **Total:** 16,200 photos are spread evenly over the 30 seen models. A model with fewer photos gives all it has, and the others share the rest.
   - **Use cap:** a mixed model uses at most 75 % of its photos (`MAX_USE`).
   - **Val and test_seen:** the 1,000 val and 1,700 test_seen photos are shared equally over the 28 mixed models.
5. **Leakage rule:**
   - **Blocks:** each model's photos, in ID order, are cut into blocks of 10 consecutive photos. A whole block goes to one split.
   - **Gaps:** blocks of different splits are never next to each other. An empty block always separates them, so the smallest ID gap between two splits of one model is 11.
   - **Runs:** val and test_seen blocks are placed in runs of up to 3 blocks, so fewer empty blocks are needed.
   - **Fallbacks:** 0 (no split blocks touch).
   - **Why 10:** ALASKA IDs follow capture order, so a burst of one scene has consecutive IDs. 10 is an assumed maximum burst length; it is not measured.
6. **Seed:**
   - **Blocks:** chosen by most v1 photos first (fewer files to copy), then by a seeded hash of (version, seed 0, model, block).
   - **Photos:** inside the chosen blocks, v1 photos are taken first, then photos in seeded-hash order.

## Totals

| split | photos | tiles512 | bytes | already on PC (in_v1) | to copy | bytes to copy |
|---|---|---|---|---|---|---|
| train | 13,500 | 21,139 | 21,286,565,208 | 3,244 | 10,256 | 16,125,765,536 |
| val | 1,000 | 1,535 | 1,559,841,040 | 223 | 777 | 1,193,453,082 |
| test_seen | 1,700 | 2,648 | 2,671,494,248 | 360 | 1,340 | 2,105,125,208 |
| test_unseen | 2,000 | 3,153 | 3,173,550,368 | 2,000 | 0 | 0 |
| **all** | **18,200** | **28,475** | **28,691,450,864** | **5,827** | **12,373** | **19,424,343,826** |

Train has 21,139 tiles, above the 19,000 target.

## Per model

"min gap" is the smallest ID distance between two photos of different splits in one model.

| model | available | train (tiles) | val | test_seen | test_unseen | min gap | note |
|---|---|---|---|---|---|---|---|
| Apple_IpadPro-1stGen | 2052 | 459(680) | 36 | 61 | 0 | 11 |  |
| Apple_IpadPro-2ndGen | 971 | 460(715) | 36 | 60 | 0 | 11 |  |
| Canon_EOS-100D | 3732 | 460(726) | 35 | 61 | 0 | 11 |  |
| Canon_EOS-500D | 2800 | 459(734) | 36 | 61 | 0 | 12 |  |
| Canon_EOS-60D | 2750 | 0(0) | 0 | 0 | 250 | None |  |
| Canon_EOS20D | 2388 | 459(708) | 36 | 60 | 0 | 31 |  |
| Canon_EOS700D | 624 | 372(576) | 35 | 61 | 0 | 11 |  |
| Canon_PowerShot-G11 | 333 | 333(513) | 0 | 0 | 0 | None | train only |
| Google_Pixel-XL | 1812 | 0(0) | 0 | 0 | 250 | None |  |
| HTC_OneA9 | 1318 | 459(769) | 35 | 61 | 0 | 11 |  |
| HTC_OneM9-OPJA100 | 835 | 459(711) | 36 | 61 | 0 | 11 |  |
| Huawei_Honor8-FRDL09 | 1134 | 459(731) | 36 | 61 | 0 | 11 |  |
| Huawei_P20lite-ANE-LX1 | 1040 | 0(0) | 0 | 0 | 250 | None |  |
| Kodak_Z990 | 832 | 458(733) | 36 | 61 | 0 | 11 |  |
| LG_G5-H840 | 1876 | 460(708) | 36 | 60 | 0 | 11 |  |
| LG_G7fit-LMQ850 | 1098 | 460(715) | 36 | 60 | 0 | 11 |  |
| Leica_M9 | 1874 | 459(751) | 35 | 61 | 0 | 11 |  |
| Nikon_1-AW1 | 850 | 460(738) | 35 | 61 | 0 | 11 |  |
| Nikon_D3000 | 2620 | 460(696) | 35 | 61 | 0 | 12 |  |
| Nikon_D5200 | 4020 | 461(698) | 35 | 60 | 0 | 21 |  |
| Nikon_D5600 | 2656 | 459(724) | 36 | 61 | 0 | 11 |  |
| Nikon_D610 | 2721 | 459(730) | 36 | 61 | 0 | 11 |  |
| Nikon_D7100 | 907 | 459(739) | 36 | 61 | 0 | 11 |  |
| Nikon_D810 | 3344 | 460(726) | 36 | 60 | 0 | 21 |  |
| Nikon_D90 | 395 | 395(637) | 0 | 0 | 0 | None | train only |
| Panasonic_DMC-GM1 | 1675 | 458(705) | 36 | 61 | 0 | 11 |  |
| Panasonic_DMC-TZ60 | 1313 | 0(0) | 0 | 0 | 250 | None |  |
| Panasonic_FZ28 | 1667 | 460(675) | 36 | 60 | 0 | 11 |  |
| Pentax_K10D | 5461 | 460(708) | 36 | 60 | 0 | 21 |  |
| Pentax_K50 | 3151 | 0(0) | 0 | 0 | 250 | None |  |
| Pentax_ist-DS | 1076 | 460(730) | 35 | 61 | 0 | 11 |  |
| Ricoh_GRDigital-III | 1262 | 0(0) | 0 | 0 | 250 | None |  |
| Samsung_GalaxyS8-SM-G950F | 2188 | 458(728) | 36 | 61 | 0 | 11 |  |
| Samsung_GalaxyS9plus-SM-G965F | 1311 | 0(0) | 0 | 0 | 250 | None |  |
| Sigma_SD10 | 2380 | 459(706) | 36 | 61 | 0 | 11 |  |
| Sigma_SD1Merrill | 3546 | 458(708) | 36 | 61 | 0 | 11 |  |
| Sony_A7R | 2032 | 0(0) | 0 | 0 | 250 | None |  |
| Sony_Alpha6000 | 1585 | 458(721) | 36 | 61 | 0 | 11 |  |

## Notes

- **History:** the first version (SHA-256 `437fcfef…ff85`) had 6 train-only models. That list was chosen on the partial download, where Kodak_Z990, Pentax_K10D, Pentax_ist-DS and Sigma_SD10 seemed to have only 250 photos. With full counts (832–5,461 photos), these 4 models are now normal mixed models.
- **Duplicates:** the selection has no repeated file SHA-256. In the whole pool there is one byte-identical pair (58794, 59282). It is in `RAW_wesaturate`, which is excluded.
- **Copies:** none yet. 12,373 files (19.42 GB) must come from the Pi. PC home had 736 GB free when the selection was made.
