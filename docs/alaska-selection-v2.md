# ALASKA selection v2

`data/alaska/selection_v2.csv` lists the ALASKA#2 various-size colour TIFFs for training and evaluation.

- **Created by:** `scripts/alaska_select_v2.py`, using header scans from `scripts/alaska_scan.py`.
- **Status:** final, made after the complete Pi download (70,282/70,282 files, 0 failures).
- **SHA-256:** `437fcfef817689cd1af5347f08f948cfddfad6523667afd9dace37eb5ef1ff85`
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
   - 6 models: Canon_PowerShot-G11, Kodak_Z990, Pentax_K10D, Pentax_ist-DS, Sigma_SD10, Nikon_D90.
   - They have no val or test_seen photos (Orchestrator decision).
4. **Photo budget:**
   - **Total:** 16,200 photos are spread evenly over the 30 seen models. A model with fewer photos gives all it has, and the others share the rest.
   - **Use cap:** a mixed model uses at most 75 % of its photos (`MAX_USE`).
   - **Val and test_seen:** the 1,000 val and 1,700 test_seen photos are shared equally over the 24 mixed models.
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
| train | 13,500 | 21,141 | 21,301,332,312 | 3,285 | 10,215 | 16,082,622,534 |
| val | 1,000 | 1,552 | 1,562,059,792 | 214 | 786 | 1,213,316,916 |
| test_seen | 1,700 | 2,672 | 2,686,627,688 | 342 | 1,358 | 2,133,106,316 |
| test_unseen | 2,000 | 3,153 | 3,173,550,368 | 2,000 | 0 | 0 |
| **all** | **18,200** | **28,518** | **28,723,570,160** | **5,841** | **12,359** | **19,429,045,766** |

Train has 21,141 tiles, above the 19,000 target.

## Per model

"min gap" is the smallest ID distance between two photos of different splits in one model.

| model | available | train (tiles) | val | test_seen | test_unseen | min gap | note |
|---|---|---|---|---|---|---|---|
| Apple_IpadPro-1stGen | 2052 | 443(653) | 42 | 71 | 0 | 11 |  |
| Apple_IpadPro-2ndGen | 971 | 444(693) | 42 | 70 | 0 | 11 |  |
| Canon_EOS-100D | 3732 | 444(700) | 41 | 71 | 0 | 11 |  |
| Canon_EOS-500D | 2800 | 443(711) | 42 | 71 | 0 | 12 |  |
| Canon_EOS-60D | 2750 | 0(0) | 0 | 0 | 250 | None |  |
| Canon_EOS20D | 2388 | 443(687) | 42 | 70 | 0 | 11 |  |
| Canon_EOS700D | 624 | 356(546) | 41 | 71 | 0 | 11 |  |
| Canon_PowerShot-G11 | 333 | 333(513) | 0 | 0 | 0 | None | train only |
| Google_Pixel-XL | 1812 | 0(0) | 0 | 0 | 250 | None |  |
| HTC_OneA9 | 1318 | 443(745) | 41 | 71 | 0 | 11 |  |
| HTC_OneM9-OPJA100 | 835 | 443(685) | 42 | 71 | 0 | 11 |  |
| Huawei_Honor8-FRDL09 | 1134 | 443(709) | 42 | 71 | 0 | 11 |  |
| Huawei_P20lite-ANE-LX1 | 1040 | 0(0) | 0 | 0 | 250 | None |  |
| Kodak_Z990 | 832 | 555(876) | 0 | 0 | 0 | None | train only |
| LG_G5-H840 | 1876 | 444(686) | 42 | 70 | 0 | 11 |  |
| LG_G7fit-LMQ850 | 1098 | 443(689) | 42 | 71 | 0 | 11 |  |
| Leica_M9 | 1874 | 443(725) | 41 | 71 | 0 | 11 |  |
| Nikon_1-AW1 | 850 | 444(714) | 41 | 71 | 0 | 11 |  |
| Nikon_D3000 | 2620 | 444(674) | 41 | 71 | 0 | 11 |  |
| Nikon_D5200 | 4020 | 445(670) | 41 | 70 | 0 | 11 |  |
| Nikon_D5600 | 2656 | 443(696) | 42 | 71 | 0 | 21 |  |
| Nikon_D610 | 2721 | 443(708) | 42 | 71 | 0 | 11 |  |
| Nikon_D7100 | 907 | 443(712) | 42 | 71 | 0 | 11 |  |
| Nikon_D810 | 3344 | 443(696) | 42 | 71 | 0 | 11 |  |
| Nikon_D90 | 395 | 395(637) | 0 | 0 | 0 | None | train only |
| Panasonic_DMC-GM1 | 1675 | 442(684) | 42 | 71 | 0 | 11 |  |
| Panasonic_DMC-TZ60 | 1313 | 0(0) | 0 | 0 | 250 | None |  |
| Panasonic_FZ28 | 1667 | 443(649) | 42 | 71 | 0 | 11 |  |
| Pentax_K10D | 5461 | 556(856) | 0 | 0 | 0 | None | train only |
| Pentax_K50 | 3151 | 0(0) | 0 | 0 | 250 | None |  |
| Pentax_ist-DS | 1076 | 556(894) | 0 | 0 | 0 | None | train only |
| Ricoh_GRDigital-III | 1262 | 0(0) | 0 | 0 | 250 | None |  |
| Samsung_GalaxyS8-SM-G950F | 2188 | 442(705) | 42 | 71 | 0 | 11 |  |
| Samsung_GalaxyS9plus-SM-G965F | 1311 | 0(0) | 0 | 0 | 250 | None |  |
| Sigma_SD10 | 2380 | 556(849) | 0 | 0 | 0 | None | train only |
| Sigma_SD1Merrill | 3546 | 442(681) | 42 | 71 | 0 | 11 |  |
| Sony_A7R | 2032 | 0(0) | 0 | 0 | 250 | None |  |
| Sony_Alpha6000 | 1585 | 443(698) | 41 | 71 | 0 | 11 |  |

## Notes

- **Small models:** with the full download, only Canon_PowerShot-G11 (333 photos) and Nikon_D90 (395 photos) are small.
- **Why the list has six models:** the list was decided on the partial download, where 4 of the 6 models seemed to have only 250 photos.
- **Other train-only models:** the other 4 have 832–5,461 photos. They are capped at their even share (555 or 556 photos).
- **Duplicates:** the selection has no repeated file SHA-256. In the whole pool there is one byte-identical pair (58794, 59282). It is in `RAW_wesaturate`, which is excluded.
- **Copies:** none yet. 12,359 files (19.43 GB) must come from the Pi. PC home had 736 GB free when the selection was made.
