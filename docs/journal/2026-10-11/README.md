# Steganalysis project journal

This journal records the stego work in this repository. Entries are in date order.
Numbers are kept when they are known. Pending work is marked.

## 2026-10-11 — Dataset and first baseline

### Did

- Replaced the old v5 generator with v6. The work is in `LSB-Dataset-Generator/`.
  - The embed engine supports `lowest` and `plane` bit modes, `k=1..8`,
    `replace` and `match`, and contiguous, stride, or random placement.
  - Payloads are uniform random bytes or letters. The bulk runner uses separate
    random-number streams, resumes work, replays output, and writes PNG text
    metadata and `metadata.csv`.
  - The cover tool decodes RAW files with rawpy 0.27.1, camera white balance,
    sRGB, and 8-bit output. It cuts 512 x 512 tiles. It does not resize.
  - Main commits: `b208258`, `8f1c52d`, `2255168`, and `a95cf61`.
- Chose plan A: cut tiles first, then embed. Plan B (embed full photos, then
  tile) is parked. Plane matching means `v +/- 2^(k-1)`. Random bytes are the
  encrypted-like payload.
- Downloaded 1,000 RAISE NEFs with seed 0. Selection was stratified. The
  download was about 18 GB. The downloader is in `scripts/download_raise.py`.
- Built `Datasets/raise512-run1`.
  - 49,796 clean tiles, 49,796 stego tiles, and 49,796 masks.
  - `labels.csv` has 99,592 data rows.
  - Generator seed 1 and one variant per cover.
  - Random settings per tile, with bpp 0.05 to 0.5 and stride 2 or 16.
  - Tiles by camera: D7000 38,556; D90 11,120; D40 120.
  - Replay verification passed 49,796 of 49,796 tiles.
- Split by source photo, not by tile. The split is 70/15/15 and is stratified
  by camera. The committed source split is `data/raise/split_seed0.csv`.
- Built baseline notebook `notebooks/01_baseline_kv_xunet.ipynb` in `d2fddcc`.
  - It uses a fixed KV filter. A KV filter is a high-pass filter that exposes
    small pixel changes. It is followed by an Xu-Net style CNN with 19.5 K
    parameters.
  - Clean and stego twins are in the same batch. Training uses flips and
    90-degree rotations. Inputs are raw 0-255 float32 values.
  - It uses Adam at 1e-3 and StepLR, with a tenfold drop at epoch 5.
  - P_E means the minimum average of false-alarm and miss rates over thresholds.
    P_E=0 is perfect detection and P_E=0.5 is guessing. AUC is the area under
    the receiver-operating-characteristic curve.
- Added the rebuild record in `docs/archive/rebuild-raise512-run1.md` and the VM files
  in `configs/raise512-run1.yaml`, `data/raise/selection.csv`, and
  `scripts/rebuild_raise512_run1.sh` (`9435257`, `672c627`).

### Found

- The RAISE subset is large but repeatable. The clean and stego counts match.
- Plan A keeps each stego tile tied to a known clean tile.
- The team Spot VM has an NVIDIA L4, 8 CPUs, and 32 GB RAM in Seoul. Its disk
  was grown from 100 GB to 200 GB with Balanced PD.
- The first VM rebuild had four of 1,000 downloads fail during a short network
  problem. The script stopped at step 1 and shut the VM down as designed.

### Decided

- Rebuild the dataset on the VM from committed files instead of uploading
  about 38 GB.
- Use `--verify` and file counts as checks. Do not add SHA-256 fingerprints.
- Keep the baseline in float32. Small KV residuals must not be rounded away.

### Next

- Pending: rerun the VM rebuild after the network failure.
- Pending: finish a baseline run and record held-out results.
