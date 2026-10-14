# Notebook 04 diagnostic speed and parity

Status: checks complete. The PC QUICK and VM full-size comparisons below
used an existing, fixed best validation checkpoint on each machine. The checks
ran notebook 04's diagnostic code alone, without training or changing saved
checkpoints.

## Change

In OTF mode, notebook 04 previously indexed a random `PairSubset` from the
main process for each diagnostic batch. `FixedPairs` caches only its last full
photo. Random indices therefore loaded the same full photo repeatedly,
including each mixed/clean/stego mode. The diagnostic now groups its existing
sample indices by photo ID, gets each clean/stego pair through the unchanged
`FixedPairs.__getitem__` path, and retains only the resulting uint8 tiles in
RAM. The grouped load does **not** change sample order: a small dataset returns
the pairs in the original subset index order. The same RNG seed, labels,
model mode, batch sizes, GPU conversion, and metric code are used. PNG mode
stays unchanged. The sample is 32 pairs per split in QUICK or 1,000 per split
in full mode; the full working set is about 1.57 GB for one 512×512 split.
The photo cache reference is cleared after preloading, and the temporary
loader/tiles are released after each split.

A synthetic unit test checks reordered photo IDs, preserved pair order and
one load per distinct photo. A real QUICK test compared every pixel in all
32 sampled clean/stego pairs with the original indexed dataset: identical.

## Timed identical-check comparisons

The harness loaded each machine's existing best v3c/clip3/GroupNorm checkpoint,
then executed the same diagnostic cell with the original and new OTF branches.
It did not write to `checkpoints/`. Timings include preload/decode, inference,
and all mixed/clean/stego passes. The environment kept the original native
TF32 settings. Every printed `CHECK` and `GroupNorm` line is compared exactly,
not just within a tolerance.

| Machine/source | Size | Original seconds | Preloaded seconds | Printed check lines |
| --- | --- | ---: | ---: | --- |
| PC, ROCm / full-photo decode | QUICK | 194.128251 | 17.332828 | exact match |
| VM, L4 / full-photo cache | full | 371.642150 | 100.451854 | exact match |
| PC, ROCm / full-photo decode | full | Not completed (>45 min in the user's run; scratch stopped after 26 min) | 464.575727 | Old scratch prefix matches; different PC/VM checkpoints, so no cross-machine equality claim |

The PC QUICK comparison included evaluation accuracy/loss, train-normalization
accuracy/mean-probability in three modes for both train and validation, and
all GroupNorm train/eval max-logit differences. The VM full comparison included
the same fields on 1,000 fixed pairs per split (2,000 images each). Each
GroupNorm difference printed zero in both variants. No full training run was
started for this timing. The full PC baseline was intentionally stopped because
it was still decoding in its first normalization mode after 26 minutes. The
user's earlier PC run took over 45 minutes without finishing the old cell.
Only the revised full PC cell was timed to completion: 464.575727 seconds
(7.74 minutes). Its original-cell scratch run printed `CHECK train a` and
`CHECK train c` and the GroupNorm difference before stopping; those three
lines exactly match the new full PC run. The PC and VM checkpoints are
different (`best-epoch07-val_pe0.0149` versus `best-epoch07-val_pe0.0146`),
so their full CHECK lines must not be compared directly. The complete full
VM old/new lines and complete PC QUICK old/new lines match exactly with the
same per-machine checkpoint. The updated PC full run completed every CHECK
line for train and validation, with zero GroupNorm max-logit difference.

## Notebook 02 breakdown on PC predictions

Executed all code cells of `notebooks/02_eval_breakdown.ipynb` with the PC run's
21,912-row `test_predictions.csv` copied to a temporary output folder. The
notebook wrote its per-setting CSV and four heatmaps into that temporary
folder, not the checkpoint folder. It detected self-contained OTF metadata,
so it did not read `labels.csv` for joins. Overall P_E was 0.017296, AUC was
0.997441.

| Source | Pairs | P_E | Detection rate at global threshold |
| --- | ---: | ---: | ---: |
| DIV2K | 1,228 | 0.061889 | 0.911238 |
| RAISE | 9,728 | 0.011513 | 0.983861 |

## Cache-record comparison

The earlier PC-versus-VM cache-record comparison is complete in
[cache verification](cache-verification.md): 1,900 matching normalized JSON
records, excluding the machine-local `master_mtime_ns` field, and zero
mismatches.
