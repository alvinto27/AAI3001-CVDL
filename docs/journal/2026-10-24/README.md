# 2026-10-24

## Submission model changed to run 18

### Did

- Showed the user run 18's cell figure (`runs/verification/cell-curriculum50-paired-2026-10-23/cells/cells.png`): an exact hit on the RAISE contiguous band, a partial hit on the ALASKA contiguous band, flat on the random-placement tile (as for run 13 and R2).

### Decided

- **User: run 18 is the submission model for now** (`srm-cell-aux-curriculum50-paired-20261023T133537Z-1366`, `last.ckpt`, SHA-256 `eee9c63a...0b80`, eval `b3d9a49137b5dbd7bd5a`, tile cut-off .5094). It replaces run 13.
  - Standard P_E 2.62% (run 13 2.97%, R2 2.43%); cell head .867 (run 13 .863, R2 .860); contiguous cell recall / FA 51.5% / 3.8% (R2 52.7% / 4.5%).
  - It failed its frozen rule against R2 (no established difference either way); the choice is the user's.

### Next

- Pending: re-fit the photo verdict rule (c) on run 18's scores (same frozen method) and rebuild the deploy bundle on `pipeline` (no G4).
