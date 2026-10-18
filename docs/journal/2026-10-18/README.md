## 2026-10-18 — Project survey, train-matched validation design approved

Recorded by the Orchestrator.

### Did

- Worker_1 surveyed the project (read-only) and wrote a design for the train-matched validation (work list #1). The Orchestrator checked the survey claims against the files.
- Design record: [train-matched validation plan](../../active/plan-train-matched-validation.md).
- Container `aai3001` started (it was stopped).

### Found

- Validation pools at 256 overlap-last (val split only): RAISE 36,484 tiles (150 photos), DIV2K 5,720 (120), ALASKA 8,966 (1,000), LSDIR 30,584 (1,500).
- Photo split (train/val/test): RAISE 701/150/149, DIV2K 680/120/100, LSDIR 7,000/1,500/2,500, ALASKA 13,500/1,000/1,700 seen + 2,000 unseen.
- The tiling result table labels its P_E as "validation min-P_E cut-off" (`docs/results/result-training-tiling.md` line 14); the 2026-10-17 handoff section 4 labels the same numbers "threshold 0.5". Pending check (side job A).
- Stale records: `README.md` calls notebook 04 the current baseline; `docs/status.md` says the Pi is online (handoff: shut down); the 2026-10-17 journal "Next" still says launch `srm-hard`; run queue row 8 has no result and "After the runs" still asks for the padding check.

### Decided

- User: validation size = **10% of the training pairs per epoch**, exact training source mix, from the existing `val` photo split; no split changes; unique tiles; the builder fails if a pool is too small. SRM recipe 18,000 pairs (7,200 / 7,200 / 3,600); `srm-hard` 21,600 (7,200 each).
- User: no augmentation; k, bpp and stego type follow the training config.
- DIV2K threshold-calibration panel moves to the per-source threshold task (#4).
- User: delete leftovers (old `runs/.scratch/` worktrees except `srm-hard-launch`, speed/profile/BF16/map-demo runs, `g4-prime-1`, `bt2-job`, `early-pc`; `g4-v4-r1b` and `g4-v4-r2` only if no record links them).
- User: side job A (resolve the threshold label) after B.

### Next

- Implement the train-matched validation on a fresh worker (Worker_1 out of context). Pending.
- Leftover cleanup. Pending.
- Side job A. Pending.
