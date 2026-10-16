# Project status — 2026-10-16

**Start here:** All research and training work is **on hold at the user's request**. Do not start a new PC experiment, G4 run, or work from a plan without a new user decision. The last mixed-source wide XuNet run completed, and the PC validation-threshold and ALASKA false-alarm check B completed. The [per-cell design](active/plan-per-cell-training.md) is written, **not approved**, and not implemented. Use [findings](findings.md) for the run table and [journal days](journal/README.md) for dated records.

## Current state and items

- **Completed:** ALASKA/LSDIR source records and mixed sampler; final verified masters; G4 mixed run and [result](results/result-newdata-1.md); PC threshold/noise check B and [result](results/result-check-b.md). The frozen standard test and RAISE/DIV2K records remain in use.
- **On hold:** A, per-cell training (design only, no code or model run); C, capacity/longer training/BF16; fair R0 control, seed repeats, front-end changes, and other free checks. No follow-up is authorized by the plans.
- **Deferred records:** prebuilt writer, OTF plan step 17: [historical design](archive/plan-otf-generator.md), OTF implemented but writer deferred. Noisy covers: [parked by the user](archive/plan-noisy-covers.md). G4 GPU LSB generator: [shelved design](archive/plan-gpu-generator.md), not built.
- **Hardware and delivery:** the G4 finished the mixed job and shut itself down; it is **stopped**. The Pi is online. Its USB-SSD swap file was removed; **2 GiB zram only** remains. See the [Pi record](active/pi-host-changes.md) and [2026-10-16 journal](journal/2026-10-16/README.md). Do not start the VM to resend a result.

## User decisions still pending

These rows carry the pending decisions from the archived [research checklist](archive/research-checklist.md). They are **not** authorizations to act. The checklist's ALASKA option-B row is historical: the later source record and mixed run already use the selected option B; confirm any further selection request with the user before changing records.

| Decision | Owner | Current record |
| --- | --- | --- |
| ALASKA train-only models: six or two (option B) | User | Checklist marked Pending; Orchestrator recommended option B. Final selection used option B; do not reopen it without a new decision. |
| BF16 for G4 | User | Pending; speed matrix complete, batch 32 retained. |
| Fair control R0 | User / Orchestrator | Pending before a causal source/model gain claim. |
| Seed repeats | User / Orchestrator | Pending; one seed is not enough for small differences. |
| Front-end upgrade | User / Orchestrator | Pending after the wide and new-source results. |

The old checkpoint-selection validation decision is **done**: the mixed run used the fixed 8,035-pair capped list, not the earlier checklist estimate. Free checks still listed as pending in that checklist are noise-versus-miss, denoise-and-compare, tanh saturation, and a filter linear probe. Check B completed a noise-versus-error analysis; do not relabel the other checks as done or start them while work is on hold.

## Key commits and records

| Commit | Record |
| --- | --- |
| `e592cce`, `1b54858` | Opt-in source readers/mixer and capped checkpoint validation. |
| `daf893c`, `2659b3d` | Final ALASKA and LSDIR records. |
| `1a6a6bf`, `3a6b1fd` | Sanity-validation fix and successful G4 checkout. |
| `ab9d3f1` | Mixed-source [result](results/result-newdata-1.md). |
| `042903d` | PC [check B](results/result-check-b.md). |
| `a6170ed` | Per-cell [design only](active/plan-per-cell-training.md). |
| `465de12` | Latest dated journal update before this documentation reorganization. |
| `45ec074` | Documentation reorganization (status, findings, journal by day). |

The [documentation index](README.md) lists each current, result, and historical record. The [run guide](active/g4-batch-runner.md) is reference material only while work is on hold.
