# Shared `cvdl` package plan

Notebook 04 is frozen (including its cell positions). New work starts from
notebook 05. `cvdl/` holds reusable Python code; notebooks hold the experiment,
its narrative and result cells. `LSB-Dataset-Generator/` remains the independent
flat generator library. `cvdl` adds its directory to the import path once;
entry-point scripts add the repository root, not each library folder.

## Settings contract (for the later settings phase)

| Kind | Owner / source | Examples |
| --- | --- | --- |
| Experiment | Notebook's top code cell, tracked in git | Model/loss, LR/schedule, source mix, epochs, validation selection |
| Machine | Automatically selected machine profile; never typed into a notebook | PC workers 6/4; G4 workers 32/16; QUICK, precision capability limits, paths |
| One-off trial | Explicit `run_notebook.py --set key=value` | A local experiment override; every use is recorded |

One schema defines each setting once: name, type, default, allowed values or
range, and one-line meaning. Unknown names and invalid values stop before data
loading with the setting name, supplied value and allowed values in the error.
No second table of defaults is maintained in notebooks, profiles or scripts.

Resolution: **defaults < notebook top cell < explicit one-off override**.
Automatic profiles supply machine-owned fields in their own namespace (the
notebook cannot set them); capability limits validate the final result rather
than silently changing an experiment. Profiles do not override experiment
fields. Save the final values, source of each value (`default`, `notebook`,
`profile`, `override`), override history, profile identity and git commit in the
run folder. Build RUN_NAME from resolved values with today's spelling.

A G4 job names a **notebook + commit**, not an experiment env file. During the
transition, all existing `V3_*` env names continue to work unchanged: a temporary
adapter maps experiment fields into the experiment layer and worker/path/device
fields into the machine layer. Keep the golden's existing env interpretation
until the settings phase explicitly changes it. Then migrate the experiment
contents of `newdata-1.env` and `wide-1.env` into tracked notebook top cells;
replace worker entries with the automatic G4 profile. `speed-1.env` is a
separate speed-matrix tool, not a training notebook; preserve its tool settings
until an explicit tool-settings migration. Never silently translate a conflicting
notebook/env value: reject it or require the explicit one-off override.

Open points before that phase:
- Reliable profile detection (GCE metadata versus explicit machine identity),
  unknown-machine behaviour, and how local users select QUICK automatically.
- Whether an explicit override may lower worker counts/precision, and how to
  prevent overriding hard device capability limits.
- A notebook top-cell format that supports a prototype-specific model/loss
  without importing and executing the whole notebook to read settings.

## Module ownership and phases

| Phase | Work | Required evidence |
| --- | --- | --- |
| 0, complete | [Before-values and comparison harness](refactor-golden.md) | Exact CPU/settings/data/four-checkpoint results; documented GPU noise rules |
| 1 | `models`: KV/XuNet/wide/features/cell maps; `metrics`: tied-threshold metrics, summaries and validation cutoffs. Repoint analysis scripts | Entire golden compare; saved-panel metric equality; full tests |
| 2 | Data/readers/splits/loader construction | Train epochs 0/1 and val/test hashes identical |
| 3 | Lightning module, scheduling, callbacks | CPU losses/weights equal; checkpoint keys and resume; GPU QUICK |
| 4 | Settings schema/profiles, evaluation, diagnostics; thin notebook 05/02 | All golden checks; explicitly reviewed QUICK extra-panel capping differences |

Phase 1 leaves notebook 05's model/metric definitions intact as an independent
before-reference for parity tests; they become imports when the notebook is
thinned. Notebook 04 and the G4 tools that extract its cells stay untouched.
Future prototypes import reusable helpers and only define the model/loss or
settings that are special to that experiment.

Safety: retain parameter names (`net.*`), hyperparameter names, callback state
keys, seed domains, tensor dtype, operation order and run-name spelling. Do not
replace numerically equivalent metric formulas if their floating results differ.
Record git commit in run status in the settings/output phase. No G4 boot is
needed for phase 1; run a QUICK smoke at the next planned G4 boot before a real
job. The notebook whitelist and runner globals remain unchanged.

Phase-1 evidence: default golden compare is ALL EQUAL; GPU package/notebook
parity is exactly zero for all four real checkpoints and the same 64 tiles;
130 tests pass. Re-scoring the saved g4-v4-r2 bpp-1 prediction CSV reproduces
its complete saved summary exactly (round-trip float parsing). Notebook 04/05
and G4/Pi job scripts are unchanged.
