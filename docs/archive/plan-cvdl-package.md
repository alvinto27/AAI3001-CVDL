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

Phase 2 adds `cvdl.data`: reader/filter helpers, explicit-root PNG datasets,
label/pair/split helpers and explicit-argument PNG/OTF builders. The OTF builders
delegate decoding, seeds and sampling to the existing LSB library. Notebook 05
keeps its definitions until thinning. `check_b_tiles.py` uses the shared KV
filter (CHW and batched BCHW supported). Same-process tests execute notebook
05 cells 6/8/12 against synthetic decoded fixtures: paired/unpaired PNG loaders
and three OTF settings (256 px, paired 512 px, mixed/capped 512 px), workers 0,
train epochs 0/1 and fixed evaluation. Frames and batch tensors match exactly.
Default golden compare is ALL EQUAL; **132 tests pass**. No new GPU checks.

Phase 3 adds `cvdl.lit` and `cvdl.callbacks`, preserving the notebook's class
names and checkpoint parameter names. Timing callbacks, validation data,
priming loader and optional conv1 conversion are explicit runtime arguments;
validation data and the conversion factory are excluded from saved hparams.
The package-built CPU run (data/models/lit/callbacks) matches all 20 golden
losses, final weights SHA `5525eb37…`, metrics.csv and checkpoint structure
exactly. The test replays only the still-unmoved setup and throwaway model
probe to preserve the original RNG stream. All four old real checkpoints load
strictly with identical state tensors. Resuming the fresh last.ckpt restores
optimizer/callback/epoch state, reaches step 22, and preserves prior CSV rows.
The capped-validation sanity/source guard is tested with explicit validation
data. Default compare is ALL EQUAL; **135 tests pass**. No GPU runs added;
notebooks 04/05 remain unchanged until thinning.

### Phase 4a: settings API and transition

`cvdl/settings.py` owns the `SCHEMA` definition table. Each entry has a name,
type parser, default (possibly dependent on QUICK/epochs), allowed values,
one-line meaning, legacy env name and machine-ownership flag. `load()` returns
a frozen `Settings`; nested maps and sequences are immutable. Example for the
future notebook top cell:

```python
from cvdl.settings import load
settings = load(model="xunet_wide", width=4, lr=3e-4, norm="none")
# Later, after the run directory is known:
settings.write(run_dir)
```

Resolution order is **defaults/profile < notebook < override**. Legacy env
values remain supported: they fill unset fields; equal notebook/env values
keep notebook provenance. Conflicting notebook/env experiment values fail with
both values and a `--set` instruction unless that field has an explicit override.
Machine env fields continue to override profile defaults.
Unknown names and bad values stop before data loading; close names/enum values
include a `did you mean` hint. `settings.provenance` maps each name to default,
notebook, profile, env or override. `settings.sources` remains the selected data
source tuple. `to_run_settings()` builds the existing LSB settings without
reading photos. `legacy_globals()` is the temporary golden/runner adapter.

| Machine choice | Rule |
| --- | --- |
| Detection | Google DMI plus CUDA RTX PRO 6000 selects `g4`; contradictory hardware or a job/profile mismatch fails loudly (phase 5) |
| Train/eval workers | PC 6/4; G4 32/16 |
| QUICK default | PC true; G4 false; legacy env and explicit overrides can change it |
| Precision | Default 32; bf16-mixed only with GPU BF16 support |
| Output root | PC `runs/`; G4 `runs-out/`; runner `--output` selects its checkpoint root |
| Paths and performance flags | Machine-owned; notebook arguments cannot set them |

A notebook cannot set machine fields, even to the same value. Explicit
one-off overrides can set workers/QUICK/paths/precision, but cannot change
read-only device, precision limits or checkout root. The runner supports
repeatable canonical `--set extra_tests=none --set workers=4` and legacy
`--set V3_EXTRA_TESTS=none`. Phase 4c also passes settings with no legacy env
reader (for example `--set seed=9`) directly to the thin notebook's `load()`.
If a changed knob changes the run prefix, supply a matching run_name or use
`--set run_name=` to request a new derived name. Existing prefix guards and
all five golden env-file names remain unchanged.

The notebook-05 headless runner records `run_settings.json` in the actual run
folder after setup: final values, a source for every value, ordered override
history, machine profile, git commit and dirty flag. It records this once per
run folder, before training. The unchanged notebook gets profile worker/QUICK
defaults through its env bridge. Existing explicit G4 env files remain valid;
notebook 04 still uses its historical runner path. The saved runner output root
is an override from `--output`. Compile-cache env values are also recorded.

The QUICK extra-panel cap is an [intended difference](refactor-golden.md#intended-difference-quick-extra-panel-cap-phase-4a).
Settings tests compare every captured effective global and LSB setting for
newdata-1, wide-1, CPU and GPU QUICK, plus the speed-tool configuration/name.
They use captured hardware profiles without GPU computation; only absolute
checkout paths are mapped when the test checkout is not `/work`. Tests also
cover provenance, errors, machine guards, immutable values, repeated CLI
settings, saved git metadata, the runner-to-JSON path with a no-training shell
stub, notebook/env conflicts, and deterministic QUICK panel selection. Default
golden compare is ALL EQUAL; **144 tests pass**. Notebooks remain independent.

### Phase 4b: evaluation and diagnostics

`cvdl.evaluation` provides explicit-device paired prediction, prediction CSV
saving, notebook-05 per-setting tables, fixed checkpoint panels, and saved-file
breakdowns with four heatmaps. `cvdl.metrics.global_summary` preserves notebook
02's threshold scan, including its first tied minimum and infinite endpoints;
this differs from the validation-cutoff tie rule. Other metric formulas remain
in `cvdl.metrics`. The checkpoint scorer is moved, not copied:
`scripts/eval_checkpoint.py` is its CLI and legacy env-aware boundary (archived 2026-10-21 to `scripts/archive/eval_checkpoint.py`; [legacy-512 retirement](legacy-512-retirement.md)). Package
calls receive root, device, QUICK and panel selection as arguments and do not
read env values. QUICK still caps extra panels at 16 pairs with seed 1.

`cvdl.diagnostics` selects/preloads seed-0 pairs and returns full-precision
batch-dependence results for eval and train modes c/e/f. The caller passes a
split dataset and owns the diagnostic pair-ID record. The original network is
not changed. Workers, normalization, device and QUICK are explicit.

Notebooks 02, 04 and 05 are unchanged. Notebook 02 remains the standalone
saved-file reference until notebook thinning; the package already accepts
self-contained OTF predictions and PNG predictions with an explicit label CSV.
Same-process tests compare real g4-v4-r2 saved predictions and synthetic PNG
metadata against notebook 02: tables, CSV bytes, global metrics and all four
PNG heatmap bytes match exactly. Notebook-05 tests compare CPU predictions,
CSV bytes and setting tables in OTF/PNG mode, plus diagnostic numbers for
none/group/batch normalization on both train and val. The panel scorer is
checked with a small CPU checkpoint and full/QUICK subsets. Default golden
compare is ALL EQUAL; **151 tests pass**. No GPU runs.

### Phase 4c: thin notebook 05

The [current notebook](../../notebooks/05_xunet_v4_nonorm.ipynb) contains **10 code
cells / 4,284 code characters**, down from **16 / 81,097**. Its first code cell
uses `settings = load()` with a commented experiment example and no explicit
experiment values. Existing G4/golden env files therefore remain valid without
notebook/env conflicts. It imports package helpers, builds data/loaders, runs
the model checks, builds/fits the Trainer, predicts, evaluates extra panels,
writes breakdowns, and runs batch diagnostics.

`cvdl.runtime` supplies explicit-argument wiring and the historical count-model
and KV-probe checks. Keeping those two allocations preserves the seeded model
initialization sequence. It has no env readers. Run settings are written before
fit and at completion. `cvdl.settings` also owns compiler environment setup,
the headless/diagnostic-worker settings, and a scoped runner context. That
context lets the literal `load()` top cell retain the ordered `--set` overrides
and their provenance without passing runner globals into the notebook. Resume
checks the override list as well as the bridged env settings.

The [frozen reference](../../notebooks/archive/05_baseline_kv_xunet_v4_pre-cvdl.ipynb)
retains every old source cell and has no outputs. Independent parity tests use
it; capped-sanity and output-retention behaviour tests use package code. The
golden settings/data/CPU drivers execute the new notebook, not the archive.
They preserve exact comparisons except for the explicitly named new settings
file. All [allowed differences](refactor-golden.md#thin-notebook-drivers-and-allow-list-phase-4c)
are printed and regression-tested; unknown changes still fail.

Notebook 04 is untouched. Notebook 02 remains standalone and unchanged as the
independent saved-prediction reference; the thin notebook calls its equivalent
package breakdown functions directly. G4 whitelist names are unchanged.

Evidence: default compare is ALL EQUAL; **156 tests pass**. The single GPU
QUICK smoke finished in **79 seconds**, all **10 code cells** passed, and the
structural comparison is ALL EQUAL. Standard prediction max absolute delta and
all recorded train/validation scalar deltas were **0** (informational, not a new
GPU equality rule). The real runner's `--set lr=0.0003` appears in final values,
override history and source `override` in `run_settings.json`. Used allow-list
entries were only `run-settings-file`, `thin-cell-layout` and
`quick-extra-panel-cap`. The 16-pair extra-panel settings SHA is
`e50a63091a5a6abb82f09fca11619a3a5641f102a8f6e2c743684ed4e20061cb`.

### Phase 5: pinned notebook jobs

G4 submissions use `run --notebook PATH [--commit SHA] [--timeout-min N]
[--dry-run]`. A clean checkout and a commit reachable from `origin/yx` are
required even in dry-run. The PC container validates the selected commit's
notebook and settings/config code in a temporary snapshot with the declared
G4 profile; it reads literal experiment keywords from the first code cell and
prints effective values, provenance and RUN_NAME before any cloud command.
No data or GPU work is needed for pre-flight.

[Newdata-1](../../notebooks/newdata-1.ipynb) and
[wide-1](../../notebooks/wide-1.ipynb) copy notebook 05 with only their top code
cell changed. Historical env files now live in
[scripts/g4/jobs/archive](../../scripts/g4/jobs/archive/README.md) as golden
fixtures, not submission inputs. Every experiment value, RunSettings field,
RUN_NAME and comparable global matches the golden. The two hardware differences
are asserted separately: the output root is `runs-out/` and the ROCm conv1 fix
is inactive on CUDA G4. G4 owns compile=default and next-epoch prime=true as
machine performance flags, along with 32/16 workers and QUICK=false.

The existing `job-env` channel carries only `V3_NOTEBOOK` and `CVDL_JOB_ID`.
The checked-out job script validates the path at the pinned checkout; archive,
missing, symlink and traversal paths fail. The installed bootstrap is unchanged
(SHA-256 `89cc83e9c73cc43476f6daf5e77825ddc298a32a22a52af8ded0909f15b1f2af`)
and already execs the checked-out `job_run.sh`, which calls its checked-out
`run_notebook.py`. No selector shim, reinstall or maintenance boot is needed.
The [runbook](g4-batch-runner.md#submit-a-run) records the handover and strict
DMI/CUDA GPU detection checks. Runtime jobs require the detected G4 profile;
wrong detection cannot silently select PC workers.

The separate speed matrix retains `speed-1.env` and its existing
`run --env-file ...` command. That exception is accepted only for the speed
tool, whose settings also receive a PC pre-flight. It still dispatches
`scripts/speed_matrix.py`, not notebook cells.

Evidence: default compare is ALL EQUAL; **166 tests pass**, including 10
no-cloud recipe/profile/submission tests. Valid dry runs print G4 settings and
only the two metadata lines; dirty/unpublished commits, bad/archive/symlink
paths, top-cell typos and old metadata-incompatible commits fail before cloud
commands. The installed-bootstrap → checkout-job-script → checked-out-runner
handover is tested with a stub runner, without training or network access.
No VM boot or GPU run is part of this phase.
