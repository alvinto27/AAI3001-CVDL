# Archived one-off scripts

These scripts made a record or checked a feature once. They are kept as evidence and are not run by any job, test or notebook.
Each one reads the PNG dataset `Datasets/generated/raise512-run1-records` (or other data), so it **needs the rebuilt PNG dataset** (see `docs/archive/rebuild-raise512-run1.md`) to run again. The `sys.path` lines use `parents[2]` (repository root).

| Script | What it verified / made | Record |
| --- | --- | --- |
| `alaska_select.py` | ALASKA selection v1 (superseded). `scripts/alaska_select_v2.py` still imports its constants from here. | [alaska-selection-v2.md](../../docs/active/alaska-selection-v2.md) |
| `check_sources.py` | Full read-only verification of source records and readers. | [source-verification.md](../../docs/archive/source-verification.md) |
| `check_settings.py` | Legacy settings redraw and three-cover production generation. | [settings-and-seeds.md](../../docs/active/settings-and-seeds.md) |
| `check_embed_fast.py` | Fast embed core parity with stored legacy samples. | [embed-core-verification.md](../../docs/archive/embed-core-verification.md) |
| `check_mode1.py` | Mode 1 replay gates and OTF identity. | [mode1-verification.md](../../docs/archive/mode1-verification.md) |
| `check_torch_loader.py` | Torch loader parity, counts, speed and GPU idle checks. | [torch-loader-verification.md](../../docs/archive/torch-loader-verification.md) |
| `map_demo_gn_check.py` | Showed nonlocal GroupNorm influence in `map_demo.py` edits. | [findings.md](../../docs/findings.md) |
| `map_demo_nonorm_check.py` | Showed no-norm suspicion cells stay unchanged outside the edited chunk. | [findings.md](../../docs/findings.md) |

`check_notebook_quick.py` was deleted (replaced by `scripts/run_notebook.py`); use git history if needed.
