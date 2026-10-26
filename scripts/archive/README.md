# Archived one-off scripts

These scripts made a record or checked a feature once. They are kept as evidence, not as active job or notebook entry points. Archive-only tests are in `tests/archive/`; the default CPU suite does not discover them. New active code must not import the seven scripts archived on 2026-10-26. The older ALASKA constants import listed below is unchanged.

Some older scripts read `Datasets/generated/raise512-run1-records` and **need the rebuilt PNG dataset** (see `docs/archive/rebuild-raise512-run1.md`). Other scripts need their recorded source data. Root paths account for the archive location, including the nested `g4/` directory. Archiving a command does not approve its execution or a new evaluation.

| Script | What it verified / made | Record |
| --- | --- | --- |
| `evaluate_gapcov_context.py` | Retained GAP+covariance adapter for the original #11 panel; routine non-colour context scoring is dropped. | Frozen #11 panel and #21 implementation history. |
| `evaluate_learn_srm_context.py` | Retained learned-SRM adapter for the original #11 panel; routine non-colour context scoring is dropped. | Run20 implementation history; dropped output remains quarantined. |
| `evaluate_rgb_candidate_context.py` | Retained B36 adapter for the original #11 panel; not an active scoring entry point. | Run19 colour-context history. |
| `g4/step_profile.py` | Compared notebook-04 raw and Lightning training-step timings. | [2026-10-14 journal](../../docs/journal/2026-10-14/README.md). |
| `loader_speed_matrix.py` | Compared packed-loader timings serially on CPU. | Historical packed-loader verification logs. |
| `measure_hard_noise_256.py` | Scanned or checked the frozen 256-pixel hard-test noise records. | [Hard-test recipe](../../docs/active/plan-hard-test-256.md); no rebuild is approved. |
| `build_new_source_manifests.py` | Built ALASKA/LSDIR manifests from frozen selections; its provenance test moved to `tests/archive/test_new_source_manifests.py`. | [New-source code plan](../../docs/archive/plan-new-data-code.md). |
| `alaska_select.py` | ALASKA selection v1 (superseded). `scripts/alaska_select_v2.py` still imports its constants from here. | [alaska-selection-v2.md](../../docs/active/alaska-selection-v2.md) |
| `check_sources.py` | Full read-only verification of source records and readers. | [source-verification.md](../../docs/archive/source-verification.md) |
| `check_settings.py` | Legacy settings redraw and three-cover production generation. | [settings-and-seeds.md](../../docs/active/settings-and-seeds.md) |
| `check_embed_fast.py` | Fast embed core parity with stored legacy samples. | [embed-core-verification.md](../../docs/archive/embed-core-verification.md) |
| `check_mode1.py` | Mode 1 replay gates and OTF identity. | [mode1-verification.md](../../docs/archive/mode1-verification.md) |
| `check_torch_loader.py` | Torch loader parity, counts, speed and GPU idle checks. | [torch-loader-verification.md](../../docs/archive/torch-loader-verification.md) |
| `map_demo_gn_check.py` | Showed nonlocal GroupNorm influence in `map_demo.py` edits. | [findings.md](../../docs/findings.md) |
| `map_demo_nonorm_check.py` | Showed no-norm suspicion cells stay unchanged outside the edited chunk. | [findings.md](../../docs/findings.md) |

`check_notebook_quick.py` was deleted (replaced by `scripts/run_notebook.py`); use git history if needed.
