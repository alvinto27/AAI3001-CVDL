# Saved-output report notebook (work list #5)

Use the shared template, `notebooks/templates/report.ipynb`.
The generator writes and executes `runs/<run>/report.ipynb` on the CPU.
It does not run a model, fit a threshold, or export HTML.
No additional package is required in the `aai3001` container.
The [readability plan](plan-report-readability.md) specifies this presentation.

## Generate a report

```sh
podman exec -w /work aai3001 python -B scripts/build_report.py \
  runs/srm-e50-20261017T111310Z-0214 \
  --evaluation-id 35afd1c7938bae78968c
```

Use `--reference-run runs/<other-run> --reference-evaluation-id <id>` for an
optional reference. Both runs must use the same frozen standard and hard test
recipes. Each model keeps its own saved validation cut-off.

For a run with cell outputs, add `--cell-dir <cell_eval output>` and, for the control map's cut-off row, `--cell-reference-dir <control cell_eval output>` (`5981ab2`). The report then has a "Where in the tile? (cell heatmap)" section with the saved figure and tables. Without these options the report does not change. Run 13 example: `--cell-dir runs/verification/cell-run13-2026-10-21/cells --cell-reference-dir runs/verification/cell-control-2026-10-21`.

If there is one evaluation folder, the generator can select it without an id.
If there are zero or multiple folders, it stops and lists the available ids.
It refuses missing, pending, QUICK, or mismatched calibration results.
It also requires the saved epoch metrics, learning-rate trace, and training plot.

The generator refuses an existing report. Use `--overwrite` only to replace all
author notes with a fresh template. It executes the new report in memory and
replaces the target only after all cells pass.

## Shared data illustration

The report embeds [the saved data figure](../figures/report/data-sample.png). It does
not open datasets or generate pixels. To rebuild the image on CPU, run
`python -B scripts/build_data_figure.py`; use `--check` to rebuild without writing
and compare the PNG bytes. The script pins seven pair ids from the frozen standard
and hard-test lists. It replays saved embedding arguments and seed keys, checks
both clean and message pixels against the evaluator's `StandardTestPairs` output,
and draws the recorded overlap-last grid. No settings are drawn again and no model
is loaded. A shared 256-colour palette keeps the saved PNG small; pixel equality
is checked before rendering. Rebuild with the same image and plotting software
versions for byte-identical output. The real-data rebuild test skips when the
frozen datasets are not installed.

## Read the report

Each section starts with a question. Chart hints (dashed lines, colours,
shading) are in the chart legend or in a short note under the chart. Charts come first;
full tables are in the collapsed appendix. The report does not generate conclusions.

| Section | Contents |
| --- | --- |
| What does the data look like? | Shared saved figure: four source photos with tile grids, clean/message tiles, amplified change maps for three placements, and a noisy ALASKA example. |
| 0. How good is this model, in short? | Four test-panel error rates, each with a plain "about 1 in N wrong" line, and standard-test AUC; definitions of false alarms, misses, overall error and AUC. |
| 1. What was trained? | A short training card and listed differences from an optional reference. |
| 2. Did training go well? | Training and original-validation loss and accuracy; learning-rate drops marked. |
| 3. Where does it fail? | False alarms beside misses for each test panel. |
| 4. Which hidden messages does it miss? | Misses by embedding strength k, then a strength-by-message-size colour grid; dark means more misses. |
| 5. Does it break on hard images? | False alarms and misses with saved Wilson 95% ranges and a 50% coin-flip line. |
| 6. Better or worse than the reference? | Change in overall error only; green is lower and red is higher. |
| 7. Would knowing the source help? | One line per source with deployment and source-known error rates. |
| 8. Author notes | Placeholders for the author's evidence, limits and decisions. |
| Appendix | Collapsed full result, diagnostic, miss, hard-image, source-known and optional higher-k tables; reference tables when supplied. |

- Original training curves retain their checkpoint-selection validation label.
  Old capped or standard 512-pixel validation is not the later 256-pixel
  calibration sample. The old training PNG is required as a saved record but is
  not shown: readable charts are drawn from the saved epoch CSV instead.
- k is embedding strength; bpp is requested message size in bits per pixel.
  The last bpp bin includes 0.50. Miss rates use saved stego probabilities and
  the saved cut-off, with no new threshold fit.
- Hard-test ranges use tile/pair trials. They do not account for tiles from the
  same photo. Small messages use 0.05–0.15 bpp; noisy images use the frozen noise
  selection. Hard-test cells can share images.
- Source-known results are reference results only, not deployment results.
  There is no DIV2K or mixed-source reference. Both ALASKA camera groups appear
  on one source line and use the same source-specific cut-off.
- Single-source panels do not repeat an overall row as a source row.
- Display rates have one decimal, AUC and cut-offs have three decimals. The
  saved inputs retain full precision. Detailed tables have horizontal scrolling.

## Hidden details and checks

Code inputs are collapsed with JupyterLab `source_hidden`, `hide_input`, and
`hide-input` tags; outputs remain visible. Use a notebook viewer that supports
these input-hiding settings. Expand inputs only to inspect the implementation.
Identifiers, hashes, paths and full-precision cut-offs are absent from displayed
content, including the appendix. Evaluation, threshold, calibration recipe,
checkpoint and test-recipe identities remain in `metadata.report` for checks.
The saved evaluation files retain the complete records.

All charts are embedded PNG outputs. The notebook reads local saved files only.
GPU visibility is disabled; model and threshold-fitting imports and external
Python network requests are blocked. Local Jupyter communication is permitted.
This is a report guard, not a security sandbox for arbitrary user code.
