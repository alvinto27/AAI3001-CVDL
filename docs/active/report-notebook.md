# Saved-output report notebook (work list #5)

Use one shared template: `notebooks/templates/report.ipynb`.
The generator writes and executes `runs/<run>/report.ipynb` on the CPU.
It does not run a model, fit a threshold, or export HTML.
No additional package is required in the `aai3001` container.

## Generate a report

```sh
podman exec -w /work aai3001 python -B scripts/build_report.py \
  runs/srm-e50-20261017T111310Z-0214 \
  --evaluation-id 35afd1c7938bae78968c
```

Use `--reference-run runs/<other-run> --reference-evaluation-id <id>` for an
optional reference. Both runs must use the same frozen standard and hard test
recipes. Each model keeps its own saved validation threshold.

If there is one evaluation folder, the generator can use it without an id.
If there are zero or multiple folders, it stops and lists the available ids.
It refuses missing, pending, QUICK, or mismatched calibration results.
It also requires the saved epoch metrics, learning-rate trace, and training plot.

The generator refuses an existing report. Use `--overwrite` only if you intend
to replace all author notes with a fresh template. It executes the fresh report
in memory and replaces the target only after all cells pass.

## Read the report

- Read the run card and the exact validation cutoff first.
- Original training curves retain their checkpoint-selection validation label.
  Old capped or standard 512 validation is not the new post-hoc 256 calibration.
- Main tables use the one saved validation cutoff. Test minimum P_E and fixed
  0.5 results are separate diagnostics.
- Miss tables use saved stego probabilities and requested bpp. The final bpp bin
  includes 0.50. No threshold is fitted.
- Hard-test plots show both FA and miss with saved Wilson 95% ranges. These
  ranges do not account for correlation between tiles from the same photo.
- Source-known thresholds are reference results only. There is no DIV2K or
  mixed RAISE+DIV2K reference result.
- Fill in the author notes. The template does not generate conclusions.

Plots, including the original training PNG, are embedded in the notebook.
The notebook reads local saved files only. GPU visibility is disabled; model and
threshold-fitting imports and external Python network requests are blocked.
Local Jupyter communication is permitted. This is a report guard, not a security
sandbox for arbitrary user code.

The report sections are run card, training curves, headline results, misses by
k/bpp/k x bpp, hard tests, optional model reference, optional higher-k results,
source-known references, and author notes.
