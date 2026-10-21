# Archived experiment env files

These files are historical parity fixtures, not submission inputs.

| Old file | Replacement tracked notebook |
| --- | --- |
| `newdata-1.env` | [newdata-1.ipynb](../../../../notebooks/newdata-1.ipynb) |
| `wide-1.env` | [wide-1.ipynb](../../../../notebooks/wide-1.ipynb) |

Submit a notebook plus commit. Its first code cell owns experiment values;
the G4 profile owns workers, paths and performance flags. See the
[G4 runbook](../../../../docs/active/g4-batch-runner.md#submit-a-run).

`../speed-1.env` remains active: it configures the separate speed-matrix tool,
not a notebook experiment. The recipe tests resolve these archived env files
directly to check the replacement notebooks. The archived comparison tool can
also read them for manual historical reproduction; it is not a required check.
