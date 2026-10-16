"""Shared metrics retain the frozen notebook results and saved panel summaries."""
import ast
import glob
import json
from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from cvdl import metrics


def frozen_metrics(notebook, index):
    source = "".join(json.loads((ROOT / notebook).read_text())["cells"][index]["source"])
    names = {"_threshold_counts", "auc_numpy", "probability_error", "reference_auc", "reference_probability_error"}
    nodes = [n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name in names]
    ns = {"np": np}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), notebook, "exec"), ns)
    return ns


class SharedMetricsTests(unittest.TestCase):
    def test_frozen_notebook_functions_exact(self):
        rng = np.random.default_rng(123)
        samples = [(np.array([], dtype=int), np.array([])),
                   (np.zeros(5, dtype=int), np.linspace(0, 1, 5)),
                   (np.ones(5, dtype=int), np.linspace(0, 1, 5)),
                   (rng.integers(0, 2, 500), np.round(rng.random(500), 2))]
        for notebook, cell in (("notebooks/05_xunet_v4_nonorm.ipynb", 19),
                               ("notebooks/02_eval_breakdown.ipynb", 4)):
            ns = frozen_metrics(notebook, cell)
            for labels, scores in samples:
                for name in ("auc_numpy", "probability_error"):
                    with self.subTest(notebook=notebook, name=name, count=len(labels)):
                        self.assertEqual(ns[name](labels, scores), getattr(metrics, name)(labels, scores))

    def test_saved_bpp1_panel_if_present(self):
        paths = glob.glob(str(ROOT / "runs/g4-v4-r2/checkpoints/*/extra-test-bpp1/test_predictions.csv"))
        if not paths:
            self.skipTest("saved g4-v4-r2 bpp-1 panel is not present")
        for path in paths:
            frame = pd.read_csv(path, float_precision="round_trip")
            frame["bpp_bin"] = pd.cut(frame["requested_bpp"],
                bins=[0., .05, .15, .3, .5, .75, 1.],
                labels=['<=0.05', '0.05-0.15', '0.15-0.30', '0.30-0.50', '0.50-0.75', '0.75-1.0'],
                include_lowest=True, right=True)
            actual = pd.DataFrame(metrics.summary_rows(frame))
            saved = pd.read_csv(Path(path).with_name("summary.csv"), float_precision="round_trip")
            pd.testing.assert_frame_equal(actual, saved, check_exact=True)


if __name__ == "__main__":
    unittest.main()
