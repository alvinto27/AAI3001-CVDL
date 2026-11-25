"""Notebook 04 OTF diagnostic preload keeps pair order and avoids repeated decodes.

The notebook helper delegates to lsb_torch.preload_pairs, which has its own
exactness tests; this test checks the notebook wiring only."""
import ast
import json
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

from torch.utils.data import Dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "LSB-Dataset-Generator"))
import lsb_torch


class FakePairs:
    def __init__(self):
        self.entries = [({"photo_id": photo},) for photo in ("a", "b", "a", "c", "b")]
        self._last_id = None
        self._last_photo = None
        self.loads = 0

    def __getitem__(self, index):
        photo = self.entries[index][0]["photo_id"]
        if photo != self._last_id:
            self.loads += 1
            self._last_id = photo
            self._last_photo = photo
        return (photo, index, index)


class DiagnosticPreloadTests(unittest.TestCase):
    def test_group_by_photo_but_return_in_subset_order(self):
        notebook = Path(__file__).resolve().parents[1] / "notebooks/04_baseline_kv_xunet_v3.ipynb"
        cell = "".join(json.loads(notebook.read_text())["cells"][29]["source"])
        function = next(node for node in ast.parse(cell).body
                        if isinstance(node, ast.FunctionDef) and node.name == "preload_diagnostic_pairs")
        calls = []

        def fake_preload(subset, workers):
            calls.append(workers)
            return [("pair", index) for index in range(len(subset))]

        namespace = {"PreloadedPairs": lsb_torch.PreloadedPairs, "preload_pairs": fake_preload,
                     "DIAG_WORKERS": 7}
        exec(compile(ast.Module(body=[function], type_ignores=[]), str(notebook), "exec"), namespace)
        subset = type("Subset", (), {"__len__": lambda self: 3})()
        preloaded = namespace["preload_diagnostic_pairs"](subset)
        self.assertIsInstance(preloaded, lsb_torch.PreloadedPairs)
        self.assertEqual(calls, [7])
        self.assertEqual([preloaded[i] for i in range(len(preloaded))],
                         [("pair", 0), ("pair", 1), ("pair", 2)])

    def test_notebook_reads_worker_count_and_imports_helpers(self):
        notebook = json.loads((Path(__file__).resolve().parents[1]
                               / "notebooks/04_baseline_kv_xunet_v3.ipynb").read_text())
        cell = "".join(notebook["cells"][29]["source"])
        self.assertIn('os.environ.get("V3_DIAG_WORKERS", str(EVAL_WORKERS))', cell)
        setup = "".join(notebook["cells"][2]["source"])
        self.assertIn("preload_pairs, PreloadedPairs", setup)

    def test_helper_returns_ordered_pairs_and_releases_last_photo(self):
        base = FakePairs()
        base.frame = None
        with patch.object(lsb_torch, "OrderedPairLoader") as loader:
            loader.return_value = iter([(lsb_torch.torch.zeros(2, 1), lsb_torch.torch.zeros(2, 1),
                                         lsb_torch.torch.tensor([0, 1]))])
            subset = type("Subset", (), {"dataset": base, "__len__": lambda self: 2})()
            pairs = lsb_torch.preload_pairs(subset, 0)
        self.assertEqual([pair[2] for pair in pairs], [0, 1])
        self.assertIsNone(base._last_photo)
        self.assertIsNone(base._last_id)


if __name__ == "__main__":
    unittest.main()
