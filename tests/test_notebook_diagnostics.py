"""Notebook 04 OTF diagnostic preload keeps pair order and avoids repeated decodes."""
import ast
import json
import unittest
from pathlib import Path

from torch.utils.data import Dataset


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
        namespace = {"Dataset": Dataset}
        exec(compile(ast.Module(body=[function], type_ignores=[]), str(notebook), "exec"), namespace)
        base = FakePairs()
        subset = type("Subset", (), {"dataset": base, "indices": [1, 0, 4, 2, 3],
                                      "__len__": lambda self: len(self.indices)})()
        preloaded = namespace["preload_diagnostic_pairs"](subset)
        self.assertEqual([preloaded[i] for i in range(len(preloaded))],
                         [("b", 1, 0), ("a", 0, 1), ("b", 4, 2), ("a", 2, 3), ("c", 3, 4)])
        self.assertEqual(base.loads, 3)
        self.assertIsNone(base._last_photo)
        self.assertIsNone(base._last_id)


if __name__ == "__main__":
    unittest.main()
