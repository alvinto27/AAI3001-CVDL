"""The golden comparator must not hide a refactor regression as GPU noise."""
import contextlib
from copy import deepcopy
import io
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from refactor_golden import compare_part


class GoldenComparisonTests(unittest.TestCase):
    def gpu(self):
        return {"folder": {
            "metrics_csv": {"kept_columns": ["val_loss", "val_auc", "val_acc"],
                            "rows": [[0.69, 0.625, 0.5]]},
            "test_predictions.csv": {"prob_stego": [0.5], "identities_sha256": "ids", "sha256": "bytes"},
            "files": ["last.ckpt"]},
            "extra_panels": {"bpp1": {"pairs": 10956, "summary": {
                "kept_columns": ["category", "value", "pairs", "P_E", "AUC"],
                "rows": [["overall", "all", 10956, 0.31, 0.75]]}}}}

    def check(self, part, a, b):
        with contextlib.redirect_stdout(io.StringIO()):
            return compare_part(part, a, b)

    def test_cpu_and_data_are_exact(self):
        for part in ("cpu", "data", "settings"):
            self.assertTrue(self.check(part, {"loss": 0.7}, {"loss": 0.70000001}))

    def test_model_gpu_hash_is_exact(self):
        a = {"checkpoints": {"m": {"class_source": "old", "cuda": {"logits_sha": "before"}}}}
        b = deepcopy(a)
        b["checkpoints"]["m"]["class_source"] = "cvdl.models"
        self.assertFalse(self.check("models", a, b))
        b["checkpoints"]["m"]["cuda"]["logits_sha"] = "changed"
        self.assertTrue(self.check("models", a, b))

    def test_gpu_loss_and_score_absolute_tolerance_only(self):
        a = self.gpu()
        b = deepcopy(a)
        b["folder"]["metrics_csv"]["rows"][0][0] += 0.00005
        b["folder"]["test_predictions.csv"]["prob_stego"][0] += 0.00005
        b["folder"]["test_predictions.csv"]["sha256"] = "different bytes"
        self.assertFalse(self.check("gpu", a, b))
        b["folder"]["test_predictions.csv"]["prob_stego"][0] += 0.001
        self.assertTrue(self.check("gpu", a, b))

    def test_gpu_auc_counts_files_and_identities_remain_exact(self):
        a = self.gpu()
        for kind in ("auc", "ids", "files", "panel_pairs"):
            b = deepcopy(a)
            if kind == "auc":
                b["folder"]["metrics_csv"]["rows"][0][1] += 0.0000001
            elif kind == "ids":
                b["folder"]["test_predictions.csv"]["identities_sha256"] = "changed"
            elif kind == "files":
                b["folder"]["files"].append("unexpected")
            else:
                b["extra_panels"]["bpp1"]["pairs"] -= 1
            with self.subTest(kind=kind):
                self.assertTrue(self.check("gpu", a, b))

    def test_panel_pe_auc_report_only_but_warn(self):
        a = self.gpu()
        b = deepcopy(a)
        b["extra_panels"]["bpp1"]["summary"]["rows"][0][3] += 0.02
        with contextlib.redirect_stdout(io.StringIO()) as log:
            self.assertFalse(compare_part("gpu", a, b))
        self.assertIn("WARNING", log.getvalue())
        self.assertEqual(a, self.gpu())  # comparator does not mutate the golden


if __name__ == "__main__":
    unittest.main()
