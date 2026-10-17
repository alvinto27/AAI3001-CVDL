"""Exact CPU/data proof; GPU QUICK is structural smoke, never a numeric gate."""
import contextlib
from copy import deepcopy
import io
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from refactor_golden import compare_part, repo_relative_paths


class GoldenComparisonTests(unittest.TestCase):
    def gpu(self):
        return {"run_notebook_state": "done", "folder": {
            "metrics_csv": {"kept_columns": ["step", "val_loss", "val_auc"], "rows": [[20, .69, .625]]},
            "test_predictions.csv": {"prob_stego": [.5], "identities_sha256": "ids", "sha256": "bytes"},
            "files": ["last.ckpt"], "last_ckpt": {"state_dict_keys": ["net.weight"]}},
            "extra_panels": {"bpp1": {"pairs": 10956, "summary": {
                "kept_columns": ["category", "value", "pairs", "P_E", "AUC"],
                "rows": [["overall", "all", 10956, .31, .75]]}}}}

    def check(self, part, a, b):
        with contextlib.redirect_stdout(io.StringIO()):
            return compare_part(part, a, b)

    def test_cpu_data_settings_are_exact(self):
        for part in ("cpu", "data", "settings"):
            self.assertTrue(self.check(part, {"loss": .7}, {"loss": .70000001}))

    def test_repo_relocation_only_changes_absolute_root(self):
        old={'paths':['/work/configs/train.yaml'],'value':.7}
        new={'paths':['/relocated/repo/configs/train.yaml'],'value':.7}
        with patch('refactor_golden.ROOT',Path('/relocated/repo')):
            self.assertFalse(self.check('settings',old,new))
            new['paths']=['/relocated/repo/other/train.yaml']
            self.assertTrue(self.check('settings',old,new))  # Same basename, different relative path.
            new['paths']=old['paths']
            self.assertTrue(self.check('settings',old,new))  # Unrebased external path must fail.
        self.assertEqual(old['paths'],['/work/configs/train.yaml'])

    def test_path_rebasing_does_not_hide_non_path_changes(self):
        for other in ('<ROOT>/a','/work-other/a','text /work/a',7,.70000001):
            with self.subTest(other=other),patch('refactor_golden.ROOT',Path('/relocated/repo')):
                self.assertTrue(self.check('settings',{'value':'/work/a'},{'value':other}))
        self.assertEqual(repo_relative_paths({'path':'/outside/a','n':7},'/work'),{'path':'/outside/a','n':7})

    def test_model_cpu_exact_gpu_cross_run_ignored(self):
        a = {"checkpoints": {"m": {"class_source": "old", "cpu": {"logits_sha": "before"}, "cuda": {"logits_sha": "gpu"}}}}
        b = deepcopy(a)
        b["checkpoints"]["m"]["class_source"] = "cvdl.models"
        b["checkpoints"]["m"].pop("cuda")
        self.assertFalse(self.check("models", a, b))
        b["checkpoints"]["m"]["cpu"]["logits_sha"] = "changed"
        self.assertTrue(self.check("models", a, b))

    def test_gpu_numeric_deltas_are_information_only(self):
        a = self.gpu()
        b = deepcopy(a)
        b["folder"]["metrics_csv"]["rows"][0][1:] = [2., .2]
        b["folder"]["test_predictions.csv"]["prob_stego"] = [.9]
        b["folder"]["test_predictions.csv"]["sha256"] = "changed bytes"
        b["extra_panels"]["bpp1"]["summary"]["rows"][0][3:] = [.8, .1]
        with contextlib.redirect_stdout(io.StringIO()) as log:
            self.assertFalse(compare_part("gpu", a, b))
        self.assertIn("delta", log.getvalue())
        self.assertEqual(a, self.gpu())

    def test_gpu_counts_files_ids_keys_stay_exact(self):
        a = self.gpu()
        for kind in ("ids", "files", "keys", "panel_pairs", "steps", "scores_count"):
            b = deepcopy(a)
            if kind == "ids":
                b["folder"]["test_predictions.csv"]["identities_sha256"] = "changed"
            elif kind == "files":
                b["folder"]["files"].append("unexpected")
            elif kind == "keys":
                b["folder"]["last_ckpt"]["state_dict_keys"].append("unexpected")
            elif kind == "steps":
                b["folder"]["metrics_csv"]["rows"][0][0] += 1
            elif kind == "scores_count":
                b["folder"]["test_predictions.csv"]["prob_stego"].append(.5)
            else:
                b["extra_panels"]["bpp1"]["pairs"] -= 1
            with self.subTest(kind=kind):
                self.assertTrue(self.check("gpu", a, b))

    def test_gpu_must_finish(self):
        a = self.gpu()
        b = deepcopy(a)
        b["run_notebook_state"] = "failed"
        self.assertTrue(self.check("gpu", a, b))


if __name__ == "__main__":
    unittest.main()
