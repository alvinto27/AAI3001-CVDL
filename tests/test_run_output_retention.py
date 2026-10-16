"""Protect notebook 05's single-checkpoint and single-full-log policy."""
import ast
import io
import json
from pathlib import Path
import tempfile
import threading
import time
from types import SimpleNamespace
import unittest

from scripts.run_notebook import CellStream

ROOT = Path(__file__).resolve().parents[1]


class RunOutputRetentionTests(unittest.TestCase):
    def test_notebook_05_has_one_best_and_last_callback(self):
        notebook = json.loads((ROOT / "notebooks/05_xunet_v4_nonorm.ipynb").read_text())
        code = "\n".join("".join(cell["source"]) for cell in notebook["cells"] if cell["cell_type"] == "code")
        self.assertNotIn("save_top_k=-1", code)
        self.assertIn('monitor="val_pe", mode="min", save_top_k=1, save_last=True', code)
        self.assertIn("assert trainer.checkpoint_callbacks == [checkpoint_callback]", code)
        self.assertIn("best_checkpoint = select_best_checkpoint(checkpoint_callback.best_model_path, checkpoint_dir)", code)
        self.assertIn("last_saved = torch.load(checkpoint_callback.last_model_path", code)
        self.assertIn('assert not list(checkpoint_dir.glob("epoch*.ckpt"))', code)
        self.assertIn("QuietHeadlessProgress()", code)

    def test_last_updates_even_when_validation_does_not_improve(self):
        import lightning as L
        import torch

        notebook = json.loads((ROOT / "notebooks/05_xunet_v4_nonorm.ipynb").read_text())
        code = next("".join(c["source"]) for c in notebook["cells"] if "class TimedModelCheckpoint" in "".join(c.get("source", [])))
        node = next(n for n in ast.parse(code).body if isinstance(n, ast.ClassDef) and n.name == "TimedModelCheckpoint")
        scope = {"L": L, "time": time, "epoch_timing_callback": SimpleNamespace(add=lambda *args: None)}
        exec(compile(ast.Module(body=[node], type_ignores=[]), "notebook05", "exec"), scope)

        class Model(L.LightningModule):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.tensor(0.0))

            def training_step(self, batch, batch_idx):
                return self.weight.square()

            def validation_step(self, batch, batch_idx):
                self.log("val_pe", torch.tensor(0.1 if self.current_epoch == 0 else 0.9))

            def configure_optimizers(self):
                return torch.optim.SGD(self.parameters(), lr=0.1)

        loader = torch.utils.data.DataLoader(torch.zeros(2, 1), batch_size=2)
        with tempfile.TemporaryDirectory() as folder:
            callback = scope["TimedModelCheckpoint"](
                dirpath=folder, filename="best-epoch{epoch:02d}-val_pe{val_pe:.4f}",
                auto_insert_metric_name=False, monitor="val_pe", mode="min",
                save_top_k=1, save_last=True, every_n_epochs=1,
                save_on_train_epoch_end=False,
            )
            L.Trainer(max_epochs=2, accelerator="cpu", devices=1, callbacks=[callback], logger=False,
                      enable_progress_bar=False, enable_model_summary=False,
                      num_sanity_val_steps=0).fit(Model(), loader, loader)
            files = sorted(p.name for p in Path(folder).glob("*.ckpt"))
            self.assertEqual(files, ["best-epoch00-val_pe0.1000.ckpt", "last.ckpt"])
            self.assertEqual(torch.load(Path(folder) / "last.ckpt", weights_only=False)["epoch"], 1)
            self.assertEqual(torch.load(Path(folder) / files[0], weights_only=False)["epoch"], 0)

    def test_compact_stream_keeps_full_text_only_in_run_log(self):
        log, console, outputs = io.StringIO(), io.StringIO(), []
        stream = CellStream("stdout", log, console, outputs, threading.RLock(), compact=True)
        stream.write("many progress redraws\r")
        stream.write("TRAIN progress epoch=1/2 step=15 steps/s=2.00 it/s=2.00")
        stream.write("TRAIN epoch-end epoch=1/2 step=20 steps/s=2.10 it/s=2.10")
        self.assertEqual(outputs, [])
        self.assertIn("many progress redraws", log.getvalue())
        self.assertNotIn("many progress redraws", console.getvalue())
        self.assertEqual(len(console.getvalue().splitlines()), 2)

    def test_notebook_04_stream_is_unchanged(self):
        log, console, outputs = io.StringIO(), io.StringIO(), []
        stream = CellStream("stdout", log, console, outputs, threading.RLock())
        stream.write("ordinary output")
        self.assertEqual(log.getvalue(), console.getvalue())
        self.assertEqual(outputs[0]["text"], ["ordinary output"])


if __name__ == "__main__":
    unittest.main()
