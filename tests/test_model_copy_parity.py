"""Shared cvdl models must equal the frozen notebook 04/05 model cells.

eval_checkpoint.py, check_b_tiles.py and speed_matrix.py build models from the
shared package. The frozen notebook cells define the models that were trained. This test runs
the notebook model cell and compares both copies: same state_dict keys, same
logits, same 32 px cell map. Weights are random (seeded). A real checkpoint is
used when present (runs/ is git-ignored, so that part skips on a fresh clone).
"""
import contextlib
import glob
import json
import os
import re
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from cvdl import models as map_demo  # alias keeps this before-reference test stable
except ImportError as error:  # pragma: no cover
    torch = None
    SKIP = str(error)


def notebook_model_namespace(path="notebooks/archive/05_baseline_kv_xunet_v4_pre-cvdl.ipynb"):
    """Run the notebook's KV_ARRAY statement and its model cell, nothing else."""
    cells = json.loads((ROOT / path).read_text())["cells"]
    ns = dict(torch=torch, nn=nn, F=F, np=np, CLIP_T=3.0, NORM="group")
    sources = ["".join(c["source"]) for c in cells if c["cell_type"] == "code"]
    for source in sources:
        match = re.search(r"^KV_ARRAY = np\.array\(.*?\)\s*/\s*12\.0[^\n]*\n", source, re.S | re.M)
        if match:
            exec(match.group(0), ns)
            break
    for source in sources:
        if re.search(r"^class KVFilter", source, re.M):
            exec(source[:source.index("model_for_count")], ns)
            return ns
    raise AssertionError("model cell not found")


@contextlib.contextmanager
def capture_pre_mean():
    """Record the tensor the notebook forward passes to .mean(dim=(-2, -1))."""
    seen, original = [], torch.Tensor.mean

    def mean(self, *args, **kwargs):
        if kwargs.get("dim") == (-2, -1) or (args and args[0] == (-2, -1)):
            seen.append(self)
        return original(self, *args, **kwargs)
    torch.Tensor.mean = mean
    try:
        yield seen
    finally:
        torch.Tensor.mean = original


def notebook_outputs(model, x):
    with capture_pre_mean() as seen, torch.inference_mode():
        logits = model(x)
        cells = F.conv2d(seen[-1], model.classifier.weight[:, :, None, None], model.classifier.bias)
    return logits, cells[:, 1] - cells[:, 0]


def copy_outputs(model, x):
    with torch.inference_mode():
        cells = model.cell_logits(x)
        return model(x), cells[:, 1] - cells[:, 0]


@unittest.skipIf(torch is None, "torch or map_demo import failed")
class ModelCopyParity(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ns = notebook_model_namespace()

    def pair(self, hparams, state=None, namespace=None):
        ns = self.ns if namespace is None else namespace
        kw = dict(clip_t=hparams.get("clip_t", 3.0), norm=hparams.get("norm", "group"))
        if hparams.get("model") == "xunet_wide":
            nb = ns["XuNetWide"](width=hparams.get("width", 4), **kw)
        else:
            nb = ns["XuNet"](**kw)
        md = map_demo.model_from_hparams(hparams)
        if state is None:
            torch.manual_seed(0)
            for module in md.modules():  # non-trivial norm statistics and weights
                if isinstance(module, (nn.BatchNorm2d, nn.GroupNorm)):
                    nn.init.uniform_(module.weight, 0.5, 1.5)
                    nn.init.uniform_(module.bias, -0.2, 0.2)
                if isinstance(module, nn.BatchNorm2d):
                    module.running_mean.uniform_(-0.2, 0.2)
                    module.running_var.uniform_(0.5, 1.5)
            state = md.state_dict()
        for model in (nb, md):
            result = model.load_state_dict(state, strict=True)
            self.assertEqual((result.missing_keys, result.unexpected_keys), ([], []))
            model.eval()
        self.assertEqual(list(nb.state_dict()), list(md.state_dict()))
        return nb, md

    def assert_same(self, nb, md, size=128, x=None):
        device = os.environ.get("CVDL_PARITY_DEVICE", "cpu")
        nb, md = nb.to(device), md.to(device)
        if x is None:
            gen = torch.Generator().manual_seed(1)
            x = torch.randint(0, 256, (2, 3, size, size), generator=gen).float()
        x = x.to(device)
        nb_logits, nb_cells = notebook_outputs(nb, x)
        md_logits, md_cells = copy_outputs(md, x)
        self.assertEqual(tuple(nb_cells.shape), (len(x), size // 32, size // 32))
        self.assertEqual(float((nb_logits - md_logits).abs().max()), 0.0)
        self.assertEqual(float((nb_cells - md_cells).abs().max()), 0.0)

    def test_kv_array_equal(self):
        self.assertTrue(np.array_equal(self.ns["KV_ARRAY"], map_demo.KV_ARRAY))

    def test_random_weights_all_variants(self):
        for model in ("xunet", "xunet_wide"):
            for norm in ("none", "group", "batch"):
                for clip in (3.0, None):
                    with self.subTest(model=model, norm=norm, clip=clip):
                        hp = dict(model=model, norm=norm, clip_t=clip, width=2)
                        self.assert_same(*self.pair(hp))

    def test_frozen_notebook_04(self):
        ns = notebook_model_namespace("notebooks/04_baseline_kv_xunet_v3.ipynb")
        self.assertTrue(np.array_equal(ns["KV_ARRAY"], map_demo.KV_ARRAY))
        for norm in ("group", "batch"):
            with self.subTest(norm=norm):
                self.assert_same(*self.pair(dict(model="xunet", norm=norm, clip_t=3.0), namespace=ns))

    def test_real_checkpoints_if_present(self):
        patterns = ["runs/g4-v4-r2/checkpoints/*/best*.ckpt",
                    "runs/g4-prime-1/checkpoints/*/best*.ckpt",
                    "runs/xunet*wide-w4_mixr2-*/checkpoints/*/best*.ckpt",
                    "runs/xunet*wide-w4_r2-*/checkpoints/*/best*.ckpt"]
        found = [p for pattern in patterns for p in glob.glob(str(ROOT / pattern))]
        if not found:
            self.skipTest("no real checkpoint in runs/")
        for path in found:
            with self.subTest(path=Path(path).name):
                ckpt = torch.load(path, map_location="cpu", weights_only=False)
                state = {k.removeprefix("net."): v for k, v in ckpt["state_dict"].items()}
                ns = (notebook_model_namespace("notebooks/04_baseline_kv_xunet_v3.ipynb")
                      if "/g4-prime-1/" in path else self.ns)
                pair = self.pair(ckpt["hyper_parameters"], state, namespace=ns)
                # Same seeded synthetic input on CPU or an explicitly selected GPU.
                self.assert_same(*pair, size=256)


if __name__ == "__main__":
    unittest.main()
