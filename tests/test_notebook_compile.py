"""Notebook 04 optional compilation and loader-cleanup placement checks."""
import ast
import json
import os
import unittest
from pathlib import Path
from unittest.mock import patch


NOTEBOOK=Path(__file__).resolve().parents[1]/"notebooks/04_baseline_kv_xunet_v3.ipynb"


class CompileSwitchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cells=json.loads(NOTEBOOK.read_text())["cells"]
        cls.setup="".join(cells[2]["source"])
        cls.training="".join(cells[19]["source"])
        for cell in cells:
            if cell["cell_type"]=="code":
                ast.parse("".join(cell["source"]))

    def test_switch_accepts_only_off_and_default(self):
        block=self.setup[self.setup.index("COMPILE_MODE = os.environ"):self.setup.index("MAIN_THREADS =")]
        for raw,expected in [(None,"off"),("off","off"),("default","default"),("0","off")]:
            with self.subTest(raw=raw), patch.dict(os.environ,{"V3_COMPILE":raw} if raw is not None else {},clear=True):
                scope={"os":os};exec(block,scope)
                self.assertEqual(scope["COMPILE_MODE"],expected)
        with patch.dict(os.environ,{"V3_COMPILE":"reduce-overhead"},clear=True):
            with self.assertRaisesRegex(ValueError,"off or default"):
                exec(block,{"os":os})

    def test_compile_guard_refuses_backend_override_and_in_repo_caches(self):
        guard=self.setup[self.setup.index('if COMPILE_MODE == "default":\n    if "MIOPEN_GEMM_ENFORCE_BACKEND"'):self.setup.index("DATA_ROOT =")]
        root=NOTEBOOK.resolve().parents[1]
        base={"COMPILE_MODE":"default","ROOT":root,"Path":Path,"os":os}
        with patch.dict(os.environ,{"MIOPEN_GEMM_ENFORCE_BACKEND":"5"},clear=True):
            with self.assertRaisesRegex(RuntimeError,"Unset MIOPEN"):
                exec(guard,base.copy())
        for name,value in [("TORCHINDUCTOR_CACHE_DIR","relative-cache"),
                           ("TRITON_CACHE_DIR",str(root / "Datasets/generated/otf-cache/v1"))]:
            with self.subTest(name=name),patch.dict(os.environ,{name:value},clear=True):
                with self.assertRaisesRegex(ValueError,"absolute path outside"):
                    exec(guard,base.copy())
        with patch.dict(os.environ,{"TRITON_CACHE_DIR":"/tmp/triton-test"},clear=True):
            exec(guard,base.copy())

    def test_in_place_compile_after_construction_before_fit(self):
        constructed=self.training.index("lightning_model = SteganalysisModule(")
        compiled=self.training.index('lightning_model.net.compile(mode="default", dynamic=False, fullgraph=False)')
        fitted=self.training.index("trainer.fit(lightning_model, train_loader, val_loader")
        shutdown=self.training.index("train_loader.shutdown()")
        self.assertLess(constructed,compiled)
        self.assertLess(compiled,fitted)
        self.assertLess(fitted,shutdown)
        self.assertNotIn("lightning_model.net = torch.compile(",self.training)
        self.assertIn('train_loader.loader._iterator = None',self.training)

    def test_guards_and_native_tf32_flags(self):
        self.assertIn('"MIOPEN_GEMM_ENFORCE_BACKEND" in os.environ',self.setup)
        for name in ("TORCHINDUCTOR_CACHE_DIR","TRITON_CACHE_DIR"):
            self.assertIn(name,self.setup)
        self.assertIn('"tf32_matmul": torch.backends.cuda.matmul.allow_tf32',self.training)
        self.assertIn('"tf32_cudnn": torch.backends.cudnn.allow_tf32',self.training)
        self.assertNotIn("allow_tf32 =",self.setup+self.training)
        self.assertNotIn("V3_TF32",self.setup+self.training)


if __name__=="__main__":
    unittest.main()
