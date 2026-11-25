"""ONNX engine: shipped model files, parity with PyTorch, refusals and no torch import."""
import importlib.util
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

import numpy as np
from PIL import Image

from cvdl.pipeline import load_pipeline, BundleError
from cvdl.pipeline.contract import contract_hash, file_hash

ROOT=Path(__file__).resolve().parents[1]
BUNDLES=[ROOT/'deploy'/name for name in ('run13-rule-c-v1','run18-rule-c-v1','run22-rule-c-v1')]
DEFAULT=ROOT/'deploy/run22-rule-c-v1'
HAS_ORT=importlib.util.find_spec('onnxruntime') is not None


def photo(seed=3,shape=(300,520)):
    """Smooth photo-like RGB8 image; the right half gets random LSB flips."""
    rng=np.random.default_rng(seed)
    yy,xx=np.mgrid[0:shape[0],0:shape[1]]
    base=np.stack([(yy*a+xx*b)%256 for a,b in ((.3,.2),(.1,.5),(.7,.05))],-1)
    pixels=np.clip(base+rng.normal(0,3,base.shape),0,255).astype(np.uint8)
    pixels[:,shape[1]//2:]^=(rng.random((shape[0],shape[1]-shape[1]//2,3))<.5).astype(np.uint8)
    stream=io.BytesIO();Image.fromarray(pixels).save(stream,format='PNG');return stream.getvalue()


class ContractCopyTests(unittest.TestCase):
    def test_torch_free_contract_copies_match_cov_pool(self):
        from cvdl import cov_pool
        from cvdl.pipeline import contract
        self.assertEqual(contract.SPEC,cov_pool.SPEC)
        self.assertEqual(contract.GAPCOV,cov_pool.MODEL)

    def test_shipped_onnx_records_bind_each_bundle(self):
        for folder in BUNDLES:
            with self.subTest(folder.name):
                config=json.loads((folder/'deploy.json').read_text())
                record=json.loads((folder/'onnx.json').read_text())
                self.assertEqual(record['onnx_sha256'],file_hash(folder/'model.onnx'))
                self.assertEqual(record['weights_sha256'],config['weights_sha256'])
                self.assertEqual(record['contract_sha256'],contract_hash(config))
                self.assertEqual(record['precision'],'fp32')
                self.assertTrue(all(v<=1e-3 for v in record['export']['check']['max_abs_difference'].values()))


@unittest.skipUnless(HAS_ORT,'onnxruntime is not installed')
class OnnxEngineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data=photo()

    def test_onnx_matches_torch_on_every_shipped_bundle(self):
        for folder in BUNDLES:
            with self.subTest(folder.name):
                reference=load_pipeline(folder,engine='torch').analyze(self.data)
                result=load_pipeline(folder,engine='onnx',threads=1).analyze(self.data)
                self.assertEqual((result['engine'],reference['engine'],result['device']),('onnx','torch','cpu'))
                self.assertEqual(result['bundle'],reference['bundle'])
                self.assertEqual(result['photo_verdict'],reference['photo_verdict'])
                self.assertEqual([t['box'] for t in result['tiles']],[t['box'] for t in reference['tiles']])
                np.testing.assert_allclose([t['probability'] for t in result['tiles']],
                    [t['probability'] for t in reference['tiles']],atol=1e-4)
                np.testing.assert_allclose(result['heatmap']['values'],reference['heatmap']['values'],atol=1e-3)
                self.assertAlmostEqual(result['photo_rule']['logit'],reference['photo_rule']['logit'],places=3)
                self.assertEqual(result['counts'],reference['counts'])

    def test_refuses_mismatched_or_changed_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            for case in ('tampered','foreign_record','int8','missing','cuda'):
                with self.subTest(case):
                    folder=Path(tmp)/case;shutil.copytree(DEFAULT,folder)
                    device='cpu'
                    if case=='tampered':
                        data=bytearray((folder/'model.onnx').read_bytes());data[-1]^=1
                        (folder/'model.onnx').write_bytes(bytes(data))
                    elif case=='foreign_record':
                        shutil.copy(ROOT/'deploy/run18-rule-c-v1/onnx.json',folder/'onnx.json')
                    elif case=='int8':
                        record=json.loads((folder/'onnx.json').read_text());record['precision']='int8'
                        (folder/'onnx.json').write_text(json.dumps(record))
                    elif case=='missing':
                        (folder/'model.onnx').unlink()
                    else:
                        device='cuda'
                    with self.assertRaises(BundleError):load_pipeline(folder,engine='onnx',device=device)

    def test_onnx_path_never_imports_torch(self):
        code=('import sys;from cvdl.pipeline import load_pipeline;'
              'from tests.test_pipeline_onnx import photo;'
              'r=load_pipeline(engine="onnx",threads=1).analyze(photo(),include_images=True);'
              'assert r["engine"]=="onnx" and "images" in r;'
              'assert "torch" not in sys.modules,"torch imported";print("ok")')
        env={k:v for k,v in os.environ.items() if k!='CVDL_BUNDLE'}
        done=subprocess.run([sys.executable,'-c',code],cwd=ROOT,env=env,capture_output=True,text=True,timeout=120)
        self.assertEqual(done.returncode,0,done.stderr[-2000:])
        self.assertEqual(done.stdout.strip(),'ok')

    def test_rejects_unknown_engine_and_threads(self):
        with self.assertRaises(ValueError):load_pipeline(engine='tensorrt')
        with self.assertRaises(BundleError):load_pipeline(engine='onnx',threads=0)


if __name__=='__main__':
    unittest.main()
