"""SRM front end, front_end setting, must-set settings and the srm notebook."""
import json
from pathlib import Path
import hashlib
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import torch
from torch.nn import functional as F
from cvdl.models import (SRM_ARRAY, SRMFilter, XuNet, XuNetWide, model_from_hparams)
from cvdl.settings import load
from scripts.g4.preflight import top_values
from test_cvdl_settings import profile

ROOT=Path(__file__).resolve().parents[1]
NOTEBOOK=ROOT/'notebooks/experiments/srm.ipynb'
# Published kernels: SRM_Kernels1.npy (shape 5x5x1x30) in
# https://github.com/BioAITeam/Steganalysis (GBRAS-Net, Reinel et al. 2021), the 30-filter set of
# Fridrich & Kodovsky, IEEE TIFS 7(3) 2012, used by Ye-Net (Ye, Ni, Yi, IEEE TIFS 12(11) 2017).
PUBLISHED=ROOT/'tests/fixtures/srm_kernels_published.npy'
PUBLISHED_SHA='3bfe80839bf93d9f8b80f2281af3f4e4c4d3ad2e50caa3612961ae197a93db70'
OPEN=('tile_size','train_tile_size','train_grid','batch_size','tiles_per_epoch')

class SRMKernelTests(unittest.TestCase):
    def test_kernels_match_published_values(self):
        self.assertEqual(hashlib.sha256(PUBLISHED.read_bytes()).hexdigest(),PUBLISHED_SHA)
        published=np.load(PUBLISHED)
        self.assertEqual(published.shape,(5,5,1,30))
        np.testing.assert_allclose(SRM_ARRAY,np.moveaxis(published[:,:,0,:],-1,0),atol=1e-6,rtol=0)

    def test_each_kernel_is_high_pass_and_normalised(self):
        self.assertEqual(SRM_ARRAY.shape,(30,5,5))
        for kernel in SRM_ARRAY:
            self.assertAlmostEqual(float(kernel.sum(dtype=np.float64)),0.0,places=6)  # constant image -> 0
            self.assertAlmostEqual(float(np.abs(kernel).max()),1.0,places=6)
        self.assertEqual(len({k.tobytes() for k in SRM_ARRAY}),30)
        # Classes: 8 first order, 4 second, 8 third, 1 SQUARE3x3, 4 EDGE3x3, 1 SQUARE5x5, 4 EDGE5x5.
        nonzero=[int((k!=0).sum()) for k in SRM_ARRAY]
        self.assertEqual(nonzero[:8],[2]*8);self.assertEqual(nonzero[8:12],[3]*4);self.assertEqual(nonzero[12:20],[4]*8)

    def test_filter_shape_channels_and_frozen(self):
        front=SRMFilter();x=torch.randn(2,3,16,20)
        y=front(x);self.assertEqual(tuple(y.shape),(2,90,16,20))
        self.assertFalse(front.conv.weight.requires_grad)
        for c in (0,2):
            for f in (0,13,29):
                expected=F.conv2d(x[:,c:c+1],torch.tensor(SRM_ARRAY[f])[None,None],padding=2)
                torch.testing.assert_close(y[:,30*c+f:30*c+f+1],expected)
        flat=front(torch.full((1,3,12,12),77.))  # constant image: zero away from the zero-padded border
        self.assertLess(float(flat[:,:,2:-2,2:-2].abs().max()),1e-4)

    def test_wide_model_srm(self):
        torch.manual_seed(0)
        model=XuNetWide(clip_t=3.0,norm='none',width=4,front_end='srm').eval()
        self.assertEqual(tuple(model.conv1.weight.shape),(16,90,5,5))
        out=model(torch.rand(2,3,64,64)*255);self.assertEqual(tuple(out.shape),(2,2))
        self.assertEqual([n for n,p in model.named_parameters() if not p.requires_grad],['kv.conv.weight'])
        probe=model.train();before=probe.kv.conv.weight.clone()
        opt=torch.optim.Adam(probe.parameters(),1e-3)
        F.cross_entropy(probe(torch.rand(4,3,32,32)*255),torch.tensor([0,1,0,1])).backward();opt.step()
        self.assertTrue(torch.equal(before,probe.kv.conv.weight))
        # Layers after the first are the same shapes as the KV model.
        kv=XuNetWide(clip_t=3.0,norm='none',width=4)
        a,b=dict(kv.state_dict()),dict(model.state_dict())
        self.assertEqual(set(a)-set(b),set());
        for key in a:
            if key not in ('conv1.weight','kv.conv.weight'):self.assertEqual(a[key].shape,b[key].shape,key)

    def test_kv_default_unchanged(self):
        torch.manual_seed(3);default=XuNetWide(clip_t=3.0,norm='none',width=4)
        torch.manual_seed(3);explicit=XuNetWide(clip_t=3.0,norm='none',width=4,front_end='kv')
        for (n1,p1),(n2,p2) in zip(default.state_dict().items(),explicit.state_dict().items()):
            self.assertEqual(n1,n2);self.assertTrue(torch.equal(p1,p2))
        self.assertEqual(tuple(default.kv.conv.weight.shape),(3,1,5,5));self.assertEqual(tuple(default.conv1.weight.shape),(16,3,5,5))
        self.assertIn('kv.conv.weight',default.state_dict())
        # An old checkpoint (no front_end hparam) still loads strictly into the default model.
        old=model_from_hparams(dict(model='xunet_wide',norm='none',clip_t=3.0,width=4))
        old.load_state_dict(default.state_dict(),strict=True)
        self.assertEqual(old.front_end,'kv')
        self.assertEqual(model_from_hparams(dict(model='xunet_wide',norm='none',front_end='srm')).front_end,'srm')
        self.assertEqual(tuple(XuNet(front_end='srm').conv1.weight.shape),(16,90,5,5))
        with self.assertRaises(ValueError):XuNet(front_end='bad')

class SettingTests(unittest.TestCase):
    def test_front_end_setting(self):
        self.assertEqual(load(environ={},_profile=profile()).front_end,'kv')
        self.assertEqual(load(environ={},_profile=profile(),front_end='srm').front_end,'srm')
        with self.assertRaises(ValueError):load(environ={},_profile=profile(),front_end='bad')

    def test_must_set(self):
        with self.assertRaisesRegex(ValueError,'must be set.*tile_size, train_tile_size'):
            load(environ={},_profile=profile(),must_set=OPEN)
        with self.assertRaisesRegex(ValueError,'unknown'):load(environ={},_profile=profile(),must_set=('nope',))
        s=load(environ={},_profile=profile(),must_set=('batch_size',),batch_size=32);self.assertEqual(s.batch_size,32)
        s=load(environ={'V3_BATCH_SIZE':'64'},_profile=profile(),must_set=('batch_size',));self.assertEqual(s.batch_size,64)
        s=load(environ={},_profile=profile(),must_set=('batch_size',),overrides=['batch_size=8']);self.assertEqual(s.batch_size,8)
        # A machine-profile default does not count as set.
        self.assertEqual(load(environ={},_profile=profile(),must_set=()).batch_size,load(environ={},_profile=profile()).batch_size)

    def open_notebook(self):
        """Keep a synthetic unset recipe to test must_set, not the ready-to-run notebook."""
        nb=json.loads(NOTEBOOK.read_text());cell=next(c for c in nb['cells'] if c['cell_type']=='code')
        lines=[line for line in cell['source'] if not any(line.lstrip().startswith(key+'=') for key in OPEN)]
        at=next(i for i,line in enumerate(lines) if 'front_end' in line)+1
        lines.insert(at,'    must_set=("tile_size", "train_tile_size", "train_grid", "batch_size", "tiles_per_epoch"),\n')
        cell['source']=lines
        return nb

    def test_notebook_top_cell(self):
        nb=json.loads(NOTEBOOK.read_text());values=top_values(nb)
        self.assertEqual((values['name'],values['front_end'],values['post_fit_eval']),('srm','srm','none'))
        self.assertNotIn('must_set',values)
        self.assertEqual(tuple(values[key] for key in OPEN),(512,256,'overlap_last',128,180000))
        base=top_values(json.loads((ROOT/'notebooks/experiments/tiling-overlap-256.ipynb').read_text()))
        self.assertEqual({k:v for k,v in values.items() if k not in {'name','front_end','tile_size'}},
                         {k:v for k,v in base.items() if k!='name'})
        base_nb=json.loads((ROOT/'notebooks/experiments/tiling-overlap.ipynb').read_text())
        self.assertEqual(nb['cells'][2:],base_nb['cells'][2:])
        resolved=load(environ={},_profile=profile(),**values)
        self.assertEqual(resolved.run_name,'srm');self.assertEqual(resolved.front_end,'srm')

    def test_preflight_and_runner_fail_naming_open_values(self):
        from scripts.g4.preflight import preflight
        from scripts.run_notebook import run
        from types import SimpleNamespace
        opened=self.open_notebook()
        with patch('scripts.g4.preflight.validate_notebook',return_value=opened):
            with self.assertRaisesRegex(ValueError,'must be set.*tile_size.*tiles_per_epoch'):
                preflight('notebooks/experiments/srm.ipynb','HEAD')
        with tempfile.TemporaryDirectory(dir=ROOT/'runs',prefix='.srm-test-') as directory:
            parent=Path(directory)/'experiments';parent.mkdir();path=parent/'srm.ipynb'
            path.write_text(json.dumps(opened))
            env=Path(directory)/'e.env';env.write_text('')
            args=SimpleNamespace(notebook=path,env_file=env,set=[],resume=False,
                                 output=Path(directory)/'output',host_monitor='off')
            with self.assertRaisesRegex(ValueError,'must be set.*train_grid'):run(args)

if __name__=='__main__':unittest.main()
