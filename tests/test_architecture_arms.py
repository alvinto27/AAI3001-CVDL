"""Opt-in architecture arms: defaults and SRM checkpoint shapes stay stable."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import torch
from torch import nn
from torch.nn import functional as F
from cvdl.models import XuNetWide, XuNetWideDelayed, model_from_hparams
from cvdl.lit import SteganalysisModule
from cvdl.settings import load, machine_profile

ROOT = Path(__file__).resolve().parents[1]


class DelayedTests(unittest.TestCase):
    def test_identical_layer_list_parameter_shapes_and_seeded_weights(self):
        torch.manual_seed(3)
        normal = XuNetWide(front_end='srm', norm='none')
        torch.manual_seed(3)
        delayed = XuNetWideDelayed()
        self.assertEqual([(n, type(m)) for n,m in normal.named_modules() if n],
                         [(n, type(m)) for n,m in delayed.named_modules() if n])
        self.assertEqual(list(normal.state_dict()), list(delayed.state_dict()))
        for key,value in normal.state_dict().items():
            self.assertEqual(value.shape, delayed.state_dict()[key].shape)
            torch.testing.assert_close(value, delayed.state_dict()[key], rtol=0, atol=0)
        self.assertFalse(delayed.kv.conv.weight.requires_grad)

    def test_only_first_two_pools_removed_and_last_convs_retained(self):
        model = XuNetWideDelayed().eval()
        shapes = {}; pool_shapes = []
        hooks = []
        for name in ('conv1','conv2','conv3','conv4','conv5','conv6','conv7'):
            hooks.append(getattr(model,name).register_forward_hook(
                lambda module,args,out,name=name: shapes.update({name:tuple(out.shape)})))
        hooks.append(model.pool.register_forward_hook(lambda module,args,out: pool_shapes.append(tuple(out.shape))))
        x = torch.randn(1,3,64,64)
        with torch.no_grad():
            features = model.features(x)
            for h in hooks:h.remove()
            z = model.kv(x).clamp(-3,3)
            z = torch.tanh(model.bn1(model.conv1(z).abs()))
            z = torch.tanh(model.bn2(model.conv2(z)))
            for name in (3,4,5):z = model.pool(F.relu(getattr(model,f'bn{name}')(getattr(model,f'conv{name}')(z))))
            for name in (6,7):z = F.relu(getattr(model,f'bn{name}')(getattr(model,f'conv{name}')(z)))
            torch.testing.assert_close(features,z,rtol=0,atol=0)
            torch.testing.assert_close(model.cell_logits(x).mean((-2,-1)),model(x),rtol=1e-5,atol=1e-6)
        self.assertEqual([shapes[f'conv{i}'][-1] for i in range(1,8)],[64,64,64,32,16,8,8])
        self.assertEqual([s[-1] for s in pool_shapes],[32,16,8])
        self.assertEqual(features.shape,(1,256,8,8))

    def test_model_factory_lightning_checkpoint_and_strict_state_reload(self):
        h = dict(model='xunet_wide_delayed',front_end='srm',norm='none',width=4)
        module = SteganalysisModule(**h)
        self.assertIsInstance(module.net,XuNetWideDelayed)
        restored = model_from_hparams(dict(module.hparams))
        restored.load_state_dict(module.net.state_dict(),strict=True)
        with TemporaryDirectory() as folder:
            path = Path(folder)/'model.ckpt'
            import lightning
            torch.save(dict(state_dict=module.state_dict(),hyper_parameters=dict(module.hparams),
                **{'pytorch-lightning_version':lightning.__version__}),path)
            copy = SteganalysisModule.load_from_checkpoint(path,map_location='cpu',weights_only=False)
            self.assertIsInstance(copy.net,XuNetWideDelayed)
        self.assertIsInstance(model_from_hparams(dict(model='xunet_wide')),XuNetWide)
        with self.assertRaises(ValueError):XuNetWideDelayed(front_end='kv')

    def test_notebook_changes_only_approved_experiment_settings(self):
        values=[]
        for name in ('srm','srm-delayed'):
            n=json.loads((ROOT/f'notebooks/experiments/{name}.ipynb').read_text());ns={}
            with patch('cvdl.settings.automatic_profile',return_value=machine_profile('pc',device='cpu')):
                exec(''.join(n['cells'][1]['source']),ns)
            d=dict(ns['settings'].values)
            for key in ('name','run_name','job_id'):d.pop(key)
            values.append(d)
        diff={k for k in values[0].keys()|values[1].keys() if values[0].get(k)!=values[1].get(k)}
        self.assertEqual(diff,{'model','checkpoint_validation','validation_recipe'})
        self.assertEqual(values[1]['validation_recipe'],'60da4391043e8ce48ff2')
        self.assertEqual(values[1]['lr_drops'],((5,.3),(20,.1)))
        self.assertEqual(values[1]['batch_size'],128)
        self.assertEqual(values[1]['epochs'],30)
        with self.assertRaises(ValueError):
            load(_profile=machine_profile('pc',device='cpu'),environ={},model='xunet_wide_delayed')


if __name__=='__main__':unittest.main()
