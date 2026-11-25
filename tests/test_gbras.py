"""Arm G against the pinned authors' layer graph, not an approximate GBRAS."""
import ast
import copy
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import lightning as L
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

from cvdl.gbras import GBRASBody, GBRASBatchNorm, GBRASNet
from cvdl.models import model_from_hparams, SRMFilter
from cvdl.lit import SteganalysisModule

ROOT=Path(__file__).resolve().parents[1]
HP=dict(model='gbras',front_end='srm',norm='batch',clip_t=3.)


class GBRASArchitectureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(1)

    def test_author_layers_shapes_skips_and_counts(self):
        # Independent transcription of saved model.summary at 0b40ab0,
        # excluding its grayscale SRM (780 frozen values).
        rows=[('bn1',30,256,90),('dw2',30,256,60),('sep2',30,256,3540),('bn2',30,256,90),
            ('dw3',30,256,60),('sep3',30,256,3540),('bn3',30,256,90),
            ('conv4',30,256,8130),('bn4',30,256,90),('conv5',30,256,8130),('bn5',30,256,90),
            ('pool6',30,128,0),('conv7',60,128,16260),('bn7',60,128,180),
            ('dw8',60,128,120),('sep8',60,128,12480),('bn8',60,128,180),
            ('dw9',60,128,120),('sep9',60,128,12480),('bn9',60,128,180),
            ('conv10',60,128,32460),('bn10',60,128,180),('pool11',60,64,0),
            ('conv12',60,64,32460),('bn12',60,64,180),('pool13',60,32,0),
            ('conv14',60,32,32460),('bn14',60,32,180),('pool15',60,16,0),
            ('conv16',30,16,1830),('bn16',30,16,90),('conv17',2,16,62),('bn17',2,16,6)]
        body=GBRASBody().eval();events=[];outputs={};inputs={};hooks=[]
        for name,c,size,count in rows:
            layer=getattr(body,name)
            self.assertEqual(sum(t.numel() for t in layer.state_dict().values() if t.is_floating_point()),count)
            def capture(layer,args,result,name=name):
                events.append(name);inputs[name]=args[0].detach();outputs[name]=result.detach()
            hooks.append(layer.register_forward_hook(capture))
        with torch.inference_mode():out=body(torch.randn(1,30,256,256))
        for hook in hooks:hook.remove()
        self.assertEqual(events,[r[0] for r in rows])
        for name,c,size,_ in rows:self.assertEqual(tuple(outputs[name].shape),(1,c,size,size))
        torch.testing.assert_close(inputs['conv4'],outputs['bn1']+outputs['bn3'],rtol=0,atol=0)
        torch.testing.assert_close(inputs['conv10'],outputs['bn7']+outputs['bn9'],rtol=0,atol=0)
        for num in (2,3,8,9):
            dw=getattr(body,f'dw{num}');sep=getattr(body,f'sep{num}');c=dw.in_channels
            self.assertEqual((dw.kernel_size,dw.stride,dw.groups,dw.out_channels),((1,1),(1,1),c,c))
            self.assertIsNotNone(dw.bias)
            self.assertEqual((sep.depthwise.kernel_size,sep.depthwise.groups,sep.depthwise.out_channels),((3,3),c,3*c))
            self.assertIsNone(sep.depthwise.bias)
            self.assertEqual(sep.pointwise.kernel_size,(1,1));self.assertIsNotNone(sep.pointwise.bias)
            torch.testing.assert_close(inputs[f'bn{num}'],F.elu(outputs[f'sep{num}']),rtol=0,atol=0)
        for num,ni,no,k in ((4,30,30,3),(5,30,30,3),(7,30,60,3),(10,60,60,3),
                           (12,60,60,3),(14,60,60,3),(16,60,30,1),(17,30,2,1)):
            conv=getattr(body,f'conv{num}')
            self.assertEqual((conv.in_channels,conv.out_channels,conv.kernel_size,conv.stride,conv.padding),
                             (ni,no,(k,k),(1,1),(k//2,k//2)))
            torch.testing.assert_close(inputs[f'bn{num}'],F.elu(outputs[f'conv{num}']),rtol=0,atol=0)
        for num in (6,11,13,15):
            pool=getattr(body,f'pool{num}');self.assertEqual((pool.kernel_size,pool.stride,pool.padding),(2,2,0))
        norms=[m for m in body.modules() if isinstance(m,GBRASBatchNorm)]
        self.assertEqual(len(norms),13)
        for bn in norms:
            self.assertEqual((bn.eps,bn.momentum),(.001,.8))
            self.assertTrue(torch.equal(bn.weight,torch.ones_like(bn.weight)))
            self.assertFalse(bn.weight.requires_grad);self.assertNotIn('weight',bn.state_dict())
            self.assertNotIn('weight',dict(bn.named_parameters()))
            self.assertTrue(bn.bias.requires_grad)
        self.assertEqual(tuple(out.shape),(1,2,16,16))
        self.assertEqual(sum(p.numel() for p in body.parameters()),164734)
        self.assertEqual(sum(t.numel() for t in body.state_dict().values() if t.is_floating_point()),165818)
        self.assertEqual(165818+780,166598)
        net=GBRASNet()
        self.assertEqual(sum(p.numel() for p in net.parameters()),169714)
        self.assertEqual(sum(p.numel() for p in net.parameters() if p.requires_grad),167464)

    def test_beta_only_bn_and_keras_initialization(self):
        bn=GBRASBatchNorm(2);x=torch.arange(24.).reshape(2,2,2,3)
        with torch.no_grad():bn.bias.copy_(torch.tensor([.5,-.5]))
        y=bn(x);mean=x.mean((0,2,3));var=x.var((0,2,3),correction=0)
        torch.testing.assert_close(y,(x-mean[None,:,None,None])/torch.sqrt(var[None,:,None,None]+.001)+bn.bias[None,:,None,None])
        torch.testing.assert_close(bn.running_mean,.8*mean)
        torch.testing.assert_close(bn.running_var,.2+.8*x.var((0,2,3),correction=1))
        body=GBRASBody()
        for name,layer in body.named_modules():
            if isinstance(layer,nn.Conv2d):
                ni=layer.in_channels;k=layer.kernel_size[0]
                fan=(k*k*(ni+layer.out_channels//ni) if layer.groups==ni
                     else k*k*(ni+layer.out_channels))
                self.assertLessEqual(float(layer.weight.detach().abs().max()),(6/fan)**.5)
                if layer.bias is not None:self.assertTrue(torch.equal(layer.bias,torch.zeros_like(layer.bias)))

    def test_cpu_fixed_gamma_matches_previous_none_weight_outputs_and_gradients(self):
        for training in (True,False):
            bn=GBRASBatchNorm(3).train(training)
            with torch.no_grad():
                bn.running_mean.copy_(torch.tensor([.1,.2,.3]));bn.running_var.fill_(2)
                bn.bias.copy_(torch.tensor([.5,-.5,.2]))
            mean,var=bn.running_mean.clone(),bn.running_var.clone()
            bias=bn.bias.detach().clone().requires_grad_()
            x=torch.randn(2,3,4,5,requires_grad=True);reference=x.detach().clone().requires_grad_()
            actual=bn(x)
            previous=F.batch_norm(reference,mean,var,None,bias,training,.8,.001)
            torch.testing.assert_close(actual,previous,rtol=1e-5,atol=1e-6)
            gradient=torch.randn_like(actual)
            actual.backward(gradient);previous.backward(gradient)
            torch.testing.assert_close(x.grad,reference.grad,rtol=1e-5,atol=1e-6)
            torch.testing.assert_close(bn.bias.grad,bias.grad,rtol=1e-5,atol=1e-6)
            torch.testing.assert_close(bn.running_mean,mean,rtol=0,atol=0)
            torch.testing.assert_close(bn.running_var,var,rtol=0,atol=0)

    @unittest.skipUnless(torch.cuda.is_available(),'small explicit GPU backward check')
    def test_gpu_fixed_gamma_full_body_forward_backward(self):
        net=GBRASNet().cuda().train()
        x=torch.randint(0,256,(2,3,256,256),device='cuda').float()
        loss=F.cross_entropy(net(x),torch.tensor([0,1],device='cuda'))
        loss.backward();torch.cuda.synchronize()
        self.assertTrue(torch.isfinite(loss))
        for name,p in net.named_parameters():
            if p.requires_grad:self.assertTrue(p.grad is not None and torch.isfinite(p.grad).all(),name)
        for layer in net.modules():
            if isinstance(layer,GBRASBatchNorm):
                self.assertEqual(layer.weight.device.type,'cuda')
                self.assertFalse(layer.weight.requires_grad)
                self.assertNotIn('weight',layer.state_dict())

    def test_256_forward_backward_frozen_front_and_unmodified_clamp(self):
        torch.manual_seed(20261021);net=GBRASNet().train();x=torch.randint(0,256,(2,3,256,256)).float()
        seen=[];hook=net.adapter.register_forward_pre_hook(lambda m,args:seen.append(args[0].detach()))
        before=net.kv.conv.weight.detach().clone();opt=torch.optim.AdamW(net.parameters(),lr=3e-4)
        logits=net(x);hook.remove();self.assertEqual(tuple(logits.shape),(2,2))
        torch.testing.assert_close(seen[0],net.kv(x).clamp(-3,3),rtol=0,atol=0)
        loss=F.cross_entropy(logits,torch.tensor([0,1]));loss.backward()
        self.assertGreater(float(net.adapter.weight.grad.abs().sum()),0)
        for name,p in net.named_parameters():
            if p.requires_grad:self.assertTrue(p.grad is not None and torch.isfinite(p.grad).all(),name)
        self.assertIsNone(net.kv.conv.weight.grad);opt.step()
        self.assertTrue(torch.equal(before,net.kv.conv.weight));self.assertTrue(torch.equal(before,SRMFilter().conv.weight))

    def test_cpu_compile_aot_eager_and_clean_checkpoint_keys(self):
        eager=GBRASNet().train();compiled=copy.deepcopy(eager)
        compiled.compile(backend='aot_eager',dynamic=False,fullgraph=False)
        x=torch.randint(0,256,(2,3,256,256)).float();target=torch.tensor([0,1])
        a=eager(x);b=compiled(x);torch.testing.assert_close(b,a,rtol=1e-5,atol=1e-6)
        F.cross_entropy(a,target).backward();F.cross_entropy(b,target).backward()
        for (name,p),(_,q) in zip(eager.named_parameters(),compiled.named_parameters()):
            if p.requires_grad:torch.testing.assert_close(q.grad,p.grad,rtol=1e-4,atol=1e-5,msg=name)
        self.assertEqual(set(eager.state_dict()),set(compiled.state_dict()))
        self.assertFalse(any('_orig_mod' in k for k in compiled.state_dict()))

    def test_class_score_cells_preserve_logits_without_extra_body_pool(self):
        from cvdl.cell_diagnostics import cell_maps
        net=GBRASNet().eval();x=torch.randn(1,3,256,256)
        with torch.inference_mode():
            native=net.features(x);cells=net.cell_logits(x);logits=net(x)
            torch.testing.assert_close(cells,F.avg_pool2d(native,2,2),rtol=0,atol=0)
            torch.testing.assert_close(cells.mean((-2,-1)),logits,rtol=1e-5,atol=1e-6)
            margin,aux,p=cell_maps(net,x)
            torch.testing.assert_close(margin,cells[:,1]-cells[:,0],rtol=0,atol=0)
            torch.testing.assert_close(p,logits.softmax(1)[:,1],rtol=0,atol=0)
        self.assertEqual(tuple(cells.shape),(1,2,8,8));self.assertIsNone(aux)
        self.assertEqual(sum(isinstance(m,nn.AvgPool2d) for m in net.body.modules()),4)
        net.train()
        with self.assertRaisesRegex(ValueError,'eval mode'):net.cell_logits(x)

    def test_diagnostic_export_is_labelled_class_score_not_cam(self):
        from scripts.cell_eval import score
        dataset=TensorDataset(torch.zeros(1,2,256,256,3,dtype=torch.uint8),
            torch.zeros(1,8,8,dtype=torch.int64),torch.ones(1,8,8,dtype=torch.bool),torch.arange(1))
        dataset.corpus=None
        with patch('scripts.cell_eval.CellPairs',return_value=dataset):
            arrays,_=score(GBRASNet().eval(),ROOT,[{}],device='cpu',workers=0,batch_pairs=1)
        self.assertEqual(set(arrays),{'counts','valid_negative','class_score'})
        self.assertEqual(arrays['class_score'].shape,(1,2,8,8))

    def test_checkpoint_evaluator_validation_and_deployment_eval_batch_invariance(self):
        from cvdl.standardtest import load_network
        from cvdl.hardtest import score  # shared scorer for standard256 and hard256
        clean=torch.stack((torch.randint(0,256,(3,256,256)).float(),torch.zeros(3,256,256),torch.full((3,256,256),255.)))
        stego=clean.clone();stego[:,0,0,0]+=1
        module=SteganalysisModule(**HP)
        # Nontrivial training statistics, not the all-zero initial running means.
        module.net.train();module.net(clean[:2])
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'model.ckpt'
            torch.save(dict(state_dict=module.state_dict(),hyper_parameters=dict(module.hparams),
                            **{'pytorch-lightning_version':L.__version__}),path)
            restored=SteganalysisModule.load_from_checkpoint(path,map_location='cpu',weights_only=False)
            network=load_network(path,'cpu')
            self.assertIsInstance(network,GBRASNet);self.assertFalse(network.training)
            for name,value in module.state_dict().items():torch.testing.assert_close(restored.state_dict()[name],value,rtol=0,atol=0)
            # Deployment-style checkpoint load: eval+inference, independent of companions.
            with torch.inference_mode():
                alone=network(clean[:1]).softmax(1)[0,1]
                mixed=network(clean).softmax(1)[0,1]
            torch.testing.assert_close(mixed,alone,rtol=1e-5,atol=1e-6)
            before={n:t.clone() for n,t in network.named_buffers()};modes=[]
            hook=network.register_forward_hook(lambda m,args,out:modes.append((m.training,torch.is_grad_enabled())))
            fixture=SimpleNamespace(frame=pd.DataFrame({'source_photo':['a','b','c']}))
            tasks=[(torch.arange(3),clean,stego)]
            # Both suites call this scorer, which must reapply eval even if a caller forgot.
            network.train()
            with patch('lsb_torch.PhotoPairTasks',return_value=tasks):
                scored=score(network,fixture,device='cpu',batch_size=3,workers=0)
            hook.remove();self.assertEqual(modes,[(False,False)])
            np.testing.assert_allclose(scored.query('label==0').prob_stego.iloc[0],alone.item(),rtol=1e-5,atol=1e-6)
            for n,t in network.named_buffers():torch.testing.assert_close(t,before[n],rtol=0,atol=0)
            # Real Lightning validation, with train-matched per-source metric wiring.
            sources=('raise','alaska','lsdir')
            restored.validation_data=SimpleNamespace(checkpoint_validation='train_matched',
                val=SimpleNamespace(frame=pd.DataFrame({'source':sources})),
                corpus=SimpleNamespace(settings=SimpleNamespace(sources=sources)))
            modes=[];outputs=[]
            def capture(m,args,out):
                modes.append((m.training,torch.is_grad_enabled(),all(not b.training for b in m.modules() if isinstance(b,nn.BatchNorm2d))))
                outputs.append(out.detach())
            hook=restored.net.register_forward_hook(capture)
            before={n:t.clone() for n,t in restored.net.named_buffers()}
            loader=DataLoader(TensorDataset(clean,stego,torch.arange(3)),batch_size=3)
            trainer=L.Trainer(accelerator='cpu',devices=1,logger=False,enable_checkpointing=False,
                enable_progress_bar=False,enable_model_summary=False,num_sanity_val_steps=0)
            result=trainer.validate(restored,dataloaders=loader,verbose=False)
            hook.remove();self.assertEqual(modes,[(False,False,True)])
            for n,t in restored.net.named_buffers():torch.testing.assert_close(t,before[n],rtol=0,atol=0)
            self.assertTrue(all(f'val_pe_{s}' in result[0] for s in sources))
            torch.testing.assert_close(outputs[0].softmax(1)[0,1],alone,rtol=1e-5,atol=1e-6)

    def test_notebook_recipe_and_settings_opt_in(self):
        from cvdl.settings import load
        from cvdl import runtime
        notebook=json.loads((ROOT/'notebooks/experiments/srm-gbras.ipynb').read_text())
        code=''.join(notebook['cells'][1]['source']);tree=ast.parse(code)
        call=next(n for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='load')
        values={kw.arg:ast.literal_eval(kw.value) for kw in call.keywords}
        self.assertEqual(values['name'],'srm-gbras');self.assertEqual(values['model'],'gbras')
        for key,value in dict(front_end='srm',norm='batch',clip_t=3.,lr=3e-4,warmup_epochs=1,
                epochs=30,batch_size=128,tiles_per_epoch=180000,train_tile_size=256,train_grid='overlap_last',
                train_yaml='configs/train-k1to4.yaml',data='otf',paired_batches=False,
                checkpoint_validation='train_matched',validation_recipe='60da4391043e8ce48ff2',
                evaluation_checkpoint='best',post_fit_eval='none').items():self.assertEqual(values[key],value)
        self.assertEqual(values['source_mix'],{'raise':.4,'alaska':.4,'lsdir':.2})
        # Compile remains machine-owned: G4 defaults to it; PC uses an override.
        settings=load(environ={},overrides=['compile=default'],**values)
        self.assertEqual(settings.compile,'default')
        self.assertEqual(settings.model,'gbras');self.assertEqual(settings.batch_size,128)
        probe=SimpleNamespace(model='gbras',rocm_conv1_fix=True,rocm=True,device='cuda')
        self.assertIsNone(runtime.conv1_factory(probe))
        for bad in (dict(norm='none'),dict(front_end='kv'),dict(clip_t=None)):
            with self.assertRaisesRegex(ValueError,'GBRAS'):model_from_hparams(HP|bad)
        for c in notebook['cells']:
            if c['cell_type']=='code':self.assertIsNone(c['execution_count']);self.assertEqual(c['outputs'],[])


if __name__=='__main__':unittest.main()
