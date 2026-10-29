"""GAP is retained; covariance math, honest maps and saved identity are opt-in."""
import copy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
import torch
from torch.nn import functional as F
from cvdl.cov_pool import descriptor,regularized_covariance,sqrt_newton_schulz,SPEC,MODEL
from cvdl.models import XuNetWide,XuNetWideGAPCov32,model_from_hparams
from cvdl.settings import load,machine_profile,plain
from scripts.g4.preflight import top_values
ROOT=Path(__file__).resolve().parents[1]
NOTEBOOK=ROOT/'notebooks/experiments/srm-e50-paired-gapcov32.ipynb'
H=dict(model=MODEL,front_end='srm',norm='none',width=4,clip_t=3.)


class MathTests(unittest.TestCase):
    def test_independent_spectral_polynomial_and_true_root_error(self):
        torch.manual_seed(1);h=torch.randn(2,4,13,dtype=torch.float64)
        c=regularized_covariance(h);values,vectors=torch.linalg.eigh(c)
        norm=values.sum(-1,keepdim=True);q=(values/norm).sqrt()
        for _ in range(5):q=q*(1.5-.5*q.square())
        expected=(vectors*(values.sqrt()*q)[:,None,:])@vectors.transpose(1,2)
        self.assertTrue(torch.allclose(sqrt_newton_schulz(c),expected,atol=1e-12,rtol=1e-12))
        s=sqrt_newton_schulz(torch.eye(32,dtype=torch.float64)[None])
        self.assertAlmostEqual(1-float(s[0,0,0]),.0925068,places=6)
        self.assertFalse(torch.allclose(s,torch.eye(32,dtype=torch.float64)[None],rtol=1e-4,atol=0))

    def test_gradcheck_constant_rank_batch_independence_and_replication(self):
        torch.manual_seed(2);h=torch.randn(2,3,7,dtype=torch.float64,requires_grad=True)
        self.assertTrue(torch.autograd.gradcheck(descriptor,(h,),eps=1e-6,atol=1e-5,rtol=1e-3))
        a=descriptor(h);repeat=descriptor(torch.cat((h,h,h,h),2))
        self.assertTrue(torch.allclose(a,repeat,atol=1e-12,rtol=1e-12))
        self.assertTrue(torch.allclose(a[0],descriptor(h[:1])[0],atol=1e-12,rtol=1e-12))
        self.assertTrue(torch.allclose(a,descriptor(h+3),atol=1e-12,rtol=1e-12))
        for v in (torch.zeros(2,32,64),torch.ones(2,32,256),torch.randn(2,1,64).expand(2,32,64)):
            v=v.clone().requires_grad_();d=descriptor(v)
            self.assertEqual(d.shape,(2,528));self.assertTrue(torch.isfinite(d).all())
            d.sum().backward();self.assertTrue(torch.isfinite(v.grad).all())
        c=torch.diag(torch.tensor([1.,2.,3.]))[None]
        s=sqrt_newton_schulz(c);indices=torch.triu_indices(3,3)
        self.assertTrue(torch.equal(s[:,indices[0],indices[1]],s.flatten(1)[:,[0,1,2,4,5,8]]))


class ModelTests(unittest.TestCase):
    def test_seeded_body_frozen_bank_and_head_shapes(self):
        torch.manual_seed(9);base=XuNetWide(front_end='srm',norm='none');rng=torch.get_rng_state()
        torch.manual_seed(9);net=XuNetWideGAPCov32()
        self.assertTrue(torch.equal(rng,torch.get_rng_state()))
        for name,v in base.state_dict().items():
            if not name.startswith('classifier.'):self.assertTrue(torch.equal(v,net.state_dict()[name]),name)
        self.assertEqual(sum(p.numel() for p in net.parameters()),640604)
        self.assertEqual(sum(p.numel() for p in net.parameters() if p.requires_grad),638354)
        self.assertEqual((net.classifier.in_features,net.cov_reduce.out_channels,net.cov_reduce.bias),(784,32,None))
        with torch.no_grad():
            for size in (256,512):
                features=net.features(torch.zeros(1,3,size,size));gap,cov=net.pooled_parts(features)
                self.assertEqual(features.shape,(1,256,size//32,size//32))
                self.assertTrue(torch.equal(gap,features.mean((-2,-1))))
                self.assertEqual(cov.shape,(1,528))
        for change in (dict(front_end='kv'),dict(width=3),dict(norm='batch'),dict(clip_t=2)):
            with self.assertRaises(ValueError):model_from_hparams(dict(H,cov_pool_identity=SPEC,**change))
            with self.assertRaises(ValueError):load(environ={},_profile=machine_profile('pc',device='cpu'),**dict(H,**change))
        with self.assertRaisesRegex(ValueError,'attribution'):net.cell_logits(torch.zeros(1,3,64,64))

    def test_updates_amp_and_checkpoint_resume(self):
        from cvdl.lit import SteganalysisModule
        from cvdl.standardtest import load_network
        import lightning
        torch.manual_seed(3);m=SteganalysisModule(**H,paired_batches=True,max_epochs=50,steps_per_epoch=3,val_batch_images=64,batch_size=128)
        m.log=lambda *a,**k:None;c=m.configure_optimizers();opt=c['optimizer'];sched=c['lr_scheduler']['scheduler']
        initial=m.net.kv.conv.weight.clone();batch=(torch.rand(2,3,64,64),torch.rand(2,3,64,64),torch.arange(2))
        def step(n,o,s):
            o.zero_grad(set_to_none=True);r=n.training_step(batch,0);self.assertEqual(r['count'],4)
            r['loss'].backward()
            for p in (n.net.conv1.weight,n.net.conv7.weight,n.net.cov_reduce.weight,n.net.classifier.weight):
                self.assertIsNotNone(p.grad);self.assertTrue(torch.isfinite(p.grad).all());self.assertGreater(p.grad.abs().max(),0)
            self.assertIsNone(n.net.kv.conv.weight.grad);o.step();s.step()
        step(m,opt,sched)
        self.assertTrue(torch.equal(initial,m.net.kv.conv.weight));self.assertEqual(m.hparams['cov_pool_identity'],SPEC)
        with torch.autocast('cpu',dtype=torch.bfloat16):
            gap,cov=m.net.pooled_parts(torch.randn(2,256,8,8))
            self.assertEqual((gap.dtype,cov.dtype),(torch.float32,torch.float32))
        saved=dict(state_dict=copy.deepcopy(m.state_dict()),hyper_parameters=dict(m.hparams),global_step=1,
            optimizer_states=[copy.deepcopy(opt.state_dict())],lr_schedulers=[copy.deepcopy(sched.state_dict())],
            **{'pytorch-lightning_version':lightning.__version__})
        with TemporaryDirectory() as tmp:
            path=Path(tmp)/'best-test.ckpt';torch.save(saved,path);net=load_network(path,'cpu')
            self.assertIsInstance(net,XuNetWideGAPCov32);self.assertFalse(net.kv.conv.weight.requires_grad)
            for identity in (None,dict(SPEC,iterations=6)):
                wrong=copy.deepcopy(saved)
                if identity is None:wrong['hyper_parameters'].pop('cov_pool_identity')
                else:wrong['hyper_parameters']['cov_pool_identity']=identity
                bad=Path(tmp)/'bad.ckpt';torch.save(wrong,bad)
                with self.assertRaisesRegex(ValueError,'head identity'):load_network(bad,'cpu')
                with self.assertRaisesRegex(ValueError,'head identity'):SteganalysisModule.load_from_checkpoint(bad,map_location='cpu',weights_only=False)
            other=SteganalysisModule.load_from_checkpoint(path,map_location='cpu',weights_only=False);other.log=lambda *a,**k:None
            oc=other.configure_optimizers();oo=oc['optimizer'];ss=oc['lr_scheduler']['scheduler']
            oo.load_state_dict(copy.deepcopy(saved['optimizer_states'][0]));ss.load_state_dict(copy.deepcopy(saved['lr_schedulers'][0]))
            step(m,opt,sched);step(other,oo,ss)
            for k,v in m.state_dict().items():self.assertTrue(torch.equal(v,other.state_dict()[k]),k)
            with self.assertRaises(RuntimeError):model_from_hparams(dict(H,model='xunet_wide')).load_state_dict(net.state_dict())
            from cvdl.curriculum import initialize_scratch50
            from cvdl.cov_training import check_start
            settings=load(environ={},_profile=machine_profile('pc',device='cpu'),**top_values(json.loads(NOTEBOOK.read_text())))
            initialize_scratch50(settings,Path(tmp),False)
            record=dict(model=MODEL,cov_pool_identity=SPEC,srm_trainable=False,initial_weights_sha256='0'*64)
            (Path(tmp)/'model_run_settings.json').write_text(json.dumps(record));torch.save(saved,Path(tmp)/'last.ckpt')
            self.assertTrue(check_start(settings,tmp))
            wrong=copy.deepcopy(saved);wrong['hyper_parameters']['cov_pool_identity']=dict(SPEC,iterations=6)
            torch.save(wrong,Path(tmp)/'last.ckpt')
            with self.assertRaisesRegex(ValueError,'identity'):check_start(settings,tmp)

    def test_nested_inference_attribution_and_explicit_photo_kind(self):
        from cvdl.cell_diagnostics import cell_maps
        from cvdl.photo_inference import infer_photo
        net=XuNetWideGAPCov32().eval();x=torch.rand(2,3,64,64)
        net.classifier.weight.grad=torch.ones_like(net.classifier.weight)
        before=net.classifier.weight.grad.clone();backwards=[]
        hook=net.conv7.register_full_backward_hook(lambda *a:backwards.append(True))
        with torch.inference_mode():cells,logits=net.class_score_cells(x);score,aux,prob=cell_maps(net,x)
        hook.remove()
        self.assertEqual(cells.shape,(2,2,2,2));self.assertIsNone(aux);self.assertFalse(backwards)
        self.assertTrue(torch.allclose(logits,net(x),atol=1e-7,rtol=1e-5))
        self.assertTrue(torch.equal(score,cells[:,1]-cells[:,0]));self.assertTrue(torch.allclose(prob,logits.softmax(1)[:,1]))
        self.assertTrue(torch.equal(before,net.classifier.weight.grad));self.assertFalse(net.training)
        with torch.no_grad():net.classifier.weight[:,256:]=0
        with torch.inference_mode():cells,logits=net.class_score_cells(x);f=net.features(x)
        expected=F.conv2d(f,net.classifier.weight[:,:256,None,None].detach())
        self.assertTrue(torch.allclose(cells[:,1],expected[:,1]-expected[:,0],atol=1e-7,rtol=1e-5))
        with self.assertRaisesRegex(ValueError,'requires attribution'):infer_photo(net,None,None,'cpu',1,map_kinds=('cam',))
        with self.assertRaisesRegex(ValueError,'GAP'):infer_photo(XuNetWide(front_end='srm'),None,None,'cpu',1,map_kinds=('attribution',))

    def test_covariance_attribution_direction_and_photo_stitch(self):
        import numpy as np
        from cvdl.cov_pool import descriptor
        from cvdl.photo_inference import infer_photo
        from scripts.cell_eval import describe_map
        net=XuNetWideGAPCov32().eval()
        with torch.no_grad():net.classifier.weight[:,:256]=0
        x=torch.rand(1,3,64,64)
        with torch.inference_mode():cells,_=net.class_score_cells(x);h=net.cov_reduce(net.features(x))
        h=h.clone().double();delta=h[:,:,0,0]-h.mean((-2,-1))
        plus=h.clone();minus=h.clone();eps=1e-4
        plus[:,:,0,0]+=eps*delta;minus[:,:,0,0]-=eps*delta
        weight=net.classifier.weight[:,256:].detach().double()
        def margin(v):
            y=F.linear(descriptor(v),weight);return y[:,1]-y[:,0]
        expected=(margin(plus)-margin(minus))/(2*eps)*4
        self.assertTrue(torch.allclose(cells[:,1,0,0].double(),expected,atol=1e-8,rtol=1e-4))
        photo=np.zeros((270,256,3),dtype=np.uint8)
        origins,scores,maps=infer_photo(net,photo,dict(source_height=270,source_width=256),'cpu',2,map_kinds=('attribution',))
        self.assertEqual(origins,[(0,0),(14,0)]);self.assertEqual(len(scores),2)
        self.assertEqual(maps.shape,(1,9,8));self.assertTrue(np.isfinite(maps).all())
        summary={};describe_map(summary,net)
        self.assertTrue(summary['context_only']);self.assertFalse(summary['main_rule_affected'])
        self.assertEqual(summary['map_kind'],'attribution');self.assertIn('not a cell probability',summary['map_definition'])

    def test_notebook_exact_diff_rocm_and_statistics_no_state_change(self):
        from cvdl.runtime import conv1_factory
        from cvdl.cov_pool import branch_statistics
        base=(ROOT/'notebooks/experiments/srm-e50-paired.ipynb').read_text();new=NOTEBOOK.read_text()
        self.assertEqual(new.replace('    "    val_batch_images=64,\\n",\n','',1),
            base.replace('name=\\"srm-e50-paired\\"','name=\\"srm-e50-paired-gapcov32\\"').replace("model='xunet_wide',","model='xunet_wide_gapcov32',",1))
        for profile in ('pc','g4'):
            a,b=[plain(load(environ={},_profile=machine_profile(profile,device='cpu'),**top_values(json.loads(t))).values) for t in (base,new)]
            self.assertEqual({k for k in a.keys()|b.keys() if a.get(k)!=b.get(k)},{'name','model','run_name','val_batch_images'})
        s=load(environ={},_profile=machine_profile('pc',device='cuda',rocm=True),**H)
        self.assertIsNotNone(conv1_factory(s));self.assertTrue(s.legacy_globals()['ROCM_CONV1_FIX_ACTIVE'])
        from cvdl.runtime import model_checks
        cpu=load(environ={},_profile=machine_profile('pc',device='cpu'),**H)
        model_checks(cpu,torch.rand(2,3,64,64),torch.rand(2,3,64,64))
        n=XuNetWideGAPCov32();before={k:v.clone() for k,v in n.state_dict().items()}
        stats=branch_statistics(n,torch.rand(2,3,64,64))
        self.assertIn('ratio_above_10',stats)
        for k,v in before.items():self.assertTrue(torch.equal(v,n.state_dict()[k]))


if __name__=='__main__':unittest.main()
