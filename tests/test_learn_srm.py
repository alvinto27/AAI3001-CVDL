"""Opt-in SRM autograd, same-shaped checkpoint identity and frozen decisions."""
import copy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch
import numpy as np
import torch
from cvdl.models import XuNetWide,XuNetWideLearnSRM,model_from_hparams,clipped_residuals
from cvdl.lit import SteganalysisModule
from cvdl.runtime import conv1_factory,model_checks
from cvdl.settings import load,machine_profile,plain,SCHEMA
from scripts.g4.preflight import top_values
from scripts.training_decision import (learn_srm_identity,rule_checks,sha,checkpoint_path,
    LEARN_NOTEBOOK,LEARN_NOTEBOOK_SHA,WISER_REFERENCE)
ROOT=Path(__file__).resolve().parents[1]
H=dict(model='xunet_wide_learn_srm',front_end='srm',norm='none',width=4,clip_t=3.)


def historical_learn_sources(test):
    """Exercise the old rule against its REAL approved source, not a fake hash.

    The live cov32 checkout intentionally fails the old source guard. Keep the
    old valid-provenance fixture useful without broadening that frozen scope.
    """
    import functools
    @functools.wraps(test)
    def wrapped(*args,**kwargs):
        import subprocess
        original=ROOT
        with TemporaryDirectory() as tmp:
            root=Path(tmp)
            for name in ('configs','data','runs','notebooks','scripts','LSB-Dataset-Generator'):
                (root/name).symlink_to(original/name,target_is_directory=True)
            (root/'cvdl').mkdir()
            for p in (original/'cvdl').glob('*.py'):
                if p.name=='models.py':(root/'cvdl'/p.name).write_bytes(subprocess.check_output(['git','show','c0412e0:cvdl/models.py'],cwd=original))
                else:(root/'cvdl'/p.name).symlink_to(p)
            gitdir=subprocess.check_output(['git','rev-parse','--absolute-git-dir'],cwd=original,text=True).strip()
            (root/'.git').write_text('gitdir: '+gitdir+'\n')
            with patch(__name__+'.ROOT',root),patch('scripts.training_decision.ROOT',root):return test(*args,**kwargs)
    return wrapped


class ModelTests(unittest.TestCase):
    def test_only_trainability_diff_and_invalid_choices(self):
        torch.manual_seed(7);base=XuNetWide(front_end='srm',norm='none');rng=torch.get_rng_state()
        torch.manual_seed(7);net=XuNetWideLearnSRM()
        self.assertTrue(torch.equal(rng,torch.get_rng_state()))
        self.assertEqual(list(base.state_dict()),list(net.state_dict()))
        for k,v in base.state_dict().items():self.assertTrue(torch.equal(v,net.state_dict()[k]),k)
        self.assertEqual([n for (n,p),(n2,q) in zip(base.named_parameters(),net.named_parameters())
            if p.requires_grad!=q.requires_grad],['kv.conv.weight'])
        self.assertEqual((net.kv.conv.groups,net.conv1.groups,net.kv.conv.bias,net.clip_t),(3,1,None,3.))
        self.assertEqual(sum(p.numel() for p in net.parameters() if p.requires_grad),631356)
        for change in (dict(front_end='kv'),dict(norm='batch'),dict(width=3),dict(clip_t=2.),dict(cell_supervision='changed_cells_v1')):
            with self.subTest(change=change),self.assertRaises(ValueError):model_from_hparams(dict(H,**change))
            with self.assertRaises(ValueError):load(environ={},_profile=machine_profile('pc',device='cpu'),**dict(H,**change))

    def test_clamp_autograd_and_fp32_autocast(self):
        net=XuNetWideLearnSRM();x=torch.randn(2,3,9,11)*.2
        raw=net.kv(x);raw.retain_grad();raw.clamp(-3,3).square().mean().backward()
        self.assertTrue(torch.isfinite(net.kv.conv.weight.grad).all());self.assertGreater(net.kv.conv.weight.grad.abs().max(),0)
        net.zero_grad(set_to_none=True)
        # Use actual guarded front (not a duplicate convolution) under autocast.
        with torch.autocast('cpu',dtype=torch.bfloat16):residual=clipped_residuals(net.kv,x,net.clip_t)
        self.assertEqual(residual.dtype,torch.float32);residual.square().mean().backward()
        self.assertEqual(net.kv.conv.weight.grad.dtype,torch.float32)
        self.assertGreater(net.kv.conv.weight.grad.abs().max(),0)
        observed=[]
        def capture(m,a,o):o.retain_grad();observed.append(o)
        hook=net.kv.conv.register_forward_hook(capture)
        clipped_residuals(net.kv,x*100,3.).sum().backward();hook.remove()
        raw=observed[0];saturated=raw.detach().abs()>3;inside=raw.detach().abs()<3
        self.assertTrue(saturated.any());self.assertTrue(inside.any())
        self.assertTrue((raw.grad[saturated]==0).all());self.assertTrue((raw.grad[inside]==1).all())

    def test_adam_resume_reconstruction_and_real_probe(self):
        from cvdl.standardtest import load_network
        torch.manual_seed(4)
        module=SteganalysisModule(**H,paired_batches=True,max_epochs=50,steps_per_epoch=3,
            lr_drops=((8,.3),(33,.1)),val_batch_images=64,batch_size=128)
        module.log=lambda *a,**k:None
        batch=(torch.rand(2,3,32,32),torch.rand(2,3,32,32),torch.arange(2))
        config=module.configure_optimizers();opt=config['optimizer'];sched=config['lr_scheduler']['scheduler']
        initial=module.net.kv.conv.weight.detach().clone()
        self.assertTrue(any(p is module.net.kv.conv.weight for g in opt.param_groups for p in g['params']))
        def step(m,o,s):
            o.zero_grad(set_to_none=True);result=m.training_step(batch,0)
            self.assertEqual(result['count'],4);result['loss'].backward()
            self.assertGreater(m.net.kv.conv.weight.grad.abs().max(),0)
            torch.nn.utils.clip_grad_norm_(m.parameters(),1.);o.step();s.step()
        step(module,opt,sched)
        self.assertFalse(torch.equal(initial,module.net.kv.conv.weight))
        import lightning
        saved=dict(**{'pytorch-lightning_version':lightning.__version__},global_step=1,hyper_parameters=dict(module.hparams),state_dict=copy.deepcopy(module.state_dict()),
            optimizer_states=[copy.deepcopy(opt.state_dict())],lr_schedulers=[copy.deepcopy(sched.state_dict())])
        with TemporaryDirectory() as tmp:
            p=Path(tmp)/'best.ckpt';torch.save(saved,p);net=load_network(p,'cpu')
            self.assertIsInstance(net,XuNetWideLearnSRM);self.assertTrue(net.kv.conv.weight.requires_grad)
            other=SteganalysisModule.load_from_checkpoint(p,map_location='cpu',weights_only=False);other.log=lambda *a,**k:None
            other.load_state_dict(saved['state_dict']);c=other.configure_optimizers();o=c['optimizer'];s=c['lr_scheduler']['scheduler']
            o.load_state_dict(copy.deepcopy(saved['optimizer_states'][0]));s.load_state_dict(copy.deepcopy(saved['lr_schedulers'][0]))
            step(module,opt,sched);step(other,o,s)
            for k,v in module.state_dict().items():self.assertTrue(torch.equal(v,other.state_dict()[k]),k)
            self.assertTrue(torch.equal(opt.state[module.net.kv.conv.weight]['exp_avg'],o.state[other.net.kv.conv.weight]['exp_avg']))
            self.assertEqual(float(o.state[other.net.kv.conv.weight]['step']),2.)
            from cvdl.curriculum import initialize_scratch50
            from cvdl.learn_srm_training import check_start
            settings=load(environ={},_profile=machine_profile('pc',device='cpu'),**top_values(json.loads(LEARN_NOTEBOOK.read_text())))
            folder=Path(tmp);initialize_scratch50(settings,folder,False)
            (folder/'model_run_settings.json').write_text(json.dumps(dict(model=H['model'],srm_trainable=True,initial_weights_sha256='0'*64)))
            torch.save(saved,folder/'last.ckpt');self.assertTrue(check_start(settings,folder))
            wrong=copy.deepcopy(saved);wrong['hyper_parameters']['model']='xunet_wide'
            torch.save(wrong,folder/'last.ckpt')
            with self.assertRaisesRegex(ValueError,'checkpoint identity'):check_start(settings,folder)
            wrong=copy.deepcopy(saved);wrong['optimizer_states'][0]['state'].pop(0)
            torch.save(wrong,folder/'last.ckpt')
            with self.assertRaisesRegex(ValueError,'SRM optimizer state'):check_start(settings,folder)
        for tag in ('xunet_wide','xunet_wide_learn_srm'):
            settings=load(environ={},_profile=machine_profile('pc',device='cpu'),**dict(H,model=tag))
            model_checks(settings,batch[0][:1],batch[1][:1])

    def test_notebook_diff_and_rocm_factory(self):
        base=(ROOT/'notebooks/experiments/srm-e50-paired.ipynb').read_text();new=LEARN_NOTEBOOK.read_text()
        self.assertEqual(sha(LEARN_NOTEBOOK),LEARN_NOTEBOOK_SHA)
        self.assertEqual(new.replace('    "    val_batch_images=64,\\n",\n','',1),
            base.replace('name=\\"srm-e50-paired\\"','name=\\"srm-e50-paired-learn-srm\\"').replace("model='xunet_wide',","model='xunet_wide_learn_srm',",1))
        for profile in ('pc','g4'):
            values=[plain(load(environ={},_profile=machine_profile(profile,device='cpu'),**top_values(json.loads(text))).values) for text in (base,new)]
            self.assertEqual({k for k in values[0].keys()|values[1].keys() if values[0].get(k)!=values[1].get(k)},
                {'name','run_name','model','val_batch_images'})
            self.assertEqual((values[1]['batch_size'],values[1]['val_batch_images']),(128,64))
        settings=load(environ={},_profile=machine_profile('pc',device='cuda',rocm=True),**H)
        self.assertIsNotNone(conv1_factory(settings));self.assertTrue(settings.legacy_globals()['ROCM_CONV1_FIX_ACTIVE'])
        conv=conv1_factory(settings)(XuNetWideLearnSRM().conv1)
        self.assertEqual(conv.groups,1)


    def test_pilot_settings_and_tolerance_are_explicit(self):
        from scripts.pilot_learn_srm import one_model,TOLERANCE,relative_error
        self.assertEqual(TOLERANCE,1e-4)
        self.assertEqual(relative_error(torch.zeros(1),torch.zeros(1)),0.)
        with self.assertRaises(ValueError):relative_error(torch.ones(1),torch.zeros(1))
        for tag in ('xunet_wide','xunet_wide_learn_srm'):
            with patch('cvdl.runtime.configure',side_effect=StopIteration) as configure:
                with self.assertRaises(StopIteration):one_model(tag,Path('/unused'))
            s=configure.call_args.args[0]
            self.assertEqual((s.batch_size,s.workers,s.eval_workers),(16,4,4))
            self.assertTrue(s.pilot);self.assertFalse(s.prime_next_epoch);self.assertEqual(s.compile,'off')
            self.assertEqual((s.checkpoint_validation,s.validation_recipe,s.post_fit_eval),('train_matched','60da4391043e8ce48ff2','none'))


class IdentityTests(unittest.TestCase):
    def test_both_routes_no_epoch_or_cell_gate(self):
        for hard in (-.02,-.01,0.,.01,.02):
            for pe in (-.003,-.002,0.,.0025,.003):
                h=dict(point=hard,low=hard,high=hard);s=dict(point=pe,low=pe,high=pe)
                self.assertEqual(rule_checks('learn_srm',h,s),rule_checks('paired_r1',h,s))
        h=dict(point=.01,low=np.nextafter(0.,1.),high=.03);s=dict(point=0.,low=-.01,high=.0025)
        self.assertTrue(rule_checks('learn_srm',h,s)['passed'])
        self.assertFalse(rule_checks('learn_srm',dict(h,low=0.),s)['passed'])
        self.assertFalse(rule_checks('learn_srm',h,dict(s,high=np.nextafter(.0025,1.)))['passed'])

    @historical_learn_sources
    def test_same_shaped_wrong_tag_and_incomplete_provenance_rejected(self):
        values=plain(load(environ={},_profile=machine_profile('g4'),**top_values(json.loads(LEARN_NOTEBOOK.read_text()))).values)
        reference=json.loads((ROOT/WISER_REFERENCE/'evaluation_settings.json').read_text())
        data=json.loads((checkpoint_path(reference).parent/'data_run_settings.json').read_text());data['val_batch_images']=64
        h=dict(H,paired_batches=True,max_epochs=50,batch_size=128,precision='32',lr=3e-4,weight_decay=1e-4,
            warmup_epochs=1,grad_clip=1.,lr_drops=[[8,.3],[33,.1]],steps_per_epoch=2813,
            val_batch_images=64,effective_val_batch_images=64)
        with TemporaryDirectory() as tmp:
            root=Path(tmp);best=root/'best-epoch01.ckpt';last=root/'last.ckpt'
            def save(hp=h,epoch=49,steps=140650):
                torch.save(dict(epoch=0,global_step=2813,hyper_parameters=hp),best)
                torch.save(dict(epoch=epoch,global_step=steps,hyper_parameters=hp),last)
                return dict(values=dict(values,batch_size=32,compile='off'),checkpoint=str(best),checkpoint_sha256=sha(best))
            for name,v in [('run_settings',dict(values=values)),('data_run_settings',data),
                ('model_run_settings',dict(model=H['model'],srm_trainable=True)),
                ('training_start',dict(initialization='random',weights_only=False,
                    settings=dict({k:v for k,v in values.items() if not SCHEMA[k].machine},
                        train_yaml_sha256=sha(ROOT/'configs/train-k1to4.yaml'))))]:(root/(name+'.json')).write_text(json.dumps(v))
            folders=[ROOT/WISER_REFERENCE,root];meta=save()
            self.assertEqual(learn_srm_identity(folders,[reference,meta])[0],1)
            for bad in (dict(h,model='xunet_wide'),{k:v for k,v in h.items() if k!='model'}):
                meta=save(bad)
                with self.assertRaisesRegex(ValueError,'checkpoint identity'):learn_srm_identity(folders,[reference,meta])
            for epoch,steps in ((48,140650),(49,140649)):
                meta=save(epoch=epoch,steps=steps)
                with self.assertRaisesRegex(ValueError,'full 50-epoch'):learn_srm_identity(folders,[reference,meta])
            meta=save();(root/'model_run_settings.json').write_text(json.dumps(dict(model=H['model'],srm_trainable=False)))
            with self.assertRaisesRegex(ValueError,'trainability'):learn_srm_identity(folders,[reference,meta])
            (root/'model_run_settings.json').write_text(json.dumps(dict(model=H['model'],srm_trainable=True)))
            for change in (dict(val_batch_images=32),dict(init_checkpoint='r1.ckpt'),dict(epochs=49),dict(pilot=True)):
                (root/'run_settings.json').write_text(json.dumps(dict(values=dict(values,**change))))
                with self.assertRaises(ValueError):learn_srm_identity(folders,[reference,meta])


class ContextAndMovementTests(unittest.TestCase):
    @historical_learn_sources
    def test_scoped_context_and_drift_guards(self):
        from cvdl import learn_srm_context as ctx
        panel=ROOT/'runs/verification/rgb-channel-analysis-2026-10-24'
        self.assertEqual(ctx.METHOD,'learn-srm-rgb-context-v1');self.assertEqual(sha(ROOT/'cvdl/models.py'),ctx.MODEL_SHA)
        _,base,variants=ctx.validate_original(ROOT,panel);self.assertEqual(len(base)+len(variants),29280)
        with patch.object(ctx,'MODEL_SHA','0'*64):
            with self.assertRaisesRegex(ValueError,'undeclared code drift'):ctx.validate_original(ROOT,panel)
        with TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError,'new sibling'):ctx.freeze(ROOT,panel,ROOT,Path(tmp))

    def test_kernel_movement_reports_decay_and_zero_coefficients(self):
        from cvdl.models import SRMFilter
        from scripts.report_srm_movement import movement,decay_only
        a=SRMFilter().conv.weight;pure=movement(a,a*.9,.9)
        self.assertGreater(pure['relative_l2'],.0999);self.assertLess(pure['beyond_decay_l2'],1e-6)
        self.assertEqual(pure['rgb_bank_divergence_l2']['red_blue'],0.)
        b=a.clone();b[0,0,0,0]=.1;r=movement(a,b,1.)
        self.assertEqual(r['filters'][0]['initially_zero_moved'],1)
        self.assertGreater(r['rgb_bank_divergence_l2']['red_green'],0.)
        self.assertTrue(all(x['colour']=='blue' for x in r['filters'][60:]))
        self.assertEqual(decay_only(dict(lr=.1,weight_decay=.2,steps_per_epoch=2,max_epochs=50,warmup_epochs=0,lr_drops=[]),2),.98**2)


if __name__=='__main__':unittest.main()
