"""Arm B36 opt-in, frozen rules and original-context provenance."""
import difflib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch
import numpy as np
import torch
from torch import nn
from cvdl.models import XuNetWide,XuNetWideColourSep36,model_from_hparams
from cvdl.settings import load,machine_profile,plain
from cvdl.runtime import conv1_factory
from scripts.g4.preflight import top_values
from scripts.training_decision import rule_checks,wiser_identity,WISER_REFERENCE,WISER_REFERENCE_SHA
ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'notebooks/experiments/srm-e50-paired.ipynb'
NEW=ROOT/'notebooks/experiments/srm-e50-paired-coloursep36.ipynb'
H=dict(model='xunet_wide_coloursep36',front_end='srm',norm='none',width=4)


class ArchitectureTests(unittest.TestCase):
    def test_shapes_pool_order_and_mac_counts(self):
        net=XuNetWideColourSep36();shapes=[];macs=[];hooks=[]
        for m in net.modules():
            if isinstance(m,nn.Conv2d):
                hooks.append(m.register_forward_hook(lambda m,a,o:macs.append(o.numel()*m.weight[0].numel())))
        hooks.append(net.pool.register_forward_hook(lambda m,a,o:shapes.append(tuple(o.shape))))
        with torch.no_grad():out=net(torch.zeros(1,3,256,256))
        for hook in hooks:hook.remove()
        self.assertEqual(out.shape,(1,2))
        self.assertEqual(shapes,[(1,36,128,128),(1,64,64,64),(1,128,32,32),(1,256,16,16),(1,512,8,8)])
        self.assertEqual(net.features(torch.zeros(1,3,32,32)).shape,(1,256,1,1))
        self.assertEqual((net.conv1.groups,net.conv1.weight.shape,net.conv2.groups,net.conv2.in_channels),(3,torch.Size([36,30,5,5]),1,36))
        self.assertEqual(sum(p.numel() for p in net.parameters()),654376)
        self.assertEqual(sum(p.numel() for p in net.parameters() if p.requires_grad),652126)
        self.assertEqual(sum(macs)+512,2986476032)
        self.assertFalse(net.kv.conv.weight.requires_grad)

    def test_colour_isolation_and_unchanged_body_rng(self):
        torch.manual_seed(4);base=XuNetWide(front_end='srm',norm='none');state=torch.get_rng_state()
        torch.manual_seed(4);net=XuNetWideColourSep36()
        self.assertTrue(torch.equal(state,torch.get_rng_state()))
        for key,value in base.state_dict().items():
            if not key.startswith(('conv1.','conv2.')):self.assertTrue(torch.equal(value,net.state_dict()[key]),key)
        a=torch.randn(1,90,16,16);b=a.clone();b[:,:30]+=1
        first=lambda x:net.pool(torch.tanh(net.conv1(x).abs()))
        self.assertTrue(torch.equal(first(a)[:,12:],first(b)[:,12:]))
        self.assertFalse(torch.equal(first(a)[:,:12],first(b)[:,:12]))

    def test_factory_and_checkpoint_roundtrip(self):
        from cvdl.lit import SteganalysisModule
        from cvdl.standardtest import load_network
        import lightning
        module=SteganalysisModule(**H)
        with TemporaryDirectory() as tmp:
            p=Path(tmp)/'best-test.ckpt'
            torch.save(dict(state_dict=module.state_dict(),hyper_parameters=dict(module.hparams),epoch=0,
                **{'pytorch-lightning_version':lightning.__version__}),p)
            network=load_network(p,'cpu')
            self.assertIsInstance(network,XuNetWideColourSep36)
            self.assertEqual(network.conv1.groups,3)
            copy=SteganalysisModule.load_from_checkpoint(p,map_location='cpu',weights_only=False)
            for key,value in module.state_dict().items():self.assertTrue(torch.equal(value,copy.state_dict()[key]))
        with self.assertRaises(RuntimeError):model_from_hparams(dict(H,model='xunet_wide')).load_state_dict(module.net.state_dict())
        with self.assertRaises(RuntimeError):model_from_hparams({}).load_state_dict(module.net.state_dict())

    def test_reject_invalid_configuration(self):
        for change in (dict(front_end='kv'),dict(norm='batch'),dict(width=3),dict(clip_t=2),dict(cell_supervision='changed_cells_v1')):
            with self.subTest(change=change),self.assertRaises(ValueError):model_from_hparams(dict(H,**change))
            with self.assertRaises(ValueError):load(environ={},_profile=machine_profile('pc',device='cpu'),**dict(H,**change))

    def test_cpu_finite_paired_updates_and_srm_fp32_boundary(self):
        from cvdl.lit import SteganalysisModule
        module=SteganalysisModule(**H,paired_batches=True);module.log=lambda *a,**k:None
        optimizer=module.configure_optimizers()['optimizer']
        original=module.net.kv.conv.weight.detach().clone()
        batch=(torch.randint(0,256,(2,3,32,32),dtype=torch.uint8),torch.randint(0,256,(2,3,32,32),dtype=torch.uint8),torch.arange(2))
        for i in range(2):
            optimizer.zero_grad(set_to_none=True);result=module.training_step(batch,i)
            self.assertEqual(result['count'],4);self.assertTrue(torch.isfinite(result['loss']))
            result['loss'].backward();optimizer.step()
        self.assertIsNone(module.net.kv.conv.weight.grad)
        self.assertTrue(torch.equal(original,module.net.kv.conv.weight))
        observed=[]
        hook=module.net.conv1.register_forward_pre_hook(lambda m,a:observed.append(a[0].dtype))
        with torch.autocast('cpu',dtype=torch.bfloat16):module.net(batch[0].float())
        hook.remove();self.assertEqual(observed,[torch.float32])

    def test_notebook_exact_two_lines_and_effective_parity(self):
        a=BASE.read_text();b=NEW.read_text()
        self.assertEqual(b,a.replace('name=\\"srm-e50-paired\\"','name=\\"srm-e50-paired-coloursep36\\"').replace("model='xunet_wide',","model='xunet_wide_coloursep36',",1))
        self.assertEqual(len([x for x in difflib.ndiff(a.splitlines(),b.splitlines()) if x.startswith('- ')]),2)
        for profile in ('pc','g4'):
            values=[plain(load(environ={},_profile=machine_profile(profile,device='cpu'),**top_values(json.loads(p.read_text()))).values) for p in (BASE,NEW)]
            difference={k for k in values[0] if values[0][k]!=values[1].get(k)}
            self.assertEqual(difference,{'name','model','run_name'})
            self.assertEqual(values[1]['checkpoint_validation'],'capped')
            self.assertEqual(values[1]['batch_size'],128)

    def test_pilot_uses_explicit_machine_overrides_and_frozen_tolerance(self):
        from scripts.pilot_wiser_b import one_model,TOLERANCE,relative_error
        self.assertEqual(TOLERANCE,1e-4)
        self.assertEqual(relative_error(torch.zeros(2),torch.zeros(2)),0.)
        with self.assertRaises(ValueError):relative_error(torch.ones(2),torch.zeros(2))
        for name in ('xunet_wide','xunet_wide_coloursep36'):
            with patch('cvdl.runtime.configure',side_effect=StopIteration) as configure:
                with self.assertRaises(StopIteration):one_model(name,Path('/tmp/not-created'))
            settings=configure.call_args.args[0]
            self.assertEqual((settings.batch_size,settings.workers,settings.eval_workers),(16,4,4))
            self.assertTrue(settings.pilot);self.assertFalse(settings.prime_next_epoch)
            self.assertEqual(settings.compile,'off')
            self.assertEqual(settings.checkpoint_validation,'train_matched')
            self.assertEqual(settings.validation_recipe,'60da4391043e8ce48ff2')

    def test_native_grouped_rocm_and_legacy_dense_fix(self):
        for name,active in [('xunet_wide_coloursep36',False),('xunet_wide',True)]:
            s=load(environ={},_profile=machine_profile('pc',device='cuda',rocm=True),**dict(H,model=name))
            self.assertEqual(conv1_factory(s) is not None,active)
            self.assertEqual(s.legacy_globals()['ROCM_CONV1_FIX_ACTIVE'],active)
        self.assertIn('native grouped',load(environ={},_profile=machine_profile('pc',device='cpu'),**H).legacy_globals()['ROCM_CONV1_FIX_REASON'])


class RuleTests(unittest.TestCase):
    def test_both_routes_boundaries_equal_old_rule(self):
        hard=dict(point=.01,low=np.nextafter(0.,1.),high=.02);standard=dict(point=0.,low=-.01,high=.0025)
        self.assertTrue(rule_checks('wiser_b',hard,standard)['passed'])
        for h,s in [(dict(hard,point=np.nextafter(.01,0)),standard),(dict(hard,low=0.),standard),
                    (hard,dict(standard,high=np.nextafter(.0025,1.)))]:
            self.assertFalse(rule_checks('wiser_b',h,s)['passed'])
        hard=dict(point=0.,low=-.01,high=.02);standard=dict(point=-.002,low=-.01,high=np.nextafter(0.,-1.))
        self.assertTrue(rule_checks('wiser_b',hard,standard)['passed'])
        for h,s in [(dict(hard,low=np.nextafter(-.01,-1.)),standard),(hard,dict(standard,point=np.nextafter(-.002,0.))),
                    (hard,dict(standard,high=0.))]:self.assertFalse(rule_checks('wiser_b',h,s)['passed'])
        for h in (-.02,-.01,0.,.01,.02):
            for p in (-.003,-.002,0.,.0025,.003):
                a=dict(point=h,low=h,high=h);b=dict(point=p,low=p,high=p)
                self.assertEqual(rule_checks('paired_r1',a,b),rule_checks('wiser_b',a,b))

    def test_identity_rejections_and_best_sha(self):
        values=plain(load(environ={},_profile=machine_profile('g4'),**top_values(json.loads(NEW.read_text()))).values)
        with TemporaryDirectory() as tmp:
            path=Path(tmp)/'best-epoch01.ckpt'
            h=dict(H,clip_t=3.,paired_batches=True,max_epochs=50,batch_size=128,precision='32',lr=3e-4,
                weight_decay=1e-4,warmup_epochs=1,grad_clip=1.,lr_drops=[[8,.3],[33,.1]],steps_per_epoch=2813)
            torch.save(dict(epoch=0,hyper_parameters=h),path)
            torch.save(dict(epoch=49,global_step=140650,hyper_parameters=h),path.parent/'last.ckpt')
            from scripts.training_decision import sha
            record=path.parent/'run_settings.json';record.write_text(json.dumps(dict(values=values)))
            # Real evaluation changes inference batch/compile, never the training recipe.
            meta=dict(values=dict(values,batch_size=32,compile='off'),checkpoint=str(path),checkpoint_sha256=sha(path))
            from scripts.training_decision import checkpoint_path
            reference=json.loads((ROOT/WISER_REFERENCE/'evaluation_settings.json').read_text())
            refpath=checkpoint_path(reference)
            data=json.loads((refpath.parent/'data_run_settings.json').read_text())
            (path.parent/'data_run_settings.json').write_text(json.dumps(data))
            folders=[ROOT/WISER_REFERENCE,Path(tmp)]
            self.assertEqual(wiser_identity(folders,[reference,meta])[0],1)
            lastpath=path.parent/'last.ckpt'
            torch.save(dict(epoch=48,global_step=140650,hyper_parameters=h),lastpath)
            with self.assertRaisesRegex(ValueError,'full 50-epoch'):wiser_identity(folders,[reference,meta])
            torch.save(dict(epoch=49,global_step=140650,hyper_parameters=dict(h,model='xunet_wide')),lastpath)
            with self.assertRaisesRegex(ValueError,'completed checkpoint identity'):wiser_identity(folders,[reference,meta])
            torch.save(dict(epoch=49,global_step=140650,hyper_parameters=h),lastpath)
            for change in (dict(model='xunet_wide'),dict(init_checkpoint='foo'),dict(checkpoint_validation='train_matched'),dict(paired_batches=False),dict(epochs=49),dict(seed=1),dict(run_seed=1),dict(precision='bf16-mixed'),dict(channels_last=True),dict(compile='off')):
                record.write_text(json.dumps(dict(values=dict(values,**change))))
                with self.assertRaises(ValueError):wiser_identity(folders,[reference,meta])
            record.write_text(json.dumps(dict(values=values)))
            (path.parent/'data_run_settings.json').write_text(json.dumps(dict(data,train_photo_selection=['drift'])))
            with self.assertRaisesRegex(ValueError,'training data recipe/selection'):wiser_identity(folders,[reference,meta])
            (path.parent/'data_run_settings.json').write_text(json.dumps(data))
            with self.assertRaises(ValueError):wiser_identity([ROOT/'wrong',Path(tmp)],[reference,meta])
            with self.assertRaises(ValueError):wiser_identity(folders,[reference,dict(meta,checkpoint_sha256='0'*64)])
            last=path.with_name('last.ckpt');last.write_bytes(path.read_bytes())
            with self.assertRaises(ValueError):wiser_identity(folders,[reference,dict(meta,checkpoint=str(last))])


class ContextTests(unittest.TestCase):
    def test_model_only_declared_addition_and_original_blob(self):
        import ast,subprocess
        from cvdl.channel_context import MODEL_SHA,ORIGINAL_COMMIT
        from cvdl.channel_analysis import file_hash
        self.assertEqual(file_hash(ROOT/'cvdl/models.py'),MODEL_SHA)
        old=ast.parse(subprocess.check_output(['git','show',ORIGINAL_COMMIT+':cvdl/models.py'],cwd=ROOT,text=True))
        new=ast.parse((ROOT/'cvdl/models.py').read_text())
        old_nodes={n.name:ast.dump(n) for n in old.body if isinstance(n,(ast.ClassDef,ast.FunctionDef))}
        new_nodes={n.name:ast.dump(n) for n in new.body if isinstance(n,(ast.ClassDef,ast.FunctionDef))}
        for key,value in old_nodes.items():
            if key!='model_from_hparams':self.assertEqual(value,new_nodes[key],key)
        text=(ROOT/'cvdl/models.py').read_text()
        start=text.index("    elif model == 'xunet_wide_coloursep36':")
        end=text.index("    elif model == 'gbras':",start)
        factory=next(n for n in ast.parse(text[:start]+text[end:]).body if isinstance(n,ast.FunctionDef) and n.name=='model_from_hparams')
        self.assertEqual(ast.dump(factory),old_nodes['model_from_hparams'])

    def test_frozen_full_panel_replay_workers_zero_four(self):
        from cvdl.channel_analysis import ChannelDataset,digest
        from cvdl.channel_context import validate_original
        panel=ROOT/'runs/verification/rgb-channel-analysis-2026-10-24'
        _,base,variants=validate_original(ROOT,panel)
        results=[]
        for workers in (0,4):
            kwargs=dict(batch_size=32,num_workers=workers,shuffle=False)
            if workers:kwargs['multiprocessing_context']='spawn'
            observed=[]
            for images,indices,counts,valid in torch.utils.data.DataLoader(ChannelDataset(panel,base,variants),**kwargs):
                for i,image,count,mask in zip(indices.tolist(),images,counts,valid):
                    observed.append((i,digest(image.numpy().tobytes()),digest(count.numpy().tobytes()),digest(mask.numpy().tobytes())))
            self.assertEqual([r[0] for r in observed],list(range(29280)))
            results.append(observed)
        self.assertEqual(*results)

    def test_context_manifest_drift_and_sibling_guards(self):
        from cvdl import channel_context as ctx
        with TemporaryDirectory() as tmp:
            root=Path(tmp);panel=root/'panel';panel.mkdir();out=root/'context';out.mkdir()
            with self.assertRaisesRegex(ValueError,'new sibling'):ctx.freeze(root,panel,root/'evaluation',out)
            candidate=root/'best.ckpt';candidate.write_bytes(b'checkpoint')
            evaluation=root/'evaluation';evaluation.mkdir();(evaluation/'calibration.json').write_text('calibration')
            original=ctx.snapshot(panel)
            m=dict(original_panel=str(panel),original_sha256=original,current_model_sha256=ctx.MODEL_SHA,
                context_code_sha256=ctx.old.file_hash(Path(ctx.__file__)),checkpoint=str(candidate),
                checkpoint_sha256=ctx.old.file_hash(candidate),evaluation=str(evaluation),
                evaluation_inputs={'calibration.json':ctx.old.file_hash(evaluation/'calibration.json')})
            ctx.old.dump(out/'manifest.json',m);(out/'manifest.sha256').write_text(ctx.old.digest(ctx.old.canonical(m).encode()))
            with patch.object(ctx,'validate_original',return_value=({},[],[])):
                ctx.checked(root,out)
                candidate.write_bytes(b'changed')
                with self.assertRaisesRegex(ValueError,'checkpoint drift'):ctx.checked(root,out)
                candidate.write_bytes(b'checkpoint');(evaluation/'calibration.json').write_text('wrong cutoff')
                with self.assertRaisesRegex(ValueError,'calibration drift'):ctx.checked(root,out)
                (panel/'unexpected').write_text('drift')
                with self.assertRaisesRegex(ValueError,'original folder changed'):ctx.checked(root,out)
            (out/'manifest.json').write_text('{}')
            with self.assertRaises((ValueError,KeyError)):ctx.checked(root,out)

    def test_original_validation_and_code_drift(self):
        from cvdl.channel_context import validate_original
        panel=ROOT/'runs/verification/rgb-channel-analysis-2026-10-24'
        recipe,base,variants=validate_original(ROOT,panel)
        self.assertEqual(len(base)+len(variants),29280)
        with patch('cvdl.channel_context.MODEL_SHA','0'*64):
            with self.assertRaisesRegex(ValueError,'undeclared code drift'):validate_original(ROOT,panel)


if __name__=='__main__':unittest.main()
