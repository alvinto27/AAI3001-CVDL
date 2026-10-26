"""Covariance provenance, descriptive FA/RMS and immutable scoped replay."""
import copy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd
import torch
from cvdl.cov_pool import SPEC,MODEL
from cvdl.models import XuNetWideGAPCov32
from cvdl.settings import SCHEMA,load,machine_profile,plain
from scripts.g4.preflight import top_values
from scripts.training_decision import (rule_checks,cov_pool_identity,checkpoint_path,sha,noisy_clean_context,
    COV_NOTEBOOK,WISER_REFERENCE,TARGETS,COV_APPROVAL)
ROOT=Path(__file__).resolve().parents[1]
H=dict(model=MODEL,front_end='srm',norm='none',width=4,clip_t=3.)


class IdentityTests(unittest.TestCase):
    def test_same_literal_routes_and_frozen_source(self):
        for hard in (-.02,-.01,0.,.01,.02):
            for pe in (-.003,-.002,0.,.0025,.003):
                h=dict(point=hard,low=hard,high=hard);s=dict(point=pe,low=pe,high=pe)
                self.assertEqual(rule_checks('cov_pool',h,s),rule_checks('paired_r1',h,s))
        for key,p in (('model_source_sha256','cvdl/models.py'),('math_source_sha256','cvdl/cov_pool.py'),('notebook_sha256',str(COV_NOTEBOOK))):
            self.assertEqual(sha(ROOT/p),COV_APPROVAL[key])

    def test_head_scratch_recipe_and_frozen_bank_identity(self):
        values=plain(load(environ={},_profile=machine_profile('g4'),**top_values(json.loads(COV_NOTEBOOK.read_text()))).values)
        reference=json.loads((ROOT/WISER_REFERENCE/'evaluation_settings.json').read_text())
        data=json.loads((checkpoint_path(reference).parent/'data_run_settings.json').read_text());data['val_batch_images']=64
        h=dict(H,cov_pool_identity=SPEC,paired_batches=True,max_epochs=50,batch_size=128,precision='32',lr=3e-4,weight_decay=1e-4,
            warmup_epochs=1,grad_clip=1.,lr_drops=[[8,.3],[33,.1]],steps_per_epoch=2813,val_batch_images=64,effective_val_batch_images=64)
        state={'net.'+k:v for k,v in XuNetWideGAPCov32().state_dict().items()}
        with TemporaryDirectory() as tmp:
            root=Path(tmp);best=root/'best-epoch00.ckpt';last=root/'last.ckpt'
            def save(hp=h,bank=None,epoch=49,step=140650):
                sd=dict(state)
                if bank is not None:sd['net.kv.conv.weight']=bank
                torch.save(dict(epoch=0,global_step=2813,hyper_parameters=hp,state_dict=sd),best)
                torch.save(dict(epoch=epoch,global_step=step,hyper_parameters=hp,state_dict=sd),last)
                return dict(values=dict(values,batch_size=32,compile='off'),checkpoint=str(best),checkpoint_sha256=sha(best))
            record=dict(model=MODEL,cov_pool_identity=SPEC,srm_trainable=False,tf32_matmul=False,initial_weights_sha256='0'*64)
            start=dict(initialization='random',weights_only=False,settings=dict({k:v for k,v in values.items() if not SCHEMA[k].machine},
                train_yaml_sha256=sha(ROOT/'configs/train-k1to4.yaml')))
            for name,v in [('run_settings',dict(values=values)),('data_run_settings',data),('model_run_settings',record),('training_start',start)]:
                (root/(name+'.json')).write_text(json.dumps(v))
            folders=[ROOT/WISER_REFERENCE,root];meta=save()
            self.assertEqual(cov_pool_identity(folders,[reference,meta])[0],1)
            for bad in (dict(h,model='xunet_wide'),dict(h,cov_pool_identity=dict(SPEC,iterations=6)),dict(h,cov_pool_identity=dict(SPEC,post_normalization='L2'))):
                meta=save(bad)
                with self.assertRaises(ValueError):cov_pool_identity(folders,[reference,meta])
            for kwargs in (dict(epoch=48),dict(step=140649),dict(epoch=50),dict(bank=state['net.kv.conv.weight']+.0001)):
                meta=save(**kwargs)
                with self.assertRaises(ValueError):cov_pool_identity(folders,[reference,meta])
            meta=save()
            for change in (dict(val_batch_images=32),dict(init_checkpoint='r1.ckpt'),dict(pilot=True),dict(epochs=49)):
                (root/'run_settings.json').write_text(json.dumps(dict(values=dict(values,**change))))
                with self.assertRaises(ValueError):cov_pool_identity(folders,[reference,meta])
            (root/'run_settings.json').write_text(json.dumps(dict(values=values)))
            with patch('scripts.training_decision.COV_APPROVAL',dict(COV_APPROVAL,math_source_sha256='0'*64)):
                with self.assertRaisesRegex(ValueError,'source drift'):cov_pool_identity(folders,[reference,meta])

    def test_noisy_union_is_distinct_and_context_only(self):
        base=dict(label=0,panel=next(iter(TARGETS)),condition='noisy')
        rows=[dict(base,cover_id='a',source_photo='one',reference=.8,candidate=.9),
            dict(base,cover_id='a',source_photo='one',reference=.1,candidate=.1),
            dict(base,condition='low_noisy',cover_id='b',source_photo='two',reference=.1,candidate=.9),
            dict(base,label=1,cover_id='c',source_photo='three',reference=.1,candidate=.1),
            dict(base,condition='low',cover_id='d',source_photo='four',reference=.1,candidate=.1)]
        result,draws=noisy_clean_context(pd.DataFrame(rows),dict(reference=.5,candidate=.5),replicates=32)
        self.assertTrue(result['context_only']);self.assertFalse(result['main_rule_affected'])
        self.assertEqual((result['distinct_tiles'],result['source_photos']),(2,2))
        self.assertEqual(result['models'],dict(reference=.5,candidate=1.));self.assertEqual(result['difference']['point'],.5)
        self.assertEqual(draws.shape,(32,2));self.assertTrue(np.isfinite(draws).all())

    def test_usage_stream_is_image_weighted_and_no_source_mutation(self):
        from cvdl.cov_usage import BlockMoments
        a=torch.tensor([[1.,3.],[2.,4.],[10.,12.]])
        m=BlockMoments();m.add(a[:2],a[:2]*.5);m.add(a[2:],a[2:]*.5);r=m.report()
        self.assertEqual((r['images'],r['batch_sizes']),(3,{2:1,1:1}))
        self.assertAlmostEqual(r['blocks']['gap']['raw_rms'],float(a.double().square().mean().sqrt()))
        self.assertAlmostEqual(r['blocks']['gap']['batch_centered_rms'],(1/6)**.5)
        self.assertAlmostEqual(r['blocks']['gap']['dataset_centered_rms'],float((a.double()-a.double().mean(0)).square().mean().sqrt()))
        self.assertAlmostEqual(r['blocks']['covariance']['raw_rms'],r['blocks']['gap']['raw_rms']/2)
        with self.assertRaises(ValueError):BlockMoments().add(a[:1],a)

    def test_best_last_usage_report_validates_complete_list_and_writes_only_output(self):
        from scripts.report_gapcov_usage import run
        state={'net.'+k:v for k,v in XuNetWideGAPCov32().state_dict().items()}
        saved=dict(epoch=49,global_step=140650,hyper_parameters=dict(H,cov_pool_identity=SPEC),state_dict=state)
        frame=pd.DataFrame(dict(split=['val','val'],cover_id=['a','b']))
        batch=(torch.arange(2),torch.rand(2,3,256,256),torch.rand(2,3,256,256))
        with TemporaryDirectory() as tmp:
            root=Path(tmp);folder=root/'ckpts';folder.mkdir()
            for name in ('best-test.ckpt','last.ckpt'):torch.save(saved,folder/name)
            before={p.name:sha(p) for p in folder.iterdir()}
            with patch('cvdl.deployment.load_recipe',return_value=(frame,dict(tile_size=256),dict(recipe_sha256='r',pairs_sha256='p'),'')),\
                    patch('lsb_torch.Corpus'),patch('cvdl.validation.MatchedValidationPairs'),patch('lsb_torch.PhotoPairTasks'),\
                    patch('torch.utils.data.DataLoader',return_value=[batch]):
                result=run(root,root/'output',workers=0,device='cpu')
            self.assertTrue(result['context_only']);self.assertTrue(result['inputs_unchanged'])
            self.assertFalse(result['main_rule_affected']);self.assertEqual(result['validation']['images'],4)
            self.assertEqual(result['checkpoints']['best']['images'],4)
            self.assertEqual(result['checkpoints']['best']['blocks'],result['checkpoints']['last']['blocks'])
            self.assertEqual(before,{p.name:sha(p) for p in folder.iterdir()})
            self.assertIn('not a cell probability',(root/'output/REPORT.md').read_text())

    def test_probe_recipe_and_independent_cpu_reference(self):
        from scripts.pilot_gapcov32 import one_model,TOLERANCE,relative_error,fp64_head
        self.assertEqual(TOLERANCE,1e-4)
        for tag in ('xunet_wide',MODEL):
            with patch('cvdl.runtime.configure',side_effect=StopIteration) as configure:
                with self.assertRaises(StopIteration):one_model(tag,Path('/unused'))
            s=configure.call_args.args[0]
            self.assertEqual((s.batch_size,s.workers,s.eval_workers),(16,4,4));self.assertTrue(s.pilot)
            self.assertFalse(s.prime_next_epoch);self.assertEqual(s.compile,'off')
            self.assertEqual((s.checkpoint_validation,s.validation_recipe,s.post_fit_eval),('train_matched','60da4391043e8ce48ff2','none'))
        n=XuNetWideGAPCov32();x=torch.randn(2,256,8,8)
        gap,cov=n.pooled_parts(x);actual=n.classifier(torch.cat((gap,cov),1))
        ref,rcov,_,_=fp64_head(x.double(),n.cov_reduce.weight.double(),n.classifier.weight.double(),n.classifier.bias.double())
        self.assertLess(relative_error(actual,ref),1e-4);self.assertLess(relative_error(cov,rcov),1e-4)


if __name__=='__main__':unittest.main()
