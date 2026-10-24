"""Explicit evaluation-only batches: unchanged defaults, training, rows and scores."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
import numpy as np
import pandas as pd
import torch
from cvdl import data
from cvdl.lit import SteganalysisModule
from cvdl.settings import load,machine_profile,plain
from lsb_notebook import NotebookData
from lsb_torch import FixedPairs
from scripts.g4.preflight import top_values,validation_srm_elements

ROOT=Path(__file__).resolve().parents[1]
# Freeze before checking rebatching: absolute probability error <=1e-6.
SCORE_ATOL=1e-6


class TinyPairs(FixedPairs):
    def __init__(self):
        self.frame=pd.DataFrame(dict(pair_id=range(65)))
        self.entries=[({'photo_id':str(i%7)},None) for i in range(65)]
        self.corpus=SimpleNamespace(settings=SimpleNamespace(paired_batches=True,tensor_dtype='uint8'))
    def __len__(self):return len(self.frame)
    def __getitem__(self,index):
        clean=torch.randint(0,256,(3,32,32),dtype=torch.uint8,generator=torch.Generator().manual_seed(index))
        stego=clean.clone();stego[0,0,0]^=1
        return clean,stego,index


def fixture(cap=None,workers=0):
    pairs=TinyPairs();otf=NotebookData.__new__(NotebookData)
    otf.corpus=pairs.corpus;otf.train=pairs;otf.fixed_train=pairs;otf.val=pairs;otf.test=pairs;otf.record={}
    loaders=data.build_otf_loaders(otf,batch_size=128,workers=0,eval_workers=workers,pin_memory=False,
        val_batch_images=cap)
    return otf,loaders


class EvaluationBatchTests(unittest.TestCase):
    def test_none_default_preserves_all_old_settings_and_hparams(self):
        for profile in ('pc','g4'):
            p=machine_profile(profile,device='cpu')
            a=load(environ={},_profile=p)
            b=load(environ={},_profile=p,val_batch_images=None)
            self.assertEqual(plain(a.values),plain(b.values))
            self.assertEqual(plain(a.provenance),plain(b.provenance))
            self.assertNotIn('val_batch_images',a.values)
            self.assertNotIn('val_batch_images',a.legacy_globals())
        torch.manual_seed(23);a=SteganalysisModule();rng=torch.get_rng_state()
        torch.manual_seed(23);b=SteganalysisModule(val_batch_images=None)
        self.assertEqual(dict(a.hparams),dict(b.hparams))
        self.assertNotIn('val_batch_images',a.hparams)
        self.assertNotIn('effective_val_batch_images',a.hparams)
        self.assertTrue(torch.equal(rng,torch.get_rng_state()))
        for k,v in a.state_dict().items():self.assertTrue(torch.equal(v,b.state_dict()[k]))

    def test_cap_changes_only_eval_boundaries_full_counts_order_and_tails(self):
        for workers in (0,2):
            old,uncapped=fixture(workers=workers);new,capped=fixture(64,workers)
            try:
                self.assertEqual(dict(new.record),dict(old.record,val_batch_images=64))
                self.assertEqual((uncapped.train.batch_size,capped.train.batch_size),(64,64))
                for role in ('val','test','train_eval'):
                    original=list(getattr(uncapped,role));limited=list(getattr(capped,role))
                    self.assertEqual([len(b[0]) for b in original],[64,1])
                    self.assertEqual([len(b[0]) for b in limited],[32,32,1])
                    for column in (0,1,2):
                        self.assertTrue(torch.equal(torch.cat([b[column] for b in original]),torch.cat([b[column] for b in limited])))
                    self.assertEqual(torch.cat([b[2] for b in limited]).tolist(),list(range(65)))
                a=list(uncapped.train);b=list(capped.train)
                self.assertEqual([len(x[0])+len(x[1]) for x in b],[128,2])
                for column in (0,1,2):
                    self.assertTrue(torch.equal(torch.cat([x[column] for x in a]),torch.cat([x[column] for x in b])))
            finally:
                for bundle in (uncapped,capped):
                    bundle.train.shutdown()
                    for role in ('val','test','train_eval'):getattr(bundle,role).shutdown()

    def test_probability_parity_for_b36_capped_and_uncapped(self):
        _,a=fixture();_,b=fixture(64)
        torch.manual_seed(19)
        module=SteganalysisModule(model='xunet_wide_coloursep36',front_end='srm',norm='none',width=4,
            batch_size=128,val_batch_images=64).eval()
        def scores(loader):
            result={}
            with torch.inference_mode():
                for batch in loader:
                    logits,targets=module.pair_logits_targets(batch)
                    probabilities=logits.softmax(1)[:,1].numpy();ids=batch[2].tolist()
                    for index,label,p in zip(ids+ids,targets.tolist(),probabilities):result[index,label]=float(p)
            return result
        try:
            uncapped=scores(a.val);capped=scores(b.val)
            self.assertEqual(sorted(uncapped),sorted(capped));self.assertEqual(len(capped),130)
            x=np.array([uncapped[k] for k in sorted(uncapped)]);y=np.array([capped[k] for k in sorted(capped)])
            np.testing.assert_allclose(x,y,rtol=0,atol=SCORE_ATOL)
            from cvdl.metrics import probability_error
            labels=np.array([k[1] for k in sorted(capped)])
            self.assertEqual(probability_error(labels,x),probability_error(labels,y))
        finally:
            a.train.shutdown();b.train.shutdown();a.val.shutdown();b.val.shutdown()

    def test_cap_is_not_an_increase_and_records_saved_effective_value(self):
        otf,bundle=fixture(256)
        self.assertEqual(bundle.val.batch_size,64)
        self.assertEqual(otf.record['val_batch_images'],128)
        bundle.train.shutdown()
        with TemporaryDirectory() as tmp:
            s=load(environ={},_profile=machine_profile('pc',device='cpu'),batch_size=16,val_batch_images=64)
            record=s.write(tmp)
            self.assertEqual(record['values']['val_batch_images'],64)
            self.assertEqual(record['effective_val_batch_images'],16)
            m=SteganalysisModule(batch_size=16,val_batch_images=64)
            self.assertEqual(m.hparams.val_batch_images,64)
            self.assertEqual(m.hparams.effective_val_batch_images,16)
            import lightning
            path=Path(tmp)/'model.ckpt'
            torch.save(dict(state_dict=m.state_dict(),hyper_parameters=dict(m.hparams),
                **{'pytorch-lightning_version':lightning.__version__}),path)
            copy=SteganalysisModule.load_from_checkpoint(path,map_location='cpu',weights_only=False)
            self.assertEqual(dict(m.hparams),dict(copy.hparams))

    def test_shared_runtime_wires_cap_and_saves_settings_model_and_data(self):
        from cvdl import runtime
        otf,unused=fixture();unused.train.shutdown()
        otf.checkpoint_validation='standard'
        otf.corpus.save_training_plan=lambda folder:None
        with TemporaryDirectory() as tmp:
            s=load(environ={},_profile=machine_profile('pc',device='cpu'),name='cap-test',
                model='xunet_wide_coloursep36',front_end='srm',norm='none',width=4,
                batch_size=128,val_batch_images=64,post_fit_eval='none',
                overrides=['workers=0','eval_workers=0','output_root='+tmp])
            loaders,_,_=runtime.create_loaders(s,otf,None)
            bundle=runtime.create_training(s,otf,loaders)
            try:
                self.assertEqual(loaders.train.batch_size*2,128)
                self.assertEqual(loaders.val.batch_size*2,64)
                self.assertEqual(bundle.lightning_model.hparams.batch_size,128)
                self.assertEqual(bundle.lightning_model.hparams.val_batch_images,64)
                self.assertEqual(bundle.lightning_model.hparams.effective_val_batch_images,64)
                self.assertEqual(bundle.lightning_model.hparams.steps_per_epoch,len(loaders.train))
                for filename in ('run_settings.json','model_run_settings.json'):
                    record=json.loads((bundle.checkpoint_dir/filename).read_text())
                    self.assertEqual(record['effective_val_batch_images'],64)
                    self.assertEqual(record.get('values',record)['val_batch_images'],64)
                record=json.loads((bundle.checkpoint_dir/'data_run_settings.json').read_text())
                self.assertEqual(record['val_batch_images'],64)
                self.assertEqual(record['batch_images'],128)
            finally:
                loaders.train.shutdown();loaders.val.shutdown()
                for logger in bundle.trainer.loggers:
                    experiment=getattr(logger,'experiment',None)
                    if hasattr(experiment,'close'):experiment.close()
                    logger.finalize('success')

    def test_invalid_caps_and_png_are_refused(self):
        for cap in (0,1,3,-2):
            with self.assertRaises(ValueError):load(environ={},_profile=machine_profile('pc',device='cpu'),val_batch_images=cap)
            with self.assertRaises(ValueError):data.build_otf_loaders(None,batch_size=128,workers=0,eval_workers=0,pin_memory=False,val_batch_images=cap)
            with self.assertRaises(ValueError):SteganalysisModule(val_batch_images=cap)
        with self.assertRaisesRegex(ValueError,'OrderedPairLoader'):
            load(environ={},_profile=machine_profile('pc',device='cpu'),data='png',val_batch_images=64)

    def test_frozen_real_g4_shape_and_preflight_output_bound(self):
        values=top_values(json.loads((ROOT/'notebooks/experiments/srm-e50-paired-coloursep36.ipynb').read_text()))
        s=load(environ={},_profile=machine_profile('g4'),**values)
        self.assertEqual(s.batch_size,128);self.assertEqual(s.val_batch_images,64)
        self.assertEqual(s.tile_size,512);self.assertEqual(s.train_tile_size,256)
        self.assertEqual(validation_srm_elements(plain(s.values)),1509949440)
        self.assertLess(validation_srm_elements(plain(s.values)),2**31-1)
        self.assertGreater(128*90*512*512,2**31-1)
        del values['val_batch_images']
        legacy=load(environ={},_profile=machine_profile('g4'),**values)
        self.assertIsNone(validation_srm_elements(plain(legacy.values)))


if __name__=='__main__':unittest.main()
