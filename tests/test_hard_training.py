"""Opt-in hard training draws; default byte streams and evaluation stay frozen."""
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np

import cvdl
from lsb_batch import read_config,write_config
from lsb_settings import draw_otf_settings,draw_legacy_settings
from lsb_torch import Corpus,RunSettings,FixedPairs
from scripts.g4.preflight import top_values
from test_lsb_source import record

ROOT=Path(__file__).resolve().parents[1]

class HardTrainingTests(unittest.TestCase):
    def setUp(self):
        self.base=read_config(ROOT/'configs/train-k1to4.yaml')
        self.hard=read_config(ROOT/'configs/train-hard-v1.yaml')
        self.image=np.zeros((256,256,3),dtype=np.uint8)

    def test_default_draws_and_rng_state_match_pre_feature_fingerprints(self):
        expected=('cc13ae2f203fa9366fa0a2559ca20d7b8a6e6684951b5cabf3e2527dddd20e04',
                  'b7b50794746ce61a30a738dbd789ed373b863c845d12629c4ca29eec3e6ef8f5')
        self.assertNotIn('k_weights',self.base.as_dict())
        self.assertNotIn('bpp_mixture',self.base.as_dict())
        for draw,digest in zip((draw_legacy_settings,draw_otf_settings),expected):
            rows=[]
            for seed in range(100):
                rng=np.random.default_rng(seed);kw,meta=draw(self.base,self.image,rng);kw.pop('rng')
                rows.append(dict(kwargs=kw,metadata=meta,next=rng.integers(0,2**31,size=8).tolist()))
            actual=hashlib.sha256(json.dumps(rows,sort_keys=True,separators=(',',':')).encode()).hexdigest()
            self.assertEqual(actual,digest)

    def test_only_the_explicit_sampling_fields_change_in_training_config(self):
        a=self.base.as_dict();b=self.hard.as_dict()
        self.assertEqual({k:a.get(k) for k in a.keys()|b.keys() if a.get(k)!=b.get(k)},
                         {'k_weights':None,'bpp_mixture':None})
        self.assertEqual(self.hard.k_weights,{1:3.,2:1.,3:1.,4:1.})
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'roundtrip.yaml';write_config(path,self.hard)
            self.assertEqual(read_config(path).as_dict(),self.hard.as_dict())

    def test_invalid_k_weights_are_rejected(self):
        for value in ({1:1},{1:0,2:0,3:0,4:0},{1:-1,2:1,3:1,4:1},
                      {1:True,2:1,3:1,4:1},{1:float('nan'),2:1,3:1,4:1},
                      {'1':3,2:1,3:1,4:1},[3,1,1,1]):
            with self.subTest(value=value),self.assertRaises(ValueError):replace(self.base,k_weights=value).normalized()
        with self.assertRaises(ValueError):replace(self.hard,k=[1,1,2,3,4]).normalized()

    def test_invalid_bpp_components_are_rejected(self):
        for value in ([],[{'weight':0,'bpp':[.05,.15]}],
                      [{'weight':True,'bpp':[.05,.15]}],
                      [{'weight':1,'bpp':[.01,.15]}],
                      [{'weight':1,'bpp':[.05,.6]}],
                      [{'weight':1,'bpp':[.15,.05]}],
                      [{'weight':1,'range':[.05,.15]}]):
            with self.subTest(value=value),self.assertRaises(ValueError):replace(self.base,bpp_mixture=value).normalized()
        with self.assertRaises(ValueError):replace(self.hard,size={'payload_bytes':[1,100]}).normalized()

    def test_weighted_draw_frequencies_and_unchanged_other_choices(self):
        rng=np.random.default_rng(2041);rows=[]
        for _ in range(5000):
            kwargs,meta=draw_otf_settings(self.hard,self.image,rng);rows.append((kwargs,meta))
        def share(test):return sum(test(kw,meta) for kw,meta in rows)/len(rows)
        for k,expected in ((1,.5),(2,1/6),(3,1/6),(4,1/6)):
            self.assertAlmostEqual(share(lambda kw,m:kw['k']==k),expected,delta=.03)
        self.assertAlmostEqual(share(lambda kw,m:m['bpp_mixture_component']==0),.5,delta=.03)
        for lo,hi,expected in ((.05,.10,11/36),(.10,.15,11/36),(.15,.35,2/9),(.35,.50,1/6)):
            self.assertAlmostEqual(share(lambda kw,m:lo<=m['requested_bpp']<hi),expected,delta=.03)
        for field,levels in (('method',('replace','match')),('placement',('contiguous','stride','random')),
                             ('payload',('random_bytes','letters')),('bit_mode',('lowest','plane'))):
            for level in levels:self.assertAlmostEqual(share(lambda kw,m:kw[field]==level),1/len(levels),delta=.03)
        for kw,meta in rows:
            self.assertGreaterEqual(meta['requested_bpp'],.05);self.assertLessEqual(meta['requested_bpp'],.5)
            self.assertLessEqual(meta['effective_bpp'],meta['capacity_payload_bytes']*8/(256**2))

    def corpora(self):
        records=[]
        for source in ('raise','div2k','alaska','lsdir'):
            for split in ('train','val','test','test_seen','test_unseen'):
                records.append({**record(1024,1024,kind='nef'),'photo_index':str(len(records)),
                    'photo_id':source+'-'+split,'source':source,'split':split,'camera_model':'test',
                    'grid_policy':'trim512' if source in ('alaska','lsdir') else 'overlap_last'})
        s=RunSettings(sources=('raise','alaska','lsdir'),settings_yaml=str(ROOT/'configs/train-k1to4.yaml'),
            source_mix={'raise':1/3,'alaska':1/3,'lsdir':1/3},tiles_per_epoch=18)
        return tuple(Corpus(ROOT,replace(s,settings_yaml=str(ROOT/'configs'/name)),records=records,
                            train_grid='overlap_last',train_tile=256)
                     for name in ('train-k1to4.yaml','train-hard-v1.yaml'))

    def test_exact_source_quotas_without_pool_cycle_and_recorded_weights(self):
        base,hard=self.corpora();groups,repeats=hard.train_tile_schedule(0)
        self.assertEqual(repeats,{'raise':0,'alaska':0,'lsdir':0})
        for source in hard.settings.sources:
            keys=[(r['photo_id'],*tile) for r,tiles in groups if r['source']==source for tile in tiles]
            self.assertEqual(len(keys),6);self.assertEqual(len(set(keys)),6)
        self.assertNotIn('training_sampling',base.run_record)
        saved=hard.run_record['training_sampling'];self.assertEqual(saved['config'],self.hard.as_dict())
        self.assertEqual(saved['config_sha256'],hashlib.sha256((ROOT/'configs/train-hard-v1.yaml').read_bytes()).hexdigest())

    def test_train_yaml_cannot_change_any_validation_or_test_pair_metadata(self):
        base,hard=self.corpora()
        for purpose,sources in (('val',('raise','div2k','alaska','lsdir')),
            ('test',('raise','div2k','lsdir')),('test_seen',('alaska',)),('test_unseen',('alaska',))):
            a=FixedPairs(base,purpose,sources=sources).frame.to_csv(index=False).encode()
            b=FixedPairs(hard,purpose,sources=sources).frame.to_csv(index=False).encode()
            self.assertEqual(hashlib.sha256(a).hexdigest(),hashlib.sha256(b).hexdigest())
        # Changing training YAML does not alter the frozen evaluation config either.
        self.assertEqual(base.evaluation_config.as_dict(),hard.evaluation_config.as_dict())

    def test_notebook_changes_only_approved_values(self):
        base=top_values(json.loads((ROOT/'notebooks/experiments/srm.ipynb').read_text()))
        hard=top_values(json.loads((ROOT/'notebooks/experiments/srm-hard.ipynb').read_text()))
        self.assertEqual({k for k in base.keys()|hard.keys() if base.get(k)!=hard.get(k)},
                         {'name','tiles_per_epoch','source_mix','train_yaml',
                          'checkpoint_validation','validation_recipe','val_cap_raise','val_cap_lsdir'})
        self.assertEqual((hard['epochs'],hard['lr_drops'],hard['batch_size'],hard['front_end']),
                         (30,((5,.3),(20,.1)),128,'srm'))
        self.assertEqual(hard['tiles_per_epoch'],216000)
        self.assertTrue(all(round(216000*v)==72000 for v in hard['source_mix'].values()))


if __name__=='__main__':unittest.main()
