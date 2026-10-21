"""Training-only geometry, byte/payload invariants, identity and offline evaluation."""
import ast
import csv
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import torch
import cvdl
from cvdl.settings import load, validate_notebook_name
from scripts.g4.preflight import top_values
from lsb_train_grid import starts, axes, summary, crop, POLICIES, pad_crop
from lsb_mode1 import make_sample, TileSpec, OTFSeed
from lsb_batch import Config
from lsb_torch import Corpus, RunSettings, TrainingImages
from test_lsb_source import record
from test_cvdl_settings import profile

ROOT=Path(__file__).resolve().parents[1]

class TrainingGridTests(unittest.TestCase):
    def test_exact_rules_and_odd_margins(self):
        expected={512:((0,),(0,),(0,)),513:((0,1),(0,),(-255,257)),
                  1023:((0,511),(255,), (0,512)),1024:((0,512),(0,512),(0,512)),
                  1025:((0,512,513),(0,512),(-255,257,769))}
        for length,values in expected.items():
            for policy,expected_starts in zip(POLICIES,values):
                self.assertEqual(starts(length,512,policy),expected_starts)
        for size in (256,512):
            for length in (size,size+1,2*size-1,2*size,2*size+1):
                for policy in POLICIES:
                    values=starts(length,size,policy)
                    self.assertEqual(len(values),len(set(values)))
                    coverage=np.zeros(length,dtype=int)
                    for start in values:coverage[max(0,start):min(length,start+size)]+=1
                    if policy!='centre_trim':self.assertTrue((coverage>=1).all())
                    if policy!='overlap_last':self.assertTrue((coverage<=1).all())
        for value in (511,0,-1,512.,True):
            with self.assertRaises(ValueError):starts(value,512,'centre_pad')

    def test_frozen_pool_counts_and_digests(self):
        records=[]
        for source in ('raise','alaska','lsdir'):
            with (ROOT/f'data/{source}/photo_manifest.csv').open() as f:records.extend(csv.DictReader(f))
        expected={'overlap_last':dict(raise_=45674,alaska=41391,lsdir=41520),
                  'centre_trim':dict(raise_=34890,alaska=21139,lsdir=14475),
                  'centre_pad':dict(raise_=45674,alaska=41391,lsdir=41520)}
        for policy,counts in expected.items():
            counts['raise']=counts.pop('raise_')
            self.assertEqual(summary(ROOT,records,512,policy)['pool_counts'],counts)
        damaged=[dict(r) for r in records]
        next(r for r in damaged if r['split']=='train')['source_width']='2000'
        with self.assertRaisesRegex(ValueError,'digest/count'):summary(ROOT,damaged,512,'centre_pad')

    def test_pad_contains_no_payload_real_capacity_and_pair_geometry(self):
        r={**record(513,1025,kind='nef'),'photo_index':'5','split':'train'}
        photo=np.random.default_rng(9).integers(0,256,(513,1025,3),dtype=np.uint8);before=photo.copy()
        for mode in ('lowest','plane'):
            for k in (1,2,3,4):
                cfg=Config(seed=1,input='unused',output='unused',bit_mode=mode,k=k,method='replace',
                           placement='random',size={'bpp':.25})
                spec=TileSpec('centre_pad',512,0,0)
                clean,cm=make_sample(photo,r,spec,cfg,OTFSeed(0,'train',0),'clean')
                stego,sm=make_sample(photo,r,spec,cfg,OTFSeed(0,'train',0),'stego')
                real,_,pad=crop(photo,r,512,0,0,'centre_pad')
                np.testing.assert_array_equal(clean,pad_crop(real,pad))
                self.assertEqual(cm['pad'],sm['pad']);self.assertEqual(cm['real_shape'],sm['real_shape'])
                mask=pad_crop(np.ones(real.shape,dtype=np.uint8),pad).astype(bool)
                self.assertTrue((stego[~mask]==0).all());self.assertTrue((clean[~mask]==0).all())
                self.assertTrue(np.any(stego[mask]!=clean[mask]))
                a=real.shape[0]*real.shape[1]
                self.assertEqual(sm['settings']['payload_bytes'],max(1,round(.25*a/8)))
                self.assertEqual(sm['settings']['effective_bpp'],8*sm['settings']['payload_bytes']/a)
                from lsb_settings import draw_otf_settings
                from lsb_engine import embed_fast
                kwargs,_=draw_otf_settings(cfg,real,np.random.default_rng(np.random.SeedSequence(sm['settings_seed_key'])))
                kwargs['rng']=np.random.default_rng(np.random.SeedSequence(sm['embedding_seed_key']))
                expected,_,_=embed_fast(real,**kwargs)
                np.testing.assert_array_equal(stego,pad_crop(expected,pad))
                again,am=make_sample(photo,r,spec,cfg,OTFSeed(0,'train',0),'stego')
                np.testing.assert_array_equal(again,stego);self.assertEqual(am,sm)
        np.testing.assert_array_equal(photo,before)
        with self.assertRaisesRegex(ValueError,'training-only'):
            make_sample(photo,r,spec,cfg,OTFSeed(0,'test',0),'stego')

    def test_uniform_source_rules_and_sampler(self):
        records=[]
        for i,source in enumerate(('raise','lsdir','alaska')):
            records.append({**record(513,1025,kind='nef'),'photo_index':str(i),'photo_id':source,
                            'source':source,'split':'train','camera_model':'test','grid_policy':'trim512' if source!='raise' else 'overlap_last'})
        settings=RunSettings(sources=('raise','lsdir','alaska'),source_mix={'raise':.4,'lsdir':.2,'alaska':.4},
                             tiles_per_epoch=10,settings_yaml=str(ROOT/'configs/train-k1to4.yaml'))
        for policy in POLICIES:
            corpus=Corpus(ROOT,settings,records=records,train_grid=policy)
            with tempfile.TemporaryDirectory() as folder:
                corpus.save_training_plan(folder)
                digest=hashlib.sha256((Path(folder)/'train_grid_plan.jsonl').read_bytes()).hexdigest()
                self.assertEqual(digest,corpus.grid_record['sha256'])
                self.assertEqual(json.loads((Path(folder)/'train_grid_summary.json').read_text()),corpus.grid_record)
            for r in records:self.assertEqual(corpus.training_axes(r),axes(r,512,policy))
            groups,repeats=corpus.train_tile_schedule(0)
            counts={source:sum(len(tiles) for r,tiles in groups if r['source']==source) for source in settings.sources}
            self.assertEqual(counts,{'raise':4,'lsdir':2,'alaska':4})
            self.assertEqual(len(TrainingImages(corpus)),20)
            self.assertEqual(corpus.train_tile_schedule(0)[1],repeats)

    def test_notebook_names_recipe_and_post_fit_guards(self):
        base=json.loads((ROOT/'notebooks/newdata-1.ipynb').read_text())
        # The tiling recipe checks apply only to the tiling notebooks. Other experiments (srm, srm-e50, ...) have
        # their own recipes and tests; they still must satisfy the notebook-name rule below.
        tiling=[ROOT/f'notebooks/experiments/tiling-{arm}{suffix}.ipynb'
                for arm in ('overlap','trim','pad') for suffix in ('','-256')]
        self.assertEqual(sorted(p.stem for p in tiling if p.exists()),sorted(p.stem for p in tiling))
        for path in sorted((ROOT/'notebooks/experiments').glob('*.ipynb')):
            nb=json.loads(path.read_text());values=top_values(nb)
            validate_notebook_name(path,values)
            self.assertEqual(values['name'],path.stem)
            if path not in tiling:continue
            self.assertEqual(values['name'],path.stem);self.assertEqual(values['post_fit_eval'],'none')
            self.assertEqual(values['train_yaml'],'configs/train-k1to4.yaml')
            recipe=top_values(base);recipe.pop('run_name')
            ignored={'name','train_grid','post_fit_eval'}
            if path.stem.endswith('-256'):
                self.assertEqual((values['train_tile_size'],values['batch_size'],values['tiles_per_epoch']),(256,128,180000))
                # Same pixels per epoch and per optimiser step as the 512 arm.
                self.assertEqual(values['tiles_per_epoch']*256**2,45000*512**2)
                self.assertEqual(values['batch_size']*256**2,32*512**2)
                ignored|={'train_tile_size','batch_size','tiles_per_epoch'}
                recipe={k:v for k,v in recipe.items() if k not in ignored}
            else:
                self.assertNotIn('train_tile_size',values)
                self.assertEqual((values['batch_size'],values['tiles_per_epoch']),(32,45000))
            self.assertEqual({k:v for k,v in values.items() if k not in ignored},recipe)
            self.assertEqual(nb['cells'][2:],base['cells'][2:])
            resolved=load(environ={},_profile=profile(),**values)
            self.assertEqual(resolved.run_name,path.stem)
            with self.assertRaisesRegex(ValueError,'stem'):validate_notebook_name(path,{**values,'name':'wrong'})
        codes=[ast.parse(''.join(c['source'])) for c in base['cells'] if c['cell_type']=='code']
        for tree in codes[-4:]:
            self.assertIsInstance(tree.body[0],ast.If)
            self.assertEqual(ast.unparse(tree.body[0].test),'settings.post_fit_eval == \'full\'')
            # No unguarded calls into prediction/panels/breakdown/diagnostics.
            self.assertEqual(len(tree.body),1)
        default=load(environ={},_profile=profile())
        self.assertEqual((default.train_grid,default.post_fit_eval),('recorded','full'))

    def test_tile256_summary_pools_and_digests(self):
        records=[]
        for source in ('raise','alaska','lsdir'):
            with (ROOT/f'data/{source}/photo_manifest.csv').open() as f:records.extend(csv.DictReader(f))
        expected=json.loads((ROOT/'data/train-grids/v1/summary-256.json').read_text())
        self.assertEqual(expected['inputs'],json.loads((ROOT/'data/train-grids/v1/summary.json').read_text())['inputs'])
        for policy in POLICIES:
            self.assertEqual(summary(ROOT,records,256,policy),expected['plans'][policy])
        self.assertEqual(expected['plans']['overlap_last']['pool_counts'],{'raise':170356,'alaska':121704,'lsdir':140863})
        damaged=[dict(r) for r in records];next(r for r in damaged if r['split']=='train')['source_width']='2000'
        with self.assertRaisesRegex(ValueError,'digest/count'):summary(ROOT,damaged,256,'overlap_last')

    def test_train_tile_256_is_training_only(self):
        from lsb_torch import FixedPairs
        records=[{**record(649,777,kind='nef'),'photo_index':str(i),'photo_id':f'{source}-{split}','source':source,
                  'split':split,'camera_model':'test','grid_policy':'overlap_last'}
                 for i,(source,split) in enumerate((s,p) for s in ('raise','lsdir','alaska') for p in ('train','val','test'))]
        settings=RunSettings(sources=('raise','lsdir','alaska'),source_mix={'raise':.4,'lsdir':.2,'alaska':.4},
                             tiles_per_epoch=10,settings_yaml=str(ROOT/'configs/train-k1to4.yaml'))
        base=Corpus(ROOT,settings,records=records)
        corpus=Corpus(ROOT,settings,records=records,train_grid='overlap_last',train_tile=256)
        self.assertEqual(settings.tile_size,512);self.assertEqual(corpus.grid_record['tile_size'],256)
        self.assertEqual(corpus.run_record['train_tile_size'],256)
        self.assertNotIn('train_tile_size',base.run_record)
        train=next(r for r in records if r['split']=='train')
        self.assertEqual(corpus.training_axes(train),axes(train,256,'overlap_last'))
        self.assertEqual(len(corpus.training_axes(train)[0])*len(corpus.training_axes(train)[1]),3*4)
        groups,_=corpus.train_tile_schedule(0)
        self.assertEqual(sum(len(t) for _,t in groups),10)
        photo=np.random.default_rng(1).integers(0,256,(649,777,3),dtype=np.uint8)
        for r in records[:1]:
            clean,meta=make_sample(photo,r,TileSpec('train_overlap_v1',256,2,3),corpus.config,OTFSeed(0,'train',0),'stego')
            self.assertEqual(clean.shape,(256,256,3));self.assertEqual(meta['tile_size'],256)
        for split in ('val','test'):
            one,two=FixedPairs(base,split),FixedPairs(corpus,split)
            self.assertEqual(one.frame.to_csv(index=False),two.frame.to_csv(index=False))
            self.assertTrue((two.frame.tile_size==512).all())
        with self.assertRaisesRegex(ValueError,'explicit train_grid'):Corpus(ROOT,settings,records=records,train_tile=256)
        with self.assertRaisesRegex(ValueError,'invalid train tile'):Corpus(ROOT,settings,records=records,train_grid='overlap_last',train_tile=128)
        with self.assertRaises(ValueError):RunSettings(sources=('raise','lsdir'),tile_size=256)  # eval guard unchanged

    def test_train_tile_size_setting(self):
        self.assertIsNone(load(environ={},_profile=profile()).train_tile_size)
        self.assertEqual(load(environ={},_profile=profile(),train_grid='overlap_last',train_tile_size=256).train_tile_size,256)
        for bad in (dict(train_tile_size=256),dict(train_grid='overlap_last',train_tile_size=128)):
            with self.assertRaises(ValueError):load(environ={},_profile=profile(),**bad)

    def test_named_notebook_still_validates_yaml_before_launch(self):
        with self.assertRaisesRegex(ValueError,'invalid train_yaml'):
            load(name='tiling-pad',train_yaml='configs/missing.yaml',environ={},_profile=profile())

    def test_name_mismatch_rejected_by_preflight_and_runner(self):
        from scripts.g4.preflight import preflight
        from scripts.run_notebook import run
        from types import SimpleNamespace
        nb=json.loads((ROOT/'notebooks/experiments/tiling-pad.ipynb').read_text())
        with patch('scripts.g4.preflight.validate_notebook',return_value=nb):
            with self.assertRaisesRegex(ValueError,'stem'):
                preflight('notebooks/experiments/wrong.ipynb','HEAD')
        with tempfile.TemporaryDirectory(dir=ROOT/'runs/.scratch') as directory:
            parent=Path(directory)/'experiments';parent.mkdir();path=parent/'wrong.ipynb'
            path.write_text(json.dumps(nb))
            args=SimpleNamespace(notebook=path,env_file=Path('/dev/null'),set=[],resume=False,
                                 output=Path(directory)/'output',host_monitor='off')
            with self.assertRaisesRegex(ValueError,'stem'):run(args)

    def test_none_executes_no_post_fit_work(self):
        from types import SimpleNamespace
        nb=json.loads((ROOT/'notebooks/experiments/tiling-pad.ipynb').read_text())
        codes=[c for c in nb['cells'] if c['cell_type']=='code']
        namespace={'settings':SimpleNamespace(post_fit_eval='none')}
        for cell in codes[-4:]:exec(''.join(cell['source']),namespace)
        self.assertNotIn('best_model',namespace)
        self.assertNotIn('predictions',namespace)

    def test_eval_frames_do_not_follow_training_grid(self):
        from lsb_torch import FixedPairs
        records=[{**record(649,777,kind='nef'),'photo_index':str(i),'photo_id':split,
                  'split':split,'camera_model':'test'} for i,split in enumerate(('train','val','test'))]
        settings=RunSettings(sources=('raise',))
        before=Corpus(ROOT,settings,records=records)
        for policy in POLICIES:
            after=Corpus(ROOT,settings,records=records,train_grid=policy)
            for split in ('val','test'):
                self.assertEqual(FixedPairs(before,split).frame.to_csv(index=False),
                                 FixedPairs(after,split).frame.to_csv(index=False))



    def test_offline_settings_rebase_uses_pc_workers_and_quick(self):
        from scripts.evaluate_run import settings_for_run
        with tempfile.TemporaryDirectory() as directory:
            folder=Path(directory)
            s=load(environ={},_profile=profile(),name='tiling-pad',post_fit_eval='none')
            record={'values':{**dict(s.values),'quick':True,'eval_workers':16,'job_id':'tiling-pad-20260101T000000Z-0001'}}
            from cvdl.settings import plain
            (folder/'run_settings.json').write_text(json.dumps(plain(record)))
            with patch('cvdl.settings.automatic_profile',return_value=profile()):resolved=settings_for_run(folder)
            self.assertTrue(resolved.quick);self.assertEqual(resolved.eval_workers,profile().values['eval_workers'])
            self.assertEqual(resolved.job_id,record['values']['job_id'])

if __name__=='__main__':unittest.main()
