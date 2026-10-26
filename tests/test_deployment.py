import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'LSB-Dataset-Generator')]
import json
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
import pandas as pd
from cvdl import deployment as C, standardtest as S, hardtest256 as H, validation as V
from cvdl.metrics import threshold,rates


def scores(frame=None):
    if frame is None:
        frame=C.load_recipe(ROOT,'60da4391043e8ce48ff2')[0].groupby('source',sort=True).head(2).copy()
    clean=frame.copy();clean['label']=0;clean['prob_stego']=.2
    stego=frame.copy();stego['label']=1;stego['prob_stego']=.8
    return pd.concat([clean,stego],ignore_index=True)


def mock_resolution():
    frame=C.load_recipe(ROOT,'60da4391043e8ce48ff2')[0].groupby('source',sort=True).head(2).reset_index(drop=True)
    return frame,dict(recipe_id='60da4391043e8ce48ff2',threshold_origin='quick_post_hoc_256',quick=True,deployment=False)


class CalibrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.frame=scores();cls.identity=dict(recipe_id='test',threshold_origin='post_hoc_256',checkpoint_sha256='test')

    def test_analytical_optimum_and_test_change_independence(self):
        a,refs,_=C.fit(self.frame,self.identity)
        self.assertEqual(a['cutoff'],.8);self.assertEqual(a['P_E_at_validation_cutoff'],0)
        self.assertEqual(set(refs.source),{'raise','alaska','lsdir'})
        test=self.frame.copy();test['split']='test';test['prob_stego']=.99
        self.assertEqual(C.fit(self.frame,self.identity)[0],a)
        with self.assertRaisesRegex(ValueError,'val rows'):C.fit(test,self.identity)

    def test_ties_saturation_and_float64_round_trip(self):
        f=self.frame.copy();f['prob_stego']=np.float32(1.)
        a,_,text=C.fit(f,self.identity)
        self.assertEqual(a['cutoff'],np.nextafter(1.,np.inf));self.assertEqual(a['FA'],0);self.assertEqual(a['miss'],1)
        with tempfile.TemporaryDirectory() as t:
            C.write(t,f,self.identity);saved=json.loads((Path(t)/'calibration.json').read_text())
            replay=pd.read_csv(Path(t)/'predictions.csv',float_precision='round_trip')
            self.assertEqual(saved['cutoff'],a['cutoff']);self.assertEqual(rates(replay,saved['cutoff'])['fa'],0)
            self.assertEqual(saved['score_csv_sha256'],V.digest(text.encode()))

    def test_bad_labels_scores_and_pair_duplicates(self):
        for mutate in ('nan','label','duplicate'):
            f=self.frame.copy()
            if mutate=='nan':f.loc[0,'prob_stego']=np.nan
            elif mutate=='label':f.loc[0,'label']=3
            else:f=pd.concat([f,f.iloc[:1]],ignore_index=True)
            with self.assertRaises(ValueError):C.fit(f,self.identity)

    def test_pooled_not_mean_source_minimum(self):
        # Majority source supports .8, minority supports .4; source-specific optimal cuts differ.
        f=pd.DataFrame(dict(label=[0]*10+[1]*10+[0,1],prob_stego=[.6]*10+[.8]*10+[.2,.4],k=1,source=['raise']*20+['alaska']*2))
        t,_=threshold(f,'pe');self.assertEqual(t,.8)
        self.assertGreater(rates(f,t)['pe_at_cutoff'],0)
        self.assertEqual(np.mean([rates(g,threshold(g,'pe')[0])['pe_at_cutoff'] for _,g in f.groupby('source')]),0)

    def test_every_standard_group_and_hard_cell_share_threshold(self):
        a,refs,_=C.fit(self.frame,self.identity)
        test=self.frame.copy();test['split']='test';test['panel']='synthetic'
        d=S.diagnostic_rows(test);d.insert(0,'panel','synthetic')
        m=C.main_rows(test,d,a)
        self.assertEqual(set(m.cutoff),{a['cutoff']});self.assertEqual(set(m.threshold_id),{a['threshold_id']})
        test['cell']='synthetic/1/all';test['condition']='all'
        h=H.cell_report(test,a)
        self.assertEqual(set(h.cutoff),{a['cutoff']});self.assertEqual(set(h.threshold_id),{a['threshold_id']})
        self.assertEqual(h.iloc[0].FA,0);self.assertEqual(h.iloc[0].miss,0)
        self.assertTrue(h.iloc[0].FA_ci95_high>0)

    def test_no_div2k_reference_or_mixed_reference(self):
        f=self.frame.copy();f.loc[f.source=='alaska','source']='div2k'
        a,refs,_=C.fit(f,self.identity)
        self.assertNotIn('div2k',set(refs.source))
        r=C.reference_rows(f,refs,panel='standard')
        self.assertNotIn('div2k',set(r.source));self.assertTrue((r.threshold_origin=='if_source_known_reference').all())
        self.assertNotIn('overall',set(r.source))
        empty=C.reference_rows(f[f.source=='div2k'],refs,panel='standard')
        self.assertTrue(empty.empty);self.assertIn('cutoff',empty.columns)
        with tempfile.TemporaryDirectory() as t:
            empty.to_csv(Path(t)/'reference.csv',index=False)
            self.assertTrue(pd.read_csv(Path(t)/'reference.csv').empty)

    def test_mapping_is_pinned_and_exact_roster(self):
        entries=C.mapping(ROOT);self.assertEqual(len(entries),14)
        self.assertEqual(sum(e['recipe_id']=='60da4391043e8ce48ff2' for e in entries.values()),13)
        # Removing only learn-SRM must reproduce the old thirteen-entry mapping exactly.
        raw=(ROOT/C.MAPPING).read_bytes()
        old=json.loads(raw);appended=old['entries'].pop()
        old_raw=(V.canonical(old)+'\n').encode()
        self.assertEqual(V.digest(old_raw),'d233ee8e0ac04af136f9fc92cc920b939ec714b5462e3b98580a1b2ca0a04a6b')
        prefix=old_raw[:old_raw.index(b'],"version"')]
        self.assertEqual(raw[:len(prefix)],prefix)
        self.assertEqual(appended['checkpoint_sha256'],'6201ad87013e5ddfffdec4a08e4d683d67cc7e0231c8314b30cbced6195d04e4')
        baseline=entries['bf64848a49bdc6e20b7fcdba47cb8eee0ed247b909a928cc45e4a4102db9f076']
        paired=entries['b00cd4b6d1f4c3648580691c9cf0e2a4bd83c55740a3cf2a7a561a8adf12f180']
        self.assertEqual(paired,dict(baseline,run_name='srm-e50-paired-20261023T022339Z-4518',
            checkpoint_file='best-epoch39-val_pe0.0444-val_auc0.9917.ckpt',
            checkpoint_sha256='b00cd4b6d1f4c3648580691c9cf0e2a4bd83c55740a3cf2a7a561a8adf12f180'))
        curriculum=entries['14c8e05c3d476912dd5540970060d3634b3a72b7d375ba5dad5ccafd50905c3a']
        self.assertEqual(curriculum,dict(baseline,run_name='srm-e50-curriculum50-paired-20261023T090435Z-3325',
            checkpoint_file='best-epoch39-val_pe0.0419-val_auc0.9922.ckpt',
            checkpoint_sha256='14c8e05c3d476912dd5540970060d3634b3a72b7d375ba5dad5ccafd50905c3a'))
        b36=entries['2f45fb4d20203d690a6a074ac3cbf8d45dae08a82713f0a504148b3ca450f0d5']
        self.assertEqual(b36,dict(baseline,run_name='srm-e50-paired-coloursep36-20261024T115642Z-1325',
            checkpoint_file='best-epoch48-val_pe0.0572-val_auc0.9860.ckpt',
            checkpoint_sha256='2f45fb4d20203d690a6a074ac3cbf8d45dae08a82713f0a504148b3ca450f0d5'))
        learned=entries['6201ad87013e5ddfffdec4a08e4d683d67cc7e0231c8314b30cbced6195d04e4']
        self.assertEqual(learned,dict(baseline,run_name='srm-e50-paired-learn-srm-20261026T022114Z-0923',
            checkpoint_file='best-epoch40-val_pe0.0538-val_auc0.9884.ckpt',
            checkpoint_sha256='6201ad87013e5ddfffdec4a08e4d683d67cc7e0231c8314b30cbced6195d04e4'))
        with tempfile.TemporaryDirectory() as t:
            p=Path(t)/C.MAPPING;p.parent.mkdir();p.write_bytes((ROOT/C.MAPPING).read_bytes()+b' ')
            with self.assertRaisesRegex(ValueError,'mapping SHA'):C.mapping(t)

    def test_unmapped_fails_no_borrowed_recipe_fallback(self):
        settings=SimpleNamespace(root=ROOT,quick=False,validation_recipe='60da4391043e8ce48ff2')
        with tempfile.TemporaryDirectory() as t:
            with self.assertRaisesRegex(ValueError,'no fallback'):C.resolve(settings,Path(t)/'best.ckpt','unmapped')

    def test_mapping_resolution_and_quick_distinct(self):
        sha,entry=next(iter(C.mapping(ROOT).items()))
        with tempfile.TemporaryDirectory() as t:
            s=SimpleNamespace(root=ROOT,quick=False)
            f,identity=C.resolve(s,Path(t)/'best.ckpt',sha)
            self.assertEqual(len(f),18000);self.assertEqual(identity['recipe_id'],entry['recipe_id'])
            s.quick=True;q,qi=C.resolve(s,Path(t)/'best.ckpt',sha)
            self.assertLess(len(q),len(f));self.assertNotEqual(qi['selected_pairs_sha256'],identity['selected_pairs_sha256'])
            self.assertFalse(qi['deployment']);self.assertEqual(qi['threshold_origin'],'quick_post_hoc_256')

    def test_saved_recipe_precedes_mapping_and_is_checked(self):
        with tempfile.TemporaryDirectory() as t:
            p=Path(t)/'best.ckpt';(p.parent/'run_settings.json').write_text(json.dumps({'values':{'validation_recipe':'bf7ede374cab01f73f5d'}}))
            s=SimpleNamespace(root=ROOT,quick=False,validation_recipe='bf7ede374cab01f73f5d',
                train_grid='overlap_last',train_tile_size=256,to_run_settings=lambda:None)
            frame,recipe,summary,text=C.load_recipe(ROOT,s.validation_recipe)
            with patch.object(C,'mapping',side_effect=AssertionError('must not map new run')),patch.object(V,'load_frozen',return_value=(frame,recipe,summary,text)):
                f,i=C.resolve(s,p,'new');self.assertEqual(i['threshold_origin'],'train_matched');self.assertEqual(len(f),21600)
                (p.parent/'validation_pairs.csv').write_text('changed')
                with self.assertRaisesRegex(ValueError,'saved validation identity'):C.resolve(s,p,'new')

    def test_validation_test_photo_disjoint(self):
        frames,_,_=S.load_frozen(ROOT, higher_k=True)
        C.assert_disjoint(self.frame,list(frames.values()))
        with self.assertRaisesRegex(ValueError,'photo overlap'):C.assert_disjoint(self.frame,[self.frame])

    def test_frozen_wide_counts_and_tampering(self):
        f,r,s,_=C.load_recipe(ROOT,'8328cc6b0673cade0312')
        self.assertEqual(f.source.value_counts().to_dict(),{'raise':15239,'div2k':2761})
        evidence=json.loads((ROOT/'data/val/8328cc6b0673cade0312/training-evidence.json').read_text())
        self.assertEqual(evidence['sources']['raise']['training_pairs'],45674)
        self.assertEqual(evidence['sources']['div2k']['training_pairs'],8276)
        with patch.object(V,'describe',return_value=({**r,'inputs':{}},{},None,None,None)):
            with self.assertRaises(ValueError):C.load_recipe(ROOT,'8328cc6b0673cade0312')

    def test_mapping_list_pin_rejects_self_consistent_replacement(self):
        sha,entry=next(iter(C.mapping(ROOT).items()))
        frame,recipe,summary,text=C.load_recipe(ROOT,entry['recipe_id'])
        with tempfile.TemporaryDirectory() as t,patch.object(C,'mapping',return_value={sha:entry}), \
             patch.object(C,'load_recipe',return_value=(frame,recipe,{**summary,'pairs_sha256':'changed'},text)):
            with self.assertRaisesRegex(ValueError,'differs from frozen checkpoint mapping'):
                C.resolve(SimpleNamespace(root=ROOT,quick=False),Path(t)/'best.ckpt',sha)

    def test_offline_rebuild_exact_and_training_records_unchanged(self):
        from scripts.build_calibration import build,RUNS
        paths=[p for run in RUNS for p in (ROOT/'runs'/run/'checkpoints').rglob('*') if p.is_file() and (p.suffix=='.ckpt' or p.name.endswith('settings.json') or p.name=='validation_pairs.csv')]
        before={p:V.digest(p.read_bytes()) for p in paths}
        files,report=build(ROOT)
        for path,text in files.items():self.assertEqual(path.read_bytes(),text.encode())
        self.assertEqual({p:V.digest(p.read_bytes()) for p in paths},before)
        self.assertEqual(report['training_counts'],{'raise':45674,'div2k':8276})

    @unittest.skipUnless((ROOT/'Datasets/originals/raise/NEF').is_dir(),'photos not present')
    def test_wide_real_cpu_replay_zero_two_workers(self):
        import torch
        from lsb_torch import Corpus,RunSettings,OrderedPairLoader
        old=torch.get_num_threads();torch.set_num_threads(1)
        try:
            frame=C.load_recipe(ROOT,'8328cc6b0673cade0312')[0].groupby('source').head(1).reset_index(drop=True)
            dataset=V.MatchedValidationPairs(Corpus(ROOT,RunSettings(sources=('raise','div2k'))),frame)
            def replay(workers):
                loader=OrderedPairLoader(dataset,2,workers,False)
                try:return [(c.numpy().tobytes(),s.numpy().tobytes(),i.tolist()) for c,s,i in loader]
                finally:loader.shutdown()
            self.assertEqual(replay(0),replay(2))
        finally:torch.set_num_threads(old)

    def test_hard_reuse_denominators_at_global_cutoff(self):
        f=self.frame[self.frame.source=='raise'].copy();f['cell']='one';f['panel']='standard';f['condition']='all'
        extra=f[f.cover_id==f.cover_id.iloc[0]].copy();extra['pair_id']+='repeat';f=pd.concat([f,extra])
        f['prob_stego']=.9
        a=dict(cutoff=.8,threshold_origin='post_hoc_256',threshold_id='global')
        r=H.cell_report(f,a).iloc[0]
        self.assertEqual(r.distinct_clean_tiles,2);self.assertEqual(r.pairs,3)
        self.assertEqual(r.FA_events,2);self.assertEqual(r.miss_events,0)

if __name__=='__main__':unittest.main()
