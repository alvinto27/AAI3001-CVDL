"""Hard256 noise, immutable lists, reuse bounds, CPU replay and evaluator wiring."""
import json
from pathlib import Path
import shutil
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch
from PIL import Image

from cvdl import hardnoise as N, hardtest256 as H, standardtest as S
from cvdl.validation import csv_text
from scripts.build_hard_test_256 import publish

ROOT=Path(__file__).resolve().parents[1]


class Hard256Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.files,cls.summary=H.build(ROOT)
        cls.frame,cls.recipe,_=H.load_frozen(ROOT)
        cls.noise,cls.noise_info=N.load_noise(ROOT)

    def test_lists_rebuild_exactly(self):
        self.assertEqual(self.summary['recipe_id'],'d6d91a5a98dae856d1ae')
        for name,text in self.files.items():self.assertEqual((ROOT/H.DATA_DIR/name).read_bytes(),text.encode())
        self.assertEqual(len(self.frame),32256)
        self.assertEqual(self.frame.cell.nunique(),32)
        self.assertFalse(self.frame.pair_id.duplicated().any())

    def test_cell_types_ranges_test_only_and_unique(self):
        for cell,g in self.frame.groupby('cell'):
            self.assertEqual(len(g),1008)
            self.assertEqual(g.cover_id.nunique(),1008)
            self.assertEqual(set(g.k),{1} if '/k1/' in cell else {2})
            counts=g.groupby(['bit_mode','method','placement','payload_kind']).size()
            self.assertEqual(set(counts.index),set(H.COMBINATIONS))
            self.assertEqual(set(counts),{42})
            self.assertTrue(g.requested_bpp.between(.05,.15 if g.condition.iloc[0] in ('low','low_noisy') else .5).all())
            self.assertFalse(set(g['split'])&{'train','val'})
            self.assertEqual(set(g.tile_size),{256});self.assertEqual(set(g.grid),{'overlap_last'})
            if g.condition.iloc[0] in ('noisy','low_noisy'):
                self.assertTrue((g.tile_kv_std>g.source.map(self.recipe['noise_edges'])).all())

    def test_noise_covers_all_val_test_tiles_and_edges_use_val_only(self):
        self.assertEqual(len(self.noise['noise-val.csv']),81754)
        self.assertEqual(len(self.noise['noise-test.csv']),125057)
        H.validate_noise(self.noise,ROOT)
        for source,g in self.noise['noise-val.csv'].groupby('source'):
            self.assertEqual(self.recipe['noise_edges'][source],float(np.quantile(g.kv_std,.75,method='linear')))
        self.assertEqual(set(self.recipe['noise_edges']),{'raise','div2k','alaska','lsdir'})
        self.assertEqual(self.recipe['noise_definition']['clip'],[-3,3])
        self.assertEqual(self.recipe['noise_definition']['dtype'],'float32')

    def test_measured_pool_sizes_and_overlap_report(self):
        expected={'standard':10507,'alaska-seen':3475,'alaska-unseen':3278,'lsdir':12980}
        std,_,_=S.load_frozen(ROOT)
        for stats in self.summary['cells']:
            if stats['condition'] in ('noisy','low_noisy'):self.assertEqual(stats['pool_tiles'],expected[stats['panel']])
            self.assertEqual(stats['max_tile_use'],1)
            g=self.frame[self.frame.cell==stats['cell']]
            overlap=set(std[S.filename(stats['panel'])].cover_id)&set(g.cover_id)
            self.assertEqual(stats['overlap_standard_distinct_tiles'],len(overlap))
            self.assertGreater(len(overlap),0)

    def test_pool_rules_fail_without_adjusting_edges_or_size(self):
        for size in (1008,5000):self.assertEqual(H.pool_rule('standard',size),1)
        for size in (504,1007):self.assertEqual(H.pool_rule('alaska-seen',size),2)
        for panel,size in [('standard',1007),('lsdir',504),('alaska-seen',503),('alaska-unseen',0)]:
            with self.assertRaisesRegex(ValueError,'fails'):H.pool_rule(panel,size)
        for size in (504,670,1007,1008,3000):
            ids=H.assign_tiles(size,'alaska-seen',1,'noisy')
            counts=pd.Series(ids).value_counts()
            self.assertEqual(len(ids),1008);self.assertLessEqual(counts.max(),2)
            self.assertLessEqual(counts.max()-counts.min(),1)
            self.assertEqual(len(counts),min(size,1008))
            self.assertTrue(np.array_equal(ids,H.assign_tiles(size,'alaska-seen',1,'noisy')))

    def test_cell_assignment_and_draws_independent_of_global_rng(self):
        state=np.random.get_state()
        try:
            np.random.seed(923)
            recipe,pools,_=H.describe(ROOT)
            f=H.build_cell(pools[('standard','all')],'standard',1,'all',self.summary['recipe_sha256'])
            original=self.frame[self.frame.cell=='standard/k1/all'].reset_index(drop=True)
            self.assertEqual(csv_text(f),csv_text(original))
        finally:np.random.set_state(state)

    def test_reject_tampered_noise_inputs_lists_and_geometry(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp)
            for name in ('noise-scan.json','noise-val.csv','noise-test.csv'):shutil.copy(ROOT/H.DATA_DIR/name,folder/name)
            p=folder/'noise-test.csv';p.write_bytes(p.read_bytes()+b'\n')
            with self.assertRaisesRegex(ValueError,'SHA256 mismatch'):N.load_noise(ROOT,folder)
            with patch.object(N,'inputs',return_value={}):
                with self.assertRaisesRegex(ValueError,'input/definition'):N.load_noise(ROOT,folder)
        recipe,pools,overlaps=H.describe(ROOT)
        with tempfile.TemporaryDirectory() as tmp,patch.object(H,'describe',return_value=(recipe,pools,overlaps)):
            root=Path(tmp);out=root/H.DATA_DIR;out.mkdir(parents=True)
            for name,text in self.files.items():(out/name).write_text(text)
            p=out/'pairs-standard.csv';p.write_bytes(p.read_bytes()+b'\n')
            with self.assertRaisesRegex(ValueError,'list SHA256 mismatch'):H.load_frozen(root)
        bad=self.noise['noise-val.csv'].copy();bad.loc[0,'x']+=1
        with self.assertRaisesRegex(ValueError,'geometry'):H.validate_noise({**self.noise,'noise-val.csv':bad},ROOT)

    def test_runtime_never_writes_or_builds(self):
        with patch.object(H,'build',side_effect=AssertionError('runtime build')), \
             patch.object(N,'measure_photo',side_effect=AssertionError('runtime decode')), \
             patch.object(Path,'write_text',side_effect=AssertionError('runtime write')):
            frame,_,_=H.load_frozen(ROOT)
        self.assertEqual(len(frame),32256)

    def test_builder_check_twice_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp);publish(self.files,folder)
            publish(self.files,folder,check=True);publish(self.files,folder,check=True)
            with self.assertRaisesRegex(ValueError,'refusing'):publish(self.files,folder)
            with self.assertRaisesRegex(ValueError,'MISMATCH'):publish({**self.files,'recipe.json':'bad'},folder,check=True)

    def test_fa_counts_distinct_clean_tiles_miss_counts_all_stegos(self):
        rows=[]
        for label,scores in [(0,[.8,.8,.1]),(1,[.3,.6,.9])]:
            for cover,score in zip(['a','a','b'],scores):
                rows.append(dict(cell='a',panel='standard',k=1,condition='all',cover_id=cover,
                                 source_photo=cover,prob_stego=score,label=label))
        report=H.cell_report(pd.DataFrame(rows)).iloc[0]
        self.assertEqual(report['distinct_clean_tiles'],2)
        self.assertEqual(report['pairs'],3)
        self.assertEqual(report['FA_at_0_5'],.5)
        self.assertEqual(report['miss_at_0_5'],1/3)
        self.assertEqual(report['max_tile_use'],2)
        self.assertNotIn('P_E',report.index)
        self.assertEqual(report['threshold_origin'],'pending_validation_min_pe')
        self.assertIsNone(report['P_E_at_validation_cutoff'])

    def test_mock_hard_writer_has_expected_files_and_pending_main(self):
        sample=self.frame.groupby('cell',sort=False).head(2).reset_index(drop=True)
        def score(network,ds,**kwargs):
            clean=ds.frame.copy();clean['label']=0;clean['prob_stego']=.2
            stego=ds.frame.copy();stego['label']=1;stego['prob_stego']=.8
            return pd.concat([clean,stego],ignore_index=True)
        settings=SimpleNamespace(root=ROOT,quick=True,device='cpu',batch_size=4,eval_workers=0)
        with tempfile.TemporaryDirectory() as tmp,patch('cvdl.hardtest.score',side_effect=score), \
             patch.object(S,'StandardTestPairs',side_effect=lambda c,f:SimpleNamespace(frame=f)):
            report=H.run(settings,torch.nn.Identity(),None,sample,self.summary,tmp)
            out=Path(tmp)/'hard-test'
            self.assertEqual({p.name for p in out.iterdir()},set(H.OUTPUT_FILES))
            self.assertEqual(report['status'],'complete');self.assertEqual(report['cells'],32)
            self.assertEqual(set(report['overlap_standard']),set(H.PANELS))
            self.assertIn('overlap_standard',pd.read_csv(out/'hard_predictions.csv').columns)
            cells=pd.read_csv(out/'hard_cells.csv')
            self.assertTrue(cells.cutoff.isna().all());self.assertNotIn('P_E',cells.columns)

    def test_standard_default_wires_hard_identity_and_no_hard_separately(self):
        sample=self.frame.groupby('cell',sort=False).head(2).reset_index(drop=True)
        def score(network,ds,**kwargs):
            clean=ds.frame.copy();clean['label']=0;clean['prob_stego']=.2
            stego=ds.frame.copy();stego['label']=1;stego['prob_stego']=.8
            return pd.concat([clean,stego],ignore_index=True)
        settings=SimpleNamespace(root=ROOT,quick=True,eval_workers=0,device='cpu',batch_size=4,
            rocm_conv1_fix=False,values={},train_tile_size=256,tile_size=512,train_grid='overlap_last',source='decode')
        with tempfile.TemporaryDirectory() as tmp,patch('cvdl.models.model_from_hparams',return_value=torch.nn.Identity()), \
             patch('lsb_torch.Corpus',return_value=SimpleNamespace()), \
             patch.object(S,'StandardTestPairs',side_effect=lambda c,f:SimpleNamespace(frame=f)), \
             patch('cvdl.hardtest.score',side_effect=score), \
             patch.object(H,'load_frozen',return_value=(sample,self.recipe,self.summary)) as load, \
             patch.object(H,'run',return_value={'status':'complete'}) as run:
            folder=Path(tmp);checkpoint=folder/'best.ckpt';torch.save({'hyper_parameters':{},'state_dict':{}},checkpoint)
            record=folder/'data_run_settings.json';record.write_text('unchanged')
            before=record.read_bytes()
            with_hard=S.run(settings,checkpoint,folder)
            self.assertEqual(with_hard['hard_test']['status'],'complete')
            run.assert_called_once();load.assert_called_once()
            without=S.run(settings,checkpoint,folder,hard=False)
            self.assertEqual(without['hard_test']['status'],'disabled')
            self.assertNotEqual(with_hard['evaluation_id'],without['evaluation_id'])
            meta=json.loads((Path(with_hard['output'])/'evaluation_settings.json').read_text())
            self.assertEqual(meta['hard_recipe_sha256'],self.summary['recipe_sha256'])
            self.assertTrue(meta['hard_enabled'])
            self.assertEqual((Path(with_hard['output'])/'diagnostic_summary.csv').read_bytes(),
                             (Path(without['output'])/'diagnostic_summary.csv').read_bytes())
            with self.assertRaisesRegex(ValueError,'identity differs'):
                S.run(settings,checkpoint,folder,hard=False,output=Path(with_hard['output']),overwrite=True)
            self.assertEqual(record.read_bytes(),before)

    @unittest.skipUnless((ROOT/'Datasets/originals/raise/NEF').is_dir(),'photos not present')
    def test_cpu_replay_each_panel_zero_two_workers_exact(self):
        from lsb_torch import Corpus,RunSettings,OrderedPairLoader
        old=torch.get_num_threads();torch.set_num_threads(1)
        try:
            corpus=Corpus(ROOT,RunSettings(sources=('raise','div2k','alaska','lsdir')))
            sample=self.frame.groupby('panel',sort=False).head(2).reset_index(drop=True)
            ds=S.StandardTestPairs(corpus,sample)
            def replay(workers):
                loader=OrderedPairLoader(ds,2,workers,False)
                try:return [(c.numpy().tobytes(),s.numpy().tobytes(),i.tolist()) for c,s,i in loader]
                finally:loader.shutdown()
            self.assertEqual(replay(0),replay(2))
        finally:torch.set_num_threads(old)


class NoiseDefinitionTests(unittest.TestCase):
    def test_noise_matches_fixed_check_b_definition_on_synthetic_photo(self):
        from cvdl.data import kv_residual
        from lsb_train_grid import axes
        old=torch.get_num_threads();torch.set_num_threads(1)
        try:
            with tempfile.TemporaryDirectory() as tmp:
                photo=np.random.default_rng(1).integers(0,256,(513,530,3),dtype=np.uint8)
                Image.fromarray(photo).save(Path(tmp)/'photo.png')
                record=dict(master_path='photo.png',photo_id='synthetic',source='lsdir',source_type='png8',
                    noise='not_applicable',source_height='513',source_width='530',split='val',photo_index='1',camera_model='')
                rows,timing=N.measure_photo((tmp,record))
                ys,xs=axes(record,256,'overlap_last')
                batch=torch.from_numpy(np.stack([photo[y:y+256,x:x+256] for y in ys for x in xs])).permute(0,3,1,2).contiguous().float()
                expected=kv_residual(batch)[:,:,2:-2,2:-2].clamp(-3,3).flatten(1).std(dim=1,unbiased=False).tolist()
                self.assertEqual([r['kv_std'] for r in rows],expected)
                self.assertEqual(timing['tiles'],9)
        finally:torch.set_num_threads(old)

    def test_noise_manifest_and_scan_definition_are_frozen(self):
        frames,info=N.load_noise(ROOT)
        self.assertEqual(info['definition'],N.definition())
        self.assertEqual(info['inputs'],N.inputs(ROOT))
        self.assertEqual(info['scan']['workers'],8)
        self.assertEqual(info['scan']['photos'],9219)
        self.assertEqual(info['scan']['tiles'],206811)
        self.assertLess(info['scan']['wall_seconds'],45*60)


if __name__=='__main__':unittest.main()
