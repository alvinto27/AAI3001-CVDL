"""Frozen standard 256 suite: metadata, CPU replay and explicit legacy CLI compatibility."""
import hashlib
import io
import json
from pathlib import Path
import shutil
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch

from cvdl import standardtest as S
from cvdl.validation import canonical, digest, csv_text
from scripts import evaluate_run as er
from scripts.build_standard_test import publish

ROOT = Path(__file__).resolve().parents[1]


class StandardTestTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.files, cls.summary = S.build(ROOT)
        cls.frames, cls.recipe, _ = S.load_frozen(ROOT, higher_k=True)

    def test_exact_rebuild_all_bytes(self):
        for name, text in self.files.items():
            self.assertEqual((ROOT/S.DATA_DIR/name).read_bytes(), text.encode())
        self.assertEqual(self.summary['recipe_id'], '5045c1076ab3f540a61c')
        self.assertEqual(self.summary['recipe_sha256'], digest(canonical(self.recipe).encode()))

    def test_exact_panel_source_k_type_quotas(self):
        for panel, sources in S.PANELS.items():
            for higher in (False, True):
                frame = self.frames[S.filename(panel, higher)]
                self.assertEqual(len(frame), sum(n for _, n in sources.values()))
                for source, (_, n) in sources.items():
                    group = frame.query('source == @source')
                    self.assertEqual(len(group), n)
                    combos = group.groupby(['k','bit_mode','method','placement','payload_kind']).size()
                    self.assertEqual(len(combos), 96)
                    self.assertEqual(set(combos), {n//96})

    def test_geometry_test_only_no_repeated_tiles_every_photo(self):
        _, pools, _ = S.describe(ROOT)
        for panel, sources in S.PANELS.items():
            f = self.frames[S.filename(panel)]
            self.assertFalse(f.cover_id.duplicated().any())
            self.assertFalse(f.pair_id.duplicated().any())
            self.assertFalse(f[['source_photo','tile_row','tile_col']].duplicated().any())
            S.verify_frame(f, panel, False, pools)
            for source in sources:
                manifest = pd.read_csv(ROOT/f'data/{source}/photo_manifest.csv')
                photos = set(f.loc[f.source == source, 'source_photo'])
                self.assertTrue(photos.isdisjoint(set(manifest.query('split in ["train", "val"]').photo_id)))
                counts = f.loc[f.source == source].groupby('source_photo').size()
                self.assertLessEqual(counts.max()-counts.min(), 1)

    def test_pool_and_quota_statistics(self):
        expected = {'raise':(149,36280), 'div2k':(100,4808), 'lsdir':(2500,50586)}
        for name, stats in self.summary['panels'].items():
            for source, p in stats['per_source'].items():
                if source in expected:
                    self.assertEqual((p['photos'],p['tiles']), expected[source])
                elif 'unseen' in name:
                    self.assertEqual((p['photos'],p['tiles']), (2000,18070))
                else:
                    self.assertEqual((p['photos'],p['tiles']), (1700,15313))
                self.assertEqual(p['photos'], p['selected_photos'])

    def test_higher_k_same_tiles_slots_draws_and_main_unchanged(self):
        main, _, _ = S.load_frozen(ROOT)
        for panel in S.PANELS:
            low = self.frames[S.filename(panel)]
            high = self.frames[S.filename(panel,True)]
            self.assertTrue(np.array_equal(low.k+4,high.k))
            self.assertTrue(main[S.filename(panel)].equals(low))
            keys = ['cover_id','source_photo','slot','bit_mode','method','placement','payload_kind','stride',
                    'requested_bpp','settings_seed_key','embedding_seed_key']
            self.assertTrue(low[keys].equals(high[keys]))
            self.assertFalse(set(low.pair_id) & set(high.pair_id))

    def test_clipping_and_requested_bpp_are_retained(self):
        for name, f in self.frames.items():
            self.assertTrue(f.requested_bpp.between(.05,.5).all())
            self.assertTrue((f.effective_bpp <= f.requested_bpp + 8/65536).all())
            self.assertEqual(int(f.clipped.sum()), self.summary['panels'][name]['histograms']['clipped'])
            self.assertEqual(len(f), self.summary['panels'][name]['pairs'])

    def test_identity_independent_of_training_rng_workers_models(self):
        state = np.random.get_state()
        try:
            np.random.seed(941)
            recipe, pools, config = S.describe(ROOT)
            for source in S.PANELS['standard']:
                count = S.PANELS['standard'][source][1]
                tiles = S.select_tiles(pools[('standard', source)], count, 'standard', source)
                rows = S.make_rows(tiles,'standard',source,config,self.summary['recipe_sha256'])
                original = self.frames[S.filename('standard')].query('source == @source').reset_index(drop=True)
                self.assertEqual(csv_text(pd.DataFrame(rows)), csv_text(original))
            self.assertEqual(recipe, self.recipe)
        finally:
            np.random.set_state(state)

    def test_canonical_identity_and_semantic_changes(self):
        self.assertEqual(digest(canonical(dict(reversed(list(self.recipe.items())))).encode()), self.summary['recipe_sha256'])
        for key, changed in [('seed',2),('geometry','other'),('inputs',{}),('panels',{}),('config',{})]:
            self.assertNotEqual(digest(canonical({**self.recipe,key:changed}).encode()), self.summary['recipe_sha256'])

    def test_runtime_is_read_only_and_does_not_build(self):
        with patch.object(Path,'write_text',side_effect=AssertionError('write')), \
             patch.object(Path,'write_bytes',side_effect=AssertionError('write')), \
             patch.object(S,'build',side_effect=AssertionError('runtime build')), \
             patch.object(S,'rng',side_effect=AssertionError('runtime RNG')):
            frames, _, _ = S.load_frozen(ROOT)
        self.assertEqual(len(frames), 4)

    def test_reject_changed_missing_lists_and_inputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            folder = root/S.DATA_DIR
            shutil.copytree(ROOT/S.DATA_DIR, folder)
            desc = S.describe(ROOT)
            with patch.object(S,'describe',return_value=desc):
                path = folder/S.filename('standard')
                path.write_bytes(path.read_bytes()+b'\n')
                with self.assertRaisesRegex(ValueError,'SHA256 mismatch'): S.load_frozen(root)
                path.unlink()
                with self.assertRaises(FileNotFoundError): S.load_frozen(root)
            changed = {**desc[0], 'inputs':{}}
            with patch.object(S,'describe',return_value=(changed,desc[1],desc[2])):
                with self.assertRaisesRegex(ValueError,'recipe/input mismatch'): S.load_frozen(root)

    def test_reject_bad_geometry_and_balance_even_with_valid_csv(self):
        desc = S.describe(ROOT)
        frame = self.frames[S.filename('standard')].copy()
        frame.loc[0,'x'] += 1
        with self.assertRaisesRegex(ValueError,'coordinate'): S.verify_frame(frame,'standard',False,desc[1])
        frame = self.frames[S.filename('standard')].copy()
        frame.loc[0,'k'] = 8
        with self.assertRaisesRegex(ValueError,'balance'): S.verify_frame(frame,'standard',False,desc[1])

    def test_builder_no_overwrite_and_check_twice(self):
        tiny = {'recipe.json':'{}\n', 'pairs.csv':'a,b\n1,2\n'}
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)/'new'
            publish(tiny,out)
            with self.assertRaisesRegex(ValueError,'refusing to overwrite'): publish(tiny,out)
            publish(tiny,out,check=True)
            publish(tiny,out,check=True)
            with self.assertRaisesRegex(ValueError,'MISMATCH'): publish({**tiny,'pairs.csv':'changed'},out,check=True)

    def test_named_metric_families_differ_on_synthetic_scores(self):
        # Three distinct errors: fixed .5=1/3, test optimum .9=0, validation .95=1/6.
        from cvdl.metrics import threshold, rates
        test = pd.DataFrame(dict(label=[0,0,0,1,1,1],prob_stego=[.1,.55,.6,.9,.95,.99],
            k=[1]*6,source=['raise']*6,source_photo=['a','b','c','a','b','c']))
        validation = test.copy(); validation.prob_stego = [.1,.8,.9,.95,.96,.99]
        d = S.diagnostic_rows(test).iloc[0]
        self.assertEqual(d['test_min_P_E'],0)
        self.assertEqual(d['P_E_at_0_5'],1/3)
        self.assertNotEqual(d['test_min_cutoff'],threshold(validation,'pe')[0])
        self.assertEqual(rates(test,threshold(validation,'pe')[0])['pe_at_cutoff'],1/6)
        self.assertNotIn('P_E', d.index)
        rows = S.diagnostic_rows(test); rows.insert(0,'panel','standard')
        main = S.main_placeholder(rows)
        self.assertTrue(main.P_E_at_validation_cutoff.isna().all())
        self.assertTrue((main.threshold_origin == 'pending_validation_min_pe').all())

    def test_prepare_output_no_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = S.prepare_output(Path(tmp)/'eval')
            (out/'record').write_text('existing')
            with self.assertRaisesRegex(ValueError,'refusing to overwrite'): S.prepare_output(out)
            self.assertEqual(S.prepare_output(out,overwrite=True),out)

    def test_mock_evaluation_writes_only_fresh_outputs_no_training_records(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp)
            record = run/'data_run_settings.json'; record.write_text('original')
            before = record.read_bytes()
            ckpt = run/'best-test.ckpt'; torch.save({'hyper_parameters':{},'state_dict':{}}, ckpt)
            settings = SimpleNamespace(root=ROOT, quick=True, eval_workers=0, device='cpu',batch_size=4,
                rocm_conv1_fix=False,values={},train_tile_size=512,tile_size=512,train_grid='recorded',manifest_root=None,source='decode')
            def fake_score(network, ds, **kwargs):
                clean = ds.frame.copy(); clean['label']=0; clean['prob_stego']=.2
                stego = ds.frame.copy(); stego['label']=1; stego['prob_stego']=.8
                return pd.concat([clean,stego],ignore_index=True)
            with patch('cvdl.models.model_from_hparams',return_value=torch.nn.Identity()), \
                 patch('lsb_torch.Corpus',return_value=SimpleNamespace(by_id={})), \
                 patch.object(S,'StandardTestPairs',side_effect=lambda corpus,f:SimpleNamespace(frame=f)), \
                 patch('cvdl.hardtest.score',side_effect=fake_score):
                result = S.run(settings,ckpt,run,higher_k=True)
                out = Path(result['output'])
                self.assertEqual(result['hard_test']['status'],'pending')
                self.assertEqual(len(list((out/'panels').glob('*/predictions.csv'))),4)
                self.assertEqual(len(list((out/'higher-k').glob('*/predictions.csv'))),4)
                self.assertNotIn('P_E',pd.read_csv(out/'diagnostic_summary.csv').columns)
                self.assertTrue(pd.read_csv(out/'main_summary.csv').cutoff.isna().all())
                with self.assertRaisesRegex(ValueError,'refusing to overwrite'): S.run(settings,ckpt,run,higher_k=True)
                with self.assertRaisesRegex(ValueError,'must be under'): S.run(settings,ckpt,run,output=run)
            self.assertEqual(record.read_bytes(),before)
            self.assertEqual(list(run.glob('test_predictions.csv')),[])

    @unittest.skipUnless((ROOT/'Datasets/originals/raise/NEF').is_dir(), 'real source photos not present')
    def test_cpu_real_replay_zero_two_workers_and_512_model_on_256(self):
        from lsb_torch import Corpus, RunSettings, OrderedPairLoader
        from cvdl.models import model_from_hparams
        old_threads = torch.get_num_threads(); torch.set_num_threads(1)
        try:
            corpus = Corpus(ROOT,RunSettings(sources=('raise','div2k','alaska','lsdir')))
            pieces = [f.groupby('source',sort=False).head(1) for f in self.frames.values()]
            f = pd.concat(pieces,ignore_index=True)
            ds = S.StandardTestPairs(corpus,f)
            def replay(workers):
                loader = OrderedPairLoader(ds,batch_size=2,num_workers=workers,pin_memory=False)
                try:
                    return [(c.numpy().tobytes(),s.numpy().tobytes(),i.tolist()) for c,s,i in loader]
                finally:
                    loader.shutdown()
            self.assertEqual(replay(0),replay(2))
            path = next((ROOT/'runs/newdata-1-20261015T173117Z-1917').rglob('best-*.ckpt'),None)
            if path is None:
                candidates = list((ROOT/'runs').glob('tiling-overlap-2026*/checkpoints/*/best-*.ckpt'))
                self.assertTrue(candidates,'512-trained checkpoint required for the CPU verification')
                path = candidates[0]
            saved = torch.load(path,map_location='cpu',weights_only=False)
            net = model_from_hparams(saved['hyper_parameters']).eval()
            net.load_state_dict({k.removeprefix('net.'):v for k,v in saved['state_dict'].items()})
            with torch.inference_mode():
                logits = net(ds[0][0].float()[None])
            self.assertEqual(tuple(logits.shape),(1,2))
            self.assertTrue(torch.isfinite(logits).all())
        finally:
            torch.set_num_threads(old_threads)


class StandardCLITests(unittest.TestCase):
    def invoke(self, arguments):
        settings = SimpleNamespace(eval_workers=0)
        with patch.object(sys,'argv',['evaluate_run.py','run',*arguments]), \
             patch.object(er,'checkpoint_folder',return_value=Path('/run/checkpoints/name')), \
             patch('cvdl.lit.select_best_checkpoint',return_value='/run/checkpoints/name/best.ckpt'), \
             patch.object(er,'settings_for_evaluation',return_value=(settings,{},None)), \
             patch('cvdl.runtime.configure'), patch('scripts.pc_safety.own_process_group'), \
             patch('scripts.pc_safety.check_workers'), patch.object(S,'run') as new, \
             patch.object(er,'evaluate') as legacy:
            er.main()
            return new, legacy

    def test_new_suite_is_default_without_flag(self):
        new, legacy = self.invoke([])
        new.assert_called_once(); legacy.assert_not_called()
        self.assertFalse(new.call_args.kwargs['higher_k'])

    def test_higher_k_opt_in(self):
        new, legacy = self.invoke(['--higher-k'])
        self.assertTrue(new.call_args.kwargs['higher_k']); legacy.assert_not_called()

    def test_explicit_legacy_retains_default_hard(self):
        new, legacy = self.invoke(['--test-suite','legacy-512'])
        new.assert_not_called(); legacy.assert_called_once()
        self.assertTrue(legacy.call_args.kwargs['hard'])

    def test_new_never_silently_runs_legacy_stages(self):
        for args in (['--only','hard'],['--temperature'],['--tables'],['--padding-check']):
            with self.assertRaisesRegex(ValueError,'legacy-512'): self.invoke(args)
        with self.assertRaisesRegex(ValueError,'requires'): self.invoke(['--test-suite','legacy-512','--higher-k'])


if __name__ == '__main__':
    unittest.main()
