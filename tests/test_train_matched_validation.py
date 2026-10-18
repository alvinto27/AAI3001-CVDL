"""Frozen train-matched validation: metadata, compatibility and CPU-only replay."""
from dataclasses import replace
import hashlib
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

from cvdl import validation as val
from cvdl.settings import load, machine_profile
from lsb_batch import read_config, write_config
from lsb_notebook import NotebookData
from lsb_settings import draw_otf_settings
from lsb_torch import Corpus, OrderedPairLoader
from scripts.build_validation import notebook_settings

ROOT = Path(__file__).resolve().parents[1]
NAMES = ('srm', 'srm-hard')
IDS = ('60da4391043e8ce48ff2', 'bf7ede374cab01f73f5d')


def settings(name='srm'):
    return notebook_settings(ROOT/f'notebooks/experiments/{name}.ipynb')


def loader_bytes(dataset, workers):
    loader = OrderedPairLoader(dataset, batch_size=2, num_workers=workers, pin_memory=False)
    try:
        return [(c.numpy().tobytes(), s.numpy().tobytes(), i.tolist()) for c, s, i in loader]
    finally:
        loader.shutdown()


class MatchedValidationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.settings = [settings(n) for n in NAMES]
        cls.builds = [val.build(ROOT, s.to_run_settings(), s.train_grid, s.train_tile_size) for s in cls.settings]

    def test_published_lists_rebuild_exactly_and_full_hashes(self):
        for expected, (rid, files, summary) in zip(IDS, self.builds):
            self.assertEqual(rid, expected)
            for name, text in files.items():
                self.assertEqual((ROOT/'data/val'/rid/name).read_bytes(), text.encode())
            self.assertEqual(summary['recipe_sha256'][:20], rid)
            self.assertEqual(summary['pairs_sha256'], val.digest(files['pairs.csv'].encode()))

    def test_exact_quotas_pools_unique_val_geometry(self):
        for s, (rid, _, summary) in zip(self.settings, self.builds):
            frame, recipe, _, _ = val.load_frozen(ROOT, s.to_run_settings(), s.train_grid, s.train_tile_size, rid)
            self.assertEqual(len(frame), s.tiles_per_epoch // 10)
            self.assertFalse(frame.cover_id.duplicated().any())
            self.assertEqual(set(frame['split']), {'val'})
            self.assertEqual(set(frame.tile_size), {256})
            self.assertEqual(set(frame.grid), {'overlap_last'})
            self.assertEqual(set(frame.variant), {1})
            self.assertFalse(recipe['augment'])
            self.assertEqual(frame.source.value_counts().to_dict(), recipe['source_quotas'])
            self.assertEqual({k:v['tiles'] for k,v in summary['pools'].items()},
                             {'raise':36484, 'alaska':8966, 'lsdir':30584})
            # Membership in validation manifests ensures train/test photos cannot enter.
            for source, group in frame.groupby('source'):
                manifest = pd.read_csv(ROOT/f'data/{source}/photo_manifest.csv')
                allowed = set(manifest.query('split == "val"').photo_id)
                self.assertTrue(set(group.source_photo) <= allowed)

    def test_rounding_and_source_order_are_deterministic(self):
        self.assertEqual(val.quotas(180000, {'raise':.4,'lsdir':.2,'alaska':.4}),
                         (18000, {'alaska':7200,'lsdir':3600,'raise':7200}))
        self.assertEqual(val.quotas(216000, {'raise':1/3,'lsdir':1/3,'alaska':1/3})[1],
                         {'alaska':7200,'lsdir':7200,'raise':7200})
        self.assertEqual(val.quotas(105, {'b':.5,'a':.5}), (11, {'a':6,'b':5}))
        self.assertEqual(val.quotas(105, {'a':.5,'b':.5}), val.quotas(105, {'b':.5,'a':.5}))
        with self.assertRaises(ValueError): val.quotas(1, {'a':1})

    def test_quota_above_pool_fails_without_capping(self):
        s = replace(self.settings[0].to_run_settings(), tiles_per_epoch=300000)
        with self.assertRaisesRegex(ValueError, 'alaska: validation quota 12000 exceeds unique val tile pool 8966'):
            val.describe(ROOT, s, 'overlap_last', 256)

    def test_hash_canonical_excludes_model_lr_epoch_hardware_run_seed(self):
        s = self.settings[0]
        recipe = val.describe(ROOT, s.to_run_settings(), 'overlap_last', 256)[0]
        reverse = dict(reversed(list(recipe.items())))
        self.assertEqual(val.recipe_id(recipe), val.recipe_id(reverse))
        changed_seed = replace(s.to_run_settings(), run_seed=909, fixed_variants=True)
        self.assertEqual(recipe, val.describe(ROOT, changed_seed, 'overlap_last', 256)[0])
        self.assertEqual(val.recipe_id(recipe), val.recipe_id(val.describe(ROOT, settings('srm-e50').to_run_settings(), 'overlap_last', 256)[0]))
        larger = replace(s.to_run_settings(), tiles_per_epoch=180010)
        self.assertNotEqual(val.recipe_id(recipe), val.recipe_id(val.describe(ROOT, larger, 'overlap_last', 256)[0]))
        ordered = replace(s.to_run_settings(), sources=tuple(reversed(s.sources)),
                          source_mix=dict(reversed(list(s.source_mix.items()))))
        self.assertEqual(recipe, val.describe(ROOT, ordered, 'overlap_last', 256)[0])
        self.assertNotEqual(val.recipe_id(recipe), IDS[1])

    def test_cosmetic_yaml_paths_and_comments_do_not_change_id(self):
        s = self.settings[0].to_run_settings()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'cosmetic.yaml'
            cfg = replace(read_config(s.settings_yaml), input='unused-in', output='unused-out', png_compress_level=5)
            write_config(path, cfg)
            path.write_text(path.read_text()+'\n# unchanged draws\n')
            changed = val.describe(ROOT, replace(s, settings_yaml=str(path)), 'overlap_last', 256)[0]
            self.assertEqual(val.recipe_id(changed), IDS[0])
            write_config(path, replace(cfg, k=[1,2]))
            changed = val.describe(ROOT, replace(s, settings_yaml=str(path)), 'overlap_last', 256)[0]
            self.assertNotEqual(val.recipe_id(changed), IDS[0])

    def test_semantic_geometry_and_seed_changes_change_hash(self):
        recipe = val.describe(ROOT, self.settings[0].to_run_settings(), 'overlap_last', 256)[0]
        for key, value in [('selection_seed',2),('geometry','next-version'),('tile_size',512),
                           ('inputs',{}),('source_quotas',{}),('config',{})]:
            self.assertNotEqual(val.recipe_id({**recipe,key:value}), IDS[0])
        with self.assertRaisesRegex(ValueError, 'overlap_last'):
            val.describe(ROOT, self.settings[0].to_run_settings(), 'centre_trim', 256)

    def test_missing_mismatched_or_tampered_list_is_rejected(self):
        s = self.settings[0].to_run_settings()
        with self.assertRaisesRegex(ValueError, 'does not match'):
            val.load_frozen(ROOT, s, 'overlap_last', 256, IDS[1])
        with self.assertRaises(ValueError): val.load_frozen(ROOT, s, 'overlap_last', 256, '../escape')
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            folder = root/'data/val'/IDS[0]
            # Use real metadata resolution but a separate frozen-list root for failure cases.
            desc = val.describe(ROOT, s, 'overlap_last', 256)
            with patch.object(val, 'describe', return_value=desc):
                with self.assertRaisesRegex(ValueError, 'missing or invalid'):
                    val.load_frozen(root, s, 'overlap_last', 256, IDS[0])
                shutil.copytree(ROOT/'data/val'/IDS[0], folder)
                with (folder/'pairs.csv').open('a') as f: f.write('\n')
                with self.assertRaisesRegex(ValueError, 'SHA256 mismatch'):
                    val.load_frozen(root, s, 'overlap_last', 256, IDS[0])

    def test_recorded_draws_replay_shared_weighted_sampler(self):
        image = np.zeros((256,256,3), dtype=np.uint8)
        for s, rid in zip(self.settings, IDS):
            frame, _, _, _ = val.load_frozen(ROOT, s.to_run_settings(), 'overlap_last', 256, rid)
            config = read_config(s.train_yaml)
            for row in frame.iloc[::997].itertuples():
                kwargs, meta = draw_otf_settings(config, image, np.random.default_rng(np.random.SeedSequence(json.loads(row.settings_seed_key))))
                kwargs.pop('rng')
                self.assertEqual(kwargs, json.loads(row.draw_kwargs))
                self.assertEqual(meta['payload_bytes'], row.payload_bytes)
                self.assertEqual(meta['requested_bpp'], row.requested_bpp)
                self.assertEqual(meta.get('bpp_mixture_component', -1), row.bpp_mixture_component)
                key = json.loads(row.embedding_seed_key)
                self.assertEqual((key[1],key[2],key[3],key[-1]), (2,1,0,1))

    def test_realized_histograms_match_config_distributions(self):
        for index, (_, _, summary) in enumerate(self.builds):
            h = summary['histograms']; n = summary['pairs']
            expected = [.25]*4 if index == 0 else [.5,1/6,1/6,1/6]
            for k, target in enumerate(expected, 1): self.assertAlmostEqual(h['k'][str(k)]/n, target, delta=.02)
            expected_bpp = [1/9,1/9,4/9,1/3] if index == 0 else [11/36,11/36,2/9,1/6]
            for observed, target in zip(h['requested_bpp'].values(), expected_bpp):
                self.assertAlmostEqual(observed/n, target, delta=.02)
            for field in ('bit_mode','method','payload_kind','placement'):
                for count in h[field].values(): self.assertAlmostEqual(count/n, 1/len(h[field]), delta=.02)

    def test_settings_opt_in_and_legacy_serialization(self):
        profile = machine_profile('pc', root=ROOT, device='cpu', precision_limits=('32',))
        legacy = load(environ={}, _profile=profile)
        self.assertNotIn('validation_recipe', legacy.values)
        self.assertNotIn('validation_recipe', legacy.provenance)
        with self.assertRaisesRegex(ValueError, 'needs validation_recipe'):
            load(environ={}, _profile=profile, checkpoint_validation='train_matched')
        with self.assertRaisesRegex(ValueError, 'requires checkpoint_validation'):
            load(environ={}, _profile=profile, validation_recipe=IDS[0])
        s = self.settings[0]
        from cvdl.settings import SCHEMA, plain
        values = {k:v for k,v in plain(s.values).items() if not SCHEMA[k].machine}
        values.update(checkpoint_validation='train_matched', validation_recipe=IDS[0])
        matched = load(environ={}, _profile=profile, **values)
        self.assertEqual(matched.validation_recipe, IDS[0])

    def test_quick_runtime_is_read_only_and_records_parent_and_subset(self):
        s = self.settings[0]
        with patch.object(Path, 'write_text', side_effect=AssertionError('runtime write')), \
             patch.object(Path, 'write_bytes', side_effect=AssertionError('runtime write')):
            data = NotebookData(ROOT, s.to_run_settings(), quick=True, eval_pairs=9,
                checkpoint_validation='train_matched', train_grid='overlap_last', train_tile=256, validation_recipe=IDS[0])
        record = data.record['validation_recipe']
        self.assertEqual(record['pairs_sha256'], self.builds[0][2]['pairs_sha256'])
        self.assertEqual(record['selected_pairs_sha256'], val.digest(data.validation_pair_csv.encode()))
        self.assertNotEqual(record['selected_pairs_sha256'], record['pairs_sha256'])
        self.assertEqual(record['source_counts'], {'raise':3,'alaska':3,'lsdir':3})
        self.assertEqual(len(data.val), 9)

    def test_runtime_bridge_and_saved_evaluator_settings_select_recipe(self):
        from cvdl import runtime
        from cvdl.settings import SCHEMA, plain
        from scripts.evaluate_run import settings_for_run
        profile = machine_profile('pc', root=ROOT, device='cpu', precision_limits=('32',))
        values = {k:v for k,v in plain(self.settings[0].values).items() if not SCHEMA[k].machine}
        values.update(checkpoint_validation='train_matched', validation_recipe=IDS[0])
        s = load(environ={}, _profile=profile, overrides={'quick':False}, **values)
        fake = SimpleNamespace(record={})
        with patch.object(runtime.data, 'build_otf_data', return_value=fake) as build, \
             patch.object(runtime.data, 'otf_pair_tables', return_value='tables'):
            self.assertEqual(runtime.create_data(s), (fake, 'tables'))
            self.assertEqual(build.call_args.kwargs['validation_recipe'], IDS[0])
        with tempfile.TemporaryDirectory() as tmp, patch('cvdl.settings.automatic_profile', return_value=profile):
            folder = Path(tmp)
            s.write(folder)
            restored = settings_for_run(folder)
            self.assertEqual(restored.validation_recipe, IDS[0])
            self.assertEqual(restored.checkpoint_validation, 'train_matched')

    def test_source_metrics_use_configured_sources_not_legacy_triple(self):
        from cvdl.lit import SteganalysisModule
        fake = SimpleNamespace(checkpoint_validation='train_matched',
            corpus=SimpleNamespace(settings=SimpleNamespace(sources=('raise','div2k'))),
            val=SimpleNamespace(frame=pd.DataFrame({'source':['raise','div2k']})))
        model = SteganalysisModule(validation_data=fake)
        model._trainer = SimpleNamespace(sanity_checking=False)
        scores = torch.tensor([[2.,0.],[2.,0.],[0.,2.],[0.,2.]])
        labels = torch.tensor([0,0,1,1])
        batch = (torch.zeros(2,3,1,1),torch.zeros(2,3,1,1),torch.tensor([0,1]))
        with patch.object(model, 'pair_logits_targets', return_value=(scores, labels)), patch.object(model, 'log') as log:
            model.validation_step(batch,0)
            model.on_validation_epoch_end()
            names = {call.args[0] for call in log.call_args_list}
            self.assertTrue({'val_pe_raise','val_pe_div2k','val_loss_raise','val_loss_div2k'} <= names)
            self.assertNotIn('val_pe_alaska',names)

    def test_old_capped_identity_unchanged(self):
        s = self.settings[0]
        data = NotebookData(ROOT, s.to_run_settings(), checkpoint_validation='capped', train_grid='overlap_last', train_tile=256)
        self.assertEqual(len(data.val), 8035)
        self.assertEqual(hashlib.sha256(data.validation_pair_csv.encode()).hexdigest(),
                         '6bb3c8efadad06af50ba014a95c98b4fc57008119bdd55b794ec2a1729e4d5ca')

    @unittest.skipUnless((ROOT/'Datasets/originals/raise/NEF').is_dir(), 'real source photos not present')
    def test_real_cpu_replay_zero_vs_two_workers_no_augmentation(self):
        threads = torch.get_num_threads()
        torch.set_num_threads(1)
        try:
            for s, rid in zip(self.settings, IDS):
                corpus = Corpus(ROOT, s.to_run_settings(), train_grid='overlap_last', train_tile=256)
                frame, _, _, _ = val.load_frozen(ROOT, s.to_run_settings(), 'overlap_last', 256, rid)
                few = frame.groupby('source', sort=False).head(2).reset_index(drop=True)
                dataset = val.MatchedValidationPairs(corpus, few)
                a = loader_bytes(dataset, 0)
                b = loader_bytes(dataset, 2)
                self.assertEqual(a, b)
                self.assertEqual(a, loader_bytes(dataset, 0))
                self.assertEqual(sum(len(indices) for _,_,indices in a), 6)
        finally:
            torch.set_num_threads(threads)


if __name__ == '__main__':
    unittest.main()
