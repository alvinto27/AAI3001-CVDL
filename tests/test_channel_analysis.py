"""CPU-only channel-panel tests; never load a real model on GPU."""
import itertools
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from cvdl import channel_analysis as C


def toy_base(count=2):
    return pd.DataFrame([dict(base_index=i, source_photo=f'photo{i}', cover_id=f'cover{i}',
        stratum=f'stratum{i%2}', source='toy', split='test', noise='other') for i in range(count)])


def toy_variants(base, images=None):
    rows = []
    for item in base.to_dict('records'):
        for k, rate, placement, arm in itertools.product((1, 2), C.RATES, C.PLACEMENTS, C.ARMS):
            row = dict(variant_id=len(rows), base_index=item['base_index'], k=k, rate=rate, placement=placement, arm=arm)
            if images is not None:
                stego, _, changed, units, payload = C.make_variant(images[item['base_index']], item, k, rate, placement, arm)
                row.update(stego_sha256=C.digest(stego.tobytes()), changed_sha256=C.digest(changed.astype('<i8').tobytes()),
                    units_sha256=C.digest(units.astype('<i8').tobytes()), message_sha256=C.digest(payload))
            rows.append(row)
    return pd.DataFrame(rows)


class ChannelPanelTests(unittest.TestCase):
    def test_exact_payload_sizes_capacity_and_channel_pairs(self):
        base = toy_base().iloc[0].to_dict()
        image = np.random.default_rng(33).integers(0, 256, (256, 256, 3), dtype=np.uint8)
        for rate, count in zip(C.RATES, (410, 1229, 4096)):
            message = C.payload_for(base, rate)
            self.assertEqual(len(message), count)
            for k, placement in itertools.product((1, 2), C.PLACEMENTS):
                support = None
                for arm in C.ARMS:
                    stego, meta, changed, units, payload = C.make_variant(image, base, k, rate, placement, arm)
                    self.assertEqual(payload, message)
                    self.assertEqual(meta['embedded_bits'], count*8)
                    self.assertEqual(meta['bpp_actual'], count*8/65536)
                    self.assertEqual(len(units), count*8//k)
                    np.testing.assert_array_equal(changed, np.flatnonzero(stego.reshape(-1) != image.reshape(-1)))
                    np.testing.assert_array_equal(stego & (255-((1<<k)-1)), image & (255-((1<<k)-1)))
                    if arm != 'native_rgb':
                        pixels = units//3
                        self.assertEqual(len(np.unique(pixels)), len(units))
                        if support is None:support = pixels
                        np.testing.assert_array_equal(pixels, support)
                        if arm in ('r', 'g', 'b'):
                            channel = C.ARMS.index(arm)
                            self.assertTrue(np.all(units % 3 == channel))
                            for other in range(3):
                                if other != channel:np.testing.assert_array_equal(stego[..., other], image[..., other])
                        else:
                            counts = np.bincount(units % 3, minlength=3)
                            self.assertLessEqual(counts.max()-counts.min(), 1)
                        fields = stego.reshape(-1)[units] & ((1<<k)-1)
                        bits = ((fields[:, None] >> np.arange(k-1, -1, -1)) & 1).reshape(-1)
                        self.assertEqual(np.packbits(bits).tobytes(), message)

    def test_native_reference_is_fresh_existing_engine_bytes_and_has_distinct_support(self):
        from lsb_engine import embed_fast
        base = toy_base().iloc[0].to_dict()
        image = np.zeros((256, 256, 3), dtype=np.uint8)
        for k, placement in itertools.product((1, 2), C.PLACEMENTS):
            stego, _, changed, units, payload = C.make_variant(image, base, k, .15, placement, 'native_rgb')
            expected, _, actual = embed_fast(image, 'lowest', k, 'replace', placement,
                np.random.default_rng(C.seed('message', base['source_photo'], base['cover_id'], .15)),
                payload_bytes=len(payload), return_changed_units=True)
            np.testing.assert_array_equal(stego, expected)
            np.testing.assert_array_equal(changed, actual)
            if placement == 'contiguous':
                self.assertLess(len(np.unique(units//3)), len(units))

    def test_selection_is_deterministic_photo_disjoint_and_noise_from_clean(self):
        rows = []
        for source, split in C.STRATA:
            for photo in range(7):
                for noisy in (False, True):
                    rows.append(dict(source=source, split=split, source_photo=f'{source}/{split}/{photo}',
                        cover_id=f'{source}/{split}/{photo}/{noisy}', kv_std=.9 if noisy else .1))
        frame = pd.DataFrame(rows)
        edges = {source:.5 for source, _ in C.STRATA}
        a = C.select_base(frame, edges, photos_per_stratum=2)
        b = C.select_base(frame.sample(frac=1, random_state=22), edges, photos_per_stratum=2)
        self.assertEqual(a, b)
        self.assertEqual(len(a), 20)
        self.assertEqual(len({r['source_photo'] for r in a}), 20)
        for row in a:self.assertEqual(row['noise']=='noisy', row['kv_std'] > .5)
        with self.assertRaisesRegex(ValueError, 'insufficient distinct'):C.select_base(frame, edges, photos_per_stratum=4)

    def test_two_tile_sampling_keeps_photo_clusters(self):
        rows = []
        for source, split in C.STRATA:
            for photo in range(6):
                for noise in (False, True):
                    for tile in range(2):
                        rows.append(dict(source=source, split=split, source_photo=f'{source}/{split}/{photo}',
                            cover_id=f'{source}/{split}/{photo}/{noise}/{tile}', kv_std=.9 if noise else .1))
        selected = pd.DataFrame(C.select_base(pd.DataFrame(rows), {s:.5 for s, _ in C.STRATA}, 2, 2))
        self.assertEqual(len(selected), 40)
        self.assertTrue((selected.groupby('source_photo').size()==2).all())
        weights = C.bootstrap_weights(selected, 30)
        for _, group in selected.groupby('source_photo'):
            np.testing.assert_array_equal(weights[:, group.index[0]], weights[:, group.index[1]])

    def test_bootstrap_weights_photo_pairing_and_reproducibility(self):
        base = toy_base(12)
        first = C.bootstrap_weights(base, 40)
        np.testing.assert_array_equal(first, C.bootstrap_weights(base, 40))
        self.assertEqual(first.shape, (41, 12))
        for _, group in base.groupby('stratum'):
            np.testing.assert_array_equal(first[:, group.index].sum(1), np.full(41, len(group)))
        clean, stego = np.array([.1, .8]), np.array([.2, .9])
        weights = np.array([[1., 1.], [2., 0.], [0., 2.]])
        np.testing.assert_allclose(C.weighted_auc(clean, stego, weights), [.75, 1., 1.])
        np.testing.assert_allclose(C.weighted_auc(clean, clean, weights), [.5, .5, .5])

    def test_frozen_reading_rule_and_threshold_metrics_no_test_fit(self):
        base = toy_base(12)
        variants = toy_variants(base)
        predictions = pd.DataFrame(dict(image_index=range(len(base)+len(variants)),
            prob_stego=np.r_[np.full(len(base), .1), np.where(variants.arm.eq('r'), 0., .9)]))
        result = C.summarize_model(base, variants, predictions, .5, C.bootstrap_weights(base, 40))
        self.assertEqual(result['primary']['r']['AUC']['point'], 0.)
        self.assertEqual(result['primary']['rgb']['AUC']['point'], 1.)
        self.assertEqual(result['primary']['r']['P_E']['point'], .5)
        self.assertTrue(result['primary_contrasts'][0]['meaningful'])
        self.assertFalse(result['all_primary_equivalent'])
        for row in result['contrasts']:
            self.assertEqual(row['differences']['FA']['point'], 0.)
        self.assertEqual(C.reading_limits()['auc_minimum'], .02)
        self.assertEqual(C.reading_limits()['bootstrap_replicates'], 5000)
        with self.assertRaisesRegex(ValueError, 'incomplete'):C.summarize_model(base, variants, predictions.iloc[:-1], .5, C.bootstrap_weights(base, 5))

    def test_cpu_report_all_models_cells_shared_draws_and_overwrite_guard(self):
        base = toy_base(8);variants = toy_variants(base)
        recipe = dict(models={name:dict(cutoff=.5) for name in C.MODEL_SPECS},
            reading=dict(C.reading_limits(), bootstrap_replicates=25))
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            for name in C.MODEL_SPECS:
                target = output/'models'/name;target.mkdir(parents=True)
                predictions = pd.DataFrame(dict(image_index=range(len(base)+len(variants)),
                    prob_stego=np.r_[np.full(len(base), .1), np.full(len(variants), .9)]))
                if name == 'run18':
                    predictions['cell_AUC'] = np.r_[np.full(len(base), np.nan), np.full(len(variants), .75)]
                    predictions['cell_eligible'] = np.r_[np.zeros(len(base), bool), np.ones(len(variants), bool)]
                    predictions['positive_cells'] = np.r_[np.zeros(len(base)), np.ones(len(variants))]
                    predictions['negative_cells'] = 32
                    predictions['positive_hits'] = predictions.positive_cells
                    predictions['negative_hits'] = 0
                predictions.to_csv(target/'predictions.csv', index=False)
                C.dump(target/'execution.json', dict(recipe_sha256=C.digest(C.canonical(recipe).encode()),
                    source=recipe['models'][name], predictions_sha256=C.file_hash(target/'predictions.csv')))
            with patch.object(C, 'validate_panel', return_value=(recipe, base, variants)):
                result = C.report('.', output)
                with self.assertRaisesRegex(ValueError, 'overwrite'):C.report('.', output)
            self.assertTrue(result['run18']['all_primary_equivalent'])
            self.assertEqual(result['run18']['clean_cell_FA']['point'], 0.)
            self.assertEqual(len(result['run18']['cell_contiguous']), 30)
            self.assertEqual(result['run18']['cell_contiguous'][0]['mean_eligible_tile_AUC']['point'], .75)
            document = json.loads((output/'report.json').read_text())
            self.assertEqual(len(document['bootstrap_draws_sha256']), 64)
            self.assertIn('No training/model/threshold decision', (output/'REPORT.md').read_text())

    def test_interval_equivalence_and_inconclusive_are_not_significance(self):
        tight = C.interval(np.linspace(-.005, .005, 100), 1-.05/3)
        self.assertFalse(C.practical_effect(0., tight))
        wide = C.interval(np.linspace(-.1, .1, 100), 1-.05/3)
        self.assertFalse(C.practical_effect(.03, wide))
        self.assertEqual(C.interval([float('nan')])['valid_replicates'], 0)
        self.assertTrue(C.practical_effect(-.025, dict(low=-.04, high=-.005)))
        self.assertFalse(C.practical_effect(.019, dict(low=.01, high=.025)))

    def test_cell_targets_use_changes_and_undefined_has_eligibility_counts(self):
        from lsb_cells import changed_cell_targets
        counts, valid = changed_cell_targets(np.array([], dtype=np.int64))
        row = C.cell_statistics(np.zeros((8, 8)), counts, valid)
        self.assertFalse(row['cell_eligible'])
        self.assertIsNone(row['cell_AUC'])
        self.assertEqual(row['negative_cells'], 64)
        counts, valid = changed_cell_targets(np.array([3*40*256+3*40]))
        values = (counts>0).astype(float)
        row = C.cell_statistics(values, counts, valid)
        self.assertTrue(row['cell_eligible']);self.assertEqual(row['cell_AUC'], 1.)
        self.assertEqual(row['positive_hits'], 1)
        full = C.cell_statistics(np.ones((8, 8)), np.ones((8, 8)), np.zeros((8, 8), bool))
        self.assertFalse(full['cell_eligible']);self.assertEqual(full['negative_cells'], 0)

    def test_dataset_replay_hashes_and_worker0_worker4_exact_order(self):
        import torch
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            base = toy_base(2)
            images = np.random.default_rng(3).integers(0, 256, (2, 256, 256, 3), dtype=np.uint8)
            np.save(output/'clean.npy', images)
            variants = toy_variants(base, images).iloc[:6].copy()
            dataset = C.ChannelDataset(output, base, variants)
            def fingerprint(workers):
                loader = torch.utils.data.DataLoader(dataset, batch_size=3, num_workers=workers,
                    **({'multiprocessing_context':'spawn'} if workers else {}))
                return [(C.digest(tensor.numpy().tobytes()), indices.tolist(),
                         C.digest(counts.numpy().tobytes()), C.digest(valid.numpy().tobytes()))
                        for tensor, indices, counts, valid in loader]
            self.assertEqual(fingerprint(0), fingerprint(4))
            broken = variants.copy();broken.loc[0, 'stego_sha256'] = '0'*64
            with self.assertRaisesRegex(ValueError, 'variant byte'):C.ChannelDataset(output, base, broken)[2]

    def test_cpu_scoring_pipeline_with_synthetic_model_and_four_workers(self):
        from cvdl.models import model_from_hparams
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory);base = toy_base(2)
            images = np.random.default_rng(17).integers(0, 256, (2, 256, 256, 3), dtype=np.uint8)
            np.save(output/'clean.npy', images)
            variants = toy_variants(base, images).iloc[:6].copy()
            pinned = dict(checkpoint='synthetic.ckpt', cutoff=.5)
            recipe = dict(models={'r1':pinned})
            network = model_from_hparams(dict(model='xunet', front_end='kv', norm='none', width=1))
            with patch.object(C, 'validate_panel', return_value=(recipe, base, variants)), \
                 patch('cvdl.standardtest.load_network', return_value=network.eval()) as load, \
                 patch('scripts.pc_safety.available_gb', return_value=20), patch('scripts.pc_safety.check_workers'):
                C.score('.', output, 'r1', device='cpu')
                with self.assertRaisesRegex(ValueError, 'overwrite'):C.score('.', output, 'r1', device='cpu')
            load.assert_called_once_with(Path('synthetic.ckpt'), 'cpu')
            predictions = pd.read_csv(output/'models/r1/predictions.csv')
            self.assertEqual(predictions.image_index.tolist(), list(range(8)))
            self.assertTrue(predictions.prob_stego.between(0, 1).all())
            context = json.loads((output/'models/r1/execution.json').read_text())
            self.assertEqual(context['batch_size'], 32);self.assertEqual(context['workers'], 4)
            self.assertFalse(context['rocm_conv1_fix']);self.assertEqual(context['precision'], 'fp32')

    def test_freeze_refuses_overwrite_before_loading_scores(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(C, 'estimate_precision') as estimate:
            with self.assertRaisesRegex(ValueError, 'overwrite'):C.build('.', directory, 'unused')
            estimate.assert_not_called()

    def test_recipe_and_file_tampering_refused(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory);(output/'file').write_text('before')
            recipe = dict(version=C.VERSION, files_sha256={'file':C.file_hash(output/'file')})
            C.dump(output/'recipe.json', recipe)
            (output/'recipe.sha256').write_text(C.digest(C.canonical(recipe).encode())+'\n')
            (output/'file').write_text('after')
            with self.assertRaisesRegex(ValueError, 'frozen panel file changed'):C.validate_panel('.', output)
            (output/'recipe.sha256').write_text('0'*64)
            with self.assertRaisesRegex(ValueError, 'recipe identity'):C.validate_panel('.', output)

    def test_model_calibration_mismatch_refused_before_checkpoint_loading(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run, eid, sha, cutoff, threshold = C.MODEL_SPECS['r1']
            folder = root/'runs'/run/'evaluation/standard-256-v1'/eid/'validation'
            folder.mkdir(parents=True)
            calibration = dict(checkpoint_sha256=sha, cutoff=cutoff, threshold_id=threshold,
                deployment=True, quick=False, geometry=dict(grid='overlap_last', tile_size=256))
            for field, bad in (('checkpoint_sha256', '0'*64), ('cutoff', .123), ('threshold_id', 'wrong'),
                    ('deployment', False), ('quick', True), ('geometry', dict(grid='trim', tile_size=256))):
                with self.subTest(field=field):
                    C.dump(folder/'calibration.json', dict(calibration, **{field:bad}))
                    with self.assertRaisesRegex(ValueError, 'model/calibration identity'):C.models(root)

    def test_settings_and_channel_selection_fail_closed(self):
        with self.assertRaisesRegex(ValueError, '32 images'):C.score('.', 'unused', 'run18', device='cpu', batch_size=64)
        with self.assertRaisesRegex(ValueError, 'unsupported model'):C.score('.', 'unused', 'other', device='cpu')
        with self.assertRaisesRegex(ValueError, 'unsupported channel'):C.make_variant(np.zeros((256,256,3), np.uint8),
            toy_base().iloc[0].to_dict(), 1, .05, 'stride', 'r')

    def test_precision_estimate_existing_scores_only_and_one_tile_choice(self):
        root = Path(__file__).resolve().parents[1]
        path = root/'runs'/C.RUN18/'evaluation/standard-256-v1/b3d9a49137b5dbd7bd5a/hard-test/hard_predictions.csv'
        if not path.exists():self.skipTest('local saved run18 score evidence absent')
        result = C.estimate_precision(root)
        self.assertEqual(result['tiles_per_photo'], 1)
        self.assertEqual(len(result['strata']), 10)
        self.assertAlmostEqual(result['expected_half_width_one_tile'], .01944511481259135)
        self.assertGreater(result['sensitivity_one_tile']['0.0'], .02)
        self.assertEqual(C.estimate_precision(root, correlation=.4)['tiles_per_photo'], 2)
        with self.assertRaisesRegex(ValueError, 'BLOCKER'):C.estimate_precision(root, correlation=0.)


if __name__ == '__main__':
    unittest.main()
