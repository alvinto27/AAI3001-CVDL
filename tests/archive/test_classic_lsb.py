import unittest
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
from cvdl.archive.classic_lsb import weighted_stego, sample_pairs, scores


class ClassicLSBTests(unittest.TestCase):
    def test_random_replacement_tracks_rate_but_matching_does_not(self):
        # Smooth continuous synthetic covers, rounded only at the end. No test data.
        for seed in (174, 279):
            rng = np.random.default_rng(seed)
            x = rng.normal(128, 30, (512, 512))
            for _ in range(5):
                x = (x+np.roll(x, 1, 0)+np.roll(x, -1, 0)+np.roll(x, 1, 1)+np.roll(x, -1, 1))/5
            cover = np.rint(x).astype(np.uint8)
            for rate in (0, .1, .25, .5, .8):
                selected = rng.random(cover.shape) < rate
                bits = rng.integers(0, 2, cover.shape, dtype=np.uint8)
                replacement = cover.copy()
                replacement[selected] = (cover[selected] & 254) | bits[selected]
                matching = cover.astype(np.int16)
                change = selected & ((cover & 1) != bits)
                direction = rng.choice([-1, 1], size=cover.shape)
                matching[change] += direction[change]
                self.assertTrue(np.all((matching >= 0) & (matching <= 255)))
                for detector in (weighted_stego, sample_pairs):
                    with self.subTest(seed=seed, rate=rate, detector=detector.__name__):
                        self.assertAlmostEqual(detector(replacement), rate, delta=.06)
                        self.assertAlmostEqual(detector(matching.astype(np.uint8)), 0, delta=.04)

    def test_reference_constant_and_degenerate_channels(self):
        for detector in (weighted_stego, sample_pairs):
            self.assertEqual(detector(np.full((8, 8), 128, dtype=np.uint8)), 0)
        # SPA's documented no-upper-pair fallback, not an unnoticed NaN.
        self.assertEqual(sample_pairs(np.tile(np.arange(0, 16, 2, dtype=np.uint8), (8, 1))), -1)

    def test_combined_is_exact_arithmetic_mean(self):
        rgb = np.random.default_rng(91).integers(0, 256, (32, 32, 3), dtype=np.uint8)
        result = scores(rgb)
        for name in ('ws', 'spa'):
            self.assertEqual(result[name], np.mean([result[f'{name}_{c}'] for c in 'rgb']))

    def test_validation_threshold_is_conservative_with_ties(self):
        from scripts.archive.check1_classic import validation_cutoff
        clean = np.array([0]*90+[1]*10, dtype=float)
        cutoff = validation_cutoff(clean, .05)
        self.assertLessEqual(np.mean(clean >= cutoff), .05)
        self.assertGreater(cutoff, 1)
        with self.assertRaises(ValueError): validation_cutoff([], .05)

    def test_photo_bootstrap_is_deterministic_and_keeps_pairs_together(self):
        import pandas as pd
        from scripts.archive.check1_classic import auc_intervals
        frame = pd.DataFrame(dict(source_photo=['a','a','b','b','c','c'], label=[0,1]*3,
                                  ws=[0,1,0,1,0,1]))
        first = auc_intervals(frame, ['ws'], replicates=25)
        self.assertEqual(first, auc_intervals(frame, ['ws'], replicates=25))
        self.assertEqual(first['ws'], (1,1,1))

    def test_saved_score_join_requires_exact_pair_and_photo_identity(self):
        import pandas as pd
        import tempfile
        from pathlib import Path
        from scripts.archive.check1_classic import attach_saved
        frame = pd.DataFrame(dict(pair_id=['a','a'], label=[0,1], source_photo=['photo']*2))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'saved.csv'
            saved = frame.assign(prob_stego=[.1,.9]); saved.to_csv(path, index=False)
            np.testing.assert_array_equal(attach_saved(frame, path).srm, [.1,.9])
            saved['source_photo'] = 'different'; saved.to_csv(path, index=False)
            with self.assertRaisesRegex(ValueError, 'do not match'): attach_saved(frame, path)

    def test_descriptive_roc_keeps_ties_and_matches_auc(self):
        from scripts.archive.check1_classic import descriptive_roc
        from cvdl.metrics import auc_numpy
        labels = np.array([0,1,0,1]); values = np.array([.1,.5,.5,.9])
        fa, miss = descriptive_roc(labels, values)
        self.assertEqual(len(fa), 4)
        self.assertEqual(np.trapezoid(1-miss, fa), auc_numpy(labels, values))
        np.testing.assert_array_equal(fa[[0,-1]], [0,1])
        np.testing.assert_array_equal(miss[[0,-1]], [1,0])

    def test_invalid_inputs(self):
        for detector in (weighted_stego, sample_pairs):
            with self.assertRaises(ValueError): detector(np.zeros((3, 3)))
            with self.assertRaises(ValueError): detector(np.zeros((2, 3), dtype=np.uint8))
        with self.assertRaises(ValueError): scores(np.zeros((3, 3), dtype=np.uint8))
