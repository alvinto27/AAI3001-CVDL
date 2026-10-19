"""Saved-file joins and shared photo-cluster bootstrap; no inference."""
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd
from cvdl.metrics import auc_numpy
from scripts.curriculum_decision import join_predictions, intervals, NAMES


class DecisionTests(unittest.TestCase):
    def frame(self):
        return pd.DataFrame(dict(pair_id=['a', 'a', 'b', 'b', 'c', 'c'],
            label=[0, 1, 0, 1, 0, 1], source_photo=['p', 'p', 'p', 'p', 'q', 'q'],
            prob_stego=[.1, .8, .4, .4, .7, .9]))

    def paths(self, folder, frames):
        paths = []
        for i, frame in enumerate(frames):
            path = Path(folder) / f'{i}.csv'
            frame.to_csv(path, index=False, float_format='%.17g')
            paths.append(path)
        return paths

    def test_exact_join_accepts_reordering_and_retains_repeated_rows(self):
        with TemporaryDirectory() as folder:
            f = self.frame()
            joined = join_predictions(self.paths(folder, [f, f.iloc[::-1], f]))
            self.assertEqual(len(joined), 6)
            for name in NAMES:
                np.testing.assert_array_equal(joined[name], f.prob_stego)

    def test_rejects_missing_extra_duplicate_keys_and_metadata_mismatch(self):
        f = self.frame()
        mismatches = [f.iloc[:-1], pd.concat([f, f.iloc[:1]]),
                      f.assign(pair_id=['wrong', 'wrong', 'b', 'b', 'c', 'c']),
                      f.assign(source_photo='wrong'), f.assign(label=[0, 0, 0, 1, 0, 1])]
        for other in mismatches:
            with self.subTest(other=other.to_dict()), TemporaryDirectory() as folder:
                with self.assertRaises(ValueError):
                    join_predictions(self.paths(folder, [f, other, f]))

    def test_requires_complete_clean_stego_pairs(self):
        with TemporaryDirectory() as folder:
            f = self.frame().iloc[:-1]
            with self.assertRaisesRegex(ValueError, 'one clean and one stego'):
                join_predictions(self.paths(folder, [f, f, f]))

    def test_shared_draws_match_independent_cluster_replay(self):
        frame = self.frame().drop(columns='prob_stego')
        for i, name in enumerate(NAMES):
            frame[name] = np.array([.1, .8, .4, .4, .7, .9]) + i*.01
        result, draws = intervals(frame, replicates=20, seed=12)
        groups = [np.array([0, 1, 2, 3]), np.array([4, 5])]
        rng = np.random.default_rng(12)
        expected = []
        for _ in range(20):
            idx = np.concatenate([groups[i] for i in rng.integers(0, 2, 2)])
            expected.append([auc_numpy(frame.label.to_numpy()[idx], frame[name].to_numpy()[idx]) for name in NAMES])
        np.testing.assert_array_equal(draws, expected)
        np.testing.assert_array_equal(draws[:, 0], draws[:, 1])
        for record in result['differences'].values():
            self.assertEqual(record, dict(point=0., low=0., high=0.))
        self.assertEqual(result['pairs'], 3)
        self.assertEqual(result['photos'], 2)
        np.testing.assert_array_equal(draws, intervals(frame, replicates=20, seed=12)[1])

    def test_pe_uses_own_cutoff_and_keeps_pair_balance(self):
        frame = self.frame().drop(columns='prob_stego')
        for name in NAMES:
            frame[name] = self.frame().prob_stego
        result, draws = intervals(frame, metric='pe', cutoffs=[.5, .8, .95], replicates=20)
        expected = [1/3, 1/6, 1/2]
        for name, point in zip(NAMES, expected):
            self.assertAlmostEqual(result['models'][name]['point'], point)
        ci = np.quantile(draws[:, 1] - draws[:, 0], [.025, .975])
        diff = result['differences']['curriculum-srm_e50']
        self.assertAlmostEqual(diff['point'], -1/6)
        np.testing.assert_array_equal([diff['low'], diff['high']], ci)


if __name__ == '__main__':
    unittest.main()
