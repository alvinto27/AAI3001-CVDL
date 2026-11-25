import unittest

import numpy as np
import pandas as pd

from cvdl.metrics import rates, threshold


class ThresholdTests(unittest.TestCase):
    def test_five_percent_uses_validation_clean_only_and_never_exceeds_target(self):
        f = pd.DataFrame({'label': [0] * 20 + [1] * 2,
                          'prob_stego': [0.1] * 18 + [.7, .9] + [.01, .99],
                          'k': [1] * 22})
        cutoff, fa = threshold(f, 'fa5')
        self.assertEqual(cutoff, .9)
        self.assertAlmostEqual(fa, .05)
        self.assertAlmostEqual(rates(f, cutoff)['k1_miss'], .5)

    def test_tied_scores_are_not_broken_apart_to_force_five_percent(self):
        f = pd.DataFrame({'label': [0] * 20 + [1],
                          'prob_stego': [.1] * 18 + [.9] * 2 + [.9],
                          'k': [1] * 21})
        cutoff, fa = threshold(f, 'fa5')
        self.assertGreater(cutoff, .9)
        self.assertEqual(fa, 0.)
        self.assertEqual(rates(f, cutoff)['miss'], 1.)

    def test_pe_minimum_prefers_lower_false_alarm_and_stays_on_validation(self):
        f = pd.DataFrame({'label': [0, 0, 1, 1],
                          'prob_stego': [.1, .4, .3, .9], 'k': [1] * 4})
        cutoff, fa = threshold(f, 'pe')
        self.assertEqual(cutoff, .9)
        self.assertEqual(fa, 0.)
        self.assertAlmostEqual(rates(f, cutoff)['pe_at_cutoff'], .25)
        with self.assertRaises(ValueError):
            threshold(pd.DataFrame({'label': [0], 'prob_stego': [.1]}), 'fa5')


if __name__ == '__main__':
    unittest.main()
