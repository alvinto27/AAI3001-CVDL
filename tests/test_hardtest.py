"""Shared Wilson interval boundary checks."""
import math
import unittest
from cvdl import hardtest as H


class MetricTests(unittest.TestCase):
    def test_wilson_matches_closed_form(self):
        low,high=H.wilson(50,1000)
        # closed form (Wilson 1927), computed independently: centre +- z*sqrt(p(1-p)/n + z^2/4n^2), over 1+z^2/n
        self.assertAlmostEqual(low,0.03813,places=4);self.assertAlmostEqual(high,0.06531,places=4)
        self.assertTrue(all(math.isnan(x) for x in H.wilson(0,0)))
        self.assertLess(abs(H.wilson(0,100)[0]),1e-12)
