"""Archived env-aware checkpoint shim; package callers use explicit quick flags."""
import unittest
from unittest.mock import patch
import pandas as pd
from scripts.archive.eval_checkpoint import panel_dataset


class CheckpointShimTests(unittest.TestCase):
    def test_quick_environment_caps_deterministically_and_explicit_false_wins(self):
        class FakePairs:
            frame=pd.DataFrame({'source':['raise']*100,'camera_model':['A','B']*50})
            def __len__(self):return len(self.frame)
        dataset=FakePairs()
        with patch.dict('os.environ',{'QUICK':'true'}):
            small=panel_dataset(dataset)
            self.assertEqual(len(small),16)
            self.assertEqual(small.indices,panel_dataset(dataset).indices)
            self.assertIs(panel_dataset(dataset,quick=False),dataset)
