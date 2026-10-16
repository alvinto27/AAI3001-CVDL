"""Partial Lightning sanity validation cannot require all capped sources."""
import ast
import json
from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np
import torch


class CappedSanityTests(unittest.TestCase):
    def test_partial_sanity_skips_source_metrics_but_real_epoch_requires_all(self):
        from cvdl.lit import SteganalysisModule
        hook = SteganalysisModule.on_validation_epoch_end
        def sample(sanity,source_names):
            logged={}
            obj=SimpleNamespace(trainer=SimpleNamespace(sanity_checking=sanity),
                validation_data=SimpleNamespace(checkpoint_validation='capped'),
                validation_scores=[torch.tensor([.1]*len(source_names)+[.9]*len(source_names))],
                validation_labels=[torch.tensor([0]*len(source_names)+[1]*len(source_names))],
                validation_sources=[np.array(source_names*2)],
                validation_losses=[torch.ones(2*len(source_names))],
                log=lambda key,value,**kwargs:logged.update({key:value}))
            return obj,logged
        sanity,logged=sample(True,['raise','raise'])
        hook(sanity)
        self.assertIn('val_pe',logged)
        self.assertFalse(any(key.startswith(('val_pe_','val_loss_')) for key in logged))
        self.assertEqual(sanity.validation_sources,[])
        real,logged=sample(False,['raise','raise'])
        with self.assertRaisesRegex(ValueError,'no alaska examples'):
            hook(real)
        complete,logged=sample(False,['raise','alaska','lsdir'])
        hook(complete)
        self.assertEqual({key for key in logged if key.startswith(('val_pe_','val_loss_'))},
                         {f'val_{metric}_{source}' for metric in ('pe','loss')
                          for source in ('raise','alaska','lsdir')})


if __name__=='__main__':
    unittest.main()
