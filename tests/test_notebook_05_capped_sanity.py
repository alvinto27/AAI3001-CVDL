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
        notebook=Path(__file__).resolve().parents[1]/'notebooks/05_xunet_v4_nonorm.ipynb'
        cells=json.loads(notebook.read_text())['cells']
        cell=next(''.join(c['source']) for c in cells if 'class SteganalysisModule' in ''.join(c['source']))
        model=next(node for node in ast.parse(cell).body
                   if isinstance(node,ast.ClassDef) and node.name=='SteganalysisModule')
        hook=next(node for node in model.body
                  if isinstance(node,ast.FunctionDef) and node.name=='on_validation_epoch_end')
        namespace=dict(torch=torch,np=np,V3_DATA='otf',
                       otf_data=SimpleNamespace(checkpoint_validation='capped'),
                       probability_error=lambda labels,scores:float(np.mean(scores)),
                       auc_numpy=lambda labels,scores:.5)
        exec(compile(ast.Module(body=[hook],type_ignores=[]),str(notebook),'exec'),namespace)
        def sample(sanity,source_names):
            logged={}
            obj=SimpleNamespace(trainer=SimpleNamespace(sanity_checking=sanity),
                validation_scores=[torch.tensor([.1]*len(source_names)+[.9]*len(source_names))],
                validation_labels=[torch.tensor([0]*len(source_names)+[1]*len(source_names))],
                validation_sources=[np.array(source_names*2)],
                validation_losses=[torch.ones(2*len(source_names))],
                log=lambda key,value,**kwargs:logged.update({key:value}))
            return obj,logged
        sanity,logged=sample(True,['raise','raise'])
        namespace['on_validation_epoch_end'](sanity)
        self.assertIn('val_pe',logged)
        self.assertFalse(any(key.startswith(('val_pe_','val_loss_')) for key in logged))
        self.assertEqual(sanity.validation_sources,[])
        real,logged=sample(False,['raise','raise'])
        with self.assertRaisesRegex(ValueError,'no alaska examples'):
            namespace['on_validation_epoch_end'](real)
        complete,logged=sample(False,['raise','alaska','lsdir'])
        namespace['on_validation_epoch_end'](complete)
        self.assertEqual({key for key in logged if key.startswith(('val_pe_','val_loss_'))},
                         {f'val_{metric}_{source}' for metric in ('pe','loss')
                          for source in ('raise','alaska','lsdir')})


if __name__=='__main__':
    unittest.main()
