"""CPU checks for frozen whole-photo aggregation."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from cvdl.photo_verdict import (features, largest_group, fit_logistic, predict,
                                calibrate_count, evaluate_scope)
from cvdl.cell_metrics import fa_cutoff


class PhotoVerdictTests(unittest.TestCase):
    def test_synthetic_features(self):
        grid=np.array([[1,0,0],[0,1,0]])
        result,counts=features([.1,.6,.8,.9],grid,.5,.5)
        self.assertEqual(result['cell_fraction'],2/6)
        self.assertEqual(result['group_fraction'],2/6)
        self.assertEqual(result['log_group_count'],np.log1p(2))
        self.assertEqual(result['log_cell_count'],np.log1p(6))
        self.assertEqual(result['max_tile'],.9)
        self.assertAlmostEqual(result['top3_tile'],(.6+.8+.9)/3)
        self.assertEqual(result['tile_fraction'],.75)
        self.assertEqual(result['log_tile_count'],np.log1p(4))
        self.assertEqual(counts['tiles_above'],3)
        result,_=features([.5],np.zeros((1,1)),.5,.5)
        self.assertEqual(result['top3_tile'],.5)
        self.assertEqual(result['log_group_count'],0)
        self.assertEqual(result['group_fraction'],0)

    def test_groups_diagonal_and_empty(self):
        self.assertEqual(largest_group(np.eye(4)),4)
        self.assertEqual(largest_group([[1,0,0,1],[0,1,0,1]]),2)
        self.assertEqual(largest_group(np.zeros((3,4))),0)
        self.assertEqual(largest_group(np.ones((2,3))),6)

    def test_cutoff_ties_and_smallest_integer_count(self):
        scores=np.r_[np.zeros(212),np.ones(12)]
        cutoff,fa=fa_cutoff(scores)
        self.assertEqual(cutoff,np.nextafter(1.,np.inf))
        self.assertEqual(fa,0)
        n,fa=calibrate_count(scores)
        self.assertEqual(n,2)
        self.assertEqual(fa,0)
        counts=np.r_[np.zeros(213),np.full(11,3)]
        n,fa=calibrate_count(counts)
        self.assertEqual(n,1)
        self.assertEqual(fa,11/224)
        self.assertGreater((counts>=n-1).mean(),.05)

    def test_logistic_toy_cpu(self):
        x=np.array([[-3,1],[-2,1],[-1,1],[1,1],[2,1],[3,1]],float)
        y=np.array([0,0,0,1,1,1])
        model=fit_logistic(x,y,np.ones(6))
        self.assertEqual(model['scale'][1],1)
        self.assertGreater(model['coefficients'][0],0)
        self.assertLess(abs(model['coefficients'][1]),1e-10)
        self.assertLess(abs(model['intercept']),1e-7)
        np.testing.assert_array_equal(predict(model,x)>=0,y)
        self.assertLessEqual(model['iterations'],1000)

    def test_photo_balancing_and_paired_differences(self):
        rows=[]
        for photo,variants in [('one',1),('two',3)]:
            rows.append(dict(photo_id=photo,panel='panel',label=0,a=0,b=0,c=0))
            for i in range(variants):
                rows.append(dict(photo_id=photo,panel='panel',label=1,a=i,b=i,c=2))
        result=evaluate_scope(rows,dict(a=1,b=1,c=1),replicates=20)
        self.assertEqual(result['rules']['a']['FA'],0)
        self.assertAlmostEqual(result['rules']['a']['miss'],2/3)
        self.assertEqual(result['rules']['c']['P_E'],0)
        self.assertEqual(result['rules']['c']['AUC'],1)
        self.assertEqual(result['differences']['c-a'],result['differences']['c-b'])

    def test_scope_bootstrap_draws_full_population(self):
        rows=[dict(photo_id='eligible',panel='p',label=0,a=0,b=0,c=0),
              dict(photo_id='eligible',panel='p',label=1,a=1,b=1,c=1)]
        result=evaluate_scope(rows,dict(a=.5,b=.5,c=.5),replicates=100,
            population={'eligible':'p','ineligible':'p'})
        # Some full-population samples contain no eligible partner. They must
        # be ineligible replicates, not replaced by draws from eligible only.
        self.assertLess(result['rules']['c']['P_E_interval']['valid_replicates'],100)
        self.assertEqual(result['rules']['c']['P_E'],0)

    def test_all_12_forced_draw_settings(self):
        from dataclasses import replace
        from lsb_batch import read_config
        from lsb_settings import draw_otf_settings
        root=Path(__file__).resolve().parents[1]
        config=read_config(root/'configs/train-k1to4.yaml')
        image=np.zeros((256,256,3),dtype=np.uint8)
        for k in range(1,5):
            for placement in ('contiguous','random','stride'):
                cfg=replace(config,k=k,placement=placement,stride=config.stride if placement=='stride' else 1)
                kwargs,metadata=draw_otf_settings(cfg,image,np.random.default_rng(k))
                self.assertEqual(kwargs['k'],k)
                self.assertEqual(kwargs['placement'],placement)
                self.assertTrue(2<=kwargs['stride']<=16 if placement=='stride' else kwargs['stride']==1)
                self.assertTrue(.05<=metadata['requested_bpp']<=.5)
                self.assertLessEqual(metadata['effective_bpp'],metadata['requested_bpp']+8/(256*256))

    def test_fit_freezes_without_opening_test(self):
        from scripts.photo_verdict_fit import fit
        from cvdl.photo_verdict import FEATURES
        rows=[]
        for photo in range(224):
            for variant in range(13):
                value=(photo%7)/100+bool(variant)
                rows.append(dict(label=int(bool(variant)),features={f:value for f in FEATURES},
                    counts=dict(tiles_above=int(bool(variant)))))
        with tempfile.TemporaryDirectory() as directory:
            output=Path(directory)
            (output/'photo-list.json').write_text(json.dumps(dict(tile_cutoff=.5,head_cutoff=.3)))
            (output/'cache-identity.json').write_text('{}')
            with patch('scripts.photo_verdict_fit.load_partition',return_value=(rows,{})) as loader:
                fit(output)
                self.assertEqual([call.args[2] for call in loader.call_args_list],['fit','calibration'])
                frozen=json.loads((output/'frozen-rules.json').read_text())
                self.assertEqual(frozen['rules']['a']['N'],1)
                self.assertEqual(frozen['rules']['c']['features'],list(FEATURES))
                self.assertEqual(frozen['fit_photos'],224)
                self.assertTrue(frozen['rules']['c']['calibration_clean_FA']<=.05)
                with self.assertRaises(ValueError):fit(output)

    def test_score_refuses_unfrozen_test(self):
        from scripts.photo_verdict_fit import score
        with tempfile.TemporaryDirectory() as directory:
            output=Path(directory)
            with patch('scripts.photo_verdict_fit.load_partition') as loader:
                with self.assertRaises(FileNotFoundError):score(output)
                loader.assert_not_called()


if __name__=='__main__':unittest.main()
