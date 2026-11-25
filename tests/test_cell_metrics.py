"""Focused synthetic guards for cell localization and saved-score comparison."""
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np
import pandas as pd
from cvdl.cell_metrics import fa_cutoff,weighted_auc,localization,photo_auc_bootstrap
from scripts.compare_cell_tiles import compare,joined
from scripts.cell_eval import report


class CellMetricTests(unittest.TestCase):
    def test_fa_cutoff_keeps_ties_together_and_does_not_skip_intermediate_positives(self):
        cutoff,fa=fa_cutoff(np.arange(20))
        self.assertEqual(fa,.05);self.assertTrue(18<cutoff<18.5)
        cutoff,fa=fa_cutoff([0]*18+[1]*2)
        self.assertGreater(cutoff,1);self.assertEqual(fa,0)
        with self.assertRaises(ValueError):fa_cutoff([])
        with self.assertRaises(ValueError):fa_cutoff([np.nan])

    def test_tied_weighted_auc_and_shared_photo_draws(self):
        labels=np.array([0,1,1,0]);scores=np.array([.5,.5,.8,.2])
        self.assertAlmostEqual(weighted_auc(labels,scores),.875)
        weights=np.array([2,2,1,1])
        self.assertEqual(weighted_auc(labels,scores,weights),weighted_auc(np.repeat(labels,weights),np.repeat(scores,weights)))
        point,draws=photo_auc_bootstrap(labels,np.stack([scores,scores],axis=1),['a','a','b','b'],replicates=32)
        np.testing.assert_array_equal(draws[:,0],draws[:,1]);self.assertEqual(point[0],point[1])
        self.assertIsNone(weighted_auc([1,1],[0,1]))

    def test_within_tile_auc_rejects_between_image_shortcut_and_random_is_ineligible(self):
        counts=np.zeros((3,64),int);counts[0,0]=1;counts[1,:63]=1;counts[2,:]=1
        neg=counts==0
        cam=np.stack([np.zeros(64),np.ones(64),np.ones(64)*2])
        aux=(counts>0).astype(float)
        result=localization(counts,neg,np.stack([cam,aux],axis=1),['a','b','c'],replicates=32)
        self.assertEqual(result['eligible_tiles'],2);self.assertEqual(result['ineligible_tiles'],1)
        self.assertEqual(result['maps'][0]['within_stego_auc']['point'],.5)
        self.assertGreater(result['maps'][0]['pooled_cell_auc']['point'],.9)
        self.assertEqual(result['aux_minus_cam']['within_stego_auc']['low'],.5)
        random=localization(counts[2:],neg[2:],np.stack([cam[2:],aux[2:]],axis=1),['c'],replicates=32)
        self.assertIsNone(random['maps'][0]['within_stego_auc'])

    def test_cell_report_validation_only_cutoff_float32_ties_and_fa_guard(self):
        rows=[dict(diagnostic_panel='raise',placement=p,source_photo=p,method='replace',bit_mode='plane',payload_kind='letters')
            for p in ('contiguous','random')]
        counts=np.zeros((2,8,8),int);counts[0,0,0]=1;counts[1]=1
        negative=counts==0;negative[0,0,1]=False  # halo cell, not a negative
        aux=np.stack([np.zeros_like(counts),counts>0],axis=1).astype(np.float32)
        arrays=dict(counts=counts,valid_negative=negative,cam=aux,aux=aux)
        first=report(rows,rows,arrays,arrays,replicates=16)
        self.assertTrue(first['auxiliary_target_mark'])
        self.assertEqual(first['maps']['aux']['test'][0]['negative_cells'],190)
        self.assertEqual(first['maps']['aux']['test'][0]['FA'],0)
        changed={**arrays,'aux':aux+100}
        second=report(rows,rows,arrays,changed,replicates=16)
        self.assertEqual(first['maps']['aux']['cutoff'],second['maps']['aux']['cutoff'])
        self.assertEqual(first['color_limits'],second['color_limits'])
        self.assertEqual(second['maps']['aux']['test'][0]['FA'],1)
        self.assertFalse(second['auxiliary_target_mark'])

    def test_saved_comparison_own_cutoffs_photo_pairing_and_distinct_noise_union(self):
        with TemporaryDirectory() as folder:
            folders=[Path(folder)/name for name in ('candidate','control')]
            for index,root in enumerate(folders):
                (root/'panels/standard').mkdir(parents=True);(root/'hard-test').mkdir()
                pd.DataFrame([dict(panel='standard',category='overall',value='all',cutoff=.65 if index==0 else .5,
                    threshold_origin='train_matched')]).to_csv(root/'main_summary.csv',index=False)
                def pair(pair_id,panel,condition,cover,clean,stego,k=1):
                    return [dict(pair_id=pair_id,label=y,source_photo=cover,cover_id=cover,panel=panel,k=k,
                        condition=condition,placement='contiguous',prob_stego=s) for y,s in enumerate((clean,stego))]
                rows=pair('standard','standard','all','s',.6 if index==0 else .4,.7 if index==0 else .8)
                pd.DataFrame(rows).to_csv(root/'panels/standard/predictions.csv',index=False)
                rows=[]
                for panel in ('alaska-seen','alaska-unseen','lsdir'):
                    rows+=pair(panel+'/target',panel,'low',panel+'/target',.4,.9 if index==0 else .8)
                    for condition in ('noisy','low_noisy'):
                        rows+=pair(panel+'/'+condition,panel,condition,panel+'/noise',.7 if index==0 else .4,.9,k=2)
                pd.DataFrame(rows).to_csv(root/'hard-test/hard_predictions.csv',index=False)
            result=compare(*folders,replicates=32)
            self.assertEqual(result['standard']['candidate']['P_E'],0)
            self.assertEqual(result['standard']['control']['P_E'],0)
            self.assertEqual(result['hard_target']['candidate_minus_control']['high'],0)
            self.assertEqual(result['noisy_clean']['distinct_tiles'],3)
            self.assertEqual(result['noisy_clean']['candidate_minus_control'],1)
            self.assertFalse(result['noisy_clean']['guard_plus_2pp'])
            path=folders[0]/'hard-test/hard_predictions.csv'
            broken=pd.read_csv(path);broken.loc[0,'source_photo']='wrong';broken.to_csv(path,index=False)
            with self.assertRaises(ValueError):joined(path,folders[1]/'hard-test/hard_predictions.csv')


if __name__=='__main__':unittest.main()
