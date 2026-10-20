import unittest
from pathlib import Path
import sys
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import torch
from cvdl.models import XuNet, XuNetWide
from cvdl.saturation import Stats
from cvdl.archive.tile_saturation import first_tanh_inputs, tile_saturated_share, first_tanh_share
from scripts.archive.check2b_saturation import (rank_quarters, spearman, correlation_interval,
                                      pooled_quarter_difference, join_saved)


class TileSaturationTests(unittest.TestCase):
    def test_strict_threshold_and_original_definition(self):
        x=torch.tensor([[[[3.,-3.,3.01,-3.01]]],[[[0.,2.,4.,5.]]]])
        shares=tile_saturated_share(x)
        self.assertEqual(shares.dtype,torch.float64)
        torch.testing.assert_close(shares,torch.tensor([.5,.5],dtype=torch.float64),rtol=0,atol=0)
        original=Stats();original.add(x)
        self.assertEqual(original.result(False)['share_abs_gt3'],float(shares.mean()))
        self.assertEqual(x.shape,(2,1,1,4))

    def test_prefix_matches_actual_first_tanh_for_both_front_ends(self):
        torch.manual_seed(19)
        images=torch.randint(0,256,(2,3,32,32),dtype=torch.uint8)
        for cls in (XuNet,XuNetWide):
            for front in ('kv','srm'):
                for norm in ('none','group','batch'):
                    with self.subTest(model=cls.__name__,front=front,norm=norm):
                        model=cls(front_end=front,norm=norm).eval();captured={}
                        hook=model.bn1.register_forward_hook(lambda m,i,o:captured.update(input=o.detach().clone()))
                        try:
                            with torch.inference_mode():model.features(images.float())
                        finally:hook.remove()
                        before={k:v.clone() for k,v in model.state_dict().items()}
                        with patch.object(model.classifier,'forward',side_effect=AssertionError('classifier called')):
                            with patch.object(model.conv2,'forward',side_effect=AssertionError('later layer called')):
                                with torch.inference_mode():
                                    actual=first_tanh_inputs(model,images)
                                    torch.testing.assert_close(actual,captured['input'],rtol=0,atol=0)
                                    torch.testing.assert_close(first_tanh_share(model,images),tile_saturated_share(actual),rtol=0,atol=0)
                        for key,value in model.state_dict().items():torch.testing.assert_close(value,before[key],rtol=0,atol=0)

    def test_invalid_activation_shapes(self):
        for x in (torch.zeros(3),torch.zeros(0,1,2,2),torch.zeros(2,1,2,2,dtype=torch.long)):
            with self.assertRaises(ValueError):tile_saturated_share(x)
        with self.assertRaises(ValueError):first_tanh_inputs(XuNet(),torch.zeros(2,1,3,3))

    def test_rank_quarters_equal_values_use_id_only(self):
        frame=pd.DataFrame(dict(pair_id=list('hgfedcba'),value=[0.]*8,prob_stego=range(8)))
        quarters,boundaries=rank_quarters(frame,'value','pair_id')
        self.assertEqual(list(quarters.pair_id),list('abcdefgh'))
        self.assertEqual(list(quarters.quarter),[0,0,1,1,2,2,3,3])
        self.assertTrue(all(b['tied'] and b['tied_group_size']==8 for b in boundaries))
        again,_=rank_quarters(frame.iloc[::-1],'value','pair_id')
        pd.testing.assert_frame_equal(quarters,again)

    def test_rank_quarters_remainder_and_invalid_input(self):
        frame=pd.DataFrame(dict(id=list('abcde'),v=range(5)))
        quarters,_=rank_quarters(frame,'v','id')
        self.assertEqual(list(quarters.quarter),[0,0,1,2,3])
        for bad in (frame.iloc[:3],pd.concat([frame,frame]),frame.assign(v=np.inf)):
            with self.assertRaises(ValueError):rank_quarters(bad,'v','id')

    def test_spearman_ties_constant_and_photo_bootstrap(self):
        self.assertAlmostEqual(spearman([1,1,2,3],[5,5,7,9]),1)
        self.assertAlmostEqual(spearman([1,2,3],[3,2,1]),-1)
        self.assertIsNone(spearman([1,1,1],[1,2,3]))
        frame=pd.DataFrame(dict(source_photo=['a','a','b','b'],tile_kv_std=[1.,2.,3.,4.],saturated_share=[.1,.2,.3,.4]))
        result=correlation_interval(frame,replicates=20)
        self.assertAlmostEqual(result['rho'],1)
        self.assertAlmostEqual(result['low'],1)
        self.assertAlmostEqual(result['high'],1)
        self.assertEqual(result['photos'],2)
        self.assertEqual(result,correlation_interval(frame,replicates=20))
        self.assertIsNone(correlation_interval(frame.assign(saturated_share=0),replicates=10)['low'])

    def test_pooled_fixed_quarters_photo_pairing_and_borderline(self):
        rows=[dict(source_photo=p,quarter=q,missed=(q==3 and i==0))
              for p in ('a','b') for q in (0,3) for i in range(10)]
        result=pooled_quarter_difference(pd.DataFrame(rows),replicates=20)
        self.assertEqual(result['bottom_n'],20)
        self.assertEqual(result['top_n'],20)
        self.assertEqual((result['difference'],result['low'],result['high']),(.1,.1,.1))
        self.assertTrue(result['borderline_10_points'])
        frame=pd.DataFrame(rows).assign(missed=lambda x:x.quarter==3)
        self.assertEqual(pooled_quarter_difference(frame,replicates=20)['difference'],1)
        self.assertFalse(pooled_quarter_difference(frame,replicates=20)['borderline_10_points'])

    def test_report_handles_undefined_correlation(self):
        import tempfile
        from pathlib import Path
        from scripts.archive.check2b_saturation import write_report
        result=dict(cutoff=.5,correlation=dict(rho=None,low=None,high=None,
            distinct_clean_tiles=4,photos=2),pooled_miss_difference=dict(difference=0.,low=0.,high=0.,
            bottom_miss=.5,top_miss=.5,bottom_n=4,top_n=4,borderline_10_points=False))
        with tempfile.TemporaryDirectory() as tmp:
            with patch('scripts.archive.check2b_saturation.OUT',Path(tmp)):
                write_report({'constant':result},[],[],[])
                self.assertIn('Spearman: undefined undefined',(Path(tmp)/'report.md').read_text())

    def test_saved_prediction_join_and_failure_modes(self):
        frozen=pd.DataFrame([dict(pair_id=p,cover_id='c'+p,source_photo='photo'+p,cell='target/k1/noisy',
            tile_kv_std=1.,k=1,condition='noisy',placement='random',method='replace',y=0,x=0) for p in ('a','b')])
        saved=pd.concat([frozen.assign(label=0,prob_stego=.2),frozen.assign(label=1,prob_stego=.7)],ignore_index=True)
        measured=saved[['pair_id','label']].assign(saturated_share=.3)
        joined=join_saved(measured,saved,frozen)
        pd.testing.assert_series_equal(joined.set_index(['pair_id','label']).prob_stego.sort_index(),
                                       saved.set_index(['pair_id','label']).prob_stego.sort_index())
        self.assertEqual(len(joined),4)
        for m,s in ((measured.iloc[:-1],saved),(pd.concat([measured,measured]),saved),
                    (measured,saved.assign(tile_kv_std=2.)),(measured,saved[saved.label==0])):
            with self.assertRaises(ValueError):join_saved(m,s,frozen)


if __name__=='__main__':unittest.main()
