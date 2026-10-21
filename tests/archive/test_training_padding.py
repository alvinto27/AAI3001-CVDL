"""Archived 512 padding diagnostic geometry and frozen-reference controls."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np


class PaddingDiagnosticTests(unittest.TestCase):


    def test_padding_strip_corner_controls_geometry(self):
        from cvdl.archive.padding import geometry
        for pattern in ('strip','corner'):
            for level in (.1,.25,.5):
                rect,pad=geometry('parent',pattern,level);y,x,h,w=rect
                self.assertEqual(h+pad[0]+pad[1],512);self.assertEqual(w+pad[2]+pad[3],512)
                self.assertEqual(sum(v>0 for v in pad),1 if pattern=='strip' else 2)
                self.assertFalse(pad[0] and pad[1]);self.assertFalse(pad[2] and pad[3])
                self.assertLess(abs(1-h*w/512**2-level),.003)
                self.assertEqual(geometry('parent',pattern,level),(rect,pad))

    def test_padding_report_contains_both_controls_and_frozen_reference(self):
        import pandas as pd
        from types import SimpleNamespace
        from cvdl.archive.padding import check
        from unittest.mock import Mock
        class Pair:
            frame=pd.DataFrame([dict(cover_id='a',source='raise',k=1)])
            entries=[({'photo_id':'p'},0,0,1,None)]
            def __len__(self):return 1
        predictions=pd.DataFrame([dict(cover_id='a',source='raise',k=1,label=label,prob_stego=score)
                                  for label,score in ((0,.2),(1,.8))])
        pixels=np.zeros((256,512,3),dtype=np.uint8)
        with tempfile.TemporaryDirectory() as folder, \
             patch('cvdl.archive.padding.diagnostic_pair',return_value=((pixels,pixels),(pixels,pixels),{'actual_fraction':.5})), \
             patch('cvdl.archive.padding.probabilities',side_effect=lambda net,images,device:[.2,.8]*(len(images)//2)):
            result=check(Mock(),Pair(),predictions,predictions,folder,device='cpu',quick=True)
            self.assertEqual(result['summary'][0]['control'],'frozen')
            self.assertEqual({row['control'] for row in result['summary']},{'frozen','padded','unpadded'})
            rows=pd.read_csv(Path(folder)/'padding-check/predictions.csv')
            self.assertEqual(len(rows),12);self.assertTrue((rows.delta==0).all())
