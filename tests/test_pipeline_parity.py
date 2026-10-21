"""Small CPU parity replay on local VALIDATION artifacts, never TEST."""
import json
from pathlib import Path
import unittest

import numpy as np
import torch

from cvdl.pipeline import load_pipeline
from cvdl.photo_verdict import FEATURES, features, predict
from scripts.photo_heatmap import infer_photo

ROOT=Path(__file__).resolve().parents[1]
OUTPUT=ROOT/'runs/verification/photo-verdict-run13-2026-10-21'


class PipelineValidationParityTests(unittest.TestCase):
    def test_three_validation_photos_clean_and_fixed_stego(self):
        if not (OUTPUT/'photo-list.json').exists():self.skipTest('local validation feature cache absent')
        from lsb_source import load_photo
        from lsb_engine import embed_fast
        selection=json.loads((OUTPUT/'photo-list.json').read_text())
        chosen=[]
        for source in ('div2k','alaska','lsdir'):
            candidates=[p for p in selection['photos'] if p['partition']=='fit' and p['record']['source']==source
                        and (int(p['record']['source_height'])%32 or int(p['record']['source_width'])%32)]
            if not candidates:
                candidates=[p for p in selection['photos'] if p['partition']=='fit' and p['record']['source']==source]
            chosen.append(min(candidates,key=lambda p:(p['tiles'],p['record']['photo_id'])))
        if any(not (ROOT/p['record']['master_path']).exists() for p in chosen):self.skipTest('local validation pixels absent')
        torch.set_num_threads(1)
        pipeline=load_pipeline(device='cpu')
        frozen=json.loads((OUTPUT/'frozen-rules.json').read_text())
        for index,selected in enumerate(chosen):
            record=selected['record'];clean=load_photo(record,ROOT)
            cached=json.loads((OUTPUT/'features/fit'/(record['photo_id']+'.json')).read_text())['rows']
            for variant in (0,index+1):  # contiguous/random/stride; one fixed version per photo
                with self.subTest(photo=record['photo_id'],variant=variant):
                    if variant:
                        settings=dict(cached[variant]['settings'])
                        settings['rng']=np.random.default_rng(np.random.SeedSequence(selected['versions'][variant]['embedding_seed']))
                        photo,_,_=embed_fast(clean,**settings)
                    else:photo=clean
                    origins,scores,maps=infer_photo(pipeline._network,photo,record,'cpu',8)
                    result=pipeline.analyze(photo)
                    self.assertEqual([r['box'] for r in result['tiles']],[[x,y,x+256,y+256] for y,x in origins])
                    # Shared helper two-map path vs selected-map public path.
                    np.testing.assert_array_equal([r['probability'] for r in result['tiles']],scores)
                    np.testing.assert_array_equal(result['heatmap']['values'],maps[0])
                    values,counts=features(scores,maps[0],frozen['tile_cutoff'],frozen['head_cutoff'])
                    self.assertEqual(result['features'],values)
                    self.assertEqual(result['counts'],counts)
                    logit=float(predict(frozen['rules']['c']['model'],[[values[f] for f in FEATURES]])[0])
                    self.assertEqual(result['photo_rule']['logit'],logit)
                    self.assertEqual(result['photo_verdict']=='message_detected',logit>=frozen['rules']['c']['cutoff'])
                    # Saved cache came from GPU; predeclared float tolerances,
                    # exact counts/flags. No threshold or coefficient adjustment.
                    np.testing.assert_allclose([values[f] for f in FEATURES],
                        [cached[variant]['features'][f] for f in FEATURES],rtol=1e-5,atol=1e-6)
                    self.assertEqual(result['counts'],cached[variant]['counts'])
                    saved_logit=float(predict(frozen['rules']['c']['model'],
                        [[cached[variant]['features'][f] for f in FEATURES]])[0])
                    self.assertAlmostEqual(logit,saved_logit,delta=1e-5)
                    self.assertEqual(logit>=frozen['rules']['c']['cutoff'],saved_logit>=frozen['rules']['c']['cutoff'])
                    print('Validation parity:',record['photo_id'],'variant',variant,'tiles',len(scores))


if __name__=='__main__':unittest.main()
