"""Temperature scaling: fit, ECE, ties, per-source summary."""
import unittest
import numpy as np
import pandas as pd
from cvdl.archive.calibration import ece, fit_temperature, nll, sigmoid, summarise, tie_share


class CalibrationTests(unittest.TestCase):
    def test_fit_recovers_known_temperature(self):
        rng=np.random.default_rng(0);n=40000
        true=rng.normal(size=n)*3;y=(rng.random(n)<sigmoid(true)).astype(int)
        T=3.0;logits=true*T  # over-confident by a factor T
        self.assertAlmostEqual(fit_temperature(logits,y),T,delta=.15)
        self.assertLess(nll(logits,y,fit_temperature(logits,y)),nll(logits,y,1.0))
        self.assertAlmostEqual(fit_temperature(true,y),1.0,delta=.05)

    def test_ece_and_ties(self):
        y=np.array([0,0,1,1]);self.assertAlmostEqual(ece(np.array([.5,.5,.5,.5]),y),0.0)
        self.assertGreater(ece(np.array([1.,1.,1.,1.]),y),.4)
        t=tie_share(np.array([1.0,1-1e-7,.5,.9,1.0]),np.array([0,0,0,0,1]))
        self.assertEqual(t['clean'],4);self.assertAlmostEqual(t['at_one'],.25);self.assertAlmostEqual(t['above_1_minus_1e6'],.5)

    def test_ties_resolved_by_logits_and_saturated_logits_stay_tied(self):
        rng=np.random.default_rng(1);n=2000
        label=np.r_[np.zeros(n),np.ones(n)].astype(int)
        logit=np.r_[rng.normal(-2,8,n),rng.normal(4,8,n)]  # float32 softmax saturates above ~17
        saved=np.float32(sigmoid(logit)).astype(np.float64)
        frame=pd.DataFrame(dict(source='alaska',k=1,label=label,logit=logit,prob_stego=saved))
        result=summarise(frame,T=5.0)['alaska']
        self.assertGreater(result['ties_before']['at_one'],0.0)
        self.assertLess(result['ties_after']['at_one'],result['ties_before']['at_one'])
        self.assertEqual(result['fa5_after']['fa'],result['fa5_after']['fa'])
        self.assertIn('all',summarise(frame,T=5.0))
        # Huge logits cannot be fixed by T: identical logits remain tied.
        flat=frame.assign(logit=np.where(label==0,200.,200.),prob_stego=1.0)
        r=summarise(flat,T=5.0)['alaska'];self.assertEqual(r['ties_after']['at_one'],1.0)
        self.assertEqual(r['logits']['distinct_clean_logits'],1)

if __name__=='__main__':unittest.main()
