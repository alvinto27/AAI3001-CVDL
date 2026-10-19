"""Saved-score joins, shared photo draws, frozen cut-offs and AUC-only rule."""
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
import numpy as np
import pandas as pd
from cvdl.metrics import auc_numpy
from scripts.architecture_decision import (NAMES, REPLICATES, SEED, auc_intervals,
    join_predictions, literal_checks, rates)


def fixture():
    rows = []
    for photo in range(4):
        for pair in range(2):
            for label in [0, 1]:
                rows.append(dict(pair_id=f'p{photo}-{pair}', source_photo=f'photo{photo}',
                    label=label, placement=['random', 'contiguous'][pair],
                    cover_id=f'cover{photo}', srm30=(photo + 2*label)/7,
                    delayed=(photo + 3*label)/8))
    return pd.DataFrame(rows)


class ArchitectureDecisionTests(unittest.TestCase):
    def write_pair(self, folder, frame=None):
        frame = fixture() if frame is None else frame
        paths = [Path(folder)/f'{name}.csv' for name in NAMES]
        for name, path in zip(NAMES, paths):
            frame.drop(columns=list(NAMES)).assign(prob_stego=frame[name]).iloc[::-1].to_csv(path, index=False)
        return paths

    def test_join_preserves_pairs_and_repeated_clean_tiles(self):
        with TemporaryDirectory() as tmp:
            frame = join_predictions(self.write_pair(tmp))
            self.assertEqual(len(frame),16)
            self.assertEqual(frame.source_photo.nunique(),4)
            self.assertEqual(len(frame[(frame.cover_id=='cover0') & (frame.label==0)]),2)
            self.assertTrue(frame.groupby('pair_id').label.apply(lambda s: sorted(s)==[0,1]).all())

    def test_key_duplicate_or_missing_metadata_and_score_fail(self):
        for mutation, message in [
            (lambda f:pd.concat([f,f.iloc[:1]]),'duplicate'),
            (lambda f:f.iloc[1:],'mismatch'),
            (lambda f:f.assign(source_photo='wrong'),'metadata'),
            (lambda f:f.assign(prob_stego=np.nan),'invalid'),
            (lambda f:f.assign(pair_id=np.nan),'missing')]:
            with self.subTest(message=message), TemporaryDirectory() as tmp:
                paths=self.write_pair(tmp);f=pd.read_csv(paths[1]);mutation(f).to_csv(paths[1],index=False)
                with self.assertRaisesRegex(ValueError,message):join_predictions(paths)

    def test_incomplete_pair_refused_even_when_both_models_match(self):
        with TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError,'one clean and one stego'):
                join_predictions(self.write_pair(tmp,fixture().iloc[1:]))

    def test_photo_shared_draws_match_manual_reference_exactly(self):
        frame=fixture();result,draws=auc_intervals(frame,replicates=31,seed=SEED)
        rng=np.random.default_rng(SEED)
        groups=[g.index.to_numpy() for _,g in frame.groupby('source_photo',sort=True)]
        reference=[]
        for _ in range(31):
            indices=np.concatenate([groups[i] for i in rng.integers(0,4,4)])
            reference.append([auc_numpy(frame.label.to_numpy()[indices],frame[n].to_numpy()[indices]) for n in NAMES])
        np.testing.assert_array_equal(draws,reference)
        lo,hi=np.quantile(draws[:,1]-draws[:,0],[.025,.975])
        self.assertEqual((result['difference']['low'],result['difference']['high']),(lo,hi))
        self.assertEqual((result['pairs'],result['photos']),(8,4))

    def test_deterministic_and_paired_identical_models_have_zero_difference(self):
        frame=fixture();frame['delayed']=frame.srm30
        first,a=auc_intervals(frame,replicates=40)
        second,b=auc_intervals(frame.iloc[::-1],replicates=40)
        np.testing.assert_array_equal(a,b)
        self.assertEqual(first,second)
        self.assertEqual(first['difference'],dict(point=0.,low=0.,high=0.))
        self.assertEqual((REPLICATES,SEED),(2000,20261019))

    def test_ties_are_half_credit(self):
        frame=fixture();frame[list(NAMES)]=.5
        result,draws=auc_intervals(frame,replicates=7)
        np.testing.assert_array_equal(draws,np.full((7,2),.5))
        self.assertEqual(result['models']['srm30']['point'],.5)

    def test_rates_use_own_frozen_cutoffs_and_ge_ties(self):
        frame=pd.DataFrame(dict(label=[0,0,1,1],srm30=[.5,.4,.5,.2],delayed=[.7,.4,.7,.8]))
        actual=rates(frame,{'srm30':.5,'delayed':.7})
        self.assertEqual((actual['srm30']['fa'],actual['srm30']['miss'],actual['srm30']['pe']),(.5,.5,.5))
        self.assertEqual((actual['delayed']['fa'],actual['delayed']['miss'],actual['delayed']['pe']),(.5,0.,.25))

    def test_literal_rule_exact_boundaries_and_auc_guard_direction(self):
        main=dict(point=.03,low=.001,high=.07)
        self.assertTrue(all(literal_checks(main,dict(point=-.001,low=-.02,high=0.)).values()))
        self.assertFalse(literal_checks(dict(main,low=0.),dict(high=.02))['pooled_difference_interval_above_zero'])
        self.assertFalse(literal_checks(dict(main,point=.029999),dict(high=.02))['pooled_point_gain_at_least_003'])
        self.assertFalse(literal_checks(main,dict(high=-.000001))['standard_auc_difference_interval_not_wholly_below_zero'])
        self.assertTrue(literal_checks(main,dict(high=.000001))['standard_auc_difference_interval_not_wholly_below_zero'])


if __name__=='__main__':unittest.main()
