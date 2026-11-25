"""Padding check: cap determinism, equal-shape batching and the worker path."""
import hashlib
import tempfile
import unittest
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from cvdl.archive import padding
from cvdl.archive.padding import (CaseTasks, case_shape, check, geometry, pair_table, select_cases)
from lsb_train_grid import pad_crop


def fake_case(dataset,row,pattern,level):
    """Deterministic stand-in for diagnostic_pair (module level so spawn workers can import it)."""
    identity=str(dataset.frame.iloc[row]['cover_id'])
    (y,x,h,w),pad=geometry(identity,pattern,level)
    seed=int.from_bytes(hashlib.sha256(f'{identity}/{pattern}/{level}'.encode()).digest()[:4],'little')
    rng=np.random.default_rng(seed)
    clean=rng.integers(0,256,(h,w,3),dtype=np.uint8);stego=clean^1
    record=dict(cover_id=identity,pattern=pattern,actual_fraction=1-h*w/512**2,real_shape=[h,w])
    return (clean,stego),(pad_crop(clean,pad),pad_crop(stego,pad)),record


class Dataset:
    def __init__(self,n):
        self.frame=pd.DataFrame(dict(cover_id=[f'c{i:03d}' for i in range(n)],source='raise',k=1))
        self.entries=[({'photo_id':f'p{i//3}'},0,0,1,None) for i in range(n)]
    def __len__(self):return len(self.frame)


class Net(torch.nn.Module):
    """Mean-pooled, shape-agnostic; same output for any batch composition."""
    def forward(self,x):
        m=x.mean(dim=(1,2,3),keepdim=False)/255
        return torch.stack([-m,m],1)


class PaddingCheckTests(unittest.TestCase):
    def test_cap_is_deterministic_per_group_and_order_independent(self):
        data=Dataset(400)
        first=select_cases(data.frame,25);again=select_cases(data.frame,25)
        self.assertEqual(first,again)
        groups={}
        for row,pattern,level in first:
            groups.setdefault((pattern,level,case_shape(data.frame.cover_id[row],pattern,level)),[]).append(row)
        for key,rows in groups.items():self.assertEqual(len(rows),min(25,len(rows)),key)
        self.assertTrue(all(len(rows)==25 for rows in groups.values()))
        # Identity-keyed: shuffling the list changes the rows, not the chosen cover IDs.
        shuffled=data.frame.sample(frac=1,random_state=0).reset_index(drop=True)
        ids=lambda frame,cases:{(frame.cover_id[r],p,l) for r,p,l in cases}
        self.assertEqual(ids(data.frame,first),ids(shuffled,select_cases(shuffled,25)))
        # Off (None or 0) keeps every case; the capped set is a subset of it.
        everything=select_cases(data.frame,0)
        self.assertEqual(len(everything),400*6);self.assertEqual(select_cases(data.frame,None),everything)
        self.assertTrue(set(first)<=set(everything))
        text,sha=pair_table(data.frame,first);self.assertEqual(sha,hashlib.sha256(text.encode()).hexdigest())
        self.assertEqual(pair_table(data.frame,again)[1],sha)
        self.assertNotEqual(pair_table(data.frame,select_cases(data.frame,10))[1],sha)

    def test_workers_match_single_process_and_batches_are_equal_shape(self):
        data=Dataset(30);net=Net()
        with tempfile.TemporaryDirectory() as a,tempfile.TemporaryDirectory() as b:
            shapes=[]
            original=padding.probabilities
            def spy(network,images,device):
                shapes.append({im.shape for im in images});return original(network,images,device)
            padding.probabilities=spy
            try:
                one=check(net,data,pd.DataFrame(dict(label=[0,1],prob_stego=[.2,.8],cover_id='x',source='raise',k=1)),
                          pd.DataFrame(dict(label=[0,1],prob_stego=[.2,.8])),a,device='cpu',quick=True,limit=8,workers=0,
                          batch_cases=4,case_fn=fake_case)
            finally:padding.probabilities=original
            self.assertTrue(all(len(s)==1 for s in shapes))  # no hidden collation padding
            many=check(net,data,pd.DataFrame(dict(label=[0,1],prob_stego=[.2,.8],cover_id='x',source='raise',k=1)),
                       pd.DataFrame(dict(label=[0,1],prob_stego=[.2,.8])),b,device='cpu',quick=True,limit=8,workers=2,
                       batch_cases=4,case_fn=fake_case)
            for key in ('pairs_sha256','padding_pairs_sha256','cases'):self.assertEqual(one[key],many[key])
            x=pd.read_csv(Path(a)/'padding-check/predictions.csv');y=pd.read_csv(Path(b)/'padding-check/predictions.csv')
            pd.testing.assert_frame_equal(x,y)
            self.assertEqual(len(x),2*one['cases']);self.assertTrue(one['cases']<30*6)
            self.assertEqual((Path(a)/'padding-check/padding-pairs.csv').read_text(),
                             (Path(b)/'padding-check/padding-pairs.csv').read_text())
            # Row order is case order (row, pattern, level), as in the old single-process check.
            order=x.drop_duplicates(['cover_id','pattern','padding'])[['cover_id']].cover_id.tolist()
            self.assertEqual(order,sorted(order))
            # Scores equal a direct one-case-at-a-time scoring.
            cover=x.iloc[0];real,padded,_=fake_case(data,int(cover.cover_id[1:]),cover.pattern,cover.padding)
            self.assertAlmostEqual(cover.prob_stego,padding.probabilities(net,padded,'cpu')[int(cover.label)],places=6)
            self.assertAlmostEqual(cover.unpadded_prob_stego,padding.probabilities(net,real,'cpu')[int(cover.label)],places=6)

if __name__=='__main__':unittest.main()
