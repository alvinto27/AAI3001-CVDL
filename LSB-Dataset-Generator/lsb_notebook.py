"""Notebook adapters; no source transforms or embedding logic live here."""
from __future__ import annotations
import dataclasses
import json
import numpy as np
import pandas as pd
from lsb_torch import (torch,RunSettings,Corpus,TrainingImages,FixedPairs,PairSubset,
                       training_loader,DataLoader)


def fixed_subset(dataset,count,seed=0):
    """Stable camera/source-stratified subset, with every nonempty stratum."""
    if count>=len(dataset):
        return dataset
    groups=list(dataset.frame.groupby(["source","camera_model"],sort=True).groups.values())
    rng=np.random.default_rng(seed)
    queues=[list(rng.permutation(list(group))) for group in groups]
    indices=[]
    while len(indices)<count and any(queues):
        for queue in queues:
            if queue and len(indices)<count:
                indices.append(queue.pop())
    return PairSubset(dataset,indices)


class NotebookData:
    def __init__(self,root,settings,*,quick=False,profile="overlap_full_mix",eval_pairs=16):
        self.corpus=Corpus(root,settings)
        self.train=TrainingImages(self.corpus)
        chosen={r["photo_id"] for r in self.corpus.selected}
        self.fixed_train=FixedPairs(self.corpus,"train",profile,photo_ids=chosen)
        self.val=FixedPairs(self.corpus,"val",profile)
        self.test=FixedPairs(self.corpus,"test",profile)
        if quick:
            self.val=fixed_subset(self.val,eval_pairs,0)
            self.test=fixed_subset(self.test,eval_pairs,1)
        self.record={**self.corpus.run_record,"data":"otf","quick":quick,"eval_profile":profile,
                     "quick_eval_pair_ids":{s:d.frame["cover_id"].tolist() if quick else None
                                            for s,d in (("val",self.val),("test",self.test))}}
        self.pairs=pd.concat([self.fixed_train.frame,self.val.frame,self.test.frame],ignore_index=True)

    def loaders(self,batch_size,workers,pin_memory,prefetch=4):
        self.record.update(workers=workers,prefetch_factor=prefetch,in_order=True,group_images=4,
                           batch_images=batch_size,tensor_dtype="uint8")
        kwargs=dict(num_workers=workers,pin_memory=pin_memory,persistent_workers=workers>0)
        if workers:
            kwargs.update(multiprocessing_context="spawn",prefetch_factor=prefetch)
        paired=self.corpus.settings.paired_batches
        train=training_loader(self.train,batch_size=max(1,batch_size//2) if paired else batch_size,
                              num_workers=workers,pin_memory=pin_memory,prefetch_factor=prefetch)
        val=DataLoader(self.val,batch_size=max(1,batch_size//2),**kwargs)
        test=DataLoader(self.test,batch_size=max(1,batch_size//2),**kwargs)
        diagnostic=fixed_subset(self.fixed_train,min(1000,len(self.fixed_train)),0)
        train_eval=DataLoader(diagnostic,batch_size=max(1,batch_size//2),**kwargs)
        return train,val,test,train_eval

    def print_record(self):
        print("OTF run settings:\n"+json.dumps(self.record,indent=2),flush=True)

    def save_record(self,path):
        path.write_text(json.dumps(self.record,indent=2)+"\n")

    def check_epochs(self):
        """Check actual Mode 1 bytes for one stable tile key, plus fixed eval."""
        from lsb_mode1 import TileSpec,OTFSeed,make_sample
        record=self.corpus.selected[0]
        photo=self.corpus.photo(record)
        spec=TileSpec(self.corpus.settings.grid,self.corpus.settings.tile_size,0,0)
        a,_=make_sample(photo,record,spec,self.corpus.config,OTFSeed(self.corpus.settings.run_seed,"train",0),"stego")
        b,_=make_sample(photo,record,spec,self.corpus.config,OTFSeed(self.corpus.settings.run_seed,"train",1),"stego")
        assert not np.array_equal(a,b)
        for name,dataset in (("val",self.val),("test",self.test)):
            before=dataset[0][:2]
            self.train.set_epoch(1)
            after=dataset[0][:2]
            assert all(torch.equal(x,y) for x,y in zip(before,after))
            print(f"EPOCH CHECK {name} unchanged: True")
        self.train.set_epoch(0)
        print("EPOCH CHECK same train tile key stego epoch0 != epoch1: True")


def complete_bpp_bins(values):
    numeric=pd.to_numeric(values,errors="raise")
    if not np.isfinite(numeric).all():
        raise ValueError("nonfinite bpp")
    return pd.cut(numeric,bins=[-np.inf,.05,.15,.30,.50,np.inf],
                  labels=["<=0.05","0.05-0.15","0.15-0.30","0.30-0.50",">0.50"],right=True)
