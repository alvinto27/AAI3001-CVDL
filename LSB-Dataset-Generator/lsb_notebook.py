"""Notebook adapters; no source transforms or embedding logic live here."""
from __future__ import annotations
import dataclasses
import json
import numpy as np
import pandas as pd
from lsb_torch import (torch,RunSettings,Corpus,TrainingImages,FixedPairs,PairSubset,
                       training_loader,DataLoader,prefetch_for_images)


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

    def loaders(self,batch_size,workers,pin_memory,prefetch=None,*,eval_workers=4,prefetch_images=128):
        if workers<0 or eval_workers<0:
            raise ValueError("worker counts must be nonnegative")
        paired=self.corpus.settings.paired_batches
        train_batch=max(1,batch_size//2) if paired else batch_size
        pair_batch=max(1,batch_size//2)
        task_images=min(2 if paired else 4,train_batch)*(2 if paired else 1)
        train_prefetch=prefetch_for_images(task_images,prefetch_images) if prefetch is None else prefetch
        eval_prefetch=prefetch_for_images(pair_batch*2,prefetch_images)
        self.record.update(workers=workers,eval_workers=eval_workers,main_threads=torch.get_num_threads(),
                           prefetch_images=prefetch_images,prefetch_override=prefetch,
                           prefetch_factor=train_prefetch,train_prefetch_images=task_images*train_prefetch,
                           eval_prefetch_factor=eval_prefetch,eval_persistent_workers=False,
                           in_order=True,group_images=4,batch_images=batch_size,
                           tensor_dtype=self.corpus.settings.tensor_dtype)
        # Evaluation is intermittent: release workers/photo arrays after each pass.
        kwargs=dict(num_workers=eval_workers,pin_memory=pin_memory,persistent_workers=False)
        if eval_workers:
            kwargs.update(multiprocessing_context="spawn",prefetch_factor=eval_prefetch)
        train=training_loader(self.train,batch_size=train_batch,num_workers=workers,
                              pin_memory=pin_memory,prefetch_factor=prefetch,prefetch_images=prefetch_images)
        val=DataLoader(self.val,batch_size=pair_batch,**kwargs)
        test=DataLoader(self.test,batch_size=pair_batch,**kwargs)
        diagnostic=fixed_subset(self.fixed_train,min(1000,len(self.fixed_train)),0)
        train_eval=DataLoader(diagnostic,batch_size=pair_batch,**kwargs)
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
