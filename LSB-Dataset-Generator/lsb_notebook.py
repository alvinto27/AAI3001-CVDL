"""Notebook adapters; no source transforms or embedding logic live here."""
from __future__ import annotations
import dataclasses
import hashlib
import json
import numpy as np
import pandas as pd
from lsb_torch import (torch,RunSettings,Corpus,TrainingImages,FixedPairs,PairSubset,
                       training_loader,OrderedPairLoader,prefetch_for_images)


def capped_validation(dataset,raise_cap,lsdir_cap):
    """Seeded pair sampling; preserve the original photo/tile distribution in expectation."""
    if any(type(cap) is not int or cap<1 for cap in (raise_cap,lsdir_cap)):
        raise ValueError("validation caps must be positive integers")
    source=dataset.frame["source"].to_numpy()
    selected=[]
    for name,cap,seed in (("raise",raise_cap,2401),("alaska",None,0),("lsdir",lsdir_cap,2402)):
        available=np.flatnonzero(source==name)
        if not len(available):
            raise ValueError(f"capped validation needs {name} validation pairs")
        chosen=(np.random.default_rng(seed).choice(available,size=cap,replace=False)
                if cap is not None and len(available)>cap else available)
        selected.extend(map(int,chosen))
    return PairSubset(dataset,sorted(selected))


def source_balanced_subset(dataset,count,seed=0):
    """Keep every capped-validation source in the tiny QUICK smoke subset."""
    groups=list(dataset.frame.groupby("source",sort=True).groups.values())
    if count<len(groups):
        raise ValueError("QUICK validation needs one pair per source")
    if count>=len(dataset):
        return dataset
    rng=np.random.default_rng(seed)
    queues=[list(rng.permutation(list(group))) for group in groups]
    indices=[]
    while len(indices)<count and any(queues):
        for queue in queues:
            if queue and len(indices)<count:
                indices.append(queue.pop())
    if isinstance(dataset,PairSubset):
        return PairSubset(dataset.dataset,[dataset.indices[int(i)] for i in indices])
    return PairSubset(dataset,indices)


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
    if isinstance(dataset,PairSubset):
        return PairSubset(dataset.dataset,[dataset.indices[int(i)] for i in indices])
    return PairSubset(dataset,indices)


class NotebookData:
    def __init__(self,root,settings,*,quick=False,profile="overlap_full_mix",eval_pairs=16,
                 checkpoint_validation="standard",val_cap_raise=5000,val_cap_lsdir=1500,train_grid='recorded',train_tile=None):
        self.corpus=Corpus(root,settings,train_grid=train_grid,train_tile=train_tile)
        self.train=TrainingImages(self.corpus)
        chosen={r["photo_id"] for r in self.corpus.selected}
        self.fixed_train=FixedPairs(self.corpus,"train",profile,photo_ids=chosen)
        if checkpoint_validation not in {"standard","combined","capped"}:
            raise ValueError("checkpoint_validation must be standard, combined, or capped")
        if checkpoint_validation=="capped" and not {"raise","alaska","lsdir"}<=set(settings.sources):
            raise ValueError("capped validation needs RAISE, ALASKA, and LSDIR")
        if profile!="overlap_full_mix" and set(settings.sources)&{"alaska","lsdir"}:
            raise ValueError("new-source panels require overlap_full_mix; legacy_v6 is RAISE only")
        self.checkpoint_validation=checkpoint_validation
        if checkpoint_validation=="capped":
            self.val=capped_validation(FixedPairs(self.corpus,"val",profile,
                                                    sources=("raise","alaska","lsdir")),
                                       val_cap_raise,val_cap_lsdir)
        else:
            self.val=FixedPairs(self.corpus,"val",profile,
                                sources=("raise","div2k") if checkpoint_validation=="standard" else None)
        self.test=FixedPairs(self.corpus,"test",profile,sources=("raise","div2k"))
        self.extra_pairs={}
        for source,splits in (("alaska",("val","test_seen","test_unseen")),
                              ("lsdir",("val","test"))):
            if source in settings.sources:
                for split in splits:
                    self.extra_pairs[f"{source}-{split}"]=FixedPairs(self.corpus,split,profile,sources=(source,))
        if quick:
            self.val=(source_balanced_subset(self.val,eval_pairs,0)
                      if checkpoint_validation=="capped" else fixed_subset(self.val,eval_pairs,0))
            self.test=fixed_subset(self.test,eval_pairs,1)
            self.extra_pairs={name:fixed_subset(dataset,eval_pairs,2+i)
                              for i,(name,dataset) in enumerate(self.extra_pairs.items())}
        self.record={**self.corpus.run_record,"data":"otf","quick":quick,"eval_profile":profile,
                     "checkpoint_validation":checkpoint_validation,
                     "extra_pair_counts":{name:len(ds) for name,ds in self.extra_pairs.items()},
                     "quick_eval_pair_ids":{s:d.frame["cover_id"].tolist() if quick else None
                                            for s,d in (("val",self.val),("test",self.test))}}
        if checkpoint_validation=="capped":
            self.validation_pair_csv=self.val.frame[["cover_id","source"]].to_csv(index=False)
            self.record.update(val_cap_raise=val_cap_raise,val_cap_lsdir=val_cap_lsdir,
                               validation_pair_counts=self.val.frame["source"].value_counts().to_dict(),
                               validation_pairs_sha256=hashlib.sha256(self.validation_pair_csv.encode()).hexdigest())
        self.pairs=pd.concat([self.fixed_train.frame,self.val.frame,self.test.frame],ignore_index=True)

    def loaders(self,batch_size,workers,pin_memory,prefetch=None,*,eval_workers=4,prefetch_images=128,
                eval_persistent_workers=False):
        if workers<0 or eval_workers<0:
            raise ValueError("worker counts must be nonnegative")
        paired=self.corpus.settings.paired_batches
        train_batch=max(1,batch_size//2) if paired else batch_size
        pair_batch=max(1,batch_size//2)
        task_images=min(2 if paired else 4,train_batch)*(2 if paired else 1)
        train_prefetch=prefetch_for_images(task_images,prefetch_images) if prefetch is None else prefetch
        train=training_loader(self.train,batch_size=train_batch,num_workers=workers,
                              pin_memory=pin_memory,prefetch_factor=prefetch,prefetch_images=prefetch_images)
        # Group pairs by photo in a worker, then restore their fixed table order.
        kwargs=dict(batch_size=pair_batch,num_workers=eval_workers,pin_memory=pin_memory,
                    prefetch_images=prefetch_images)
        val=OrderedPairLoader(self.val,persistent_workers=eval_persistent_workers,**kwargs)
        test=OrderedPairLoader(self.test,**kwargs)
        diagnostic=fixed_subset(self.fixed_train,min(1000,len(self.fixed_train)),0)
        train_eval=OrderedPairLoader(diagnostic,**kwargs)
        self.record.update(workers=workers,eval_workers=eval_workers,main_threads=torch.get_num_threads(),
                           prefetch_images=prefetch_images,prefetch_override=prefetch,
                           prefetch_factor=train_prefetch,train_prefetch_images=task_images*train_prefetch,
                           eval_prefetch_factor=val.prefetch_factor,
                           eval_prefetch_by_split={"val":val.prefetch_factor,"test":test.prefetch_factor,
                                                   "train_eval":train_eval.prefetch_factor},
                           eval_task="one_photo",eval_persistent_workers=val.persistent_workers,
                           in_order=True,group_images=4,batch_images=batch_size,
                           tensor_dtype=self.corpus.settings.tensor_dtype)
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
        spec=TileSpec(self.corpus.settings.grid if self.corpus.train_grid=='recorded' else self.corpus.training_grid(record),self.corpus.train_tile,0,0)
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
