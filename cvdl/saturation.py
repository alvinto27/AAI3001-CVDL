"""Measure how saturated the tanh layers of a trained XuNet are. No model code is changed.

The tanh calls in cvdl/models.py are functional (torch.tanh), so a forward hook on the module that
produces each tanh input (bn1 and bn2; Identity when norm=none) reads the input x, and tanh(x) is
recomputed for the output. The network is run in eval mode (the models have no norm layers or dropout
here, so train and eval mode are the same). Layers: tanh1 = tanh(bn1(|conv1(x)|)), tanh2 = tanh(bn2(conv2(x))).
"""
from __future__ import annotations
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

HIST_MAX=50.0
HIST_BINS=50000
LAYERS=(('tanh1','bn1'),('tanh2','bn2'))


class Stats:
    """Streaming |x| statistics: counts above thresholds, mean, histogram quantiles, max."""
    def __init__(self):
        self.n=0;self.sum=0.0;self.gt2=0;self.gt3=0;self.out99=0;self.max=0.0
        self.hist=None;self.over=0
    def add(self,x,out=None):
        a=x.detach().abs().double().flatten()  # stays on the device: statistics are reductions
        self.n+=a.numel();self.sum+=float(a.sum());self.gt2+=int((a>2).sum());self.gt3+=int((a>3).sum())
        self.max=max(self.max,float(a.max()));self.over+=int((a>=HIST_MAX).sum())
        h=torch.histc(a[a<HIST_MAX].float(),bins=HIST_BINS,min=0,max=HIST_MAX).double().cpu()
        self.hist=h if self.hist is None else self.hist+h
        if out is not None:self.out99+=int((out.detach().abs()>0.99).sum())
    def quantile(self,q):
        target=q*self.n;cumulative=torch.cumsum(self.hist,0)
        index=int(torch.searchsorted(cumulative,torch.tensor([target],dtype=torch.float64)))
        return float('inf') if index>=HIST_BINS else (index+1)*HIST_MAX/HIST_BINS
    def result(self,tanh_output=True):
        r=dict(values=self.n,mean_abs=self.sum/self.n,p50_abs=self.quantile(.5),p95_abs=self.quantile(.95),
               p99_abs=self.quantile(.99),max_abs=self.max,share_abs_gt2=self.gt2/self.n,share_abs_gt3=self.gt3/self.n)
        if tanh_output:r['share_tanh_out_gt_0.99']=self.out99/self.n
        return r


def measure(network,batches,device='cpu'):
    """batches yields (uint8 or float NCHW images, int labels). Returns {layer: {label: stats}} plus logit stats."""
    network.eval().to(device)
    stats={name:{0:Stats(),1:Stats()} for name,_ in LAYERS};logit={0:Stats(),1:Stats()}
    current={}
    hooks=[getattr(network,module).register_forward_hook(lambda m,i,o,key=name:current.__setitem__(key,o))
           for name,module in LAYERS]
    try:
        with torch.inference_mode():
            for images,labels in batches:
                labels=torch.as_tensor(labels).cpu()
                output=network(images.to(device,dtype=torch.float32))
                diff=(output[:,1]-output[:,0])
                for label in (0,1):
                    mask=(labels==label).cpu()
                    if not mask.any():continue
                    for name,_ in LAYERS:
                        x=current[name][mask.to(current[name].device)];stats[name][label].add(x,torch.tanh(x))
                    logit[label].add(diff[mask.to(diff.device)])
    finally:
        for hook in hooks:hook.remove()
    table={name:{str(label):s.result() for label,s in per.items() if s.n} for name,per in stats.items()}
    table['final_logit']={str(label):s.result(tanh_output=False) for label,s in logit.items() if s.n}
    return table


def fixed_pairs(count,seed,total):
    return sorted(int(i) for i in np.random.default_rng(seed).choice(total,size=min(count,total),replace=False))


def pair_batches(loader):
    for clean,stego,_ in loader:
        yield clean,torch.zeros(len(clean),dtype=torch.long)
        yield stego,torch.ones(len(stego),dtype=torch.long)


OUTPUT_FILES=('tanh_saturation.json','tanh_saturation.csv','tanh_pairs.txt')
CSV_FIELDS=('measurement','layer','label','values','mean_abs','p50_abs','p95_abs','p99_abs','max_abs',
            'share_abs_gt2','share_abs_gt3','share_tanh_out_gt_0.99')


def write_csv(path,result):
    with Path(path).open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=CSV_FIELDS);writer.writeheader()
        for measurement,table in (('validation',result['validation']),
                                  *((('training_batches',result['training_batches']['stats']),) if 'training_batches' in result else ())):
            for layer,labels in table.items():
                for label,row in labels.items():
                    writer.writerow(dict(measurement=measurement,layer=layer,label=label,**row))


def run(settings,otf,network,output,*,pairs=2000,seed=0,train_batches=0,validation_sha256=None):
    """Fixed validation tile list (unaugmented, eval mode); optionally training-mode batches with augmentation."""
    from lsb_torch import PairSubset, OrderedPairLoader
    from . import data
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    identity=hashlib.sha256(otf.validation_pair_csv.encode()).hexdigest()
    if validation_sha256 and identity!=validation_sha256:
        raise ValueError(f'capped validation identity differs: {identity}')
    indices=fixed_pairs(pairs,seed,len(otf.val))
    val=otf.val
    subset=(PairSubset(val.dataset,[val.indices[i] for i in indices]) if isinstance(val,PairSubset)
            else PairSubset(val,indices))
    ids='\n'.join(subset.frame['cover_id'].astype(str))+'\n'
    (output/'tanh_pairs.txt').write_text(ids)
    result=dict(version='tanh-saturation-v1',seed=seed,pairs=len(indices),
                pair_ids_sha256=hashlib.sha256(ids.encode()).hexdigest(),validation_pairs_sha256=identity,
                layers='tanh1 = tanh(bn1(|conv1(x)|)), tanh2 = tanh(bn2(conv2(x))); final_logit = logit1 - logit0',
                note='the tile list is the same for every model; labels 0 = clean, 1 = stego')
    loader=OrderedPairLoader(subset,32,settings.eval_workers,settings.device=='cuda')
    result['validation']=measure(network,pair_batches(loader),settings.device)
    if train_batches:
        train=data.build_otf_loaders(otf,batch_size=settings.batch_size,workers=settings.workers,
            eval_workers=0,pin_memory=settings.device=='cuda',prefetch=settings.prefetch,
            prefetch_images=settings.prefetch_images,prime_next_epoch=False).train
        train.dataset.set_epoch(0)
        def batches():
            for count,(images,labels,_) in enumerate(train):
                if count>=train_batches:break
                yield images,labels
        try:
            result['training_batches']=dict(batches=train_batches,epoch=0,batch_size=settings.batch_size,
                augmentation=True,stats=measure(network,batches(),settings.device))
        finally:
            train.shutdown()
    (output/'tanh_saturation.json').write_text(json.dumps(result,indent=2)+'\n')
    write_csv(output/'tanh_saturation.csv',result)
    return result
