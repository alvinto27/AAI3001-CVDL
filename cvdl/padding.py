"""Separate, versioned strip/corner diagnostic with exact unpadded controls."""
import hashlib
import json
import math
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from lsb_engine import embed_fast
from lsb_settings import draw_otf_settings, otf_seed_sequence
from lsb_train_grid import pad_crop
from .metrics import metrics, rates, threshold

VERSION='padding-check-v1'


def rng_for(identity,domain):
    digest=hashlib.sha256(f'{VERSION}/1/{identity}/{domain}'.encode()).digest()
    return np.random.default_rng(np.random.SeedSequence(np.frombuffer(digest,dtype='<u4')))


def geometry(identity,pattern,level):
    if pattern not in {'strip','corner'} or level not in (.1,.25,.5):
        raise ValueError('padding check needs strip/corner at .1/.25/.5')
    orientation=int(rng_for(identity,pattern+'/orientation').integers(4))
    if pattern=='strip':
        h,w=512,math.floor(512*(1-level)+.5)
        if orientation in (2,3):h,w=w,h
        top=(512-h if orientation==2 else 0)
        left=(512-w if orientation==0 else 0)
    else:
        h=w=math.floor(512*math.sqrt(1-level)+.5)
        top=(512-h if orientation//2 else 0);left=(512-w if orientation%2 else 0)
    return (top,left,h,w),(top,512-top-h,left,512-left-w)


def diagnostic_pair(dataset,index,pattern,level):
    base=dataset;original=index
    while hasattr(base,'indices'):
        original=base.indices[original];base=base.dataset
    record,row,col,variant,legacy=base.entries[original]
    if legacy is not None:raise ValueError('padding check needs frozen OTF test pairs')
    identity=str(dataset.frame.iloc[index]['cover_id'])
    full=dataset[index][0].permute(1,2,0).numpy()
    (y,x,h,w),pad=geometry(identity,pattern,level)
    clean=np.ascontiguousarray(full[y:y+h,x:x+w])
    sequence=otf_seed_sequence(run_seed=base.corpus.settings.run_seed,purpose=base.purpose,epoch=0,
        photo_index=int(record['photo_index']),grid_policy=base.corpus.effective_grid(record),tile_size=512,
        tile_row=row,tile_col=col,variant=variant,stream='settings',fixed_variants=base.corpus.settings.fixed_variants)
    kwargs,drawn=draw_otf_settings(base.corpus.evaluation_config,clean,np.random.default_rng(sequence))
    kwargs['rng']=rng_for(identity,f'{pattern}/{level}/embedding')
    stego,embedded,_=embed_fast(clean,**kwargs)
    provenance=dict(cover_id=identity,pattern=pattern,requested_fraction=level,pad=list(pad),
        real_shape=[h,w],actual_fraction=1-h*w/512**2,settings={k:v for k,v in kwargs.items() if k!='rng'},
        requested_bpp=drawn['requested_bpp'],effective_bpp=drawn['effective_bpp'],
        payload_bytes=drawn['payload_bytes'],embedding=embedded)
    return (clean,stego),(pad_crop(clean,pad),pad_crop(stego,pad)),provenance


def probabilities(network,pair,device):
    batch=torch.stack([torch.from_numpy(np.ascontiguousarray(p)).permute(2,0,1) for p in pair]).to(device,dtype=torch.float32)
    with torch.inference_mode():return network(batch).softmax(1)[:,1].cpu().tolist()


def score_summary(frame, cutoff):
    return {**metrics(frame), **rates(frame,cutoff)}


def check(network,dataset,baseline,validation,folder,*,device,quick):
    network.eval();folder=Path(folder)/'padding-check';folder.mkdir(exist_ok=True)
    cutoff,_=threshold(validation,'pe')
    rows=[];manifest=hashlib.sha256()
    # The passed test dataset is already QUICK-capped when requested.
    with (folder/'pairs.jsonl').open('w') as out:
        for i in range(len(dataset)):
            common=dataset.frame.iloc[i].to_dict()
            for pattern in ('strip','corner'):
                for level in (.1,.25,.5):
                    real,padded,record=diagnostic_pair(dataset,i,pattern,level)
                    line=json.dumps(record,sort_keys=True,default=str)+'\n';out.write(line);manifest.update(line.encode())
                    unpadded_scores=probabilities(network,real,device)
                    padded_scores=probabilities(network,padded,device)
                    for label in (0,1):
                        rows.append(dict(**common,label=label,pattern=pattern,padding=level,
                            actual_fraction=record['actual_fraction'],prob_stego=padded_scores[label],
                            unpadded_prob_stego=unpadded_scores[label],delta=padded_scores[label]-unpadded_scores[label]))
    frame=pd.DataFrame(rows);frame.to_csv(folder/'predictions.csv',index=False)
    summaries=[dict(pattern='reference',padding=0,control='frozen',**score_summary(baseline,cutoff))]
    for (pattern,level),group in frame.groupby(['pattern','padding']):
        for control in ('padded','unpadded'):
            selected=group.copy()
            if control=='unpadded':selected['prob_stego']=selected['unpadded_prob_stego']
            summaries.append(dict(pattern=pattern,padding=level,control=control,**score_summary(selected,cutoff)))
        for setting in ('source','k'):
            for value,part in group.groupby(setting):
                for control in ('padded','unpadded'):
                    selected=part.copy()
                    if control=='unpadded':selected['prob_stego']=selected['unpadded_prob_stego']
                    summaries.append(dict(pattern=pattern,padding=level,control=control,setting=setting,value=str(value),
                                          **score_summary(selected,cutoff)))
    result=dict(version=VERSION,seed=1,quick=quick,pairs=len(dataset),validation_threshold=cutoff,
        pairs_sha256=manifest.hexdigest(),pooling_area_changes=True,
        note='padded versus unpadded uses identical content/payload; versus p=0 changes content',summary=summaries)
    (folder/'summary.json').write_text(json.dumps(result,indent=2,default=str)+'\n')
    return result
