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
from cvdl.metrics import metrics, rates, threshold

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


def probabilities(network,images,device):
    """Score a list of equal-shape HxWx3 uint8 images (arrays or tensors) in one forward pass."""
    batch=torch.stack([torch.as_tensor(p) for p in images]).to(device)
    batch=batch.permute(0,3,1,2).to(torch.float32).contiguous()
    with torch.inference_mode():return network(batch).softmax(1)[:,1].cpu().tolist()


PATTERNS=('strip','corner')
LEVELS=(.1,.25,.5)
DEFAULT_PAIRS=2000


def case_shape(identity,pattern,level):
    (_,_,h,w),_=geometry(identity,pattern,level)
    return h,w


def select_cases(frame,limit=DEFAULT_PAIRS):
    """Choose cases from the frozen test pair list; independent of the model and of row order.

    Cases are (row, pattern, level). Within each (pattern, level, real shape) group keep the
    `limit` cases with the smallest keyed hash of (cover_id, pattern, level); `limit` of
    None or 0 keeps every case. Output order: row, pattern, level (the order of the full check).
    """
    ids=frame['cover_id'].astype(str).tolist()
    groups={}
    for row,identity in enumerate(ids):
        for pattern in PATTERNS:
            for level in LEVELS:
                groups.setdefault((pattern,level,case_shape(identity,pattern,level)),[]).append(row)
    keep=set()
    for (pattern,level,_),rows in groups.items():
        if limit and len(rows)>limit:
            rank=lambda r:hashlib.sha256(f'{VERSION}/cap/1/{ids[r]}/{pattern}/{level}'.encode()).hexdigest()
            rows=sorted(rows,key=rank)[:limit]
        keep.update((row,pattern,level) for row in rows)
    order={(p,l):n for n,(p,l) in enumerate((p,l) for p in PATTERNS for l in LEVELS)}
    return sorted(keep,key=lambda c:(c[0],order[(c[1],c[2])]))


def pair_table(frame,cases):
    """CSV text of the chosen cases and its SHA-256 (saved beside the results)."""
    lines=['cover_id,pattern,padding,real_height,real_width']
    for row,pattern,level in cases:
        identity=str(frame.iloc[row]['cover_id']);h,w=case_shape(identity,pattern,level)
        lines.append(f'{identity},{pattern},{level},{h},{w}')
    text='\n'.join(lines)+'\n'
    return text,hashlib.sha256(text.encode()).hexdigest()


class CaseTasks(torch.utils.data.Dataset):
    """One task per photo: decode once, then crop/embed/pad every chosen case of its tiles."""
    def __init__(self,dataset,cases,case_fn=None):
        self.case_fn=case_fn or diagnostic_pair
        by_row={}
        for row,pattern,level in cases:by_row.setdefault(row,[]).append((pattern,level))
        base=dataset.dataset if hasattr(dataset,'dataset') else dataset
        groups={}
        for row in sorted(by_row):
            original=dataset.indices[row] if hasattr(dataset,'indices') else row
            groups.setdefault(base.entries[original][0]['photo_id'],[]).append(row)
        self.dataset=dataset;self.by_row=by_row;self.tasks=list(groups.values())
    def __len__(self):return len(self.tasks)
    def __getitem__(self,index):
        out=[]
        for row in self.tasks[index]:
            for pattern,level in self.by_row[row]:
                real,padded,record=self.case_fn(self.dataset,row,pattern,level)
                out.append((row,pattern,level,real,padded,record))
        return out


def _score_buckets(network,device,results,batch_cases):
    """Score cases in equal-shape buckets; returns {case: {'padded': [clean, stego], 'unpadded': [...]}}."""
    scores={};buckets={}
    def score(part):
        probs=probabilities(network,[im for _,_,pair in part for im in pair],device)
        for n,(case,kind,_) in enumerate(part):
            scores.setdefault(case,{})[kind]=probs[2*n:2*n+2]
    for case,real,padded in results:
        for kind,pair in (('unpadded',real),('padded',padded)):
            bucket=buckets.setdefault((kind,)+tuple(pair[0].shape),[])
            bucket.append((case,kind,pair))
            if len(bucket)>=batch_cases:
                score(bucket[:batch_cases]);del bucket[:batch_cases]
    for bucket in buckets.values():
        for start in range(0,len(bucket),batch_cases):score(bucket[start:start+batch_cases])
    return scores


def score_summary(frame, cutoff):
    return {**metrics(frame), **rates(frame,cutoff)}


def check(network,dataset,baseline,validation,folder,*,device,quick,limit=DEFAULT_PAIRS,workers=0,batch_cases=32,case_fn=None):
    network.eval();folder=Path(folder)/'padding-check';folder.mkdir(exist_ok=True)
    cutoff,_=threshold(validation,'pe')
    cases=select_cases(dataset.frame,limit)
    table,table_sha=pair_table(dataset.frame,cases)
    (folder/'padding-pairs.csv').write_text(table)
    tasks=CaseTasks(dataset,cases,case_fn)
    kwargs=dict(batch_size=None,num_workers=workers)
    if workers:
        from lsb_torch import initialize_worker
        kwargs.update(multiprocessing_context='spawn',prefetch_factor=2,worker_init_fn=initialize_worker)
    loader=torch.utils.data.DataLoader(tasks,**kwargs)
    records={}
    def stream():
        for task in loader:
            for row,pattern,level,real,padded,record in task:
                records[(row,pattern,level)]=record
                yield (row,pattern,level),real,padded
    scores=_score_buckets(network,device,stream(),batch_cases)
    rows=[];manifest=hashlib.sha256()
    # The passed test dataset is already QUICK-capped when requested.
    with (folder/'pairs.jsonl').open('w') as out:
        previous=None;common=None
        for case in cases:
            row,pattern,level=case
            if row!=previous:common=dataset.frame.iloc[row].to_dict();previous=row
            record=records[case]
            line=json.dumps(record,sort_keys=True,default=str)+'\n';out.write(line);manifest.update(line.encode())
            padded_scores=scores[case]['padded'];unpadded_scores=scores[case]['unpadded']
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
    result=dict(version=VERSION,seed=1,quick=quick,pairs=len(dataset),cases=len(cases),pairs_per_shape_and_level=limit or None,
        padding_pairs_sha256=table_sha,validation_threshold=cutoff,
        pairs_sha256=manifest.hexdigest(),pooling_area_changes=True,
        note='padded versus unpadded uses identical content/payload; versus p=0 changes content',summary=summaries)
    (folder/'summary.json').write_text(json.dumps(result,indent=2,default=str)+'\n')
    return result
