#!/usr/bin/env python3
"""Score frozen cell diagnostics and same-tile CAM/aux figure (not a tile evaluator)."""
import argparse
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import pandas as pd
from cvdl.cell_diagnostics import MANIFEST, selected_rows, replay_cell_pair, cell_maps, render_cell_figure
from cvdl.cell_metrics import fa_cutoff, localization, REPLICATES, SEED


class CellPairs:
    def __init__(self,root,rows):
        from lsb_torch import Corpus,RunSettings
        self.corpus=Corpus(root,RunSettings(sources=('raise','div2k','alaska','lsdir')))
        self.rows=rows
    def __len__(self):return len(self.rows)
    def __getitem__(self,index):
        clean,stego,counts,valid=replay_cell_pair(self.corpus,self.rows[index])
        return np.stack((clean,stego)),counts,valid,index


def score(network,root,rows,*,device,workers,batch_pairs):
    import torch
    from torch.utils.data import DataLoader
    from lsb_torch import initialize_worker
    dataset=CellPairs(root,rows)
    kwargs=dict(batch_size=batch_pairs,num_workers=workers,pin_memory=device=='cuda',worker_init_fn=initialize_worker)
    if workers:kwargs.update(multiprocessing_context='spawn',prefetch_factor=1)
    loader=DataLoader(dataset,**kwargs)
    primary='class_score' if hasattr(network,'class_score_cells') else 'cam'
    collected={k:[] for k in ('counts','valid_negative',primary)}
    if hasattr(network,'cell_head'):collected['aux']=[]
    for images,counts,valid,indices in loader:
        if indices.tolist()!=list(range(sum(len(v) for v in collected['counts']),sum(len(v) for v in collected['counts'])+len(images))):
            raise ValueError('diagnostic loader changed pair order')
        batch=len(images)
        images=images.reshape(-1,256,256,3).permute(0,3,1,2).to(device=device,dtype=torch.float32)
        cam,aux,_=cell_maps(network,images)
        collected[primary].append(cam.cpu().numpy().reshape(batch,2,8,8))
        if aux is not None:collected['aux'].append(aux.cpu().numpy().reshape(batch,2,8,8))
        collected['counts'].append(counts.numpy());collected['valid_negative'].append(valid.numpy())
    return {k:np.concatenate(v) for k,v in collected.items()},dataset.corpus


def report(val_rows,test_rows,val,test,*,replicates=REPLICATES):
    maps=[k for k in ('cam','class_score','aux') if k in val]
    result={'primary_localization':'mean_per_eligible_stego_tile_auc; contiguous only',
        'secondary_localization':'pooled eligible-stego-cell AUC',
        'bootstrap':dict(unit='source_photo',replicates=replicates,seed=SEED,interval='percentile95; shared draws for maps'),
        'cutoff_rule':'score >= cutoff; just above first excluded validation negative; ties cannot split',
        'maps':{},'color_limits':{},'localization':[]}
    valneg=np.concatenate((np.ones_like(val['valid_negative'],dtype=bool)[:,None],val['valid_negative'][:,None]),axis=1)
    for name in maps:
        cutoff,fa=fa_cutoff(val[name][valneg])
        lo,hi=np.quantile(val[name],[.01,.99])
        if lo==hi:lo,hi=lo-1,hi+1
        result['color_limits'][name]=[float(lo),float(hi)]
        result['maps'][name]={'cutoff':cutoff,'validation_negative_cells':int(valneg.sum()),'validation_FA':fa,'test':[]}
        for panel in ['pooled',*dict.fromkeys(r['diagnostic_panel'] for r in test_rows)]:
            for placement in ('all','contiguous','random'):
                take=np.array([(panel=='pooled' or r['diagnostic_panel']==panel) and (placement=='all' or r['placement']==placement) for r in test_rows])
                neg=test['valid_negative'][take];pos=test['counts'][take]>0;s=test[name][take].astype(np.float64)
                negatives=np.concatenate((s[:,0].ravel(),s[:,1][neg]))
                positives=s[:,1][pos]
                result['maps'][name]['test'].append(dict(panel=panel,placement=placement,negative_cells=len(negatives),positive_cells=len(positives),
                    FA=float((negatives>=cutoff).mean()) if len(negatives) else None,
                    recall=float((positives>=cutoff).mean()) if len(positives) else None))
    for split,rows,arrays in (('val',val_rows,val),('test',test_rows,test)):
        for panel in ['pooled',*dict.fromkeys(r['diagnostic_panel'] for r in rows)]:
            for placement in ('contiguous','random'):
                take=np.array([(panel=='pooled' or r['diagnostic_panel']==panel) and r['placement']==placement for r in rows])
                scores=np.stack([arrays[name][take,1].reshape(-1,64) for name in maps],axis=1)
                item=localization(arrays['counts'][take],arrays['valid_negative'][take],scores,
                    [r['source_photo'] for r,t in zip(rows,take) if t],replicates=replicates)
                item['maps']={name:entry for name,entry in zip(maps,item['maps'])}
                result['localization'].append(dict(split=split,panel=panel,placement=placement,**item))
    from collections import Counter
    result['selection_counts']={split:{field:dict(Counter(r[field] for r in rows)) for field in
        ('diagnostic_panel','method','bit_mode','payload_kind','placement')} for split,rows in (('val',val_rows),('test',test_rows))}
    if 'aux' in maps:
        primary=next(r for r in result['localization'] if r['split']=='test' and r['panel']=='pooled' and r['placement']=='contiguous')['maps']['aux']['within_stego_auc']
        test_fa=next(r for r in result['maps']['aux']['test'] if r['panel']=='pooled' and r['placement']=='all')['FA']
        result['auxiliary_target_mark']=bool(primary and primary['point']>=.75 and primary['low']>.5 and test_fa<=.07)
    return result


def describe_map(summary,network):
    if getattr(network,'cell_map_kind',None)=='attribution':
        from scripts.training_decision import COV_ATTRIBUTION
        summary.update(context_only=True,main_rule_affected=False,map_kind='attribution',
            map_storage_key='class_score',map_label=network.cell_map_label,map_definition=COV_ATTRIBUTION)
    elif hasattr(network,'class_score_cells'):
        summary['map_definition']='class_score: post-ELU/BN two-class margin, native stride16 averaged to32; not a CAM'


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('checkpoint',type=Path)
    parser.add_argument('--manifest',type=Path,default=ROOT/MANIFEST)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--device',choices=('cpu','cuda'),default='cuda')
    parser.add_argument('--workers',type=int,default=2)
    parser.add_argument('--batch-pairs',type=int,default=4)
    parser.add_argument('--replicates',type=int,default=REPLICATES)
    parser.add_argument('--figure',action='store_true')
    parser.add_argument('--control-scores',type=Path,help='optional prior cell_eval output for a control CAM column')
    args=parser.parse_args()
    if not 0<=args.workers<=12 or args.batch_pairs<1 or args.replicates<1:parser.error('invalid workers/batch/replicates')
    from scripts.pc_safety import own_process_group,check_workers
    own_process_group();check_workers(args.workers)
    import threading
    from scripts.pc_safety import available_gb,kill_group
    stop=threading.Event()
    def watch_memory():
        while not stop.wait(.5):
            if available_gb()<12:
                print('PC memory safety stop: less than 12 GiB available',flush=True)
                kill_group()
    watcher=threading.Thread(target=watch_memory,daemon=True);watcher.start()
    import torch
    from cvdl.models import model_from_hparams
    from cvdl.pilot import refuse_run
    refuse_run(args.checkpoint.parent)
    if args.output.exists() and any(args.output.iterdir()):raise ValueError('use an empty output folder')
    args.output.mkdir(parents=True,exist_ok=True)
    start=time.perf_counter();torch.set_num_threads(1)
    checkpoint=torch.load(args.checkpoint,map_location='cpu',weights_only=False)
    network=model_from_hparams(checkpoint.get('hyper_parameters',{})).eval()
    network.load_state_dict({k.removeprefix('net.'):v for k,v in checkpoint['state_dict'].items()},strict=True)
    del checkpoint
    network.to(args.device)
    manifest=json.loads(args.manifest.read_text())
    val_rows=selected_rows(ROOT,'val',manifest=manifest);test_rows=selected_rows(ROOT,'test',manifest=manifest)
    if set(r['source_photo'] for r in val_rows)&set(r['source_photo'] for r in test_rows):raise ValueError('val/test photos overlap')
    val,_=score(network,ROOT,val_rows,device=args.device,workers=args.workers,batch_pairs=args.batch_pairs)
    test,corpus=score(network,ROOT,test_rows,device=args.device,workers=args.workers,batch_pairs=args.batch_pairs)
    summary=report(val_rows,test_rows,val,test,replicates=args.replicates)
    describe_map(summary,network)
    for split,rows,arrays in (('val',val_rows,val),('test',test_rows,test)):
        pd.DataFrame(rows).to_csv(args.output/f'{split}-pairs.csv',index=False)
        np.savez_compressed(args.output/f'{split}-scores.npz',**arrays)
    if args.figure:
        control=None;limits=None
        if args.control_scores:
            frame=pd.read_csv(args.control_scores/'test-pairs.csv')
            with np.load(args.control_scores/'test-scores.npz') as saved:
                control=dict(zip(frame.pair_id,saved['cam'][:,1]))
            limits=json.loads((args.control_scores/'summary.json').read_text())['color_limits']['cam']
        render_cell_figure(network,corpus,ROOT,args.output/'cells.png',score_limits=summary['color_limits'],manifest=manifest,
            control_cam=control,control_limits=limits)
    summary['seconds']=time.perf_counter()-start
    (args.output/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
    stop.set();watcher.join()
    print(json.dumps(dict(seconds=summary['seconds'],validation_pairs=len(val_rows),test_pairs=len(test_rows),maps=list(summary['maps'])),indent=2))


if __name__=='__main__':
    from scripts.pc_safety import guarded
    guarded(main)
