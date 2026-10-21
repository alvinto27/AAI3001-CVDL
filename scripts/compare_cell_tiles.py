#!/usr/bin/env python3
"""Compare two SAVED standard256/hard256 evaluations; CSV only, no inference."""
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import pandas as pd
from cvdl.cell_metrics import weighted_auc,photo_auc_bootstrap,interval,REPLICATES,SEED

TARGETS=('alaska-seen','alaska-unseen','lsdir')


def joined(candidate,control):
    frames=[]
    for path in (candidate,control):
        frame=pd.read_csv(path,float_precision='round_trip')
        if frame[['pair_id','label','source_photo','cover_id']].isna().any().any() or frame.duplicated(['pair_id','label']).any():
            raise ValueError('missing or duplicate saved prediction keys')
        if not frame.label.isin((0,1)).all() or not np.isfinite(frame.prob_stego).all():raise ValueError('invalid saved scores/labels')
        frame=frame.set_index(['pair_id','label']).sort_index();frames.append(frame)
    a,b=frames
    try:pd.testing.assert_frame_equal(a.drop(columns='prob_stego'),b.drop(columns='prob_stego'),check_exact=True)
    except AssertionError as e:raise ValueError('saved prediction keys/metadata differ') from e
    out=a.drop(columns='prob_stego').reset_index()
    out['candidate']=a.prob_stego.to_numpy();out['control']=b.prob_stego.to_numpy()
    pairs=out.groupby('pair_id')
    if not pairs.label.agg(list).map(lambda x:x==[0,1]).all() or not pairs.source_photo.nunique().eq(1).all():
        raise ValueError('each pair needs clean and stego from the same photo')
    return out


def own_cutoff(folder):
    summary=pd.read_csv(Path(folder)/'main_summary.csv',float_precision='round_trip')
    row=summary[(summary.panel=='standard')&(summary.category=='overall')&(summary.value=='all')]
    if len(row)!=1 or not np.isfinite(row.cutoff.iloc[0]):raise ValueError('one saved deployment cutoff required')
    origin=row.threshold_origin.iloc[0]
    if not isinstance(origin,str) or not origin or origin.startswith('quick'):raise ValueError('full deployment evaluation required')
    return float(row.cutoff.iloc[0])


def compare(candidate,control,*,replicates=REPLICATES):
    folders={'candidate':Path(candidate),'control':Path(control)}
    cuts={name:own_cutoff(p) for name,p in folders.items()}
    standard=joined(folders['candidate']/'panels/standard/predictions.csv',folders['control']/'panels/standard/predictions.csv')
    hard=joined(folders['candidate']/'hard-test/hard_predictions.csv',folders['control']/'hard-test/hard_predictions.csv')
    result={'bootstrap':dict(replicates=replicates,seed=SEED,unit='source_photo',interval='paired percentile95'),
        'standard':{},'hard_target':{},'noisy_clean':{}}
    for name in folders:
        clean=standard[standard.label==0][name];stego=standard[standard.label==1][name]
        fa=float((clean>=cuts[name]).mean());miss=float((stego<cuts[name]).mean())
        result['standard'][name]=dict(cutoff=cuts[name],pairs=len(stego),FA=fa,miss=miss,P_E=(fa+miss)/2,AUC=weighted_auc(standard.label,standard[name]))
    delta=result['standard']['candidate']['P_E']-result['standard']['control']['P_E']
    result['standard'].update(candidate_minus_control=delta,guard_plus_025pp=delta<=.0025)
    target=hard[hard.panel.isin(TARGETS)&(hard.k==1)&(hard.condition=='low')&(hard.placement=='contiguous')]
    if set(target.panel)!=set(TARGETS) or target.empty:raise ValueError('hard target missing panels')
    point,draws=photo_auc_bootstrap(target.label.to_numpy(),target[['candidate','control']].to_numpy(),target.source_photo.to_numpy(),replicates=replicates)
    difference=interval(point[0]-point[1],draws[:,0]-draws[:,1])
    result['hard_target']=dict(pairs=int((target.label==1).sum()),photos=int(target.source_photo.nunique()),
        candidate=interval(point[0],draws[:,0]),control=interval(point[1],draws[:,1]),candidate_minus_control=difference,
        benefit_plus_002_and_interval_positive=difference['point']>=.02 and difference['low']>0,
        per_panel={p:{name:weighted_auc(g.label,g[name]) for name in folders} for p,g in target.groupby('panel')})
    noise=hard[(hard.label==0)&hard.panel.isin(TARGETS)&hard.condition.isin(('noisy','low_noisy'))].drop_duplicates('cover_id')
    if noise.empty:raise ValueError('no noisy clean tiles')
    for name in folders:result['noisy_clean'][name]=float((noise[name]>=cuts[name]).mean())
    delta=result['noisy_clean']['candidate']-result['noisy_clean']['control']
    result['noisy_clean'].update(distinct_tiles=len(noise),candidate_minus_control=delta,guard_plus_2pp=delta<=.02,
        scope='ALASKA seen/unseen + LSDIR; noisy/low_noisy, both k; distinct cover_id union')
    result['hard_context_auc']={placement:{name:weighted_auc(g.label,g[name]) for name in folders}
        for placement,g in hard[hard.panel.isin(TARGETS)&(hard.k==1)&(hard.condition=='low')].groupby('placement')}
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('candidate',type=Path,help='saved standard256 evaluation folder')
    parser.add_argument('control',type=Path,help='saved standard256 evaluation folder')
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():raise ValueError('use a fresh output file')
    result=compare(args.candidate,args.control)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps(result,indent=2,allow_nan=False))


if __name__=='__main__':main()
