"""CHECK 2b: per-tile first-tanh saturation; predictions are joined, never inferred.

Quarter membership: stable value/id sort, equal-count quarters (earlier quarters
get remainder rows). Membership stays fixed during photo-cluster bootstraps.
Primary noise cohort: distinct clean hard tiles of the three registered targets.
Miss cohort: stegos in target noisy/low_noisy cells, both k values and methods.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT/'runs/verification/diagnostics-2026-10-19/check2b'
TARGETS = ('alaska-seen','alaska-unseen','lsdir')
SEED = 20261019
MODELS = {
    'srm-e50': ('srm-e50-20261017T111310Z-0214', '35afd1c7938bae78968c', 'srm'),
    'overlap-256': ('tiling-overlap-256-20261017T022212Z-9778', 'c144162a404460293797', 'kv'),
}


def snapshot():
    from scripts.archive.check1_classic import snapshot as first_snapshot
    result = first_snapshot()
    other = ROOT/'runs'/MODELS['overlap-256'][0]
    for p in sorted(other.rglob('*')):
        if not p.is_file(): continue
        h = hashlib.sha256()
        with p.open('rb') as stream:
            for block in iter(lambda: stream.read(1024*1024), b''): h.update(block)
        result[str(p.relative_to(ROOT))] = h.hexdigest()
    return result


def join_saved(measured, saved, frozen):
    """Exact one-to-one pair/label join; saved cell/noise/probability is authoritative."""
    keys = ['pair_id','label']
    if measured.duplicated(keys).any() or saved.duplicated(keys).any():
        raise ValueError('ambiguous saved/measurement keys')
    if frozen.pair_id.duplicated().any(): raise ValueError('ambiguous frozen pair identity')
    reference = frozen.set_index('pair_id')
    expected = reference.loc[saved.pair_id].reset_index(drop=True)
    for col in ('cover_id','source_photo','cell','tile_kv_std','k','condition','placement','method','y','x'):
        if not saved[col].reset_index(drop=True).equals(expected[col]):
            raise ValueError('saved prediction metadata differs from frozen list: '+col)
    if set(saved.label) != {0,1} or not saved.groupby('pair_id').label.agg(list).map(lambda v: sorted(v)==[0,1]).all():
        raise ValueError('saved predictions need both labels for every pair')
    if set(saved.pair_id) != set(frozen.pair_id): raise ValueError('incomplete saved pair list')
    result = saved.merge(measured, on=keys, how='outer', validate='one_to_one', indicator=True)
    if not result['_merge'].eq('both').all(): raise ValueError('missing measured or saved pair/label')
    order={pair:i for i,pair in enumerate(frozen.pair_id)}
    result['_frozen_index']=result.pair_id.map(order)
    return result.sort_values(['_frozen_index','label']).drop(columns=['_merge','_frozen_index']).reset_index(drop=True)


def infer(workers):
    import numpy as np
    import pandas as pd
    import torch
    from torch.utils.data import DataLoader
    from cvdl.hardtest256 import load_frozen
    from cvdl.standardtest import StandardTestPairs
    from cvdl.models import model_from_hparams
    from cvdl.lit import select_best_checkpoint
    from cvdl.archive.tile_saturation import first_tanh_inputs, tile_saturated_share
    from lsb_torch import Corpus, RunSettings, PhotoPairTasks, initialize_worker, prefetch_for_images
    torch.set_num_threads(1); torch.manual_seed(0); np.random.seed(0)
    torch.backends.cudnn.benchmark = False
    if not torch.cuda.is_available(): raise RuntimeError('granted GPU unavailable')
    frozen, _, _ = load_frozen(ROOT)
    corpus = Corpus(ROOT, RunSettings(sources=('raise','div2k','alaska','lsdir'), source='decode'))
    dataset = StandardTestPairs(corpus, frozen)
    for alias, (run, eid, front) in MODELS.items():
        start = time.perf_counter(); folder = ROOT/'runs'/run
        evaluation = folder/'evaluation/standard-256-v1'/eid
        meta = json.loads((evaluation/'evaluation_settings.json').read_text())
        if meta['values']['source'] != 'decode': raise ValueError('unexpected evaluator pixel source')
        checkpoint = Path(select_best_checkpoint('',folder/'checkpoints'/run))
        sha = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        if sha != meta['checkpoint_sha256']: raise ValueError('wrong checkpoint')
        state = torch.load(checkpoint,map_location='cpu',weights_only=False)
        network = model_from_hparams(state['hyper_parameters']).eval()
        network.load_state_dict({k.removeprefix('net.'):v for k,v in state['state_dict'].items()},strict=True)
        if network.front_end != front: raise ValueError('wrong front end')
        network.to('cuda')
        tasks = PhotoPairTasks(dataset)
        kwargs = dict(batch_size=None,num_workers=workers,pin_memory=True,worker_init_fn=initialize_worker)
        if workers: kwargs.update(multiprocessing_context='spawn',prefetch_factor=prefetch_for_images(tasks.max_images,128))
        rows = []; pixel_checks = []; total = 0
        torch.cuda.synchronize()
        with torch.inference_mode():
            for indices,clean,stego in DataLoader(tasks,**kwargs):
                if len(pixel_checks) < 8:
                    index = int(indices[0]); c,s,_ = dataset[index]
                    assert torch.equal(clean[0],c) and torch.equal(stego[0],s)
                    pixel_checks.append(dict(pair_id=str(frozen.iloc[index].pair_id), layout='CHW uint8',
                        clean_sha256=hashlib.sha256(c.numpy().tobytes()).hexdigest(),
                        stego_sha256=hashlib.sha256(s.numpy().tobytes()).hexdigest()))
                for lo in range(0,len(indices),16):
                    part = slice(lo,lo+16); count = len(clean[part])
                    images = torch.cat([clean[part],stego[part]],0).to('cuda',dtype=torch.float32)
                    inputs = first_tanh_inputs(network,images)
                    values = inputs[0].numel(); shares = tile_saturated_share(inputs).cpu().numpy()
                    for position,index in enumerate(indices[part].tolist()):
                        for label,offset in ((0,0),(1,count)):
                            share = float(shares[position+offset])
                            rows.append(dict(index=int(index),pair_id=str(frozen.iloc[int(index)].pair_id),label=label,
                                saturated_share=share,saturated_events=int(round(share*values)),tanh_inputs=values))
                total += len(indices)
                if total % 2000 < len(indices): print(alias,total,'/',len(dataset),flush=True)
        torch.cuda.synchronize()
        measured = pd.DataFrame(rows).sort_values(['index','label']).drop(columns='index').reset_index(drop=True)
        saved_path = evaluation/'hard-test/hard_predictions.csv'
        saved = pd.read_csv(saved_path,float_precision='round_trip')
        joined = join_saved(measured,saved,frozen)
        measured.to_csv(OUT/f'{alias}-measurements.csv',index=False,float_format='%.17g')
        joined.to_csv(OUT/f'{alias}-joined.csv',index=False,float_format='%.17g')
        (OUT/f'{alias}-pixel-parity.json').write_text(json.dumps(pixel_checks,indent=2)+'\n')
        cutoff = float(json.loads((evaluation/'validation/calibration.json').read_text())['cutoff'])
        if alias == 'srm-e50' and abs(cutoff-0.45057889819145203)>1e-15:
            raise ValueError('SRM frozen threshold differs from registered threshold')
        record = dict(model=alias,run=run,evaluation_id=eid,checkpoint=str(checkpoint),checkpoint_sha256=sha,
            front_end=network.front_end,norm=state['hyper_parameters'].get('norm'),clip_t=network.clip_t,
            rocm_conv1_fix=False,eval_batch_size=32,workers=workers,pairs=len(frozen),images=len(joined),
            torch=str(torch.__version__),numpy=np.__version__,seconds=time.perf_counter()-start,
            cutoff=cutoff,pixel_parity_samples=len(pixel_checks),saved_predictions_sha256=hashlib.sha256(saved_path.read_bytes()).hexdigest(),
            no_classifier_calls=True,note='Prefix only; conv1 workaround affects backward only, not the measured forward.')
        (OUT/f'{alias}-inference.json').write_text(json.dumps(record,indent=2)+'\n')
        print(json.dumps(record),flush=True)
        del network,state,inputs,images
        torch.cuda.empty_cache()


def rank_quarters(frame, value, key):
    """Equal-count ranks; deterministic id-only tie break, with boundary audit."""
    import numpy as np
    if len(frame)<4 or frame[key].duplicated().any() or not np.isfinite(frame[value]).all():
        raise ValueError('quarters need at least four finite uniquely identified observations')
    result = frame.sort_values([value,key],kind='stable').reset_index(drop=True).copy()
    result['quarter'] = -1; groups = np.array_split(np.arange(len(result)),4)
    for q,indices in enumerate(groups): result.loc[indices,'quarter'] = q
    boundaries = []
    for q in range(3):
        at = int(groups[q][-1]); left=float(result.iloc[at][value]); right=float(result.iloc[at+1][value])
        tied = left==right
        boundaries.append(dict(after_quarter=q+1,left=left,right=right,tied=tied,
                               tied_group_size=int((result[value]==left).sum()) if tied else 0))
    return result,boundaries


def spearman(x,y):
    import numpy as np
    import pandas as pd
    a = pd.Series(x).rank(method='average').to_numpy(); b = pd.Series(y).rank(method='average').to_numpy()
    a=a-a.mean(); b=b-b.mean(); denominator=np.sqrt(np.dot(a,a)*np.dot(b,b))
    return None if denominator==0 else float(np.clip(np.dot(a,b)/denominator,-1,1))


def correlation_interval(frame,replicates=2000):
    import numpy as np
    x=frame.tile_kv_std.to_numpy(); y=frame.saturated_share.to_numpy()
    groups=[g.index.to_numpy() for _,g in frame.reset_index(drop=True).groupby('source_photo',sort=True)]
    rng=np.random.default_rng(SEED); draws=[]; undefined=0
    for _ in range(replicates):
        indices=np.concatenate([groups[i] for i in rng.integers(0,len(groups),len(groups))])
        r=spearman(x[indices],y[indices])
        if r is None: undefined+=1
        else: draws.append(r)
    low,high=(None,None) if undefined else [float(v) for v in np.quantile(draws,[.025,.975])]
    return dict(rho=spearman(x,y),low=low,high=high,undefined_bootstraps=undefined,
                distinct_clean_tiles=len(frame),photos=len(groups))


def pooled_quarter_difference(frame,replicates=2000):
    """Top-minus-bottom miss; fixed per-cell quarters, joint photo resampling."""
    import numpy as np
    # For each photo: bottom events/n, top events/n. Middle quarters keep their
    # photos in the bootstrap population even when a photo has no extreme row.
    contributions=[]
    for _,g in frame.groupby('source_photo',sort=True):
        bottom=g[g.quarter==0]; top=g[g.quarter==3]
        contributions.append([bottom.missed.sum(),len(bottom),top.missed.sum(),len(top)])
    a=np.asarray(contributions,dtype=float); rng=np.random.default_rng(SEED); draws=[]
    for _ in range(replicates):
        s=a[rng.integers(0,len(a),len(a))].sum(0)
        if not s[1] or not s[3]: raise ValueError('bootstrap has an empty extreme quarter')
        draws.append(s[2]/s[3]-s[0]/s[1])
    totals=a.sum(0); low,high=np.quantile(draws,[.025,.975])
    return dict(bottom_miss=float(totals[0]/totals[1]),top_miss=float(totals[2]/totals[3]),
        difference=float(totals[2]/totals[3]-totals[0]/totals[1]),low=float(low),high=float(high),
        bottom_n=int(totals[1]),top_n=int(totals[3]),photos=len(a),
        borderline_10_points=bool(low<=.10<=high))


def summarize():
    import numpy as np
    import pandas as pd
    from cvdl.hardtest import wilson
    results={}; noise_rows=[]; miss_rows=[]; boundaries=[]; duplicate_audits=[]
    for alias in MODELS:
        data=pd.read_csv(OUT/f'{alias}-joined.csv',float_precision='round_trip')
        meta=json.loads((OUT/f'{alias}-inference.json').read_text()); cutoff=meta['cutoff']
        clean=data[(data.label==0)&data.panel.isin(TARGETS)].copy()
        if (clean.groupby('cover_id').tile_kv_std.nunique()>1).any(): raise ValueError('noise is inconsistent for a clean tile')
        spread=clean.groupby('cover_id').saturated_share.agg(['min','max'])
        duplicate_audits.append(dict(model=alias,max_clean_share_spread=float((spread['max']-spread['min']).max()),
            covers_with_gpu_share_variation=int((spread['max']!=spread['min']).sum())))
        clean=clean.drop_duplicates('cover_id').reset_index(drop=True)
        correlation=correlation_interval(clean)
        quartered,boundary=rank_quarters(clean,'tile_kv_std','cover_id')
        boundaries.extend(dict(model=alias,cohort='distinct-clean-noise',**b) for b in boundary)
        means=[]
        for q,g in quartered.groupby('quarter'):
            mean=float(g.saturated_share.mean());means.append(mean)
            noise_rows.append(dict(model=alias,quarter=int(q)+1,tiles=len(g),photos=int(g.source_photo.nunique()),
                noise_min=float(g.tile_kv_std.min()),noise_max=float(g.tile_kv_std.max()),
                mean_noise=float(g.tile_kv_std.mean()),mean_saturated_share=mean))
        stego=data[(data.label==1)&data.panel.isin(TARGETS)&data.condition.isin(['noisy','low_noisy'])].copy()
        fixed=[]
        for cell,g in stego.groupby('cell',sort=True):
            quarters,boundary=rank_quarters(g,'saturated_share','pair_id')
            quarters['missed']=quarters.prob_stego<cutoff; fixed.append(quarters)
            boundaries.extend(dict(model=alias,cohort=str(cell),**b) for b in boundary)
            for q,f in quarters.groupby('quarter'):
                events=int(f.missed.sum());lo,hi=wilson(events,len(f));row=f.iloc[0]
                miss_rows.append(dict(model=alias,cell=str(cell),panel=str(row.panel),k=int(row.k),condition=str(row.condition),
                    quarter=int(q)+1,pairs=len(f),miss_events=events,miss=events/len(f),miss_low=lo,miss_high=hi,
                    share_min=float(f.saturated_share.min()),share_max=float(f.saturated_share.max()),
                    mean_saturated_share=float(f.saturated_share.mean())))
        fixed=pd.concat(fixed,ignore_index=True)
        fixed[['pair_id','cell','source_photo','quarter','saturated_share','prob_stego','missed']].to_csv(
            OUT/f'{alias}-fixed-quarters.csv',index=False,float_format='%.17g')
        difference=pooled_quarter_difference(fixed)
        rises=(correlation['rho'] is not None and correlation['low'] is not None and
               correlation['rho']>0 and correlation['low']>0 and means[3]>means[0])
        results[alias]=dict(cutoff=cutoff,correlation=correlation,noise_quarter_means=means,
            rises_with_noise=bool(rises),pooled_miss_difference=difference,
            miss_difference_at_least_10_points=bool(difference['difference']>=.10))
        print(alias,json.dumps(results[alias]),flush=True)
    pd.DataFrame(noise_rows).to_csv(OUT/'noise-quarters.csv',index=False,float_format='%.17g')
    pd.DataFrame(miss_rows).to_csv(OUT/'miss-by-saturation-quarter.csv',index=False,float_format='%.17g')
    (OUT/'quarter-boundary-ties.json').write_text(json.dumps(boundaries,indent=2)+'\n')
    (OUT/'duplicate-clean-audit.json').write_text(json.dumps(duplicate_audits,indent=2)+'\n')
    (OUT/'summary.json').write_text(json.dumps(dict(models=results,bootstrap_seed=SEED,bootstrap_replicates=2000,
        decision_model='srm-e50 only; overlap-256 is contrast',quarter_membership='fixed in bootstraps',
        clean_duplicate_rule='first canonical pair/label row per cover_id'),indent=2)+'\n')
    write_report(results,noise_rows,miss_rows,boundaries)


def write_report(results,noise,miss,boundaries):
    lines=['# CHECK 2b: first-tanh saturation versus noise','',
        'All scores/cut-offs come from saved evaluations. First-tanh input share uses |x| > 3. '
        'Only front-end/clip/conv1/abs/bn1 ran; no classifier scores were inferred. '
        'Photo bootstraps: 2,000 replicates, seed20261019; quarter memberships fixed. '
        'Primary noise cohort: distinct clean hard tiles of ALASKA seen, ALASKA unseen and LSDIR.','']
    for alias,result in results.items():
        c=result['correlation'];d=result['pooled_miss_difference']
        rho = 'undefined' if c['rho'] is None else f'{c["rho"]:.5f}'
        interval = 'undefined' if c['low'] is None else f'[{c["low"]:.5f}, {c["high"]:.5f}]'
        lines += ['## '+alias,'',f'Saved validation cut-off: {result["cutoff"]:.17g}.', '',
            f'Spearman: {rho} {interval} (photo-bootstrap 95%). '
            f'{c["distinct_clean_tiles"]} distinct clean tiles; {c["photos"]} photos.', '',
            '| Noise quarter | Tiles | Mean noise | Mean saturated share |','| ---: | ---: | ---: | ---: |']
        for r in noise:
            if r['model']==alias:lines.append(f'| {r["quarter"]} | {r["tiles"]} | {r["mean_noise"]:.5f} | {r["mean_saturated_share"]:.2%} |')
        lines += ['', '| Cell | Q1 miss | Q2 miss | Q3 miss | Q4 miss |', '| --- | ---: | ---: | ---: | ---: |']
        cells=sorted({r['cell'] for r in miss if r['model']==alias})
        for cell in cells:
            values=sorted((r for r in miss if r['model']==alias and r['cell']==cell),key=lambda r:r['quarter'])
            lines.append('| '+cell+' | '+' | '.join(f'{r["miss"]:.1%} ({r["miss_events"]}/{r["pairs"]})' for r in values)+' |')
        lines += ['', f'Pooled Q4-minus-Q1 miss: {100*d["difference"]:+.2f} points '
            f'[{100*d["low"]:+.2f}, {100*d["high"]:+.2f}] (photo-bootstrap 95%). '
            f'Q1 {d["bottom_miss"]:.2%} ({d["bottom_n"]} pairs), Q4 {d["top_miss"]:.2%} ({d["top_n"]} pairs). '
            f'Borderline at 10 points: {d["borderline_10_points"]}.','']
    tied=[b for b in boundaries if b['tied']]
    lines += ['## Quarter boundary audit','',
        f'{len(tied)}/{len(boundaries)} boundaries split equal measured values. '
        'Stable id-only tie-breaks; no score/label tie-breaks. See quarter-boundary-ties.json.', '',
        'Decision uses SRM-e50 only. No decision-plan or board changes are made by this script.']
    (OUT/'report.md').write_text('\n'.join(lines)+'\n')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase',choices=['infer','summarize'],required=True)
    parser.add_argument('--workers',type=int,default=4)
    args=parser.parse_args()
    if not 0<=args.workers<=12:parser.error('workers must be 0 through 12')
    sys.path.insert(0,str(ROOT))
    os.environ.update(PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
    if args.phase=='summarize':os.environ.update(CUDA_VISIBLE_DEVICES='',HIP_VISIBLE_DEVICES='')
    OUT.mkdir(parents=True,exist_ok=True)
    before=snapshot();(OUT/f'{args.phase}-inputs-before.json').write_text(json.dumps(before,indent=2)+'\n')
    started=time.perf_counter()
    try:
        if args.phase=='infer':infer(args.workers)
        else:summarize()
    finally:
        after=snapshot();(OUT/f'{args.phase}-inputs-after.json').write_text(json.dumps(after,indent=2)+'\n')
        (OUT/f'{args.phase}-timing.json').write_text(json.dumps(dict(seconds=time.perf_counter()-started),indent=2)+'\n')
        if before!=after:raise RuntimeError('protected input SHA-256 values changed')


if __name__=='__main__':main()
