"""Frozen paired/curriculum decisions from saved scores only; CPU, no inference.

Reference/candidate run directories must contain exactly one complete evaluation,
otherwise pass its exact evaluation directory. Cutoffs are read, never test-fit.
"""
from pathlib import Path
import argparse
import hashlib
import json
import os
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.architecture_decision import (join_predictions, auc_intervals, rates,
    PANELS, TARGETS, SEED, REPLICATES)

NAMES=('reference','candidate')
RULES=('paired_r1','curriculum50','paired_r2','wiser_b')
WISER_REFERENCE='runs/srm-e50-paired-20261023T022339Z-4518/evaluation/standard-256-v1/6ef349895655136bef90'
WISER_REFERENCE_SHA='b00cd4b6d1f4c3648580691c9cf0e2a4bd83c55740a3cf2a7a561a8adf12f180'
WISER_NOTEBOOK=ROOT/'notebooks/experiments/srm-e50-paired-coloursep36.ipynb'
WISER_NOTEBOOK_SHA='c49e8b80233a5cfd51cf47495a78d9b16e3c9b8956c0ec8c127ffbdc5d03968b'
WISER_APPROVAL=dict(arm='ideas20-B36',date='2026-10-24',routes='paired_r1: a OR b',
    original_plan_sha256='6d1ff40e574bc55f504265cb136959082dbe4733a6e8f7c81527a3c5df76b48f',
    approved_override='both paired_r1 routes; no cell or minimum selected-epoch gate',
    model='xunet_wide_coloursep36',selection='full scratch50 capped best',
    reference=WISER_REFERENCE,reference_sha256=WISER_REFERENCE_SHA,notebook_sha256=WISER_NOTEBOOK_SHA)
LIMITS=dict(standard_pe_route_a_point_max=-.002,standard_pe_route_a_upper_strict_max=0.,
    hard_auc_guard_lower_min=-.01,hard_auc_gain_point_min=.01,hard_auc_gain_lower_strict_min=0.,
    standard_pe_guard_upper_max=.0025,curriculum_selected_epoch_one_based_min=8,
    cell_auc_gain_point_min=.02,cell_auc_gain_lower_strict_min=0.,candidate_cell_fa_max=.07)
PLAN=ROOT/'docs/active/plan-paired-training.md'
CURRICULUM_PLAN=ROOT/'docs/active/plan-curriculum50.md'


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def evaluation_folder(path):
    path=Path(path).resolve()
    if (path/'summary.json').is_file() and (path/'evaluation_settings.json').is_file():return path
    found=sorted(path.glob('evaluation/standard-256-v1/*/evaluation_settings.json'))
    if len(found)!=1:raise ValueError('run needs exactly one evaluation; pass an explicit evaluation directory')
    return found[0].parent


def evaluation_inputs(folder):
    return [folder/'summary.json',folder/'evaluation_settings.json',folder/'validation/calibration.json',
        folder/'validation/predictions.csv',folder/'hard-test/hard_predictions.csv',
        *[folder/f'panels/{panel}/predictions.csv' for panel in PANELS]]


def validate_evaluation(folder):
    import numpy as np
    summary=json.loads((folder/'summary.json').read_text())
    meta=json.loads((folder/'evaluation_settings.json').read_text())
    calibration=json.loads((folder/'validation/calibration.json').read_text())
    threshold=summary['validation_threshold']
    if (len(folder.name)!=20 or any(c not in '0123456789abcdef' for c in folder.name) or
            summary['evaluation_id']!=folder.name or meta['evaluation_id']!=folder.name or
            summary['suite']!='standard-256-v1' or meta['suite']!='standard-256-v1' or
            summary.get('quick') is not False or meta.get('quick') is not False or
            summary.get('hard_test',{}).get('status')!='complete' or
            threshold.get('status')!='complete' or threshold.get('deployment') is not True or
            threshold.get('quick') is not False or not np.isfinite(threshold['cutoff']) or
            not 0<=threshold['cutoff']<=1):
        raise ValueError('incomplete, quick, or non-deployment evaluation')
    if any(threshold.get(k)!=v for k,v in calibration.items()):
        raise ValueError('saved calibration and summary threshold disagree')
    if sha(folder/'validation/predictions.csv')!=calibration['score_csv_sha256']:
        raise ValueError('validation score hash mismatch')
    fields=('checkpoint_sha256','recipe_id','recipe_sha256','selected_pairs_sha256',
            'full_pairs_sha256','mapping_sha256','geometry','policy','deployment','quick')
    if any(meta['calibration_inputs'][k]!=threshold[k] for k in fields):
        raise ValueError('calibration input metadata mismatch')
    if (meta['checkpoint_sha256']!=threshold['checkpoint_sha256'] or
            threshold['selected_pairs_sha256']!=threshold['full_pairs_sha256']):
        raise ValueError('checkpoint mismatch or partial deployment calibration')
    hard_threshold=summary['hard_test']['validation_threshold']
    if any(hard_threshold[k]!=threshold[k] for k in (*fields,'cutoff','threshold_id')):
        raise ValueError('hard evaluation did not use the same frozen cutoff')
    return summary,meta,threshold


def checkpoint_path(meta):
    path=Path(meta['checkpoint'])
    if path.is_file():return path.resolve()
    if 'runs' in path.parts:
        rebased=ROOT.joinpath(*path.parts[path.parts.index('runs'):])
        if rebased.is_file():return rebased
    raise ValueError('selected checkpoint not found; cannot verify curriculum exposure')


def selected_epoch(meta):
    """Verify scratch50 identity and read its actual evaluated checkpoint epoch."""
    import torch
    from cvdl.curriculum import SCRATCH50,scratch50_record
    values=meta.get('values',{})
    if values.get('embedding_schedule')!=SCRATCH50 or values.get('tiles_per_epoch')!=180000:
        raise ValueError('curriculum50 requires the versioned 180000-draw scratch schedule')
    path=checkpoint_path(meta)
    if not path.name.startswith('best-') or meta.get('values',{}).get('evaluation_checkpoint','best')!='best':
        raise ValueError('curriculum50 requires the selected best checkpoint')
    if sha(path)!=meta['checkpoint_sha256']:raise ValueError('evaluated checkpoint SHA mismatch')
    saved=torch.load(path,map_location='cpu',weights_only=False)
    if saved.get('hyper_parameters',{}).get('embedding_schedule_identity')!=scratch50_record(180000):
        raise ValueError('selected checkpoint curriculum50 schedule identity mismatch')
    epoch=saved['epoch']
    if type(epoch) is not int or not 0<=epoch<50:raise ValueError('invalid checkpoint epoch')
    return epoch+1,path


def wiser_identity(folders, metadata):
    """Verify the fixed reference and scratch/best recipe before reading scores."""
    import torch
    from scripts.g4.preflight import top_values
    if Path(folders[0]).resolve()!=(ROOT/WISER_REFERENCE).resolve() or metadata[0]['checkpoint_sha256']!=WISER_REFERENCE_SHA:
        raise ValueError('wiser_b requires the frozen R1 reference')
    if sha(WISER_NOTEBOOK)!=WISER_NOTEBOOK_SHA:raise ValueError('wiser_b frozen recipe fingerprint changed')
    from cvdl.settings import load,machine_profile,plain,SCHEMA
    expected=plain(load(environ={},_profile=machine_profile('g4'),
        **top_values(json.loads(WISER_NOTEBOOK.read_text()))).values)
    path=checkpoint_path(metadata[1])
    record=path.parent/'run_settings.json'
    if not record.is_file():raise ValueError('missing original training settings')
    values=json.loads(record.read_text())['values']
    for key,value in expected.items():
        if key in ('name','run_name') or (SCHEMA[key].machine and key not in ('precision','channels_last','quick','compile','workers','eval_workers','main_threads',
            'prefetch_images','pack_blocks','prime_next_epoch')):continue
        actual=values.get(key)
        if key=='train_yaml':
            # Different machine roots are not experiment differences.
            if not actual or Path(actual).parts[-2:]!=('configs','train-k1to4.yaml'):
                raise ValueError('wiser_b recipe mismatch: train_yaml')
            continue
        if json.dumps(actual,sort_keys=True)!=json.dumps(value,sort_keys=True):
            raise ValueError('wiser_b recipe mismatch: '+key)
    for key in ('model','front_end','norm','width','clip_t'):
        if metadata[1].get('values',{}).get(key)!=expected[key]:
            raise ValueError('wiser_b evaluated model mismatch: '+key)
    for key in ('init_checkpoint','embedding_schedule','cell_supervision','pilot'):
        if values.get(key) not in (None,False,'off'):
            raise ValueError('wiser_b forbids '+key)
    if not path.name.startswith('best-') or values.get('evaluation_checkpoint','best')!='best' or metadata[1].get('values',{}).get('evaluation_checkpoint','best')!='best':
        raise ValueError('wiser_b requires selected best checkpoint')
    if sha(path)!=metadata[1]['checkpoint_sha256']:raise ValueError('evaluated checkpoint SHA mismatch')
    from cvdl.lit import select_best_checkpoint
    if Path(select_best_checkpoint(None,path.parent)).resolve()!=path.resolve():raise ValueError('not selected best checkpoint')
    last=path.parent/'last.ckpt'
    if not last.is_file():raise ValueError('wiser_b needs completed scratch50 last checkpoint evidence')
    completed=torch.load(last,map_location='cpu',weights_only=False)
    if completed.get('epoch') not in (49,50) or completed.get('global_step')!=140650:
        raise ValueError('wiser_b requires full 50-epoch fit evidence')
    saved=torch.load(path,map_location='cpu',weights_only=False)
    h=saved.get('hyper_parameters',{})
    for key,value in dict(model='xunet_wide_coloursep36',front_end='srm',norm='none',width=4,paired_batches=True,max_epochs=50).items():
        if h.get(key)!=value:raise ValueError('wiser_b checkpoint identity mismatch: '+key)
    epoch=saved.get('epoch')
    if type(epoch) is not int or not 0<=epoch<50:raise ValueError('invalid checkpoint epoch')
    return epoch+1,path


def rule_checks(rule, hard, standard, *, epoch=None, cells=None):
    """Candidate minus reference. Literal boundary directions are intentional."""
    if rule not in RULES:raise ValueError('unknown frozen rule')
    if rule=='wiser_b':
        return rule_checks('paired_r1',hard,standard)
    if rule=='paired_r1':
        a=dict(standard_pe_gain_at_least_002=standard['point']<=-.002,
            standard_pe_upper_below_zero=standard['high']<0,
            hard_auc_lower_at_least_minus_001=hard['low']>=-.01)
        b=dict(hard_auc_gain_at_least_001=hard['point']>=.01,
            hard_auc_lower_above_zero=hard['low']>0,
            standard_pe_upper_at_most_0025=standard['high']<=.0025)
        return dict(passed=all(a.values()) or all(b.values()),
            routes={'a':dict(passed=all(a.values()),checks=a),'b':dict(passed=all(b.values()),checks=b)})
    if rule=='curriculum50':
        if epoch is None:raise ValueError('curriculum50 needs verified selected checkpoint epoch')
        checks=dict(hard_auc_gain_at_least_001=hard['point']>=.01,
            hard_auc_lower_above_zero=hard['low']>0,
            standard_pe_upper_at_most_0025=standard['high']<=.0025,
            selected_epoch_one_based_at_least_8=epoch>=8)
    else:
        if cells is None or not cells['head_vs_head']:raise ValueError('paired_r2 requires aux head versus aux head')
        delta=cells['primary']['difference']
        checks=dict(cell_auc_gain_at_least_002=delta is not None and delta['point']>=.02,
            cell_auc_lower_above_zero=delta is not None and delta['low']>0,
            standard_pe_upper_at_most_0025=standard['high']<=.0025,
            hard_auc_lower_at_least_minus_001=hard['low']>=-.01,
            candidate_cell_fa_at_most_007=cells['operating_points']['candidate']['test_fa']<=.07)
    return dict(passed=all(checks.values()),checks=checks)


def cell_inputs(folder):
    return [Path(folder)/name for name in ('val-pairs.csv','test-pairs.csv','val-scores.npz','test-scores.npz','summary.json')]


def compare_cells(reference, candidate, *, reference_map='aux', candidate_map='aux', replicates=REPLICATES):
    """Reuse cell_eval's saved target/score layout and cell_metrics.localization.

    Non-aux maps are permitted for historical diagnostic reproduction ONLY;
    rule_checks refuses to use that context comparison as an R2 verdict.
    """
    import numpy as np
    import pandas as pd
    from cvdl.cell_metrics import localization, SEED as CELL_SEED
    folders=[Path(reference),Path(candidate)];keys=[reference_map,candidate_map]
    summaries=[json.loads((p/'summary.json').read_text()) for p in folders]
    rows={};arrays={}
    for split in ('val','test'):
        aligned=[];values=[]
        for folder,key in zip(folders,keys):
            frame=pd.read_csv(folder/f'{split}-pairs.csv',float_precision='round_trip')
            if frame[['pair_id','source_photo']].isna().any().any() or frame.pair_id.duplicated().any() or len(frame)==0:
                raise ValueError('invalid cell row keys')
            allowed={'val'} if split=='val' else {'test','test_seen','test_unseen'}
            if not frame.split.isin(allowed).all():raise ValueError('cell split mismatch')
            order=np.argsort(frame.pair_id.to_numpy(),kind='stable')
            aligned.append(frame.iloc[order].reset_index(drop=True))
            with np.load(folder/f'{split}-scores.npz',allow_pickle=False) as saved:
                if key not in saved:raise ValueError(f'missing saved {key} scores in {folder}')
                count=saved['counts'];negative=saved['valid_negative'];scores=saved[key]
                if count.shape!=(len(frame),8,8) or negative.shape!=count.shape or scores.shape!=(len(frame),2,8,8):
                    raise ValueError('cell array shape mismatch')
                if (not np.isfinite(scores).all() or not np.isfinite(count).all() or
                        (count<0).any() or (count!=np.floor(count)).any() or
                        not np.isin(negative,[False,True]).all() or ((count>0)&negative.astype(bool)).any()):
                    raise ValueError('invalid cell targets/scores')
                values.append(dict(counts=count[order],valid_negative=negative[order].astype(bool),scores=scores[order].astype(np.float64)))
        try:pd.testing.assert_frame_equal(*aligned)
        except AssertionError as e:raise ValueError('cell row metadata mismatch') from e
        if any(not np.array_equal(values[0][key],values[1][key]) for key in ('counts','valid_negative')):
            raise ValueError('cell target/eligibility mismatch')
        rows[split]=aligned[0];arrays[split]=values
    if set(rows['val'].source_photo)&set(rows['test'].source_photo):raise ValueError('cell validation/test photos overlap')
    operating={}
    for name,key,summary,val,test in zip(NAMES,keys,summaries,arrays['val'],arrays['test']):
        record=summary['maps'][key];cutoff=record['cutoff']
        validation_negatives=np.concatenate((val['scores'][:,0].ravel(),val['scores'][:,1][val['valid_negative']]))
        test_negatives=np.concatenate((test['scores'][:,0].ravel(),test['scores'][:,1][test['valid_negative']]))
        if not len(test_negatives) or not len(validation_negatives) or not np.isfinite(cutoff):
            raise ValueError('empty cell negatives or invalid frozen cutoff')
        fa=float((validation_negatives>=cutoff).mean())
        if (fa>.05 or fa!=record['validation_FA'] or len(validation_negatives)!=record['validation_negative_cells']):
            raise ValueError('saved cell cutoff is not validation-frozen at 5%')
        test_fa=float((test_negatives>=cutoff).mean())
        reported=next(r for r in record['test'] if r['panel']=='pooled' and r['placement']=='all')
        if test_fa!=reported['FA']:raise ValueError('saved cell FA does not match scores')
        operating[name]=dict(map=key,cutoff=float(cutoff),validation_fa=fa,
            validation_negative_cells=len(validation_negatives),test_fa=test_fa,test_negative_cells=len(test_negatives))
    frame=rows['test'];take=(frame.placement=='contiguous').to_numpy()
    if not take.any():raise ValueError('no contiguous test cells for the primary scope')
    counts=arrays['test'][0]['counts'][take];negative=arrays['test'][0]['valid_negative'][take]
    scores=np.stack([a['scores'][take,1] for a in arrays['test']],axis=1)
    samples={}
    result=localization(counts,negative,scores,frame.source_photo.to_numpy()[take],
        replicates=replicates,difference_name='candidate_minus_reference',draws_out=samples)
    maps={name:r for name,r in zip(NAMES,result.pop('maps'))}
    delta=result.pop('candidate_minus_reference',{}).get('within_stego_auc')
    primary=dict(scope='TEST pooled contiguous; mean per eligible stego-tile AUC',
        **result,models={name:r['within_stego_auc'] for name,r in maps.items()},difference=delta)
    eligible=(counts>0).reshape(-1,64).any(1)&negative.reshape(-1,64).any(1)
    primary['eligible_photos']=int(np.unique(frame.source_photo.to_numpy()[take][eligible]).size)
    return dict(seed=CELL_SEED,replicates=replicates,head_vs_head=keys==['aux','aux'],
        rows={split:dict(pairs=len(f),photos=int(f.source_photo.nunique())) for split,f in rows.items()},
        score_maps=dict(zip(NAMES,keys)),primary=primary,operating_points=operating,
        secondary_pooled_cell_auc={name:r['pooled_cell_auc'] for name,r in maps.items()}),samples


def summarize(reference, candidate, output, rule, *, reference_cells=None, candidate_cells=None):
    import numpy as np
    if rule not in RULES:raise ValueError('unknown frozen rule')
    if rule!='paired_r2' and (reference_cells or candidate_cells):raise ValueError('cell inputs are only used by paired_r2')
    folders=[evaluation_folder(reference),evaluation_folder(candidate)];output=Path(output).resolve()
    if output.exists() and any(output.iterdir()):raise ValueError('use an empty decision output folder')
    sources=[*folders,*[Path(p).resolve() for p in (reference_cells,candidate_cells) if p]]
    if any(output==p or p in output.parents for p in sources):raise ValueError('output must not modify input folders')
    paths=[p for folder in folders for p in evaluation_inputs(folder)]
    paths+=[PLAN,CURRICULUM_PLAN,Path(__file__),ROOT/'scripts/architecture_decision.py',
        ROOT/'cvdl/cell_metrics.py',ROOT/'cvdl/metrics.py',ROOT/'cvdl/curriculum.py']
    if rule=='wiser_b':
        paths.extend([WISER_NOTEBOOK,*[ROOT/p for p in ('cvdl/models.py','cvdl/settings.py','cvdl/runtime.py',
            'cvdl/lit.py','configs/train-k1to4.yaml','configs/test-256-v1.yaml',
            'configs/evaluation-calibration-v1.json')]])
        paths.extend(sorted((ROOT/'data/val/60da4391043e8ce48ff2').glob('*')))
        # Resolve and hash before any score joins or bootstrap work.
        raw=[json.loads((p/'evaluation_settings.json').read_text()) for p in folders]
        wiser_epoch,wiser_path=wiser_identity(folders,raw)
        paths.extend([wiser_path,*[wiser_path.parent/p for p in ('last.ckpt','run_settings.json',
            'model_run_settings.json','data_run_settings.json')]])
    if rule=='paired_r2':
        if not reference_cells or not candidate_cells:raise ValueError('paired_r2 needs both saved cell folders')
        paths.extend(cell_inputs(reference_cells)+cell_inputs(candidate_cells))
    def snapshot():return {str(p.resolve()):sha(p) for p in paths}
    before=snapshot();validated=[validate_evaluation(p) for p in folders]
    thresholds=[v[2] for v in validated];metadata=[v[1] for v in validated]
    shared=('recipe_id','recipe_sha256','selected_pairs_sha256','full_pairs_sha256','geometry','policy')
    if any(thresholds[0][k]!=thresholds[1][k] for k in shared):raise ValueError('deployment validation recipe mismatch')
    if metadata[0]['hard_recipe_sha256']!=metadata[1]['hard_recipe_sha256']:raise ValueError('hard recipe mismatch')
    cutoffs={name:t['cutoff'] for name,t in zip(NAMES,thresholds)}
    def joined(paths):
        frame=join_predictions(paths,names=NAMES)
        if not frame[list(NAMES)].ge(0).all().all() or not frame[list(NAMES)].le(1).all().all():
            raise ValueError('saved probabilities outside [0,1]')
        return frame
    hard=joined([p/'hard-test/hard_predictions.csv' for p in folders])
    if any(int((hard.label==1).sum())!=v[0]['hard_test']['pairs'] for v in validated):
        raise ValueError('hard prediction count disagrees with complete summary')
    low=hard[hard.panel.isin(TARGETS)&(hard.k==1)&(hard.condition=='low')]
    if (set(low.panel)!=set(TARGETS) or set(low.method)!={'replace','match'} or
            set(low.placement)!={'contiguous','random','stride'}):raise ValueError('hard scope coverage mismatch')
    main,main_draws=auc_intervals(low,names=NAMES)
    draws={'pooled_hard_auc':main_draws};panels={}
    for panel in PANELS:
        frame=joined([p/f'panels/{panel}/predictions.csv' for p in folders])
        for summary,_,_ in validated:
            expected=next(t['pairs'] for t in summary['timings'] if t['panel']==panel and not t.get('higher_k',False))
            if int((frame.label==1).sum())!=expected:raise ValueError('panel prediction count disagrees with complete summary')
        item,draws['standard_'+panel]=auc_intervals(frame,names=NAMES)
        item['rates']=rates(frame,cutoffs,names=NAMES)
        if panel=='standard':
            guard,draws['standard_pe']=auc_intervals(frame,names=NAMES,metric='pe',cutoffs=cutoffs)
        panels[panel]=item
    context={}
    for field,values in (('panel',TARGETS),('placement',('contiguous','random','stride'))):
        for value in values:
            key=field+'/'+value
            context[key],draws[key]=auc_intervals(low[low[field]==value],names=NAMES)
    for condition in ('noisy','low_noisy'):
        context[condition],draws[condition]=auc_intervals(hard[hard.panel.isin(TARGETS)&(hard.k==1)&(hard.condition==condition)],names=NAMES)
    epoch=wiser_epoch if rule=='wiser_b' else None;cells=None
    if rule=='curriculum50':
        epoch,path=selected_epoch(metadata[1]);paths.append(path)
        # The checkpoint was hashed/verified against the already snapshotted meta.
        before[str(path.resolve())]=metadata[1]['checkpoint_sha256']
    if rule=='paired_r2':
        cells,cell_draws=compare_cells(reference_cells,candidate_cells)
        draws.update({'cell_'+k:v for k,v in cell_draws.items()})
    verdict=rule_checks(rule,main['difference'],guard['difference'],epoch=epoch,cells=cells)
    after=snapshot()
    if before!=after:raise RuntimeError('saved inputs changed during decision')
    result=dict(rule_set=rule,rule_parameters=LIMITS,**verdict,model_order=NAMES,evaluations=dict(zip(NAMES,map(str,folders))),
        tile_bootstrap=dict(seed=SEED,replicates=REPLICATES,interval='percentile95',
            unit='sorted source_photo clusters; paired/repeated clean rows retained; shared model draws'),
        uncertainty='test-photo uncertainty, not training-seed uncertainty',
        input_sha256=before,input_sha256_after=after,inputs_unchanged=True,
        threshold_metadata=dict(zip(NAMES,thresholds)),pooled_hard_auc=main,standard_pe=guard,
        standard_panels=panels,context_hard_auc=context,selected_epoch_one_based=epoch,cells=cells)
    if rule=='wiser_b':
        result['approved_identity']=WISER_APPROVAL
        result['approved_identity_sha256']=hashlib.sha256(json.dumps(WISER_APPROVAL,sort_keys=True).encode()).hexdigest()
    output.mkdir(parents=True,exist_ok=True)
    (output/'evidence.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    np.savez_compressed(output/'bootstrap-replicates.npz',**draws)
    write_report(result,output)
    return result


def write_report(result,output):
    def number(r):return 'ineligible' if r is None else f'{r["point"]:+.8f} [{r["low"]:+.8f}, {r["high"]:+.8f}]'
    lines=[f'# {result["rule_set"]}: '+('PASS' if result['passed'] else 'FAIL'),'',
        'Saved scores only. Candidate minus reference; paired source-photo draws, 2,000 replicates. '
        'Tile seed20261019; cell seed20261021. Percentile95% intervals; no test-fitted cutoffs. '
        'These intervals do not measure training-seed uncertainty.','',
        f'- Standard ΔP_E (fraction): {number(result["standard_pe"]["difference"])}',
        f'- Pooled hard k1 low ΔAUC: {number(result["pooled_hard_auc"]["difference"])}',
        f'- Selected epoch (one-based): {result["selected_epoch_one_based"]}']
    for scope in ('standard_pe','pooled_hard_auc'):
        for name,record in result[scope]['models'].items():lines.append(f'- {scope}, {name}: {number(record)}')
    for name,t in result['threshold_metadata'].items():lines.append(f'- {name} deployment cutoff: {t["cutoff"]:.17g}')
    if result['cells']:
        cell=result['cells'];lines.append('- Head-vs-head primary ΔAUC: '+number(cell['primary']['difference']))
        lines.append(f'- Eligible contiguous stego tiles: {cell["primary"]["eligible_tiles"]}; source photos: {cell["primary"]["eligible_photos"]}')
        for name,point in cell['primary']['models'].items():lines.append(f'- {name} head primary AUC: '+number(point))
        for name,point in cell['operating_points'].items():lines.append(f'- {name} cell FA: {point["test_fa"]:.8f}; validation-frozen cutoff {point["cutoff"]:.17g}')
    if result['rule_set']=='curriculum50' and result['selected_epoch_one_based']<8:
        lines.append('- No selected curriculum exposure: no curriculum-benefit claim.')
    lines+=['','## Literal checks','','```json',json.dumps({k:result[k] for k in ('passed','checks','routes') if k in result},indent=2),'```',
        '','## Context only','','| Scope | ΔAUC [95%] |','| --- | --- |']
    for key,item in result['context_hard_auc'].items():lines.append(f'| {key} | {number(item["difference"])} |')
    for key,item in result['standard_panels'].items():lines.append(f'| standard/{key} | {number(item["difference"])} |')
    lines+=['','All numbers, input hashes and frozen threshold metadata: `evidence.json`. Shared draws: `bootstrap-replicates.npz`.']
    (output/'report.md').write_text('\n'.join(lines)+'\n')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference',type=Path,required=True,help='run or exact evaluation directory')
    parser.add_argument('--candidate',type=Path,required=True,help='run or exact evaluation directory')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--rule',choices=RULES,required=True)
    parser.add_argument('--reference-cells',type=Path)
    parser.add_argument('--candidate-cells',type=Path)
    args=parser.parse_args()
    result=summarize(args.reference,args.candidate,args.output,args.rule,
        reference_cells=args.reference_cells,candidate_cells=args.candidate_cells)
    print(json.dumps({k:result[k] for k in ('rule_set','passed','standard_pe','pooled_hard_auc')},indent=2))
    return 0  # A measured FAIL is a valid analysis, not a tool error.


if __name__=='__main__':
    os.environ.update(CUDA_VISIBLE_DEVICES='',HIP_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1',
        OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
    raise SystemExit(main())
