#!/usr/bin/env python3
"""Saved scores only: run21 detector reference and run18 auxiliary-map reference.

Numeric paired intervals, not a frozen verdict or model-adoption command.
"""
import argparse
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))

from cvdl.cell_artifacts import sha,validate_scores,validate_cutoffs,checkpoint_identity,FILES
from scripts import training_decision as d
from scripts.architecture_decision import join_predictions,auc_intervals,PANELS,TARGETS

PINS=ROOT/'configs/run21-cell-comparison-v1.json'
PINS_SHA='8f7deb06dcab576001ef69055cc840054c329e2592e2db265242cc7aadd606f4'


def load_pins():
    if sha(PINS)!=PINS_SHA:raise ValueError('reference pin record changed')
    return json.loads(PINS.read_text())


def pinned(folder,files):
    if any(sha(folder/name)!=value for name,value in files.items()):raise ValueError('pinned reference artifact changed')


def compatible_calibration(reference,candidate):
    shared=('recipe_id','recipe_sha256','selected_pairs_sha256','full_pairs_sha256','geometry','policy')
    if any(reference[k]!=candidate[k] for k in shared):raise ValueError('deployment calibration recipe/list/geometry/policy mismatch')
    if reference.get('threshold_origin')!='post_hoc_256' or candidate.get('threshold_origin')!='train_matched':
        raise ValueError('expected pinned post_hoc_256 reference and candidate own train_matched calibration')


def summarize(candidate,candidate_cells,output):
    import numpy as np
    import pandas as pd
    from cvdl.metrics import threshold
    from cvdl.gapcov_provenance import validate_sources,CELL_AUX
    from cvdl.pilot import refuse_run
    pins=load_pins();reference=ROOT/pins['detector']['evaluation'];cells_reference=ROOT/pins['heatmap']['folder']
    candidate=d.evaluation_folder(candidate);candidate_cells=Path(candidate_cells).resolve();output=Path(output).resolve()
    if output.exists() and any(output.iterdir()):raise ValueError('use an empty comparison output')
    sources=[reference,candidate,cells_reference,candidate_cells]
    if any(output==p.resolve() or p.resolve() in output.parents or output in p.resolve().parents for p in sources):
        raise ValueError('output must not overlap inputs')
    pinned(reference,pins['detector']['files_sha256']);pinned(cells_reference,pins['heatmap']['files_sha256'])
    metadata=[json.loads((p/'evaluation_settings.json').read_text()) for p in (reference,candidate)]
    checkpoints=[d.checkpoint_path(m) for m in metadata]
    for path in checkpoints:refuse_run(path.parent)
    if sha(checkpoints[0])!=pins['detector']['checkpoint_sha256']:raise ValueError('pinned detector checkpoint changed')
    if checkpoints[1].name!='last.ckpt':raise ValueError('candidate comparison requires its last checkpoint')
    identity=checkpoint_identity(checkpoints[1]);provenance=validate_sources(ROOT,CELL_AUX)
    import torch
    h=torch.load(checkpoints[1],map_location='cpu',weights_only=False)['hyper_parameters']
    if (identity['source_provenance']!=provenance or identity['model']!='xunet_wide_gapcov32' or
        identity['cell_supervision']!='changed_cells_v1' or
        h.get('initialization_checkpoint_sha256')!=pins['detector']['checkpoint_sha256']):
        raise ValueError('candidate checkpoint source/head/initializer mismatch')
    if any(m.get('quick') or m.get('values',{}).get('pilot') or m.get('values',{}).get('quick') for m in metadata):
        raise ValueError('quick/pilot comparison refused')
    paths=[PINS,checkpoints[0],checkpoints[1],*[p for folder in (reference,candidate) for p in d.evaluation_inputs(folder)],
           *[cells_reference/name for name in pins['heatmap']['files_sha256']],
           *[candidate_cells/name for name in (*FILES,'artifact_identity.json')],ROOT/'configs/cell-diagnostic-v1.json',
           *[ROOT/name for name in provenance['current_files_sha256']]]
    def snapshot():return {str(p.resolve()):sha(p) for p in paths}
    before=snapshot()
    validated=[d.validate_evaluation(p) for p in (reference,candidate)];calibrations=[v[2] for v in validated]
    if [v[1] for v in validated]!=metadata or any(t['checkpoint_sha256']!=sha(p) for t,p in zip(calibrations,checkpoints)):
        raise ValueError('evaluation/checkpoint inputs changed or mismatched')
    compatible_calibration(*calibrations)
    if metadata[0]['hard_recipe_sha256']!=metadata[1]['hard_recipe_sha256']:raise ValueError('hard recipe mismatch')
    cell_summary=validate_scores(candidate_cells,checkpoint=checkpoints[1],expected_sha=metadata[1]['checkpoint_sha256'],require_v2=True)
    manifest=json.loads((candidate_cells/'manifest.json').read_text())
    if manifest!=json.loads((ROOT/'configs/cell-diagnostic-v1.json').read_text()):raise ValueError('candidate cell manifest mismatch')
    reference_summary=json.loads((cells_reference/'summary.json').read_text())
    for folder,summary in ((cells_reference,reference_summary),(candidate_cells,cell_summary)):
        validate_cutoffs(folder,summary,['aux'])
        val=pd.read_csv(folder/'val-pairs.csv');test=pd.read_csv(folder/'test-pairs.csv')
        if set(val.pair_id)&set(test.pair_id):raise ValueError('cell validation/test row overlap')
    def joined(relative):
        frame=join_predictions([p/relative for p in (reference,candidate)],names=d.NAMES)
        if not np.isfinite(frame[list(d.NAMES)]).all().all() or not frame[list(d.NAMES)].ge(0).all().all() or not frame[list(d.NAMES)].le(1).all().all():
            raise ValueError('invalid saved probabilities')
        return frame
    validation=joined('validation/predictions.csv')
    if 'split' not in validation or set(validation.split)!={'val'}:raise ValueError('detector calibration needs val rows')
    for name,calibration in zip(d.NAMES,calibrations):
        frame=validation.drop(columns=[n for n in d.NAMES if n!=name]).rename(columns={name:'prob_stego'})
        fitted,_=threshold(frame,'pe')
        if fitted!=calibration['cutoff']:raise ValueError('detector cutoff is not own validation min-PE cutoff')
    hard=joined('hard-test/hard_predictions.csv');panels={panel:joined(f'panels/{panel}/predictions.csv') for panel in PANELS}
    if set(validation.source_photo)&(set(hard.source_photo)|set().union(*(set(f.source_photo) for f in panels.values()))):
        raise ValueError('detector validation/test photo overlap')
    for summary,_,_ in validated:
        if int((hard.label==1).sum())!=summary['hard_test']['pairs']:raise ValueError('incomplete hard rows')
        for panel,frame in panels.items():
            expected=next(t['pairs'] for t in summary['timings'] if t['panel']==panel and not t.get('higher_k',False))
            if int((frame.label==1).sum())!=expected:raise ValueError('incomplete panel rows')
    low=hard[hard.panel.isin(TARGETS)&(hard.k==1)&(hard.condition=='low')]
    if set(low.panel)!=set(TARGETS) or set(low.method)!={'replace','match'} or set(low.placement)!={'contiguous','random','stride'}:
        raise ValueError('hard scope coverage mismatch')
    cutoffs={name:t['cutoff'] for name,t in zip(d.NAMES,calibrations)}
    hard_result,hard_draws=auc_intervals(low,names=d.NAMES)
    standard,standard_draws=auc_intervals(panels['standard'],names=d.NAMES,metric='pe',cutoffs=cutoffs)
    cells,cell_draws=d.compare_cells(cells_reference,candidate_cells,reference_map='aux',candidate_map='aux')
    proposed=dict(status='PROPOSED / Pending (user)',standard_pe_upper_at_most_0025=standard['difference']['high']<=.0025,
        hard_auc_lower_at_least_minus_001=hard_result['difference']['low']>=-.01,
        candidate_cell_fa_at_most_007=cells['operating_points']['candidate']['test_fa']<=.07)
    after=snapshot()
    if load_pins()!=pins:raise ValueError('reference pin inputs changed')
    pinned(reference,pins['detector']['files_sha256']);pinned(cells_reference,pins['heatmap']['files_sha256'])
    if before!=after:raise ValueError('comparison inputs changed during processing')
    result=dict(version='run21-cell-comparison-v1',adoption='not decided; no frozen PASS/FAIL',
        detector_reference=dict(evaluation=str(reference),checkpoint_sha256=pins['detector']['checkpoint_sha256']),
        heatmap_reference=dict(folder=str(cells_reference),checkpoint_sha256=pins['heatmap']['checkpoint_sha256'],
            binding='exact original saved artifacts; no retroactive checkpoint metadata'),
        candidate=dict(evaluation=str(candidate),cells=str(candidate_cells),**identity),
        detector_bootstrap=dict(seed=20261019,replicates=2000,unit='source_photo',interval='percentile95'),
        cell_bootstrap=dict(seed=20261021,replicates=2000,unit='source_photo',interval='percentile95'),
        calibrations=dict(zip(d.NAMES,calibrations)),standard_pe=standard,pooled_hard_auc=hard_result,cells=cells,
        proposed_guards=proposed,input_sha256=before,input_sha256_after=after,inputs_unchanged=True)
    output.mkdir(parents=True,exist_ok=True)
    (output/'comparison.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    np.savez_compressed(output/'bootstrap-replicates.npz',standard_pe=standard_draws,hard_auc=hard_draws,**{'cell_'+k:v for k,v in cell_draws.items()})
    lines=['# Two-reference saved-score comparison','','PROPOSED guards: Pending (user). No frozen verdict or automatic adoption.',
        'Detector reference: run21 BEST. Auxiliary heatmap reference: run18 last. Own validation cutoffs only.',
        'Shared paired source-photo draws: 2,000; detector seed20261019, cell seed20261021. Not training-seed uncertainty.',
        '',f'Standard delta P_E: {standard["difference"]}',f'Hard k1 low delta AUC: {hard_result["difference"]}',
        f'Cell head delta AUC: {cells["primary"]["difference"]}',f'Candidate cell FA: {cells["operating_points"]["candidate"]["test_fa"]}',
        '',json.dumps(proposed,indent=2)]
    (output/'REPORT.md').write_text('\n'.join(lines)+'\n')
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('candidate',type=Path);parser.add_argument('--candidate-cells',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    summarize(args.candidate,args.candidate_cells,args.output)


if __name__=='__main__':main()
