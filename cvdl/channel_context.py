"""Opt-in B36 context on original #11 bytes; never edits the original panel."""
from pathlib import Path
import json
import subprocess
import time
import numpy as np
import pandas as pd
from . import channel_analysis as old

ORIGINAL_COMMIT='9361d8b'
RECIPE_SHA='3604b153f8aee490a93742dfda727baf26af55de19090d74223e1f5c9cb96e11'
# Only models.py may differ from the original frozen code. Frozen before scoring.
MODEL_SHA='515a9f99d791a0c104812cf6c34e9bd78510a1c6b134d3a13a0db7ea39a269a8'
METHOD='wiser-b-rgb-context-v1'


def validate_original(root,panel):
    root,panel=Path(root),Path(panel)
    recipe=json.loads((panel/'recipe.json').read_text())
    if old.digest(old.canonical(recipe).encode())!=RECIPE_SHA or (panel/'recipe.sha256').read_text().strip()!=RECIPE_SHA:
        raise ValueError('original recipe changed')
    for path,expected in recipe['files_sha256'].items():
        if old.file_hash(panel/path)!=expected:raise ValueError('original panel file changed: '+path)
    for path,expected in recipe['inputs'].items():
        if old.file_hash(root/path)!=expected:raise ValueError('original input changed: '+path)
    for path,expected in recipe['code_sha256'].items():
        blob=subprocess.check_output(['git','show',ORIGINAL_COMMIT+':'+path],cwd=root)
        if old.digest(blob)!=expected:raise ValueError('untrusted original code: '+path)
        current=old.file_hash(root/path)
        if current!=(MODEL_SHA if path=='cvdl/models.py' else expected):
            raise ValueError('undeclared code drift: '+path)
    if old.models(root)!=recipe['models']:raise ValueError('original model drift')
    base=pd.read_csv(panel/'base-tiles.csv',float_precision='round_trip')
    variants=pd.read_csv(panel/'variants.csv',float_precision='round_trip')
    cache=np.load(panel/'clean.npy',mmap_mode='r',allow_pickle=False)
    if (base.source_photo.nunique()!=480 or len(base)!=recipe['tiles'] or len(variants)!=recipe['stego_images'] or
        cache.shape!=(len(base),256,256,3) or cache.dtype!=np.uint8 or base.cover_id.duplicated().any() or
        list(base.base_index)!=list(range(len(base))) or list(variants.variant_id)!=list(range(len(variants)))):
        raise ValueError('original geometry drift')
    for row in base.to_dict('records'):
        if old.digest(cache[row['base_index']].tobytes())!=row['clean_sha256']:raise ValueError('clean byte drift')
    execution=json.loads((panel/'models/r1/execution.json').read_text())
    if (execution['recipe_sha256']!=RECIPE_SHA or execution['source']!=recipe['models']['r1'] or
        execution['predictions_sha256']!=old.file_hash(panel/'models/r1/predictions.csv')):
        raise ValueError('original R1 execution drift')
    return recipe,base,variants


def snapshot(panel):
    return {str(p.relative_to(panel)):old.file_hash(p) for p in sorted(panel.rglob('*')) if p.is_file()}


def freeze(root,panel,evaluation,output):
    """Pin provenance before inference; output must be a new sibling folder."""
    from scripts.training_decision import validate_evaluation,wiser_identity,WISER_REFERENCE
    root,panel,evaluation,output=map(lambda p:Path(p).resolve(),(root,panel,evaluation,output))
    if output.parent!=panel.parent or output==panel or output.exists():raise ValueError('use a new sibling output')
    recipe,_,_=validate_original(root,panel)
    reference=root/WISER_REFERENCE
    _,meta,calibration=validate_evaluation(evaluation)
    epoch,checkpoint=wiser_identity([reference,evaluation],[json.loads((reference/'evaluation_settings.json').read_text()),meta])
    manifest=dict(method=METHOD,context_only=True,original_panel=str(panel),original_sha256=snapshot(panel),
        recipe_sha256=RECIPE_SHA,original_commit=ORIGINAL_COMMIT,original_model_sha256=recipe['code_sha256']['cvdl/models.py'],
        current_model_sha256=MODEL_SHA,context_code_sha256=old.file_hash(Path(__file__)),
        original_code_sha256=recipe['code_sha256'],
        current_code_sha256=dict(recipe['code_sha256'],**{'cvdl/models.py':MODEL_SHA}),
        candidate_identity=dict(model='xunet_wide_coloursep36',front_end='srm',norm='none',width=4,
            paired_batches=True,max_epochs=50,selection='best'),
        checkpoint=str(checkpoint),checkpoint_sha256=old.file_hash(checkpoint),selected_epoch=epoch,
        evaluation=str(evaluation),evaluation_inputs={p:old.file_hash(evaluation/p) for p in ('summary.json','evaluation_settings.json','validation/calibration.json')},
        calibration=calibration,batch_size=32,workers=4,precision='fp32',rocm_conv1_fix=False,
        bootstrap_replicates=5000,bootstrap_seed=20261024)
    output.mkdir();old.dump(output/'manifest.json',manifest)
    (output/'manifest.sha256').write_text(old.digest(old.canonical(manifest).encode())+'\n')
    return manifest


def checked(root,output):
    output=Path(output);m=json.loads((output/'manifest.json').read_text());panel=Path(m['original_panel'])
    if old.digest(old.canonical(m).encode())!=(output/'manifest.sha256').read_text().strip():raise ValueError('context manifest drift')
    if m['context_code_sha256']!=old.file_hash(Path(__file__)) or m['current_model_sha256']!=MODEL_SHA:raise ValueError('context code drift')
    recipe,base,variants=validate_original(root,panel)
    if snapshot(panel)!=m['original_sha256']:raise ValueError('original folder changed')
    if old.file_hash(m['checkpoint'])!=m['checkpoint_sha256']:raise ValueError('candidate checkpoint drift')
    for p,h in m['evaluation_inputs'].items():
        if old.file_hash(Path(m['evaluation'])/p)!=h:raise ValueError('candidate calibration drift')
    return m,recipe,base,variants


def score(root,output):
    import torch
    from .standardtest import load_network
    from scripts.pc_safety import available_gb,check_workers
    output=Path(output);m,_,base,variants=checked(root,output)
    if (output/'predictions.csv').exists():raise ValueError('refusing score overwrite')
    check_workers(4)
    if available_gb()<12 or not torch.cuda.is_available():raise ValueError('GPU/RAM safety floor')
    torch.set_num_threads(1)
    network=load_network(Path(m['checkpoint']),'cuda')
    loader=torch.utils.data.DataLoader(old.ChannelDataset(m['original_panel'],base,variants),batch_size=32,
        num_workers=4,shuffle=False,multiprocessing_context='spawn')
    rows=[];start=time.monotonic()
    with torch.inference_mode():
        for images,indices,_,_ in loader:
            if available_gb()<12 or time.monotonic()-start>1800:raise ValueError('context safety limit')
            probabilities=network(images.to('cuda',dtype=torch.float32)).softmax(1)[:,1].cpu().numpy()
            if not np.isfinite(probabilities).all():raise ValueError('nonfinite context scores')
            rows.extend(dict(image_index=i,prob_stego=float(p)) for i,p in zip(indices.tolist(),probabilities))
    if [r['image_index'] for r in rows]!=list(range(len(base)+len(variants))):raise ValueError('context row mismatch')
    checked(root,output)
    pd.DataFrame(rows).to_csv(output/'predictions.csv',index=False,lineterminator='\n')
    old.dump(output/'execution.json',dict(manifest_sha256=old.file_hash(output/'manifest.json'),
        predictions_sha256=old.file_hash(output/'predictions.csv'),seconds=time.monotonic()-start,images=len(rows)))


def report(root,output):
    output=Path(output);m,recipe,base,variants=checked(root,output)
    if (output/'report.json').exists() or (output/'REPORT.md').exists():raise ValueError('refusing report overwrite')
    execution=json.loads((output/'execution.json').read_text())
    if execution['manifest_sha256']!=old.file_hash(output/'manifest.json') or execution['predictions_sha256']!=old.file_hash(output/'predictions.csv'):
        raise ValueError('context score drift')
    weights=old.bootstrap_weights(base,5000);results={}
    for name,path,cutoff in [('r1',Path(m['original_panel'])/'models/r1/predictions.csv',recipe['models']['r1']['cutoff']),
                              ('b36',output/'predictions.csv',m['calibration']['cutoff'])]:
        frame=pd.read_csv(path,float_precision='round_trip')
        if list(frame.image_index)!=list(range(len(base)+len(variants))):raise ValueError('context join mismatch')
        results[name]=old.summarize_model(base,variants,frame,cutoff,weights)
    checked(root,output)
    old.dump(output/'report.json',dict(context_only=True,main_rule_affected=False,models=results,
        bootstrap_draws_sha256=old.digest(weights.astype('<f8').tobytes()),manifest_sha256=old.file_hash(output/'manifest.json')))
    lines=['# B36 frozen RGB context', '', 'Reused TEST context only. This report cannot change wiser_b or select a width.',
        '480 photo clusters; 5,000 shared draws, seed20261024. Validation-only fixed cutoffs.',
        'Single-channel density and colour confounds remain. No cell-head comparison.', '',
        '| Model | Arm | k1 random low-rate macro AUC | P_E |', '|---|---|---:|---:|']
    for name,result in results.items():
        for arm,value in result['primary'].items():
            lines.append(f"| {name} | {arm} | {value['AUC']['point']:.8f} | {value['P_E']['point']:.8f} |")
    lines+=['', 'Arm contrasts and source/noise/placement results with paired intervals: report.json.',
        'Original #11 scores, execution records and panel files remain unchanged.']
    (output/'REPORT.md').write_text('\n'.join(lines)+'\n')
