"""Evaluate one downloaded run on the PC; never fit or modify its checkpoints."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
VALIDATION_SHA256='6bb3c8efadad06af50ba014a95c98b4fc57008119bdd55b794ec2a1729e4d5ca'
sys.path.insert(0,str(ROOT))


def checkpoint_folder(path):
    paths=sorted({p.parent for p in Path(path).resolve().rglob('best-*.ckpt')})
    if len(paths)!=1:raise ValueError('run folder must contain exactly one checkpoint directory')
    return paths[0]


def portable_config(value):
    if value is None:return None
    path=Path(value)
    if path.is_absolute():
        if 'configs' not in path.parts:raise ValueError(f'cannot rebase saved config {value}')
        path=Path(*path.parts[path.parts.index('configs'):])
    target=(ROOT/path).resolve()
    if ROOT not in target.parents or not target.is_file():raise ValueError(f'missing saved config: {value}')
    return str(target)


def settings_for_run(folder):
    from cvdl.settings import SCHEMA, automatic_profile, load
    profile=automatic_profile()
    if profile.name!='pc':raise ValueError('evaluate_run is PC-only; G4 jobs only train')
    saved=folder/'run_settings.json'
    if saved.exists():
        values=json.loads(saved.read_text())['values']
        experiment={k:v for k,v in values.items() if k in SCHEMA and not SCHEMA[k].machine}
        overrides=dict(quick=values['quick'],job_id=values.get('job_id'))
    else:
        # Explicit migration for pre-package G4 artefacts; never guess a notebook.
        model=json.loads((folder/'model_run_settings.json').read_text())
        record=json.loads((folder/'data_run_settings.json').read_text());old=record['settings']
        experiment={k:model[k] for k in ('norm','clip_t','lr','warmup_epochs','lr_drops','batch_size','model','width','train_yaml','extra_tests')}
        experiment.update(epochs=model['max_epochs'],grad_clip=model['grad_clip_norm'],run_name=folder.name,
            checkpoint_validation=record['checkpoint_validation'],val_cap_raise=record.get('val_cap_raise',5000),
            val_cap_lsdir=record.get('val_cap_lsdir',1500),eval_profile=record['eval_profile'],data=record['data'])
        for key in ('sources','source','tile_size','source_mix','tiles_per_epoch','run_seed','train_photo_fraction','paired_batches'):
            experiment[key]=old[key]
        overrides=dict(quick=record['quick'])
    experiment['train_yaml']=portable_config(experiment.get('train_yaml'))
    return load(environ={},_profile=profile,overrides=overrides,**experiment)


def evaluate(folder, *, padding_check=False, padding_pairs=2000):
    from cvdl import runtime, data, post_fit
    from cvdl.settings import plain
    from cvdl.lit import select_best_checkpoint
    folder=checkpoint_folder(folder);settings=settings_for_run(folder)
    checkpoint=Path(select_best_checkpoint('',folder))
    runtime.configure(settings)
    otf,tables=runtime.create_data(settings)
    if otf is not None:
        existing=folder/'validation_pairs.csv'
        if existing.exists() and existing.read_text()!=otf.validation_pair_csv:
            raise ValueError('saved validation identity differs; refuse evaluation')
        loaders=data.build_otf_loaders(otf,batch_size=settings.batch_size,workers=settings.workers,
            eval_workers=settings.eval_workers,pin_memory=settings.device=='cuda',prefetch=settings.prefetch,
            prefetch_images=settings.prefetch_images,prime_next_epoch=False,eval_persistent_workers=False)
    else:
        loaders=data.build_png_loaders(tables.run_pairs,settings.data_root,batch_size=settings.batch_size,
            workers=settings.workers,eval_workers=settings.eval_workers,pin_memory=settings.device=='cuda',
            paired_batches=settings.paired_batches,seed=settings.seed)
    import torch, numpy, PIL
    provenance=dict(runtime=dict(torch=str(torch.__version__),numpy=numpy.__version__,pillow=PIL.__version__),
        values=plain(settings.values),sources=plain(settings.provenance),profile=settings.profile,
        checkpoint=str(checkpoint),checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest())
    (folder/'evaluation_settings.json').write_text(json.dumps(provenance,indent=2)+'\n')
    model,predictions,result=post_fit.run(settings,otf,tables,loaders,checkpoint,folder)
    if padding_check:
        if otf is None:raise ValueError('padding diagnostic requires OTF data')
        from cvdl.padding import check
        check(model.net,otf.test,predictions['test'],predictions['val'],folder,
              device=settings.device,quick=settings.quick,limit=padding_pairs,workers=settings.eval_workers)
    print(f'EVALUATION COMPLETE: {folder}',flush=True)
    return result


def score_logits(model, loader, *, device):
    """Raw two-class logits for every clean/stego row, in the same order as predict_pairs."""
    import numpy as np, pandas as pd, torch
    model.eval();network=getattr(model,'net',model);frame=loader.dataset.frame;rows=[]
    with torch.inference_mode():
        for clean,stego,indices in loader:
            images=torch.cat([clean,stego],dim=0).to(device,dtype=torch.float32)
            output=network(images)
            logits=output.double().cpu().numpy()
            probability=torch.softmax(output,dim=1)[:,1].cpu().numpy()
            count=clean.shape[0]
            for position,index in enumerate(indices.tolist()):
                row=frame.iloc[int(index)]
                for label,at in ((0,position),(1,count+position)):
                    rows.append(dict(cover_id=row['cover_id'],source=row['source'],k=row['k'],label=label,
                        prob_stego=float(probability[at]),logit0=logits[at,0],logit1=logits[at,1],logit=logits[at,1]-logits[at,0]))
    return pd.DataFrame(rows)


def calibrate(folder, *, settings_from=None, output=None):
    """Temperature scaling on the capped validation logits; writes NEW files only.

    One T minimises NLL of sigmoid(logit/T) on the capped validation set. Outputs:
    validation_logits.csv, test_logits.csv, *_predictions_calibrated.csv, calibration.json.
    """
    import pandas as pd
    from cvdl import runtime, data
    from cvdl.calibration import fit_temperature, summarise, sigmoid
    from cvdl.lit import SteganalysisModule, select_best_checkpoint
    folder=checkpoint_folder(folder)
    settings=settings_for_run(checkpoint_folder(settings_from) if settings_from else folder)
    out=Path(output) if output else folder;out.mkdir(parents=True,exist_ok=True)
    checkpoint=Path(select_best_checkpoint('',folder))
    runtime.configure(settings)
    otf,_=runtime.create_data(settings)
    if otf is None:raise ValueError('temperature scaling requires OTF data')
    identity=hashlib.sha256(otf.validation_pair_csv.encode()).hexdigest()
    if identity!=VALIDATION_SHA256:raise ValueError(f'capped validation identity differs: {identity}')
    loaders=data.build_otf_loaders(otf,batch_size=settings.batch_size,workers=0,eval_workers=settings.eval_workers,
        pin_memory=settings.device=='cuda',prime_next_epoch=False,eval_persistent_workers=False)
    model=SteganalysisModule.load_from_checkpoint(checkpoint,map_location=settings.device,weights_only=False).to(settings.device)
    frames={name:score_logits(model,loader,device=settings.device) for name,loader in (('validation',loaders.val),('test',loaders.test))}
    saved=folder/'test_predictions.csv'
    check=None
    if saved.exists():
        old=pd.read_csv(saved,dtype={'cover_id':'string'})
        if len(old)==len(frames['test']):check=float((old.prob_stego.to_numpy()-frames['test'].prob_stego.to_numpy()).__abs__().max())
    T=fit_temperature(frames['validation'].logit.to_numpy(),frames['validation'].label.to_numpy())
    result=dict(version='temperature-v1',checkpoint=str(checkpoint),checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        validation_pairs_sha256=identity,temperature=T,saved_test_prob_max_abs_diff=check,
        fitted_on='capped validation (NLL of sigmoid(logit/T), both labels)')
    for name,frame in frames.items():
        frame.to_csv(out/f'{name}_logits.csv',index=False)
        calibrated=frame.copy();calibrated['prob_stego']=sigmoid(frame.logit.to_numpy()/T)
        calibrated.to_csv(out/f'{name}_predictions_calibrated.csv',index=False)
        result[name]=summarise(frame,T=T)
    (out/'calibration.json').write_text(json.dumps(result,indent=2)+'\n')
    print(f'CALIBRATION COMPLETE: T={T:.4f} -> {out}',flush=True)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run_folder',type=Path)
    parser.add_argument('--padding-check',action='store_true')
    parser.add_argument('--padding-pairs',type=int,default=2000,
        help='cases per (pattern, level, real shape), chosen by a fixed hash from the standard test list; 0 = all')
    parser.add_argument('--temperature',action='store_true',
        help='only fit temperature scaling on the capped validation logits; writes new files')
    parser.add_argument('--settings-from',type=Path,help='take data settings from another run folder (old runs)')
    parser.add_argument('--output',type=Path,help='folder for the new calibration files (default: the checkpoint folder)')
    args=parser.parse_args()
    if args.temperature:
        calibrate(args.run_folder,settings_from=args.settings_from,output=args.output);return
    evaluate(args.run_folder,padding_check=args.padding_check,padding_pairs=args.padding_pairs)


if __name__=='__main__':main()
