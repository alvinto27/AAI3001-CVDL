"""Evaluate one downloaded run on the PC; never fit or modify its checkpoints."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
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


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run_folder',type=Path)
    parser.add_argument('--padding-check',action='store_true')
    parser.add_argument('--padding-pairs',type=int,default=2000,
        help='cases per (pattern, level, real shape), chosen by a fixed hash from the standard test list; 0 = all')
    args=parser.parse_args()
    evaluate(args.run_folder,padding_check=args.padding_check,padding_pairs=args.padding_pairs)


if __name__=='__main__':main()
