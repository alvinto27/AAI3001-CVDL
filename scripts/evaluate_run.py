"""Evaluate a downloaded run on the frozen 256 suites; never fit its checkpoints.

--settings-from borrows data settings, not model architecture. Saved training
records stay intact. Legacy-512 evaluation is retired from this command.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def checkpoint_folder(path):
    from cvdl.pilot import refuse_run
    refuse_run(path)
    paths=sorted({p.parent for p in Path(path).resolve().rglob('best-*.ckpt')})
    if len(paths)!=1:raise ValueError('run folder must contain exactly one checkpoint directory')
    refuse_run(paths[0])
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


EVALUATION_OVERRIDE_KEYS={'rocm_conv1_fix':'rocm_conv1_fix','eval_batch_size':'batch_size',
                          'batch_size':'batch_size','eval_workers':'eval_workers'}


def evaluation_override_records(arguments):
    from cvdl.settings import parse
    records=[]
    for argument in arguments:
        key,separator,value=argument.partition('=')
        if not separator or key not in EVALUATION_OVERRIDE_KEYS:
            raise ValueError('forbidden evaluation --set '+repr(argument)+'; allowed: '+
                ', '.join(EVALUATION_OVERRIDE_KEYS)+'. Model, data and test identity cannot be overridden.')
        target=EVALUATION_OVERRIDE_KEYS[key]
        records.append(dict(argument=argument,key=target,value=parse(target,value),source='CLI --set (evaluation only)'))
    return records


def evaluation_override_values(arguments):
    return {r['key']:r['value'] for r in evaluation_override_records(arguments)}


# This historical notebook predates checkpoint-validation and mixed-epoch fields.
# Its saved loader output has 22,476 validation images (all 11,238 RAISE/DIV2K
# val pairs), and its RunSettings call has neither mixture nor epoch quotas.
# See docs/active/legacy-wide-r2-migration.md. Never infer from a run name alone.
WIDE_R2_EVIDENCE={
    'data_run_settings.json':'bb9945fbd236497823c6b8e84bdf3f68b9e9300b39ed7683d5372154ed0b4ebb',
    'model_run_settings.json':'9835987926ba07f439e262c7bd6d3d5a66d3edc58ab42cc847dd3e5e8eb5d24c',
    'executed.ipynb':'431f0d2188f2641f0cc0a9b1af04a18e6dd4be9ae420f196164ed3447f52ed64',
}


def legacy_data_record(folder):
    """Resolve only the evidenced old wide-R2 omissions; never change saved files."""
    folder=Path(folder)
    record=json.loads((folder/'data_run_settings.json').read_text())
    old=record.get('settings',{})
    required=('sources','source','tile_size','source_mix','tiles_per_epoch','run_seed','train_photo_fraction','paired_batches')
    missing=[f'settings.{k}' for k in required if k not in old]
    missing += [k for k in ('checkpoint_validation','eval_profile','data','quick') if k not in record]
    if not missing:return record,None
    evidence={name:folder/name for name in ('data_run_settings.json','model_run_settings.json')}
    evidence['executed.ipynb']=folder.parent.parent/'executed.ipynb'
    if set(missing)!={'checkpoint_validation','settings.source_mix','settings.tiles_per_epoch'} or any(
        not p.is_file() or hashlib.sha256(p.read_bytes()).hexdigest()!=WIDE_R2_EVIDENCE[name]
        for name,p in evidence.items()):
        raise ValueError(f'missing legacy settings {missing} in {folder}; no approved saved-record evidence for migration')
    # The three pinned files prove the historical standard/full-source loader.
    record={**record,'checkpoint_validation':'standard',
            'settings':{**old,'source_mix':None,'tiles_per_epoch':None}}
    return record,dict(migration='wide-r2-full-source-v1',evidence_sha256=WIDE_R2_EVIDENCE,
        derived={'checkpoint_validation':'standard','source_mix':None,'tiles_per_epoch':None},
        original_validation_pairs=11238,original_training_pairs=53950)


def settings_for_run(folder, *, eval_overrides=()):
    from cvdl.pilot import refuse_run
    refuse_run(folder)
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
        record,migration=legacy_data_record(folder);old=record['settings']
        if migration:print('LEGACY SETTINGS MIGRATION: '+json.dumps(migration,sort_keys=True),flush=True)
        experiment={k:model[k] for k in ('norm','clip_t','lr','warmup_epochs','lr_drops','batch_size','model','width','train_yaml','extra_tests')}
        experiment.update(epochs=model['max_epochs'],grad_clip=model['grad_clip_norm'],run_name=folder.name,
            checkpoint_validation=record['checkpoint_validation'],val_cap_raise=record.get('val_cap_raise',5000),
            val_cap_lsdir=record.get('val_cap_lsdir',1500),eval_profile=record['eval_profile'],data=record['data'])
        for key in ('sources','source','tile_size','source_mix','tiles_per_epoch','run_seed','train_photo_fraction','paired_batches'):
            experiment[key]=old[key]
        overrides=dict(quick=record['quick'])
    experiment['train_yaml']=portable_config(experiment.get('train_yaml'))
    overrides.update(evaluation_override_values(eval_overrides))
    # Resolve/validate the original training recipe FIRST. The fixed scratch
    # curriculum intentionally refuses non-128 training batches; an evaluation
    # batch is not a training-recipe change. Never relax its settings guard.
    evaluation_batch=(overrides.pop('batch_size',None)
                      if experiment.get('embedding_schedule')=='low_bpp50_v1' else None)
    settings=load(environ={},_profile=profile,overrides=overrides,**experiment)
    if evaluation_batch is None:return settings
    from dataclasses import replace
    from cvdl.settings import freeze
    return replace(settings,values=freeze(dict(settings.values,batch_size=evaluation_batch)),
        provenance=freeze(dict(settings.provenance,batch_size='override: evaluation only')),
        overrides=freeze((*settings.overrides,dict(name='batch_size',value=evaluation_batch))))


# These are exactly the architecture choices read by model_from_hparams.
MODEL_DEFAULTS=dict(model='xunet',width=4,norm='group',clip_t=3.0,front_end='kv')
MODEL_TRAINING_KEYS=('lr','warmup_epochs','lr_drops','grad_clip','epochs','seed')


def settings_for_evaluation(folder, checkpoint, settings_from=None, *, eval_overrides=()):
    """Borrow data, not the target model. Inspect trusted checkpoints on CPU only."""
    context={'evaluation_overrides':evaluation_override_records(eval_overrides)} if eval_overrides else {}
    if settings_from is None:
        settings=settings_for_run(folder,eval_overrides=eval_overrides) if eval_overrides else settings_for_run(folder)
        if not (folder/'run_settings.json').exists() and (folder/'data_run_settings.json').is_file():
            _,migration=legacy_data_record(folder)
            if migration:context['legacy_settings_migration']=migration
        return settings,context,None
    import torch
    from cvdl.settings import SCHEMA, automatic_profile, load, plain, parse
    source=checkpoint_folder(settings_from)
    borrowed=settings_for_run(source)
    saved=folder/'run_settings.json'
    if saved.exists():
        own=json.loads(saved.read_text())['values']
    else:
        own=json.loads((folder/'model_run_settings.json').read_text())
        own={**own,**{new:own[old] for old,new in (('max_epochs','epochs'),('grad_clip_norm','grad_clip')) if old in own}}
    hparams=torch.load(checkpoint,map_location='cpu',weights_only=False).get('hyper_parameters',{})
    architecture={k:parse(k,hparams.get(k,default)) for k,default in MODEL_DEFAULTS.items()}
    needed=[k for k in MODEL_DEFAULTS if k!='width' or architecture['model']=='xunet_wide']
    for key in needed:
        if key in own and parse(key,own[key])!=architecture[key]:
            raise ValueError(f'evaluated run disagrees with checkpoint model setting {key!r}')
        if borrowed.values[key]!=architecture[key]:
            raise ValueError(f'borrowed settings disagree with checkpoint model setting {key!r}: '
                             f'{borrowed.values[key]!r} != {architecture[key]!r}')
    experiment={k:v for k,v in plain(borrowed.values).items() if k in SCHEMA and not SCHEMA[k].machine}
    experiment.update(architecture)
    experiment.update({k:own[k] for k in MODEL_TRAINING_KEYS if k in own})
    # An old derived run name describes its original data recipe, not this evaluation.
    # Keep an explicit target identity when available; never borrow the donor identity.
    experiment.update(name=own.get('name'),run_name=own.get('run_name') if own.get('name') else None)
    experiment['train_yaml']=portable_config(experiment.get('train_yaml'))
    settings=load(environ={},_profile=automatic_profile(),
        overrides=dict(quick=borrowed.quick,job_id=own.get('job_id'),**evaluation_override_values(eval_overrides)),**experiment)
    original=source/'run_settings.json'
    if original.exists():
        snapshot=original.read_bytes();kind='saved'
        inputs={'run_settings.json':hashlib.sha256(snapshot).hexdigest()}
    else:
        # This is an OUTPUT snapshot of resolved legacy settings, never a fabricated input.
        snapshot=(json.dumps(dict(values=plain(borrowed.values),sources=plain(borrowed.provenance),
            profile=borrowed.profile,overrides=plain(borrowed.overrides)),indent=2)+'\n').encode()
        kind='resolved-legacy'
        inputs={name:hashlib.sha256((source/name).read_bytes()).hexdigest()
                for name in ('model_run_settings.json','data_run_settings.json')}
    provenance=dict(borrowed_settings_from=str(Path(settings_from).resolve()),
        borrowed_checkpoint_folder=str(source),borrowed_run_settings_sha256=hashlib.sha256(snapshot).hexdigest(),
        borrowed_run_settings_kind=kind,borrowed_source_files_sha256=inputs,
        checkpoint_model_settings=architecture)
    return settings,{**provenance,**context},snapshot


def job_folder(folder):
    """The outer job folder (parent of checkpoints/) when the layout has one."""
    folder=Path(folder)
    return folder.parent.parent if folder.parent.name=='checkpoints' else folder


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run_folder',type=Path)
    parser.add_argument('--test-suite',choices=('standard-256-v1',),default='standard-256-v1',
        help='frozen 256 suite; legacy-512 is retired')
    parser.add_argument('--only',choices=('main',),help='compatibility alias for the default main stage')
    parser.add_argument('--higher-k',action='store_true',help='add matched-tile k 5-8 panels')
    parser.add_argument('--hard',action='store_true',help='include hard256 (already enabled by default)')
    parser.add_argument('--no-hard',action='store_true',help='do not run hard256')
    parser.add_argument('--overwrite',action='store_true',help='allow overwriting evaluation output')
    parser.add_argument('--settings-from',type=Path,help='borrow data settings, retain the checkpoint model')
    parser.add_argument('--output',type=Path,help='folder under runs/<run>/evaluation/standard-256-v1/')
    parser.add_argument('--set',dest='eval_overrides',action='append',default=[],metavar='KEY=VALUE',
        help='evaluation override: rocm_conv1_fix, eval_batch_size (or batch_size), eval_workers only')
    args=parser.parse_args()  # Retired options/selectors fail before checkpoint, GPU or file work.
    evaluation_override_values(args.eval_overrides)
    override_options={'eval_overrides':tuple(args.eval_overrides)} if args.eval_overrides else {}
    folder=checkpoint_folder(args.run_folder)
    from cvdl import standardtest, runtime
    from cvdl.lit import select_evaluation_checkpoint as select_best_checkpoint
    from scripts.pc_safety import check_workers, own_process_group
    checkpoint=Path(select_best_checkpoint('',folder))
    settings,borrowing,_=settings_for_evaluation(folder,checkpoint,args.settings_from,**override_options)
    check_workers(settings.eval_workers)
    own_process_group()
    runtime.configure(settings)
    standardtest.run(settings,checkpoint,job_folder(folder),higher_k=args.higher_k,hard=not args.no_hard,
        output=args.output,overwrite=args.overwrite,provenance=borrowing)


if __name__=='__main__':main()
