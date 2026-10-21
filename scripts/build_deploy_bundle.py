#!/usr/bin/env python3
"""Export a trusted evaluated checkpoint + its own fitted photo rule, CPU only."""
import argparse
import copy
import json
from pathlib import Path
import sys
import tempfile

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import torch
from cvdl.models import model_from_hparams
from cvdl.pipeline.bundle import contract_hash, file_hash, load_bundle, validate_config

LEGACY_RUN13='4dc9ab8e8ce6ed98e36da71625b3eff386999a0ac57b67b3cfbfb6bb87dd78b9'


def _agrees(source,fitted):
    # The accepted run-13 fit intentionally froze ten-decimal cutoffs.
    return source==fitted or round(source,10)==fitted


def build_bundle(run,evaluation_id,photo_rule,cells_summary,output,*,map_kind='trained_head',name=None,version='1'):
    run=Path(run);output=Path(output)
    if output.exists():raise ValueError('refusing to overwrite an existing bundle')
    evaluation=run/'evaluation/standard-256-v1'/evaluation_id
    calibration=json.loads((evaluation/'validation/calibration.json').read_text())
    evaluated=json.loads((evaluation/'evaluation_settings.json').read_text())
    fit=json.loads(Path(photo_rule).read_text())
    cells=json.loads(Path(cells_summary).read_text())
    if not calibration['deployment'] or calibration['quick'] or evaluated['quick']:
        raise ValueError('requires full deployment validation')
    checkpoint_sha=calibration['checkpoint_sha256']
    if evaluated['checkpoint_sha256']!=checkpoint_sha or evaluated['evaluation_id']!=evaluation_id:
        raise ValueError('evaluation/checkpoint identity mismatch')
    identity=fit['cache_identity']
    if identity['checkpoint_sha256']!=checkpoint_sha:
        raise ValueError('photo rule belongs to another checkpoint')
    # Older accepted run-13 artifacts predate explicit map/evaluation fields.
    # Other fits must carry them, so a head rule cannot masquerade as CAM.
    legacy=checkpoint_sha==LEGACY_RUN13 and 'map_kind' not in identity
    if legacy:
        if map_kind!='trained_head':raise ValueError('legacy run-13 rule is trained-head only')
    elif identity.get('map_kind')!=map_kind or identity.get('evaluation_id')!=evaluation_id:
        raise ValueError('photo-rule map/evaluation identity mismatch')
    if map_kind not in ('trained_head','cam'):raise ValueError('unsupported map kind')
    geometry=calibration['geometry']
    if geometry!={'grid':'overlap_last','tile_size':256}:raise ValueError('unsupported evaluated geometry')
    cell_key='aux' if map_kind=='trained_head' else 'cam'
    tile_cutoff=fit['tile_cutoff'];cell_cutoff=fit['head_cutoff']
    if identity['tile_cutoff']!=tile_cutoff or identity['head_cutoff']!=cell_cutoff:
        raise ValueError('photo-rule/cache cutoff mismatch')
    if not _agrees(calibration['cutoff'],tile_cutoff) or not _agrees(cells['maps'][cell_key]['cutoff'],cell_cutoff):
        raise ValueError('source thresholds do not match photo-rule thresholds')
    candidates=[p for p in sorted(run.rglob('*.ckpt')) if file_hash(p)==checkpoint_sha]
    if not candidates:raise ValueError('no checkpoint matches the saved evaluation')
    # This offline builder accepts only the caller's trusted training checkpoint.
    checkpoint=torch.load(candidates[0],map_location='cpu',weights_only=False)
    hparams=checkpoint['hyper_parameters']
    architecture={k:hparams.get(k,default) for k,default in dict(model='xunet',clip_t=3.,norm='group',
        front_end='kv',width=4,cell_supervision='off').items()}
    network=model_from_hparams(architecture)
    state={k.removeprefix('net.'):v.detach().cpu().contiguous() for k,v in checkpoint['state_dict'].items()}
    network.load_state_dict(state,strict=True)
    fitted=fit['rules']['c']
    config=dict(format_version=1,name=name or run.name,version=version,architecture=architecture,
        input='rgb8_0_255',tile=dict(size=256,grid='overlap_last',cutoff=tile_cutoff),
        map=dict(kind=map_kind,cell_size=32,score='logit' if map_kind=='trained_head' else 'class1_minus_class0',cutoff=cell_cutoff),
        photo_rule=dict(type='logistic_v1',features=copy.deepcopy(fitted['features']),
            model={k:copy.deepcopy(fitted['model'][k]) for k in ('mean','scale','coefficients','intercept')},cutoff=fitted['cutoff']),
        source=dict(checkpoint_sha256=checkpoint_sha,evaluation_id=evaluation_id,
            threshold_id=calibration['threshold_id'],photo_fit_sha256=file_hash(photo_rule)),weights_sha256='0'*64)
    validate_config(config)
    output.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.bundle-',dir=output.parent) as directory:
        temporary=Path(directory)
        torch.save(dict(state_dict=state,contract_sha256=contract_hash(config)),temporary/'weights.pt')
        config['weights_sha256']=file_hash(temporary/'weights.pt')
        (temporary/'deploy.json').write_text(json.dumps(config,indent=2,allow_nan=False)+'\n')
        load_bundle(temporary,device='cpu')
        temporary.rename(output)
    return config


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--evaluation-id',required=True)
    parser.add_argument('--photo-rule',type=Path,required=True)
    parser.add_argument('--cells-summary',type=Path,required=True)
    parser.add_argument('--map-kind',choices=('trained_head','cam'),default='trained_head')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--name')
    parser.add_argument('--version',default='1')
    args=parser.parse_args(argv)
    torch.set_num_threads(1)
    config=build_bundle(args.run,args.evaluation_id,args.photo_rule,args.cells_summary,args.output,
        map_kind=args.map_kind,name=args.name,version=args.version)
    print(json.dumps(dict(bundle=str(args.output),weights_sha256=config['weights_sha256'])))


if __name__=='__main__':main()
