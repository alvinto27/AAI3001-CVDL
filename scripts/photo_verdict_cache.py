#!/usr/bin/env python3
"""Freeze photo IDs first, then resume a bounded whole-photo feature cache."""
import argparse
import csv
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import random
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import cvdl  # registers the generator import path
import numpy as np
from scripts.photo_heatmap import photo_selection, RUN, EID
from cvdl.photo_inference import infer_photo
from cvdl.photo_verdict import features

OUTPUT = ROOT/'runs/verification/photo-verdict-run13-2026-10-21'
TILE_CUTOFF = .5116998553
HEAD_CUTOFF = .3110628426
PANELS = (('raise', 'raise', 'test', 96), ('div2k', 'div2k', 'test', 96),
          ('alaska-seen', 'alaska', 'test_seen', 128),
          ('alaska-unseen', 'alaska', 'test_unseen', 128), ('lsdir', 'lsdir', 'test', 128))


def dump(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    temporary.replace(path)


def seed_key(record, partition, variant, stream):
    from lsb_settings import otf_seed_sequence
    return otf_seed_sequence(run_seed=1, purpose='test' if partition == 'test' else 'val',
        epoch=0, photo_index=int(record['photo_index']), grid_policy='none', tile_size=0,
        tile_row=0, tile_col=0, variant=variant, stream=stream, carrier='photo').entropy


def selection(*,tile_cutoff=TILE_CUTOFF,head_cutoff=HEAD_CUTOFF):
    from lsb_train_grid import axes
    excluded = sorted(r['photo_id'] for r in photo_selection())
    manifests = {}
    for source in ('raise', 'div2k', 'alaska', 'lsdir'):
        path = ROOT/f'data/{source}/photo_manifest.csv'
        with path.open() as stream:
            manifests[source] = list(csv.DictReader(stream))
    photos = []
    groups = [('raise', 'raise', 'val', 96), ('div2k', 'div2k', 'val', 96),
              ('alaska-seen', 'alaska', 'val', 128), ('lsdir', 'lsdir', 'val', 128)] + list(PANELS)
    for group, (panel, source, split, cap) in enumerate(groups):
        candidates = sorted((r for r in manifests[source] if r['split'] == split and
                             (split == 'val' or r['photo_id'] not in excluded)), key=lambda r:r['photo_id'])
        random.Random(20261021+group).shuffle(candidates)
        chosen = candidates[:cap]
        if len(chosen) != cap:
            raise ValueError('insufficient photo pool')
        if split == 'val':
            random.Random(20261121+group).shuffle(chosen)
        for i, record in enumerate(chosen):
            partition = ('fit' if i < cap//2 else 'calibration') if split == 'val' else 'test'
            ys, xs = axes(record, 256, 'overlap_last')
            versions = [dict(variant=0, k=0, placement='clean')]
            for variant, (k, placement) in enumerate(((k,p) for k in range(1,5)
                    for p in ('contiguous','random','stride')), 1):
                versions.append(dict(variant=variant, k=k, placement=placement,
                    settings_seed=seed_key(record,partition,variant,'settings'),
                    embedding_seed=seed_key(record,partition,variant,'embedding')))
            photos.append(dict(panel=panel, partition=partition, group=group,
                tiles=len(ys)*len(xs), record=record, versions=versions))
    val_ids={p['record']['photo_id'] for p in photos if p['partition']!='test'}
    test_ids={p['record']['photo_id'] for p in photos if p['partition']=='test'}
    if val_ids & test_ids:
        raise ValueError('validation/TEST overlap')
    return dict(selection_seed=20261021, split_seed=20261121, group_numbering='zero-based; four val sources then five TEST panels',
        excluded_demos=excluded, tile_cutoff=tile_cutoff, head_cutoff=head_cutoff,
        manifest_hashes={s:hashlib.sha256((ROOT/f'data/{s}/photo_manifest.csv').read_bytes()).hexdigest() for s in manifests},
        config_sha256=hashlib.sha256((ROOT/'configs/train-k1to4.yaml').read_bytes()).hexdigest(), photos=photos)


def collection_inputs(run,evaluation_id,cell_summary,map_kind,checkpoint=None):
    """Resolve model-specific validation thresholds before collecting any scores."""
    from cvdl.pipeline.bundle import file_hash
    run=Path(run)
    calibration=json.loads((run/'evaluation/standard-256-v1'/evaluation_id/'validation/calibration.json').read_text())
    if calibration['quick'] or not calibration['deployment'] or calibration['geometry']!={'grid':'overlap_last','tile_size':256}:
        raise ValueError('full 256 overlap-last deployment calibration required')
    if checkpoint is None:
        matches=[p for p in sorted(run.rglob('*.ckpt')) if file_hash(p)==calibration['checkpoint_sha256']]
        if not matches:raise ValueError('no evaluated checkpoint found')
        checkpoint=matches[0]
    if file_hash(checkpoint)!=calibration['checkpoint_sha256']:raise ValueError('checkpoint/calibration mismatch')
    cells=json.loads(Path(cell_summary).read_text())
    tile=calibration['cutoff'];head=cells['maps']['aux' if map_kind=='trained_head' else 'cam']['cutoff']
    if run.name==RUN and evaluation_id==EID and map_kind=='trained_head':
        if round(tile,10)!=TILE_CUTOFF or round(head,10)!=HEAD_CUTOFF:raise ValueError('accepted run-13 cutoffs changed')
        tile,head=TILE_CUTOFF,HEAD_CUTOFF
    return Path(checkpoint),tile,head


def collect(output, frozen,*,checkpoint=None,evaluation_id=EID,map_kind='trained_head',device='cuda',validation_only=False):
    from scripts.pc_safety import own_process_group, available_gb, kill_group, check_workers
    own_process_group()
    check_workers(0)
    started=time.monotonic()
    stop=threading.Event()
    def watch():
        while not stop.wait(.5):
            if available_gb()<12 or time.monotonic()-started>=180*60:
                print('STOP: RAM floor or 180-minute budget reached; resume cache after review',flush=True)
                kill_group()
    thread=threading.Thread(target=watch,daemon=True)
    thread.start()
    import torch
    torch.set_num_threads(1)
    from cvdl.standardtest import load_network
    from lsb_source import load_photo
    from lsb_settings import draw_otf_settings
    from lsb_batch import read_config
    from lsb_engine import embed_fast
    checkpoint=Path(checkpoint) if checkpoint else ROOT/'runs'/RUN/'checkpoints'/RUN/'last.ckpt'
    identity=dict(selection_sha256=hashlib.sha256((output/'photo-list.json').read_bytes()).hexdigest(),
        checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(), batch_size=8,workers=0,
        tile_cutoff=frozen['tile_cutoff'],head_cutoff=frozen['head_cutoff'],
        evaluation_id=evaluation_id,map_kind=map_kind)
    identity_path=output/'cache-identity.json'
    if identity_path.exists():
        saved_identity=json.loads(identity_path.read_text())
        legacy={k:v for k,v in identity.items() if k not in ('map_kind','evaluation_id')}
        legacy_ok=(saved_identity==legacy and evaluation_id==EID and map_kind=='trained_head' and
            identity['checkpoint_sha256']=='4dc9ab8e8ce6ed98e36da71625b3eff386999a0ac57b67b3cfbfb6bb87dd78b9')
        if saved_identity!=identity and not legacy_ok:raise ValueError('cache identity mismatch')
        identity=saved_identity  # Never rewrite an existing immutable cache identity.
    else:dump(identity_path,identity)
    network=load_network(checkpoint,device)
    if map_kind=='trained_head' and not hasattr(network,'cell_head'):raise ValueError('headless model requires CAM and its own rule fit')
    config=read_config(ROOT/'configs/train-k1to4.yaml')
    for number, selected in enumerate(frozen['photos'],1):
        record=selected['record'];partition=selected['partition']
        if validation_only and partition=='test':continue
        path=output/'features'/partition/(record['photo_id']+'.json')
        path.parent.mkdir(parents=True,exist_ok=True)
        if path.exists():
            # Atomic photo writes plus the shared identity permit a blind TEST
            # resume: never open TEST scores before the CPU rule freeze.
            if partition=='test':continue
            cached=json.loads(path.read_text())
            if len(cached['rows'])!=13 or cached['selection_sha256']!=identity['selection_sha256']:
                raise ValueError('incomplete or mismatched photo cache')
            continue
        clean=load_photo(record,ROOT)
        rows=[]
        for version in selected['versions']:
            variant=version['variant'];metadata=None;settings=None
            if not variant:
                photo=clean
            else:
                cfg=replace(config,k=version['k'],placement=version['placement'],
                    stride=config.stride if version['placement']=='stride' else 1)
                kwargs,metadata=draw_otf_settings(cfg,clean,np.random.default_rng(np.random.SeedSequence(version['settings_seed'])))
                settings={k:v for k,v in kwargs.items() if k!='rng'}
                kwargs['rng']=np.random.default_rng(np.random.SeedSequence(version['embedding_seed']))
                photo,_,_=embed_fast(clean,**kwargs)
            origins,scores,maps=infer_photo(network,photo,record,device,8,map_kinds=(map_kind,))
            values,counts=features(scores,maps[0],frozen['tile_cutoff'],frozen['head_cutoff'])
            rows.append(dict(photo_id=record['photo_id'],panel=selected['panel'],partition=partition,
                camera=record['camera_model'],height=int(clean.shape[0]),width=int(clean.shape[1]),
                variant=variant,label=int(variant>0),k=version['k'],placement=version['placement'],
                embedding=metadata,settings=settings,features=values,counts=counts))
            del photo,maps,scores,origins
        dump(path,dict(selection_sha256=identity['selection_sha256'],rows=rows))
        del clean,rows
        # Do not print TEST features or scores while collecting.
        print(json.dumps(dict(completed=number,total=len(frozen['photos']),partition=partition,
            photo_id=record['photo_id'],elapsed_minutes=round((time.monotonic()-started)/60,2))),flush=True)
    stop.set();thread.join()
    print('CACHE COMPLETE',flush=True)


def main():
    from scripts.pc_safety import own_process_group
    own_process_group()
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=OUTPUT)
    parser.add_argument('--select-only',action='store_true')
    parser.add_argument('--run',type=Path,default=ROOT/'runs'/RUN)
    parser.add_argument('--evaluation-id',default=EID)
    parser.add_argument('--cell-summary',type=Path,default=ROOT/'runs/verification/cell-run13-2026-10-21/cells/summary.json')
    parser.add_argument('--map-kind',choices=('trained_head','cam'),default='trained_head')
    parser.add_argument('--checkpoint',type=Path)
    parser.add_argument('--device',choices=('cpu','cuda'),default='cuda')
    parser.add_argument('--validation-only',action='store_true')
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    path=args.output/'photo-list.json'
    checkpoint,tile,head=collection_inputs(args.run,args.evaluation_id,args.cell_summary,args.map_kind,args.checkpoint)
    frozen=selection(tile_cutoff=tile,head_cutoff=head)
    if path.exists():
        if json.loads(path.read_text())!=frozen:raise ValueError('frozen selection differs')
    else:dump(path,frozen)
    print('photo-list SHA256 '+hashlib.sha256(path.read_bytes()).hexdigest(),flush=True)
    for partition in ('fit','calibration','test'):
        photos=[p for p in frozen['photos'] if p['partition']==partition]
        print(partition,len(photos),sum(p['tiles'] for p in photos),flush=True)
    if not args.select_only:collect(args.output,frozen,checkpoint=checkpoint,evaluation_id=args.evaluation_id,
        map_kind=args.map_kind,device=args.device,validation_only=args.validation_only)


if __name__=='__main__':
    from scripts.pc_safety import guarded
    guarded(main)
