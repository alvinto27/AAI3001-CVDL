"""Offline wide-R2 validation and frozen best-checkpoint mapping builder."""
import argparse
import csv
from fractions import Fraction
import json
from pathlib import Path
import sys
from types import SimpleNamespace

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(ROOT/'LSB-Dataset-Generator'))
WIDE='xunet_v4_clip3_nonorm_lr0.0003_warm1_drops5x0.3-20x0.1_k1to4_e30_wide-w4_r2-20261015T121753Z-29499'
MIX='xunet_v4_clip3_nonorm_lr0.0003_warm1_drops5x0.3-20x0.1_k1to4_e30_wide-w4_mixr2-20261016T024859Z-9032'
RUNS=(WIDE,MIX,'tiling-overlap-20261016T135926Z-8501','tiling-trim-20261016T144117Z-3222',
 'tiling-pad-20261017T013055Z-7210','tiling-overlap-256-20261017T022212Z-9778',
 'tiling-trim-256-20261017T032025Z-8579','tiling-pad-256-20261017T043145Z-9721',
 'srm-20261017T052857Z-4521','srm-e50-20261017T111310Z-0214',
 'srm-e50-paired-20261023T022339Z-4518',
 'srm-e50-curriculum50-paired-20261023T090435Z-3325',
 'srm-e50-paired-coloursep36-20261024T115642Z-1325')


def build(root=ROOT):
    from cvdl import validation as V
    from cvdl.lit import select_best_checkpoint
    from lsb_source import grid_axes
    root=Path(root)
    wide_folders=list((root/'runs'/WIDE/'checkpoints').iterdir())
    if len(wide_folders)!=1:raise ValueError('ambiguous wide training evidence folder')
    folder=wide_folders[0]
    record_path=folder/'data_run_settings.json'; raw=record_path.read_bytes();record=json.loads(raw)
    old=record['settings']
    if old['sources']!=['raise','div2k'] or old['grid']!='overlap_last' or old['tile_size']!=512 or old['source_weights'] is not None or old['variants_per_tile']!=1 or old.get('source_mix'):
        raise ValueError('unsupported wide training record; cannot derive source shares')
    counts={}; evidence={}
    for source in old['sources']:
        groups=[g for g in record['train_photo_selection'] if g['source']==source]
        listed=[i for g in groups for i in g['photo_ids']]
        ids=set(listed)
        if not ids or len(ids)!=len(listed) or any(g['selected']!=len(g['photo_ids']) for g in groups):
            raise ValueError('missing/duplicate saved selected training photos')
        path=root/'data'/source/'photo_manifest.csv'
        with path.open() as f:photos=[r for r in csv.DictReader(f) if r['photo_id'] in ids]
        if len(photos)!=len(ids) or any(r['split']!='train' for r in photos):raise ValueError('training evidence mismatch')
        counts[source]=sum(len(grid_axes(r,'overlap_last',512)[0])*len(grid_axes(r,'overlap_last',512)[1]) for r in photos)
        evidence[source]=dict(selected_photos=len(ids),selected_ids_sha256=V.digest(V.canonical(sorted(ids)).encode()),
                              training_pairs=counts[source],manifest_sha256=V.digest(path.read_bytes()))
    settings=SimpleNamespace(sources=tuple(counts),source_mix={s:Fraction(n) for s,n in counts.items()},
        tiles_per_epoch=180000,tile_size=512,variants_per_tile=1,settings_yaml=root/'configs/train-k1to4.yaml',manifest_root=None)
    rid,files,summary=V.build(root,settings,'overlap_last',256)
    files['training-evidence.json']=V.canonical(dict(run=WIDE,record=str(record_path.relative_to(root)),
        record_sha256=V.digest(raw),selection='saved selected training photos; one pair per recorded overlap512 tile',
        training_pairs=sum(counts.values()),sources=evidence,
        calibration_pairs=18000,calibration_budget='user-approved fixed size; NOT 10% of original training epoch'))+'\n'
    outputs={root/'data/val'/rid/name:text for name,text in files.items()}
    entries=[]
    for run in RUNS:
        folders=list((root/'runs'/run/'checkpoints').iterdir())
        if len(folders)!=1:raise ValueError(f'ambiguous checkpoint folder: {run}')
        checkpoint=Path(select_best_checkpoint(None,folders[0]))
        list_id=rid if run==WIDE else '60da4391043e8ce48ff2'
        pinned=summary if run==WIDE else json.loads((root/'data/val'/list_id/'summary.json').read_text())
        entries.append(dict(run_name=run,checkpoint_file=checkpoint.name,checkpoint_sha256=V.digest(checkpoint.read_bytes()),
            recipe_id=list_id,recipe_sha256=pinned['recipe_sha256'],pairs_sha256=pinned['pairs_sha256'],
            threshold_origin='post_hoc_256'))
    if len({r['checkpoint_sha256'] for r in entries})!=len(RUNS):raise ValueError('duplicate checkpoint mapping')
    mapping=V.canonical(dict(version='evaluation-calibration-v1',entries=entries))+'\n'
    outputs[root/'configs/evaluation-calibration-v1.json']=mapping
    return outputs,dict(recipe_id=rid,summary=summary,training_counts=counts,mapping_sha256=V.digest(mapping.encode()),entries=entries)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--check',action='store_true');args=parser.parse_args()
    outputs,report=build()
    for path,text in outputs.items():
        if args.check or path.exists():
            if not path.is_file() or path.read_bytes()!=text.encode():raise ValueError(f'frozen output differs: {path}')
    if not args.check:
        for path,text in outputs.items():
            if not path.exists():path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(text.encode())
    print('REBUILD IDENTICAL' if args.check else 'BUILT');print(json.dumps(report,indent=2))

if __name__=='__main__':main()
