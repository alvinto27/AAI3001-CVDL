"""Build (or check) the frozen hard test v1 pair lists under data/hard-test/v1/.

CPU only, light (reads saved prediction CSVs and the saved clean-tile noise file; no photo decode, no model).
  python scripts/build_hard_test.py --run RUN --noise CSV            # write lists + summary
  python scripts/build_hard_test.py --run RUN --noise CSV --check    # rebuild in memory, compare to the files
Inputs identify the standard clean test tiles (the frozen panels' clean rows) and the clean-tile KV noise.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from cvdl import hardtest as H

OUT=ROOT/'data/hard-test/v1'


def checkpoint_dir(run):
    found=sorted({p.parent for p in Path(run).resolve().rglob('best-*.ckpt')})
    if len(found)!=1:raise ValueError('run folder must contain exactly one checkpoint directory')
    return found[0]


def load_inputs(run, noise_path):
    folder=checkpoint_dir(run)
    noise=pd.read_csv(noise_path)
    if noise.cover_id.duplicated().any():raise ValueError('duplicate cover ids in the noise file')
    noise_by=noise.set_index('cover_id').kv_std
    panels={}
    for panel,sub in H.PANEL_FILES.items():
        frame=pd.read_csv(folder/sub/'test_predictions.csv')
        clean=frame[frame.label==0].drop_duplicates('cover_id')
        keep=['cover_id','source','source_photo','camera_model','split','grid','tile_row','tile_col','y','x','photo_index']
        panels[panel]=clean[keep].reset_index(drop=True)
    validation=pd.concat([pd.read_csv(folder/'extra-standard-val'/'test_predictions.csv'),
                          pd.read_csv(folder/'extra-alaska-val'/'test_predictions.csv'),
                          pd.read_csv(folder/'extra-lsdir-val'/'test_predictions.csv')],ignore_index=True)
    validation=validation[validation.label==0][['cover_id','source']].drop_duplicates('cover_id')
    return panels,validation,noise_by


def photo_splits():
    result={}
    for source in ('raise','div2k','alaska','lsdir'):
        with (ROOT/f'data/{source}/photo_manifest.csv').open() as stream:
            for row in csv.DictReader(stream):result[row['photo_id']]=row['split']
    return result


def assert_test_only(panels, splits):
    """Zero train/val overlap: every photo of every panel must be a test-split photo, and no photo is in two splits."""
    for panel,frame in panels.items():
        got={splits[p] for p in frame.source_photo.unique()}
        if got-set(H.PANEL_SPLITS[panel]):raise ValueError(f'{panel} uses non-test photos: {got}')
    owners={}
    for panel,frame in panels.items():
        for p in frame.source_photo.unique():owners.setdefault(p,set()).add(panel)
    bad=[p for p,v in owners.items() if len(v)>1]
    if bad:raise ValueError(f'photos in more than one hard-test panel: {bad[:3]}')
    return True


def build(run, noise_path):
    panels,validation,noise_by=load_inputs(run,noise_path)
    assert_test_only(panels,photo_splits())
    edges=H.validation_edges(noise_by,validation)
    lists={};cells=[]
    for panel,k,condition in H.cell_names():
        clean=panels[panel]
        rows=H.build_pairs(clean,noise_by,edges,panel,k,condition)
        pool=H.tile_pool(clean,noise_by,edges,condition)
        H.check_cell(rows,len(pool))
        lists[(panel,k,condition)]=rows
        frame=pd.DataFrame(rows)
        cells.append(dict(cell=f'{panel}/k{k}/{condition}',panel=panel,k=k,condition=condition,pairs=len(frame),
            pool_tiles=len(pool),pool_photos=int(pool.source_photo.nunique()),distinct_tiles=int(frame.cover_id.nunique()),
            distinct_photos=int(frame.source_photo.nunique()),max_tile_use=int(frame.tile_use.max()),
            by_source={s:int(n) for s,n in frame.source.value_counts().sort_index().items()}))
    return panels,edges,lists,cells


def write_files(lists, cells, edges, out, run, noise_path):
    out.mkdir(parents=True,exist_ok=True)
    shas={}
    for panel in H.PANELS:
        rows=[r for key,v in lists.items() if key[0]==panel for r in v]
        text=H.pair_list_csv(rows)
        (out/f'pairs-{panel}.csv').write_text(text)
        shas[f'pairs-{panel}.csv']=H.sha256_text(text)
    summary=dict(version=H.VERSION,pairs_per_cell=H.PAIRS_PER_CELL,combinations=len(H.COMBINATIONS),per_combination=H.PER_COMBINATION,
        noise_quantile=H.NOISE_QUANTILE,noise_edges_from_validation_only=edges,files_sha256=shas,
        noise_file=str(Path(noise_path).name),noise_file_sha256=hashlib.sha256(Path(noise_path).read_bytes()).hexdigest(),cells=cells)
    (out/'summary.json').write_text(json.dumps(summary,indent=2,sort_keys=True)+'\n')
    return summary


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True,help='a finished run whose saved panels define the clean test tiles')
    p.add_argument('--noise',type=Path,required=True)
    p.add_argument('--check',action='store_true')
    p.add_argument('--output',type=Path,default=OUT)
    a=p.parse_args()
    _,edges,lists,cells=build(a.run,a.noise)
    if a.check:
        summary=json.loads((a.output/'summary.json').read_text())
        for panel in H.PANELS:
            rows=[r for key,v in lists.items() if key[0]==panel for r in v]
            text=H.pair_list_csv(rows)
            if H.sha256_text(text)!=summary['files_sha256'][f'pairs-{panel}.csv']:raise SystemExit(f'REBUILD MISMATCH: {panel}')
            if (a.output/f'pairs-{panel}.csv').read_text()!=text:raise SystemExit(f'FILE MISMATCH: {panel}')
        print('REBUILD IDENTICAL: 4 pair lists, SHA-256 equal')
        return
    if (a.output/'summary.json').exists():raise SystemExit('refusing to overwrite a frozen hard test: '+str(a.output))
    summary=write_files(lists,cells,edges,a.output,a.run,a.noise)
    print(json.dumps({k:summary[k] for k in ('files_sha256','noise_edges_from_validation_only')},indent=2))


if __name__=='__main__':main()
