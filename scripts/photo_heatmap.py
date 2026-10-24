#!/usr/bin/env python3
"""Whole-photo cell-head/CAM demo on frozen TEST photos; no training.

Optional positional input: a frozen photo ID or its master_path. Omit it for the
seeded six-photo selection. --select-only records the list before inference.
"""
import argparse
import csv
import json
from pathlib import Path
import sys
import threading

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np

SEED=20261021
RUN='srm-cell-aux-20261021T054704Z-6997'
EID='ea32a4be0e2ce3597fbd'
VERSIONS=(('clean',0,'clean'),('k1-contiguous',1,'contiguous'),('k2-contiguous',2,'contiguous'),('k1-random',1,'random'))


from cvdl.photo_inference import cell_edges, cell_to_pixels, reduce_cells, stitch_cells, infer_photo


def photo_selection(photo=None):
    rng=np.random.default_rng(SEED);chosen=[]
    for source,split,count in (('raise','test',2),('div2k','test',1),('alaska','test_unseen',2),('lsdir','test',1)):
        with (ROOT/f'data/{source}/photo_manifest.csv').open() as stream:
            candidates=sorted((r for r in csv.DictReader(stream) if r['split']==split),key=lambda r:r['photo_id'])
        indices=rng.choice(len(candidates),count,replace=False)
        if photo:
            chosen.extend(r for r in candidates if photo in (r['photo_id'],r['master_path'],str(ROOT/r['master_path'])))
        else:chosen.extend(candidates[i] for i in indices)
    if photo and len(chosen)!=1:raise ValueError('input must identify one photo in the frozen TEST splits')
    return chosen


def selection_record(records):
    photos=[]
    for r in records:
        versions=[]
        for name,k,placement in VERSIONS:
            seed=[SEED,int(r['photo_index']),2 if placement=='random' else 1]
            versions.append(dict(name=name,seed=seed if k else None,
                settings=dict(bit_mode='plane',k=k,method='replace',placement=placement,bpp=.10,payload='random_bytes') if k else None))
        photos.append(dict(record=r,versions=versions))
    return dict(seed=SEED,photos=photos)


def plot_rows(rows,output,limits,cutoffs):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    fig,axes=plt.subplots(len(rows),4,figsize=(16,3.6*len(rows)),squeeze=False)
    for axrow,row in zip(axes,rows):
        h,w=row['shape'];ys=cell_edges(h);xs=cell_edges(w)
        axrow[0].imshow(row['thumbnail'],extent=(0,w,h,0))
        for y,x in row['origins']:
            axrow[0].plot([x,x+256,x+256,x,x],[y,y,y+256,y+256,y],color='white',lw=.25,alpha=.6)
        axrow[0].set_title(f"{row['photo_id']} / {row['name']}\nany tile: {'message' if row['decision'] else 'clean'}; max score {row['max_score']:.3f}")
        axrow[1].pcolormesh(xs,ys,row['truth'],cmap='gray',vmin=0,vmax=1,shading='flat',rasterized=True)
        axrow[1].set_title('Actually changed cells')
        for axis,values,key,title in ((axrow[2],row['maps'][0],'aux','Trained cell head'),(axrow[3],row['maps'][1],'cam','Tile-head CAM margin')):
            im=axis.pcolormesh(xs,ys,values,cmap='coolwarm',vmin=limits[key][0],vmax=limits[key][1],shading='flat',rasterized=True)
            segments=[]
            for r,c in np.argwhere(values>=cutoffs[key]):
                segments.append([(xs[c],ys[r]),(xs[c+1],ys[r]),(xs[c+1],ys[r+1]),(xs[c],ys[r+1]),(xs[c],ys[r])])
            axis.add_collection(LineCollection(segments,colors='black',linewidths=.35))
            axis.set_title(title);fig.colorbar(im,ax=axis,fraction=.035,pad=.02)
        for axis in axrow:
            axis.set_xlim(0,w);axis.set_ylim(h,0);axis.set_aspect('equal');axis.set_xticks([]);axis.set_yticks([])
    fig.suptitle('256px overlap-last tiles; 32px photo cells; overlaps averaged at pixels\nBlack outlines: above frozen cell cut-off. Cells summarise 48×48px patches, not exact causal pixels.',fontsize=12)
    fig.tight_layout(rect=(0,0,1,.97))
    try:fig.savefig(output,dpi=130)
    finally:plt.close(fig)


def main():
    from scripts.pc_safety import own_process_group
    own_process_group()
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('photo',nargs='?')
    parser.add_argument('--run',default=RUN)
    parser.add_argument('--evaluation-id',default=EID)
    parser.add_argument('--cell-summary',type=Path,default=ROOT/'runs/verification/cell-run13-2026-10-21/cells/summary.json')
    parser.add_argument('--output',type=Path,default=ROOT/'runs/verification/photo-heatmaps-run13-2026-10-21')
    parser.add_argument('--select-only',action='store_true')
    parser.add_argument('--device',choices=('cpu','cuda'),default='cuda')
    parser.add_argument('--batch-size',type=int,default=8)
    args=parser.parse_args()
    if args.batch_size<1:parser.error('batch size must be positive')
    selection=selection_record(photo_selection(args.photo));args.output.mkdir(parents=True,exist_ok=True)
    list_path=args.output/'photo-list.json'
    if list_path.exists() and json.loads(list_path.read_text())!=selection:raise ValueError('existing frozen photo list differs')
    if list(args.output.glob('*.png')) or any(p!=list_path for p in args.output.glob('photo-*.json')):
        raise ValueError('refusing to overwrite prior photo results')
    list_path.write_text(json.dumps(selection,indent=2)+'\n')
    if args.select_only:
        print(json.dumps([(p['record']['source'],p['record']['photo_id'],p['record']['split']) for p in selection['photos']]))
        return
    from scripts.pc_safety import check_workers,available_gb,kill_group
    check_workers(0)
    stop=threading.Event()
    def watch():
        while not stop.wait(.5):
            if available_gb()<12:print('Host RAM safety stop',flush=True);kill_group()
    thread=threading.Thread(target=watch,daemon=True);thread.start()
    import torch
    torch.set_num_threads(1)
    from scripts.evaluate_run import checkpoint_folder
    from cvdl.lit import select_evaluation_checkpoint
    from cvdl.standardtest import load_network
    from lsb_source import load_photo
    from lsb_engine import embed_fast
    run=ROOT/'runs'/args.run
    checkpoint=select_evaluation_checkpoint('',checkpoint_folder(run))
    calibration=json.loads((run/'evaluation/standard-256-v1'/args.evaluation_id/'validation/calibration.json').read_text())
    if not calibration['deployment'] or calibration['quick']:raise ValueError('requires full saved deployment calibration')
    tile_cutoff=calibration['cutoff']
    cell_summary=json.loads(args.cell_summary.read_text())
    cutoffs={k:cell_summary['maps'][k]['cutoff'] for k in ('aux','cam')};limits=cell_summary['color_limits']
    network=load_network(checkpoint,args.device)
    if not hasattr(network,'cell_head'):raise ValueError('checkpoint has no trained cell head')
    overview=[]
    from PIL import Image
    for selected in selection['photos']:
        record=selected['record'];clean=load_photo(record,ROOT);shape=clean.shape[:2];rows=[];records=[]
        for version in selected['versions']:
            name=version['name']
            if name=='clean':photo=clean;changed=None;metadata=None
            else:
                kwargs={**version['settings'],'rng':np.random.default_rng(np.random.SeedSequence(version['seed']))}
                photo,metadata,changed=embed_fast(clean,**kwargs,return_changed_units=True)
            mask=np.zeros(shape,bool)
            if changed is not None:mask.ravel()[changed//3]=True
            truth=reduce_cells(mask)>0
            origins,scores,maps=infer_photo(network,photo,record,args.device,args.batch_size)
            decision=bool(max(scores)>=tile_cutoff);map_counts={}
            for i,key in enumerate(('aux','cam')):
                predicted=maps[i]>=cutoffs[key];true_positive=int((predicted&truth).sum())
                map_counts[key]=dict(cells_above_cutoff=int(predicted.sum()),true_positive_cells=true_positive,
                    changed_cell_recall=true_positive/int(truth.sum()) if truth.any() else None,
                    false_cells=int((predicted&~truth).sum()),unchanged_cells=int((~truth).sum()))
            records.append(dict(version=name,embedding=metadata,photo_decision=decision,max_tile_score=max(scores),
                tile_scores=[dict(y=int(y),x=int(x),score=s) for (y,x),s in zip(origins,scores)],
                changed_cells=int(truth.sum()),maps=map_counts))
            thumbnail=Image.fromarray(photo);thumbnail.thumbnail((1024,1024))
            row=dict(photo_id=record['photo_id'],name=name,shape=shape,thumbnail=np.asarray(thumbnail).copy(),truth=truth,maps=maps,
                origins=origins,decision=decision,max_score=max(scores))
            rows.append(row)
            if name in ('clean','k1-contiguous'):overview.append(row)
        output=args.output/f"photo-{record['photo_id']}"
        plot_rows(rows,output.with_suffix('.png'),limits,cutoffs)
        data=dict(photo_id=record['photo_id'],source=record['source'],split=record['split'],shape=shape,cell_shape=list(rows[0]['truth'].shape),
            tile_cutoff=tile_cutoff,cell_cutoffs=cutoffs,color_limits=limits,
            photo_rule='any tile >= saved deployment cutoff; not a calibrated photo false-alarm guarantee',
            false_cell_definition='predicted photo cell with no actually changed pixel in its 32px bin; demo only, 48px receptive fields can include neighboring changes',versions=records)
        output.with_suffix('.json').write_text(json.dumps(data,indent=2,allow_nan=False)+'\n')
        print(json.dumps(dict(photo_id=record['photo_id'],tiles=len(origins),versions=len(records))),flush=True)
        del rows,clean,photo,mask,changed
    plot_rows(overview,args.output/'overview.png',limits,cutoffs)
    stop.set();thread.join()


if __name__=='__main__':
    from scripts.pc_safety import guarded
    guarded(main)
