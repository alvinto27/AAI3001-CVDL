"""Small fixed cell-target diagnostic selection and label/figure replay.

No training, threshold fitting, bootstrap or benchmark recipe creation here.
"""
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

MANIFEST = 'configs/cell-diagnostic-v1.json'
SEED = 20261021
SOURCES = {
    'val': {p:'data/val/60da4391043e8ce48ff2/pairs.csv' for p in ('raise','alaska','lsdir')},
    'test': {'raise':'data/test/standard-256-v1/pairs-standard.csv',
             **{p:f'data/hard-test/256-v1/pairs-{p}.csv' for p in ('alaska-seen','alaska-unseen','lsdir')}},
}


def select_manifest(root):
    """Seeded pair-ID order; 32 distinct photos per placement and source/split."""
    root=Path(root)
    result={'seed':SEED,'val':{},'test':{}}
    photos={}
    for split,panels in SOURCES.items():
        photos[split]=set()
        for panel,path in panels.items():
            with (root/path).open() as f: rows=list(csv.DictReader(f))
            used=set();result[split][panel]={}
            for placement in ('contiguous','random'):
                candidates=[r for r in rows if int(r['k'])==1 and r['placement']==placement
                    and .05<=float(r['effective_bpp'])<=.15
                    and (split!='val' or r['source']==panel)
                    and (panel!='raise' or r['source']=='raise')
                    and (split!='test' or panel=='raise' or r['condition']=='low')]
                candidates.sort(key=lambda r:hashlib.sha256(
                    f"{SEED}/cell-diag-v1/{split}/{panel}/{placement}/{r['pair_id']}".encode()).digest())
                selected=[]
                for row in candidates:
                    if row['source_photo'] in used:continue
                    selected.append(row['pair_id']);used.add(row['source_photo'])
                    photos[split].add(row['source_photo'])
                    if len(selected)==32:break
                if len(selected)!=32:
                    raise ValueError(f'cell selection cap cannot be met: {split}/{panel}/{placement}: {len(selected)}')
                result[split][panel][placement]=selected
    if photos['val']&photos['test']:raise ValueError('cell diagnostic val/test photos overlap')
    result['figure']=[dict(split='test',panel=panel,placement=placement,
        pair_id=result['test'][panel][placement][0]) for panel,placement in
        (('raise','contiguous'),('alaska-seen','contiguous'),('raise','random'))]
    return result


def selected_rows(root, split, *, manifest=None):
    root=Path(root)
    manifest=json.loads((root/MANIFEST).read_text()) if manifest is None else manifest
    result=[]
    for panel,placements in manifest[split].items():
        with (root/SOURCES[split][panel]).open() as f: rows={r['pair_id']:r for r in csv.DictReader(f)}
        for ids in placements.values():
            result.extend(dict(rows[pair_id],diagnostic_panel=panel) for pair_id in ids)
    return result


def replay_cell_pair(corpus, row):
    """Replay one recorded embedding; reduce its true changed units to targets."""
    from lsb_engine import embed_fast
    from lsb_train_grid import crop
    from lsb_cells import changed_cell_targets
    record=corpus.by_id[row['source_photo']]
    clean,(y,x),pad=crop(corpus.photo(record),record,256,int(row['tile_row']),int(row['tile_col']),'overlap_last')
    if (y,x)!=(int(row['y']),int(row['x'])) or any(pad):
        raise ValueError('cell replay geometry differs from frozen pair')
    kwargs=json.loads(row['draw_kwargs'])
    kwargs['rng']=np.random.default_rng(np.random.SeedSequence(json.loads(row['embedding_seed_key'])))
    stego,_,changed=embed_fast(clean,**kwargs,return_changed_units=True)
    counts,valid=changed_cell_targets(changed)
    return clean.copy(),stego,counts,valid


def cell_maps(network, images):
    """XuNet CAM/aux, or GBRAS class-score cells (not a linear-feature CAM)."""
    import torch
    from torch.nn import functional as F
    with torch.inference_mode():
        if hasattr(network,'class_score_cells'):
            cells,logits=network.class_score_cells(images.float())
            return cells[:,1]-cells[:,0],None,logits.softmax(1)[:,1]
        features=network.features(images.float())
        cam=F.conv2d(features,network.classifier.weight[:,:,None,None],network.classifier.bias)
        aux=network.cell_head(features)[:,0] if hasattr(network,'cell_head') else None
        tile=network.classifier(features.mean((-2,-1))).softmax(1)[:,1]
    return cam[:,1]-cam[:,0],aux,tile


def render_cell_figure(network, corpus, root, output, *, score_limits, manifest=None, control_cam=None, control_limits=None):
    """Required same-tile comparison. Limits must be frozen from validation.

    score_limits = {'cam':(low,high),'aux':(low,high)} for XuNet. GBRAS uses
    'class_score' instead of 'cam', with no aux. No per-image scaling.
    This does not choose cutoffs, evaluate a suite or compute result statistics.
    """
    import torch
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    root=Path(root);output=Path(output)
    manifest=json.loads((root/MANIFEST).read_text()) if manifest is None else manifest
    rows={r['pair_id']:r for r in selected_rows(root,'test',manifest=manifest)}
    has_aux=hasattr(network,'cell_head')
    primary='class_score' if hasattr(network,'class_score_cells') else 'cam'
    if set(score_limits)!=({primary,'aux'} if has_aux else {primary}) or any(len(v)!=2 or not np.isfinite(v).all() or v[0]>=v[1] for v in score_limits.values()):
        raise ValueError('CAM and auxiliary limits must be finite validation-derived ranges')
    if control_cam is not None and (control_limits is None or len(control_limits)!=2 or not np.isfinite(control_limits).all() or control_limits[0]>=control_limits[1]):
        raise ValueError('control CAM requires fixed validation limits')
    columns=4+int(has_aux)+int(control_cam is not None)
    fig,axes=plt.subplots(len(manifest['figure']),columns,figsize=(3*columns,9),squeeze=False)
    records=[];device=next(network.parameters()).device
    network.eval()
    for index,chosen in enumerate(manifest['figure']):
        row=rows[chosen['pair_id']]
        clean,stego,counts,valid=replay_cell_pair(corpus,row)
        images=torch.from_numpy(np.stack([clean,stego])).permute(0,3,1,2).to(device)
        cam,aux,tile=cell_maps(network,images)
        cam=cam.cpu().numpy();aux=aux.cpu().numpy() if aux is not None else None
        axes[index,0].imshow(stego);axes[index,0].set_title(f"{row['diagnostic_panel']} / {row['placement']}\nk=1, bpp={float(row['effective_bpp']):.3f}")
        inset=axes[index,0].inset_axes([.02,.02,.28,.28]);inset.imshow(clean);inset.set_axis_off();inset.set_title('clean',fontsize=8)
        axes[index,1].imshow(np.abs(stego.astype(float)-clean.astype(float)).max(2)*16,vmin=0,vmax=255,cmap='gray')
        axes[index,1].set_title('Actual changes ×16')
        for axis in (axes[index,0],axes[index,1]):
            for edge in range(32,256,32):
                axis.axvline(edge-.5,color='white',linewidth=.3);axis.axhline(edge-.5,color='white',linewidth=.3)
        axes[index,2].imshow(np.minimum(counts/16,1),vmin=0,vmax=1,interpolation='nearest',cmap='viridis')
        axes[index,2].set_title('Cell target / sparse weight')
        for r,c in zip(*np.where((counts==0)&~valid)):
            axes[index,2].add_patch(Rectangle((c-.5,r-.5),1,1,fill=False,hatch='//',edgecolor='gray',linewidth=0))
        maps=[(cam[1],score_limits[primary],getattr(network,'cell_map_label','Tile-head CAM margin'))]
        if has_aux:maps.append((aux[1],score_limits['aux'],'Trained auxiliary logit'))
        if control_cam is not None:maps.append((control_cam[row['pair_id']],control_limits,'Control CAM margin'))
        for axis,(values,limits,title) in zip(axes[index,3:],maps):
            im=axis.imshow(values,interpolation='nearest',cmap='coolwarm',vmin=limits[0],vmax=limits[1])
            axis.set_title(title);fig.colorbar(im,ax=axis,fraction=.046,pad=.04)
        for axis in axes[index]:axis.set_xticks([]);axis.set_yticks([])
        records.append(dict(pair_id=row['pair_id'],source_photo=row['source_photo'],k=int(row['k']),
            bpp=float(row['effective_bpp']),method=row['method'],placement=row['placement'],
            tile_clean_score=float(tile[0]),tile_stego_score=float(tile[1]),counts=counts.tolist(),valid_negative=valid.tolist()))
    rf=getattr(network,'cell_receptive_field',48)
    label='class-score map' if hasattr(network,'class_score_cells') else 'tile-head CAM'
    fig.suptitle('Same tiles: changed-cell targets, '+label+(', trained auxiliary head' if has_aux else ' (no auxiliary head)')+f'\n32-pixel spacing; {rf}×{rf} receptive field. Maps are not causal pixel saliency.')
    fig.tight_layout();output.parent.mkdir(parents=True,exist_ok=True)
    try:fig.savefig(output,dpi=160)
    finally:plt.close(fig)
    output.with_suffix('.json').write_text(json.dumps(dict(score_limits=score_limits,tiles=records),indent=2)+'\n')
