"""One frozen validation-derived deployment threshold; no test fitting here."""
from fractions import Fraction
import json
from pathlib import Path
import re
from types import SimpleNamespace

import numpy as np
import pandas as pd
from . import validation as V
from .metrics import threshold, rates, roc_auc

POLICY='validation-min-pe-v1/atol1e-12/min-fa/first-ascending/float64-ge'
MAPPING='configs/evaluation-calibration-v1.json'
MAPPING_SHA='14f2674478b765f957f6bd7e6b9843201229e4a9adfdd06776b456e9647a5275'


def mapping(root):
    raw=(Path(root)/MAPPING).read_bytes()
    if V.digest(raw)!=MAPPING_SHA:raise ValueError('frozen calibration mapping SHA256 mismatch')
    data=json.loads(raw)
    if data['version']!='evaluation-calibration-v1':raise ValueError('invalid calibration mapping')
    entries=data['entries']
    if len(entries)!=len({e['checkpoint_sha256'] for e in entries}):raise ValueError('duplicate calibration mapping')
    return {e['checkpoint_sha256']:e for e in entries}


def load_recipe(root,rid):
    """Standalone evaluation descriptor, not fabricated training RunSettings.

    In particular legacy DIV2K source shares must not pass through the training
    source_mix API, which deliberately does not support DIV2K mixtures.
    """
    if not isinstance(rid,str) or not re.fullmatch('[0-9a-f]{20}',rid):raise ValueError('invalid calibration recipe id')
    root=Path(root);recipe=json.loads((root/'data/val'/rid/'recipe.json').read_text())
    from lsb_batch import read_config
    configs=[root/'configs/train-k1to4.yaml',root/'configs/train-hard-v1.yaml']
    configs=[p for p in configs if V.canonical({k:v for k,v in read_config(p).as_dict().items() if k in V.SEMANTIC_KEYS})==V.canonical(recipe['config'])]
    if len(configs)!=1:raise ValueError('calibration recipe has no unique supported config')
    descriptor=SimpleNamespace(sources=tuple(recipe['source_mix']),
        source_mix={s:Fraction(w) for s,w in recipe['source_mix'].items()},
        tiles_per_epoch=recipe['train_pairs_per_epoch'],tile_size=512,variants_per_tile=1,
        settings_yaml=configs[0],manifest_root=None)
    return V.load_frozen(root,descriptor,'overlap_last',recipe['tile_size'],rid)


def resolve(settings,checkpoint,checkpoint_sha):
    """Own saved recipe wins. Unmapped old checkpoints fail; never borrow calibration."""
    root=Path(settings.root);checkpoint=Path(checkpoint)
    saved=checkpoint.parent/'run_settings.json'
    own=json.loads(saved.read_text())['values'] if saved.exists() else {}
    rid=own.get('validation_recipe')
    if rid:
        if settings.validation_recipe!=rid:raise ValueError('calibration must use evaluated run own validation recipe, not borrowed settings')
        frame,recipe,summary,text=V.load_frozen(root,settings.to_run_settings(),settings.train_grid,settings.train_tile_size,rid)
        identity=dict(threshold_origin='train_matched',recipe_id=rid,run_name=own.get('run_name'),mapping_sha256=None)
        prior=checkpoint.parent/'validation_pairs.csv'
        if prior.exists() and prior.read_bytes()!=text.encode():raise ValueError('saved validation identity differs')
    else:
        entry=mapping(root).get(checkpoint_sha)
        if entry is None:raise ValueError('no saved validation recipe or frozen checkpoint calibration mapping; approve an offline mapping (no fallback)')
        rid=entry['recipe_id'];frame,recipe,summary,text=load_recipe(root,rid)
        if summary['pairs_sha256']!=entry['pairs_sha256'] or summary['recipe_sha256']!=entry['recipe_sha256'] or V.digest(text.encode())!=entry['pairs_sha256']:
            raise ValueError('calibration list differs from frozen checkpoint mapping')
        identity=dict(threshold_origin=entry['threshold_origin'],recipe_id=rid,run_name=entry['run_name'],mapping_sha256=MAPPING_SHA)
    if set(frame['split'])!={'val'} or frame.cover_id.duplicated().any():raise ValueError('calibration requires unique validation tiles')
    if recipe['tile_size']!=256:raise ValueError('standard256 calibration requires256 geometry')
    full_sha=summary['pairs_sha256']
    if settings.quick:frame=frame.groupby('source',sort=True).head(8).reset_index(drop=True)
    identity.update(checkpoint_sha256=checkpoint_sha,policy=POLICY,recipe_sha256=summary['recipe_sha256'],
        full_pairs_sha256=full_sha,selected_pairs_sha256=V.digest(V.csv_text(frame).encode()),
        geometry=dict(tile_size=recipe['tile_size'],grid='overlap_last'),quick=bool(settings.quick),deployment=not settings.quick)
    if settings.quick:identity['threshold_origin']='quick_'+identity['threshold_origin']
    return frame,identity


def assert_disjoint(validation,tests):
    photos=set(validation.source_photo)
    if any(photos & set(frame.source_photo) for frame in tests):raise ValueError('validation/test photo overlap')


def fit(scored,identity):
    """Fit on validation only. Identity and score bytes bind the frozen cutoff."""
    scored=scored.copy();scored['prob_stego']=scored.prob_stego.astype(np.float64)
    if set(scored['split'])!={'val'}:raise ValueError('threshold fitting requires val rows only')
    if set(scored.label.unique())!={0,1} or not np.isfinite(scored.prob_stego).all():raise ValueError('invalid calibration scores/labels')
    if scored.groupby('pair_id').label.agg(list).map(lambda v:sorted(v)!=[0,1]).any():
        raise ValueError('calibration requires exactly one clean/stego per pair')
    if scored[scored.label==0].cover_id.duplicated().any():raise ValueError('calibration requires unique clean tiles')
    cutoff,_=threshold(scored,'pe');r=rates(scored,cutoff)
    text=V.csv_text(scored)
    result=dict(**identity,score_csv_sha256=V.digest(text.encode()),cutoff=cutoff,FA=r['fa'],miss=r['miss'],
        P_E_at_validation_cutoff=r['pe_at_cutoff'],AUC=roc_auc(scored.prob_stego.to_numpy(),scored.label.to_numpy().astype(bool)),
        pairs=r['pairs'],clean_tiles=int((scored.label==0).sum()),images=len(scored),photos=int(scored.source_photo.nunique()))
    result['threshold_id']=V.digest(V.canonical(result).encode())[:20]
    refs=[]
    for source,g in scored.groupby('source',sort=True):
        if source=='div2k':continue
        t,_=threshold(g,'pe');v=rates(g,t)
        refs.append(dict(source=source,threshold_origin='if_source_known_reference',cutoff=t,
            FA=v['fa'],miss=v['miss'],P_E_at_validation_cutoff=v['pe_at_cutoff'],pairs=v['pairs'],
            threshold_id=result['threshold_id'],recipe_id=identity['recipe_id'],
            AUC=roc_auc(g.prob_stego.to_numpy(),g.label.to_numpy().astype(bool)),
            distinct_clean_tiles=int((g.label==0).sum()),images=len(g),photos=int(g.source_photo.nunique())))
    return result,pd.DataFrame(refs),text


def write(target,scored,identity):
    result,refs,text=fit(scored,identity)
    target=Path(target);target.mkdir(parents=True,exist_ok=True)
    (target/'predictions.csv').write_bytes(text.encode())
    (target/'calibration.json').write_text(V.canonical(result)+'\n')
    refs.to_csv(target/'source_reference_thresholds.csv',index=False,float_format='%.17g')
    return result,refs


def main_rows(scored,diagnostics,calibration):
    scored=scored.copy();scored['prob_stego']=scored.prob_stego.astype(np.float64)
    result=[]
    for row in diagnostics.itertuples(index=False):
        g=scored
        if row.category=='source':g=g[g.source.astype(str)==row.value]
        elif row.category=='k':g=g[g.k.astype(str)==row.value]
        elif row.category=='source/k':
            source,k=row.value.rsplit('/',1);g=g[(g.source==source)&(g.k.astype(str)==k)]
        clean=g[g.label==0].drop_duplicates('cover_id');positive=g[g.label==1]
        r=rates(pd.concat([clean,positive],ignore_index=True),np.float64(calibration['cutoff']))
        result.append(dict(panel=row.panel,category=row.category,value=row.value,pairs=len(positive),
            images=len(g),photos=int(g.source_photo.nunique()),distinct_clean_tiles=len(clean),AUC=row.AUC,
            threshold_origin=calibration['threshold_origin'],threshold_id=calibration['threshold_id'],cutoff=calibration['cutoff'],
            FA=r['fa'],miss=r['miss'],P_E_at_validation_cutoff=r['pe_at_cutoff']))
    return pd.DataFrame(result)


def reference_rows(scored,refs,*,panel,cell=None):
    scored=scored.copy();scored['prob_stego']=scored.prob_stego.astype(np.float64)
    rows=[]
    # No mixed-source aggregate. Only complete, source-specific subsets.
    for ref in refs.itertuples(index=False):
        source=scored[scored.source==ref.source]
        if source.empty:continue
        for k,g in [('all',source)]+[(str(k),g) for k,g in source.groupby('k',sort=True)]:
            clean=g[g.label==0].drop_duplicates('cover_id');stego=g[g.label==1]
            r=rates(pd.concat([clean,stego]),np.float64(ref.cutoff))
            rows.append(dict(panel=panel,cell=cell,source=ref.source,k=k,threshold_origin='if_source_known_reference',
                threshold_id=ref.threshold_id,cutoff=ref.cutoff,FA=r['fa'],miss=r['miss'],P_E_at_validation_cutoff=r['pe_at_cutoff'],
                pairs=len(stego),distinct_clean_tiles=len(clean),AUC=roc_auc(g.prob_stego.to_numpy(),g.label.to_numpy().astype(bool))))
    return pd.DataFrame(rows,columns=['panel','cell','source','k','threshold_origin','threshold_id','cutoff',
        'FA','miss','P_E_at_validation_cutoff','pairs','distinct_clean_tiles','AUC'])
