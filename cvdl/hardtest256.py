"""Frozen hard test at256; independent of models, training recipes and legacy hard-v1."""
from __future__ import annotations
import io
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd

from . import hardnoise as N, standardtest as S
from .hardtest import COMBINATIONS, CONDITIONS, KS, PAIRS_PER_CELL, PER_COMBINATION, wilson
from .validation import csv_text

VERSION = 'hard-test-256-v1'
DATA_DIR = N.DATA_DIR
PANELS = tuple(S.PANELS)
OUTPUT_FILES = ('hard_predictions.csv', 'hard_cells.csv', 'hard_summary.json','source_reference_summary.csv')


def seed_key(*parts):
    return [int(x) for x in np.frombuffer(bytes.fromhex(N.digest('/'.join(map(str,(VERSION,1,*parts))).encode())), dtype='<u4')]


def rng(*parts):
    return np.random.default_rng(np.random.SeedSequence(seed_key(*parts)))


def pool_rule(panel, size):
    if size >= PAIRS_PER_CELL: return 1
    if panel.startswith('alaska-') and size >= PAIRS_PER_CELL//2: return 2
    raise ValueError(f'{panel}: pool {size} fails1,008 unique/ALASKA>=504 rule; no edge/count adjustment allowed')


def assign_tiles(size, panel, k, condition):
    pool_rule(panel,size)
    reps, extra = divmod(PAIRS_PER_CELL,size)
    order = rng(panel,k,condition,'tiles').permutation(size)
    indices = np.concatenate([np.tile(np.arange(size),reps),order[:extra]])
    rng(panel,k,condition,'slots').shuffle(indices)
    return indices


def validate_noise(frames, root):
    """Every eligible coordinate exactly once, val/test split membership from frozen manifests."""
    from lsb_train_grid import axes
    expected = {'noise-val.csv':set(), 'noise-test.csv':set()}
    for r in N.records(root):
        ys,xs = axes(r,256,'overlap_last')
        key = 'noise-val.csv' if r['split']=='val' else 'noise-test.csv'
        expected[key].update((f"{r['photo_id']}:overlap_last:256:{row}:{col}:v1",r['source'],r['photo_id'],r['split'],y,x)
                             for row,y in enumerate(ys) for col,x in enumerate(xs))
    for name,f in frames.items():
        actual = set(f[['cover_id','source','source_photo','split','y','x']].itertuples(index=False,name=None))
        if (len(f)!=len(expected[name]) or actual!=expected[name] or f.cover_id.duplicated().any()
                or set(f.tile_size)!={256} or set(f.grid)!={'overlap_last'}
                or not np.isfinite(f.kv_std).all() or not f.kv_std.between(0,3).all()):
            raise ValueError(f'incomplete/invalid frozen noise geometry: {name}')


def describe(root):
    frames, info = N.load_noise(root)
    validate_noise(frames,root)
    val,test = frames['noise-val.csv'],frames['noise-test.csv']
    edges = {s:float(np.quantile(g.kv_std.to_numpy(),.75,method='linear')) for s,g in val.groupby('source',sort=True)}
    pools = {}
    for panel,sources in S.PANELS.items():
        mask = np.zeros(len(test),dtype=bool)
        for source,(split,_) in sources.items(): mask |= test.source.eq(source)&test['split'].eq(split)
        base = test[mask].sort_values('cover_id').reset_index(drop=True)
        noisy = base[base.kv_std>base.source.map(edges)].reset_index(drop=True)
        for condition in CONDITIONS:
            pool = noisy if condition in ('noisy','low_noisy') else base
            pool_rule(panel,len(pool)); pools[(panel,condition)] = pool
    standard, _, stdinfo = S.load_frozen(root)
    recipe = dict(version=VERSION, schema=1, inputs=N.inputs(root), noise_definition=info['definition'],
        noise_files_sha256=info['files_sha256'], geometry='train-grid-v1/overlap_last', tile_size=256,
        noise_edges=edges, quantile=dict(q=.75,method='linear',comparison='strict > source edge',split='val'),
        panels={p:{s:v[0] for s,v in S.PANELS[p].items()} for p in PANELS}, ks=list(KS), conditions=list(CONDITIONS),
        pairs_per_cell=PAIRS_PER_CELL, combinations=[list(c) for c in COMBINATIONS], per_combination=PER_COMBINATION,
        bpp=dict(all=[.05,.5],low=[.05,.15]), stride=[2,16], start='random', seed=1, seed_domain=VERSION,
        tile_assignment='keyed balanced cycles and extras; unique when pool>=1008; ALASKA max2 otherwise fail',
        standard_overlap_reference=stdinfo['recipe_sha256'])
    overlaps = {p:set(standard[S.filename(p)].cover_id) for p in PANELS}
    return recipe,pools,overlaps


def build_cell(pool, panel, k, condition, recipe_sha):
    from lsb_batch import Config
    from lsb_settings import draw_otf_settings
    indices = assign_tiles(len(pool),panel,k,condition)
    combos = np.repeat(np.arange(len(COMBINATIONS)),PER_COMBINATION)
    rng(panel,k,condition,'combos').shuffle(combos)
    image = np.zeros((256,256,3),dtype=np.uint8)
    rows,uses = [],{}
    cell = f'{panel}/k{k}/{condition}'
    for slot,(index,combo) in enumerate(zip(indices,combos)):
        tile = pool.iloc[int(index)]
        b,m,p,payload = COMBINATIONS[int(combo)]
        low,high = (.05,.15) if condition in ('low','low_noisy') else (.05,.5)
        bpp = float(rng(panel,k,condition,slot,'bpp').uniform(low,high))
        stride = int(rng(panel,k,condition,slot,'stride').integers(2,17)) if p=='stride' else 1
        config = Config(seed=1,input='unused',output='unused',bit_mode=b,k=k,method=m,placement=p,
                        stride=stride,payload=payload,start='random',size={'bpp':bpp})
        skey,ekey = seed_key(cell,slot,'settings'),seed_key(cell,slot,'embedding')
        kwargs,meta = draw_otf_settings(config,image,np.random.default_rng(np.random.SeedSequence(skey)))
        kwargs.pop('rng')
        uses[tile.cover_id] = uses.get(tile.cover_id,0)+1
        row = {key:tile[key] for key in N.COLUMNS if key!='kv_std'}
        row.update(pair_id=f'{recipe_sha[:20]}/{cell}/{slot:04d}',cell=cell,panel=panel,k=k,condition=condition,
            slot=slot,variant=1,tile_kv_std=float(tile.kv_std),tile_use=uses[tile.cover_id],bit_mode=b,method=m,
            placement=p,stride=stride,payload_kind=payload,requested_bpp=bpp,payload_bytes=meta['payload_bytes'],
            effective_bpp=meta['effective_bpp'],clipped=meta['clipped'],draw_kwargs=N.canonical(kwargs),
            settings_seed_key=N.canonical(skey),embedding_seed_key=N.canonical(ekey))
        rows.append(row)
    return pd.DataFrame(rows)


def cell_stats(frame, pool, overlap):
    counts = frame.cover_id.value_counts()
    return dict(cell=frame.cell.iloc[0],panel=frame.panel.iloc[0],k=int(frame.k.iloc[0]),condition=frame.condition.iloc[0],
        pairs=len(frame),pool_tiles=len(pool),distinct_tiles=int(len(counts)),distinct_photos=int(frame.source_photo.nunique()),
        max_tile_use=int(counts.max()),overlap_standard_distinct_tiles=len(set(frame.cover_id)&overlap),
        overlap_standard_pairs=int(frame.cover_id.isin(overlap).sum()),clipped_pairs=int(frame.clipped.sum()),
        by_source={s:int(n) for s,n in frame.source.value_counts().sort_index().items()})


def verify_cell(frame,pool,panel,k,condition):
    if (len(frame)!=PAIRS_PER_CELL or frame.pair_id.duplicated().any() or set(frame.panel)!={panel}
            or set(frame.k)!={k} or set(frame.condition)!={condition}): raise ValueError('hard256 cell counts/identity mismatch')
    combos = frame.groupby(['bit_mode','method','placement','payload_kind']).size()
    if set(combos.index)!=set(COMBINATIONS) or set(combos)!={PER_COMBINATION}: raise ValueError('hard256 type balance mismatch')
    by_cover = pool.set_index('cover_id')
    if not set(frame.cover_id)<=set(by_cover.index): raise ValueError('hard256 tile outside eligible test pool')
    used = frame.cover_id.value_counts()
    if used.max()>pool_rule(panel,len(pool)): raise ValueError('hard256 reuse limit exceeded')
    reps,extra = divmod(PAIRS_PER_CELL,len(pool))
    if len(used)!=min(len(pool),PAIRS_PER_CELL) or (used.max()-used.min()>1) or used.min()<max(1,reps):
        raise ValueError('hard256 tile assignment not balanced')
    high = .15 if condition in ('low','low_noisy') else .5
    if not frame.requested_bpp.between(.05,high).all(): raise ValueError('hard256 bpp outside cell range')
    for r in frame.itertuples():
        t = by_cover.loc[r.cover_id]
        if any(getattr(r,key)!=t[key] for key in N.COLUMNS if key not in ('kv_std','cover_id')) or r.tile_kv_std!=t.kv_std:
            raise ValueError('hard256 recorded geometry/noise mismatch')


def build(root):
    recipe,pools,overlaps = describe(root)
    sha = N.digest(N.canonical(recipe).encode())
    files,stats,shas = {},[],{}
    for panel in PANELS:
        cells = []
        for k in KS:
            for condition in CONDITIONS:
                pool = pools[(panel,condition)]
                cell = build_cell(pool,panel,k,condition,sha)
                verify_cell(cell,pool,panel,k,condition)
                stats.append(cell_stats(cell,pool,overlaps[panel]));cells.append(cell)
        text = csv_text(pd.concat(cells,ignore_index=True))
        name = f'pairs-{panel}.csv';files[name]=text;shas[name]=N.digest(text.encode())
    summary = dict(version=VERSION,recipe_sha256=sha,recipe_id=sha[:20],pairs=32*PAIRS_PER_CELL,
                   noise_edges=recipe['noise_edges'],files_sha256=shas,cells=stats)
    files['recipe.json']=N.canonical(recipe)+'\n';files['summary.json']=N.canonical(summary)+'\n'
    return files,summary


def load_frozen(root):
    recipe,pools,_ = describe(root)
    folder = Path(root)/DATA_DIR
    saved = json.loads((folder/'recipe.json').read_text());summary=json.loads((folder/'summary.json').read_text())
    sha = N.digest(N.canonical(recipe).encode())
    if saved!=recipe or summary['recipe_sha256']!=sha or summary['recipe_id']!=sha[:20] or summary['version']!=VERSION:
        raise ValueError('hard256 recipe/input mismatch')
    frames=[]
    for panel in PANELS:
        name=f'pairs-{panel}.csv';raw=(folder/name).read_bytes()
        if N.digest(raw)!=summary['files_sha256'][name]: raise ValueError(f'hard256 list SHA256 mismatch: {name}')
        frame=pd.read_csv(io.BytesIO(raw),float_precision='round_trip')
        if len(frame)!=8*PAIRS_PER_CELL or frame.pair_id.duplicated().any(): raise ValueError('hard256 panel size/identity mismatch')
        for k in KS:
            for condition in CONDITIONS:
                group=frame[(frame.k==k)&(frame.condition==condition)]
                verify_cell(group,pools[(panel,condition)],panel,k,condition)
        frames.append(frame)
    return pd.concat(frames,ignore_index=True),recipe,summary


def cell_report(scored,calibration=None):
    """FA and test-min FA use distinct clean tiles; AUC retains v1 paired-row weighting."""
    from .metrics import auc_numpy,metrics,threshold,rates
    scored=scored.copy();scored['prob_stego']=scored.prob_stego.astype(np.float64)
    rows=[]
    for cell,g in scored.groupby('cell',sort=False):
        clean=g[g.label==0].drop_duplicates('cover_id');stego=g[g.label==1]
        unique=pd.concat([clean,stego],ignore_index=True)
        fixed=rates(unique,.5);m=metrics(unique)
        fa_events=int((clean.prob_stego>=.5).sum());miss_events=int((stego.prob_stego<.5).sum())
        row=dict(cell=cell,panel=stego.panel.iloc[0],k=int(stego.k.iloc[0]),condition=stego.condition.iloc[0],
            pairs=len(stego),distinct_clean_tiles=len(clean),distinct_photos=int(g.source_photo.nunique()),
            max_tile_use=int(g[g.label==0].cover_id.value_counts().max()),FA_at_0_5=fixed['fa'],miss_at_0_5=fixed['miss'],
            P_E_at_0_5=fixed['pe_at_cutoff'],FA_events_at_0_5=fa_events,miss_events_at_0_5=miss_events,
            FA_at_0_5_ci95_low=wilson(fa_events,len(clean))[0],FA_at_0_5_ci95_high=wilson(fa_events,len(clean))[1],
            miss_at_0_5_ci95_low=wilson(miss_events,len(stego))[0],miss_at_0_5_ci95_high=wilson(miss_events,len(stego))[1],
            AUC=auc_numpy(g.label.to_numpy(),g.prob_stego.to_numpy()),test_min_P_E=m['P_E'],test_min_cutoff=threshold(unique,'pe')[0],
            overlap_standard_distinct_tiles=int(clean.overlap_standard.sum()) if 'overlap_standard' in clean else None,
            overlap_standard_pairs=int(stego.overlap_standard.sum()) if 'overlap_standard' in stego else None,
            threshold_origin='pending_validation_min_pe',cutoff=None,FA=None,miss=None,P_E_at_validation_cutoff=None)
        if calibration is not None:
            r=rates(unique,np.float64(calibration['cutoff']))
            fa=int((clean.prob_stego>=np.float64(calibration['cutoff'])).sum())
            missed=int((stego.prob_stego<np.float64(calibration['cutoff'])).sum())
            row.update(threshold_origin=calibration['threshold_origin'],threshold_id=calibration['threshold_id'],
                cutoff=calibration['cutoff'],FA=r['fa'],miss=r['miss'],P_E_at_validation_cutoff=r['pe_at_cutoff'],
                FA_events=fa,miss_events=missed,FA_ci95_low=wilson(fa,len(clean))[0],FA_ci95_high=wilson(fa,len(clean))[1],
                miss_ci95_low=wilson(missed,len(stego))[0],miss_ci95_high=wilson(missed,len(stego))[1])
        rows.append(row)
    return pd.DataFrame(rows)


def run(settings,network,corpus,frame,frozen,out,*,calibration,references):
    from .hardtest import score
    target=Path(out)/'hard-test';target.mkdir(parents=True,exist_ok=True)
    before=time.perf_counter()
    if settings.quick:frame=frame.groupby('cell',sort=False).head(8).reset_index(drop=True)
    standard,_,_=S.load_frozen(settings.root)
    standard_covers=set().union(*(set(f.cover_id) for f in standard.values()))
    frame=frame.copy();frame['overlap_standard']=frame.cover_id.isin(standard_covers)
    scored=score(network,S.StandardTestPairs(corpus,frame),device=settings.device,
                 batch_size=max(1,settings.batch_size//2),workers=settings.eval_workers)
    scored.to_csv(target/OUTPUT_FILES[0],index=False,float_format='%.17g')
    cell_report(scored,calibration).to_csv(target/OUTPUT_FILES[1],index=False,float_format='%.17g')
    from .deployment import reference_rows
    pd.concat([reference_rows(g,references,panel=str(g.panel.iloc[0]),cell=cell)
        for cell,g in scored.groupby('cell',sort=False)],ignore_index=True).to_csv(target/'source_reference_summary.csv',index=False,float_format='%.17g')
    summary=dict(status='complete',version=VERSION,recipe_sha256=frozen['recipe_sha256'],
        files_sha256=frozen['files_sha256'],noise_edges=frozen['noise_edges'],pairs=len(frame),cells=int(frame.cell.nunique()),
        seconds=time.perf_counter()-before,quick=bool(settings.quick),AUC_clean_weighting='paired rows (as v1)',
        FA_counting='distinct clean tiles within cell',validation_threshold=calibration,
        overlap_standard={p:dict(distinct_hard_tiles=int(g.cover_id.nunique()),
            shared_distinct_tiles=len(set(g.cover_id)&standard_covers),shared_pairs=int(g.overlap_standard.sum()))
            for p,g in frame.groupby('panel',sort=True)})
    (target/OUTPUT_FILES[2]).write_text(json.dumps(summary,indent=2)+'\n')
    return summary
