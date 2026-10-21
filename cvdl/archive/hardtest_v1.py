"""Hard test v1: a frozen, separate test of hard cells, reported next to the unchanged standard test.

Cells: panel x k in {1,2} x condition. Panels: RAISE+DIV2K combined, ALASKA seen, ALASKA unseen, LSDIR.
Conditions: all (bpp 0.05-0.5), low bpp (0.05-0.15), noisy (top quarter of clean-tile KV noise, edge from
that source's VALIDATION tiles only), low+noisy. Test-split photos only. Never used for training or
checkpoint choice. The pair list is derived, not drawn from a shared RNG, so it rebuilds identically.

Pair identity: one pair = one clean 512 tile of the standard test grid + embedding settings drawn from a seed
keyed by (version, panel, k, condition, slot). A tile may be reused by a different slot (different embedding).
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd

from cvdl.hardtest import COMBINATIONS, CONDITIONS, KS, PAIRS_PER_CELL, PER_COMBINATION, score, wilson

VERSION='hard-test-v1'
PANELS=('raise_div2k','alaska_seen','alaska_unseen','lsdir')
LOW_BPP=(0.05,0.15)
ALL_BPP=(0.05,0.5)
STRIDE_RANGE=(2,16)
PANEL_FILES={'raise_div2k':'.','alaska_seen':'extra-alaska-test_seen','alaska_unseen':'extra-alaska-test_unseen','lsdir':'extra-lsdir-test'}
PANEL_SPLITS={'raise_div2k':('test',),'alaska_seen':('test_seen',),'alaska_unseen':('test_unseen',),'lsdir':('test',)}
NOISE_QUANTILE=0.75


def cell_names():
    return [(p,k,c) for p in PANELS for k in KS for c in CONDITIONS]


def key_bytes(*parts):
    return hashlib.sha256('/'.join(str(x) for x in (VERSION,*parts)).encode()).digest()


def rng_for(*parts):
    return np.random.default_rng(np.random.SeedSequence(np.frombuffer(key_bytes(*parts),dtype='<u4')))


def validation_edges(noise_by_cover, validation_clean):
    """Per-source top-quarter edge from VALIDATION clean tiles only. validation_clean has cover_id, source."""
    out={}
    for source,group in validation_clean.groupby('source'):
        values=group.cover_id.map(noise_by_cover)
        if values.isna().any():
            raise ValueError(f'missing validation noise values for {source}')
        out[source]=float(np.quantile(values.to_numpy(),NOISE_QUANTILE))
    return out


def tile_pool(panel_clean, noise_by_cover, edges, condition):
    """Clean tiles eligible for a condition. panel_clean: one row per clean test tile (cover_id, source, ...)."""
    pool=panel_clean.copy()
    pool['kv_std']=pool.cover_id.map(noise_by_cover)
    if pool.kv_std.isna().any():
        raise ValueError('missing test noise values')
    if condition in ('noisy','low_noisy'):
        pool=pool[pool.kv_std>pool.source.map(edges)]
    return pool.sort_values('cover_id').reset_index(drop=True)


def assign_tiles(pool_ids, n, panel, k, condition):
    """n tile slots from a pool: every tile used floor(n/T) or ceil(n/T) times; the extras are keyed, not random."""
    T=len(pool_ids)
    if T==0:
        raise ValueError('empty tile pool')
    reps,extra=divmod(n,T)
    order=rng_for(panel,k,condition,'tiles').permutation(T)
    slots=np.concatenate([np.tile(np.arange(T),reps),order[:extra]]) if extra else np.tile(np.arange(T),reps)
    rng_for(panel,k,condition,'slots').shuffle(slots)
    return [pool_ids[i] for i in slots]


def combination_slots(n, panel, k, condition):
    """The balanced combination for each of n slots, in a keyed shuffled order."""
    if n!=PAIRS_PER_CELL:
        raise ValueError('hard-test cells have exactly PAIRS_PER_CELL pairs')
    slots=np.repeat(np.arange(len(COMBINATIONS)),PER_COMBINATION)
    rng_for(panel,k,condition,'combos').shuffle(slots)
    return slots


def draw_bpp(panel,k,condition,slot,low,high):
    return float(rng_for(panel,k,condition,slot,'bpp').uniform(low,high))


def draw_stride(panel,k,condition,slot):
    return int(rng_for(panel,k,condition,slot,'stride').integers(STRIDE_RANGE[0],STRIDE_RANGE[1]+1))


def sha256_text(text):
    return hashlib.sha256(text.encode()).hexdigest()


# ----- building and replaying the frozen pair list ------------------------------------------------------------

LIST_COLUMNS=('pair_id','cell','panel','k','condition','slot','cover_id','source','source_photo','camera_model','split',
              'grid','tile_row','tile_col','y','x','photo_index','bit_mode','method','placement','stride','payload_kind',
              'requested_bpp','tile_kv_std','tile_use')


def pair_settings(panel, k, condition, slot, combination_index):
    """The embedding settings of one slot: balanced combination, keyed stride and bpp."""
    bit_mode,method,placement,payload=COMBINATIONS[combination_index]
    low,high=LOW_BPP if condition in ('low','low_noisy') else ALL_BPP
    return dict(bit_mode=bit_mode,method=method,placement=placement,payload_kind=payload,
                stride=draw_stride(panel,k,condition,slot) if placement=='stride' else 1,
                requested_bpp=draw_bpp(panel,k,condition,slot,low,high))


def build_pairs(panel_clean, noise_by_cover, edges, panel, k, condition):
    """The 1,008 pairs of one cell, as rows of LIST_COLUMNS, from the frozen clean test tiles of one panel."""
    pool=tile_pool(panel_clean,noise_by_cover,edges,condition)
    ids=pool.cover_id.tolist()
    tiles=assign_tiles(ids,PAIRS_PER_CELL,panel,k,condition)
    combos=combination_slots(PAIRS_PER_CELL,panel,k,condition)
    by_id=pool.set_index('cover_id')
    uses={}
    rows=[]
    for slot,(cover,combo) in enumerate(zip(tiles,combos)):
        uses[cover]=uses.get(cover,0)+1
        t=by_id.loc[cover]
        settings=pair_settings(panel,k,condition,slot,int(combo))
        rows.append(dict(pair_id=f'{VERSION}/{panel}/k{k}/{condition}/{slot:04d}',cell=f'{panel}/k{k}/{condition}',panel=panel,
            k=k,condition=condition,slot=slot,cover_id=cover,source=t['source'],source_photo=t['source_photo'],
            camera_model=t['camera_model'],split=t['split'],grid=t['grid'],tile_row=int(t['tile_row']),tile_col=int(t['tile_col']),
            y=int(t['y']),x=int(t['x']),photo_index=int(t['photo_index']),tile_kv_std=float(t['kv_std']),
            tile_use=uses[cover],**settings))
    return rows


def pair_list_csv(rows):
    frame=pd.DataFrame(rows,columns=LIST_COLUMNS)
    return frame.to_csv(index=False,float_format='%.17g')


def check_cell(rows, pool_size):
    """Invariants of one built cell (also used by tests)."""
    frame=pd.DataFrame(rows)
    if len(frame)!=PAIRS_PER_CELL:
        raise ValueError(f'cell has {len(frame)} pairs')
    per=frame.groupby(['bit_mode','method','placement','payload_kind']).size()
    if len(per)!=len(COMBINATIONS) or set(per.unique())!={PER_COMBINATION}:
        raise ValueError('combinations are not balanced')
    counts=frame.cover_id.value_counts()
    reps,extra=divmod(PAIRS_PER_CELL,pool_size)
    if counts.min()<reps or counts.max()>reps+(1 if extra else 0):
        raise ValueError('tile reuse is not even')
    return True


# ----- scoring: the dataset, metrics and report ---------------------------------------------------------------

def embed_config(row, template):
    """A one-slot embedding config: this pair's fixed k, bit_mode, method, placement, stride, payload and bpp."""
    from lsb_batch import Config
    return Config(seed=int(template.seed),input=template.input,output=template.output,
                  bit_mode=row['bit_mode'],k=int(row['k']),method=row['method'],placement=row['placement'],
                  stride=int(row['stride']),payload=row['payload_kind'],start='random',size={'bpp':float(row['requested_bpp'])})


def embedding_seed(row):
    """Two SeedSequences (settings draw, embedding), keyed by the pair id, never by position."""
    key=np.frombuffer(key_bytes('embed',row['pair_id']),dtype='<u4')
    return np.random.SeedSequence(list(key)+[0]),np.random.SeedSequence(list(key)+[1])


def _fixed_pairs_class():
    from lsb_torch import FixedPairs
    return FixedPairs


class HardPairs(_fixed_pairs_class()):
    """A frozen hard-test pair list as a dataset of (clean, stego, index), one decode per photo run.

    Subclasses FixedPairs only so the photo-owned loader (one task per photo) accepts it; it does not call
    FixedPairs.__init__. Same crop rule as the standard test: the recorded 512 tile of the test photo.
    """
    def __init__(self, frame, corpus):
        self.frame=frame.reset_index(drop=True)
        self.corpus=corpus
        self._last_id=None
        self._last_photo=None
        self.records=corpus.by_id
        self.entries=[(self.records[p],) for p in self.frame.source_photo]

    def __len__(self):
        return len(self.frame)

    def __getstate__(self):
        state=self.__dict__.copy();state.update(_last_id=None,_last_photo=None);return state

    def pair(self, index):
        import torch
        from lsb_settings import draw_otf_settings
        from lsb_engine import embed_fast
        from lsb_source import tile_at
        row=self.frame.iloc[index]
        record=self.records[row['source_photo']]
        if self._last_id!=row['source_photo']:
            self._last_photo=self.corpus.photo(record);self._last_id=row['source_photo']
        view=tile_at(self._last_photo,record,row['grid'],512,int(row['tile_row']),int(row['tile_col']))
        clean=view.pixels.copy()
        settings_seed,embed_seed=embedding_seed(row)
        config=embed_config(row,self.corpus.evaluation_config)
        kwargs,meta=draw_otf_settings(config,view.pixels,np.random.default_rng(settings_seed))
        kwargs['rng']=np.random.default_rng(embed_seed)
        stego,_,_=embed_fast(view.pixels,**kwargs)
        info=dict(effective_bpp=float(meta['effective_bpp']),requested_bpp=float(meta['requested_bpp']),clipped=bool(meta['clipped']),
                  payload_bytes=int(meta['payload_bytes']))
        return clean,stego,info

    def __getitem__(self, index):
        import torch
        clean,stego,_=self.pair(index)
        tensor=lambda a:torch.from_numpy(np.ascontiguousarray(a)).permute(2,0,1).contiguous()
        return tensor(clean),tensor(stego),index


def wilson_row(events, n):
    low,high=wilson(events,n)
    return dict(rate=events/n if n else float('nan'),ci95_low=low,ci95_high=high)


def cell_report(scored, cutoffs):
    """Per-cell metrics from a scored frame (one row per pair and label).

    miss over all stego pairs; clean FA over DISTINCT clean tiles (a reused tile counts once);
    AUC over all pairs. cutoffs: {source: validation FA5 cut-off} (the standard tables' cut-off).
    """
    from cvdl.metrics import auc_numpy
    rows=[]
    for cell,group in scored.groupby('cell',sort=False):
        stego=group[group.label==1];clean=group[group.label==0]
        distinct=clean.drop_duplicates('cover_id')
        cut=distinct.source.map(cutoffs)
        fa_events=int((distinct.prob_stego>=cut).sum());stego_cut=stego.source.map(cutoffs)
        miss_events=int((stego.prob_stego<stego_cut).sum())
        by_source={s:int(n) for s,n in stego.source.value_counts().sort_index().items()}
        rows.append(dict(cell=cell,panel=stego.panel.iloc[0],k=int(stego.k.iloc[0]),condition=stego.condition.iloc[0],
            pairs=len(stego),distinct_clean_tiles=len(distinct),distinct_photos=int(group.source_photo.nunique()),
            max_tile_use=int(clean.cover_id.value_counts().max()),
            miss_events=miss_events,**{f'miss_{k}':v for k,v in wilson_row(miss_events,len(stego)).items()},
            fa_events=fa_events,**{f'fa_{k}':v for k,v in wilson_row(fa_events,len(distinct)).items()},
            auc=float(auc_numpy(group.label.to_numpy(),group.prob_stego.to_numpy())),
            clipped_pairs=int(stego.clipped.sum()),clipped_share=float(stego.clipped.mean()),
            effective_bpp_mean=float(stego.effective_bpp.mean()),by_source=json.dumps(by_source)))
    return pd.DataFrame(rows)


def embedding_info(dataset):
    """effective_bpp, clipped, payload_bytes per pair from the settings draw only (no pixels are changed)."""
    from lsb_settings import draw_otf_settings
    tile=np.zeros((512,512,3),dtype=np.uint8)
    rows=[]
    for _,row in dataset.frame.iterrows():
        seed,_=embedding_seed(row)
        config=embed_config(row,dataset.corpus.evaluation_config)
        _,meta=draw_otf_settings(config,tile,np.random.default_rng(seed))
        rows.append(dict(pair_id=row['pair_id'],effective_bpp=float(meta['effective_bpp']),clipped=bool(meta['clipped']),
                         payload_bytes=int(meta['payload_bytes'])))
    return pd.DataFrame(rows)


# ----- the evaluation stage -----------------------------------------------------------------------------------

DATA_DIR='data/hard-test/v1'
OUTPUT_FILES=('hard_predictions.csv','hard_cells.csv','hard_summary.json')


def read_pair_list(path_or_buffer):
    """Read a frozen pair list with the exact float parser: the default fast parser can change the last digit,
    and requested_bpp feeds the payload size."""
    return pd.read_csv(path_or_buffer,float_precision='round_trip')


def load_frozen(root):
    """The frozen pair lists, with their SHA-256 checked against summary.json; refuses a changed file."""
    folder=Path(root)/DATA_DIR
    summary=json.loads((folder/'summary.json').read_text())
    frames=[]
    for panel in PANELS:
        name=f'pairs-{panel}.csv'
        text=(folder/name).read_text()
        if sha256_text(text)!=summary['files_sha256'][name]:
            raise ValueError(f'frozen hard-test list changed: {name}')
        frames.append(read_pair_list(folder/name))
    return pd.concat(frames,ignore_index=True),summary


QUICK_PAIRS_PER_CELL=8


def run(settings, otf, network, folder, out, *, limit=None):
    """Score the frozen hard test with a trained network; write new files only under out/hard-test/."""
    from cvdl.archive import panels
    out=Path(out)/'hard-test';out.mkdir(parents=True,exist_ok=True)
    frame,frozen=load_frozen(settings.root)
    if limit is None:limit=QUICK_PAIRS_PER_CELL if settings.quick else 0
    if limit:frame=frame.groupby('cell',sort=False).head(limit).reset_index(drop=True)
    cutoffs=panels.source_cutoffs(folder)
    dataset=HardPairs(frame,otf.corpus)
    pair_batch=max(1,settings.batch_size//2)
    scored=score(network,dataset,device=settings.device,batch_size=pair_batch,workers=settings.eval_workers)
    info=embedding_info(dataset).set_index('pair_id')
    scored=scored.join(info,on='pair_id')
    cells=cell_report(scored,cutoffs)
    scored.to_csv(out/'hard_predictions.csv',index=False,float_format='%.9g')
    cells.to_csv(out/'hard_cells.csv',index=False)
    (out/'hard_summary.json').write_text(json.dumps(dict(version=VERSION,pairs=int(len(frame)),cells=int(len(cells)),
        frozen_files_sha256=frozen['files_sha256'],noise_edges_from_validation_only=frozen['noise_edges_from_validation_only'],
        source_validation_fa5_cutoffs=cutoffs,quick_limit_per_cell=limit or None,batch_pairs=pair_batch),indent=2,sort_keys=True)+'\n')
    return cells
