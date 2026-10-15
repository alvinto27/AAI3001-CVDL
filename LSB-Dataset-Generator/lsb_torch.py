"""Mode 1 torch boundary: image streams, fixed pair tables and keyed geometry."""
from __future__ import annotations

import csv
from concurrent.futures import ThreadPoolExecutor
import math
import os
import queue
import threading
import weakref
import zlib
from dataclasses import dataclass, asdict
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, IterableDataset, DataLoader, get_worker_info

from lsb_batch import read_config
from lsb_cache import load_cached
from lsb_source import load_photo, grid_axes, configure_source_worker
from lsb_settings import (otf_seed_sequence, legacy_seed_sequence, draw_otf_settings,
                          draw_legacy_settings, _bpp_payload_bytes)
from lsb_mode1 import TileSpec, OTFSeed, make_sample, replay_legacy

_VALIDATED_CACHE_SESSIONS = set()


@dataclass(frozen=True)
class RunSettings:
    sources: tuple[str, ...] = ("raise", "div2k")
    tile_size: int = 512
    grid: str = "overlap_last"
    settings_yaml: str | None = None
    evaluation_yaml: str | None = None  # None keeps the frozen run1 evaluation draws.
    variants_per_tile: int = 1
    fixed_variants: bool = False
    run_seed: int = 0
    train_photo_fraction: float = 1.0
    source_weights: dict[str, float] | None = None
    source: str = "decode"
    cache_root: str | None = None
    buffer_size: int = 1024  # images, not float32 tensors or photo count
    augment: bool = True
    paired_batches: bool = False
    tensor_dtype: str = "uint8"

    def __post_init__(self):
        if not self.sources or len(set(self.sources)) != len(self.sources) or set(self.sources)-{"raise","div2k"}:
            raise ValueError("sources must be a nonempty unique raise/div2k tuple")
        if type(self.tile_size) is not int or self.tile_size not in (256,512) or self.grid not in ("overlap_last","topleft_crop"):
            raise ValueError("invalid tile size/grid")
        if self.grid == "topleft_crop" and (self.tile_size != 512 or self.sources != ("raise",)):
            raise ValueError("top-left training requires RAISE-only 512")
        if type(self.variants_per_tile) is not int or self.variants_per_tile < 1:
            raise ValueError("positive variants_per_tile required")
        if type(self.run_seed) is not int or self.run_seed < 0:
            raise ValueError("run_seed must be a non-negative integer")
        if (type(self.train_photo_fraction) not in (int,float) or
                not 0 < self.train_photo_fraction <= 1 or not math.isfinite(self.train_photo_fraction)):
            raise ValueError("train_photo_fraction must be in (0,1]")
        if any(type(v) is not bool for v in (self.fixed_variants,self.augment,self.paired_batches)):
            raise ValueError("variant/augmentation/pair flags must be Boolean")
        if self.tensor_dtype not in ("uint8", "float32"):
            raise ValueError("tensor_dtype must be uint8 or float32")
        if self.source not in ("decode","cache") or (self.source == "cache" and not self.cache_root):
            raise ValueError("source is decode or cache with cache_root")
        if type(self.buffer_size) is not int or self.buffer_size < 2:
            raise ValueError("buffer_size must be at least two images")
        if self.source_weights is not None:
            if set(self.source_weights)-set(self.sources):
                raise ValueError("weights apply to selected training sources only")
            if any(type(v) not in (int,float) or not math.isfinite(v) or v < 0 for v in self.source_weights.values()):
                raise ValueError("source weights must be finite non-negative multipliers")
            if not any(self.source_weights.get(s,1)>0 for s in self.sources):
                raise ValueError("at least one source weight must be positive")


def _rows(path):
    with Path(path).open(newline="",encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def initialize_worker(worker_id):
    configure_source_worker()
    torch.set_num_threads(1)


def _rng(seed, epoch, domain):
    return np.random.default_rng(np.random.SeedSequence([seed,epoch,domain]))


def image_tensor(array, entropy, label, augment, dtype=torch.float32):
    """CHW 0..255, same keyed flip/flip/rot90 operations at either dtype."""
    image = torch.from_numpy(array).permute(2,0,1).to(dtype=dtype)
    if augment:
        words=np.random.SeedSequence(list(entropy)+[0x415547,label]).generate_state(2)
        generator=torch.Generator().manual_seed(int(words[0]) | (int(words[1])<<32))
        if bool(torch.rand((),generator=generator)<0.5):
            image=torch.flip(image,dims=(-1,))
        if bool(torch.rand((),generator=generator)<0.5):
            image=torch.flip(image,dims=(-2,))
        turns=int(torch.randint(0,4,(),generator=generator).item())
        image=torch.rot90(image,turns,dims=(-2,-1))
    return image.contiguous()


class Corpus:
    """Frozen inventories, selected train photos and one cheap cache audit/session.

    Evaluation always includes both frozen sources; training source selection,
    fraction, weights, and custom YAML cannot change overlap_full_mix.
    """
    def __init__(self, repo_root, settings: RunSettings, *, records=None):
        self.root=Path(repo_root)
        self.settings=settings
        self.photos=(list(records) if records is not None else
                     _rows(self.root/"data/raise/photo_manifest.csv")+_rows(self.root/"data/div2k/photo_manifest.csv"))
        self.by_id={r["photo_id"]:r for r in self.photos}
        if len(self.by_id)!=len(self.photos):
            raise ValueError("duplicate photo identity")
        self.config=read_config(settings.settings_yaml or self.root/"configs/raise512-run1.yaml")
        self.evaluation_config=read_config(settings.evaluation_yaml or self.root/"configs/raise512-run1.yaml")
        self.selected=[]
        self.selection_record=[]
        groups={}
        for photo in self.photos:
            if photo["split"]=="train" and photo["source"] in settings.sources:
                groups.setdefault((photo["source"],photo["camera_model"]),[]).append(photo)
        for key,group in sorted(groups.items()):
            group=sorted(group,key=lambda r:int(r["photo_index"]))
            n=max(1,int(round(len(group)*settings.train_photo_fraction)))
            code=zlib.crc32((key[0]+"/"+key[1]).encode())
            indices=_rng(settings.run_seed,0,code).permutation(len(group))[:n]
            chosen=sorted((group[int(i)] for i in indices),key=lambda r:int(r["photo_index"]))
            self.selected.extend(chosen)
            self.selection_record.append(dict(source=key[0],camera_model=key[1],available=len(group),
                                              selected=len(chosen),photo_ids=[r["photo_id"] for r in chosen]))
        self.selected.sort(key=lambda r:int(r["photo_index"]))
        self.offsets={}
        offset=0
        for record in self.photos:
            if settings.grid=="topleft_crop" and record["source"]!="raise":
                continue
            self.offsets[record["photo_id"]]=offset
            offset+=self.pairs_in_photo(record,settings.grid)
        self._legacy=None
        if settings.source=="cache":
            key=(str(self.root.resolve()),str(Path(settings.cache_root).resolve()),
                 tuple(r["photo_id"] for r in self.photos))
            if key not in _VALIDATED_CACHE_SESSIONS:
                for record in self.photos:
                    mapped=load_cached(record,settings.cache_root,self.root,mmap=True,
                                       verify_hash=False,verify_master_hash=False)
                    del mapped  # header, records, master-stat and shape only; no pixel/hash scan
                _VALIDATED_CACHE_SESSIONS.add(key)

    def photo(self, record):
        if self.settings.source=="decode":
            return load_photo(record,self.root)
        return load_cached(record,self.settings.cache_root,self.root,mmap=False,
                           verify_hash=False,verify_master_hash=False)

    def pairs_in_photo(self,record,grid="overlap_last"):
        ys,xs=grid_axes(record,grid,self.settings.tile_size)
        return len(ys)*len(xs)*self.settings.variants_per_tile

    def train_schedule(self,epoch):
        rng=_rng(self.settings.run_seed,epoch,0x534348)
        if self.settings.source_weights is None:
            schedule=list(self.selected)
        else:
            schedule=[]
            for source in self.settings.sources:
                group=[r for r in self.selected if r["source"]==source]
                n=int(round(len(group)*self.settings.source_weights.get(source,1)))
                if not group:
                    continue
                while n:
                    take=min(n,len(group))
                    schedule.extend(group[int(i)] for i in rng.permutation(len(group))[:take])
                    n-=take
        return [schedule[int(i)] for i in rng.permutation(len(schedule))]

    def legacy_rows(self):
        if self._legacy is None:
            self._legacy=_rows(self.root/"data/raise/legacy_tile_manifest.csv")
        return self._legacy

    @property
    def run_record(self):
        return dict(settings=asdict(self.settings),source=self.settings.source,grid=self.settings.grid,tile_size=self.settings.tile_size,
                    run_seed=self.settings.run_seed,train_photo_fraction=self.settings.train_photo_fraction,
                    train_photo_selection=self.selection_record,source_weights=self.settings.source_weights,
                    variants_per_tile=self.settings.variants_per_tile,fixed_variants=self.settings.fixed_variants,
                    buffer_images=self.settings.buffer_size,
                    buffer_payload_bytes=self.settings.buffer_size*self.settings.tile_size**2*3)


class TrainingImages(IterableDataset):
    """Shard photo schedule, then bounded uint8 shuffle; keys never use workers."""
    def __init__(self, corpus: Corpus):
        super().__init__()
        self.corpus=corpus
        self._epoch=torch.zeros(1,dtype=torch.int64).share_memory_()

    def set_epoch(self,epoch):
        if type(epoch) is not int or epoch<0:
            raise ValueError("epoch must be a non-negative integer")
        self._epoch[0]=epoch  # call before starting next iterator, not mid-iteration

    def __len__(self):
        count=sum(self.corpus.pairs_in_photo(r,self.corpus.settings.grid)
                  for r in self.corpus.train_schedule(int(self._epoch.item())))
        return count if self.corpus.settings.paired_batches else count*2

    def batch_count(self,workers,batch_size):
        schedule=self.corpus.train_schedule(int(self._epoch.item()))
        n=max(1,workers)
        factor=1 if self.corpus.settings.paired_batches else 2
        return sum(math.ceil(sum(self.corpus.pairs_in_photo(r,self.corpus.settings.grid)*factor
                                 for r in schedule[w::n])/batch_size) for w in range(n))

    def _items(self,schedule,epoch):
        settings=self.corpus.settings
        for record in schedule:
            photo=self.corpus.photo(record)
            ys,xs=grid_axes(record,settings.grid,settings.tile_size)
            for row in range(len(ys)):
                for col in range(len(xs)):
                    for variant in range(1,settings.variants_per_tile+1):
                        spec=TileSpec(settings.grid,settings.tile_size,row,col)
                        seed=OTFSeed(settings.run_seed,"train",epoch,variant,settings.fixed_variants)
                        clean,_=make_sample(photo,record,spec,self.corpus.config,seed,"clean")
                        stego,meta=make_sample(photo,record,spec,self.corpus.config,seed,"stego")
                        pair_id=(self.corpus.offsets[record["photo_id"]]+
                                 (row*len(xs)+col)*settings.variants_per_tile+variant-1)
                        entropy=meta["embedding_seed_key"]
                        if settings.paired_batches:
                            yield clean,stego,pair_id,entropy
                        else:
                            yield clean,0,pair_id*2,entropy
                            yield stego,1,pair_id*2+1,entropy
            del photo

    def _output(self,item):
        dtype=getattr(torch,self.corpus.settings.tensor_dtype)
        if self.corpus.settings.paired_batches:
            clean,stego,index,entropy=item
            return (image_tensor(clean,entropy,0,self.corpus.settings.augment,dtype),
                    image_tensor(stego,entropy,1,self.corpus.settings.augment,dtype),index)
        image,label,index,entropy=item
        return image_tensor(image,entropy,label,self.corpus.settings.augment,dtype),label,index

    def __iter__(self):
        initialize_worker(0)
        epoch=int(self._epoch.item())
        info=get_worker_info()
        worker,n=(0,1) if info is None else (info.id,info.num_workers)
        schedule=self.corpus.train_schedule(epoch)[worker::n]
        rng=_rng(self.corpus.settings.run_seed,epoch,0x425546+worker)
        limit=self.corpus.settings.buffer_size//(2 if self.corpus.settings.paired_batches else 1)
        buffer=[]
        for item in self._items(schedule,epoch):
            if len(buffer)<limit:
                buffer.append(item)
                continue
            index=int(rng.integers(len(buffer)))
            old=buffer[index]
            buffer[index]=item
            yield self._output(old)
        for index in rng.permutation(len(buffer)):
            yield self._output(buffer[int(index)])


class FixedPairs(Dataset):
    """Stable index/frame order, one last-photo working array per worker.

    profile overlap_full_mix: every tile from both frozen sources and run1 mix.
    profile legacy_v6: RAISE run1 keys/rows; purpose all exposes all 49,796 pairs.
    """
    def __init__(self,corpus: Corpus,purpose="val",profile="overlap_full_mix",photo_ids=None):
        if purpose not in ("train","val","test","all") or profile not in ("overlap_full_mix","legacy_v6"):
            raise ValueError("invalid purpose/profile")
        if profile=="overlap_full_mix" and purpose=="all":
            raise ValueError("overlap evaluation requires an explicit purpose")
        if profile=="legacy_v6" and (corpus.settings.tile_size!=512 or corpus.settings.variants_per_tile!=1):
            raise ValueError("legacy profile is exactly 512, one variant")
        self.corpus,self.purpose,self.profile=corpus,purpose,profile
        self._last_id=None
        self._last_photo=None
        self.entries=[]
        table=[]
        placeholder=np.zeros((corpus.settings.tile_size,corpus.settings.tile_size,3),dtype=np.uint8)
        if profile=="legacy_v6":
            for original in corpus.legacy_rows():
                record=corpus.by_id.get(original["photo_id"])
                if (record is not None and (purpose=="all" or record["split"]==purpose) and
                        (photo_ids is None or record["photo_id"] in photo_ids)):
                    self.entries.append((record,int(original["tile_row"]),int(original["tile_col"]),1,original))
        else:
            for record in corpus.photos:
                if record["split"]!=purpose or (photo_ids is not None and record["photo_id"] not in photo_ids):
                    continue
                ys,xs=grid_axes(record,"overlap_last",corpus.settings.tile_size)
                for row in range(len(ys)):
                    for col in range(len(xs)):
                        for variant in range(1,corpus.settings.variants_per_tile+1):
                            self.entries.append((record,row,col,variant,None))
        for record,row,col,variant,original in self.entries:
            grid="topleft_crop" if original is not None else "overlap_last"
            ys,xs=grid_axes(record,grid,corpus.settings.tile_size)
            if original is not None:
                sequence=legacy_seed_sequence(corpus.evaluation_config.seed,int(original["cover_index"]),1,0)
                kwargs,meta=draw_legacy_settings(corpus.evaluation_config,placeholder,np.random.default_rng(sequence))
                payload_bytes=(kwargs.get("payload_bytes") if "payload_bytes" in kwargs else
                               _bpp_payload_bytes(kwargs["bpp"],corpus.settings.tile_size**2))
                effective=payload_bytes*8/(corpus.settings.tile_size**2)
                requested=float(meta["size_value"]) if meta["size_kind"]=="bpp" else effective
                clipped=False
                identity=original["cover_id"]
            else:
                sequence=otf_seed_sequence(run_seed=corpus.settings.run_seed,purpose=purpose,epoch=0,
                    photo_index=int(record["photo_index"]),grid_policy=grid,tile_size=corpus.settings.tile_size,
                    tile_row=row,tile_col=col,variant=variant,stream="settings",
                    fixed_variants=corpus.settings.fixed_variants)
                kwargs,meta=draw_otf_settings(corpus.evaluation_config,placeholder,np.random.default_rng(sequence))
                requested,effective,clipped=meta["requested_bpp"],meta["effective_bpp"],meta["clipped"]
                identity=f"{record['photo_id']}:{grid}:{corpus.settings.tile_size}:{row}:{col}:v{variant}"
            table.append(dict(cover_id=identity,pair_id=identity,source_photo=record["photo_id"],
                camera_model=record["camera_model"] or "not_applicable",tile_row=row,tile_col=col,
                k=kwargs["k"],bit_mode=kwargs["bit_mode"],method=kwargs["method"],placement=kwargs["placement"],
                payload_kind=kwargs["payload"],stride=kwargs.get("stride"),bpp_actual=effective,photo_index=int(record["photo_index"]),
                source=record["source"],grid=grid,tile_size=corpus.settings.tile_size,y=ys[row],x=xs[col],
                requested_bpp=requested,effective_bpp=effective,clipped=clipped,variant=variant,split=record["split"]))
        self.frame=pd.DataFrame(table)
        if not self.frame.empty and self.frame["cover_id"].duplicated().any():
            raise ValueError("pair identities must be unique")

    def __len__(self):
        return len(self.entries)

    def __getstate__(self):
        state=self.__dict__.copy()
        state.update(_last_id=None,_last_photo=None)
        return state

    def __getitem__(self,index):
        record,row,col,variant,original=self.entries[index]
        if self._last_id!=record["photo_id"]:
            self._last_photo=self.corpus.photo(record)
            self._last_id=record["photo_id"]
        if original is not None:
            clean,cm=replay_legacy(self._last_photo,record,original,self.corpus.evaluation_config,"clean")
            stego,sm=replay_legacy(self._last_photo,record,original,self.corpus.evaluation_config,"stego")
        else:
            spec=TileSpec("overlap_last",self.corpus.settings.tile_size,row,col)
            seed=OTFSeed(self.corpus.settings.run_seed,self.purpose,0,variant,self.corpus.settings.fixed_variants)
            clean,cm=make_sample(self._last_photo,record,spec,self.corpus.evaluation_config,seed,"clean")
            stego,sm=make_sample(self._last_photo,record,spec,self.corpus.evaluation_config,seed,"stego")
        dtype=getattr(torch,self.corpus.settings.tensor_dtype)
        return image_tensor(clean,cm["embedding_seed_key"],0,False,dtype),image_tensor(stego,sm["embedding_seed_key"],1,False,dtype),index


def prefetch_for_images(images_per_task,prefetch_images=128):
    """Round the per-worker image lead up to whole loader tasks."""
    if images_per_task<=0 or prefetch_images<=0:
        raise ValueError("task images and prefetch images must be positive")
    return (prefetch_images+images_per_task-1)//images_per_task


def training_loader(dataset,batch_size=32,num_workers=8,pin_memory=True,persistent_workers=True,
                    prefetch_factor=None,in_order=True,group_images=4,prefetch_images=128):
    """No shuffle=True; retain all worker tails. Paired batch size is pairs."""
    if batch_size<=0 or group_images<0:
        raise ValueError("batch size must be positive and group images nonnegative")
    images_per_item=2 if dataset.corpus.settings.paired_batches else 1
    group_items=max(1,group_images//images_per_item) if group_images else batch_size
    task_images=min(group_items,batch_size)*images_per_item
    derived=prefetch_for_images(task_images,prefetch_images)
    prefetch_factor=derived if prefetch_factor is None else prefetch_factor
    if prefetch_factor<=0:
        raise ValueError("prefetch factor must be positive")
    kwargs=dict(batch_size=min(group_items,batch_size),num_workers=num_workers,pin_memory=pin_memory,
                persistent_workers=persistent_workers and num_workers>0,worker_init_fn=initialize_worker,
                drop_last=False,in_order=in_order)
    if num_workers:
        kwargs.update(multiprocessing_context="spawn",prefetch_factor=prefetch_factor)
    loader=DataLoader(dataset,**kwargs)
    return MixedLoader(loader,batch_size) if group_images else loader


class PairSubset(Dataset):
    """Fixed pair/frame indices for QUICK and diagnostic subsets."""
    def __init__(self,dataset,indices):
        self.dataset=dataset
        self.indices=list(map(int,indices))
        self.frame=dataset.frame.iloc[self.indices].reset_index(drop=True)
    def __len__(self):
        return len(self.indices)
    def __getitem__(self,index):
        clean,stego,_=self.dataset[self.indices[index]]
        return clean,stego,index


class PhotoPairTasks(Dataset):
    """One evaluation task per photo, regardless of pair table order.

    Return local pair indices so the parent can restore even a stratified
    PairSubset's original, non-photo-major order without rereading photos.
    """
    def __init__(self,dataset):
        self.dataset=dataset
        base=dataset.dataset if isinstance(dataset,PairSubset) else dataset
        if not isinstance(base,FixedPairs):
            raise TypeError("photo-owned evaluation needs FixedPairs or PairSubset")
        groups={}
        for local in range(len(dataset)):
            original=dataset.indices[local] if isinstance(dataset,PairSubset) else local
            photo_id=base.entries[original][0]["photo_id"]
            groups.setdefault(photo_id,[]).append(local)
        self.groups=list(groups.values())
        self.max_images=max((2*len(group) for group in self.groups),default=2)

    def __len__(self):
        return len(self.groups)

    def __getitem__(self,index):
        indices=self.groups[index]
        pairs=[self.dataset[local] for local in indices]
        return (torch.tensor(indices,dtype=torch.int64),
                torch.stack([pair[0] for pair in pairs]),
                torch.stack([pair[1] for pair in pairs]))


class OrderedPairLoader:
    """Restore pair order and old batch boundaries after photo-owned tasks.

    With persistent_workers=True, keep workers after a complete pass. Close
    them after an interrupted pass, including a Lightning sanity check.
    """
    def __init__(self,dataset,batch_size,num_workers=4,pin_memory=True,prefetch_images=128,
                 persistent_workers=False):
        if batch_size<=0 or num_workers<0:
            raise ValueError("batch size must be positive and workers nonnegative")
        tasks=PhotoPairTasks(dataset)
        prefetch=prefetch_for_images(tasks.max_images,prefetch_images)
        kwargs=dict(batch_size=None,num_workers=num_workers,pin_memory=pin_memory,
                    persistent_workers=persistent_workers and num_workers>0,worker_init_fn=initialize_worker)
        if num_workers:
            kwargs.update(multiprocessing_context="spawn",prefetch_factor=prefetch)
        self.loader=DataLoader(tasks,**kwargs)
        self.dataset=dataset
        self.batch_size=batch_size
        self.pin_memory=pin_memory
        self.prefetch_factor=prefetch
        self.num_workers=num_workers
        self.persistent_workers=persistent_workers and num_workers>0

    def shutdown(self):
        iterator=getattr(self.loader,"_iterator",None)
        if iterator is not None:
            try:
                iterator._shutdown_workers()
            finally:
                self.loader._iterator=None

    def __len__(self):
        return math.ceil(len(self.dataset)/self.batch_size)

    def __iter__(self):
        iterator=None
        pending={}
        next_index=0
        batch=[]
        try:
            iterator=iter(self.loader)
            for indices,clean,stego in iterator:
                for offset,local in enumerate(indices.tolist()):
                    if local in pending or local<next_index:
                        raise ValueError("duplicate evaluation pair index")
                    pending[local]=(clean[offset],stego[offset])
                while next_index in pending:
                    c,s=pending.pop(next_index)
                    batch.append((c,s,next_index))
                    next_index+=1
                    if len(batch)==self.batch_size:
                        yield self._batch(batch)
                        batch=[]
            if next_index!=len(self.dataset) or pending:
                raise ValueError("evaluation tasks omitted pair indices")
            if batch:
                yield self._batch(batch)
        finally:
            # A completed persistent iterator resets its sampler on the next
            # pass. An incomplete pass must release its workers and restart.
            incomplete=next_index<len(self.dataset) or bool(pending)
            if hasattr(iterator,"_shutdown_workers") and (incomplete or not self.persistent_workers):
                try:
                    if incomplete and self.num_workers:
                        # Stop scheduling new photos, then consume only tasks
                        # already in flight before closing the result queue.
                        iterator._sampler_iter=iter(())
                        for _ in iterator:
                            pass
                finally:
                    iterator._shutdown_workers()
                    if self.persistent_workers:
                        self.loader._iterator=None

    @staticmethod
    def _batch(batch):
        clean=torch.stack([pair[0] for pair in batch])
        stego=torch.stack([pair[1] for pair in batch])
        if batch[0][0].is_pinned():
            clean,stego=clean.pin_memory(),stego.pin_memory()
        return clean,stego,torch.tensor([pair[2] for pair in batch],dtype=torch.int64)


def preload_pairs(subset,workers=0,batch_size=16,prefetch_images=128):
    """Decode a fixed pair subset once per photo, in parallel, keeping its order.

    Returns [(clean, stego, subset_index), ...] with the exact uint8 tensors
    that ``subset[i]`` yields; only the third item is the subset position.
    One worker task owns one photo, so every photo is decoded once. Workers
    exit when the pass ends. With ``workers=0`` it runs in this process.
    """
    loader=OrderedPairLoader(subset,batch_size,workers,False,prefetch_images=prefetch_images)
    pairs=[]
    for clean,stego,indices in loader:
        for offset,index in enumerate(indices.tolist()):
            pairs.append((clean[offset],stego[offset],index))
    if [pair[2] for pair in pairs]!=list(range(len(subset))):
        raise ValueError("preloaded pairs are incomplete or out of order")
    if not workers:
        base=subset.dataset if isinstance(subset,PairSubset) else subset
        base._last_photo=None  # Release the last full-photo array.
        base._last_id=None
    return pairs


class PreloadedPairs(Dataset):
    """Read-only dataset over pairs returned by ``preload_pairs``."""
    def __init__(self,pairs):
        self.pairs=pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self,index):
        return self.pairs[index]


class PrimedTrainingLoader:
    """Prime persistent training workers during validation, with unchanged keys.

    Close only a fully consumed epoch iterator, then set the shared epoch and
    reset the *same* DataLoader. No worker spawn or second shuffle buffer is
    needed. The first batch is kept until Lightning requests the next epoch.
    """
    def __init__(self,loader):
        if not isinstance(loader,MixedLoader):
            raise TypeError("priming requires a mixed training loader")
        self._active=loader
        self._future=None
        self._executor=None
        self.dataset=loader.dataset
        self.batch_size=loader.batch_size
        self.pin_memory=loader.pin_memory

    @property
    def loader(self):
        return self._active.loader

    def __len__(self):
        return len(self._active)

    def prime_next(self,epoch):
        if self._future is not None:
            raise RuntimeError("the next epoch is already being primed")
        if self._active._producer is not None:
            if self._active._batches_yielded!=len(self._active):
                raise RuntimeError("cannot prime while the current epoch is running")
            # Lightning may stop at len(loader) without asking for StopIteration.
            # Finish the old generator's natural cleanup, keeping workers alive.
            previous=self._active._current_iterator() if self._active._current_iterator else None
            if previous is None:
                raise RuntimeError("completed epoch iterator disappeared before priming")
            previous.close()
        self.dataset.set_epoch(epoch)
        self._executor=ThreadPoolExecutor(max_workers=1,thread_name_prefix="prime-train")

        def start():
            iterator=iter(self._active)
            try:
                return iterator,next(iterator)
            except BaseException:
                iterator.close()
                raise
        self._future=self._executor.submit(start)

    def __iter__(self):
        if self._future is None:
            return iter(self._active)
        try:
            iterator,first=self._future.result()
        except BaseException:
            self.shutdown()
            raise
        self._executor.shutdown(wait=True)
        self._executor=None
        self._future=None

        def stream():
            try:
                yield first
                yield from iterator
            finally:
                iterator.close()
        return stream()

    def shutdown(self):
        try:
            if self._future is not None:
                try:
                    iterator,_=self._future.result()
                    iterator.close()
                finally:
                    self._executor.shutdown(wait=True)
                    self._executor=None
                    self._future=None
        finally:
            self._active.shutdown()


class MixedLoader:
    """Join and pin ordered groups in a bounded producer thread.

    The iterator owns its stop signal. Dropping it releases the producer and
    workers even when the producer is waiting for queue space. Normal epochs
    keep persistent workers; set_epoch updates their shared epoch value.
    """
    def __init__(self,loader,batch_size):
        self.loader=loader
        self.dataset=loader.dataset
        self.batch_size=batch_size
        self.pin_memory=loader.pin_memory
        self._producer=None
        self._batches_yielded=0
        self._current_iterator=None

    def __len__(self):
        return math.ceil(len(self.dataset)/self.batch_size)

    def _joined_batches(self,stop):
        pending=[]
        count=0
        for group in self.loader:
            if stop.is_set():
                break
            start=0
            while start<len(group[-1]):
                if stop.is_set():
                    return
                take=min(self.batch_size-count,len(group[-1])-start)
                pending.append(tuple(part[start:start+take] for part in group))
                count+=take
                start+=take
                if count==self.batch_size:
                    yield self._join(pending)
                    pending,count=[],0
        if pending and not stop.is_set():
            yield self._join(pending)

    def __iter__(self):
        iterator=self._iterate()
        self._current_iterator=weakref.ref(iterator)
        return iterator

    def _iterate(self):
        if self._producer is not None:
            raise RuntimeError("the preceding mixed iterator is still active")
        self._batches_yielded=0
        stop=threading.Event()
        ready=queue.Queue(maxsize=2)

        def offer(item):
            while not stop.is_set():
                try:
                    ready.put(item,timeout=0.1)
                    return True
                except queue.Full:
                    pass
            return False

        def produce():
            failure=None
            completed=False
            try:
                for batch in self._joined_batches(stop):
                    if stop.is_set() or not offer(("batch",batch)):
                        break
                else:
                    completed=not stop.is_set()
            except BaseException as error:
                failure=(error,error.__traceback__)
            finally:
                if not completed or failure is not None:
                    try:
                        self._stop_workers()
                    except BaseException as error:
                        if failure is None:
                            failure=(error,error.__traceback__)
                offer(("done",failure))

        thread=threading.Thread(target=produce,name="mixed-loader-join",daemon=False)
        state=(stop,thread)
        self._producer=state
        thread.start()
        batches=0
        failed=False
        try:
            while True:
                try:
                    kind,value=ready.get(timeout=0.1)
                except queue.Empty:
                    if thread.is_alive():
                        continue
                    # It may have put its last result just after the timeout.
                    try:
                        kind,value=ready.get_nowait()
                    except queue.Empty:
                        raise RuntimeError("mixed loader producer stopped without a result")
                if kind=="done":
                    if value is not None:
                        error,traceback=value
                        failed=True
                        raise error.with_traceback(traceback)
                    break
                batches+=1
                self._batches_yielded=batches
                yield value
        finally:
            natural=batches==len(self) and not failed
            if not natural:
                stop.set()
            thread.join(timeout=30)
            if thread.is_alive():
                stop.set()
                thread.join(timeout=30)
                raise RuntimeError("mixed loader producer did not stop within 30 seconds")
            try:
                if not natural:
                    self._stop_workers()
            finally:
                if self._producer is state:
                    self._producer=None
                self._current_iterator=None

    def _join(self,groups):
        parts=tuple(torch.cat([group[i] for group in groups]) for i in range(3))
        return tuple(part.pin_memory() for part in parts) if self.pin_memory and groups[0][0].is_pinned() else parts

    def _stop_workers(self):
        iterator=getattr(self.loader,"_iterator",None)
        if iterator is not None:
            try:
                iterator._shutdown_workers()
            finally:
                self.loader._iterator=None

    def shutdown(self):
        if self._producer is not None:
            stop,thread=self._producer
            stop.set()
            thread.join(timeout=30)
            if thread.is_alive():
                raise RuntimeError("mixed loader producer did not stop within 30 seconds")
            self._producer=None
        self._stop_workers()
