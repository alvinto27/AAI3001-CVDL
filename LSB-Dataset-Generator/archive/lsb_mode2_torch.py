"""Mode 2 photo-owned tile stream and unpadded, single-photo bags.

One DataLoader task processes one entire photo. Both clean and embedded tiles
come from one decode and one embedding; the model/batching policy is separate.
"""
from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from lsb_mode1 import OTFSeed
from lsb_mode2 import embed_photo, iter_photo_tiles
from lsb_source import grid_axes
from lsb_torch import Corpus, RunSettings, _rng, image_tensor, initialize_worker

PHOTO_LABEL_RULE = "any_changed_tile"


@dataclass
class PhotoBag:
    """Exactly one photo output; no padding or unrelated photo tiles."""
    tiles: torch.Tensor  # N,C,T,T, uint8
    tile_labels: torch.Tensor  # N, int64
    photo_embedded: bool
    photo_actual_changed: bool
    photo_label: int
    photo_label_rule: str
    photo_id: str
    source: str
    variant: int
    records: list[dict]

    @property
    def n(self):
        return len(self.tile_labels)


def _one_photo(items, photo_id, source, embedded):
    records=[record for _,record in items]
    labels=torch.tensor([record["label"] for record in records],dtype=torch.int64)
    tiles=torch.stack([torch.from_numpy(np.ascontiguousarray(pixels.transpose(2,0,1)))
                       for pixels,_ in items])
    actual=bool(records[0]["photo_actual_changed"])
    return PhotoBag(tiles,labels,embedded,actual,int(labels.any()),PHOTO_LABEL_RULE,
                    photo_id,source,int(records[0]["variant"]),records)


class Mode2Photos(Dataset):
    """Map-style photo tasks: ordered across worker counts, whole-photo work.

    A shared epoch counter lets persistent workers observe set_epoch before the
    next iterator. Each worker caches its schedule once per epoch, not per tile.
    """
    def __init__(self, corpus: Corpus, purpose="train"):
        if purpose not in ("train","val","test"):
            raise ValueError("purpose must be train, val or test")
        settings=corpus.settings
        if settings.grid!="overlap_last" or settings.variants_per_tile!=1 or settings.paired_batches:
            raise ValueError("Mode 2 requires overlap_last, one photo variant and unpaired image output")
        if settings.tensor_dtype!="uint8":
            raise ValueError("Mode 2 keeps tiles uint8 until the model boundary")
        self.corpus,self.purpose=corpus,purpose
        self._epoch=torch.zeros(1,dtype=torch.int64).share_memory_()
        self._local_epoch=None
        self._local_schedule=None
        self._fixed=[r for r in corpus.photos if r["split"]==purpose] if purpose!="train" else None

    def set_epoch(self,epoch):
        if type(epoch) is not int or epoch<0:
            raise ValueError("epoch must be a non-negative integer")
        self._epoch[0]=epoch
        self._local_epoch=None

    def schedule(self):
        if self._fixed is not None:
            return self._fixed
        epoch=int(self._epoch.item())
        if epoch!=self._local_epoch:
            self._local_schedule=self.corpus.train_schedule(epoch)
            self._local_epoch=epoch
        return self._local_schedule

    def __len__(self):
        return len(self.schedule())

    def __getitem__(self,index):
        record=self.schedule()[index]
        photo=self.corpus.photo(record)
        seed=OTFSeed(self.corpus.settings.run_seed,self.purpose,int(self._epoch.item()),1,
                     self.corpus.settings.fixed_variants)
        stego,metadata,changed=embed_photo(photo,record,
            self.corpus.config if self.purpose=="train" else self.corpus.evaluation_config,seed)
        clean_tiles=[]
        embedded_tiles=[]
        for pixels,tile_record in iter_photo_tiles(photo,stego,record,metadata,changed,
                                                    self.corpus.settings.tile_size):
            (embedded_tiles if tile_record["kind"]=="stego" else clean_tiles).append((pixels,tile_record))
        clean=_one_photo(clean_tiles,record["photo_id"],record["source"],False)
        embedded=_one_photo(embedded_tiles,record["photo_id"],record["source"],True)
        del photo,stego,changed
        return clean,embedded

    def n_max(self):
        result={}
        for record in self.corpus.photos:
            ys,xs=grid_axes(record,"overlap_last",self.corpus.settings.tile_size)
            result[record["source"]]=max(result.get(record["source"],0),len(ys)*len(xs))
        return result


def _one_task(items):
    return items[0]


def _photo_loader(photos,workers,prefetch_images,persistent_workers):
    if type(workers) is not int or workers<0 or type(prefetch_images) is not int or prefetch_images<1:
        raise ValueError("workers must be non-negative and prefetch_images positive")
    # A DataLoader task contains two whole-photo outputs. One task per worker
    # is the smallest possible lead; variable real grids never get padded.
    max_images=2*max(photos.n_max().values())
    kwargs=dict(batch_size=1,num_workers=workers,collate_fn=_one_task,
                worker_init_fn=initialize_worker,drop_last=False,in_order=True,
                persistent_workers=bool(workers and persistent_workers))
    if workers:
        kwargs.update(multiprocessing_context="spawn",
                      prefetch_factor=max(1,math.ceil(prefetch_images/max_images)))
    return DataLoader(photos,**kwargs)


class BagLoader:
    """Emit bags separately (batch_size=1) or lists, never padded tensors."""
    def __init__(self,photos,workers=0,batch_size=1,prefetch_images=128,persistent_workers=False):
        if type(batch_size) is not int or batch_size<1:
            raise ValueError("positive bag batch size required")
        self.photos=photos
        self.batch_size=batch_size
        self.loader=_photo_loader(photos,workers,prefetch_images,persistent_workers)

    def set_epoch(self,epoch):
        self.photos.set_epoch(epoch)

    def __len__(self):
        return math.ceil(2*len(self.photos)/self.batch_size)

    def __iter__(self):
        pending=[]
        for clean,embedded in self.loader:
            for bag in (clean,embedded):
                if self.batch_size==1:
                    yield bag
                else:
                    pending.append(bag)
                    if len(pending)==self.batch_size:
                        yield pending
                        pending=[]
        if pending:
            yield pending

    def shutdown(self):
        iterator=getattr(self.loader,"_iterator",None)
        if iterator is not None:
            iterator._shutdown_workers()
            self.loader._iterator=None


class TileLoader:
    """Bounded cross-photo image shuffle; preserve each tile's side record."""
    def __init__(self,photos,batch_size=32,workers=0,prefetch_images=128,
                 buffer_images=None,persistent_workers=False):
        if photos.purpose!="train":
            # Evaluation uses the same deterministic photo order with no shuffle.
            buffer_images=0
        if type(batch_size) is not int or batch_size<1:
            raise ValueError("positive image batch size required")
        self.photos=photos
        self.batch_size=batch_size
        self.buffer_images=photos.corpus.settings.buffer_size if buffer_images is None else buffer_images
        if type(self.buffer_images) is not int or self.buffer_images<0:
            raise ValueError("buffer_images must be non-negative")
        self.loader=_photo_loader(photos,workers,prefetch_images,persistent_workers)
        self.counts={}

    def set_epoch(self,epoch):
        self.photos.set_epoch(epoch)

    def __len__(self):
        size=self.photos.corpus.settings.tile_size
        return math.ceil(sum(2*math.prod(len(axis) for axis in
            (grid_axes(r,"overlap_last",size))) for r in self.photos.schedule())/self.batch_size)

    def _emit(self,item):
        pixels,label,key,record=item
        source=record["source"]
        self.counts.setdefault(source,Counter())[int(label)]+=1
        settings=self.photos.corpus.settings
        if settings.augment and self.photos.purpose=="train":
            # Role, not actual-change label, distinguishes independent transforms.
            role=int(record["kind"]=="stego")
            hwc=pixels.permute(1,2,0).numpy()
            # The photo embedding stream stays unchanged; tile coordinates
            # enter only the separate, keyed augmentation stream.
            entropy=[*record["embedding_seed_key"],record["tile_row"],record["tile_col"],settings.tile_size]
            pixels=image_tensor(hwc,entropy,role,True,torch.uint8)
        return pixels,int(label),key,record

    def __iter__(self):
        self.counts={}
        rng=_rng(self.photos.corpus.settings.run_seed,int(self.photos._epoch.item()),0x425546)
        buffer=[]
        batch=[]
        size=self.photos.corpus.settings.tile_size
        for clean,embedded in self.loader:
            assert clean.n==embedded.n
            record=clean.records[0]
            offset=self.photos.corpus.offsets[record["photo_id"]]
            for tile_index in range(clean.n):
                for bag,role in ((clean,0),(embedded,1)):
                    # Own the tile: a view would keep the whole photo bag alive
                    # for as long as one tile remains in the shuffle buffer.
                    item=(bag.tiles[tile_index].clone(),int(bag.tile_labels[tile_index]),
                          (offset+tile_index)*2+role,bag.records[tile_index])
                    if self.buffer_images and len(buffer)<self.buffer_images:
                        buffer.append(item)
                        continue
                    if self.buffer_images:
                        index=int(rng.integers(len(buffer)))
                        item,buffer[index]=buffer[index],item
                    batch.append(self._emit(item))
                    if len(batch)==self.batch_size:
                        yield self._batch(batch)
                        batch=[]
        for index in rng.permutation(len(buffer)):
            batch.append(self._emit(buffer[int(index)]))
            if len(batch)==self.batch_size:
                yield self._batch(batch)
                batch=[]
        if batch:
            yield self._batch(batch)

    @staticmethod
    def _batch(items):
        return (torch.stack([item[0] for item in items]),
                torch.tensor([item[1] for item in items],dtype=torch.long),
                torch.tensor([item[2] for item in items],dtype=torch.long),
                [item[3] for item in items])

    def shutdown(self):
        iterator=getattr(self.loader,"_iterator",None)
        if iterator is not None:
            iterator._shutdown_workers()
            self.loader._iterator=None


def mode2_corpus(repo_root,settings: RunSettings,*,records=None):
    """Reuse the Mode 1 inventory, source/cache audit and photo schedule."""
    return Corpus(repo_root,settings,records=records)


__all__=["PHOTO_LABEL_RULE","PhotoBag","Mode2Photos","BagLoader","TileLoader","mode2_corpus"]
