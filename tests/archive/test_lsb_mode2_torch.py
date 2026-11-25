"""Mode 2 photo-owned tile and bag loader parity, worker and epoch rules."""
import dataclasses
import hashlib
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "LSB-Dataset-Generator"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "LSB-Dataset-Generator" / "archive"))
from lsb_cache import build_entry
from lsb_mode1 import OTFSeed
from lsb_mode2 import embed_photo,iter_photo_tiles
from lsb_mode2_torch import Mode2Photos,BagLoader,TileLoader,mode2_corpus,PHOTO_LABEL_RULE
from lsb_torch import RunSettings
from tests.test_lsb_source import record


def fingerprint(items):
    return {r["sample_id"]:(int(label),hashlib.sha256(image.numpy().tobytes()).hexdigest(),r)
            for image,label,r in items}


class Mode2LoaderTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.root=Path(self.temp.name)
        (self.root/"configs").mkdir()
        shutil.copyfile(Path(__file__).resolve().parents[2]/"configs/raise512-run1.yaml",
                        self.root/"configs/raise512-run1.yaml")
        self.rows=[]
        for i,purpose in enumerate(["train"]*4+["val"]*2+["test"]*2):
            row={**record(513,649),"photo_id":f"photo{i}","master_path":f"photo{i}.png",
                 "photo_index":str(1000+i),"source":"raise" if i<4 else "div2k",
                 "camera_model":"","split":purpose}
            Image.fromarray(np.random.default_rng(i).integers(0,256,size=(513,649,3),dtype=np.uint8)).save(self.root/row["master_path"])
            self.rows.append(row)

    def tearDown(self):
        self.temp.cleanup()

    def photos(self,purpose="train",source="decode",fixed=False):
        settings=RunSettings(tile_size=256,augment=False,buffer_size=8,
                             source=source,cache_root=str(self.root/"cache") if source=="cache" else None,
                             fixed_variants=fixed)
        return Mode2Photos(mode2_corpus(self.root,settings,records=self.rows),purpose)

    def tiles(self,photos,workers,buffer=0):
        loader=TileLoader(photos,batch_size=7,workers=workers,buffer_images=buffer)
        samples=[]
        for images,labels,keys,records in loader:
            self.assertEqual(images.dtype.name if hasattr(images.dtype,'name') else str(images.dtype),"torch.uint8")
            samples.extend((image,int(label),r) for image,label,r in zip(images,labels,records))
        self.assertEqual(sum(map(sum,(v.values() for v in loader.counts.values()))),len(samples))
        loader.shutdown()
        return samples

    def reference(self,photos):
        results=[]
        for row in photos.schedule():
            cover=photos.corpus.photo(row)
            seed=OTFSeed(photos.corpus.settings.run_seed,photos.purpose,int(photos._epoch.item()),1,
                         photos.corpus.settings.fixed_variants)
            stego,metadata,changed=embed_photo(cover,row,
                photos.corpus.config if photos.purpose=="train" else photos.corpus.evaluation_config,seed)
            for image,side in iter_photo_tiles(cover,stego,row,metadata,changed,256):
                import torch
                results.append((torch.from_numpy(np.ascontiguousarray(image.transpose(2,0,1))),side["label"],side))
        return results

    def test_tiles_match_reference_decode_cache_and_worker_counts(self):
        for row in self.rows:
            build_entry(row,self.root/"cache",self.root)
        for source in ("decode","cache"):
            for purpose in ("train","val","test"):
                photos=self.photos(purpose,source)
                expected=fingerprint(self.reference(photos))
                for workers in (0,2,4):
                    with self.subTest(source=source,purpose=purpose,workers=workers):
                        actual=self.tiles(photos,workers)
                        self.assertEqual(fingerprint(actual),expected)
                        if purpose!="train":
                            self.assertEqual([r["sample_id"] for _,_,r in actual],
                                             [r["sample_id"] for _,_,r in self.reference(photos)])

    def test_one_source_read_and_embedding_per_photo_work_item(self):
        from unittest.mock import patch
        import lsb_mode2_torch as loader_module
        photos=self.photos()
        with patch.object(photos.corpus,"photo",wraps=photos.corpus.photo) as source_read, \
             patch.object(loader_module,"embed_photo",wraps=loader_module.embed_photo) as embed:
            clean,stego=photos[0]
        self.assertEqual(source_read.call_count,1)
        self.assertEqual(embed.call_count,1)
        self.assertEqual(clean.n,stego.n)

    def test_keyed_augmentation_matches_across_workers(self):
        settings=RunSettings(tile_size=256,augment=True,buffer_size=8)
        photos=Mode2Photos(self.corpus_for(settings),"train")
        single=fingerprint(self.tiles(photos,0,buffer=8))
        self.assertEqual(single,fingerprint(self.tiles(photos,2,buffer=8)))
        self.assertEqual(single,fingerprint(self.tiles(photos,4,buffer=8)))

    def corpus_for(self,settings):
        return mode2_corpus(self.root,settings,records=self.rows)

    def test_train_epochs_fixed_evaluation_and_unpadded_bags(self):
        train=self.photos()
        first=fingerprint(self.tiles(train,2,buffer=8))
        train.set_epoch(1)
        second=fingerprint(self.tiles(train,2,buffer=8))
        self.assertEqual(set(first),set(second))
        self.assertTrue(any(first[k][1]!=second[k][1] for k in first if k.endswith(':stego')))
        fixed=self.photos(fixed=True)
        fixed0=fingerprint(self.tiles(fixed,0))
        fixed.set_epoch(2)
        self.assertEqual(fixed0,fingerprint(self.tiles(fixed,2)))
        for purpose in ("val","test"):
            photos=self.photos(purpose)
            before=fingerprint(self.tiles(photos,0))
            photos.set_epoch(5)
            self.assertEqual(before,fingerprint(self.tiles(photos,4)))
        for workers in (0,2,4):
            photos=self.photos("val")
            loader=BagLoader(photos,workers=workers)
            bags=list(loader);loader.shutdown()
            self.assertEqual(len(bags),4)
            for bag in bags:
                self.assertEqual(bag.n,9)
                self.assertEqual(tuple(bag.tiles.shape),(9,3,256,256))
                self.assertEqual(len(bag.records),bag.n)
                self.assertEqual(bag.photo_label_rule,PHOTO_LABEL_RULE)
                self.assertEqual(bag.photo_label,int(bag.tile_labels.any()))
                self.assertEqual({r["photo_id"] for r in bag.records},{bag.photo_id})
                self.assertEqual({r["kind"] for r in bag.records},{"stego" if bag.photo_embedded else "clean"})
                self.assertEqual(bag.photo_actual_changed,any(r["actual_changed"] for r in bag.records))
            if workers==0:
                expected=[(b.photo_id,b.photo_embedded,b.tile_labels.tolist()) for b in bags]
            else:
                self.assertEqual(expected,[(b.photo_id,b.photo_embedded,b.tile_labels.tolist()) for b in bags])
        self.assertEqual(self.photos().n_max(),{"raise":9,"div2k":9})
        loader=BagLoader(self.photos("val"),batch_size=3)
        self.assertEqual([len(group) for group in loader],[3,1])


if __name__=="__main__":
    unittest.main()
