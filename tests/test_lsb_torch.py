"""Loader byte keys, persistent worker epochs, train selection and evaluation."""
import dataclasses
import hashlib
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"LSB-Dataset-Generator"))
from lsb_torch import Corpus,RunSettings,TrainingImages,FixedPairs,training_loader
from lsb_cache import build_entry
from test_lsb_source import record


def hashes(loader):
    result={}
    for image,label,index in loader:
        for a,l,i in zip(image,label,index):
            result[int(i)]=(int(l),hashlib.sha256(a.numpy().tobytes()).hexdigest())
    return result


class TorchTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.root=Path(self.temp.name)
        (self.root/"configs").mkdir()
        shutil.copyfile(Path(__file__).resolve().parents[1]/"configs/raise512-run1.yaml",
                        self.root/"configs/raise512-run1.yaml")
        self.photos=[]
        for i in range(4):
            row={**record(513,649),"photo_id":f"example{i}","master_path":f"example{i}.png",
                 "photo_index":str(1000+i),"camera_model":"","split":"train"}
            pixels=np.random.default_rng(i).integers(0,256,size=(513,649,3),dtype=np.uint8)
            Image.fromarray(pixels).save(self.root/row["master_path"])
            self.photos.append(row)
    def tearDown(self):
        self.temp.cleanup()
    def corpus(self,settings,records=None):
        return Corpus(self.root,settings,records=self.photos if records is None else records)
    def test_worker_count_keyed_augmentation_and_persistent_epochs(self):
        s=RunSettings(tile_size=256,buffer_size=8)
        dataset=TrainingImages(self.corpus(s))
        single=hashes(training_loader(dataset,batch_size=8,num_workers=0,pin_memory=False))
        loader=training_loader(dataset,batch_size=8,num_workers=2,pin_memory=False,persistent_workers=True)
        parallel=hashes(loader)
        self.assertEqual(single,parallel)
        unordered=hashes(training_loader(dataset,batch_size=8,num_workers=2,pin_memory=False,
                                        persistent_workers=False,prefetch_factor=4,in_order=False))
        self.assertEqual(single,unordered)
        dataset.set_epoch(1)
        next_epoch=hashes(loader)
        dataset.set_epoch(1)
        expected=hashes(training_loader(dataset,batch_size=8,num_workers=0,pin_memory=False))
        self.assertEqual(next_epoch,expected)
        self.assertEqual(set(single),set(next_epoch))
        self.assertTrue(all(single[i]!=next_epoch[i] for i in single if i%2==1))
        loader._iterator._shutdown_workers()
    def test_decode_cache_and_fixed_variants(self):
        for row in self.photos:
            build_entry(row,self.root/"cache",self.root)
        s=RunSettings(tile_size=256,buffer_size=8,fixed_variants=True)
        decoded=TrainingImages(self.corpus(s))
        cached=TrainingImages(self.corpus(dataclasses.replace(s,source="cache",cache_root=str(self.root/"cache"))))
        a=hashes(training_loader(decoded,batch_size=8,num_workers=0,pin_memory=False))
        b=hashes(training_loader(cached,batch_size=8,num_workers=0,pin_memory=False))
        self.assertEqual(a,b)
        cached.set_epoch(5)
        self.assertEqual(b,hashes(training_loader(cached,batch_size=8,num_workers=0,pin_memory=False)))
    def test_fraction_stable_and_recorded(self):
        s=RunSettings(train_photo_fraction=.5)
        first=self.corpus(s)
        second=self.corpus(s)
        self.assertEqual(first.run_record,second.run_record)
        self.assertEqual(len(first.selected),2)
        self.assertEqual({r["photo_id"] for r in first.train_schedule(0)},
                         {r["photo_id"] for r in first.train_schedule(10)})
        self.assertEqual(first.selection_record[0]["selected"],2)
    def test_evaluation_keys_frame_contract_and_counts(self):
        required={"cover_id","source_photo","camera_model","tile_row","tile_col","k","bit_mode",
                  "method","placement","payload_kind","bpp_actual","photo_index","source","grid",
                  "tile_size","y","x","requested_bpp","effective_bpp","clipped"}
        for purpose in ("val","test"):
            records=[{**r,"split":purpose} for r in self.photos]
            a=FixedPairs(self.corpus(RunSettings(tile_size=256,run_seed=1),records),purpose)
            b=FixedPairs(self.corpus(RunSettings(tile_size=256,run_seed=999,fixed_variants=True),records),purpose)
            self.assertTrue(required<=set(a.frame.columns))
            self.assertEqual(len(a),4*9)
            self.assertTrue(a.frame.equals(b.frame))
            for i in (0,len(a)-1):
                for left,right in zip(a[i][:2],b[i][:2]):
                    np.testing.assert_array_equal(left.numpy(),right.numpy())
                    self.assertEqual(str(left.dtype),"torch.float32")
                    self.assertGreaterEqual(float(left.min()),0)
                    self.assertLessEqual(float(left.max()),255)
        dataset=TrainingImages(self.corpus(RunSettings(tile_size=256)))
        self.assertEqual(len(dataset),4*9*2)
    def test_source_weight_epoch_length_and_paired_contract(self):
        s=RunSettings(tile_size=256,buffer_size=8,paired_batches=True,source_weights={"div2k":2.0})
        dataset=TrainingImages(self.corpus(s))
        self.assertEqual(len(dataset),4*9*2)
        clean,stego,index=next(iter(training_loader(dataset,batch_size=4,num_workers=0,pin_memory=False)))
        self.assertEqual(tuple(clean.shape),(4,3,256,256))
        self.assertEqual(clean.shape,stego.shape)
        self.assertEqual(len(index),4)


if __name__=="__main__":
    unittest.main()
