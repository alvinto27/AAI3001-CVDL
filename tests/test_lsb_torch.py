"""Loader byte keys, persistent worker epochs, train selection and evaluation."""
import dataclasses
import hashlib
import json
import gc
import multiprocessing
import time
import threading
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"LSB-Dataset-Generator"))
from lsb_torch import (torch,Corpus,RunSettings,TrainingImages,FixedPairs,PairSubset,
                       OrderedPairLoader,MixedLoader,training_loader,prefetch_for_images,
                       preload_pairs,PreloadedPairs,PrimedTrainingLoader)
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
        loader.shutdown()
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
    def test_optional_evaluation_yaml_changes_only_bpp_draws(self):
        import pandas as pd
        original = Path(__file__).resolve().parents[1] / 'configs/raise512-run1.yaml'
        expanded = Path(__file__).resolve().parents[1] / 'configs/test-bpp1.yaml'
        self.assertEqual(original.read_text().replace('bpp: [0.05, 0.5]', 'bpp: [0.05, 1.0]'),
                         expanded.read_text())
        records = [{**r, 'split': 'test'} for r in self.photos]
        default = FixedPairs(self.corpus(RunSettings(tile_size=256), records), 'test').frame
        explicit = FixedPairs(self.corpus(RunSettings(tile_size=256,
                              evaluation_yaml=str(original)), records), 'test').frame
        wider = FixedPairs(self.corpus(RunSettings(tile_size=256,
                             evaluation_yaml=str(expanded)), records), 'test').frame
        pd.testing.assert_frame_equal(default, explicit)
        fields = ['cover_id', 'source_photo', 'k', 'bit_mode', 'method', 'placement',
                  'payload_kind', 'stride', 'source', 'grid', 'tile_size', 'y', 'x']
        pd.testing.assert_frame_equal(default[fields], wider[fields])
        self.assertTrue((default['requested_bpp'] != wider['requested_bpp']).any())
        self.assertTrue(wider['requested_bpp'].max() > .5)

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
                    self.assertEqual(str(left.dtype),"torch.uint8")
                    self.assertGreaterEqual(float(left.min()),0)
                    self.assertLessEqual(float(left.max()),255)
        dataset=TrainingImages(self.corpus(RunSettings(tile_size=256)))
        self.assertEqual(len(dataset),4*9*2)
    def test_uint8_exact_float_path(self):
        s=RunSettings(tile_size=256,buffer_size=8)
        byte_ds=TrainingImages(self.corpus(s))
        float_ds=TrainingImages(self.corpus(dataclasses.replace(s,tensor_dtype="float32")))
        byte_loader=training_loader(byte_ds,batch_size=8,num_workers=0,pin_memory=False)
        float_loader=training_loader(float_ds,batch_size=8,num_workers=0,pin_memory=False)
        for left,right in zip(byte_loader,float_loader):
            self.assertTrue(left[0].float().equal(right[0]))
            self.assertTrue(left[1].equal(right[1]))
            self.assertTrue(left[2].equal(right[2]))
        vals=[{**r,"split":"val"} for r in self.photos]
        a=FixedPairs(self.corpus(s,vals),"val")
        b=FixedPairs(self.corpus(dataclasses.replace(s,tensor_dtype="float32"),vals),"val")
        self.assertTrue(a[0][0].float().equal(b[0][0]))
        self.assertTrue(a[0][1].float().equal(b[0][1]))

    def test_small_groups_mix_workers_without_key_changes(self):
        ds=TrainingImages(self.corpus(RunSettings(tile_size=256,buffer_size=8)))
        worker_by_photo={r["photo_id"]:w for w in range(2)
                         for r in ds.corpus.train_schedule(0)[w::2]}
        mixed=training_loader(ds,batch_size=8,num_workers=2,pin_memory=False,group_images=4)
        iterator=iter(mixed)
        first=next(iterator)
        workers={worker_by_photo[f"example{int(i)//18}"] for i in first[2]}
        self.assertEqual(workers,{0,1})
        for _ in iterator:
            pass
        mixed.shutdown()
        raw=training_loader(ds,batch_size=8,num_workers=2,pin_memory=False,group_images=0)
        expected=hashes(raw)
        raw._iterator._shutdown_workers()
        actual=hashes(training_loader(ds,batch_size=8,num_workers=2,pin_memory=False,group_images=4))
        self.assertEqual(expected,actual)
        self.assertEqual(len(actual),len(ds))
        self.assertEqual(len(mixed),9)

    def test_prime_next_epoch_preserves_three_streams_and_releases_workers(self):
        settings=RunSettings(tile_size=256,buffer_size=8)
        corpus=self.corpus(settings)
        baseline=TrainingImages(corpus)
        old=training_loader(baseline,batch_size=8,num_workers=2,pin_memory=False)
        primed_dataset=TrainingImages(corpus)
        prime=PrimedTrainingLoader(training_loader(primed_dataset,batch_size=8,
                                                  num_workers=2,pin_memory=False))
        children={p.pid for p in multiprocessing.active_children()}
        worker_pids=None
        try:
            for epoch in range(3):
                baseline.set_epoch(epoch)
                expected=[(tuple(t.numpy().tobytes() for t in batch)) for batch in old]
                actual=[(tuple(t.numpy().tobytes() for t in batch)) for batch in prime]
                self.assertEqual(actual,expected)
                current_pids=[p.pid for p in prime.loader._iterator._workers]
                if worker_pids is not None:
                    self.assertEqual(current_pids,worker_pids)  # No respawn at epoch boundaries.
                worker_pids=current_pids
                if epoch<2:
                    prime.prime_next(epoch+1)
            old.shutdown()
            prime.shutdown()
            current=[p for p in multiprocessing.active_children() if p.pid not in children]
            self.assertEqual(current,[])
        finally:
            old.shutdown()
            prime.shutdown()
        # A resumed loader started at epoch 2 must yield exactly the same stream.
        resumed=TrainingImages(corpus)
        resumed.set_epoch(2)
        last=training_loader(resumed,batch_size=8,num_workers=2,pin_memory=False)
        try:
            self.assertEqual(actual,[tuple(t.numpy().tobytes() for t in batch) for batch in last])
        finally:
            last.shutdown()

    def test_prime_resume_from_last_checkpoint_matches_uninterrupted_stream(self):
        import lightning as L
        from lightning.pytorch.callbacks import ModelCheckpoint
        corpus=self.corpus(RunSettings(tile_size=256,buffer_size=8))
        seen=[]

        class Probe(L.LightningModule):
            def __init__(self):
                super().__init__()
                self.weight=torch.nn.Parameter(torch.ones(()))
            def training_step(self,batch,batch_idx):
                image,label,index=batch
                digest=hashlib.sha256()
                for part in batch:
                    digest.update(part.numpy().tobytes())
                seen.append((int(self.trainer.current_epoch),digest.hexdigest()))
                return self.weight*image[:,0,0,0].float().mean()
            def validation_step(self,batch,batch_idx):
                return None
            def configure_optimizers(self):
                return torch.optim.SGD(self.parameters(),lr=0.00001)

        def fit(max_epochs,folder,ckpt_path=None):
            dataset=TrainingImages(corpus)
            loader=PrimedTrainingLoader(training_loader(dataset,batch_size=8,
                                              num_workers=2,pin_memory=False))
            class Epoch(L.Callback):
                def on_fit_start(self,trainer,pl_module):
                    dataset.set_epoch(int(trainer.current_epoch))
                def on_train_epoch_start(self,trainer,pl_module):
                    dataset.set_epoch(int(trainer.current_epoch))
                def on_validation_start(self,trainer,pl_module):
                    if not trainer.sanity_checking and trainer.current_epoch+1<trainer.max_epochs:
                        loader.prime_next(int(trainer.current_epoch)+1)
            checkpoint=ModelCheckpoint(dirpath=str(folder),save_last=True,save_top_k=0)
            trainer=L.Trainer(max_epochs=max_epochs,accelerator="cpu",devices=1,
                logger=False,enable_progress_bar=False,enable_model_summary=False,
                num_sanity_val_steps=0,callbacks=[Epoch(),checkpoint],
                default_root_dir=str(folder))
            val=torch.utils.data.DataLoader(torch.utils.data.TensorDataset(torch.zeros(1,1)),batch_size=1)
            try:
                trainer.fit(Probe(),loader,val,ckpt_path=ckpt_path)
            finally:
                loader.shutdown()
            return checkpoint.last_model_path

        reference=fit(3,self.root/"uninterrupted")
        expected=seen[:]
        seen.clear()
        first=fit(1,self.root/"interrupted")
        self.assertTrue(Path(first).is_file())
        fit(3,self.root/"resumed",ckpt_path=first)
        self.assertEqual(seen,expected)

    def test_prime_rejects_running_epoch_and_shutdown_cleans_future(self):
        dataset=TrainingImages(self.corpus(RunSettings(tile_size=256,buffer_size=8)))
        prime=PrimedTrainingLoader(training_loader(dataset,batch_size=8,num_workers=2,pin_memory=False))
        children={p.pid for p in multiprocessing.active_children()}
        try:
            current=iter(prime)
            next(current)
            with self.assertRaisesRegex(RuntimeError,"current epoch is running"):
                prime.prime_next(1)
            current.close()
            prime.prime_next(1)
            prime.shutdown()
            self.assertFalse([p for p in multiprocessing.active_children() if p.pid not in children])
        finally:
            prime.shutdown()

    def test_notebook_fixed_subset_and_complete_bins(self):
        from lsb_notebook import fixed_subset,complete_bpp_bins
        vals=[{**r,"split":"val"} for r in self.photos]
        pairs=FixedPairs(self.corpus(RunSettings(tile_size=256),vals),"val")
        a=fixed_subset(pairs,8,0)
        b=fixed_subset(pairs,8,0)
        self.assertTrue(a.frame.equals(b.frame))
        self.assertEqual(len(a),8)
        self.assertEqual(a[0][2],0)
        bins=complete_bpp_bins([.00001,.0499878,.05,.15,.30,.50,.60])
        self.assertFalse(bins.isna().any())
        self.assertEqual(list(bins.value_counts()),[3,1,1,1,1])

    def test_prefetch_image_budget_and_explicit_override(self):
        self.assertEqual(prefetch_for_images(4),32)
        self.assertEqual(prefetch_for_images(32),4)
        self.assertEqual(prefetch_for_images(3),43)
        for images,lead in ((0,128),(4,0),(4,-1)):
            with self.assertRaises(ValueError):
                prefetch_for_images(images,lead)
        ds=TrainingImages(self.corpus(RunSettings(tile_size=256,buffer_size=8)))
        default=training_loader(ds,num_workers=2,pin_memory=False)
        self.assertEqual(default.loader.prefetch_factor,32)
        explicit=training_loader(ds,num_workers=2,pin_memory=False,prefetch_factor=4)
        self.assertEqual(explicit.loader.prefetch_factor,4)
        whole=training_loader(ds,num_workers=2,pin_memory=False,group_images=0)
        self.assertEqual(whole.prefetch_factor,4)
        paired=TrainingImages(self.corpus(RunSettings(tile_size=256,buffer_size=8,paired_batches=True)))
        tiny=training_loader(paired,batch_size=1,num_workers=2,pin_memory=False)
        self.assertEqual(tiny.loader.prefetch_factor,64)  # One pair = two images.
        with self.assertRaises(ValueError):
            training_loader(ds,num_workers=0,prefetch_factor=0)

    def test_notebook_separate_evaluation_workers_and_record(self):
        from lsb_notebook import NotebookData
        adapter=NotebookData.__new__(NotebookData)
        adapter.corpus=self.corpus(RunSettings(tile_size=256,buffer_size=8))
        adapter.train=TrainingImages(adapter.corpus)
        adapter.fixed_train=FixedPairs(adapter.corpus,'train')
        adapter.val=FixedPairs(adapter.corpus,'val')
        adapter.test=FixedPairs(adapter.corpus,'test')
        adapter.record={}
        train,*evaluation=adapter.loaders(32,4,False)
        self.assertEqual(train.loader.num_workers,4)
        self.assertEqual(train.loader.prefetch_factor,32)
        self.assertTrue(train.loader.persistent_workers)
        for loader in evaluation:
            self.assertEqual(loader.num_workers,4)
            self.assertEqual(loader.prefetch_factor,prefetch_for_images(loader.loader.dataset.max_images))
            self.assertFalse(loader.persistent_workers)
        self.assertEqual(adapter.record['eval_workers'],4)
        self.assertEqual(adapter.record['prefetch_images'],128)
        self.assertEqual(adapter.record['train_prefetch_images'],128)
        self.assertIsNone(adapter.record['prefetch_override'])
        train,*evaluation=adapter.loaders(8,0,False,4,eval_workers=0,prefetch_images=64)
        self.assertEqual(adapter.record['prefetch_factor'],4)
        self.assertEqual(adapter.record['prefetch_override'],4)
        self.assertEqual(adapter.record['eval_prefetch_factor'],32)  # Empty validation, one pair fallback.
        self.assertTrue(all(loader.num_workers==0 for loader in evaluation))
        with self.assertRaises(ValueError):
            adapter.loaders(32,4,False,eval_workers=-1)

    def test_photo_owned_evaluation_order_batches_and_reads(self):
        from unittest.mock import patch
        rows=[{**r,"split":"val"} for r in self.photos]
        ds=FixedPairs(self.corpus(RunSettings(tile_size=256),rows),"val")
        # Stratified order revisits photos. Every photo must still be read once.
        selections=[0,9,18,27,1,10,19,28,2,11,20,29,3]
        for subset in (ds,PairSubset(ds,selections)):
            old=list(torch.utils.data.DataLoader(subset,batch_size=4,num_workers=0))
            old_bytes=[(c.numpy().tobytes(),s.numpy().tobytes(),idx.tolist()) for c,s,idx in old]
            for workers in (0,2,4):
                loader=OrderedPairLoader(subset,4,workers,False,prefetch_images=16)
                with self.subTest(subset=len(subset),workers=workers):
                    got=[(c.numpy().tobytes(),s.numpy().tobytes(),idx.tolist()) for c,s,idx in loader]
                    self.assertEqual(got,old_bytes)
                    self.assertEqual(len(loader),len(old))
                    self.assertEqual(len(loader.loader.dataset.groups),4)
            with patch.object(ds.corpus,'photo',wraps=ds.corpus.photo) as loads:
                list(OrderedPairLoader(subset,4,0,False))
                self.assertEqual(loads.call_count,4)

    def test_preload_pairs_matches_serial_reads_in_order(self):
        from unittest.mock import patch
        rows=[{**r,"split":"val"} for r in self.photos]
        ds=FixedPairs(self.corpus(RunSettings(tile_size=256),rows),"val")
        selections=[0,9,18,27,1,10,19,28,2,11,20,29,3]
        for subset in (ds,PairSubset(ds,selections)):
            serial=[(subset[i][0].numpy().tobytes(),subset[i][1].numpy().tobytes(),i) for i in range(len(subset))]
            for workers in (0,2,3):
                with self.subTest(pairs=len(subset),workers=workers):
                    pairs=preload_pairs(subset,workers,batch_size=5,prefetch_images=16)
                    got=[(c.numpy().tobytes(),s.numpy().tobytes(),i) for c,s,i in pairs]
                    self.assertEqual(got,serial)
                    self.assertTrue(all(c.dtype==torch.uint8 and s.dtype==torch.uint8 for c,s,_ in pairs))
                    dataset=PreloadedPairs(pairs)
                    self.assertEqual(len(dataset),len(subset))
                    self.assertIs(dataset[3],pairs[3])
        with patch.object(ds.corpus,'photo',wraps=ds.corpus.photo) as loads:
            preload_pairs(PairSubset(ds,selections),0)
            self.assertEqual(loads.call_count,4)  # One decode per photo.
        self.assertIsNone(ds._last_photo)

    def test_photo_owned_iterator_drop_and_interrupt_release_workers(self):
        rows=[{**r,"split":"val"} for r in self.photos]
        ds=FixedPairs(self.corpus(RunSettings(tile_size=256),rows),"val")
        loader=OrderedPairLoader(ds,4,2,False)
        before={p.pid for p in multiprocessing.active_children()}
        iterator=iter(loader)
        next(iterator)
        children=[p for p in multiprocessing.active_children() if p.pid not in before]
        self.assertEqual(len(children),2)
        start=time.perf_counter()
        del iterator
        gc.collect()
        for child in children:
            child.join(timeout=10)
            self.assertFalse(child.is_alive())
        self.assertLess(time.perf_counter()-start,12)
        try:
            for _ in loader:
                raise KeyboardInterrupt
        except KeyboardInterrupt:
            pass
        gc.collect()
        self.assertEqual({p.pid for p in multiprocessing.active_children()},before)

    def test_background_join_drop_shutdown_epoch_and_error(self):
        ds=TrainingImages(self.corpus(RunSettings(tile_size=256,buffer_size=8)))
        for workers,pin in ((0,False),(2,False),(0,True),(2,True),(4,True)):
            loader=training_loader(ds,batch_size=8,num_workers=workers,pin_memory=pin)
            before={p.pid for p in multiprocessing.active_children()}
            iterator=iter(loader)
            first=next(iterator)
            self.assertEqual(len(first[-1]),8)
            self.assertEqual(first[0].is_pinned(),pin and torch.cuda.is_available())
            del iterator
            gc.collect()
            self.assertFalse([t for t in threading.enumerate() if t.name=="mixed-loader-join"])
            self.assertEqual({p.pid for p in multiprocessing.active_children()},before)
            ds.set_epoch(1)
            next_epoch=list(loader)
            self.assertEqual(sum(len(batch[-1]) for batch in next_epoch),len(ds))
            self.assertFalse([t for t in threading.enumerate() if t.name=="mixed-loader-join"])
            retained={p.pid for p in multiprocessing.active_children()}-before
            self.assertEqual(len(retained),workers)
            ds.set_epoch(2)
            self.assertEqual(sum(len(batch[-1]) for batch in loader),len(ds))
            self.assertEqual({p.pid for p in multiprocessing.active_children()}-before,retained)
            ds.set_epoch(0)
            iterator=iter(loader)
            next(iterator)
            del iterator
            gc.collect()
            self.assertEqual({p.pid for p in multiprocessing.active_children()},before)
            iterator=iter(loader)
            next(iterator)
            try:
                raise KeyboardInterrupt
            except KeyboardInterrupt:
                del iterator
                gc.collect()
            self.assertEqual({p.pid for p in multiprocessing.active_children()},before)
            iterator=iter(loader)
            next(iterator)
            loader.shutdown()
            successor=iter(loader)
            next(successor)
            del iterator
            gc.collect()
            self.assertIsNotNone(loader._producer)
            del successor
            gc.collect()
            self.assertFalse([t for t in threading.enumerate() if t.name=="mixed-loader-join"])
            self.assertEqual({p.pid for p in multiprocessing.active_children()},before)
        loader=training_loader(ds,batch_size=8,num_workers=2,pin_memory=False)
        def failure(groups):
            raise ValueError("join failed in producer")
        loader._join=failure
        try:
            next(iter(loader))
        except ValueError as error:
            self.assertEqual(str(error),"join failed in producer")
            self.assertIn("failure",''.join(__import__('traceback').format_tb(error.__traceback__)))
        else:
            self.fail("producer error must reach the consumer")
        loader.shutdown()
        self.assertFalse([t for t in threading.enumerate() if t.name=="mixed-loader-join"])
        self.assertFalse(multiprocessing.active_children())

    def test_background_done_between_timeout_and_thread_check(self):
        import queue as standard_queue
        from types import SimpleNamespace
        from unittest.mock import patch
        instances=[]
        class ForcedEmpty(standard_queue.Queue):
            def __init__(self,*args,**kwargs):
                super().__init__(*args,**kwargs)
                self.done=threading.Event()
                self.forced=False
                self.rechecked=False
                instances.append(self)
            def put(self,item,*args,**kwargs):
                super().put(item,*args,**kwargs)
                if item[0]=="done":
                    self.done.set()
            def get(self,block=True,timeout=None):
                if timeout is not None and not self.forced:
                    self.forced=True
                    if not self.done.wait(5):
                        raise AssertionError("producer did not publish done")
                    while any(t.name=="mixed-loader-join" for t in threading.enumerate()):
                        time.sleep(.001)
                    raise standard_queue.Empty
                return super().get(block,timeout)
            def get_nowait(self):
                self.rechecked=True
                return super().get_nowait()
        class EmptyLoader:
            dataset=range(0)
            pin_memory=False
            _iterator=None
            def __iter__(self):
                return iter(())
        with patch('lsb_torch.queue',SimpleNamespace(Queue=ForcedEmpty,Empty=standard_queue.Empty,
                                                     Full=standard_queue.Full)):
            self.assertEqual(list(MixedLoader(EmptyLoader(),8)),[])
        self.assertTrue(instances[0].rechecked)

    def test_ordered_prefetch_equality(self):
        ds=TrainingImages(self.corpus(RunSettings(tile_size=256,buffer_size=8)))
        a=training_loader(ds,batch_size=8,num_workers=2,pin_memory=False,prefetch_factor=4)
        b=training_loader(ds,batch_size=8,num_workers=2,pin_memory=False)
        for left,right in zip(a,b):
            self.assertTrue(all(x.equal(y) for x,y in zip(left,right)))
        a.shutdown();b.shutdown()

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
