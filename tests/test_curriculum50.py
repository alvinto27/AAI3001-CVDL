"""Versioned scratch policy, independent d28e2bb legacy replay, no GPU required."""
from dataclasses import replace
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

import lightning as L
import numpy as np
import torch
from cvdl import runtime
from cvdl.curriculum import (SCRATCH50, SCRATCH50_STAGES, CurriculumImages,
    CurriculumCounts, stage_for_epoch, original_positions, scratch50_record,
    start_identity, check_start, initialize_scratch50)
from cvdl.lit import SteganalysisModule, v4_lr_factor
from cvdl.settings import load, machine_profile, SCHEMA
from lsb_torch import TrainingImages, training_loader, PrimedTrainingLoader, image_tensor
from scripts.g4.preflight import top_values
from test_paired_training import make_corpus, collect

ROOT=Path(__file__).resolve().parents[1]
REFERENCE=ROOT/'tests/fixtures/curriculum-d28e2bb.txt'
legacy=ModuleType('cvdl._curriculum_d28e2bb');legacy.__package__='cvdl'
exec(compile(REFERENCE.read_text(),str(REFERENCE),'exec'),legacy.__dict__)
# Give the frozen class an importable name for spawn-based worker replay.
FrozenCurriculumImages=legacy.CurriculumImages
FrozenCurriculumImages.__module__=__name__
FrozenCurriculumImages.__name__=FrozenCurriculumImages.__qualname__='FrozenCurriculumImages'


def recipe(paired=False):
    name='srm-e50-curriculum50'+('-paired' if paired else '')
    return top_values(json.loads((ROOT/f'notebooks/experiments/{name}.ipynb').read_text()))


def settings(paired=False,**updates):
    values=recipe(paired)
    machine={k:v for k,v in updates.items() if SCHEMA[k].machine}
    values.update({k:v for k,v in updates.items() if not SCHEMA[k].machine})
    return load(environ={},_profile=machine_profile('pc',device='cpu'),overrides=machine,**values)


class Curriculum50Tests(unittest.TestCase):
    def test_all_fifty_epochs_bounds_and_lr_clock(self):
        for epoch in range(50):
            i=0 if epoch<7 else 1 if epoch<20 else 2 if epoch<32 else 3
            got=stage_for_epoch(epoch,mode=SCRATCH50)
            self.assertEqual(got,dict(stage=i+1,bpp_min=.05,bpp_max=(.5,.35,.25,.15)[i],
                k1_weight=(1,1,2,3)[i],original_fraction=1. if i==0 else .5))
            flags=original_positions(180000,0,epoch,SCRATCH50)
            self.assertEqual(int(flags.sum()),180000 if epoch<7 else 90000)
        for bad in (-1,50,True,1.5):
            with self.assertRaises(ValueError):stage_for_epoch(bad,mode=SCRATCH50)
        with self.assertRaises(ValueError):original_positions(7,0,7,SCRATCH50)
        for epoch,factor in ((6,1.),(7,.3),(31,.3),(32,.03),(49,.03)):
            self.assertAlmostEqual(v4_lr_factor(epoch*2813,2813,50,1,((8,.3),(33,.1))),factor)
        flags=original_positions(20,0,7,SCRATCH50)
        self.assertFalse(np.array_equal(flags,original_positions(20,1,7,SCRATCH50)))
        self.assertFalse(np.array_equal(flags,original_positions(20,0,8,SCRATCH50)))
        np.testing.assert_array_equal(flags,original_positions(20,0,7,SCRATCH50))

    def test_settings_separate_scratch_guard_and_exact_notebooks(self):
        base=json.loads((ROOT/'notebooks/experiments/srm-e50.ipynb').read_text())
        for paired in (False,True):
            name='srm-e50-curriculum50'+('-paired' if paired else '')
            expected=json.loads(json.dumps(base));src=expected['cells'][1]['source']
            i=src.index('    name="srm-e50",\n')
            src[i:i+1]=[f'    name="{name}",\n','    embedding_schedule="low_bpp50_v1",\n']+(['    paired_batches=True,\n'] if paired else [])
            self.assertEqual(json.loads((ROOT/f'notebooks/experiments/{name}.ipynb').read_text()),expected)
            s=settings(paired)
            self.assertEqual((s.epochs,s.lr,s.lr_drops,s.paired_batches),(50,3e-4,((8,.3),(33,.1)),paired))
            for absent in ('embedding_stage_epochs','init_checkpoint','init_checkpoint_sha256','evaluation_checkpoint'):
                self.assertNotIn(absent,s.values)
            for kwargs in (dict(epochs=49),dict(embedding_stage_epochs=5),dict(lr=3e-5),
                    dict(lr_drops=()),dict(evaluation_checkpoint='last'),dict(batch_size=64),
                    dict(init_checkpoint='x',init_checkpoint_sha256='0'*64),dict(train_tile_size=512),
                    dict(model='xunet'),dict(source_mix={'raise':1.}),dict(channels_last=True),
                    dict(tiles_per_epoch=180004),dict(checkpoint_validation='full')):
                with self.assertRaises(ValueError,msg=str(kwargs)):settings(paired,**kwargs)
            pilot=settings(paired,pilot=True,batch_size=16,checkpoint_validation='train_matched',
                validation_recipe='60da4391043e8ce48ff2')
            self.assertTrue(pilot.pilot)

    def test_first_seven_epochs_normal_pixels_labels_order_and_rng(self):
        # Same private augmentation implementation (unchanged by this task),
        # independently trace every draw and each global state in the caller.
        for paired in (False,True):
            for workers,pack in ((0,1),(2,4)):
                c=make_corpus(paired)
                ordinary=TrainingImages(c);new=CurriculumImages(c,SCRATCH50,None)
                for epoch in range(7):
                    ordinary.set_epoch(epoch);new.set_epoch(epoch)
                    a,arng=collect(ordinary,workers,pack);b,brng=collect(new,workers,pack)
                    self.assertTrue(torch.equal(a[0],b[0]));self.assertTrue(torch.equal(a[1],b[1]))
                    self.assertTrue(torch.equal(arng,brng))
                    if paired:
                        self.assertTrue(torch.equal(a[2]*2,b[2][:,0,0]))
                        self.assertTrue(torch.equal(a[2]*2+1,b[2][:,1,0]))
                    else:self.assertTrue(torch.equal(a[2],b[2][:,0]))
                    self.assertTrue((b[2][...,1]==1).all())
        # Private augmentation draw trace as well as output parity, both dtypes.
        rand,randint=torch.rand,torch.randint
        def traced(dataset):
            events=[]
            def wrap(fn):
                def call(*args,**kw):
                    generator=kw.get('generator')
                    state=generator.get_state().clone() if generator is not None else None
                    value=fn(*args,**kw)
                    if state is not None:events.append((state,value.clone()))
                    return value
                return call
            torch.manual_seed(42);np.random.seed(42)
            with patch('torch.rand',wrap(rand)),patch('torch.randint',wrap(randint)):
                list(dataset)
            return events,torch.get_rng_state(),np.random.get_state()
        for dtype in ('uint8','float32'):
            c=make_corpus(False,dtype)
            a=traced(TrainingImages(c));b=traced(CurriculumImages(c,SCRATCH50,None))
            self.assertEqual(len(a[0]),len(b[0]))
            for x,y in zip(a[0],b[0]):
                for p,q in zip(x,y):self.assertTrue(torch.equal(p,q))
            self.assertTrue(torch.equal(a[1],b[1]));np.testing.assert_array_equal(a[2][1],b[2][1]);self.assertEqual(a[2][2:],b[2][2:])

    def test_legacy_run10_11_exact_tensors_draws_records_and_identities(self):
        self.assertEqual(hashlib.sha256(REFERENCE.read_bytes()).hexdigest(),
            'd0da1ec46977fecc685c30e9fb718f583fa0225345dcc179f234a6052f977ef8')
        for mode,name in (('low_bpp','srm-low-bpp-curriculum'),('original','srm-low-bpp-control')):
            values=top_values(json.loads((ROOT/f'notebooks/experiments/{name}.ipynb').read_text()))
            s=load(environ={},_profile=machine_profile('pc',device='cpu'),**values)
            self.assertEqual(start_identity(s),legacy.start_identity(s))
            for epoch in range(15):
                self.assertEqual(stage_for_epoch(epoch,5,mode),legacy.stage_for_epoch(epoch,5,mode))
                np.testing.assert_array_equal(original_positions(180000,0,epoch,mode),
                    legacy.original_positions(180000,0,epoch,mode))
            for dtype in ('uint8','float32'):
                c=make_corpus(False,dtype)
                old=legacy.CurriculumImages(c,mode,5);new=CurriculumImages(c,mode,5)
                for workers,pack in ((0,1),(2,4)):
                    for epoch in (0,5,10):
                        old.set_epoch(epoch);new.set_epoch(epoch)
                        a,arng=collect(old,workers,pack);b,brng=collect(new,workers,pack)
                        for x,y in zip(a,b):self.assertTrue(torch.equal(x,y))
                        self.assertTrue(torch.equal(arng,brng))
            with TemporaryDirectory() as folder:
                paths=[Path(folder)/f'{i}.jsonl' for i in range(2)]
                callbacks=[legacy.CurriculumCounts(paths[0],mode,5),CurriculumCounts(paths[1],mode,5)]
                d=CurriculumImages(make_corpus(False),mode,5)
                loader=training_loader(d,8,num_workers=0,pin_memory=False)
                try:
                    for epoch in (0,5,10):
                        d.set_epoch(epoch);t=SimpleNamespace(current_epoch=epoch)
                        for cb in callbacks:cb.on_train_epoch_start(t,None)
                        for i,batch in enumerate(loader):
                            for cb in callbacks:cb.on_train_batch_end(t,None,None,batch,i)
                        for cb in callbacks:cb.on_train_epoch_end(t,None)
                    self.assertEqual(paths[0].read_bytes(),paths[1].read_bytes())
                    self.assertEqual(callbacks[0].state_dict(),callbacks[1].state_dict())
                finally:loader.shutdown()
            fake=lambda *a,**k:SimpleNamespace(corpus=make_corpus(False),record={'frozen':'legacy'})
            with patch('cvdl.runtime.data.build_otf_data',side_effect=fake),patch('cvdl.runtime.data.otf_pair_tables',return_value=None):
                otf,_=runtime.create_data(s)
            self.assertEqual(otf.record,dict(frozen='legacy',embedding_schedule=dict(mode=mode,stage_epochs=5,
                original_fraction=.25 if mode=='low_bpp' else 1.,
                assignment='exact per epoch before worker sharding; counts reflect consumed stegos'),
                compile_mode=s.compile,prime_next_epoch=s.prime_next_epoch))

    def test_intact_pairs_transformed_raw_difference_and_metadata(self):
        c=make_corpus(True);d=CurriculumImages(c,SCRATCH50,None)
        with self.assertRaisesRegex(ValueError,'fixed boundaries'):CurriculumImages(c,SCRATCH50,5)
        for epoch in (0,7,20,32,49):
            groups,counts=c.train_tile_schedule(epoch)
            self.assertEqual(sum(len(tiles) for _,tiles in groups),20)
            self.assertEqual(counts,TrainingImages(c).corpus.train_tile_schedule(epoch)[1])
            for item in d._items_mixed(groups,epoch):
                clean,stego,tags,key=item;a,b,actual_tags=d._output(item)
                expected=image_tensor(stego.astype(np.int16)-clean.astype(np.int16),key,1,True,torch.int16)
                self.assertTrue(torch.equal(b.to(torch.int16)-a.to(torch.int16),expected))
                self.assertTrue(torch.equal(b,image_tensor(stego,key,1,True,torch.uint8)))
                self.assertEqual(tags[0][0]+1,tags[1][0]);self.assertEqual(tags[0][1],tags[1][1])
                self.assertTrue(torch.equal(actual_tags,torch.tensor(tags)))
        with self.assertRaisesRegex(ValueError,'original schedule'):CurriculumImages(c,'low_bpp',5)

    def test_counts_all_stage_boundaries_priming_packing_tails_budget(self):
        for paired in (False,True):
            for workers,pack in ((0,1),(2,1),(2,4)):
                c=make_corpus(paired);d=CurriculumImages(c,SCRATCH50,None)
                loader=PrimedTrainingLoader(training_loader(d,8 if paired else 16,num_workers=workers,pin_memory=False,pack_blocks=pack))
                with TemporaryDirectory() as folder:
                    cb=CurriculumCounts(Path(folder)/'counts.jsonl',SCRATCH50,None)
                    m=SimpleNamespace(hparams={'paired_batches':paired})
                    try:
                        for epoch in (6,7,19,20,31,32,49):
                            if epoch==6:d.set_epoch(epoch)
                            else:loader.prime_next(epoch)
                            t=SimpleNamespace(current_epoch=epoch);cb.on_train_epoch_start(t,m)
                            sizes=[]
                            for i,batch in enumerate(loader):
                                sizes.append(len(batch[0]));cb.on_train_batch_end(t,m,None,batch,i)
                            cb.on_train_epoch_end(t,m)
                            row=json.loads(cb.path.read_text().splitlines()[-1])
                            self.assertEqual(row['consumed_stegos'],20)
                            self.assertEqual(row['original_stegos'],20 if epoch<7 else 10)
                            self.assertEqual(row['stage_stegos'],0 if epoch<7 else 10)
                            self.assertEqual(sizes,[8,8,4] if paired else [16,16,8])
                        restored=CurriculumCounts(cb.path,SCRATCH50,None);restored.load_state_dict(cb.state_dict())
                        self.assertEqual(len(cb.path.read_text().splitlines()),7)
                    finally:loader.shutdown()
            c.settings=replace(c.settings,tiles_per_epoch=180000)
            loader=training_loader(CurriculumImages(c,SCRATCH50,None),64 if paired else 128,num_workers=0,pin_memory=False)
            self.assertEqual(len(loader),2813);self.assertEqual(len(loader.dataset),180000 if paired else 360000)
            self.assertEqual(len(loader.dataset)%loader.batch_size,32 if paired else 64);loader.shutdown()

    def test_record_scratch_identity_no_loading_and_table_change_rejected(self):
        record=scratch50_record(180000)
        self.assertEqual([x['original_draws'] for x in record['stages']],[180000,90000,90000,90000])
        self.assertEqual([x['start'] for x in record['stages']],[0,7,20,32])
        s=settings()
        with TemporaryDirectory() as folder:
            self.assertFalse(check_start(s,folder))
            torch_before=torch.get_rng_state().clone()
            with patch('cvdl.curriculum.weights_only',side_effect=AssertionError('not a weight start')):
                saved=initialize_scratch50(s,folder,False)
            self.assertTrue(torch.equal(torch_before,torch.get_rng_state()));self.assertEqual(saved['initialization'],'random')
            path=Path(folder)/'last.ckpt'
            torch.save(dict(hyper_parameters=dict(embedding_schedule_identity=record)),path)
            self.assertTrue(check_start(s,folder));self.assertEqual(initialize_scratch50(s,folder,True),saved)
            changed=((0,7,.5,1,1.),(7,20,.36,1,.5),*SCRATCH50_STAGES[2:])
            with patch('cvdl.curriculum.SCRATCH50_STAGES',changed):
                with self.assertRaisesRegex(ValueError,'fingerprint'):check_start(s,folder)
            with self.assertRaisesRegex(ValueError,'fingerprint'):check_start(settings(seed=1),folder)
            torch.save(dict(hyper_parameters={}),path)
            with self.assertRaisesRegex(ValueError,'checkpoint schedule identity'):check_start(s,folder)
        with TemporaryDirectory() as folder:
            (Path(folder)/'last.ckpt').write_bytes(b'orphan')
            with self.assertRaisesRegex(ValueError,'identity'):check_start(s,folder)

    def test_shared_runtime_scratch_resume_epoch_optimizer_lr_and_checkpoint_identity(self):
        # Fully consumed tiny CPU epochs: test the real Trainer/checkpoint loop,
        # substituting only the expensive network (not the data/loss/callbacks).
        class Capture(L.Callback):
            def on_train_epoch_start(self,trainer,module):
                self.epoch=trainer.current_epoch;self.step=trainer.global_step
                self.optim_steps=[int(v['step']) for v in trainer.optimizers[0].state.values()]
                self.scheduler=trainer.lr_scheduler_configs[0].scheduler.last_epoch
        original_module=SteganalysisModule
        def tiny_module(**kwargs):
            m=original_module(**kwargs)
            m.net=torch.nn.Sequential(torch.nn.Conv2d(3,2,1),torch.nn.AdaptiveAvgPool2d(1),torch.nn.Flatten())
            return m
        for paired in (False,True):
            with TemporaryDirectory() as folder:
                s=settings(paired,output_root=folder,compile='off',workers=0,eval_workers=0,
                    persistent_workers=False,prime_next_epoch=False)
                c=make_corpus(paired);dataset=CurriculumImages(c,SCRATCH50,None)
                train=training_loader(dataset,20 if paired else 40,num_workers=0,pin_memory=False)
                images=torch.zeros(2,3,256,256,dtype=torch.uint8)
                val=torch.utils.data.DataLoader([(images,images,torch.tensor([0,1]))],batch_size=None)
                otf=SimpleNamespace(train=dataset,corpus=c,checkpoint_validation='full',
                    save_record=lambda p:Path(p).write_text(json.dumps(scratch50_record(20))))
                loaders=SimpleNamespace(train=train,val=val)
                try:
                    with patch('cvdl.runtime.SteganalysisModule',side_effect=tiny_module),patch('cvdl.callbacks.TrainingCurves.redraw'):
                        first=runtime.create_training(s,otf,loaders,trainer_options=dict(max_steps=7,num_sanity_val_steps=0))
                        self.assertEqual(len(first.lightning_model.configure_optimizers()['optimizer'].state),0)
                        first.trainer.fit(first.lightning_model,train,val,ckpt_path='last')
                        path=first.checkpoint_dir/'last.ckpt'
                        saved=torch.load(path,map_location='cpu',weights_only=False)
                        self.assertEqual((saved['epoch'],saved['global_step']),(6,7))
                        self.assertEqual(saved['hyper_parameters']['embedding_schedule_identity'],scratch50_record(180000))
                        capture=Capture()
                        resumed=runtime.create_training(s,otf,loaders,trainer_options=dict(max_steps=8,num_sanity_val_steps=0))
                        resumed.trainer.callbacks.append(capture)
                        resumed.trainer.fit(resumed.lightning_model,train,val,ckpt_path='last')
                        self.assertEqual((capture.epoch,capture.step,capture.scheduler),(7,7,7))
                        self.assertTrue(capture.optim_steps);self.assertTrue(all(v==7 for v in capture.optim_steps))
                        rows=[json.loads(line) for line in (first.checkpoint_dir/'embedding_epoch_counts.jsonl').read_text().splitlines()]
                        self.assertEqual(len(rows),8);self.assertEqual(rows[-1]['epoch'],7)
                        self.assertEqual((rows[-1]['original_stegos'],rows[-1]['stage_stegos']),(10,10))
                        record=json.loads((first.checkpoint_dir/'model_run_settings.json').read_text())
                        self.assertEqual(record['initialization'],'random')
                        self.assertEqual(record['embedding_schedule_identity'],scratch50_record(180000))
                        # No resumed model init is substituted for the true initial hash.
                        old=json.loads((first.checkpoint_dir/'training_start.json').read_text())
                        self.assertEqual(old['settings']['embedding_schedule_identity'],scratch50_record(180000))
                finally:train.shutdown()
        # Real architecture strict eager reload also accepts the additive identity.
        with TemporaryDirectory() as folder:
            module=SteganalysisModule(model='xunet_wide',width=4,front_end='srm',norm='none')
            module.hparams['embedding_schedule_identity']=scratch50_record(180000)
            path=Path(folder)/'model.ckpt'
            torch.save(dict(state_dict=module.state_dict(),hyper_parameters=dict(module.hparams),
                **{'pytorch-lightning_version':L.__version__}),path)
            restored=SteganalysisModule.load_from_checkpoint(path,map_location='cpu',strict=True,weights_only=False)
            for k,v in module.state_dict().items():self.assertTrue(torch.equal(v,restored.state_dict()[k]))

    def test_deterministic_shared_cpu_interrupted_resume_matches_uninterrupted_across_7_to_8(self):
        from cvdl.checkpoint_io import DurableCheckpointIO
        class Trace(L.Callback):
            def __init__(self):self.rows=[]
            def on_train_batch_start(self,trainer,module,batch,batch_idx):
                digest=hashlib.sha256()
                def add(value):
                    if isinstance(value,torch.Tensor):
                        digest.update(str((value.dtype,tuple(value.shape))).encode());digest.update(value.cpu().numpy().tobytes())
                    elif isinstance(value,(tuple,list)):
                        for item in value:add(item)
                    else:raise TypeError(type(value))
                add(batch)
                self.rows.append((trainer.current_epoch,trainer.global_step,
                    trainer.lr_scheduler_configs[0].scheduler.last_epoch,
                    trainer.optimizers[0].param_groups[0]['lr'],digest.hexdigest()))
        original=SteganalysisModule
        def tiny(**kwargs):
            module=original(**kwargs)
            module.net=torch.nn.Sequential(torch.nn.Conv2d(3,2,1),torch.nn.AdaptiveAvgPool2d(1),torch.nn.Flatten())
            return module
        def same(a,b):
            if isinstance(a,torch.Tensor):self.assertTrue(torch.equal(a,b))
            elif isinstance(a,dict):
                self.assertEqual(a.keys(),b.keys())
                for key in a:same(a[key],b[key])
            elif isinstance(a,(list,tuple)):
                self.assertEqual(len(a),len(b))
                for x,y in zip(a,b):same(x,y)
            else:self.assertEqual(a,b)
        def run(folder,steps,seed):
            torch.manual_seed(seed)
            s=settings(True,output_root=str(folder),compile='off',workers=0,eval_workers=0,
                persistent_workers=False,prime_next_epoch=False)
            c=make_corpus(True);dataset=CurriculumImages(c,SCRATCH50,None)
            train=training_loader(dataset,20,num_workers=0,pin_memory=False)
            images=torch.zeros(2,3,256,256,dtype=torch.uint8)
            val=torch.utils.data.DataLoader([(images,images,torch.tensor([0,1]))],batch_size=None)
            otf=SimpleNamespace(train=dataset,corpus=c,checkpoint_validation='full',
                save_record=lambda p:Path(p).write_text(json.dumps(scratch50_record(20))))
            try:
                bundle=runtime.create_training(s,otf,SimpleNamespace(train=train,val=val),
                    trainer_options=dict(max_steps=steps,num_sanity_val_steps=0))
                self.assertIsInstance(bundle.trainer.strategy.checkpoint_io,DurableCheckpointIO)
                trace=Trace();bundle.trainer.callbacks.append(trace)
                bundle.trainer.fit(bundle.lightning_model,train,val,ckpt_path='last')
                saved=torch.load(bundle.checkpoint_dir/'last.ckpt',map_location='cpu',weights_only=False)
                counts=(bundle.checkpoint_dir/'embedding_epoch_counts.jsonl').read_text()
                return saved,trace.rows,counts
            finally:train.shutdown()
        with TemporaryDirectory() as folder,patch('cvdl.runtime.SteganalysisModule',side_effect=tiny),patch('cvdl.callbacks.TrainingCurves.redraw'):
            base,base_trace,base_counts=run(Path(folder)/'base',8,8123)
            first,first_trace,_=run(Path(folder)/'split',7,8123)
            self.assertEqual((first['epoch'],first['global_step']),(6,7))
            # New dataset/loader/model/process RNG: only the checkpoint supplies trained state.
            resumed,resume_trace,resume_counts=run(Path(folder)/'split',8,999)
            self.assertEqual(base_trace,first_trace+resume_trace)
            self.assertEqual(resume_trace[0][:3],(7,7,7))
            self.assertEqual((resumed['epoch'],resumed['global_step']),(7,8))
            for key in ('state_dict','optimizer_states','lr_schedulers','hyper_parameters'):
                same(base[key],resumed[key])
            self.assertEqual(base_counts,resume_counts)
            rows=[json.loads(x) for x in resume_counts.splitlines()]
            self.assertEqual((rows[-1]['original_stegos'],rows[-1]['stage_stegos']),(10,10))
            self.assertEqual(len(rows),8)

    def test_real_data_records_match_normal_except_new_schedule(self):
        # Metadata/CSV verification only: no full photos or GPU are read.
        base=top_values(json.loads((ROOT/'notebooks/experiments/srm-e50.ipynb').read_text()))
        normal=load(environ={},_profile=machine_profile('pc',device='cpu'),**base)
        a,_=runtime.create_data(normal)
        for paired in (False,True):
            b,_=runtime.create_data(settings(paired))
            self.assertEqual(b.record['embedding_schedule'],scratch50_record(180000))
            self.assertEqual(b.validation_pair_csv,a.validation_pair_csv)
            ar=dict(a.record);br=dict(b.record);br.pop('embedding_schedule')
            # paired_batches is intentionally a data identity difference only.
            def strip_pairing(value):
                if isinstance(value,dict):return {k:strip_pairing(v) for k,v in value.items() if k!='paired_batches'}
                if isinstance(value,list):return [strip_pairing(v) for v in value]
                return value
            self.assertEqual(strip_pairing(ar),strip_pairing(br))


if __name__=='__main__':unittest.main()
