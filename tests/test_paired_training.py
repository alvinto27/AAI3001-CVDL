"""Paired-only behavior change; independent historical unpaired RNG reference."""
from dataclasses import replace
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch
import lightning as L
from cvdl.curriculum import CurriculumImages, CurriculumCounts, weights_only
from cvdl.data import PairDataset, build_png_loaders
from cvdl.lit import SteganalysisModule
from cvdl.settings import load, machine_profile
from cvdl.validation import load_frozen
from lsb_cells import changed_cell_targets
from lsb_torch import TrainingImages, image_tensor, training_loader, PrimedTrainingLoader
from test_cell_training import corpus, ENABLED, ROOT


def historical_image(array, entropy, label, augment, dtype=torch.float32, *, cell_targets=None):
    # Frozen pre-fix algorithm: independent label-keyed private generator.
    image=torch.from_numpy(array).permute(2,0,1).to(dtype=dtype)
    cells=[torch.from_numpy(np.asarray(a).astype(np.int64)) for a in cell_targets] if cell_targets is not None else None
    if augment:
        words=np.random.SeedSequence(list(entropy)+[0x415547,label]).generate_state(2)
        generator=torch.Generator().manual_seed(int(words[0]) | (int(words[1])<<32))
        h=bool(torch.rand((),generator=generator)<.5)
        v=bool(torch.rand((),generator=generator)<.5)
        turns=int(torch.randint(0,4,(),generator=generator).item())
        for dims,enabled in (((-1,),h),((-2,),v)):
            if enabled:
                image=torch.flip(image,dims)
                if cells is not None: cells=[torch.flip(a,dims) for a in cells]
        image=torch.rot90(image,turns,(-2,-1))
        if cells is not None: cells=[torch.rot90(a,turns,(-2,-1)) for a in cells]
    if cells is not None: return image.contiguous(),tuple(a.contiguous() for a in cells)
    return image.contiguous()


class HistoricalImages(TrainingImages):
    def _output(self,item):
        image,label,index,entropy=item
        return historical_image(image,entropy,label,self.corpus.settings.augment,getattr(torch,self.corpus.settings.tensor_dtype)),label,index


class HistoricalCells(CurriculumImages):
    def _output(self,item):
        image,label,tag,entropy=item
        if self.cell_supervision=='off':
            return historical_image(image,entropy,label,self.corpus.settings.augment,getattr(torch,self.corpus.settings.tensor_dtype)),label,torch.tensor(tag)
        image,cells=historical_image(image,entropy,label,self.corpus.settings.augment,getattr(torch,self.corpus.settings.tensor_dtype),cell_targets=tag[2:])
        return image,label,torch.cat((torch.tensor(tag[:2]),*(a.flatten() for a in cells)))


def make_corpus(paired=True,dtype='uint8',count=20):
    c=corpus();c.settings=replace(c.settings,paired_batches=paired,tensor_dtype=dtype,tiles_per_epoch=count)
    return c


def collect(dataset,workers=0,pack=1):
    torch.manual_seed(37)
    loader=training_loader(dataset,batch_size=8,num_workers=workers,pin_memory=False,pack_blocks=pack)
    try:
        batches=list(loader)
        return tuple(torch.cat([b[i] for b in batches]) for i in range(3)),torch.get_rng_state().clone()
    finally:loader.shutdown()


class PairedTrainingTests(unittest.TestCase):
    def test_unpaired_exact_tensors_private_draws_and_global_rng(self):
        array=np.random.default_rng(7).integers(0,256,(256,256,3),dtype=np.uint8)
        counts,valid=changed_cell_targets(np.array([31*256*3+27*3]))
        original_rand,original_int=torch.rand,torch.randint
        def trace(fn,**kwargs):
            events=[]
            def wrapped(original,name):
                def call(*args,**kw):
                    state=hashlib.sha256(kw['generator'].get_state().numpy().tobytes()).hexdigest()
                    out=original(*args,**kw);events.append((name,state,out.item()));return out
                return call
            before=torch.get_rng_state().clone();np_before=np.random.get_state()
            with patch('torch.rand',wrapped(original_rand,'rand')),patch('torch.randint',wrapped(original_int,'randint')):
                out=fn(array,kwargs.pop('entropy'),**kwargs)
            self.assertTrue(torch.equal(before,torch.get_rng_state()))
            after=np.random.get_state();self.assertEqual(np_before[0],after[0]);np.testing.assert_array_equal(np_before[1],after[1]);self.assertEqual(np_before[2:],after[2:])
            return out,events
        for dtype in (torch.uint8,torch.float32):
            for label in (0,1):
                for key in range(16):
                    kwargs=dict(entropy=[key,3,25],label=label,augment=True,dtype=dtype,cell_targets=(counts,valid))
                    a,draws=trace(historical_image,**kwargs);b,got=trace(image_tensor,**kwargs)
                    self.assertEqual(draws,got);self.assertEqual(len(got),3)
                    for x,y in zip((a[0],*a[1]),(b[0],*b[1])):self.assertTrue(torch.equal(x,y))

    def test_unpaired_loaders_historical_bytes_order_rng_workers_packing_dtypes(self):
        for dtype in ('uint8','float32'):
            for workers,pack in ((0,1),(2,1),(2,4)):
                for cells in (None,'off','changed_cells_v1'):
                    c=make_corpus(False,dtype)
                    if cells is None:
                        old,new=HistoricalImages(c),TrainingImages(c)
                    else:
                        old,new=(cls(c,'original',5,cell_supervision=cells) for cls in (HistoricalCells,CurriculumImages))
                    for epoch in (0,1):
                        old.set_epoch(epoch);new.set_epoch(epoch)
                        a,arng=collect(old,workers,pack);b,brng=collect(new,workers,pack)
                        for x,y in zip(a,b):self.assertTrue(torch.equal(x,y),(dtype,workers,pack,cells,epoch))
                        self.assertTrue(torch.equal(arng,brng))

    def test_pairs_same_key_and_recomputed_cell_targets_across_workers_and_packing(self):
        for dtype in ('uint8','float32'):
            snapshots=[]
            for workers,pack in ((0,1),(2,1),(2,4)):
                c=make_corpus(True,dtype)
                dataset=CurriculumImages(c,'original',5,cell_supervision='changed_cells_v1')
                (clean,stego,tags),_=collect(dataset,workers,pack)
                self.assertEqual(tuple(tags.shape),(20,2,130))
                self.assertTrue((tags[:,0,0]+1==tags[:,1,0]).all())
                self.assertTrue((tags[:,:,1]==1).all())
                self.assertFalse(tags[:,0,2:66].any());self.assertTrue((tags[:,0,66:]==1).all())
                for a,b,t in zip(clean,stego,tags):
                    diff=(a.to(torch.int16)-b.to(torch.int16)).abs()
                    self.assertLessEqual(int(diff.max()),15) # k1..4 LSB only: orientation cannot differ.
                    changed=torch.nonzero(diff.any(0)).numpy()
                    counts,valid=changed_cell_targets((changed[:,0]*256+changed[:,1])*3)
                    np.testing.assert_array_equal(t[1,2:66].reshape(8,8),counts)
                    np.testing.assert_array_equal(t[1,66:].reshape(8,8),valid)
                snapshots.append((clean,stego,tags))
            for a,b in zip(snapshots[1],snapshots[2]):self.assertTrue(torch.equal(a,b))
        # Exact same transformed raw difference and historical stego orientation.
        c=make_corpus();d=TrainingImages(c)
        for item in d._items_mixed(c.train_tile_schedule(0)[0],0):
            clean,stego,index,key=item;a,b,_=d._output(item)
            expected=historical_image(stego,key,1,True,torch.uint8)
            self.assertTrue(torch.equal(b,expected))
            expected_diff=historical_image(stego.astype(np.int16)-clean.astype(np.int16),key,1,True,torch.int16)
            self.assertTrue(torch.equal(b.to(torch.int16)-a.to(torch.int16),expected_diff))

    def test_paired_original_without_cells_matches_ordinary_paired_pixels_and_budget(self):
        for dtype in ('uint8','float32'):
            c=make_corpus(True,dtype)
            ordinary=TrainingImages(c);control=CurriculumImages(c,'original',5)
            for a,b in zip(list(ordinary),list(control),strict=True):
                self.assertTrue(torch.equal(a[0],b[0]));self.assertTrue(torch.equal(a[1],b[1]))
                self.assertEqual(a[2]*2,int(b[2][0,0]));self.assertEqual(a[2]*2+1,int(b[2][1,0]))
        for paired,size in ((False,128),(True,64)):
            dataset=TrainingImages(make_corpus(paired,count=180000))
            loader=training_loader(dataset,batch_size=size,num_workers=0,pin_memory=False)
            self.assertEqual(len(dataset),180000 if paired else 360000)
            self.assertEqual(len(loader),2813)
            self.assertEqual(len(dataset)%size,32 if paired else 64)
            loader.shutdown()

    def test_paired_ce_bce_gradients_tag_order_equal_explicit_unpaired_flattening(self):
        c=make_corpus();d=CurriculumImages(c,'original',5,cell_supervision='changed_cells_v1')
        batch=next(iter(training_loader(d,batch_size=2,num_workers=0,pin_memory=False)))
        clean,stego,tags=batch
        paired=SteganalysisModule(**ENABLED,paired_batches=True)
        flat=SteganalysisModule(**ENABLED);flat.load_state_dict(paired.state_dict())
        logs={};paired.log=lambda k,v,**kw:logs.update({k:(v.detach().clone(),kw)})
        flat.log=lambda *a,**kw:None
        out=paired.training_step(batch,0)
        explicit=(torch.cat((clean,stego)),torch.tensor([0,0,1,1]),torch.cat((tags[:,0],tags[:,1])))
        ref=flat.training_step(explicit,0)
        self.assertTrue(torch.equal(out['loss'],ref['loss']));self.assertEqual(out['count'],4)
        self.assertTrue(torch.equal(out['loss'],logs['train_tile_ce'][0]+.2*logs['train_cell_loss'][0]))
        self.assertEqual(logs['train_loss'][1]['batch_size'],4)
        out['loss'].backward();ref['loss'].backward()
        for p,q in zip(paired.parameters(),flat.parameters()):
            if p.grad is not None:self.assertTrue(torch.equal(p.grad,q.grad))
        self.assertGreater(float(paired.net.cell_head.weight.grad.norm()),0)
        self.assertGreater(float(paired.net.conv1.weight.grad.norm()),0)
        self.assertIsNone(paired.net.kv.conv.weight.grad)

    def test_counts_full_epochs_priming_tail_and_original_guard(self):
        with TemporaryDirectory() as folder:
            c=make_corpus(count=20);d=CurriculumImages(c,'original',5,cell_supervision='changed_cells_v1')
            loader=PrimedTrainingLoader(training_loader(d,batch_size=8,num_workers=2,pin_memory=False,pack_blocks=4))
            cb=CurriculumCounts(Path(folder)/'counts.jsonl','original',5)
            module=SimpleNamespace(hparams={'paired_batches':True})
            try:
                for epoch in (0,1):
                    trainer=SimpleNamespace(current_epoch=epoch);cb.on_train_epoch_start(trainer,module)
                    sizes=[]
                    for i,batch in enumerate(loader):
                        sizes.append(len(batch[0]));cb.on_train_batch_end(trainer,module,None,batch,i)
                    self.assertEqual(sizes,[8,8,4]);cb.on_train_epoch_end(trainer,module)
                    if epoch==0:loader.prime_next(1)
                rows=[json.loads(x) for x in cb.path.read_text().splitlines()]
                self.assertEqual([x['consumed_stegos'] for x in rows],[20,20])
                self.assertTrue(all(x['original_stegos']==20 and x['stage_stegos']==0 for x in rows))
            finally:loader.shutdown()
            with self.assertRaisesRegex(ValueError,'original schedule'):CurriculumImages(c,'low_bpp',5)

    def test_notebook_only_identity_pairing_changes_settings_guards_and_real60da(self):
        from scripts.g4.preflight import top_values
        for base in ('srm-e50','srm-cell-aux'):
            old=json.loads((ROOT/f'notebooks/experiments/{base}.ipynb').read_text())
            new=json.loads((ROOT/f'notebooks/experiments/{base}-paired.ipynb').read_text())
            expected=json.loads(json.dumps(old));src=expected['cells'][1]['source'];i=src.index(f'    name="{base}",\n')
            src[i:i+1]=[f'    name="{base}-paired",\n','    paired_batches=True,\n']
            if base=='srm-cell-aux':
                prior=top_values(old)
                for key,value in (
                    ('init_checkpoint','runs-out/srm-e50-paired-20261023T022339Z-4518/checkpoints/srm-e50-paired-20261023T022339Z-4518/best-epoch39-val_pe0.0444-val_auc0.9917.ckpt'),
                    ('init_checkpoint_sha256','b00cd4b6d1f4c3648580691c9cf0e2a4bd83c55740a3cf2a7a561a8adf12f180')):
                    src[src.index(f'    {key}={prior[key]!r},\n')]=f'    {key}={value!r},\n'
            self.assertEqual(new,expected)
            s=load(environ={},_profile=machine_profile('g4',device='cpu'),**top_values(new))
            self.assertTrue(s.paired_batches);self.assertEqual(s.batch_size,128);self.assertEqual(s.tiles_per_epoch,180000)
            if base=='srm-cell-aux':
                self.assertEqual((s.epochs,s.lr,s.lr_drops,s.evaluation_checkpoint),(15,3e-5,(),'last'))
                frame,recipe,summary,text=load_frozen(ROOT,s.to_run_settings(),s.train_grid,s.train_tile_size,s.validation_recipe)
                self.assertEqual(len(frame),18000)
                self.assertEqual(hashlib.sha256(text.encode()).hexdigest(),'e017acb0f463bb147ff437bd1b3e833710eee7284066850729c9666b0079344d')
                bad=top_values(new);bad['embedding_schedule']='low_bpp'
                with self.assertRaisesRegex(ValueError,'paired mode needs original'):load(environ={},_profile=machine_profile('pc',device='cpu'),**bad)
            else:self.assertEqual((s.epochs,s.checkpoint_validation,getattr(s,'evaluation_checkpoint','best')),(50,'capped','best'))

    def test_curriculum50_cell_notebook_only_name_pairing_and_initializer_differ(self):
        from scripts.g4.preflight import top_values
        name='srm-cell-aux-curriculum50-paired'
        old=json.loads((ROOT/'notebooks/experiments/srm-cell-aux.ipynb').read_text())
        new=json.loads((ROOT/f'notebooks/experiments/{name}.ipynb').read_text())
        paired=json.loads((ROOT/'notebooks/experiments/srm-cell-aux-paired.ipynb').read_text())
        checkpoint='runs-out/srm-e50-curriculum50-paired-20261023T090435Z-3325/checkpoints/srm-e50-curriculum50-paired-20261023T090435Z-3325/best-epoch39-val_pe0.0419-val_auc0.9922.ckpt'
        sha='14c8e05c3d476912dd5540970060d3634b3a72b7d375ba5dad5ccafd50905c3a'
        for reference,add_pairing in ((old,True),(paired,False)):
            expected=json.loads(json.dumps(reference));src=expected['cells'][1]['source'];prior=top_values(reference)
            i=src.index(f'    name="{prior["name"]}",\n')
            src[i:i+1]=[f'    name="{name}",\n']+(['    paired_batches=True,\n'] if add_pairing else [])
            for key,value in (('init_checkpoint',checkpoint),('init_checkpoint_sha256',sha)):
                src[src.index(f'    {key}={prior[key]!r},\n')]=f'    {key}={value!r},\n'
            self.assertEqual(new,expected)  # Every cell, output and metadata field remains strict.
        for values in (top_values(paired),top_values(new)):
            p=Path(values['init_checkpoint']).parts
            self.assertEqual((p[0],p[2]),('runs-out','checkpoints'));self.assertEqual(p[1],p[3]);self.assertEqual(len(p),5)
        s=load(environ={},_profile=machine_profile('g4',device='cpu'),**top_values(new))
        self.assertTrue(s.paired_batches)
        self.assertEqual((s.embedding_schedule,s.embedding_stage_epochs),('original',5))
        self.assertEqual((s.epochs,s.lr,s.lr_drops,s.evaluation_checkpoint),(15,3e-5,(),'last'))
        self.assertEqual((s.batch_size,s.tiles_per_epoch,s.cell_supervision,s.cell_loss_weight),(128,180000,'changed_cells_v1',.2))
        self.assertEqual((s.validation_recipe,s.checkpoint_validation),('60da4391043e8ce48ff2','train_matched'))

    def test_curriculum50_source_is_weights_only_into_original_schedule_fresh_head(self):
        from cvdl.curriculum import initialize,check_start,scratch50_record,stage_for_epoch
        from scripts.g4.preflight import top_values
        with TemporaryDirectory() as folder:
            source=SteganalysisModule(**{k:v for k,v in ENABLED.items() if k!='cell_supervision'},paired_batches=True)
            source.hparams['embedding_schedule_identity']=scratch50_record(180000)
            target=SteganalysisModule(**ENABLED,paired_batches=True,lr=3e-5,max_epochs=15,lr_drops=())
            path=Path(folder)/'curriculum.ckpt'
            torch.save(dict(state_dict=source.state_dict(),hyper_parameters=dict(source.hparams),epoch=39,global_step=112520,
                optimizer_states=[{'must_not_restore':True}],lr_schedulers=[{'last_epoch':112520}]),path)
            values=top_values(json.loads((ROOT/'notebooks/experiments/srm-cell-aux-curriculum50-paired.ipynb').read_text()))
            values.update(init_checkpoint=str(path),init_checkpoint_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
            s=load(environ={},_profile=machine_profile('pc',device='cpu'),**values)
            run=Path(folder)/'fine-tune';run.mkdir()
            self.assertFalse(check_start(s,run))
            head={k:v.clone() for k,v in target.net.cell_head.state_dict().items()}
            record=initialize(s,target,run,False)
            self.assertEqual((record['weights_only'],record['optimizer_restored'],record['scheduler_restored'],record['source_epoch']),(True,False,False,39))
            self.assertEqual(record['settings']['embedding_schedule'],'original')
            self.assertNotIn('embedding_schedule_identity',record['settings'])
            self.assertNotIn('embedding_schedule_identity',target.hparams)
            for key,value in source.net.state_dict().items():self.assertTrue(torch.equal(value,target.net.state_dict()[key]),key)
            for key,value in head.items():self.assertTrue(torch.equal(value,target.net.cell_head.state_dict()[key]),key)
            optim=target.configure_optimizers()
            self.assertEqual(len(optim['optimizer'].state),0)
            self.assertEqual(optim['lr_scheduler']['scheduler'].last_epoch,0)
            self.assertEqual(stage_for_epoch(14,5,s.embedding_schedule)['original_fraction'],1.)

    def test_paired_source_weights_only_fresh_head_and_strict_compile_reload(self):
        with TemporaryDirectory() as folder:
            source=SteganalysisModule(**{k:v for k,v in ENABLED.items() if k!='cell_supervision'},paired_batches=True)
            target=SteganalysisModule(**ENABLED,paired_batches=True,lr=3e-5,max_epochs=15,lr_drops=())
            path=Path(folder)/'body.ckpt';torch.save(dict(state_dict=source.state_dict(),hyper_parameters=dict(source.hparams)),path)
            head=target.net.cell_head.weight.detach().clone()
            record=weights_only(target,path,hashlib.sha256(path.read_bytes()).hexdigest())
            self.assertFalse(record['optimizer_restored']);self.assertTrue(torch.equal(head,target.net.cell_head.weight))
            self.assertEqual(len(target.configure_optimizers()['optimizer'].state),0)
            target.net.compile(backend='aot_eager',fullgraph=True)
            batch=next(iter(training_loader(CurriculumImages(make_corpus(),'original',5,cell_supervision='changed_cells_v1'),batch_size=1,num_workers=0,pin_memory=False)))
            target.log=lambda *a,**k:None
            result=target.training_step(batch,0);result['loss'].backward()
            self.assertTrue(torch.isfinite(target.net.cell_head.weight.grad).all())
            path=Path(folder)/'cell.ckpt';torch.save(dict(state_dict=target.state_dict(),hyper_parameters=dict(target.hparams),**{'pytorch-lightning_version':L.__version__}),path)
            restored=SteganalysisModule.load_from_checkpoint(path,map_location='cpu',weights_only=False,strict=True)
            self.assertTrue(restored.hparams.paired_batches)
            with torch.no_grad():torch.testing.assert_close(restored.net(batch[0].float()),target.net(batch[0].float()))
            target.net._compiled_call_impl=None;torch._dynamo.reset()

    def test_png_augmented_pairing_refused_validation_unchanged(self):
        with self.assertRaisesRegex(ValueError,'augmented paired PNG'):PairDataset(pd.DataFrame(),ROOT,train=True)
        with self.assertRaisesRegex(ValueError,'augmented paired PNG'):
            build_png_loaders(pd.DataFrame({'split':['train']}),ROOT,batch_size=128,workers=0,eval_workers=0,pin_memory=False,paired_batches=True,seed=0)
        self.assertEqual(len(PairDataset(pd.DataFrame(),ROOT,train=False)),0)


if __name__=='__main__':unittest.main()
