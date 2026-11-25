import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch
from cvdl.curriculum import (CurriculumImages, CurriculumCounts, stage_for_epoch,
    original_positions, weights_only, check_start, initialize)
from cvdl.lit import SteganalysisModule, select_evaluation_checkpoint
from cvdl.settings import load, machine_profile
from lsb_torch import Corpus, RunSettings, TrainingImages, training_loader

ROOT=Path(__file__).resolve().parents[1]


class ToyCorpus(Corpus):
    def photo(self,record):
        return np.random.default_rng(int(record['photo_index'])).integers(0,256,(256,256,3),dtype=np.uint8)


def toy_corpus():
    records=[dict(photo_id=str(i),photo_index=str(i),source='raise',split='train',camera_model='toy',
        source_height='256',source_width='256',channels='3',grid_policy='overlap_last',
        overlap256_y='0',overlap256_x='0',overlap256_rows='1',overlap256_cols='1') for i in range(2)]
    return ToyCorpus(ROOT,RunSettings(sources=('raise',),tile_size=256,
        settings_yaml=str(ROOT/'configs/train-k1to4.yaml'),source_mix={'raise':1.},tiles_per_epoch=8),records=records)


class CurriculumTests(unittest.TestCase):
    def settings(self,**kwargs):
        return load(_profile=machine_profile('pc',device='cpu'),environ={},**kwargs)

    def test_stage_table_all_fifteen_epochs(self):
        for epoch in range(15):
            s=stage_for_epoch(epoch)
            self.assertEqual(s['stage'],epoch//5+1)
            self.assertEqual(s['bpp_max'],(.35,.25,.15)[epoch//5])
            self.assertEqual(s['k1_weight'],(1,2,3)[epoch//5])
            self.assertEqual(s['original_fraction'],.25)
            self.assertEqual(stage_for_epoch(epoch,mode='original')['original_fraction'],1)
        for epoch in (-1,15):
            with self.assertRaises(ValueError):stage_for_epoch(epoch)

    def test_exact_quarter_occurrences_seeded(self):
        a=original_positions(180000,0,0,'low_bpp')
        self.assertEqual(int(a.sum()),45000)
        np.testing.assert_array_equal(a,original_positions(180000,0,0,'low_bpp'))
        self.assertFalse(np.array_equal(a,original_positions(180000,0,1,'low_bpp')))
        self.assertFalse(np.array_equal(a,original_positions(180000,1,0,'low_bpp')))
        self.assertTrue(original_positions(20,0,0,'original').all())
        with self.assertRaises(ValueError):original_positions(7,0,0,'low_bpp')

    def test_control_pixels_and_order_match_default_training(self):
        corpus=toy_corpus();normal=TrainingImages(corpus);control=CurriculumImages(corpus,'original',5)
        for epoch in (0,5,14):
            normal.set_epoch(epoch);control.set_epoch(epoch)
            for a,b in zip(list(normal),list(control),strict=True):
                torch.testing.assert_close(a[0],b[0],rtol=0,atol=0)
                self.assertEqual(a[1],b[1]);self.assertEqual(a[2],int(b[2][0]))
                self.assertEqual(int(b[2][1]),1)

    def test_real_pixels_exact_mix_and_workers_packing(self):
        for workers,pack in ((0,1),(2,1),(2,4)):
            dataset=CurriculumImages(toy_corpus(),'low_bpp',5)
            loader=training_loader(dataset,batch_size=8,num_workers=workers,pin_memory=False,pack_blocks=pack)
            try:
                for epoch in (0,5,10):
                    dataset.set_epoch(epoch);original=stage=0
                    for images,labels,tags in loader:
                        self.assertEqual(images.dtype,torch.uint8)
                        original+=int(((labels==1)&(tags[:,1]==1)).sum())
                        stage+=int(((labels==1)&(tags[:,1]==0)).sum())
                    self.assertEqual((original,stage),(2,6))
            finally:loader.shutdown()

    def checkpoint(self,folder):
        source=SteganalysisModule(front_end='srm',lr=3e-4,norm='none',steps_per_epoch=2)
        path=Path(folder)/'start.ckpt'
        torch.save(dict(state_dict=source.state_dict(),hyper_parameters=dict(source.hparams),
            epoch=43,global_step=9999,optimizer_states=['must-not-load'],lr_schedulers=['must-not-load']),path)
        return source,path,hashlib.sha256(path.read_bytes()).hexdigest()

    def test_weights_only_sha_and_fresh_optimizer_schedule(self):
        with TemporaryDirectory() as folder:
            source,path,sha=self.checkpoint(folder)
            target=SteganalysisModule(front_end='srm',norm='none',lr=3e-5,max_epochs=15,steps_per_epoch=2,lr_drops=())
            record=weights_only(target,path,sha)
            for key,value in source.net.state_dict().items():torch.testing.assert_close(value,target.net.state_dict()[key],rtol=0,atol=0)
            self.assertEqual(record['checkpoint_sha256'],sha);self.assertFalse(record['optimizer_restored'])
            optim=target.configure_optimizers();self.assertEqual(len(optim['optimizer'].state),0)
            self.assertEqual(optim['optimizer'].param_groups[0]['initial_lr'],3e-5)
            self.assertEqual(target.current_epoch,0);self.assertEqual(target.global_step,0)
            self.assertEqual(optim['lr_scheduler']['scheduler'].last_epoch,0)

    def test_bad_sha_and_architecture_fail(self):
        with TemporaryDirectory() as folder:
            _,path,sha=self.checkpoint(folder)
            target=SteganalysisModule(front_end='srm',norm='none')
            with self.assertRaisesRegex(ValueError,'SHA-256'):weights_only(target,path,'0'*64)
            for kwargs in (dict(front_end='kv'),dict(front_end='srm',clip_t=None),dict(front_end='srm',norm='group')):
                with self.assertRaisesRegex(ValueError,'setting mismatch'):weights_only(SteganalysisModule(**kwargs),path,sha)

    def test_resume_does_not_reload_start_and_checks_fingerprint(self):
        with TemporaryDirectory() as folder:
            _,path,sha=self.checkpoint(folder)
            s=self.settings(init_checkpoint=str(path),init_checkpoint_sha256=sha)
            self.assertFalse(check_start(s,folder))
            module=SteganalysisModule(front_end='srm',norm='none')
            initialize(s,module,folder,False)
            (Path(folder)/'last.ckpt').write_bytes(b'resume-owned-by-Lightning')
            path.unlink()
            self.assertTrue(check_start(s,folder))
            with patch('cvdl.curriculum.weights_only',side_effect=AssertionError('start reloaded')):
                initialize(s,module,folder,True)
            changed=self.settings(init_checkpoint=str(path),init_checkpoint_sha256=sha,lr=1e-5)
            with self.assertRaisesRegex(ValueError,'fingerprint'):check_start(changed,folder)

    def test_defaults_do_not_gain_opt_in_fields(self):
        s=self.settings()
        for key in ('init_checkpoint','init_checkpoint_sha256','embedding_schedule','embedding_stage_epochs','evaluation_checkpoint'):
            self.assertNotIn(key,s.values)
        self.assertEqual(s.lr,.0003)
        self.assertNotEqual(s.lr_drops,())
        self.assertEqual(self.settings(lr_drops=()).lr_drops,())

    def test_missing_sha_and_invalid_schedule_fail(self):
        with self.assertRaises(ValueError):self.settings(init_checkpoint='x')
        for kwargs in (dict(batch_size=12),dict(epochs=14),dict(tiles_per_epoch=7)):
            values=dict(embedding_schedule='low_bpp',epochs=15,source_mix={'raise':1.},
                tiles_per_epoch=20,train_yaml='configs/train-k1to4.yaml',init_checkpoint='x',init_checkpoint_sha256='0'*64)
            values.update(kwargs)
            with self.assertRaises(ValueError):self.settings(**values)

    def test_notebooks_differ_only_in_schedule_and_identity(self):
        values=[]
        for name in ('srm-low-bpp-curriculum','srm-low-bpp-control'):
            notebook=json.loads((ROOT/f'notebooks/experiments/{name}.ipynb').read_text())
            ns={}
            with patch('cvdl.settings.automatic_profile',return_value=machine_profile('pc',device='cpu')):
                exec(''.join(notebook['cells'][1]['source']),ns)
            d=dict(ns['settings'].values)
            for key in ('name','run_name','job_id'):d.pop(key)
            values.append(d)
        self.assertEqual({k for k in values[0] if values[0][k]!=values[1][k]},{'embedding_schedule'})
        self.assertEqual(values[0]['lr'],3e-5);self.assertEqual(values[0]['lr_drops'],())
        self.assertEqual(values[0]['evaluation_checkpoint'],'last')

    def test_real_full_size_notebooks_pass_frozen_validation_guard_without_training(self):
        from cvdl.validation import load_frozen
        for name in ('srm-low-bpp-curriculum','srm-low-bpp-control'):
            notebook=json.loads((ROOT/f'notebooks/experiments/{name}.ipynb').read_text());ns={}
            with patch('cvdl.settings.automatic_profile',return_value=machine_profile('pc',device='cpu')):
                exec(''.join(notebook['cells'][1]['source']),ns)
            s=ns['settings']
            self.assertEqual((s.checkpoint_validation,s.validation_recipe,s.tiles_per_epoch),
                             ('train_matched','60da4391043e8ce48ff2',180000))
            frame,recipe,summary,text=load_frozen(ROOT,s.to_run_settings(),s.train_grid,s.train_tile_size,s.validation_recipe)
            self.assertEqual(len(frame),18000)
            self.assertEqual(hashlib.sha256(text.encode()).hexdigest(),summary['pairs_sha256'])
            self.assertEqual(recipe['train_pairs_per_epoch'],180000)

    def test_evaluator_last_policy_and_incomplete_failure(self):
        with TemporaryDirectory() as folder:
            folder=Path(folder)
            (folder/'best-toy.ckpt').write_bytes(b'best')
            self.assertEqual(select_evaluation_checkpoint('',folder),str(folder/'best-toy.ckpt'))
            (folder/'run_settings.json').write_text(json.dumps(dict(values=dict(evaluation_checkpoint='last',epochs=15))))
            with self.assertRaises(FileNotFoundError):select_evaluation_checkpoint('',folder)
            path=folder/'last.ckpt'
            torch.save(dict(epoch=14,global_step=30,hyper_parameters=dict(steps_per_epoch=2)),path)
            self.assertEqual(select_evaluation_checkpoint('',folder),str(path))
            torch.save(dict(epoch=9,global_step=20,hyper_parameters=dict(steps_per_epoch=2)),path)
            with self.assertRaisesRegex(ValueError,'complete'):select_evaluation_checkpoint('',folder)

    def test_actual_consumed_epoch_counts(self):
        with TemporaryDirectory() as folder:
            callback=CurriculumCounts(Path(folder)/'counts.jsonl','low_bpp',5)
            trainer=SimpleNamespace(current_epoch=10)
            callback.on_train_epoch_start(trainer,None)
            batch=(None,torch.tensor([0,1,1,1,1]),torch.tensor([[0,1],[1,1],[3,0],[5,0],[7,0]]))
            callback.on_train_batch_end(trainer,None,None,batch,0)
            callback.on_train_epoch_end(trainer,None)
            row=json.loads(callback.path.read_text())
            self.assertEqual((row['stage'],row['original_stegos'],row['stage_stegos']),(3,1,3))
            restored=CurriculumCounts(callback.path,'low_bpp',5)
            restored.load_state_dict(callback.state_dict())
            self.assertEqual(len(callback.path.read_text().splitlines()),1)
            self.assertEqual(restored.state_dict(),callback.state_dict())


if __name__=='__main__':unittest.main()
