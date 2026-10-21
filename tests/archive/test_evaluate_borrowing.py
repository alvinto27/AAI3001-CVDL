"""Borrow data without changing checkpoint architecture or fabricating old inputs."""
import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import pandas as pd
import torch
from cvdl.settings import machine_profile
from scripts.archive import evaluate_legacy_512 as er


class BorrowingTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        root=Path(self.temp.name)
        self.target=root/'target';self.source=root/'source'
        self.target.mkdir();self.source.mkdir()
        self.checkpoint=self.target/'best-target.ckpt'
        self.hparams=dict(model='xunet_wide',width=4,norm='none',clip_t=3.0)
        torch.save(dict(hyper_parameters=self.hparams),self.checkpoint)
        (self.source/'best-donor.ckpt').write_bytes(b'not the evaluated checkpoint')
        self.own=dict(self.hparams,lr=.001,max_epochs=40,warmup_epochs=1,
                      lr_drops=[[5,.3],[20,.1]],grad_clip_norm=.5)
        self.save_own()
        self.values=dict(name='donor',quick=False,model='xunet_wide',width=4,norm='none',clip_t=3.,
            front_end='kv',lr=.0003,epochs=30,lr_drops=[[5,.3],[20,.1]],
            source_mix={'raise':.4,'lsdir':.2,'alaska':.4},tiles_per_epoch=180000,
            checkpoint_validation='capped',train_grid='overlap_last',train_tile_size=256,
            tile_size=512,batch_size=128,train_yaml='configs/train-k1to4.yaml',extra_tests='test-bpp1,source-panels')
        self.save_source()
        self.profile=patch('cvdl.settings.automatic_profile',return_value=machine_profile('pc',root=er.ROOT))
        self.profile.start();self.addCleanup(self.profile.stop)

    def save_own(self):
        (self.target/'model_run_settings.json').write_text(json.dumps(self.own))

    def save_source(self):
        (self.source/'run_settings.json').write_text(json.dumps(dict(values=self.values)))

    def resolve(self):
        return er.settings_for_evaluation(self.target,self.checkpoint,self.source)


    def test_shared_borrowing_outputs_are_idempotent_but_never_silently_replaced(self):
        _,meta,snapshot=self.resolve();out=Path(self.temp.name)/'out'
        er.write_borrowing(out,meta,snapshot);er.write_borrowing(out,meta,snapshot)
        before=(out/'borrowed_run_settings.json').read_bytes()
        with self.assertRaisesRegex(ValueError,'borrowed provenance differs'):
            er.write_borrowing(out,meta,b'other donor')
        self.assertEqual((out/'borrowed_run_settings.json').read_bytes(),before)
        with self.assertRaisesRegex(ValueError,'refusing to overwrite'):
            er.refuse_existing(['main'],out,out,False)

    def test_main_uses_target_checkpoint_and_donor_identity_and_writes_provenance(self):
        settings,meta,snapshot=self.resolve()
        (self.target/'validation_pairs.csv').write_text('original target validation\n')
        (self.source/'validation_pairs.csv').write_text('borrowed validation\n')
        otf=SimpleNamespace(validation_pair_csv='borrowed validation\n')
        model=SimpleNamespace(net=object());val=pd.DataFrame({'label':[0,1],'prob_stego':[.1,.9]})
        with patch('cvdl.lit.select_best_checkpoint',return_value=str(self.checkpoint)), \
             patch('cvdl.runtime.configure'),patch('cvdl.runtime.create_data',return_value=(otf,None)), \
             patch('cvdl.data.build_otf_loaders',return_value='loaders'), \
             patch('cvdl.archive.post_fit.run',return_value=(model,{'val':val},{'ok':True})) as run, \
             patch('cvdl.archive.panels.write_validation'),patch('cvdl.archive.panels.write_standard_validation'),patch('cvdl.archive.panels.tables'):
            self.assertEqual(er.evaluate(self.target,settings_from=self.source),{'ok':True})
        self.assertEqual(run.call_args.args[4],self.checkpoint)
        self.assertEqual(run.call_args.args[0].batch_size,128)
        self.assertEqual((self.target/'validation_pairs.csv').read_text(),'original target validation\n')
        record=json.loads((self.target/'evaluation_settings.json').read_text())
        self.assertEqual(record['borrowed_run_settings_sha256'],meta['borrowed_run_settings_sha256'])
        self.assertEqual((self.target/'borrowed_run_settings.json').read_bytes(),snapshot)
        self.assertEqual(record['checkpoint_sha256'],hashlib.sha256(self.checkpoint.read_bytes()).hexdigest())

    def test_cli_forwards_settings_from_to_main(self):
        import sys
        with patch.object(sys,'argv',['evaluate_run.py',str(self.target),'--test-suite','legacy-512','--only','main','--settings-from',str(self.source)]), \
             patch.object(er,'evaluate') as run:
            er.main()
        self.assertEqual(run.call_args.kwargs['settings_from'],self.source)


if __name__=='__main__':unittest.main()
