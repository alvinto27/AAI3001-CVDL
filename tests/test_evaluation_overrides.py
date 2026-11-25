"""Evaluation overrides are explicit, narrow, recorded and never saved as training inputs."""
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from cvdl.settings import machine_profile, plain
from scripts import evaluate_run as er


class EvaluationOverrideTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name);self.folder=self.root/'run';self.folder.mkdir()
        (self.folder/'best-test.ckpt').write_bytes(b'not loaded in these CPU settings tests')
        self.saved=self.folder/'run_settings.json'
        self.saved.write_text(json.dumps(dict(values=dict(name='testeval',quick=False,model='xunet_wide',width=4,
            norm='none',clip_t=3.,front_end='kv',batch_size=128,train_yaml='configs/train-k1to4.yaml'))))
        self.original=self.saved.read_bytes()
        self.profile=patch('cvdl.settings.automatic_profile',return_value=machine_profile('pc',root=er.ROOT))
        self.profile.start();self.addCleanup(self.profile.stop)
        self.arguments=('rocm_conv1_fix=off','eval_batch_size=32','eval_workers=0')

    def resolve(self,args=()):
        return er.settings_for_evaluation(self.folder,self.folder/'best-test.ckpt',eval_overrides=args)

    def test_allowed_overrides_applied_and_recorded_without_input_edits(self):
        settings,context,snapshot=self.resolve(self.arguments)
        self.assertEqual((settings.rocm_conv1_fix,settings.batch_size,settings.eval_workers),(False,32,0))
        self.assertEqual(len(context['evaluation_overrides']),3)
        self.assertTrue(all(r['source']=='CLI --set (evaluation only)' for r in context['evaluation_overrides']))
        self.assertEqual(plain(settings.values)['batch_size'],32)
        self.assertIn('override',plain(settings.provenance)['batch_size'])
        self.assertEqual(self.saved.read_bytes(),self.original)
        self.assertIsNone(snapshot)

    def test_curriculum50_evaluation_batch_does_not_relax_training_guard(self):
        from cvdl.settings import load
        from scripts.g4.preflight import top_values
        notebook=json.loads((er.ROOT/'notebooks/experiments/srm-e50-curriculum50.ipynb').read_text())
        for paired in (False,True):
            values=top_values(notebook);values['paired_batches']=paired
            training=load(environ={},_profile=machine_profile('pc',root=er.ROOT),**values)
            self.saved.write_text(json.dumps(dict(values=plain(training.values))))
            before=self.saved.read_bytes()
            settings,context,_=self.resolve(('eval_batch_size=32',))
            self.assertEqual(settings.batch_size,32)
            self.assertEqual(settings.embedding_schedule,'low_bpp50_v1')
            self.assertEqual(settings.paired_batches,paired)
            self.assertEqual(settings.epochs,50)
            self.assertEqual(settings.lr_drops,((8,.3),(33,.1)))
            self.assertEqual(settings.checkpoint_validation,'capped')
            self.assertEqual(context['evaluation_overrides'][0]['value'],32)
            self.assertIn('override',settings.provenance['batch_size'])
            self.assertEqual(self.saved.read_bytes(),before)
            self.assertEqual(er.settings_for_run(self.folder).batch_size,128)
            values['batch_size']=32
            with self.assertRaisesRegex(ValueError,'batch128'):
                load(environ={},_profile=machine_profile('pc',root=er.ROOT),**values)

    def test_forbidden_model_data_and_identity_keys_fail(self):
        for key in ('model','width','norm','clip_t','front_end','source_mix','tiles_per_epoch','seed',
                    'run_seed','tile_size','train_grid','sources','evaluation_yaml','quick','workers','root'):
            with self.subTest(key=key),self.assertRaisesRegex(ValueError,'forbidden evaluation.*identity'):
                er.evaluation_override_records([key+'=1'])
        with self.assertRaisesRegex(ValueError,'forbidden evaluation'):
            er.evaluation_override_records(['eval_workers'])

    def test_no_override_preserves_values_sources_and_metadata(self):
        before=er.settings_for_run(self.folder);after,context,snapshot=self.resolve()
        self.assertEqual(plain(before.values),plain(after.values))
        self.assertEqual(plain(before.provenance),plain(after.provenance))
        self.assertEqual(context,{})
        self.assertIsNone(snapshot)
        self.assertFalse((self.root/'absent').exists())
        self.assertEqual(self.saved.read_bytes(),self.original)

    def test_repeatable_alias_keeps_every_argument_and_last_value(self):
        args=('eval_batch_size=16','batch_size=32')
        settings,context,_=self.resolve(args)
        self.assertEqual(settings.batch_size,32)
        self.assertEqual([r['value'] for r in context['evaluation_overrides']],[16,32])

    def test_forbidden_cli_key_refused_before_folder_or_gpu_work(self):
        with patch.object(sys,'argv',['evaluate_run.py','missing','--set','front_end=srm']), \
             patch.object(er,'checkpoint_folder') as folder:
            with self.assertRaisesRegex(ValueError,'forbidden evaluation'):
                er.main()
        folder.assert_not_called()

    def test_cli_routes_same_overrides_to_standard256(self):
        settings,context,_=self.resolve(self.arguments)
        argv=['evaluate_run.py',str(self.folder),'--only','main','--hard']
        for arg in self.arguments:argv+=['--set',arg]
        with patch.object(sys,'argv',argv), \
             patch('cvdl.lit.select_best_checkpoint',return_value=str(self.folder/'best-test.ckpt')), \
             patch.object(er,'settings_for_evaluation',return_value=(settings,context,None)) as resolve, \
             patch('cvdl.runtime.configure'),patch('scripts.pc_safety.check_workers'), \
             patch('scripts.pc_safety.own_process_group'),patch('cvdl.standardtest.run') as run:
            er.main()
        self.assertEqual(resolve.call_args.kwargs['eval_overrides'],self.arguments)
        self.assertEqual(run.call_args.kwargs['provenance'],context)
        self.assertTrue(run.call_args.kwargs['hard'])
        self.assertEqual(self.saved.read_bytes(),self.original)

    def test_cli_refuses_retired_stages_before_folder_or_gpu_work(self):
        import contextlib,io
        for stages in ('padding,tanh','main,temperature','tables','hard'):
            argv=['evaluate_run.py','missing','--only',stages,'--set','eval_batch_size=32']
            with self.subTest(stages=stages),patch.object(sys,'argv',argv), \
                 patch.object(er,'checkpoint_folder') as folder,patch('cvdl.runtime.configure') as gpu, \
                 contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as error:er.main()
                self.assertEqual(error.exception.code,2)
                folder.assert_not_called();gpu.assert_not_called()

    def test_borrowing_keeps_source_snapshot_original_and_records_overrides(self):
        import torch
        target=self.root/'target';target.mkdir()
        own=dict(model='xunet_wide',width=4,norm='none',clip_t=3.,lr=.001,max_epochs=40,
                 warmup_epochs=1,lr_drops=[[5,.3],[20,.1]],grad_clip_norm=1.)
        (target/'model_run_settings.json').write_text(json.dumps(own))
        checkpoint=target/'best-own.ckpt';torch.save(dict(hyper_parameters=own),checkpoint)
        settings,context,snapshot=er.settings_for_evaluation(target,checkpoint,self.folder,eval_overrides=self.arguments)
        self.assertEqual((settings.lr,settings.batch_size),(.001,32))
        self.assertEqual(snapshot,self.original)
        self.assertEqual(context['borrowed_run_settings_sha256'],hashlib.sha256(self.original).hexdigest())
        self.assertEqual(len(context['evaluation_overrides']),3)
        self.assertEqual(self.saved.read_bytes(),self.original)


if __name__=='__main__':unittest.main()
