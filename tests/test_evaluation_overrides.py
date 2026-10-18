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
        out=self.root/'output';er.record_evaluation_overrides(out,settings,self.arguments)
        record=json.loads((out/'evaluation_settings.json').read_text())
        self.assertEqual(record['evaluation_overrides'],context['evaluation_overrides'])
        self.assertEqual(record['values']['batch_size'],32)
        self.assertIn('override',record['sources']['batch_size'])
        self.assertEqual(self.saved.read_bytes(),self.original)
        self.assertIsNone(snapshot)

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
        er.record_evaluation_overrides(self.root/'absent',after,())
        self.assertFalse((self.root/'absent').exists())
        self.assertEqual(self.saved.read_bytes(),self.original)

    def test_repeatable_alias_keeps_every_argument_and_last_value(self):
        args=('eval_batch_size=16','batch_size=32')
        settings,context,_=self.resolve(args)
        self.assertEqual(settings.batch_size,32)
        self.assertEqual([r['value'] for r in context['evaluation_overrides']],[16,32])

    def test_override_record_is_idempotent_and_refuses_changed_inputs(self):
        settings,_,_=self.resolve(self.arguments);out=self.root/'output'
        er.record_evaluation_overrides(out,settings,self.arguments)
        old=(out/'evaluation_settings.json').read_bytes()
        er.record_evaluation_overrides(out,settings,self.arguments)
        with self.assertRaisesRegex(ValueError,'evaluation overrides differ'):
            er.record_evaluation_overrides(out,settings,('eval_batch_size=16',))
        self.assertEqual((out/'evaluation_settings.json').read_bytes(),old)

    def test_forbidden_cli_key_refused_before_folder_or_gpu_work(self):
        with patch.object(sys,'argv',['evaluate_run.py','missing','--set','front_end=srm']), \
             patch.object(er,'checkpoint_folder') as folder:
            with self.assertRaisesRegex(ValueError,'forbidden evaluation'):
                er.main()
        folder.assert_not_called()

    def test_cli_routes_same_overrides_to_main_and_temperature(self):
        argv=['evaluate_run.py',str(self.folder),'--test-suite','legacy-512','--only','main,padding,tanh,temperature']
        for arg in self.arguments:argv+=['--set',arg]
        with patch.object(sys,'argv',argv),patch.object(er,'evaluate') as main,patch.object(er,'calibrate') as temperature:
            er.main()
        self.assertEqual(main.call_args.kwargs['eval_overrides'],self.arguments)
        self.assertEqual(temperature.call_args.kwargs['eval_overrides'],self.arguments)

    def test_cli_routes_same_overrides_to_optional_stages(self):
        argv=['evaluate_run.py',str(self.folder),'--test-suite','legacy-512','--only','padding,tanh','--set','eval_batch_size=32']
        with patch.object(sys,'argv',argv),patch.object(er,'optional_stages') as optional:
            er.main()
        self.assertEqual(optional.call_args.kwargs['eval_overrides'],('eval_batch_size=32',))

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
