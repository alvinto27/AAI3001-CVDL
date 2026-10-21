"""Evaluation overrides are explicit, narrow, recorded and never saved as training inputs."""
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from cvdl.settings import machine_profile, plain
from scripts.archive import evaluate_legacy_512 as er


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


    def test_no_override_preserves_values_sources_and_metadata(self):
        before=er.settings_for_run(self.folder);after,context,snapshot=self.resolve()
        self.assertEqual(plain(before.values),plain(after.values))
        self.assertEqual(plain(before.provenance),plain(after.provenance))
        self.assertEqual(context,{})
        er.record_evaluation_overrides(self.root/'absent',after,())
        self.assertFalse((self.root/'absent').exists())
        self.assertEqual(self.saved.read_bytes(),self.original)


    def test_override_record_is_idempotent_and_refuses_changed_inputs(self):
        settings,_,_=self.resolve(self.arguments);out=self.root/'output'
        er.record_evaluation_overrides(out,settings,self.arguments)
        old=(out/'evaluation_settings.json').read_bytes()
        er.record_evaluation_overrides(out,settings,self.arguments)
        with self.assertRaisesRegex(ValueError,'evaluation overrides differ'):
            er.record_evaluation_overrides(out,settings,('eval_batch_size=16',))
        self.assertEqual((out/'evaluation_settings.json').read_bytes(),old)


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


if __name__=='__main__':unittest.main()
