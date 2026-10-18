import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'LSB-Dataset-Generator')]
from scripts import evaluate_run as E
from cvdl import deployment as D
from cvdl.settings import machine_profile

RUN='xunet_v4_clip3_nonorm_lr0.0003_warm1_drops5x0.3-20x0.1_k1to4_e30_wide-w4_r2-20261015T121753Z-29499'
FOLDER=ROOT/'runs'/RUN/'checkpoints'/'xunet_v4_clip3_nonorm_lr0.0003_warm1_drops5x0.3-20x0.1_k1to4_e30_wide-w4_r2'


class LegacyMigrationTests(unittest.TestCase):
    def setUp(self):
        t=tempfile.TemporaryDirectory();self.addCleanup(t.cleanup)
        self.run=Path(t.name)/'historical';self.folder=self.run/'checkpoints'/FOLDER.name;self.folder.mkdir(parents=True)
        record=dict(settings=dict(sources=['raise','div2k'],source='decode',tile_size=512,run_seed=0,
            train_photo_fraction=1.,paired_batches=False),eval_profile='overlap_full_mix',data='otf',quick=False)
        model=dict(norm='none',clip_t=3.,lr=.0003,warmup_epochs=1,lr_drops=[[5,.3],[20,.1]],batch_size=32,
            model='xunet_wide',width=4,train_yaml='configs/train-k1to4.yaml',extra_tests='test-bpp1',max_epochs=30,grad_clip_norm=1.)
        (self.folder/'data_run_settings.json').write_text(json.dumps(record));(self.folder/'model_run_settings.json').write_text(json.dumps(model))
        (self.run/'executed.ipynb').write_text('{"fixture": "standard full-source validation"}')
        self.evidence={n:hashlib.sha256((self.run/n if n=='executed.ipynb' else self.folder/n).read_bytes()).hexdigest() for n in E.WIDE_R2_EVIDENCE}
        self.profile=machine_profile('pc',root=ROOT,device='cpu',precision_limits=('32',))

    def test_exact_evidenced_omissions_resolve_without_original_writes(self):
        before={p:p.read_bytes() for p in self.run.rglob('*') if p.is_file()}
        with patch.object(E,'WIDE_R2_EVIDENCE',self.evidence),patch('cvdl.settings.automatic_profile',return_value=self.profile):
            settings,context,snapshot=E.settings_for_evaluation(self.folder,self.folder/'unused.ckpt')
        self.assertEqual(settings.checkpoint_validation,'standard');self.assertIsNone(settings.source_mix);self.assertIsNone(settings.tiles_per_epoch)
        self.assertEqual(settings.sources,('raise','div2k'));self.assertEqual(context['legacy_settings_migration']['derived']['checkpoint_validation'],'standard')
        self.assertIsNone(snapshot);self.assertEqual(before,{p:p.read_bytes() for p in before})
        self.assertFalse((self.folder/'run_settings.json').exists())

    def test_unknown_missing_key_fails_no_generic_defaults(self):
        with self.assertRaisesRegex(ValueError,'no approved saved-record evidence'):E.legacy_data_record(self.folder)

    def test_missing_or_changed_notebook_evidence_fails(self):
        with patch.object(E,'WIDE_R2_EVIDENCE',self.evidence):
            p=self.run/'executed.ipynb';p.write_text('changed')
            with self.assertRaisesRegex(ValueError,'no approved'):E.legacy_data_record(self.folder)
            p.unlink()
            with self.assertRaisesRegex(ValueError,'no approved'):E.legacy_data_record(self.folder)

    def test_other_missing_fields_fail_even_in_evidenced_shape(self):
        p=self.folder/'data_run_settings.json';r=json.loads(p.read_text());del r['settings']['run_seed'];p.write_text(json.dumps(r))
        with patch.object(E,'WIDE_R2_EVIDENCE',self.evidence):
            with self.assertRaisesRegex(ValueError,'settings.run_seed'):E.legacy_data_record(self.folder)

    def test_complete_existing_record_is_not_migrated(self):
        p=self.folder/'data_run_settings.json';r=json.loads(p.read_text());r['checkpoint_validation']='standard';r['settings'].update(source_mix=None,tiles_per_epoch=None);p.write_text(json.dumps(r))
        resolved,evidence=E.legacy_data_record(self.folder);self.assertEqual(resolved,r);self.assertIsNone(evidence)

    @unittest.skipUnless((FOLDER/'data_run_settings.json').is_file(),'historical artifact unavailable')
    def test_actual_wide_r2_record_hydrates_and_mapping_stays_8328(self):
        before={name:(FOLDER/name).read_bytes() for name in ('data_run_settings.json','model_run_settings.json')}
        with patch('cvdl.settings.automatic_profile',return_value=self.profile):s=E.settings_for_run(FOLDER)
        self.assertEqual(s.checkpoint_validation,'standard');self.assertIsNone(s.source_mix);self.assertIsNone(s.tiles_per_epoch)
        sha='9ba63252e8eecbf398f646f2d69641b70ad37c914ed8b7d6b4665dda54a961e1'
        frame,identity=D.resolve(s,FOLDER/'best-epoch23-val_pe0.0177-val_auc0.9957.ckpt',sha)
        self.assertEqual(identity['recipe_id'],'8328cc6b0673cade0312');self.assertEqual(identity['threshold_origin'],'post_hoc_256')
        self.assertEqual(len(frame),18000);self.assertEqual(before,{name:(FOLDER/name).read_bytes() for name in before})

if __name__=='__main__':unittest.main()
