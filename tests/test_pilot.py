"""PC pilot isolation, unchanged real recipe guards, G4/evaluation refusals."""
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import pandas as pd
from cvdl import runtime, validation
from cvdl.pilot import MAX_TILES, SLICE_SEED, VALIDATION_PAIRS, refuse_run
from cvdl.settings import load, machine_profile, plain
from scripts.g4.preflight import top_values, preflight, validate_job_env

ROOT = Path(__file__).resolve().parents[1]
RID = '60da4391043e8ce48ff2'


def settings(**changes):
    values = top_values(json.loads((ROOT/'notebooks/experiments/srm-delayed.ipynb').read_text()))
    values.update(changes)
    return load(environ={}, _profile=machine_profile('pc', device='cpu'), **values)


class PilotTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.settings = settings(pilot=True)
        cls.paths = list((ROOT/'data/val'/RID).iterdir())
        cls.before = {p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in cls.paths}
        with patch('lsb_torch.Corpus.photo', side_effect=AssertionError('pilot setup must not decode')):
            cls.otf, _ = runtime.create_data(cls.settings)
        cls.full, _, _, _ = validation.load_frozen(ROOT, cls.settings.to_run_settings(),
            cls.settings.train_grid, cls.settings.train_tile_size, RID)

    def test_fixed_slice_and_effective_budget_without_changing_declared_settings(self):
        s = self.settings;otf = self.otf
        self.assertEqual(s.tiles_per_epoch,180000)
        self.assertEqual(otf.corpus.settings.tiles_per_epoch,MAX_TILES)
        self.assertEqual(len(otf.train),2*MAX_TILES)
        self.assertEqual(len(otf.val),VALIDATION_PAIRS)
        self.assertEqual(set(otf.val.frame.tile_size),{256})
        self.assertTrue(set(otf.val.frame.pair_id)<=set(self.full.pair_id))
        expected = self.full.set_index('pair_id').loc[otf.val.frame.pair_id].reset_index()
        pd.testing.assert_frame_equal(otf.val.frame[expected.columns],expected)
        marker = otf.record['pilot']
        self.assertEqual((marker['declared_tiles_per_epoch'],marker['effective_tiles_per_epoch']),(180000,4096))
        self.assertEqual(marker['slice_seed'],SLICE_SEED)
        self.assertEqual(marker['pair_ids'],otf.val.frame.pair_id.tolist())
        self.assertEqual(marker['slice_sha256'],hashlib.sha256(otf.validation_pair_csv.encode()).hexdigest())
        self.assertEqual(marker['full_validation_pairs'],18000)
        self.assertFalse(marker['deployment_eligible'])
        self.assertEqual(otf.record['settings']['tiles_per_epoch'],4096)
        self.assertEqual(self.before,{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in self.paths})

    def test_slice_does_not_depend_on_quick_limit_or_training_seed(self):
        other = settings(pilot=True,run_seed=891,overrides=['quick_eval_pairs=4'])
        otf,_ = runtime.create_data(other)
        pd.testing.assert_frame_equal(otf.val.frame,self.otf.val.frame)
        self.assertEqual(otf.record['pilot']['slice_sha256'],self.otf.record['pilot']['slice_sha256'])

    def test_real_declared_budget_guard_is_not_bypassed(self):
        wrong = settings(pilot=True,tiles_per_epoch=180004)
        with self.assertRaisesRegex(ValueError,'does not match the training data recipe'):
            runtime.create_data(wrong)
        with patch('cvdl.validation.load_frozen',side_effect=ValueError('SHA256 mismatch')):
            with self.assertRaisesRegex(ValueError,'SHA256 mismatch'):
                runtime.create_data(self.settings)

    def test_default_absence_and_no_rewrite_of_training_controls(self):
        normal = settings();pilot = settings(pilot=True)
        self.assertNotIn('pilot',normal.values)
        self.assertNotIn('pilot',normal.provenance)
        self.assertEqual({k:v for k,v in plain(pilot.values).items() if k!='pilot'},plain(normal.values))
        with TemporaryDirectory() as folder:
            record = pilot.write(folder)
            self.assertTrue(record['values']['pilot'])
        for changes in (dict(checkpoint_validation='capped',validation_recipe=None),dict(post_fit_eval='full')):
            with self.assertRaisesRegex(ValueError,'pilot mode needs'):
                settings(pilot=True,**changes)

    def test_g4_settings_preflight_and_job_runner_refuse(self):
        values = top_values(json.loads((ROOT/'notebooks/experiments/srm-delayed.ipynb').read_text()))
        values['pilot']=True
        with self.assertRaisesRegex(ValueError,'PC-only'):
            load(environ={},_profile=machine_profile('g4'),**values)
        document={'cells':[dict(cell_type='code',source=[
            'from cvdl.settings import load\nsettings = load(name="pilot", pilot=True)'])]}
        with patch('scripts.g4.preflight.validate_notebook',return_value=document):
            with self.assertRaisesRegex(ValueError,'PC-only'):
                preflight('notebooks/experiments/pilot.ipynb','HEAD')
            with TemporaryDirectory() as tmp:
                env=Path(tmp)/'job.env';env.write_text('V3_NOTEBOOK=notebooks/experiments/pilot.ipynb\nCVDL_JOB_ID=pilot-job\n')
                with self.assertRaisesRegex(ValueError,'PC-only'):
                    validate_job_env(env,'pilot-job')

    def test_evaluator_and_report_builder_refuse_pilot_records(self):
        from scripts.evaluate_run import checkpoint_folder, settings_for_run
        from cvdl.reporting import SavedReport
        with TemporaryDirectory() as tmp:
            root=Path(tmp);run=root/'runs/pilot';folder=run/'checkpoints/pilot';folder.mkdir(parents=True)
            for filename,record in [('run_settings.json',{'values':{'pilot':True}}),
                                    ('data_run_settings.json',{'pilot':{'mode':'pc-pilot-v1'}})]:
                path=folder/filename;path.write_text(json.dumps(record))
                for function,args in [(refuse_run,(run,)),(checkpoint_folder,(run,)),
                    (settings_for_run,(folder,)),(SavedReport,(root,'pilot'))]:
                    with self.assertRaisesRegex(ValueError,'pilot runs are smoke tests'):
                        function(*args)
                path.unlink()
            refuse_run(run)


if __name__=='__main__':unittest.main()
