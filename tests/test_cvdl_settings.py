"""Settings schema errors, provenance and QUICK extra caps; no GPU runs."""
from dataclasses import FrozenInstanceError
import contextlib
import io
import json
import os
from pathlib import Path
import tempfile
import subprocess
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import pandas as pd

from cvdl.settings import (ROOT, SCHEMA, MachineProfile, automatic_profile, load,
                           extra_panel_limit, plain)
from scripts.run_notebook import settings_file
from cvdl.evaluation import panel_dataset


def profile(device='cpu'):
    return MachineProfile('pc', dict(root=str(ROOT), device=device, rocm=True,
        precision_limits=('32','bf16-mixed') if device=='cuda' else ('32',),
        quick=True, workers=6, eval_workers=4))


def resolve(**kwargs):
    return load(environ={}, _profile=profile(), **kwargs)


class SettingsTests(unittest.TestCase):
    def test_typo_bad_value_and_cross_checks(self):
        for values, message in [({'epohs':3}, 'did you mean.*epochs'),
                                ({'norm':'grop'}, 'did you mean.*group'),
                                ({'epochs':True}, 'expected int'),
                                ({'epochs':0}, 'allowed >0'),
                                ({'epochs':float('inf')}, 'setting epochs=.*allowed >0'),
                                ({'batch_size':10}, 'multiple of 4'),
                                ({'lr':float('nan')}, 'finite'),
                                ({'lr':0}, 'allowed >0'),
                                ({'epochs':2,'warmup_epochs':3}, 'warmup_epochs'),
                                ({'lr_drops':'2:0.1,2:0.1'}, 'increasing'),
                                ({'lr_drops':'2:2'}, 'factors')]:
            with self.subTest(values=values), self.assertRaisesRegex(ValueError,message):
                resolve(**values)
        self.assertEqual(resolve(epochs=1,lr_drops=None).lr_drops, ((1,.1),))
        with self.assertRaisesRegex(ValueError,'did you mean.*V3_EPOCHS'):
            load(environ={'V3_EPOHS':'3'},_profile=profile())

    def test_notebook_cannot_set_machine_fields_or_bypass_limits(self):
        for key in ('workers','eval_workers','quick','precision','output_root','root'):
            with self.subTest(key=key), self.assertRaisesRegex(ValueError,'machine-owned'):
                resolve(**{key:SCHEMA[key].default})
        self.assertEqual(resolve(overrides=['workers=2']).workers,2)
        with self.assertRaisesRegex(ValueError,'profile limits'):
            resolve(overrides=['precision=bf16-mixed'])
        with self.assertRaisesRegex(ValueError,'read-only'):
            resolve(overrides=['precision_limits=bf16-mixed'])

    def test_automatic_pc_g4_defaults(self):
        for dmi, name, workers, eval_workers, quick in [('Google Compute Engine','g4',32,16,False),
                                                       ('Workstation','pc',6,4,True)]:
            with patch.object(Path,'read_text',return_value=dmi), \
                 patch('torch.cuda.is_available',return_value=name=='g4'), \
                 patch('torch.cuda.get_device_name',return_value='NVIDIA RTX PRO 6000 Blackwell'), \
                 patch('torch.cuda.is_bf16_supported',return_value=False), patch('torch.version.hip',None):
                machine = automatic_profile()
                actual = load(environ={},_profile=machine)
            self.assertEqual((actual.profile,actual.workers,actual.eval_workers,actual.quick),
                             (name,workers,eval_workers,quick))
            self.assertEqual(actual.provenance['workers'],'profile')
            self.assertEqual(actual.precision_limits,('32',))
            self.assertTrue(actual.output_root.endswith('/runs-out' if name=='g4' else '/runs'))

    def test_priority_history_frozen_and_run_record(self):
        result = load(lr=.001, environ={'V3_LR':'.002'}, overrides=['lr=.003','V3_LR=.004'], _profile=profile())
        self.assertEqual(result.lr,.004)
        self.assertEqual(result.provenance['lr'],'override')
        self.assertEqual([v['value'] for v in result.overrides],[.003,.004])
        self.assertEqual(load(environ={'V3_LR':'.002'},_profile=profile()).provenance['lr'],'env')
        self.assertEqual(resolve(lr=.001).provenance['lr'],'notebook')
        self.assertEqual(resolve().provenance['lr'],'default')
        with self.assertRaises(FrozenInstanceError): result.lr=.01
        with self.assertRaises(TypeError): result.values['lr']=.01
        mixed=resolve(source_mix={'raise':.5,'alaska':.25,'lsdir':.25})
        with self.assertRaises(TypeError): mixed.source_mix['raise']=1.
        with tempfile.TemporaryDirectory() as tmp:
            record=result.write(tmp)
            self.assertEqual(json.loads((Path(tmp)/'run_settings.json').read_text()),record)
            self.assertEqual(record['values'],plain(result.values))
            self.assertEqual(set(record['sources']),set(record['values']))
            self.assertEqual(record['sources']['lr'],'override')
            self.assertEqual(len(record['git']['commit']),40)
            self.assertIsInstance(record['git']['dirty'],bool)

    def test_notebook_env_conflicts_require_explicit_override(self):
        with self.assertRaisesRegex(ValueError, r"lr.*0\.001.*V3_LR='\.002'.*--set"):
            load(lr=.001,environ={'V3_LR':'.002'},_profile=profile())
        equal=load(lr=.001,environ={'V3_LR':'0.001'},_profile=profile())
        self.assertEqual(equal.lr,.001)
        self.assertEqual(equal.provenance['lr'],'notebook')
        changed=load(lr=.001,environ={'V3_LR':'.002'},overrides=['lr=.003'],_profile=profile())
        self.assertEqual(changed.lr,.003)
        self.assertEqual(changed.provenance['lr'],'override')
        self.assertEqual(resolve().provenance['main_threads'],'default')

    def test_runner_repeated_canonical_and_legacy_overrides(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'job.env';path.write_text('QUICK=false\nV3_LR=.0003\n')
            env=settings_file(path,['lr=.001','V3_LR=.002','workers=2'])
            self.assertEqual(env['V3_LR'],'0.002')
            self.assertEqual(env['V3_WORKERS'],'2')
            result=load(environ=env,overrides=['lr=.001','V3_LR=.002','workers=2'],_profile=profile())
            self.assertEqual(result.lr,.002)
            self.assertEqual(result.provenance['lr'],'override')
            self.assertEqual(result.provenance['workers'],'override')
            self.assertEqual(len(result.overrides),3)
            with self.assertRaisesRegex(ValueError,'did you mean'):
                settings_file(path,['epohs=3'])
            path.write_text('QUICK=false\nV3_RUN_NAME=old-name\n')
            reset=settings_file(path,['lr=.002','run_name='])
            self.assertNotIn('V3_RUN_NAME',reset)
            renamed=load(environ=reset,overrides=['lr=.002','run_name='],_profile=profile())
            self.assertIn('lr0.002',renamed.run_name)
            self.assertEqual(renamed.provenance['run_name'],'override')

    def test_runner_writes_override_provenance_without_training(self):
        from scripts import run_notebook as runner
        def shell_instance(user_ns, **kwargs):
            def run_cell(code, **kwargs):
                exec(code,user_ns)
                return SimpleNamespace(error_before_exec=None,error_in_exec=None)
            return SimpleNamespace(enable_matplotlib=lambda *args:None, run_cell=run_cell,
                display_pub=SimpleNamespace(publish=lambda *args:None),
                display_formatter=SimpleNamespace(formatters={'image/png':SimpleNamespace(enabled=False)}))
        previous=Path.cwd()
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ,{}):
            root=Path(tmp); subprocess.check_call(['git','init','-q',str(root)])
            # The runner now records the pinned source commit, as real G4 jobs do.
            subprocess.check_call(['git','-C',str(root),'-c','user.name=Runner fixture',
                '-c','user.email=fixture@localhost','-c','core.hooksPath=/dev/null',
                'commit','--allow-empty','-qm','source identity fixture'])
            (root/'.gitignore').write_text('runs/\n')
            notebook=root/'notebooks/05_xunet_v4_nonorm.ipynb';notebook.parent.mkdir()
            name='xunet_v4_clip3_nonorm_lr0.001_warm1_drop1_e3_otf'
            notebook.write_text(json.dumps({'cells':[{'cell_type':'code',
                'source':['from cvdl.settings import load\nsettings = load()\nRUN_NAME = settings.run_name'],
                'metadata':{}}], 'metadata':{}}))
            env=root/'job.env';env.write_text('QUICK=false\n')
            args=SimpleNamespace(notebook=notebook,env_file=env,set=['lr=.001','epochs=3'],
                                 output=root/'runs/test',resume=False,host_monitor='off')
            try:
                with patch.object(runner,'ROOT',root), patch('cvdl.settings.automatic_profile',return_value=profile()), \
                     patch('IPython.terminal.interactiveshell.TerminalInteractiveShell.instance',side_effect=shell_instance), \
                     contextlib.redirect_stdout(io.StringIO()):
                    self.assertEqual(runner.run(args),0)
                record=json.loads((args.output/'checkpoints'/name/'run_settings.json').read_text())
                self.assertEqual(record['values']['lr'],.001)
                self.assertEqual(record['values']['epochs'],3)
                self.assertEqual(record['sources']['lr'],'override')
                self.assertEqual(record['sources']['epochs'],'override')
                self.assertEqual(record['values']['run_name'],name)
                self.assertEqual(record['overrides'][:2],[{'name':'lr','value':.001},{'name':'epochs','value':3}])
            finally:
                os.chdir(previous)

    def test_quick_extra_panels_are_capped_and_deterministic(self):
        class FakePairs:
            frame=pd.DataFrame({'source':['raise']*100,'camera_model':['A','B']*50})
            def __len__(self):return len(self.frame)
        dataset=FakePairs()
        self.assertIsNone(extra_panel_limit(environ={}))
        self.assertEqual(extra_panel_limit(environ={'QUICK':'true'}),16)
        self.assertIs(panel_dataset(dataset,quick=False),dataset)
        with patch.dict('os.environ',{'QUICK':'true'}):
            # Package API is explicit and must not inherit the legacy CLI environment.
            self.assertIs(panel_dataset(dataset,quick=False),dataset)
            small=panel_dataset(dataset,quick=True)
            self.assertEqual(len(small),16)
            self.assertEqual(small.indices,panel_dataset(dataset,quick=True).indices)


if __name__=='__main__':
    unittest.main()
