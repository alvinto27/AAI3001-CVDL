"""No-cloud shell tests, pinned preflight and G4 recipe parity; no GPU work."""
from contextlib import redirect_stdout
from copy import deepcopy
from dataclasses import asdict
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch

from cvdl.settings import SCHEMA, automatic_profile, load, machine_profile, plain
from scripts.g4.preflight import preflight, top_values, validate_notebook, validate_job_env
from scripts.run_notebook import settings_file

ROOT=Path(__file__).resolve().parents[1]
BOOTSTRAP_SHA='89cc83e9c73cc43476f6daf5e77825ddc298a32a22a52af8ded0909f15b1f2af'


def git(root,*args):
    return subprocess.check_output(['git',*args],cwd=root,text=True).strip()


class RecipeTests(unittest.TestCase):
    def test_two_notebooks_keep_recipe_except_post_fit_and_two_machine_globals(self):
        base=json.loads((ROOT/'notebooks/05_xunet_v4_nonorm.ipynb').read_text())
        for name in ('newdata-1','wide-1'):
            notebook=json.loads((ROOT/f'notebooks/{name}.ipynb').read_text())
            self.assertEqual(notebook['cells'][:1]+notebook['cells'][2:],base['cells'][:1]+base['cells'][2:])
            settings=load(environ={},_profile=machine_profile('g4'),**top_values(notebook))
            env=settings_file(ROOT/f'scripts/g4/jobs/archive/{name}.env',[])
            # Historical PC/ROCm context, with the explicit full G4 job values.
            legacy=load(environ=env,_profile=machine_profile('pc',rocm=True))
            self.assertEqual(settings.run_name,legacy.run_name)
            for key,spec in SCHEMA.items():
                if key in {'validation_recipe','init_checkpoint','init_checkpoint_sha256',
                           'embedding_schedule','embedding_stage_epochs','evaluation_checkpoint','pilot',
                           'cell_supervision','cell_loss_weight'}:
                    # Unset opt-ins must stay absent from both legacy records.
                    self.assertNotIn(key, settings.values)
                    self.assertNotIn(key, legacy.values)
                    continue
                if key=='post_fit_eval':
                    self.assertEqual((settings.values[key],legacy.values[key]),('none','full'),name)
                elif not spec.machine:
                    self.assertEqual(settings.values[key],legacy.values[key],f'{name}: {key}')
            self.assertEqual(plain(asdict(settings.to_run_settings())),
                             plain(asdict(legacy.to_run_settings())))
            actual=settings.legacy_globals()
            before=legacy.legacy_globals()
            changed={key for key in set(actual)|set(before) if actual.get(key)!=before.get(key)}
            self.assertEqual(changed,{'CHECKPOINT_ROOT','ROCM_CONV1_FIX_ACTIVE'})
            self.assertEqual(before['CHECKPOINT_ROOT'],str(ROOT/'runs'))
            self.assertEqual(actual['CHECKPOINT_ROOT'],str(ROOT/'runs-out'))
            self.assertIs(before['ROCM_CONV1_FIX_ACTIVE'],True)
            self.assertIs(actual['ROCM_CONV1_FIX_ACTIVE'],False)
            self.assertEqual((settings.workers,settings.eval_workers,settings.quick,settings.compile,settings.prime_next_epoch),
                             (32,16,False,'default',True))

    def test_profile_contradictions_fail_instead_of_falling_back(self):
        with patch('cvdl.settings.Path.read_text',return_value='Workstation'), \
             patch('torch.cuda.is_available',return_value=False),patch.dict(os.environ,{'CVDL_EXPECTED_PROFILE':'g4'}):
            with self.assertRaisesRegex(ValueError,'profile mismatch.*requires'):
                automatic_profile()
        with patch('cvdl.settings.Path.read_text',return_value='Google Compute Engine'), \
             patch('torch.cuda.is_available',return_value=False):
            with self.assertRaisesRegex(ValueError,'G4 profile mismatch'):
                automatic_profile()
        with patch('cvdl.settings.Path.read_text',return_value='Workstation'), \
             patch('torch.cuda.is_available',return_value=True),patch('torch.version.hip',None), \
             patch('torch.cuda.get_device_name',return_value='NVIDIA RTX PRO 6000'):
            with self.assertRaisesRegex(ValueError,'refuse PC fallback'):
                automatic_profile()

        with patch('cvdl.settings.Path.read_text',return_value='Google Compute Engine'), \
             patch('torch.cuda.is_available',return_value=True),patch('torch.version.hip',None), \
             patch('torch.cuda.get_device_name',return_value='NVIDIA RTX PRO 6000'), \
             patch.dict(os.environ,{'CVDL_EXPECTED_PROFILE':'pc'}):
            with self.assertRaisesRegex(ValueError,'requires.*pc.*detected.*g4'):
                automatic_profile()

    def test_installed_bootstrap_source_is_unchanged_and_execs_checkout(self):
        source=(ROOT/'scripts/g4/job.sh').read_text()
        self.assertEqual(hashlib.sha256(source.encode()).hexdigest(),BOOTSTRAP_SHA)
        self.assertNotIn('V3_NOTEBOOK',source)
        self.assertIn('target="$ROOT/scripts/g4/job_run.sh"',source)
        self.assertIn('exec /usr/bin/env bash "$target" "$@"',source)
        self.assertIn('scripts/run_notebook.py',(ROOT/'scripts/g4/job_run.sh').read_text())


class SubmissionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp=tempfile.TemporaryDirectory(prefix='cvdl-g4-tests-')
        parent=Path(cls.temp.name);cls.root=parent/'repo';cls.root.mkdir()
        paths=git(ROOT,'ls-files').splitlines()+[
            'scripts/g4/preflight.py','notebooks/newdata-1.ipynb','notebooks/wide-1.ipynb',
            'scripts/g4/jobs/archive/newdata-1.env','scripts/g4/jobs/archive/wide-1.env',
            'scripts/g4/jobs/archive/README.md']
        copied=[]
        for name in sorted(set(paths)):
            source=ROOT/name
            needed = (name == '.gitignore' or name.startswith(('cvdl/','configs/','notebooks/','scripts/g4/'))
                      or name in {'scripts/run_notebook.py','scripts/speed_matrix.py','scripts/host_monitor.py'}
                      or (name.startswith('LSB-Dataset-Generator/') and source.suffix in {'.py','.yaml','.txt'}))
            if source.is_file() and needed:
                target=cls.root/name;target.parent.mkdir(parents=True,exist_ok=True)
                shutil.copy2(source,target);copied.append(name)
        # A typo notebook is committed so its failure is preflight, not path/dirty validation.
        typo=json.loads((cls.root/'notebooks/newdata-1.ipynb').read_text())
        typo['cells'][1]['source']=''.join(typo['cells'][1]['source']).replace('epochs=30','epohs=30').splitlines(True)
        (cls.root/'notebooks/typo.ipynb').write_text(json.dumps(typo));copied.append('notebooks/typo.ipynb')
        (cls.root/'notebooks/link.ipynb').symlink_to('05_xunet_v4_nonorm.ipynb');copied.append('notebooks/link.ipynb')
        git(cls.root,'init','-q');git(cls.root,'config','user.name','G4 tests');git(cls.root,'config','user.email','g4-tests@example.invalid')
        git(cls.root,'add','--',*copied);git(cls.root,'-c','core.hooksPath=/dev/null','commit','-qm','published fixture')
        cls.commit=git(cls.root,'rev-parse','HEAD')
        # Local fixture remote-tracking ref, without pushing or contacting a remote.
        git(cls.root,'update-ref','refs/remotes/origin/yx',cls.commit)
        tools=parent/'bin';tools.mkdir()
        wrapper=tools/'podman'
        wrapper.write_text(f'''#!{sys.executable}
import os, pathlib, subprocess, sys
root=pathlib.Path(os.environ['TEST_REPO'])
a=sys.argv[1:]
if a[:2]==['exec','-i']:del a[1]
assert a[:4]==['exec','-w','/work','aai3001'], a
a=a[4:]
if 'scripts/run_notebook.py' in a:
    pathlib.Path(os.environ['RUN_MARKER']).write_text(' '.join(a))
    raise SystemExit(0)
a=[str(root/x[6:]) if x.startswith('/work/') else x for x in a]
a[0]=sys.executable
raise SystemExit(subprocess.call(a,cwd=root))
''');wrapper.chmod(0o755)
        cloud=tools/'gcloud';cloud.write_text('#!/bin/sh\necho UNEXPECTED_CLOUD_CALL >&2\nexit 99\n');cloud.chmod(0o755)
        cls.marker=parent/'runner-called'
        cls.env={**os.environ,'PATH':str(tools)+os.pathsep+os.environ['PATH'],
                 'TEST_REPO':str(cls.root),'RUN_MARKER':str(cls.marker)}

    @classmethod
    def tearDownClass(cls):cls.temp.cleanup()

    def submit(self,*args):
        return subprocess.run(['bash','scripts/g4/submit.sh','run',*args,'--dry-run'],
                              cwd=self.root,env=self.env,capture_output=True,text=True)

    def test_dry_run_prints_g4_settings_and_only_minimal_job_env(self):
        done=self.submit('--notebook','notebooks/newdata-1.ipynb')
        self.assertEqual(done.returncode,0,done.stdout+done.stderr)
        self.assertIn('"profile": "g4"',done.stdout)
        self.assertIn('"workers": 32',done.stdout)
        self.assertIn('wide-w4_mixr2',done.stdout)
        block=done.stdout.split('job-env (generated; no experiment values):\n')[1]
        lines=[line.strip() for line in block.splitlines() if line.startswith('  ') and '=' in line]
        self.assertEqual(len(lines),2)
        self.assertEqual(lines[0],'V3_NOTEBOOK=notebooks/newdata-1.ipynb')
        self.assertTrue(lines[1].startswith('CVDL_JOB_ID=newdata-1-'))
        self.assertIn('gcloud compute instances start',done.stdout)
        self.assertNotIn('UNEXPECTED_CLOUD_CALL',done.stderr)

    def test_job_id_has_single_separator_and_exactly_four_random_digits(self):
        done=self.submit('--notebook','notebooks/newdata-1.ipynb')
        self.assertEqual(done.returncode,0,done.stdout+done.stderr)
        job_id=next(line.split('=',1)[1] for line in done.stdout.splitlines() if line.startswith('job-id='))
        self.assertRegex(job_id,r'^newdata-1-[0-9]{8}T[0-9]{6}Z-[0-9]{4}$')
        self.assertNotIn('--',job_id)
        self.assertIn(f'  CVDL_JOB_ID={job_id}\n',done.stdout)

    def test_bad_archive_untracked_and_typo_fail_before_cloud(self):
        for path in ('../bad.ipynb','notebooks/archive/05_baseline_kv_xunet_v4_pre-cvdl.ipynb',
                     'notebooks/missing.ipynb','notebooks/typo.ipynb','notebooks/link.ipynb'):
            with self.subTest(path=path):
                done=self.submit('--notebook',path)
                self.assertNotEqual(done.returncode,0)
                self.assertNotIn('gcloud compute instances start',done.stdout)
                if path.endswith('/typo.ipynb'):
                    self.assertIn('did you mean',done.stderr);self.assertIn('epochs',done.stderr)

    def test_dirty_and_unpublished_refused_even_in_dry_run(self):
        path=self.root/'dirty.txt';path.write_text('dirty')
        try:self.assertIn('Dirty checkout',self.submit('--notebook','notebooks/newdata-1.ipynb').stderr)
        finally:path.unlink()
        other=git(self.root,'-c','user.name=G4 tests','-c','user.email=g4-tests@example.invalid',
                  'commit-tree','HEAD^{tree}','-p','HEAD','-m','unpublished')
        done=self.submit('--notebook','notebooks/newdata-1.ipynb','--commit',other)
        self.assertNotEqual(done.returncode,0);self.assertIn('not published',done.stderr)

    def test_speed_tool_keeps_legacy_env_dispatch(self):
        done=self.submit('--env-file','scripts/g4/jobs/speed-1.env','--timeout-min','50')
        self.assertEqual(done.returncode,0,done.stderr)
        self.assertIn('scripts/speed_matrix.py',done.stdout)
        done=self.submit('--env-file','scripts/g4/jobs/archive/newdata-1.env')
        self.assertNotEqual(done.returncode,0)
        self.assertIn('reserved for the speed matrix',done.stderr)

    def test_pinned_commit_not_working_tree_is_preflight_source(self):
        path=self.root/'notebooks/newdata-1.ipynb';before=path.read_bytes()
        path.write_text('bad JSON in working tree')
        try:
            report=preflight('notebooks/newdata-1.ipynb',self.commit,root=self.root)
            self.assertTrue(report['run_name'].endswith('wide-w4_mixr2'))
            self.assertEqual(report['values']['quick'],False)
        finally:path.write_bytes(before)

    def test_old_commit_metadata_incompatibility_fails_before_boot(self):
        source=(self.root/'cvdl/settings.py').read_text()
        old='\n'.join(line for line in source.splitlines() if "'job_id': Definition" not in line)+'\n'
        blob=subprocess.check_output(['git','hash-object','-w','--stdin'],cwd=self.root,input=old,text=True).strip()
        with tempfile.TemporaryDirectory() as folder:
            env={**os.environ,'GIT_INDEX_FILE':str(Path(folder)/'index')}
            for args in (['read-tree','HEAD'],['update-index','--cacheinfo','100644',blob,'cvdl/settings.py']):
                subprocess.check_call(['git',*args],cwd=self.root,env=env)
            tree=subprocess.check_output(['git','write-tree'],cwd=self.root,env=env,text=True).strip()
            commit=git(self.root,'commit-tree',tree,'-p','HEAD','-m','old metadata contract')
        with self.assertRaisesRegex(ValueError,'predates.*metadata interface'):
            preflight('notebooks/newdata-1.ipynb',commit,root=self.root)

    def run_job(self, notebook, extra=''):
        if self.marker.exists():self.marker.unlink()
        state=self.root/'runs/g4-test-state';state.mkdir(parents=True,exist_ok=True)
        job='unit-'+hashlib.sha256((notebook+extra).encode()).hexdigest()[:8]
        env=state/(job+'.input');env.write_text(f'V3_NOTEBOOK={notebook}\nCVDL_JOB_ID={job}\n'+extra)
        metadata=state/(job+'.metadata');metadata.write_text(f'job-mode=run\njob-id={job}\njob-commit=HEAD\njob-env={env}\njob-timeout-min=0.05\nmax-run-minutes=120\n')
        environment={**self.env,'G4_REPO_ROOT':str(self.root),'G4_STATE_ROOT':str(state),
                     'G4_LIB_DIR':str(state/'lib')}
        return subprocess.run(['bash','scripts/g4/job.sh','--test','--metadata-file',str(metadata),'--retry-seconds','0'],
                              cwd=self.root,env=environment,capture_output=True,text=True)

    def test_bootstrap_uses_checkout_and_job_validates_before_runner(self):
        done=self.run_job('notebooks/wide-1.ipynb')
        self.assertEqual(done.returncode,0,done.stdout+done.stderr)
        self.assertIn('job_run copy: checkout',done.stdout)
        self.assertTrue(self.marker.exists())
        self.assertIn('--notebook notebooks/wide-1.ipynb',self.marker.read_text())
        done=self.run_job('notebooks/archive/05_baseline_kv_xunet_v4_pre-cvdl.ipynb')
        self.assertNotEqual(done.returncode,0)
        self.assertFalse(self.marker.exists())
        done=self.run_job('notebooks/wide-1.ipynb','V3_LR=.1\n')
        self.assertNotEqual(done.returncode,0)
        self.assertFalse(self.marker.exists())


if __name__=='__main__':unittest.main()
