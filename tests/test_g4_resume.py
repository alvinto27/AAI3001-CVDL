"""CPU-only G4 resume gates, durable writes and mocked cloud/boot decisions."""
import hashlib
import json
import os
import signal
import time
from pathlib import Path
import stat
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from cvdl.checkpoint_io import DurableCheckpointIO
from scripts.g4 import resume
from scripts import run_notebook
import test_g4_jobs as jobs

ROOT=Path(__file__).resolve().parents[1]


def checkpoint(path):
    path.parent.mkdir(parents=True,exist_ok=True)
    DurableCheckpointIO().save_checkpoint(dict(state_dict={'x':torch.tensor([1.])},optimizer_states=[{}],
        lr_schedulers=[{}],loops={},epoch=6,global_step=7),path)


class CheckpointTests(unittest.TestCase):
    def test_same_folder_replace_and_file_then_folder_fsync_for_every_save(self):
        with tempfile.TemporaryDirectory() as folder:
            events=[];real_sync=os.fsync;real_replace=os.replace
            def sync(fd):
                events.append('folder' if stat.S_ISDIR(os.fstat(fd).st_mode) else 'file')
                return real_sync(fd)
            def replace(source,target):
                self.assertEqual(Path(source).parent,Path(target).parent)
                events.append('replace');return real_replace(source,target)
            with patch('cvdl.checkpoint_io.os.fsync',side_effect=sync),patch('cvdl.checkpoint_io.os.replace',side_effect=replace):
                for name in ('best-epoch00.ckpt','last.ckpt'):
                    path=Path(folder)/name;checkpoint(path)
                    self.assertEqual(torch.load(path,weights_only=False)['global_step'],7)
            self.assertEqual(events,['file','replace','folder']*2)
            self.assertEqual(sorted(p.name for p in Path(folder).iterdir()),['best-epoch00.ckpt','last.ckpt'])

    def test_failed_serialization_or_replace_preserves_old_checkpoint_and_cleans_temp(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'last.ckpt';checkpoint(path);before=path.read_bytes()
            for operation in ('torch.save','os.replace'):
                with patch('cvdl.checkpoint_io.'+operation,side_effect=OSError('interrupted')):
                    with self.assertRaises(OSError):checkpoint(path)
                self.assertEqual(path.read_bytes(),before)
                self.assertEqual(list(path.parent.iterdir()),[path])
            with patch('cvdl.checkpoint_io.os.fsync',side_effect=OSError('disk error')):
                with self.assertRaises(OSError):checkpoint(path)
            self.assertEqual(path.read_bytes(),before)

    def test_checkpoint_requires_full_readable_last_and_records_exact_sha(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);path=root/'checkpoints/job/last.ckpt'
            with self.assertRaises(ValueError):resume.checkpoint_info(root)
            checkpoint(path);info=resume.checkpoint_info(root)
            self.assertEqual((info['epoch'],info['step']),(6,7))
            self.assertEqual(info['checkpoint_sha256'],hashlib.sha256(path.read_bytes()).hexdigest())
            torch.save({'state_dict':{}},path)
            with self.assertRaisesRegex(ValueError,'full training state'):resume.checkpoint_info(root)
            path.write_bytes(b'truncated')
            with self.assertRaises(Exception):resume.checkpoint_info(root)

    def test_outcome_does_not_confuse_upload_with_training_completion(self):
        with tempfile.TemporaryDirectory() as folder:
            status=Path(folder)/'status.json'
            for code,expected in [('0','done'),('1','failed'),('124','timeout'),('130','interrupted'),('143','interrupted'),('137','failed')]:
                self.assertEqual(resume.outcome(folder,code),expected)
            status.write_text(json.dumps(dict(state='running',exit_code=None)))
            self.assertEqual(resume.outcome(folder,'unknown'),'interrupted')
            status.write_text(json.dumps(dict(state='failed',exit_code=1)))
            self.assertEqual(resume.outcome(folder,'unknown'),'failed')


class NotebookGuards(unittest.TestCase):
    def test_resume_missing_commit_notebook_settings_or_checkpoint_never_starts_fresh(self):
        notebook=ROOT/'notebooks/experiments/srm-e50-paired.ipynb'
        parent=ROOT/'runs/.scratch';parent.mkdir(parents=True,exist_ok=True)
        for wrong in ('missing-status','commit','notebook','settings','overrides','missing-checkpoint','corrupt-checkpoint'):
            with self.subTest(wrong=wrong),tempfile.TemporaryDirectory(dir=parent,prefix='resume-guard-') as folder:
                output=Path(folder);job='srm-e50-paired-20261023T190000Z-0000'
                env=output/'input.env';env.write_text(f'V3_NOTEBOOK={notebook.relative_to(ROOT)}\nCVDL_JOB_ID={job}\n')
                prior=dict(state='running',executed_notebook=str(output/'executed.ipynb'),
                    source_commit=jobs.git(ROOT,'rev-parse','HEAD'),notebook_sha256=hashlib.sha256(notebook.read_bytes()).hexdigest(),
                    effective_settings=dict(V3_NOTEBOOK=str(notebook.relative_to(ROOT)),CVDL_JOB_ID=job,QUICK='false',V3_OUTPUT_ROOT=str(output/'checkpoints')),
                    setting_overrides=[])
                if wrong=='commit':prior['source_commit']='0'*40
                if wrong=='notebook':prior['notebook_sha256']='0'*64
                if wrong=='settings':prior['effective_settings']['QUICK']='true'
                if wrong=='overrides':prior['setting_overrides']=['workers=1']
                if wrong!='missing-status':(output/'status.json').write_text(json.dumps(prior))
                if wrong=='corrupt-checkpoint':
                    path=output/'checkpoints'/job/'last.ckpt';path.parent.mkdir(parents=True);path.write_bytes(b'bad')
                before={p.relative_to(output):p.read_bytes() for p in output.rglob('*') if p.is_file()}
                args=SimpleNamespace(notebook=notebook,env_file=env,output=output,set=[],resume=True)
                with self.assertRaises(Exception):run_notebook.run(args)
                self.assertEqual(before,{p.relative_to(output):p.read_bytes() for p in output.rglob('*') if p.is_file()})


class RunnerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        jobs.SubmissionTests.setUpClass.__func__(cls)
        tools=Path(cls.env['PATH'].split(os.pathsep)[0])
        cloud=tools/'gcloud'
        cloud.write_text(f'''#!{os.sys.executable}
import json,os,sys
from pathlib import Path
if 'describe' in sys.argv:
    if '--format=json' in sys.argv:print(Path(os.environ['MOCK_METADATA']).read_text())
    else:print(json.loads(Path(os.environ['MOCK_METADATA']).read_text())['status'])
else:
    with open(os.environ['CLOUD_MARKER'],'a') as f:f.write(' '.join(sys.argv[1:])+'\\n')
''');cloud.chmod(0o755)
        wrapper=tools/'podman';text=wrapper.read_text()
        text=text.replace("    raise SystemExit(0)","""    out=root/a[a.index('--output')+1]
    out.mkdir(parents=True,exist_ok=True)
    with (out/'run.log').open('a') as f:f.write('MOCK RUN\\n')
    import time; time.sleep(float(os.environ.get('MOCK_RUN_SLEEP','0')))
    raise SystemExit(int(os.environ.get('MOCK_RUN_EXIT','0')))""")
        wrapper.write_text(text)

    @classmethod
    def tearDownClass(cls):cls.temp.cleanup()

    def prepare(self,state='interrupted',attempt=0):
        self._prepare_index=getattr(self,'_prepare_index',0)+1
        job='unit-resume-'+self._testMethodName.replace('test_','')+'-'+str(self._prepare_index)
        root=self.root;area=root/'runs/resume-tests'/job;area.mkdir(parents=True,exist_ok=True)
        output=root/'runs-out'/job;output.mkdir(parents=True,exist_ok=True)
        envfile=area/'input.env';notebook='notebooks/newdata-1.ipynb'
        envfile.write_text(f'V3_NOTEBOOK={notebook}\nCVDL_JOB_ID={job}\n')
        record=area/'jobs'/f'{job}.record';record.parent.mkdir(exist_ok=True)
        record.write_text(f'job-id={job}\nstate=delivered\ntraining-state={state}\nprotocol=resume-v1\ncommit={self.commit}\nnotebook={notebook}\nattempt={attempt}\nenv-sha256={hashlib.sha256(envfile.read_bytes().rstrip(b'\n')).hexdigest()}\nstarted-at=2026-10-23T00:00:00Z\ndelivered=yes\n')
        checkpoint(output/'checkpoints'/job/'last.ckpt')
        (output/'run.log').write_text('ORIGINAL\n')
        (output/'status.json').write_text(json.dumps(dict(state='running',exit_code=None)))
        metadata=area/'metadata';metadata.write_text(f'job-mode=resume\njob-id={job}\njob-commit={self.commit}\njob-env={envfile}\njob-attempt={attempt+1}\njob-timeout-min=1\nmax-run-minutes=120\n')
        if self.marker.exists():self.marker.unlink()
        return SimpleNamespace(job=job,area=area,record=record,output=output,metadata=metadata,envfile=envfile)

    def boot(self,f,code=0,delivery=False):
        return subprocess.run(['bash','scripts/g4/job.sh','--test','--metadata-file',str(f.metadata),'--retry-seconds','2' if delivery else '0','--inbox-key',str(f.area/('fake-key' if delivery else 'no-key'))],
            cwd=self.root,env={**self.env,'G4_REPO_ROOT':str(self.root),'G4_STATE_ROOT':str(f.area),'MOCK_RUN_EXIT':str(code)},text=True,capture_output=True,timeout=45)

    def test_authorized_resume_same_output_and_counter_prevents_repeat_after_reboot(self):
        f=self.prepare();done=self.boot(f,143)
        self.assertEqual(done.returncode,143,done.stdout+done.stderr)
        self.assertIn('--resume',self.marker.read_text());self.assertIn('runs-out/'+f.job,self.marker.read_text())
        self.assertEqual((f.output/'run.log').read_text(),'ORIGINAL\nMOCK RUN\n')
        self.assertIn('attempt=1',f.record.read_text());self.assertIn('training-state=interrupted',f.record.read_text())
        entries=[json.loads(x) for x in (f.output/'attempts.jsonl').read_text().splitlines()]
        self.assertEqual([e['event'] for e in entries],['start','end'])
        self.assertEqual((entries[0]['epoch_in'],entries[0]['step_in']),(6,7))
        self.assertEqual(entries[0]['checkpoint_sha256_in'],hashlib.sha256((f.output/'checkpoints'/f.job/'last.ckpt').read_bytes()).hexdigest())
        self.marker.unlink();reboot=self.boot(f)
        self.assertNotEqual(reboot.returncode,0);self.assertFalse(self.marker.exists())
        f.metadata.write_text(f.metadata.read_text().replace('job-attempt=1','job-attempt=2'))
        final=self.boot(f);self.assertEqual(final.returncode,0,final.stdout+final.stderr)
        self.assertIn('training-state=done',f.record.read_text())
        self.assertEqual(len((f.output/'attempts.jsonl').read_text().splitlines()),4)

    def test_same_pi_folder_refreshes_resumed_files_without_changing_fresh_sync(self):
        f=self.prepare();f.record.write_text(f.record.read_text().replace('delivered=yes','delivered=no'))
        (f.area/'fake-key').write_text('mock, not a credential')
        tools=Path(self.env['PATH'].split(os.pathsep)[0])
        (tools/'getent').write_text('#!/bin/sh\nexit 0\n');(tools/'getent').chmod(0o755)
        (tools/'rsync').write_text(f'''#!{os.sys.executable}
import json,os,sys
with open(os.environ['RSYNC_MARKER'],'a') as f:f.write(json.dumps(sys.argv[1:])+'\\n')
''');(tools/'rsync').chmod(0o755)
        marker=f.area/'rsync.jsonl'
        with patch.dict(self.env,RSYNC_MARKER=str(marker)):done=self.boot(f,delivery=True)
        self.assertEqual(done.returncode,0,done.stdout+done.stderr)
        rows=[json.loads(x) for x in marker.read_text().splitlines()]
        self.assertEqual(len(rows),2)
        self.assertNotIn('--delete-delay',rows[0]);self.assertIn('--delete-delay',rows[1])
        self.assertEqual(rows[0][-2:],rows[1][-2:])
        self.assertEqual(rows[1][-2],str(f.output))
        self.assertTrue(rows[1][-1].endswith('@rpi400-server:.'))

    def test_service_sigterm_is_interrupted_not_success(self):
        f=self.prepare()
        env={**self.env,'G4_REPO_ROOT':str(self.root),'G4_STATE_ROOT':str(f.area),'MOCK_RUN_SLEEP':'60'}
        command=['bash','scripts/g4/job.sh','--test','--metadata-file',str(f.metadata),'--retry-seconds','0','--inbox-key',str(f.area/'no-key')]
        process=subprocess.Popen(command,cwd=self.root,env=env,text=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE,start_new_session=True)
        try:
            deadline=time.monotonic()+20
            while not self.marker.exists() and process.poll() is None and time.monotonic()<deadline:time.sleep(.05)
            self.assertTrue(self.marker.exists())
            process.send_signal(signal.SIGTERM);out,err=process.communicate(timeout=20)
            self.assertEqual(process.returncode,143,out+err)
            self.assertIn('training-state=interrupted',f.record.read_text())
            self.assertIn('exit-code=143',f.record.read_text())
        finally:
            if process.poll() is None:os.killpg(process.pid,signal.SIGKILL);process.wait()

    def test_terminal_states_and_wrong_commit_notebook_env_attempt_never_execute(self):
        for wrong in ('done','failed','timeout','commit','notebook','env','attempt','job-id'):
            with self.subTest(wrong=wrong):
                f=self.prepare(state=wrong if wrong in ('done','failed','timeout') else 'interrupted')
                if wrong=='commit':f.metadata.write_text(f.metadata.read_text().replace(self.commit,'0'*40))
                if wrong=='notebook':f.envfile.write_text(f.envfile.read_text().replace('newdata-1','wide-1'))
                if wrong=='env':f.envfile.write_text(f.envfile.read_text()+'V3_WORKERS=2\n')
                if wrong=='attempt':f.metadata.write_text(f.metadata.read_text().replace('job-attempt=1','job-attempt=0'))
                if wrong=='job-id':f.record.write_text(f.record.read_text().replace('job-id='+f.job,'job-id=other'))
                before=f.record.read_bytes();done=self.boot(f)
                self.assertNotEqual(done.returncode,0,done.stdout+done.stderr)
                self.assertFalse(self.marker.exists());self.assertEqual(f.record.read_bytes(),before)

    def test_missing_corrupt_checkpoint_refused_and_abrupt_running_record_recovers(self):
        for wrong in ('missing','corrupt','abrupt'):
            with self.subTest(wrong=wrong):
                f=self.prepare(state='running' if wrong=='abrupt' else 'interrupted')
                path=f.output/'checkpoints'/f.job/'last.ckpt'
                if wrong=='missing':path.unlink()
                elif wrong=='corrupt':path.write_bytes(b'bad')
                else:
                    f.record.write_text(f.record.read_text().replace('delivered=yes','delivered=no'))
                    resume.attempt(f.area/'jobs'/f'{f.job}.attempts.jsonl',f.output,f.job,0,'start','run')
                done=self.boot(f)
                if wrong=='abrupt':
                    self.assertEqual(done.returncode,0,done.stdout+done.stderr);self.assertTrue(self.marker.exists())
                    self.assertIn('abrupt stop detected',(f.output/'attempts.jsonl').read_text())
                else:
                    self.assertNotEqual(done.returncode,0);self.assertFalse(self.marker.exists())
                    self.assertIn('training-state=failed',f.record.read_text())

    def test_submit_resume_reads_stopped_metadata_only_and_reuses_published_identity(self):
        f=self.prepare();cloud=f.area/'cloud';data=f.area/'instance.json'
        document=dict(status='TERMINATED',metadata=dict(items=[dict(key=k,value=v) for k,v in {
            'job-id':f.job,'job-commit':self.commit,'job-env':f.envfile.read_text(),'job-timeout-min':'100',
            'max-run-minutes':'120','job-protocol':'resume-v1','job-attempt':'0'}.items()]))
        data.write_text(json.dumps(document))
        env={**self.env,'MOCK_METADATA':str(data),'CLOUD_MARKER':str(cloud)}
        command=['bash','scripts/g4/submit.sh','resume',f.job,'--dry-run']
        done=subprocess.run(command,cwd=self.root,env=env,text=True,capture_output=True)
        self.assertEqual(done.returncode,0,done.stdout+done.stderr)
        self.assertIn('job-id='+f.job,done.stdout);self.assertIn('job-attempt=1',done.stdout)
        self.assertIn('job-mode=resume',done.stdout);self.assertFalse(cloud.exists())
        for change in ('RUNNING','different-job','old-protocol'):
            bad=json.loads(json.dumps(document))
            if change=='RUNNING':bad['status']='RUNNING'
            else:
                key='job-id' if change=='different-job' else 'job-protocol'
                next(x for x in bad['metadata']['items'] if x['key']==key)['value']=change
            data.write_text(json.dumps(bad))
            failed=subprocess.run(command,cwd=self.root,env=env,text=True,capture_output=True)
            self.assertNotEqual(failed.returncode,0);self.assertNotIn('instances start',failed.stdout);self.assertFalse(cloud.exists())
