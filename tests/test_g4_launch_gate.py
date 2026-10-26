"""Read-only launch gate: fake transports, production metadata parser, no live hosts."""
from contextlib import redirect_stderr
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from scripts.g4 import launch_gate as gate, resume_request as request
import test_g4_jobs as jobs
import test_g4_watchdog as watchdog_tests

ROOT = Path(__file__).resolve().parents[1]
PI = 'mount_source=1\nfilesystem_rw=1\nupload_owner_write=1\nsrv_search=1\nresults_search=1'
DOC = dict(scheduling=dict(provisioningModel='SPOT', instanceTerminationAction='STOP', maxRunDuration=dict(seconds='18000')))

STUB = r'''import json, os, pathlib, subprocess, sys
name=pathlib.Path(sys.argv[0]).name; args=sys.argv[1:]
with open(os.environ['CALLS'],'a') as f:f.write(json.dumps([name,*args])+'\n')
if name==os.environ.get('FAIL_TOOL'):sys.exit(127)
if name=='timeout':
    assert args[0]=='--kill-after=2s' and args[1] in ('20s','40s'), args
    if args[2]==os.environ.get('TIMEOUT_TARGET'):sys.exit(124)
    sys.exit(subprocess.call(args[2:]))
elif name=='hostname':print(os.environ.get('FAKE_HOST','pc-tests'))
elif name=='ssh':
    assert 'rpi400@rpi400-server' in args and 'BatchMode=yes' in args and 'ConnectTimeout=10' in args
    sys.stdin.read();print(os.environ['PI_OUTPUT']);sys.exit(int(os.environ.get('SSH_EXIT','0')))
elif name=='gcloud':
    if args[:3]==['compute','instances','describe']:
        if '--format=json' in args:print(pathlib.Path(os.environ['VM_FILE']).read_text())
        else:print('TERMINATED')
        sys.exit(int(os.environ.get('GCLOUD_EXIT','0')))
    with open(os.environ['WRITES'],'a') as f:f.write(' '.join(args)+'\n')
    print('{}')
elif name=='awk':
    assert args[-1]=='/proc/self/mountinfo'
    sys.exit(subprocess.call(['/usr/bin/awk',*args[:-1],os.environ['MOUNTINFO']]))
elif name=='id':
    assert args[1]==os.environ.get('UPLOAD_USER','cvdl-inbox'), args
    print('1001' if args[0]=='-u' else os.environ.get('UPLOAD_GROUPS','1001 2001'))
elif name=='stat':
    target=args[-1]
    if target=='/srv':print(os.environ.get('SRV_STAT','41ed 1000 1000'))
    elif target=='/srv/cvdl-results':
        if os.environ.get('MISSING_RESULTS'):sys.exit(1)
        print(os.environ.get('RESULTS_STAT','41ed 1001 1001'))
    else:raise RuntimeError(args)
else:raise RuntimeError((name,args))
'''


def tools(area, base_env=None):
    area=Path(area); bin=area/'probe-bin';bin.mkdir(exist_ok=True)
    for name in ('timeout','hostname','ssh','gcloud','awk','id','stat'):
        path=bin/name;path.write_text('#!'+sys.executable+'\n'+STUB);path.chmod(0o755)
    vm=area/'probe-vm.json';vm.write_text(json.dumps(DOC))
    mount=area/'mountinfo'
    # Deliberately per-mount ro, filesystem rw: the sandbox must not false-refuse.
    mount.write_text('42 1 8:1 / /srv ro,nosuid shared:1 - ext4 /dev/sda1 rw,noatime\n')
    return dict(base_env or os.environ, PATH=str(bin)+os.pathsep+(base_env or os.environ)['PATH'],
                CALLS=str(area/'probe-calls'),VM_FILE=str(vm),MOUNTINFO=str(mount),WRITES=str(area/'writes'),PI_OUTPUT=PI)


def raw(pi=PI, document=DOC, pi_rc=0, cloud_rc=0):
    return f'{pi_rc}\n{pi}\n---CVDL-GCE---\n{cloud_rc}\n{json.dumps(document)}\n'


class ParserTests(unittest.TestCase):
    def test_good_and_each_failed_check_prints_truthful_status_to_stderr_only(self):
        stream=io.StringIO();self.assertTrue(all(gate.validate(raw(),stream=stream).values()))
        self.assertEqual(stream.getvalue().count('launch gate PASS:'),7)
        for key in set(gate.LABELS)-{'duration','termination'}:
            with self.subTest(key=key):
                stream=io.StringIO()
                with self.assertRaises(gate.LaunchGateError):gate.validate(raw(pi=PI.replace(key+'=1',key+'=0')),stream=stream)
                self.assertIn('FAIL: '+gate.LABELS[key],stream.getvalue())
                self.assertEqual(stream.getvalue().count('PASS:'),6)

    def test_bad_missing_and_fractional_scheduling_and_malformed_payloads(self):
        for scheduling in ({}, {'maxRunDuration':{'seconds':7200},'instanceTerminationAction':'STOP'},
                           {'maxRunDuration':{'seconds':18000,'nanos':1},'instanceTerminationAction':'STOP'},
                           {'maxRunDuration':{'seconds':18000,'nanos':False},'instanceTerminationAction':'STOP'},
                           {'maxRunDuration':{'seconds':18000.0},'instanceTerminationAction':'STOP'},
                           {'maxRunDuration':{'seconds':True},'instanceTerminationAction':'STOP'},
                           {'maxRunDuration':{'seconds':'18000'},'instanceTerminationAction':'DELETE'},
                           {'maxRunDuration':{'seconds':'18000'}}, None):
            with self.subTest(scheduling=scheduling), self.assertRaises(gate.LaunchGateError):
                gate.validate(raw(document={'scheduling':scheduling}),stream=io.StringIO())
        for payload in ('', 'garbage', raw(pi=PI+'\nmount_source=1'), raw(pi='garbage'),
                        raw(pi=PI.replace('mount_source=1','mount_source=PASS')),raw(document=[])):
            with self.subTest(payload=payload), self.assertRaises(gate.LaunchGateError):
                gate.validate(payload,stream=io.StringIO())
        for pi_rc,cloud_rc in ((124,0),(127,0),(0,124),(0,127)):
            with self.subTest(rc=(pi_rc,cloud_rc)),self.assertRaises(gate.LaunchGateError):
                gate.validate(raw(pi_rc=pi_rc,cloud_rc=cloud_rc),stream=io.StringIO())


class ProbeTests(unittest.TestCase):
    def probe(self, env):
        result=subprocess.run(['bash',str(ROOT/'scripts/g4/launch_probe.sh')],env=env,text=True,capture_output=True,timeout=10)
        self.assertEqual(result.returncode,0,result.stderr)
        return result.stdout

    def test_local_native_metadata_uses_superblock_not_sandbox_mount_flags(self):
        with TemporaryDirectory() as area:
            env=tools(area);env['FAKE_HOST']='rpi400-server'
            stream=io.StringIO();self.assertTrue(all(gate.validate(self.probe(env),stream=stream).values()))
            calls=Path(env['CALLS']).read_text();self.assertNotIn('["ssh",',calls)
            self.assertNotIn('sudo',calls);self.assertNotIn('test -w',stream.getvalue())
            self.assertIn('upload user owns the folder with write permission (mode bits)',stream.getvalue())

    def test_each_local_metadata_failure(self):
        cases=[({'MISSING_RESULTS':'1'},'upload_owner_write'),({'RESULTS_STAT':'81ed 1001 1001'},'results_search'),
               ({'RESULTS_STAT':'41ed 1002 1001'},'upload_owner_write'),({'RESULTS_STAT':'416d 1001 1001'},'upload_owner_write'),
               ({'RESULTS_STAT':'41ac 1001 1001'},'results_search'),({'SRV_STAT':'41ec 1000 1000'},'srv_search'),
               ({'SRV_STAT':'41c7 1000 2001'},'srv_search'),({'SRV_STAT':'41bf 1001 1000'},'srv_search')]
        for changes,key in cases:
            with self.subTest(changes=changes),TemporaryDirectory() as area:
                env=tools(area);env.update(FAKE_HOST='rpi400-server',**changes);stream=io.StringIO()
                with self.assertRaises(gate.LaunchGateError):gate.validate(self.probe(env),stream=stream)
                self.assertIn('FAIL: '+gate.LABELS[key],stream.getvalue())
        for mount,key in [('', 'mount_source'),
                          ('42 1 8:1 / /srv rw - ext4 /dev/sdb1 rw\n','mount_source'),
                          ('42 1 8:1 / /srv rw - ext4 /dev/sda1 ro\n','filesystem_rw'),
                          ('42 1 8:1 / /srv rw - ext4 /dev/sda1 rw,ro\n','filesystem_rw'),
                          ('malformed\n','mount_source')]:
            with self.subTest(mount=mount),TemporaryDirectory() as area:
                env=tools(area);env['FAKE_HOST']='rpi400-server';Path(env['MOUNTINFO']).write_text(mount)
                with self.assertRaises(gate.LaunchGateError):gate.validate(self.probe(env),stream=io.StringIO())

    def test_ssh_cloud_timeouts_errors_and_missing_tools(self):
        for changes in ({'TIMEOUT_TARGET':'ssh'},{'TIMEOUT_TARGET':'gcloud'}, {'SSH_EXIT':'1'}, {'GCLOUD_EXIT':'1'}):
            with self.subTest(changes=changes),TemporaryDirectory() as area:
                env=tools(area);env.update(changes)
                with self.assertRaises(gate.LaunchGateError):gate.validate(self.probe(env),stream=io.StringIO())
        for name in ('ssh','gcloud','id','stat','awk','timeout','hostname'):
            with self.subTest(missing=name),TemporaryDirectory() as area:
                env=tools(area);env['FAIL_TOOL']=name
                if name in ('id','stat','awk'):env['FAKE_HOST']='rpi400-server'
                result=subprocess.run(['bash',str(ROOT/'scripts/g4/launch_probe.sh')],env=env,text=True,capture_output=True)
                with self.assertRaises(gate.LaunchGateError):gate.validate(result.stdout,stream=io.StringIO())
        with patch.object(gate.subprocess,'run',side_effect=subprocess.TimeoutExpired('probe',70)),redirect_stderr(io.StringIO()):
            with self.assertRaises(gate.LaunchGateError):gate.check_launch()

    def test_upload_username_is_read_from_existing_source_not_assumed(self):
        with TemporaryDirectory() as area:
            env=tools(area);here=Path(area)/'g4';here.mkdir()
            for name in ('launch_probe.sh','pi_launch_check.sh'):shutil.copy2(ROOT/'scripts/g4'/name,here/name)
            (here/'job_run.sh').write_text('INBOX_USER=alternate-upload\n')
            result=subprocess.run(['bash',str(here/'launch_probe.sh')],env=env,text=True,capture_output=True)
            self.assertEqual(result.returncode,0,result.stderr)
            self.assertIn('bash -s -- alternate-upload',Path(env['CALLS']).read_text())


class SubmissionGateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        jobs.SubmissionTests.setUpClass.__func__(cls)
        wrapper=Path(cls.env['PATH'].split(os.pathsep)[0])/'podman'
        text=wrapper.read_text().replace("if 'scripts/run_notebook.py' in a:",
            "if 'scripts/g4/preflight.py' in a:\n    print('{}');raise SystemExit(0)\nif 'scripts/run_notebook.py' in a:")
        wrapper.write_text(text)
    @classmethod
    def tearDownClass(cls):cls.temp.cleanup()

    def launch(self, area, action, dry, *, bad=None):
        env=tools(area,self.env)
        metadata=watchdog_tests.document();metadata['status']='TERMINATED'
        m=watchdog_tests.common.metadata(metadata);m['job-commit']=self.commit
        m['job-env']='V3_NOTEBOOK=notebooks/newdata-1.ipynb\nCVDL_JOB_ID=unit-job\n'
        metadata['metadata']['items']=[dict(key=k,value=v) for k,v in m.items()]
        Path(env['VM_FILE']).write_text(json.dumps(metadata))
        if bad in ('duration','termination'):
            if bad=='duration':metadata['scheduling']['maxRunDuration']['seconds']='7200'
            else:metadata['scheduling']['instanceTerminationAction']='DELETE'
            Path(env['VM_FILE']).write_text(json.dumps(metadata))
        elif bad:env['PI_OUTPUT']=PI.replace(bad+'=1',bad+'=0')
        command=['bash','scripts/g4/submit.sh',action]
        command+=['--notebook','notebooks/newdata-1.ipynb'] if action=='run' else ['unit-job']
        if dry:command+=['--dry-run']
        result=subprocess.run(command,cwd=self.root,env=env,text=True,capture_output=True,timeout=15)
        return result,env

    def test_run_and_resume_fail_before_mutations_including_dry_run(self):
        for action in ('run','resume'):
            for dry in (False,True):
                for key in gate.LABELS:
                    with self.subTest(action=action,dry=dry,key=key),TemporaryDirectory() as area:
                        result,env=self.launch(area,action,dry,bad=key)
                        self.assertNotEqual(result.returncode,0,result.stdout+result.stderr)
                        self.assertIn('FAIL: '+gate.LABELS[key],result.stderr)
                        self.assertFalse(Path(env['WRITES']).exists())

    def test_dry_run_performs_all_checks_but_never_metadata_or_start(self):
        for action in ('run','resume'):
            with self.subTest(action=action),TemporaryDirectory() as area:
                result,env=self.launch(area,action,True)
                self.assertEqual(result.returncode,0,result.stdout+result.stderr)
                self.assertEqual(result.stderr.count('launch gate PASS:'),7)
                self.assertNotIn('launch gate',result.stdout);self.assertFalse(Path(env['WRITES']).exists())
                calls=Path(env['CALLS']).read_text();self.assertIn('["ssh",',calls);self.assertIn('instances", "describe',calls)

    def test_success_is_required_before_real_metadata_start_calls(self):
        for action in ('run','resume'):
            with self.subTest(action=action),TemporaryDirectory() as area:
                result,env=self.launch(area,action,False)
                self.assertEqual(result.returncode,0,result.stdout+result.stderr)
                writes=Path(env['WRITES']).read_text().splitlines()
                self.assertEqual(len(writes),2);self.assertIn('add-metadata',writes[0]);self.assertIn('instances start',writes[1])

    def test_watchdog_forwards_named_checks_only_and_keeps_json_stdout_contract(self):
        for code in (0,2):
            result=subprocess.CompletedProcess([],code,'{"phase":"pending"}', 'launch gate FAIL: named condition\nsecret cloud stderr\n')
            stream=io.StringIO()
            with patch.object(watchdog_tests.w.subprocess,'run',return_value=result),redirect_stderr(stream):
                if code:
                    with self.assertRaises(subprocess.CalledProcessError):
                        watchdog_tests.w.Engine.canonical_submit(SimpleNamespace(root=ROOT),'unit-job','receipt',15)
                else:
                    self.assertEqual(watchdog_tests.w.Engine.canonical_submit(SimpleNamespace(root=ROOT),'unit-job','receipt',15),{'phase':'pending'})
            self.assertEqual(stream.getvalue(),'launch gate FAIL: named condition\n')

    def test_pending_native_route_rechecks_before_prepare_and_before_start_and_in_dry_run(self):
        for phase in ('new','metadata-set','dry'):
            with self.subTest(phase=phase),TemporaryDirectory() as area:
                env=tools(area);cloud=watchdog_tests.FakeCloud();cloud.doc['status']='TERMINATED';path=Path(area)/'request.json'
                with patch.dict(os.environ,env), patch.object(request,'validate_checkout',return_value=watchdog_tests.PROOF),redirect_stderr(io.StringIO()):
                    if phase=='metadata-set':request.step(path,'unit-job',cloud=cloud)
                    before=path.read_bytes() if path.exists() else None;actions=list(cloud.actions)
                    os.environ['PI_OUTPUT']=PI.replace('filesystem_rw=1','filesystem_rw=0')
                    with self.assertRaises(gate.LaunchGateError):request.step(path,'unit-job',cloud=cloud,dry=phase=='dry')
                    self.assertEqual(cloud.actions,actions)
                    self.assertEqual(path.read_bytes() if path.exists() else None,before)


if __name__=='__main__':unittest.main()
