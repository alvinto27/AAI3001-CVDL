"""No-cloud watchdog, pending-request and native identity tests."""
from copy import deepcopy
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from scripts.g4 import resume_request as request
from scripts.g4 import watchdog as w
from scripts.g4 import watchdog_common as common

ROOT=Path(__file__).resolve().parents[1]
BASE=1800000000.
SHA='a'*40
PROOF=dict(validation='resume-identity-only',commit=SHA,notebook='notebooks/experiments/unit.ipynb',notebook_sha256='b'*64)

def iso(t):return datetime.fromtimestamp(t,timezone.utc).isoformat()

def document():
    m={'job-id':'unit-job','job-commit':SHA,'job-env':'V3_NOTEBOOK=notebooks/experiments/unit.ipynb\nCVDL_JOB_ID=unit-job\n',
       'job-protocol':'resume-v1','job-mode':'run','job-attempt':'0','job-timeout-min':'200','max-run-minutes':'240'}
    return dict(id='12345',status='RUNNING',lastStartTimestamp=iso(BASE+1),
        scheduling=dict(provisioningModel='SPOT',instanceTerminationAction='STOP',maxRunDuration=dict(seconds='18000')),
        metadata=dict(items=[dict(key=k,value=v) for k,v in m.items()]))

def operation(name='preempt-1',kind='compute.instances.preempted',t=BASE+100):
    return dict(name=name,operationType=kind,targetId='12345',status='DONE',startTime=iso(t),insertTime=iso(t),
        targetLink='https://www.googleapis.com/compute/v1/projects/personal-compute-511601/zones/us-central1-c/instances/rtx6000pro',
        zone='https://www.googleapis.com/compute/v1/projects/personal-compute-511601/zones/us-central1-c')

class FakeCloud:
    def __init__(self):self.doc=document();self.ops=[];self.actions=[];self.start_error=None
    def describe(self):return deepcopy(self.doc)
    def operations(self):return deepcopy(self.ops)
    def operation(self,name):return deepcopy(next(op for op in self.ops if op['name']==name))
    def set_metadata(self,m):
        self.actions.append(('metadata',deepcopy(m)))
        self.doc['metadata']['items']=[dict(key=k,value=v) for k,v in m.items()]
    def start(self):
        self.actions.append(('start',))
        if self.start_error:raise self.start_error
        op=operation('start-'+str(len(self.actions)),'start',BASE+200+len(self.actions));op['status']='PENDING'
        self.ops.append(op);return deepcopy(op)
    def stop(self):self.actions.append(('stop',));return dict(name='stop-op')


class PendingTests(unittest.TestCase):
    def setUp(self):
        self.temp=TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.path=Path(self.temp.name)/'request.json';self.cloud=FakeCloud();self.cloud.doc['status']='TERMINATED'
        self.proof=patch.object(request,'validate_checkout',return_value=PROOF);self.proof.start();self.addCleanup(self.proof.stop)
    def step(self,now=BASE+200,**kw):return request.step(self.path,'unit-job',cloud=self.cloud,now=now,**kw)
    def test_prepare_then_start_and_repeated_pending_never_increment_twice(self):
        self.assertEqual(self.step()['phase'],'metadata-set')
        self.assertEqual(common.metadata(self.cloud.doc)['job-attempt'],'1')
        self.assertEqual(self.step()['phase'],'pending')
        for _ in range(3):self.assertEqual(self.step()['phase'],'pending')
        self.assertEqual([a[0] for a in self.cloud.actions],['metadata','start'])
        self.assertEqual(common.read_json(self.path)['start_calls'],1)
    def test_stockout_240_delay_same_attempt_and_total_start_limit(self):
        self.step(start_limit=2);self.step(start_limit=2)
        self.cloud.ops[-1].update(status='DONE',error=dict(errors=[dict(code='ZONE_RESOURCE_POOL_EXHAUSTED')]))
        self.assertEqual(self.step(BASE+300,start_limit=2)['phase'],'stockout')
        self.step(BASE+539,start_limit=2);self.assertEqual(len(self.cloud.actions),2)
        self.step(BASE+540,start_limit=2);self.assertEqual([a[0] for a in self.cloud.actions],['metadata','start','start'])
        self.assertEqual(common.metadata(self.cloud.doc)['job-attempt'],'1')
        self.cloud.ops[-1].update(status='DONE',error=dict(errors=[dict(code='ZONE_RESOURCE_POOL_EXHAUSTED')]))
        self.step(BASE+600,start_limit=2)
        self.assertEqual(self.step(BASE+840,start_limit=2)['phase'],'failed')
        self.assertEqual(common.read_json(self.path)['start_calls'],2)
    def test_noncapacity_mixed_and_quota_do_not_retry(self):
        for codes in (['QUOTA_EXCEEDED'],['PERMISSION_DENIED'],['RESOURCE_EXHAUSTED'],['ZONE_RESOURCE_POOL_EXHAUSTED','QUOTA_EXCEEDED']):
            with self.subTest(codes=codes):
                self.path.unlink(missing_ok=True);self.cloud=FakeCloud();self.cloud.doc['status']='TERMINATED'
                self.step();self.step();self.cloud.ops[-1].update(status='DONE',error=dict(errors=[dict(code=c) for c in codes]))
                self.assertEqual(self.step()['phase'],'failed');self.step(BASE+1000)
                self.assertEqual([a[0] for a in self.cloud.actions],['metadata','start'])
    def test_boot_consumes_request_even_when_already_stopped_again(self):
        self.step();self.step();self.cloud.doc['lastStartTimestamp']=iso(BASE+210)
        self.assertEqual(self.step(BASE+300)['phase'],'consumed')
        self.step(BASE+400);self.assertEqual(len(self.cloud.actions),2)
    def test_response_loss_reconciles_metadata_without_next_counter(self):
        original=self.cloud.set_metadata
        def lost(m):original(m);raise TimeoutError()
        self.cloud.set_metadata=lost
        with self.assertRaises(TimeoutError):self.step()
        self.cloud.set_metadata=original
        self.assertEqual(self.step()['phase'],'pending')
        self.assertEqual(common.metadata(self.cloud.doc)['job-attempt'],'1')
    def test_lost_start_response_reserves_try_and_never_blindly_starts_again(self):
        self.step();self.cloud.start_error=TimeoutError()
        with self.assertRaises(TimeoutError):self.step()
        for _ in range(3):self.assertEqual(self.step()['phase'],'ambiguous')
        self.assertEqual([a[0] for a in self.cloud.actions],['metadata','start'])
        self.assertEqual(common.read_json(self.path)['start_calls'],1)
    def test_lost_response_recovers_one_matching_operation(self):
        self.step();self.cloud.start_error=TimeoutError()
        with self.assertRaises(TimeoutError):self.step()
        self.cloud.ops=[operation('recovered','start',BASE+201)]
        self.step();self.assertEqual(common.read_json(self.path)['operation'],'recovered')
        self.assertEqual(self.step()['phase'],'awaiting-boot')
    def test_corrupt_receipt_counter_and_auth_failure_are_not_retried(self):
        self.step();r=common.read_json(self.path);r['start_calls']=-1;common.atomic(self.path,r)
        with self.assertRaises(ValueError):self.step()
        self.assertEqual(len(self.cloud.actions),1)
        self.path.unlink();self.cloud=FakeCloud();self.cloud.doc['status']='TERMINATED'
        def forbidden(_):raise subprocess.CalledProcessError(1,['gcloud'],stderr='PERMISSION_DENIED')
        self.cloud.set_metadata=forbidden
        self.assertEqual(self.step()['phase'],'failed')
        self.assertEqual(self.step()['phase'],'failed');self.assertEqual(self.cloud.actions,[])

    def test_identity_drift_and_limit_change_refused(self):
        self.step()
        for change in ('commit','vm','metadata','limit'):
            with self.subTest(change=change):
                doc=deepcopy(self.cloud.doc)
                try:
                    if change=='commit':self.cloud.doc['metadata']['items'][1]['value']='c'*40
                    elif change=='vm':self.cloud.doc['id']='999'
                    elif change=='metadata':self.cloud.doc['metadata']['items'].append(dict(key='operator',value='changed'))
                    with self.assertRaises(ValueError):self.step(start_limit=14 if change=='limit' else 15)
                finally:self.cloud.doc=doc
        self.assertEqual(len(self.cloud.actions),1)
    def test_dry_run_creates_no_receipt_or_cloud_write(self):
        self.assertEqual(self.step(dry=True)['phase'],'dry-run')
        self.assertFalse(self.path.exists());self.assertEqual(self.cloud.actions,[])
    def test_running_first_request_scheduling_wrong_job_refused(self):
        for change in ('running','schedule','job'):
            cloud=deepcopy(self.cloud.doc)
            if change=='running':self.cloud.doc['status']='RUNNING'
            if change=='schedule':self.cloud.doc['scheduling']['maxRunDuration']['seconds']='7200'
            if change=='job':self.cloud.doc['metadata']['items'][0]['value']='other'
            with self.assertRaises(ValueError):self.step()
            self.cloud.doc=cloud
        self.assertEqual(self.cloud.actions,[])


class EngineTests(unittest.TestCase):
    def setUp(self):
        self.temp=TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name);self.path=self.root/'state/state.json';self.cloud=FakeCloud();self.now=BASE
        self.config=dict(w.DEFAULTS,armed=True,audit_approved=True,job='unit-job')
        self.patcher=patch.object(w,'validate_checkout',return_value=PROOF);self.patcher.start();self.addCleanup(self.patcher.stop)
        self.calls=[]
        self.engine=self.engine_new()
        self.now=BASE+10
    def submit(self,job,path,limit):
        self.calls.append((job,path,limit))
        common.atomic(path,dict(start_calls=1));return dict(phase='pending')
    def engine_new(self):return w.Engine(self.path,cloud=self.cloud,inbox=self.root/'inbox',clock=lambda:self.now,submit=self.submit)
    def tick(self):return self.engine.tick(self.config)
    def enroll(self):self.tick();return self.engine.state['jobs']['unit-job']
    def stopped(self):self.cloud.doc['status']='TERMINATED';self.cloud.ops=[operation()];self.now=BASE+110
    def snapshot(self,**kw):
        value=dict(source_commit=SHA,notebook_sha256='b'*64,state='running',exit_code=None,**{})
        value.update(kw);common.atomic(self.root/'inbox/unit-job/status.json',value)
    def test_unarmed_readonly_and_no_old_job_enrollment(self):
        self.config['armed']=False;self.assertEqual(self.tick()['phase'],'UNARMED');self.assertEqual(self.engine.state['jobs'],{})
        self.config['armed']=True;self.cloud.doc['lastStartTimestamp']=iso(BASE-1)
        self.assertEqual(self.tick()['phase'],'PAUSED');self.assertEqual(self.calls,[]);self.assertEqual(self.cloud.actions,[])
    def test_fresh_preemption_reserves_before_canonical_submit(self):
        j=self.enroll();self.stopped()
        self.assertEqual(self.tick()['phase'],'RESUME_RESERVED');self.assertEqual(self.calls,[])
        self.assertEqual(len(common.read_json(self.path)['jobs']['unit-job']['requests']),1)
        self.tick();self.assertEqual(len(self.calls),1);self.assertTrue(j['inspection_used'])
        self.assertEqual(j['used_preemptions'],['preempt-1'])
    def test_terminal_priority_over_preemption_and_restart_latches(self):
        for state,code,expected in [('done',0,'DONE'),('failed',1,'FAILED'),('interrupted',124,'TIMEOUT')]:
            with self.subTest(expected=expected):
                self.engine.state['jobs']={};self.cloud.doc=document();self.now=BASE+10
                (self.root/'inbox/unit-job/status.json').unlink(missing_ok=True)
                self.enroll();self.stopped()
                self.snapshot(state=state,exit_code=code)
                self.assertEqual(self.tick()['phase'],expected);self.assertEqual(self.calls,[])
                self.engine=self.engine_new();self.now+=301;self.assertEqual(self.tick()['phase'],expected)
    def test_no_preemption_for_stale_wrong_target_id_time_or_unknown_stop(self):
        self.enroll();self.stopped()
        for field,value in [('targetId','777'),('startTime',iso(BASE-1)),('targetLink','https://x/projects/personal-compute-511601/zones/us-central1-c/instances/other'),('operationType','stop'),('status','PENDING')]:
            op=operation();op[field]=value;self.cloud.ops=[op]
            self.assertNotEqual(self.tick()['phase'],'RESUME_RESERVED')
        self.now+=601;self.cloud.ops=[];self.assertEqual(self.tick()['phase'],'PAUSED');self.assertEqual(self.calls,[])
    def test_unknown_state_inspection_is_once_per_job(self):
        j=self.enroll();j['inspection_used']=True;self.stopped()
        self.assertEqual(self.tick()['phase'],'PAUSED');self.assertEqual(self.calls,[])
    def test_known_current_interruption_does_not_use_unknown_allowance(self):
        j=self.enroll();j['inspection_used']=True;self.stopped();self.snapshot(state='interrupted',exit_code=143)
        p=self.root/'inbox/unit-job/attempts.jsonl';p.write_text(json.dumps(dict(job_id='unit-job',attempt=0,event='end',reason='interrupted exit=143'))+'\n')
        self.assertEqual(self.tick()['phase'],'RESUME_RESERVED')
    def test_budget_reserve_start_and_resume_caps(self):
        for which in ('budget','resumes','starts'):
            self.engine.state['jobs']={};self.cloud.doc=document();self.now=BASE+10;j=self.enroll();self.stopped()
            if which=='budget':j['policy']['hourly_rate_usd']=5;self.config['hourly_rate_usd']=5
            elif which=='resumes':j['requests']=[dict(consumed=True,path=str(self.root/f'r{i}.json')) for i in range(3)]
            else:j['requests']=[dict(consumed=True,path=str(self.root/'r0.json'))]
            for r in j['requests']:common.atomic(r['path'],dict(start_calls=15 if which=='starts' else 1))
            for item in self.cloud.doc['metadata']['items']:
                if item['key']=='job-attempt':item['value']=str(len(j['requests']))
            self.assertEqual(self.tick()['phase'],'CAPPED');self.assertEqual(self.calls,[])
            self.config['hourly_rate_usd']=2
    def test_external_attempt_or_unrequested_boot_pauses_and_charges_unknown_gap(self):
        j=self.enroll();self.cloud.doc['status']='TERMINATED';self.now=BASE+100;self.tick()
        self.now=BASE+3601;self.cloud.doc.update(status='RUNNING',lastStartTimestamp=iso(BASE+3500))
        for item in self.cloud.doc['metadata']['items']:
            if item['key']=='job-attempt':item['value']='2'
        self.assertEqual(self.tick()['phase'],'PAUSED')
        self.assertAlmostEqual(j['estimated_usd'],2)
        self.assertEqual(self.calls,[]);self.assertEqual(self.cloud.actions,[])
        self.engine=self.engine_new();self.assertEqual(self.tick()['phase'],'PAUSED')

    def test_unrequested_same_counter_reboot_is_not_automatic_resume_authority(self):
        self.enroll();self.now=BASE+100;self.cloud.doc['lastStartTimestamp']=iso(BASE+90)
        self.assertEqual(self.tick()['phase'],'PAUSED');self.assertEqual(self.calls,[])

    def test_original_boot_cost_interval_union_and_stop_detection(self):
        j=self.enroll();self.now=BASE+3601
        self.assertAlmostEqual(self.tick()['estimated_usd'],2)
        self.cloud.doc['status']='TERMINATED';self.tick();end=j['boots'][iso(BASE+1)]['end']
        self.now+=600;self.tick();self.assertEqual(j['boots'][iso(BASE+1)]['end'],end)
        self.assertAlmostEqual(j['estimated_usd'],2)
    def test_caps_stop_owned_running_vm_bounded_and_confirmed(self):
        j=self.enroll();j.update(phase='CAPPED',terminal_at=self.now)
        for i in range(5):self.now+=61;result=self.tick()
        self.assertEqual([a[0] for a in self.cloud.actions],['stop']*3)
        self.assertIn('HIGH-PRIORITY',result['reason'])
        self.cloud.doc['status']='TERMINATED';self.assertEqual(self.tick()['reason'],'confirmed-stopped')
    def test_different_job_vm_or_policy_never_stopped_or_resumed(self):
        self.enroll();self.cloud.doc['id']='999';self.assertEqual(self.tick()['phase'],'PAUSED')
        self.assertEqual(self.cloud.actions,[]);self.assertEqual(self.calls,[])
    def test_missing_corrupt_state_cannot_reset_caps(self):
        self.enroll();self.path.unlink()
        with self.assertRaisesRegex(ValueError,'caps'):self.engine_new()
        self.path.write_text('{')
        with self.assertRaises(ValueError):self.engine_new()
    def test_partial_or_wrong_inbox_refuses_decision(self):
        self.enroll();self.stopped();self.snapshot(source_commit='c'*40)
        with self.assertRaises(ValueError):self.tick()
        p=self.root/'inbox/unit-job/status.json';p.write_text('{')
        with self.assertRaises(ValueError):self.tick()
        self.assertEqual(self.calls,[])
    def test_duplicate_lock_and_private_atomic_modes(self):
        with common.lock(self.root/'lock'):
            with self.assertRaises(BlockingIOError):
                with common.lock(self.root/'lock'):pass
        self.assertEqual(self.path.stat().st_mode&0o777,0o600)
        self.assertEqual(list(self.path.parent.glob('.state.json.*')),[])
    def test_config_caps_and_arm_audit_gate(self):
        path=self.root/'config.json'
        for key,value in [('armed','yes'),('max_resumes',4),('max_start_calls',16),('retry_seconds',1),('budget_usd',21),('hourly_rate_usd',float('nan')),('margin_usd',0),('job','../x'),('audit_approved',False)]:
            c=dict(self.config);c[key]=value;path.write_text(json.dumps(c))
            with self.assertRaises(ValueError):w.configuration(path)
    def test_integrated_three_resumes_and_fourth_preemption_is_capped(self):
        with patch.object(request,'validate_checkout',return_value=PROOF):
            self.engine.submit=lambda job,path,limit:request.step(path,job,cloud=self.cloud,now=self.now,start_limit=limit)
            j=self.enroll()
            for n in range(1,4):
                self.now=BASE+n*300;self.cloud.doc['status']='TERMINATED'
                self.cloud.ops.append(operation('preempt-'+str(n),t=self.now-1))
                if n>1:
                    self.snapshot(state='interrupted',exit_code=143)
                    p=self.root/'inbox/unit-job/attempts.jsonl'
                    p.write_text(json.dumps(dict(job_id='unit-job',attempt=n-1,event='end',reason='interrupted exit=143'))+'\n')
                self.assertEqual(self.tick()['phase'],'RESUME_RESERVED')
                self.tick();self.tick()
                self.assertEqual(common.metadata(self.cloud.doc)['job-attempt'],str(n))
                self.now+=20;self.cloud.doc.update(status='RUNNING',lastStartTimestamp=iso(self.now))
                self.tick();self.assertTrue(j['requests'][-1]['consumed'])
                self.engine=self.engine_new()
                self.engine.submit=lambda job,path,limit:request.step(path,job,cloud=self.cloud,now=self.now,start_limit=limit)
                j=self.engine.state['jobs']['unit-job']
            self.now+=100;self.cloud.doc['status']='TERMINATED';self.cloud.ops.append(operation('preempt-4',t=self.now-1))
            self.assertEqual(self.tick()['phase'],'CAPPED')
            self.assertEqual(sum(a[0]=='start' for a in self.cloud.actions),3)
            self.assertEqual(self.engine.start_calls(j),3)

    def test_helper_observes_boot_before_outer_poll_without_false_external_pause(self):
        with patch.object(request,'validate_checkout',return_value=PROOF):
            call=lambda job,path,limit:request.step(path,job,cloud=self.cloud,now=self.now,start_limit=limit)
            self.engine.submit=call;self.enroll();self.stopped();self.tick();self.tick();self.tick()
            self.now+=20
            def race(job,path,limit):
                self.cloud.doc.update(status='RUNNING',lastStartTimestamp=iso(self.now))
                return call(job,path,limit)
            self.engine.submit=race
            self.tick()  # outer snapshot was stopped; canonical helper sees the new boot
            self.engine=self.engine_new()
            self.assertEqual(self.tick()['phase'],'OBSERVING')
            self.assertEqual(sum(a[0]=='start' for a in self.cloud.actions),1)

    def test_corrupt_counter_and_missing_request_cannot_reset_allowance(self):
        j=self.enroll();self.stopped();self.tick();self.tick()
        r=Path(j['requests'][0]['path']);r.write_text(json.dumps(dict(start_calls=-1)))
        with self.assertRaises(ValueError):self.tick()
        r.unlink()
        with self.assertRaises(ValueError):self.tick()
        self.assertEqual(len(self.calls),1)

    def test_stopped_capacity_wait_does_not_consume_an_old_boot_duration(self):
        self.enroll();self.stopped();self.now=BASE+1+241*60
        self.assertEqual(self.tick()['phase'],'RESUME_RESERVED')

    def test_stopping_wait_and_actual_metadata_duration_limit(self):
        j=self.enroll();self.cloud.doc['status']='STOPPING';self.tick();self.now+=601
        self.assertEqual(self.tick()['phase'],'PAUSED');self.assertEqual(self.calls,[])
        j['phase']='OBSERVING';self.cloud.doc['status']='RUNNING';self.now=BASE+1+240*60
        self.assertEqual(self.tick()['phase'],'TIMEOUT')

    def test_atomic_failure_preserves_previous_state(self):
        before=self.path.read_bytes()
        with patch.object(common.os,'replace',side_effect=OSError()):
            with self.assertRaises(OSError):common.atomic(self.path,dict(bad=True))
        self.assertEqual(self.path.read_bytes(),before);self.assertEqual(list(self.path.parent.glob('.state.json.*')),[])


class NativeShellTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import test_g4_jobs as jobs
        jobs.SubmissionTests.setUpClass.__func__(cls)
        cls.area=cls.root.parent
        tools=Path(cls.env['PATH'].split(os.pathsep)[0])
        (tools/'podman').write_text('#!/bin/sh\necho UNEXPECTED_PODMAN >&2\nexit 99\n')
        (tools/'ssh').write_text('#!/bin/sh\necho UNEXPECTED_SSH >&2\nexit 99\n');(tools/'ssh').chmod(0o755)
        cloud=tools/'gcloud'
        cloud.write_text('''#!'''+sys.executable+'''
import json,os,sys
from pathlib import Path
args=sys.argv[1:];path=Path(os.environ['FAKE_VM']);d=json.loads(path.read_text())
if args[:3]==['compute','instances','describe']:print(json.dumps(d))
elif args[:3]==['compute','operations','list']:print('[]')
elif args[:3]==['compute','operations','describe']:
 print(json.dumps(dict(name=args[3],status='PENDING',targetId=d['id'],targetLink=os.environ['TARGET'],zone=os.environ['ZONE_LINK'])))
elif args[:3]==['compute','instances','add-metadata']:
 m={x['key']:x['value'] for x in d['metadata']['items']}
 raw=next(a.split('=',1)[1] for a in args if a.startswith('--metadata='))
 m.update(x.split('=',1) for x in raw.split(','))
 p=next(a.split('=',2)[2] for a in args if a.startswith('--metadata-from-file='))
 m['job-env']=Path(p).read_text();d['metadata']['items']=[dict(key=k,value=v) for k,v in m.items()]
 path.write_text(json.dumps(d));print('{}')
elif args[:3]==['compute','instances','start']:
 with open(os.environ['STARTS'],'a') as f:f.write('start\\n')
 print(json.dumps([dict(name='start-one')]))
else:raise SystemExit('UNEXPECTED CLOUD '+repr(args))
''');cloud.chmod(0o755)
        cls.env.update(CVDL_G4_CONTROLLER_PYTHON=sys.executable,TARGET=operation()['targetLink'],ZONE_LINK=operation()['zone'])
    @classmethod
    def tearDownClass(cls):cls.temp.cleanup()
    def test_daemon_once_unarmed_creates_private_state_without_cloud_writes(self):
        vm=self.area/'daemon-vm.json';vm.write_text(json.dumps(document()));state=self.area/'daemon-state/state.json';starts=self.area/'daemon-starts'
        env=dict(self.env,FAKE_VM=str(vm),STARTS=str(starts))
        subprocess.run([sys.executable,'-S','scripts/g4/watchdog.py','--once',
            '--config','scripts/g4/watchdog.json','--state',str(state)],cwd=self.root,env=env,check=True,capture_output=True,timeout=20)
        self.assertFalse(starts.exists());self.assertEqual(common.read_json(state)['jobs'],{})
        status=common.read_json(state.parent/'status.json');self.assertEqual(status['phase'],'UNARMED')
        self.assertIn('+00:00',status['utc']);self.assertEqual(state.stat().st_mode&0o777,0o600)
        self.assertTrue((state.parent/'watchdog.jsonl').exists())

    def test_canonical_shell_native_pending_request_and_dry_run(self):
        doc=document();doc['status']='TERMINATED'
        m=common.metadata(doc);m['job-commit']=self.commit;m['job-env']='V3_NOTEBOOK=notebooks/newdata-1.ipynb\nCVDL_JOB_ID=unit-job\n'
        doc['metadata']['items']=[dict(key=k,value=v) for k,v in m.items()]
        vm=self.area/'fake-vm.json';vm.write_text(json.dumps(doc));receipt=self.area/'pending.json';starts=self.area/'starts'
        env=dict(self.env,FAKE_VM=str(vm),STARTS=str(starts))
        base=['bash','scripts/g4/submit.sh','resume','unit-job','--request-file',str(receipt)]
        def run(extra=()):
            p=subprocess.run(base+list(extra),cwd=self.root,env=env,capture_output=True,text=True,timeout=20)
            self.assertEqual(p.returncode,0,p.stdout+p.stderr);return json.loads(p.stdout)
        self.assertEqual(run(['--dry-run'])['phase'],'dry-run');self.assertFalse(receipt.exists());self.assertFalse(starts.exists())
        self.assertEqual(run()['phase'],'metadata-set');self.assertEqual(run()['phase'],'pending')
        self.assertEqual(run()['phase'],'pending');self.assertEqual(starts.read_text(),'start\n')
        self.assertEqual(common.metadata(common.read_json(vm))['job-attempt'],'1')


class PortabilityTests(unittest.TestCase):
    def test_identity_mode_in_python_without_site_or_training_imports(self):
        script='''import sys
class Block:
 def find_spec(self,fullname,*args):
  if fullname.split('.')[0] in {'torch','numpy','pandas','PIL','yaml','lightning'}:raise AssertionError(fullname)
sys.meta_path.insert(0,Block())
from scripts.g4.preflight import resume_identity
p=resume_identity('notebooks/experiments/srm-e50-paired-coloursep36.ipynb','HEAD')
assert p['validation']=='resume-identity-only'
'''
        subprocess.run([sys.executable,'-S','-c',script],cwd=ROOT,check=True,capture_output=True,text=True,timeout=20)
    def test_native_run_is_refused_before_any_external_cloud(self):
        env=dict(os.environ,CVDL_G4_CONTROLLER_PYTHON=sys.executable,PATH='/usr/bin:/bin')
        done=subprocess.run(['bash','scripts/g4/submit.sh','run','--notebook','notebooks/newdata-1.ipynb'],cwd=ROOT,env=env,capture_output=True,text=True)
        self.assertNotEqual(done.returncode,0);self.assertIn('resume/status only',done.stderr)
    def test_cloud_commands_explicit_project_zone_no_ssh_and_bounded(self):
        seen=[]
        def fake(args,**kw):
            seen.append((args,kw));return subprocess.CompletedProcess(args,0,'[]','')
        with patch.object(common.subprocess,'run',side_effect=fake):
            cloud=common.Cloud();cloud.operations();cloud.stop()
        for args,kw in seen:
            self.assertEqual(args[0],'gcloud');self.assertIn('--project=personal-compute-511601',args);self.assertEqual(kw['timeout'],40)
        self.assertIn('--zones=us-central1-c',seen[0][0]);self.assertNotIn('--limit',str(seen[0][0]))
    def test_arm_disarm_keep_root_config_service_readable_and_require_audit(self):
        with TemporaryDirectory() as tmp:
            path=Path(tmp)/'config.json';common.atomic(path,dict(w.DEFAULTS),mode=0o644)
            base=[sys.executable,str(ROOT/'scripts/g4/watchdog.py'),'--config',str(path)]
            p=subprocess.run(base+['--arm','unit-job'],capture_output=True,text=True)
            self.assertNotEqual(p.returncode,0);self.assertFalse(common.read_json(path)['armed'])
            c=dict(w.DEFAULTS,audit_approved=True);common.atomic(path,c,mode=0o644)
            subprocess.run(base+['--arm','unit-job'],check=True,capture_output=True)
            self.assertTrue(common.read_json(path)['armed']);self.assertEqual(path.stat().st_mode&0o777,0o644)
            subprocess.run(base+['--disarm'],check=True,capture_output=True)
            self.assertFalse(common.read_json(path)['armed']);self.assertEqual(path.stat().st_mode&0o777,0o644)

    def test_unit_and_template_have_no_armed_job_or_training_dependency(self):
        c=w.configuration(ROOT/'scripts/g4/watchdog.json');self.assertFalse(c['armed']);self.assertFalse(c['audit_approved']);self.assertIsNone(c['job'])
        text=(ROOT/'scripts/g4/cvdl-watchdog.service').read_text()
        for value in ('User=cvdl-watchdog','LoadCredential=','Restart=on-failure','ProtectSystem=strict','UMask=0077','RestartPreventExitStatus=75'):self.assertIn(value,text)
        self.assertNotIn('podman',text);self.assertNotIn('uv ',text)


if __name__=='__main__':unittest.main()
