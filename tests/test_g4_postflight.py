"""Postflight uses only local reads plus fake SSH/gcloud; no VM or Pi writes."""
from copy import deepcopy
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import time
import unittest
from unittest.mock import patch

from scripts.g4 import postflight as p

ROOT=Path(__file__).resolve().parents[1]
JOB='unit-postflight'

def iso(value):return datetime.fromtimestamp(value,timezone.utc).isoformat()


def fixtures(area):
    area=Path(area); roots=[]
    status=dict(state='done',exit_code=0,source_commit='a'*40,notebook_sha256='b'*64,
                effective_settings={'CVDL_JOB_ID':JOB},ended_at=iso(time.time()-10))
    events=[]
    for number in (0,1):
        for event in ('start','end'):
            events.append(dict(job_id=JOB,attempt=number,event=event,time_utc=iso(100+2*number+(event=='end')),
                reason=('run' if number==0 else 'resume') if event=='start' else 'interrupted exit=143' if number==0 else 'done exit=0'))
    for side in ('pc','pi'):
        root=area/side/JOB;checkpoint=root/'checkpoints/nested/deeper/model.ckpt';checkpoint.parent.mkdir(parents=True)
        checkpoint.write_bytes(b'checkpoint bytes, not a torch serialization')
        (root/'checkpoints/nested/last.ckpt').write_bytes(b'last bytes')
        (root/'checkpoints/nested/settings.json').write_text('{}')
        (root/'status.json').write_text(json.dumps(status));(root/'attempts.jsonl').write_text(''.join(json.dumps(e)+'\n' for e in events))
        roots.append(root)
    reader=dict(uid=os.getuid(),groups=os.getgroups(),name='fixture-reader')
    pc,pi=[p.snapshot(root,reader) for root in roots]
    now=time.time()
    remote=dict(pi=pi,cloud=dict(metadata=dict(items=[dict(key='job-id',value=JOB),dict(key='job-mode',value='run')])),
                watchdog=dict(utc=iso(now),last_instance_read_utc=iso(now),job=JOB,vm='rtx6000pro',phase='UNARMED'),
                config=dict(armed=False,job=JOB))
    return roots,reader,pc,remote,now


def failures(rows):return [row for row in rows if row[1]=='FAIL']


class AuditTests(unittest.TestCase):
    def setUp(self):
        self.temp=TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.roots,self.reader,self.pc,self.remote,self.now=fixtures(self.temp.name)
    def rows(self,**kw):return p.rows_for(JOB,self.pc,self.remote,now=self.now,**kw)
    def test_good_recursive_inventory_completion_interruption_and_info_only_record(self):
        rows=self.rows();self.assertFalse(failures(rows),p.table(rows))
        self.assertEqual(sum(row[0].startswith('SHA ') for row in rows),3)
        self.assertIn(('runner record','INFO','VM-only, not read'),rows)
        self.assertIn('0644',p.table(rows));self.assertIn('2 contiguous attempts',p.table(rows))
        for reason in ('interrupted exit=130','abrupt stop detected on next boot; time is detection time'):
            self.remote['pi']['attempts']['value'][1]['reason']=reason
            self.assertFalse(failures(self.rows()))

    def test_missing_extra_and_mismatched_recursive_files(self):
        key='nested/deeper/model.ckpt'
        for change in ('missing','extra','hash','empty','error'):
            with self.subTest(change=change):
                before=deepcopy(self.remote)
                if change=='missing':del self.remote['pi']['files'][key]
                elif change=='extra':self.remote['pi']['files']['other/new.ckpt']=deepcopy(self.pc['files'][key])
                elif change=='hash':self.remote['pi']['files'][key]['sha256']='0'*64
                elif change=='empty':self.remote['pi']['files']={}
                else:self.remote['pi']['errors']=['unreadable checkpoint']
                self.assertTrue(failures(self.rows()));self.remote=before

    def test_symlinks_unsafe_names_special_files_and_intended_reader_permissions(self):
        root=self.roots[1];target=root/'checkpoints/nested/deeper/model.ckpt'
        target.chmod(0o600)
        other=dict(uid=12345,groups=[12345],name='other-reader')
        snap=p.snapshot(root,other);self.assertTrue(snap['errors']);self.assertIn('mode=0600',str(snap['errors']))
        # Group read is sufficient for the matching reader; owner takes precedence.
        target.chmod(0o640)
        info=target.stat();group_reader=dict(uid=12345,groups=[info.st_gid],name='group-reader')
        self.assertFalse(p.snapshot(root,group_reader)['errors'])
        target.chmod(0o644)
        for name in ('unsafe..name.ckpt','bad\nname.ckpt'):
            path=root/'checkpoints'/name;path.write_bytes(b'x')
            self.assertTrue(p.snapshot(root,self.reader)['errors']);path.unlink()
        for kind in ('file','directory','fifo'):
            path=root/'checkpoints/link'
            if kind=='file':path.symlink_to(target)
            elif kind=='directory':path.symlink_to(root/'checkpoints/nested',target_is_directory=True)
            else:os.mkfifo(path)
            self.assertTrue(p.snapshot(root,self.reader)['errors']);path.unlink()
        alias=Path(self.temp.name)/'alias';alias.symlink_to(root,target_is_directory=True)
        self.assertTrue(p.snapshot(alias,self.reader)['errors'])
        folder=root/'checkpoints/nested';folder.chmod(0o644)
        self.assertTrue(p.snapshot(root,other)['errors']);folder.chmod(0o755)

    def test_unreadable_or_missing_status_attempts_and_unavailable_side(self):
        for key in ('status','attempts'):
            path=self.roots[1]/('status.json' if key=='status' else 'attempts.jsonl')
            before=path.read_bytes()
            path.chmod(0o600);snap=p.snapshot(self.roots[1],dict(uid=12345,groups=[12345],name='other'))
            self.assertIn('mode=0600',snap[key]['error']);path.chmod(0o644)
            path.unlink();self.assertIn('error',p.snapshot(self.roots[1],self.reader)[key]);path.write_bytes(before)
            path.write_text('malformed JSON')
            self.assertIn('error',p.snapshot(self.roots[1],self.reader)[key]);path.write_bytes(before)
        self.remote['pi']=dict(error='SSH unavailable');self.assertTrue(failures(self.rows()))

    def test_changed_during_hash_read_refuses(self):
        real=p.os.read;changed=False;target=self.roots[0]/'checkpoints/nested/deeper/model.ckpt'
        def read(fd,n):
            nonlocal changed
            value=real(fd,n)
            if not changed:
                changed=True
                with target.open('ab') as stream:stream.write(b'changed')
            return value
        with patch.object(p.os,'read',side_effect=read):snap=p.snapshot(self.roots[0],self.reader)
        self.assertTrue(snap['errors'])

    def test_unbalanced_repeated_noncontiguous_wrong_job_and_bad_final_attempts(self):
        original=deepcopy(self.remote['pi']['attempts']['value'])
        changes=[lambda e:e.pop(),lambda e:e.__setitem__(2,deepcopy(e[0])),
                 lambda e:e[2].update(attempt=3),lambda e:e[1].update(event='start'),
                 lambda e:e[3].update(job_id='wrong'),lambda e:e[3].update(reason='done exit=1'),
                 lambda e:e[3].update(reason='failed exit=0'),lambda e:e[1].update(reason='failed exit=1'),
                 lambda e:e[2].update(time_utc=iso(1)),lambda e:e[0].update(time_utc='missing timezone'),
                 lambda e:e[0].update(attempt=True)]
        for change in changes:
            with self.subTest(change=change):
                events=deepcopy(original);change(events);self.remote['pi']['attempts']['value']=events
                self.assertTrue(failures(self.rows()))
        self.remote['pi']['attempts']['value']=original

    def test_status_or_delivered_copies_disagree(self):
        for key,value in (('state','running'),('exit_code',1),('exit_code',False),('job_id','another-job')):
            before=deepcopy(self.remote)
            self.remote['pi']['status']['value'][key]=value;self.assertTrue(failures(self.rows()));self.remote=before
        for key in ('status','attempts'):
            self.remote['pi'][key]['sha256']='0'*64;self.assertTrue(failures(self.rows()))

    def test_vm_job_mode_identity_and_explicit_expected_override(self):
        self.remote['cloud']['metadata']['items'][1]['value']='resume'
        self.assertTrue(failures(self.rows()));self.assertFalse(failures(self.rows(expected_mode='resume')))
        self.remote['cloud']['metadata']['items'][0]['value']='other-job';self.assertTrue(failures(self.rows(expected_mode='resume')))
        self.remote['cloud']=dict(error='gcloud unavailable');self.assertTrue(failures(self.rows()))

    def test_watchdog_armed_phase_stale_wrong_job_and_explicit_expectations(self):
        original=deepcopy(self.remote)
        for key,value in (('phase','OBSERVING'),('job','other'),('utc',iso(self.now-301)),
                          ('last_instance_read_utc',iso(self.now-301)),('utc',iso(self.now+60))):
            self.remote['watchdog'][key]=value;self.assertTrue(failures(self.rows()));self.remote=deepcopy(original)
        for key,value in (('armed',True),('armed','false'),('job','other')):
            self.remote['config'][key]=value;self.assertTrue(failures(self.rows()));self.remote=deepcopy(original)
        self.remote['watchdog']['phase']='OBSERVING';self.remote['config']['armed']=True
        self.assertFalse(failures(self.rows(expected_phase='OBSERVING',expected_armed=True)))
        self.remote['watchdog']=dict(error='permission denied');self.assertTrue(failures(self.rows()))

    def test_unsafe_remote_paths_malformed_hashes_and_table_control_escaping(self):
        self.remote['pi']['files']['../outside.ckpt']=dict(sha256='0'*64,mode='0644')
        rows=self.rows();self.assertTrue(failures(rows))
        text=p.table([('bad\x1b\npath','FAIL','a|b')]);self.assertNotIn('\x1b',text);self.assertIn('a\\|b',text)
        for path in ('../x','/x','x/../a','x\\y','a//b','./a'):
            with self.assertRaises(ValueError):p.safe_name(path)
        for document in ('{"state":"failed","state":"done"}','{"exit_code":NaN}'):
            with self.assertRaises(ValueError):p.load_json(document)


FAKE=r'''import json,os,pathlib,sys
name=pathlib.Path(sys.argv[0]).name;args=sys.argv[1:]
with open(os.environ['CALLS'],'a') as f:f.write(json.dumps([name,*args])+'\n')
if name=='gcloud':
 assert args[:3]==['compute','instances','describe'], args
 assert '--project=personal-compute-511601' in args and '--format=json' in args
 key='cloud'
else:
 assert 'rpi400@rpi400-server' in args and 'StrictHostKeyChecking=yes' in args
 command=args[-1]
 if command=='python3 -B - unit-postflight --collect':
  source=sys.stdin.read();assert 'def snapshot(' in source and 'os.O_RDONLY' in source
  key='pi'
 elif command=='sudo -n cat /var/lib/cvdl-watchdog/status.json':key='watchdog'
 elif command=='cat /etc/cvdl-watchdog/watchdog.json':key='config'
 else:raise RuntimeError('not an allowed read: '+command)
if os.environ.get('FAIL_QUERY')==key:sys.exit(124)
print((pathlib.Path(os.environ['DATA'])/(key+'.json')).read_text())
'''


class TransportTests(unittest.TestCase):
    def test_fake_ssh_gcloud_read_only_good_bad_unavailable_and_cli_table(self):
        with TemporaryDirectory() as area:
            area=Path(area);roots,reader,pc,remote,now=fixtures(area);tools=area/'bin';tools.mkdir()
            for name in ('ssh','gcloud'):
                path=tools/name;path.write_text('#!'+sys.executable+'\n'+FAKE);path.chmod(0o755)
            for key,value in remote.items():(area/(key+'.json')).write_text(json.dumps(value))
            env=dict(os.environ,PATH=str(tools)+os.pathsep+os.environ['PATH'],DATA=str(area),CALLS=str(area/'calls'))
            base=[sys.executable,str(ROOT/'scripts/g4/postflight.py'),JOB,'--pc-root',str(area/'pc'),'--pc-reader',str(os.getuid())]
            good=subprocess.run(base,env=env,text=True,capture_output=True,timeout=15)
            self.assertEqual(good.returncode,0,good.stdout+good.stderr);self.assertEqual(good.stdout.count('| Check |'),1)
            self.assertNotIn('| FAIL |',good.stdout);self.assertIn('| runner record | INFO | VM-only, not read |',good.stdout)
            for key in ('pi','cloud','watchdog','config'):
                result=subprocess.run(base,env=dict(env,FAIL_QUERY=key),text=True,capture_output=True,timeout=15)
                self.assertEqual(result.returncode,2,result.stdout+result.stderr);self.assertIn('| FAIL |',result.stdout)
            (area/'cloud.json').write_text('malformed')
            result=subprocess.run(base,env=env,text=True,capture_output=True,timeout=15)
            self.assertEqual(result.returncode,2);self.assertIn('current VM metadata | FAIL',result.stdout)
            calls=[json.loads(line) for line in (area/'calls').read_text().splitlines()]
            self.assertTrue(all(call[0] in ('ssh','gcloud') for call in calls))
            before=len(calls)
            result=subprocess.run([*base[:2],'../unsafe'],env=env,text=True,capture_output=True,timeout=15)
            self.assertEqual(result.returncode,2);self.assertIn('unsafe job ID',result.stdout)
            self.assertEqual(len((area/'calls').read_text().splitlines()),before)

    def test_remote_stdin_never_queries_and_malformed_snapshot_is_nonzero(self):
        with TemporaryDirectory() as area:
            roots,reader,pc,remote,now=fixtures(area)
            frame='\n---CVDL-POSTFLIGHT---\n'.join('0\n'+json.dumps(remote[key]) for key in ('pi','cloud','watchdog','config'))
            args=[sys.executable,str(ROOT/'scripts/g4/postflight.py'),JOB,'--pc-root',str(Path(area)/'pc'),'--pc-reader',str(os.getuid()),'--remote-stdin']
            for text,expected in ((frame,0),('garbage',2)):
                result=subprocess.run(args,input=text,text=True,capture_output=True,timeout=15,env=dict(os.environ,PATH='/does-not-exist'))
                self.assertEqual(result.returncode,expected,result.stdout+result.stderr)


if __name__=='__main__':unittest.main()
