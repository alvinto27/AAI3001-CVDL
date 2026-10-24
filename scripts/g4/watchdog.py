#!/usr/bin/env python3
"""Bounded, explicitly enrolled Spot resume controller. Unarmed by default."""
import argparse
from datetime import datetime,timezone
import json
import logging
from logging.handlers import RotatingFileHandler
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from scripts.g4.preflight import resume_identity
from scripts.g4.resume_request import validate_checkout
from scripts.g4.watchdog_common import (Cloud,STOPPED,atomic,digest,identity,lock,matches_operation,
    metadata,read_json,scheduling,timestamp)

DEFAULTS=dict(armed=False,job=None,audit_approved=False,max_resumes=3,max_start_calls=15,
    retry_seconds=240,budget_usd=20.,hourly_rate_usd=2.,margin_usd=.5,poll_seconds=60)


def configuration(path):
    c=read_json(path,65536)
    if set(c)!=set(DEFAULTS):raise ValueError('configuration keys must match template')
    if type(c['armed']) is not bool or type(c['audit_approved']) is not bool:raise ValueError('invalid armed/audit flag')
    for key,low,high in (('max_resumes',1,3),('max_start_calls',1,15),('retry_seconds',240,86400),('poll_seconds',60,300)):
        if type(c[key]) is not int or not low<=c[key]<=high:raise ValueError('invalid '+key)
    for key in ('budget_usd','hourly_rate_usd','margin_usd'):
        v=c[key]
        if type(v) not in (int,float) or not math.isfinite(v) or v<=0:raise ValueError('invalid '+key)
    if c['retry_seconds']!=240:raise ValueError('approved capacity delay is240s')
    if c['budget_usd']>20 or c['margin_usd']<.5:raise ValueError('budget exceeds approved cap or margin too small')
    if c['job'] is not None:
        import re
        if not isinstance(c['job'],str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9._-]{0,100}',c['job']) or '..' in c['job']:raise ValueError('invalid job')
    if c['armed'] and (not c['audit_approved'] or not c['job']):raise ValueError('arming requires explicit audit approval and named job')
    return c


def terminal(snapshot,ident,sha):
    if not snapshot:return None
    if snapshot.get('source_commit')!=ident['commit'] or snapshot.get('notebook_sha256')!=sha:raise ValueError('inbox identity mismatch')
    if snapshot.get('effective_settings',{}).get('CVDL_JOB_ID',ident['job'])!=ident['job']:raise ValueError('inbox job mismatch')
    code=snapshot.get('exit_code');state=snapshot.get('state')
    if code==124:return 'TIMEOUT'
    if state=='failed' or (code is not None and code not in (0,130,143)):return 'FAILED'
    if state=='done' and code==0:return 'DONE'
    if state=='done' or code==0:raise ValueError('contradictory terminal inbox state')
    return None


def fresh_preemptions(operations,document,used):
    boot=timestamp(document['lastStartTimestamp'])
    return sorted((op for op in operations if matches_operation(op,document)
        and op.get('operationType')=='compute.instances.preempted' and op.get('status')=='DONE'
        and not op.get('error') and op.get('name') not in used
        and timestamp(op.get('startTime',op.get('insertTime')))>=boot),key=lambda op:timestamp(op.get('startTime',op.get('insertTime'))))


class Engine:
    def __init__(self,path,*,cloud=None,root=ROOT,inbox=Path('/srv/cvdl-results'),clock=time.time,submit=None):
        self.path=Path(path);self.cloud=cloud or Cloud();self.root=Path(root);self.inbox=Path(inbox);self.clock=clock
        self.submit=submit or self.canonical_submit
        anchor=self.path.parent/'installed.json'
        if self.path.exists():
            self.state=read_json(self.path)
            if self.state.get('version')!=1 or not isinstance(self.state.get('jobs'),dict):raise ValueError('corrupt controller state')
            if not anchor.exists() or read_json(anchor)['created_at']!=self.state['created_at']:raise ValueError('installation anchor mismatch')
        else:
            if anchor.exists():raise ValueError('missing state: caps cannot be reset')
            self.state=dict(version=1,created_at=self.clock(),jobs={})
            atomic(anchor,dict(created_at=self.state['created_at']));self.save()
    def save(self):atomic(self.path,self.state)
    def canonical_submit(self,job,request,limit):
        env=dict(os.environ,CVDL_G4_CONTROLLER_PYTHON=sys.executable)
        done=subprocess.run(['bash',str(self.root/'scripts/g4/submit.sh'),'resume',job,
            '--request-file',str(request),'--start-limit',str(limit)],cwd=self.root,env=env,
            capture_output=True,text=True,timeout=150,check=True)
        return json.loads(done.stdout)
    def status(self,phase,reason='',**fields):
        doc=getattr(self,'document',{})
        job=self.state.get('jobs',{}).get(self.state.get('active_job'),{})
        counters={}
        if job:
            try:
                counters=dict(resume_intents=len(job['requests']),start_calls=self.start_calls(job),
                    estimated_usd=job.get('estimated_usd'))
                if job['requests']:
                    p=Path(job['requests'][-1]['path'])
                    if p.exists():counters['next_retry_utc']=read_json(p).get('next_start_at')
            except Exception:counters=dict(counters_valid=False)
        result=dict(utc=datetime.fromtimestamp(self.clock(),timezone.utc).isoformat(),
            job=self.state.get('active_job'),vm='rtx6000pro',vm_state=doc.get('status'),boot=doc.get('lastStartTimestamp'),
            last_instance_read_utc=self.state.get('last_instance_read_utc'),phase=phase,reason=reason)
        result.update(counters);result.update(fields)
        atomic(self.path.parent/'status.json',result)
        logging.info(json.dumps(result,allow_nan=False))
        return result
    def delivered(self,j):
        path=self.inbox/j['identity']['job']/'status.json'
        return read_json(path) if path.exists() else None
    def start_calls(self,j):
        total=0
        for r in j['requests']:
            if not Path(r['path']).exists():
                if r.get('created'):raise ValueError('missing request counter')
                continue
            n=read_json(r['path'])['start_calls']
            if type(n) is not int or not 0<=n<=r.get('limit',15):raise ValueError('corrupt start counter')
            total+=n
        if total>j['policy']['max_start_calls']:raise ValueError('start counter exceeds approved cap')
        return total
    def cost(self,j,document,now):
        boot=document['lastStartTimestamp'];start=timestamp(boot)
        if start>now+5:raise ValueError('future VM boot timestamp')
        boots=j['boots']
        if boot not in boots:
            # A new boot closes the previous unknown running interval no later than new start.
            for interval in boots.values():
                if interval['end'] is None:interval['end']=start
            boots[boot]=dict(start=start,end=None)
        if document['status'] not in STOPPED:
            # Provisioning may precede an updated lastStartTimestamp. Charge the
            # unknown interval, even if the prior poll had confirmed a stop.
            boots[boot]['end']=None
        elif boots[boot]['end'] is None:boots[boot]['end']=now
        intervals=sorted((b['start'],now if b['end'] is None else b['end']) for b in boots.values())
        total=0.;end=-float('inf')
        for a,b in intervals:
            if b<a:raise ValueError('invalid billing interval')
            total+=max(0,b-max(a,end));end=max(end,b)
        return total/3600*j['policy']['hourly_rate_usd']
    def stop_owned(self,j,document,now):
        if document['status'] in STOPPED:return 'confirmed-stopped'
        if j.get('stop_tries',0)>=3:return 'HIGH-PRIORITY: stop unconfirmed; GCE backstop only'
        if now<j.get('next_stop_at',0):return 'waiting-for-stop'
        j.update(stop_tries=j.get('stop_tries',0)+1,next_stop_at=now+60);self.save()
        try:self.cloud.stop()
        except Exception:return 'HIGH-PRIORITY: stop request failed; confirmation pending'
        return 'stop-requested; confirmation pending'
    def tick(self,c):
        now=self.clock()
        document=self.cloud.describe();self.document=document
        self.state['last_instance_read_utc']=datetime.fromtimestamp(now,timezone.utc).isoformat();self.save()
        if not c['armed']:
            # Read-only acceptance checks both required API paths. Never adopt or mutate a VM.
            self.cloud.operations()
            return self.status('UNARMED','read-only get/operations succeeded')
        ident=identity(document);scheduling(document)
        if ident['job']!=c['job']:return self.status('PAUSED','different retained job; no VM action')
        policy={k:v for k,v in c.items() if k not in ('armed','job','audit_approved')}
        jobs=self.state['jobs'];self.state['active_job']=c['job']
        if c['job'] not in jobs:
            if timestamp(document['lastStartTimestamp'])<self.state['created_at'] or int(metadata(document)['job-attempt'])!=0:
                return self.status('PAUSED','future first-attempt jobs only; old/active boot cannot be enrolled')
            proof=validate_checkout(self.root,ident)
            jobs[c['job']]=dict(identity=ident,vm_id=str(document['id']),proof=proof,policy=policy,phase='OBSERVING',
                boots={},requests=[],used_preemptions=[],inspection_used=False)
            self.save()
        j=jobs[c['job']]
        if (j.get('phase') not in ('OBSERVING','PAUSED','DONE','FAILED','TIMEOUT','CAPPED')
                or not isinstance(j.get('requests'),list) or len(j['requests'])>c['max_resumes']
                or type(j.get('inspection_used')) is not bool or not isinstance(j.get('boots'),dict)
                or not isinstance(j.get('used_preemptions'),list)):
            raise ValueError('corrupt enrolled state')
        self.start_calls(j)
        if j['identity']!=ident or j['vm_id']!=str(document['id']) or j['policy']!=policy:
            return self.status('PAUSED','identity/policy drift; no VM action')
        pending=j['requests'][-1] if j['requests'] else None
        n=len(j['requests']);attempt=int(metadata(document)['job-attempt'])
        waiting=bool(pending and not pending.get('consumed'))
        expected={n-1,n} if waiting else {n}
        new_boot=bool(j['boots'] and document['lastStartTimestamp'] not in j['boots'])
        authorized_boot=bool((waiting and attempt==n and pending.get('created') and Path(pending['path']).exists())
            or any(r.get('consumed') and r.get('boot')==document['lastStartTimestamp'] for r in j['requests']))
        if attempt not in expected or (new_boot and not authorized_boot):
            # Controller downtime/manual resumes must not hide billable boots or
            # reset the three-resume cap. Charge the unknown gap conservatively.
            if new_boot:
                previous=max(j['boots'].values(),key=lambda b:b['start'])
                previous['end']=max(previous['end'] or previous['start'],timestamp(document['lastStartTimestamp']))
            if j['phase'] not in ('DONE','FAILED','TIMEOUT','CAPPED'):
                j.update(phase='PAUSED',reason='external attempt/boot; accounting needs operator review')
        cost=self.cost(j,document,now);j['estimated_usd']=cost;self.save()
        snapshot=self.delivered(j)
        finished=terminal(snapshot,ident,j['proof']['notebook_sha256'])
        if finished and j['phase'] not in ('DONE','FAILED','TIMEOUT','CAPPED'):
            j.update(phase=finished,terminal_at=now);self.save()
        if document['status'] not in STOPPED and now-timestamp(document['lastStartTimestamp'])>=ident['maximum']*60 and j['phase'] not in ('DONE','FAILED','TIMEOUT','CAPPED'):
            j.update(phase='TIMEOUT',terminal_at=now);self.save()
        if cost+c['margin_usd']>=c['budget_usd'] and j['phase'] not in ('DONE','FAILED','TIMEOUT'):
            j.update(phase='CAPPED',terminal_at=now);self.save()
        if j['phase'] in ('DONE','FAILED','TIMEOUT','CAPPED'):
            # Delivery/shutdown gets five minutes, except an exhausted budget stops immediately.
            action='delivery/shutdown grace'
            if j['phase']=='CAPPED' or now-j['terminal_at']>=300 or document['status'] in STOPPED:
                action=self.stop_owned(j,document,now)
            return self.status(j['phase'],action,estimated_usd=cost)
        if j['phase']=='PAUSED':return self.status('PAUSED',j.get('reason','operator review required'),estimated_usd=cost)
        if document['status']=='STOPPING':
            j.setdefault('stopping_at',now);self.save()
            if now-j['stopping_at']>600:j.update(phase='PAUSED',reason='STOPPING exceeded10min');self.save()
            return self.status(j['phase'],'waiting for stopped state',estimated_usd=cost)
        j.pop('stopping_at',None)
        pending=j['requests'][-1] if j['requests'] else None
        if pending and not pending.get('consumed'):
            request=Path(pending['path'])
            if not request.exists() and pending.get('created'):
                j.update(phase='PAUSED',reason='missing pending request; refuse counter reset');self.save()
                return self.status('PAUSED',j['reason'])
            # Mark before crossing the process boundary. A crash before receipt
            # creation may require review, but must never reset an uncertain try.
            pending['created']=True;self.save()
            try:result=self.submit(ident['job'],request,pending['limit'])
            except Exception:
                # The durable request, if any, owns the uncertain external action.
                result={'phase':'ambiguous'}
            if request.exists():pending['created']=True
            phase=result['phase']
            if phase=='consumed':
                pending.update(consumed=True,boot=result['boot']);j['phase']='OBSERVING';j.pop('uncertain_at',None)
            elif phase=='failed':j.update(phase='FAILED',terminal_at=now,failure_reason=result.get('reason','canonical resume rejected'))
            elif phase in ('ambiguous','awaiting-boot','pending'):
                j.setdefault('uncertain_at',now)
                if now-j['uncertain_at']>=600:j.update(phase='PAUSED',reason='unresolved resume/start result')
            else:j.pop('uncertain_at',None)
            self.save()
            calls=self.start_calls(j)
            return self.status(j['phase'],phase,resume_intents=len(j['requests']),start_calls=calls,estimated_usd=cost)
        if document['status'] not in STOPPED:
            if document['status'] not in ('RUNNING','STAGING','PROVISIONING','REPAIRING'):raise ValueError('unknown VM state')
            return self.status('OBSERVING','no start while VM active',estimated_usd=cost)
        events=fresh_preemptions(self.cloud.operations(),document,j['used_preemptions'])
        if not events:
            j.setdefault('unexplained_stop_at',now)
            if now-j['unexplained_stop_at']>=600:j.update(phase='PAUSED',reason='stopped without fresh preemption')
            self.save();return self.status(j['phase'],'waiting for preemption/delivery evidence',estimated_usd=cost)
        j.pop('unexplained_stop_at',None)
        calls=self.start_calls(j)
        if (len(j['requests'])>=c['max_resumes'] or calls>=c['max_start_calls']
                or cost+5*c['hourly_rate_usd']+c['margin_usd']>c['budget_usd']):
            j.update(phase='CAPPED',terminal_at=now);self.save();return self.status('CAPPED','resume/start/budget reserve cap',estimated_usd=cost)
        # A delivered interruption is current only when its attempt-end record agrees.
        known=False;attempts=self.inbox/ident['job']/'attempts.jsonl'
        if snapshot and snapshot.get('state')=='interrupted' and attempts.exists():
            raw=attempts.read_bytes()
            if len(raw)>16*1024*1024:raise ValueError('oversize attempt log')
            entries=[json.loads(line) for line in raw.splitlines()]
            known=any(e.get('job_id')==ident['job'] and e.get('attempt')==int(metadata(document)['job-attempt'])
                and e.get('event')=='end' and 'interrupted' in e.get('reason','') for e in entries)
        if not known and j['inspection_used']:
            j.update(phase='PAUSED',reason='one unknown-state inspection already used');self.save();return self.status('PAUSED',j['reason'])
        if not known:j['inspection_used']=True
        number=len(j['requests'])+1
        request=self.path.parent/'requests'/f'{ident["job"]}-{number}.json'
        if request.exists():raise ValueError('orphan request; refuse fresh intent')
        j['requests'].append(dict(path=str(request),limit=c['max_start_calls']-calls,created=False,consumed=False))
        j['used_preemptions'].append(events[-1]['name']);self.save()
        return self.status('RESUME_RESERVED','canonical resume on next poll',resume_intents=number,estimated_usd=cost)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--state',type=Path,default=Path('/var/lib/cvdl-watchdog/state.json'))
    parser.add_argument('--once',action='store_true')
    parser.add_argument('--arm',metavar='JOB');parser.add_argument('--disarm',action='store_true')
    args=parser.parse_args()
    if args.arm and args.disarm:parser.error('choose arm or disarm')
    if args.arm or args.disarm:
        c=configuration(args.config);c.update(armed=bool(args.arm),job=args.arm or c['job'])
        if args.arm and not c['audit_approved']:parser.error('audit_approved must be explicitly set after user audit')
        # Validate the prospective config before replacement.
        import re
        if args.arm and (not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9._-]{0,100}',args.arm) or '..' in args.arm):parser.error('invalid job')
        atomic(args.config,c,mode=0o644);return 0
    args.state.parent.mkdir(parents=True,exist_ok=True,mode=0o700)
    handler=RotatingFileHandler(args.state.parent/'watchdog.jsonl',maxBytes=5*1024*1024,backupCount=3)
    logging.basicConfig(level=logging.INFO,handlers=[handler])
    running=True
    def stop(*_):
        nonlocal running
        running=False
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    try:
        with lock(args.state.parent/'controller.lock'):
            engine=Engine(args.state)
            while running:
                c=configuration(args.config)
                try:engine.tick(c)
                except Exception as error:
                    # Do not include subprocess stderr or credential-bearing metadata.
                    engine.status('PAUSED','read/validation error: '+type(error).__name__)
                if args.once:break
                # Interruptible wait without extending shutdown by a full poll period.
                for _ in range(c['poll_seconds']):
                    if not running:break
                    time.sleep(1)
            engine.save()
    except BlockingIOError:return 75
    except Exception as error:
        atomic(args.state.parent/'status.json',dict(phase='PAUSED',reason='controller state/config error: '+type(error).__name__))
        return 2
    return 0


if __name__=='__main__':raise SystemExit(main())
