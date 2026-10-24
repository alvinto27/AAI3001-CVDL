"""Canonical submit.sh pending-resume helper. Never invoked directly by the watchdog."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from scripts.g4.preflight import resume_identity
from scripts.g4.watchdog_common import (Cloud,STOPPED,atomic,capacity,identity,lock,matches_operation,
    metadata,read_json,scheduling,timestamp)


def validate_checkout(root,ident):
    if subprocess.check_output(['git','status','--porcelain'],cwd=root,text=True).strip():raise ValueError('dirty checkout')
    subprocess.run(['git','merge-base','--is-ancestor',ident['commit'],'refs/remotes/origin/yx'],cwd=root,check=True,capture_output=True)
    return resume_identity(ident['notebook'],ident['commit'],root=root)


def validate_request(r):
    if r.get('version')!=1 or r.get('phase') not in ('prepared','metadata-set','start-inflight','pending','stockout','consumed','failed'):
        raise ValueError('corrupt request state')
    if (type(r.get('start_calls')) is not int or type(r.get('start_limit')) is not int
            or not 0<=r['start_calls']<=r['start_limit']<=15 or r['start_limit']<1):raise ValueError('corrupt start counter')
    expected=dict(r['original'])
    expected.update({'job-mode':'resume','job-attempt':str(int(expected['job-attempt'])+1)})
    if r['prepared']!=expected:raise ValueError('corrupt prepared metadata')
    timestamp(r['original_boot'])


def step(path,job,*,cloud=None,now=None,start_limit=15,root=ROOT,dry=False):
    """One reconciled transition; at most one start, only after prepared metadata."""
    cloud=cloud or Cloud();now=time.time() if now is None else now;path=Path(path)
    if type(start_limit) is not int or not 1<=start_limit<=15:raise ValueError('invalid start limit')
    document=cloud.describe();scheduling(document);ident=identity(document)
    if ident['job']!=job:raise ValueError('retained job mismatch')
    proof=validate_checkout(root,ident)
    with lock(str(path)+'.lock'):
        if path.exists():
            r=read_json(path);validate_request(r)
            if (r['identity']!=ident or r['vm_id']!=str(document['id']) or r['proof']!=proof
                    or r['start_limit']!=start_limit):raise ValueError('pending request identity/limit mismatch')
        else:
            if document['status'] not in STOPPED:raise ValueError('first resume request requires stopped VM')
            original=metadata(document);prepared=dict(original)
            prepared.update({'job-mode':'resume','job-attempt':str(int(original['job-attempt'])+1)})
            r=dict(version=1,identity=ident,proof=proof,vm_id=str(document['id']),original=original,prepared=prepared,
                original_boot=document['lastStartTimestamp'],created_at=now,phase='prepared',start_calls=0,start_limit=start_limit)
            if dry:return dict(phase='dry-run',validation=proof)
            atomic(path,r)
        if dry:return dict(phase='dry-run',validation=proof)
        if r['phase'] in ('consumed','failed'):return r
        current=metadata(document)
        if current not in (r['original'],r['prepared']):raise ValueError('metadata changed outside pending request')
        boot=timestamp(document['lastStartTimestamp'])
        if boot!=timestamp(r['original_boot']):
            if boot<timestamp(r['original_boot']) or current!=r['prepared']:raise ValueError('unexpected boot identity')
            r.update(phase='consumed',boot=document['lastStartTimestamp']);atomic(path,r);return r
        if current==r['original']:
            if document['status'] not in STOPPED:raise ValueError('cannot prepare running VM')
            # Same exact write can be reconciled after lost response. Never increment again.
            try:cloud.set_metadata(r['prepared'])
            except subprocess.CalledProcessError as error:
                if any(word in (error.stderr or '').lower() for word in ('permission','unauthenticated','quota')):
                    r.update(phase='failed',reason='auth/quota metadata rejection');atomic(path,r);return r
                raise
            r['phase']='metadata-set';atomic(path,r);return r
        if r.get('operation'):
            op=cloud.operation(r['operation'])
            if not matches_operation(op,document):raise ValueError('start operation target mismatch')
            if op.get('status')!='DONE':return dict(r,phase='pending')
            if op.get('error'):
                if not capacity(op):
                    r.update(phase='failed',reason='non-capacity start failure');atomic(path,r);return r
                if r['phase']!='stockout':
                    r.update(phase='stockout',next_start_at=now+240);atomic(path,r)
                if now<r['next_start_at']:return r
            else:return dict(r,phase='awaiting-boot')
        elif r['phase']=='start-inflight':
            # A lost API response is not a stockout. Recover exactly one matching operation.
            ops=[op for op in cloud.operations() if matches_operation(op,document)
                and op.get('operationType') in ('start','compute.instances.start')
                and timestamp(op.get('insertTime',op.get('startTime')))>=r['start_requested_at']]
            if len(ops)==1:
                r['operation']=ops[0]['name'];atomic(path,r)
            return dict(r,phase='ambiguous')
        if document['status'] not in STOPPED:return dict(r,phase='pending')
        if r['start_calls']>=r['start_limit']:
            r.update(phase='failed',reason='start call cap');atomic(path,r);return r
        # Reserve before the external write, including a possible response loss.
        r.update(phase='start-inflight',start_calls=r['start_calls']+1,start_requested_at=now)
        r.pop('operation',None);atomic(path,r)
        try:op=cloud.start()
        except subprocess.CalledProcessError as error:
            # An explicit auth/quota rejection is not capacity and must not retry.
            text=error.stderr or ''
            if any(code in text for code in ('PERMISSION_DENIED','UNAUTHENTICATED','QUOTA_EXCEEDED')):
                r.update(phase='failed',reason='auth/quota start rejection');atomic(path,r);return r
            raise
        r.update(operation=op['name'],phase='pending');atomic(path,r)
        return r


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('job');parser.add_argument('--request-file',required=True,type=Path)
    parser.add_argument('--start-limit',type=int,default=15);parser.add_argument('--dry-run',action='store_true')
    args=parser.parse_args()
    try:
        result=step(args.request_file,args.job,start_limit=args.start_limit,dry=args.dry_run)
        print(json.dumps({k:result[k] for k in ('phase','reason','start_calls','boot','validation') if k in result}))
        return 0
    except Exception as error:
        # No cloud stderr/token dumps in daemon output. Preserve request for reconciliation.
        print('resume request refused: '+type(error).__name__,file=sys.stderr);return 2


if __name__=='__main__':raise SystemExit(main())
