"""Small data-only helpers for the G4 resume protocol; no cloud or training calls."""
import argparse
from datetime import datetime,timezone
import hashlib
import io
import json
import os
from pathlib import Path
import re
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from cvdl.file_io import default_mode_temp

PROTOCOL='resume-v1'


def metadata_fields(document,job):
    if document.get('status') not in ('TERMINATED','STOPPED'):
        raise ValueError('VM must be stopped before resume')
    items=document.get('metadata',{}).get('items',[])
    if len(items)!=len({i['key'] for i in items}):raise ValueError('duplicate metadata key')
    m={i['key']:i.get('value','') for i in items}
    if m.get('job-id')!=job or m.get('job-protocol')!=PROTOCOL:
        raise ValueError('only the most recent resume-capable job can resume')
    commit=m.get('job-commit','');env=m.get('job-env','')
    if not re.fullmatch('[0-9a-f]{40}',commit):raise ValueError('resume requires original full commit SHA')
    lines=env.splitlines()
    if len(lines)!=2 or any('=' not in x for x in lines):raise ValueError('invalid retained job-env')
    e=dict(x.split('=',1) for x in lines)
    if set(e)!={'V3_NOTEBOOK','CVDL_JOB_ID'} or e['CVDL_JOB_ID']!=job:
        raise ValueError('retained job-env identity mismatch')
    if env.rstrip('\n')!=f"V3_NOTEBOOK={e['V3_NOTEBOOK']}\nCVDL_JOB_ID={job}":
        raise ValueError('noncanonical retained job-env')
    values=[m.get(k,'') for k in ('job-timeout-min','max-run-minutes','job-attempt')]
    if not all(re.fullmatch('[0-9]+',v) for v in values):raise ValueError('missing/invalid retained limits or attempt')
    timeout,maximum,attempt=map(int,values)
    if attempt>=999999999:raise ValueError('attempt counter exhausted')
    if timeout<=0 or maximum<16 or timeout>maximum-15:raise ValueError('invalid retained timeout margin')
    return [commit,e['V3_NOTEBOOK'],str(timeout),str(maximum),str(attempt+1)]


def checkpoint_info(output):
    paths=list((Path(output)/'checkpoints').glob('*/last.ckpt'))
    if len(paths)!=1:raise ValueError('resume needs exactly one existing last.ckpt')
    path=paths[0];raw=path.read_bytes()
    import torch
    saved=torch.load(io.BytesIO(raw),map_location='cpu',weights_only=False)
    if not isinstance(saved,dict) or not {'state_dict','optimizer_states','lr_schedulers','loops','epoch','global_step'}<=saved.keys():
        raise ValueError('last.ckpt lacks full training state')
    if type(saved['epoch']) is not int or type(saved['global_step']) is not int or min(saved['epoch'],saved['global_step'])<0:
        raise ValueError('invalid last checkpoint epoch/step')
    return dict(path=str(path),checkpoint_sha256=hashlib.sha256(raw).hexdigest(),epoch=saved['epoch'],step=saved['global_step'])


def outcome(output,code):
    p=Path(output)/'status.json'
    status=json.loads(p.read_text()) if p.exists() else {}
    if code=='unknown':code=str(status.get('exit_code','unknown'))
    if code=='0':return 'done'
    if code=='124':return 'timeout'
    if code in ('130','143'):return 'interrupted'
    if code not in ('unknown','None',''):return 'failed'
    if status.get('state')=='done':return 'done'
    if status.get('state')=='failed':return 'failed'
    return 'interrupted'


def atomic_text(path,text):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    fd,name=default_mode_temp(path)
    try:
        with os.fdopen(fd,'w') as f:f.write(text);f.flush();os.fsync(f.fileno())
        os.replace(name,path)
        fd=os.open(path.parent,os.O_RDONLY|os.O_DIRECTORY)
        try:os.fsync(fd)
        finally:os.close(fd)
    finally:Path(name).unlink(missing_ok=True)


def attempt(path,output,job,number,event,reason):
    path=Path(path)
    entries=[json.loads(x) for x in path.read_text().splitlines()] if path.exists() else []
    if any(e['attempt']==number and e['event']==event for e in entries):
        raise ValueError('attempt event already recorded')
    entry=dict(job_id=job,attempt=number,event=event,time_utc=datetime.now(timezone.utc).isoformat(),reason=reason)
    try:info=checkpoint_info(output)
    except Exception as error:
        info={};entry['checkpoint_error']=str(error)
    suffix='in' if event=='start' else 'out'
    entry.update({f'epoch_{suffix}':info.get('epoch'),f'step_{suffix}':info.get('step'),f'checkpoint_sha256_{suffix}':info.get('checkpoint_sha256')})
    atomic_text(path,''.join(json.dumps(e)+'\n' for e in [*entries,entry]))


def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='action',required=True)
    m=sub.add_parser('metadata');m.add_argument('job')
    r=sub.add_parser('record');r.add_argument('path')
    c=sub.add_parser('checkpoint');c.add_argument('output')
    o=sub.add_parser('outcome');o.add_argument('output');o.add_argument('code')
    a=sub.add_parser('attempt');a.add_argument('path');a.add_argument('output');a.add_argument('job');a.add_argument('number',type=int);a.add_argument('event',choices=('start','end'));a.add_argument('reason')
    args=p.parse_args()
    if args.action=='metadata':print('\n'.join(metadata_fields(json.load(sys.stdin),args.job)))
    elif args.action=='record':atomic_text(args.path,sys.stdin.read())
    elif args.action=='checkpoint':print(json.dumps(checkpoint_info(args.output)))
    elif args.action=='outcome':print(outcome(args.output,args.code))
    else:attempt(args.path,args.output,args.job,args.number,args.event,args.reason)


if __name__=='__main__':main()
