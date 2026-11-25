"""Data-only G4 controller primitives. No training imports, SSH or cloud writes on import."""
from contextlib import contextmanager
from datetime import datetime,timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile

PROJECT='personal-compute-511601'
ZONE='us-central1-c'
INSTANCE='rtx6000pro'
STOPPED={'TERMINATED','STOPPED'}
CAPACITY={'ZONE_RESOURCE_POOL_EXHAUSTED','ZONE_RESOURCE_POOL_EXHAUSTED_WITH_DETAILS','RESOURCE_POOL_EXHAUSTED'}


def timestamp(value):
    if not isinstance(value,str):raise ValueError('missing timestamp')
    dt=datetime.fromisoformat(value.replace('Z','+00:00'))
    if dt.tzinfo is None:raise ValueError('timestamp requires timezone')
    return dt.astimezone(timezone.utc).timestamp()


def digest(value):return hashlib.sha256(value.encode()).hexdigest()


def read_json(path,limit=16*1024*1024):
    with Path(path).open('rb') as stream:
        raw=stream.read(limit+1)
    if len(raw)>limit:raise ValueError('oversize JSON')
    return json.loads(raw)


def atomic(path,value,*,mode=0o600):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True,mode=0o700)
    fd,name=tempfile.mkstemp(prefix='.'+path.name+'.',dir=path.parent)
    try:
        os.fchmod(fd,mode)
        with os.fdopen(fd,'w') as stream:
            json.dump(value,stream,indent=2,allow_nan=False);stream.write('\n');stream.flush();os.fsync(stream.fileno())
        os.replace(name,path)
        fd=os.open(path.parent,os.O_RDONLY|os.O_DIRECTORY)
        try:os.fsync(fd)
        finally:os.close(fd)
    finally:Path(name).unlink(missing_ok=True)


@contextmanager
def lock(path):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True,mode=0o700)
    fd=os.open(path,os.O_CREAT|os.O_RDWR,0o600)
    try:
        fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
        yield
    finally:os.close(fd)


def metadata(document):
    items=document.get('metadata',{}).get('items',[])
    result={item['key']:item.get('value','') for item in items}
    if len(result)!=len(items):raise ValueError('duplicate metadata')
    return result


def identity(document):
    # Reuse the existing strict resume protocol parser; it requires a stopped VM.
    from scripts.g4.resume import metadata_fields
    m=metadata(document)
    copy=dict(document,status='TERMINATED')
    fields=metadata_fields(copy,m.get('job-id',''))
    job=m['job-id']
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9._-]{0,100}',job) or '..' in job:raise ValueError('invalid job ID')
    return dict(job=job,commit=fields[0],notebook=fields[1],env_sha256=digest(m['job-env'].rstrip('\n')),
        timeout=int(fields[2]),maximum=int(fields[3]))


def scheduling(document):
    s=document.get('scheduling',{})
    if (s.get('provisioningModel')!='SPOT' or s.get('instanceTerminationAction','STOP')!='STOP'
            or int(s.get('maxRunDuration',{}).get('seconds',0))!=18000):
        raise ValueError('requires SPOT/STOP with actual maxRunDuration=18000s')


def resource(value):
    # Google returns both www.googleapis.com and compute.googleapis.com links.
    marker='/projects/'
    if not isinstance(value,str) or marker not in value:raise ValueError('invalid resource link')
    return 'projects/'+value.split(marker,1)[1]


def target(document):return f'projects/{PROJECT}/zones/{ZONE}/instances/{INSTANCE}'


def matches_operation(op,document):
    try:
        return (resource(op['targetLink'])==target(document) and str(op['targetId'])==str(document['id'])
            and resource(op['zone'])==f'projects/{PROJECT}/zones/{ZONE}')
    except (KeyError,ValueError):return False


def capacity(op):
    errors=op.get('error',{}).get('errors',[])
    return bool(errors) and all(e.get('code') in CAPACITY for e in errors)


class Cloud:
    """Explicit project and zone on every API call; finite process deadline."""
    def run(self,*args):
        done=subprocess.run(['gcloud',*args,'--project='+PROJECT,'--format=json','--quiet'],
            capture_output=True,text=True,timeout=40,check=True)
        return json.loads(done.stdout or 'null')
    def describe(self):return self.run('compute','instances','describe',INSTANCE,'--zone='+ZONE)
    def operations(self):
        # gcloud paginates. Do not use --limit: a recent relevant event can be later in a page.
        return self.run('compute','operations','list','--zones='+ZONE) or []
    def operation(self,name):
        if not re.fullmatch(r'[A-Za-z0-9_-]+',name):raise ValueError('invalid operation name')
        return self.run('compute','operations','describe',name,'--zone='+ZONE)
    def set_metadata(self,values):
        fd,name=tempfile.mkstemp(prefix='cvdl-job-env-')
        try:
            with os.fdopen(fd,'w') as stream:stream.write(values['job-env'])
            keys=('job-mode','job-id','job-commit','job-timeout-min','max-run-minutes','job-protocol','job-attempt')
            return self.run('compute','instances','add-metadata',INSTANCE,'--zone='+ZONE,
                '--metadata='+','.join(k+'='+values[k] for k in keys),'--metadata-from-file=job-env='+name)
        finally:Path(name).unlink(missing_ok=True)
    def start(self):
        result=self.run('compute','instances','start',INSTANCE,'--zone='+ZONE,'--async')
        if isinstance(result,list) and len(result)==1:result=result[0]
        if not isinstance(result,dict) or not result.get('name'):raise ValueError('start returned no operation')
        return result
    def stop(self):return self.run('compute','instances','stop',INSTANCE,'--zone='+ZONE,'--async')
