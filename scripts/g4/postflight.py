"""Read-only delivered-job audit; no training imports or remote mutation commands."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import pwd
import re
import stat
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2] if __file__ != '<stdin>' else Path.cwd()
NOFOLLOW = os.O_RDONLY | os.O_NOFOLLOW


def load_json(text):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result: raise ValueError('duplicate JSON key')
            result[key] = value
        return result
    def invalid(value): raise ValueError('nonstandard JSON constant')
    return json.loads(text, object_pairs_hook=unique, parse_constant=invalid)


def job_id(value):
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9._-]{0,100}', value) or '..' in value:
        raise ValueError('unsafe job ID')
    return value


def safe_name(value):
    path = PurePosixPath(value)
    if (not value or path.is_absolute() or str(path) != value
            or any(not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9._-]*', part) or '..' in part for part in path.parts)):
        raise ValueError('unsafe relative path: ' + repr(value))
    return value


def reader_identity(value):
    entry = pwd.getpwuid(int(value)) if str(value).isdigit() else pwd.getpwnam(value)
    return dict(uid=entry.pw_uid, groups=os.getgrouplist(entry.pw_name, entry.pw_gid), name=entry.pw_name)


def allowed(info, reader, mask):
    shift = 6 if info.st_uid == reader['uid'] else 3 if info.st_gid in reader['groups'] else 0
    return (stat.S_IMODE(info.st_mode) >> shift) & mask == mask


def open_directory(path):
    # Reject symlinks in every ancestor, not merely the final path component.
    fd = os.open('/', NOFOLLOW | os.O_DIRECTORY)
    try:
        for part in Path(path).absolute().parts[1:]:
            next_fd = os.open(part, NOFOLLOW | os.O_DIRECTORY, dir_fd=fd)
            os.close(fd); fd = next_fd
        return fd
    except Exception:
        os.close(fd); raise


def read_file(fd, name, reader, *, contents=False):
    handle = os.open(name, NOFOLLOW | os.O_NONBLOCK, dir_fd=fd)
    try:
        before = os.fstat(handle)
        mode = f'{stat.S_IMODE(before.st_mode):04o}'
        if not stat.S_ISREG(before.st_mode): raise ValueError('not a regular file')
        if not allowed(before, reader, 4): raise ValueError('not readable for intended reader; mode=' + mode)
        digest = hashlib.sha256(); chunks = []
        while True:
            chunk = os.read(handle, 1024 * 1024)
            if not chunk: break
            digest.update(chunk)
            if contents: chunks.append(chunk)
        after = os.fstat(handle)
        keys = ('st_dev', 'st_ino', 'st_size', 'st_mtime_ns', 'st_ctime_ns')
        if any(getattr(before, key) != getattr(after, key) for key in keys): raise ValueError('file changed during read')
        result = dict(sha256=digest.hexdigest(), mode=mode)
        if contents: result['text'] = b''.join(chunks).decode('utf-8')
        return result
    finally: os.close(handle)


def snapshot(path, reader):
    """All regular files under checkpoints, including sidecars; never follow links."""
    result = dict(reader=reader, files={}, errors=[])
    try: root = open_directory(path)
    except (OSError, ValueError) as error:
        return dict(result, errors=[str(error)])
    def walk(fd, prefix=''):
        before = os.fstat(fd)
        if not allowed(before, reader, 5): raise ValueError(f'directory not readable/searchable: {prefix or "checkpoints"} mode={stat.S_IMODE(before.st_mode):04o}')
        with os.scandir(fd) as entries:
            names = sorted(entry.name for entry in entries)
        for name in names:
            relative = prefix + name
            try:
                safe_name(relative)
                info = os.stat(name, dir_fd=fd, follow_symlinks=False)
                if stat.S_ISDIR(info.st_mode):
                    child = os.open(name, NOFOLLOW | os.O_DIRECTORY, dir_fd=fd)
                    try: walk(child, relative + '/')
                    finally: os.close(child)
                elif stat.S_ISREG(info.st_mode): result['files'][relative] = read_file(fd, name, reader)
                else: raise ValueError('symlink or non-regular entry')
            except (OSError, ValueError) as error: result['errors'].append(relative + ': ' + str(error))
        after = os.fstat(fd)
        if (before.st_mtime_ns, before.st_ctime_ns) != (after.st_mtime_ns, after.st_ctime_ns):
            raise ValueError('directory changed during inventory: ' + prefix)
    try:
        try:
            if not allowed(os.fstat(root), reader, 5): raise ValueError('run directory not readable/searchable for intended reader')
            directory = os.open('checkpoints', NOFOLLOW | os.O_DIRECTORY, dir_fd=root)
            try: walk(directory)
            finally: os.close(directory)
        except (OSError, ValueError) as error: result['errors'].append(str(error))
        for key, filename in (('status', 'status.json'), ('attempts', 'attempts.jsonl')):
            try:
                item = read_file(root, filename, reader, contents=True)
                text = item.pop('text')
                if key == 'status':
                    value = load_json(text)
                    item['value'] = {k: value.get(k) for k in ('state','exit_code','source_commit','notebook_sha256','ended_at')}
                    item['value']['job_id'] = value.get('effective_settings', {}).get('CVDL_JOB_ID')
                else: item['value'] = [load_json(line) for line in text.splitlines()]
                result[key] = item
            except (OSError, ValueError, TypeError, AttributeError) as error: result[key] = dict(error=str(error))
    finally: os.close(root)
    return result


def utc(value):
    dt = datetime.fromisoformat(value.replace('Z', '+00:00'))
    if dt.tzinfo is None: raise ValueError('timestamp has no timezone')
    return dt.timestamp()


def attempts_done(entries, job):
    if not entries or len(entries) % 2: raise ValueError('missing/unbalanced start/end events')
    previous = float('-inf')
    for number in range(len(entries) // 2):
        start, end = entries[2*number:2*number+2]
        for event, expected in ((start, 'start'), (end, 'end')):
            if (type(event.get('attempt')) is not int or event['attempt'] != number
                    or event.get('event') != expected or event.get('job_id') != job):
                raise ValueError('attempt IDs/events/job must be contiguous ordered start/end pairs from 0')
            moment = utc(event['time_utc'])
            if moment < previous: raise ValueError('attempt timestamps out of order')
            previous = moment
        reason = end.get('reason', '')
        if number == len(entries)//2 - 1:
            if reason != 'done exit=0': raise ValueError('final attempt is not done exit=0: ' + str(reason))
        elif not (reason in ('interrupted exit=130', 'interrupted exit=143')
                  or reason == 'abrupt stop detected on next boot; time is detection time'):
            raise ValueError('predecessor is not interrupted: ' + str(reason))
    return f'{len(entries)//2} contiguous attempts; final done exit=0'


def sections(raw):
    parts = raw.split('\n---CVDL-POSTFLIGHT---\n')
    if len(parts) != 4: raise ValueError('malformed/unavailable remote snapshot')
    result = {}
    for key, part in zip(('pi','cloud','watchdog','config'), parts):
        code, body = part.split('\n', 1)
        try:
            value = load_json(body) if code == '0' else dict(error=f'{key} query failed/timeout (exit {code})')
            if not isinstance(value, dict): raise ValueError('expected JSON object')
            result[key] = value
        except (ValueError, TypeError) as error: result[key] = dict(error=f'{key}: malformed answer: {error}')
    return result


def rows_for(job, pc, remote, *, expected_mode='run', expected_phase='UNARMED', expected_armed=False, now=None, max_age=300):
    rows = []
    def row(name, function):
        try: detail = function(); rows.append((name, 'PASS', detail))
        except (ValueError, KeyError, TypeError, AttributeError) as error: rows.append((name, 'FAIL', str(error)))
    def require(condition, detail):
        if not condition: raise ValueError(detail)
        return detail
    pi = remote.get('pi', {}); cloud = remote.get('cloud', {})
    if not isinstance(pi, dict) or not isinstance(pi.get('files', {}), dict) or any(not isinstance(key,str) for key in pi.get('files',{})):
        pi = dict(error='malformed Pi checkpoint inventory')
    for side, data in (('PC',pc), ('Pi',pi)):
        row(side+' inventory', lambda data=data: require(not data.get('error') and not data.get('errors')
            and bool(data.get('files')) and any(name.endswith('.ckpt') for name in data.get('files',{})),
            str(data.get('error') or data.get('errors') or f"{len(data.get('files',{}))} regular checkpoint-tree files; reader={data.get('reader')}")))
    names = sorted(set(pc.get('files', {})) | set(pi.get('files', {})))
    row('checkpoint inventories', lambda: require(bool(names) and set(pc.get('files',{})) == set(pi.get('files',{})),
        f"PC-only={sorted(set(pc.get('files',{}))-set(pi.get('files',{})))}; Pi-only={sorted(set(pi.get('files',{}))-set(pc.get('files',{})))}"))
    for name in names:
        def compare(name=name):
            safe_name(name)
            a, b = pc['files'][name], pi['files'][name]
            for item in (a,b):
                if not re.fullmatch(r'[0-9a-f]{64}', item['sha256']) or not re.fullmatch(r'[0-7]{4}', item['mode']): raise ValueError('invalid hash/mode')
            return require(a['sha256'] == b['sha256'], f"PC={a['sha256']} Pi={b['sha256']}; modes PC={a['mode']} Pi={b['mode']}")
        row('SHA '+name, compare)
    for key in ('status','attempts'):
        def match(key=key):
            a, b = pc[key], pi[key]
            if 'error' in a or 'error' in b: raise ValueError(f'PC={a.get("error")} Pi={b.get("error")}')
            return require(a['sha256'] == b['sha256'], f"Pi=PC SHA256; readable modes PC={a['mode']} Pi={b['mode']}")
        row(key+' Pi=PC / modes', match)
    def status_done():
        value = pi['status']['value']
        return require(value['state'] == 'done' and type(value['exit_code']) is int and value['exit_code'] == 0 and value['job_id'] == job,
                       f"Pi state={value['state']} exit={value['exit_code']} job={value['job_id']}")
    row('delivered completion', status_done)
    row('attempt completion', lambda: attempts_done(pi['attempts']['value'],job))
    rows.append(('runner record', 'INFO', 'VM-only, not read'))
    def vm():
        if cloud.get('error'): raise ValueError(cloud['error'])
        items = cloud['metadata']['items']; m = {item['key']:item['value'] for item in items}
        if len(m) != len(items): raise ValueError('duplicate VM metadata keys')
        return require(m.get('job-id') == job and m.get('job-mode') == expected_mode,
                       f"job-id={m.get('job-id')}; job-mode={m.get('job-mode')} (expected {expected_mode})")
    row('current VM metadata', vm)
    def watchdog():
        state, config = remote['watchdog'], remote['config']
        for value in (state,config):
            if value.get('error'): raise ValueError(value['error'])
        current = time.time() if now is None else now
        ages = [current-utc(state[key]) for key in ('utc','last_instance_read_utc')]
        return require(state['phase'] == expected_phase and type(config['armed']) is bool and config['armed'] == expected_armed
            and state['job'] == job and config['job'] == job and state['vm'] == 'rtx6000pro'
            and all(-30 <= age <= max_age for age in ages) and not (state['phase']=='UNARMED' and config['armed']),
            f"phase={state['phase']}; armed={config['armed']}; job={state['job']}; config job={config['job']}; ages={','.join(f'{age:.0f}s' for age in ages)}")
    row('watchdog phase/armed/identity/freshness', watchdog)
    return rows


def table(rows):
    def cell(value):
        return re.sub(r'[\x00-\x1f\x7f]',lambda match: repr(match.group())[1:-1],str(value).replace('|','\\|'))
    return '\n'.join(['| Check | Result | Detail |','|---|---|---|'] + ['| '+' | '.join(map(cell,row))+' |' for row in rows])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('job'); parser.add_argument('--pc-root',type=Path,default=ROOT/'runs')
    parser.add_argument('--pc-reader',default=str(ROOT.stat().st_uid))
    parser.add_argument('--expected-job-mode',default='run')
    parser.add_argument('--expected-watchdog-phase',default='UNARMED')
    parser.add_argument('--expected-watchdog-armed',choices=('true','false'),default='false')
    parser.add_argument('--max-watchdog-age',type=int,default=300)
    parser.add_argument('--remote-stdin',action='store_true',help='host-collected probe frame on stdin; no network calls')
    parser.add_argument('--collect',action='store_true',help=argparse.SUPPRESS)
    args = parser.parse_args()
    try:
        job_id(args.job)
        if args.max_watchdog_age <= 0: raise ValueError('watchdog max age must be positive')
        if args.collect:
            print(json.dumps(snapshot(Path('/srv/cvdl-results')/args.job,reader_identity('rpi400')))); return 0
        pc = snapshot(args.pc_root/args.job,reader_identity(args.pc_reader))
        try:
            raw = sys.stdin.read() if args.remote_stdin else subprocess.run(['bash',str(Path(__file__).with_name('postflight_probe.sh')),args.job],
                capture_output=True,text=True,timeout=260,check=True).stdout
            remote = sections(raw)
        except (OSError,ValueError,subprocess.SubprocessError) as error: remote = dict(pi=dict(error=str(error)))
        rows = rows_for(args.job,pc,remote,expected_mode=args.expected_job_mode,expected_phase=args.expected_watchdog_phase,
                        expected_armed=args.expected_watchdog_armed=='true',max_age=args.max_watchdog_age)
    except (OSError,ValueError,KeyError) as error:
        rows = [('postflight','FAIL',str(error)),('runner record','INFO','VM-only, not read')]
    print(table(rows))
    return 2 if any(row[1]=='FAIL' for row in rows) else 0


if __name__=='__main__': raise SystemExit(main())
