"""Validate a pinned notebook and its literal top-cell settings without training."""
import argparse
import ast
from dataclasses import asdict
import io
import json
from pathlib import Path, PurePosixPath
import subprocess
import sys
import tarfile
import tempfile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def validate_notebook(path, *, commit='HEAD', root=ROOT):
    """Accept only a regular tracked notebook outside every archive directory."""
    candidate = PurePosixPath(path)
    if (candidate.is_absolute() or candidate.as_posix() != path or '..' in candidate.parts
            or len(candidate.parts) < 2 or candidate.parts[0] != 'notebooks'
            or 'archive' in candidate.parts or candidate.suffix != '.ipynb' or '\\' in path
            or any(ord(c)<32 or ord(c)==127 for c in path)):
        raise ValueError(f'invalid notebook path {path!r}: use a tracked notebooks/*.ipynb outside archive/')
    entry = subprocess.check_output(['git','ls-tree',commit,'--',path],cwd=root,text=True).strip()
    if not entry or entry.split()[0] not in {'100644','100755'}:
        raise ValueError(f'notebook {path!r} is not a regular tracked file at {commit}')
    raw = subprocess.check_output(['git','show',f'{commit}:{path}'],cwd=root,text=True)
    return json.loads(raw)


def top_values(notebook):
    """Read only literal keyword arguments to the first code cell's settings=load()."""
    first = next((c for c in notebook['cells'] if c['cell_type']=='code'),None)
    if first is None:
        raise ValueError('notebook needs a first code cell with settings = load(...)')
    tree = ast.parse(''.join(first['source']))
    assignments = [node for node in tree.body if isinstance(node,ast.Assign)
        and any(isinstance(t,ast.Name) and t.id=='settings' for t in node.targets)]
    if len(assignments)!=1 or not isinstance(assignments[0].value,ast.Call):
        raise ValueError('first code cell must have exactly one settings = load(...)')
    call = assignments[0].value
    imported = any(isinstance(n,ast.ImportFrom) and n.module=='cvdl.settings'
                   and any(a.name=='load' and a.asname is None for a in n.names) for n in tree.body)
    if not imported or not isinstance(call.func,ast.Name) or call.func.id!='load' or call.args:
        raise ValueError('top cell must import cvdl.settings.load and call settings = load(key=value, ...)')
    values = {}
    for keyword in call.keywords:
        if keyword.arg is None or keyword.arg in {'overrides','environ','_profile'}:
            raise ValueError('top-cell settings must be literal experiment keywords, not env/profile/override controls')
        try:
            values[keyword.arg] = ast.literal_eval(keyword.value)
        except (ValueError,TypeError) as error:
            raise ValueError(f'top-cell {keyword.arg}: expected a literal value') from error
    return values


def preflight(notebook_path, commit, *, root=ROOT, speed_env=None):
    """Use settings code/configs FROM the selected commit in an isolated snapshot."""
    notebook = validate_notebook(notebook_path,commit=commit,root=root)
    values = top_values(notebook) if speed_env is None else {}
    from cvdl.settings import machine_profile, plain, validate_notebook_name
    if speed_env is None:
        validate_notebook_name(notebook_path,values)
        refuse_pilot(values)
    archive = subprocess.check_output(['git','archive',commit],cwd=root)
    with tempfile.TemporaryDirectory(prefix='cvdl-preflight-') as temp:
        snapshot = Path(temp)
        with tarfile.open(fileobj=io.BytesIO(archive)) as tar:
            tar.extractall(snapshot,filter='data')
        profile = plain(machine_profile('g4',root=snapshot).values)
        script = '''import json, sys
from dataclasses import asdict
from cvdl.settings import load, MachineProfile, plain
values, profile, env = (json.loads(arg) for arg in sys.argv[1:4])
settings = load(environ=env, _profile=MachineProfile('g4',profile), **values)
if getattr(settings,'train_grid','recorded')!='recorded':
    import csv
    from pathlib import Path
    from lsb_train_grid import summary
    records=[]
    for source in settings.sources:
        records.extend(csv.DictReader((Path(settings.root)/f'data/{source}/photo_manifest.csv').open()))
    summary(settings.root,records,getattr(settings,'train_tile_size',None) or settings.tile_size,settings.train_grid)
print(json.dumps(dict(values=plain(settings.values), sources=plain(settings.provenance),
    profile=settings.profile, run_name=settings.run_name, globals=settings.legacy_globals(),
    run_settings=asdict(settings.to_run_settings()))))
'''
        done = subprocess.run([sys.executable,'-c',script,json.dumps(values),json.dumps(profile),json.dumps(speed_env or {})],
                              cwd=snapshot,capture_output=True,text=True)
        if done.returncode:
            raise ValueError('G4 pre-flight failed before boot:\n'+done.stderr.strip())
        report = json.loads(done.stdout)
        if speed_env is None and 'job_id' not in report['values']:
            raise ValueError('pinned commit predates the minimal notebook-job metadata interface; use a phase-5-or-newer commit')
        # No VM absolute path is assumed: its checkout root is chosen by setup.sh.
        def display(value):
            if isinstance(value,dict):return {k:display(v) for k,v in value.items()}
            if isinstance(value,list):return [display(v) for v in value]
            return value.replace(str(snapshot),'<checkout>') if isinstance(value,str) else value
        report = display(report)
    report.update(commit=commit,notebook=notebook_path)
    return report


def refuse_pilot(values):
    """PC pilot recipes cannot pass submission or VM job-env validation."""
    from cvdl.settings import boolean
    if boolean(values.get('pilot', False)):
        raise ValueError('pilot mode is PC-only; G4 preflight and runner refuse pilots')


def validate_job_env(path, job_id, *, root=ROOT, commit='HEAD'):
    from scripts.run_notebook import settings_file
    env = settings_file(Path(path), [])
    notebook = env.get('V3_NOTEBOOK','notebooks/05_xunet_v4_nonorm.ipynb')
    if env.get('V3_SPEED_MATRIX') == 'on':
        if notebook != 'notebooks/05_xunet_v4_nonorm.ipynb':
            raise ValueError('speed matrix requires notebook 05 selector')
    elif set(env) != {'V3_NOTEBOOK','CVDL_JOB_ID'} or env.get('CVDL_JOB_ID') != job_id:
        raise ValueError('notebook job-env must contain only V3_NOTEBOOK and matching CVDL_JOB_ID')
    document = validate_notebook(notebook,commit=commit,root=root)
    if env.get('V3_SPEED_MATRIX') != 'on':
        from cvdl.settings import validate_notebook_name
        values = top_values(document)
        validate_notebook_name(notebook,values)
        refuse_pilot(values)
    return notebook


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--notebook')
    parser.add_argument('--job-env',type=Path)
    parser.add_argument('--job-id')
    parser.add_argument('--speed-env-stdin',action='store_true')
    parser.add_argument('--commit',default='HEAD')
    parser.add_argument('--root',type=Path,default=ROOT)
    parser.add_argument('--check-path-only',action='store_true')
    args=parser.parse_args()
    if not (args.notebook or args.job_env or args.speed_env_stdin):
        parser.error('choose --notebook, --job-env or --speed-env-stdin')
    try:
        if args.speed_env_stdin:
            from scripts.run_notebook import settings_file
            with tempfile.NamedTemporaryFile('w+',suffix='.env') as handle:
                handle.write(sys.stdin.read());handle.flush()
                env = settings_file(Path(handle.name),[])
            if env.get('V3_SPEED_MATRIX') != 'on' or env.get('V3_NOTEBOOK') != 'notebooks/05_xunet_v4_nonorm.ipynb':
                raise ValueError('speed preflight requires the speed-matrix env and notebook-05 selector')
            print(json.dumps(preflight(env['V3_NOTEBOOK'],args.commit,root=args.root,speed_env=env),indent=2))
        elif args.job_env:
            print(validate_job_env(args.job_env,args.job_id,root=args.root,commit=args.commit))
        elif args.check_path_only:
            validate_notebook(args.notebook,commit=args.commit,root=args.root)
        else:
            print(json.dumps(preflight(args.notebook,args.commit,root=args.root),indent=2))
        return 0
    except (ValueError,KeyError,SyntaxError,subprocess.CalledProcessError) as error:
        print(f'pre-flight: {error}',file=sys.stderr)
        return 2


if __name__=='__main__':raise SystemExit(main())
