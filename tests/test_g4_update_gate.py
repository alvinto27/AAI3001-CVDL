"""Run the production shell path with all system/network/package commands stubbed."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest

ROOT = Path(__file__).resolve().parents[1]
UNITS = ('apt-daily.timer', 'apt-daily-upgrade.timer', 'apt-daily.service',
         'apt-daily-upgrade.service', 'unattended-upgrades.service')

STUB = r'''import json, os, pathlib, subprocess, sys
root = pathlib.Path(os.environ['G4_REPO_ROOT'])
name = pathlib.Path(sys.argv[0]).name
args = sys.argv[1:]
with (root/'calls').open('a') as f: f.write(json.dumps([name, *args])+'\n')
if name == 'systemctl':
    assert args[0]=='show' and args[-1]=='--value', args
    unit=args[1]; prop=args[2].split('=',1)[1]
    states=json.loads(os.environ['UNIT_STATES'])
    print(states.get(unit,{}).get(prop, 'masked' if prop=='LoadState' else 'inactive'))
    if [unit, prop] in json.loads(os.environ.get('QUERY_FAILURES','[]')): sys.exit(1)
elif name == 'curl':
    key=args[args.index('-o')-1].rsplit('/',1)[1]
    metadata=json.loads(os.environ['METADATA'])
    if key not in metadata: sys.exit(22)
    pathlib.Path(args[args.index('-o')+1]).write_text(metadata[key])
elif name == 'python':
    if args[0].endswith('resume.py'):
        if args[1]=='checkpoint': print('{}')
        else: sys.exit(subprocess.call([sys.executable, *args]))
    elif args[0].endswith('preflight.py'): print('notebooks/unit.ipynb')
    elif args[0].endswith('run_notebook.py'):
        out=pathlib.Path(args[args.index('--output')+1]); out.mkdir(parents=True,exist_ok=True)
        (out/'run.log').write_text('STUB RUNNER ONLY\n')
        (out/'status.json').write_text('{"state":"done","exit_code":0}')
    else: raise RuntimeError(args)
elif name == 'git':
    assert args[0] in ('status','rev-parse','fetch','checkout'), args
    if args[0]=='rev-parse': print('a'*40)
elif name == 'sudo':
    assert args==['shutdown','-h','now'], args
elif name in ('uv','tailscale','getent','rsync','nvidia-smi'):
    pass
else: raise RuntimeError((name,args))
'''


class UpdateGateTests(unittest.TestCase):
    def boot(self, area, mode='run', states=None, failures=()):
        root = Path(area)
        for name in ('scripts/g4/job_run.sh', 'scripts/g4/resume.py', 'cvdl/__init__.py', 'cvdl/file_io.py'):
            target = root / name; target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT / name, target)
        tools = root / 'bin'; tools.mkdir()
        for name in ('systemctl', 'curl', 'git', 'sudo', 'tailscale', 'getent', 'rsync', 'nvidia-smi'):
            path = tools / name; path.write_text('#!' + sys.executable + '\n' + STUB); path.chmod(0o755)
        for name in ('.venv/bin/python', '.local/bin/uv'):
            path = root / name; path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text('#!' + sys.executable + '\n' + STUB); path.chmod(0o755)
        key = root / '.ssh/cvdl-g4-inbox'; key.parent.mkdir(); key.write_text('not a credential')
        job = 'unit-update-gate'; notebook = 'notebooks/unit.ipynb'; commit = 'a' * 40
        job_env = f'V3_NOTEBOOK={notebook}\nCVDL_JOB_ID={job}\n'
        state = root / 'state'; out = root / 'runs-out' / job
        if mode == 'resume':
            out.mkdir(parents=True); (out/'run.log').write_text('PRIOR RUN\n')
            records = state / 'jobs'; records.mkdir(parents=True)
            (records / (job + '.record')).write_text(
                f'job-id={job}\nstate=delivered\ntraining-state=interrupted\nprotocol=resume-v1\n'
                f'commit={commit}\nnotebook={notebook}\nattempt=0\n'
                f'env-sha256={hashlib.sha256(job_env.rstrip().encode()).hexdigest()}\n'
                'started-at=2026-10-26T00:00:00Z\ndelivered=yes\n')
        metadata = dict(zip(('job-mode','job-id','job-commit','job-env','job-attempt','job-timeout-min','max-run-minutes'),
                            (mode, job, commit, job_env, '1' if mode=='resume' else '0', '1', '120')))
        env = dict(os.environ, HOME=str(root), G4_REPO_ROOT=str(root), G4_STATE_ROOT=str(state),
                   PATH=str(tools)+os.pathsep+os.environ['PATH'], UNIT_STATES=json.dumps(states or {}),
                   QUERY_FAILURES=json.dumps(failures), METADATA=json.dumps(metadata))
        result = subprocess.run(['bash', str(root/'scripts/g4/job_run.sh')], cwd=root,
                                env=env, text=True, capture_output=True, timeout=15)
        calls = [json.loads(line) for line in (root/'calls').read_text().splitlines()]
        return result, calls, state, out, job

    def assert_refusal(self, area, mode, states=None, failures=(), units=()):
        result, calls, state, out, job = self.boot(area, mode, states, failures)
        self.assertEqual(result.returncode, 2, result.stdout + result.stderr)
        self.assertFalse(any(call[0]=='uv' or any('run_notebook.py' in arg for arg in call) for call in calls))
        self.assertFalse(any(call[0]=='git' for call in calls))
        self.assertEqual(sum(call[0]=='systemctl' for call in calls), 10)
        self.assertTrue(all(call[1]=='show' for call in calls if call[0]=='systemctl'))
        record = (state/'jobs'/f'{job}.record').read_text()
        self.assertIn('training-state=failed\n', record); self.assertIn('exit-code=2\n', record)
        self.assertIn('delivered=yes\n', record)
        if mode == 'resume':
            self.assertEqual((out/'run.log').read_text(), 'PRIOR RUN\n')
        attempts = [json.loads(line) for line in (out/'attempts.jsonl').read_text().splitlines()]
        self.assertEqual([row['event'] for row in attempts], ['start','end'])
        self.assertEqual(attempts[0]['attempt'], 1 if mode=='resume' else 0)
        self.assertTrue(attempts[1]['time_utc']); reason = attempts[1]['reason']
        self.assertIn('failed exit=2; automatic-update policy refused:', reason)
        for unit in units:
            for text in (result.stdout, (state/'logs/boot.log').read_text(), (out/'job-console.log').read_text(), reason):
                self.assertIn(unit, text)
        upload = next(i for i, call in enumerate(calls) if call[0]=='rsync')
        shutdown = next(i for i, call in enumerate(calls) if call==['sudo','shutdown','-h','now'])
        ended = next(i for i, call in enumerate(calls) if 'attempt' in call and 'end' in call)
        started = next(i for i, call in enumerate(calls) if 'attempt' in call and 'start' in call)
        first_query = next(i for i, call in enumerate(calls) if call[0]=='systemctl')
        self.assertLess(started, first_query); self.assertLess(ended, upload); self.assertLess(upload, shutdown)
        return reason

    def test_each_unmasked_or_not_inactive_unit_refuses_run_and_resume(self):
        for mode in ('run', 'resume'):
            for unit in UNITS:
                for prop, value in (('LoadState','loaded'), ('ActiveState','active')):
                    with self.subTest(mode=mode, unit=unit, prop=prop), TemporaryDirectory() as area:
                        reason = self.assert_refusal(area, mode, {unit:{prop:value}}, units=[unit])
                        self.assertIn(f'{prop}={value}', reason)

    def test_query_failure_refuses_even_if_stdout_claims_safe_state(self):
        for mode in ('run', 'resume'):
            for prop in ('LoadState', 'ActiveState'):
                with self.subTest(mode=mode, prop=prop), TemporaryDirectory() as area:
                    reason = self.assert_refusal(area, mode, failures=[[UNITS[2],prop]], units=[UNITS[2]])
                    self.assertIn(f'{prop}=query-failed', reason)

    def test_multiple_bad_units_and_empty_or_unknown_results_are_named(self):
        for mode in ('run', 'resume'):
            with self.subTest(mode=mode), TemporaryDirectory() as area:
                self.assert_refusal(area, mode, {UNITS[0]:{'LoadState':''}, UNITS[-1]:{'ActiveState':'unknown'}},
                                    units=[UNITS[0], UNITS[-1]])

    def test_good_policy_allows_package_work_and_runner_then_delivery_shutdown(self):
        for mode in ('run', 'resume'):
            with self.subTest(mode=mode), TemporaryDirectory() as area:
                result, calls, state, out, job = self.boot(area, mode)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                queries = [i for i, call in enumerate(calls) if call[0]=='systemctl']
                self.assertEqual(len(queries),10)
                uv = next(i for i, call in enumerate(calls) if call[:3]==['uv','sync','--locked'])
                runner = next(i for i, call in enumerate(calls) if any('run_notebook.py' in arg for arg in call))
                self.assertLess(max(queries),uv); self.assertLess(uv,runner)
                self.assertIn('training-state=done\n', (state/'jobs'/f'{job}.record').read_text())
                self.assertTrue(any(call[0]=='rsync' for call in calls))
                self.assertEqual(calls[-1], ['sudo','shutdown','-h','now'])

    def test_maintenance_and_off_do_not_query_units_or_do_package_work(self):
        for mode in ('maintenance', 'off'):
            with self.subTest(mode=mode), TemporaryDirectory() as area:
                result, calls, _, _, _ = self.boot(area, mode, {unit:{'LoadState':'loaded'} for unit in UNITS})
                self.assertEqual(result.returncode,0,result.stdout+result.stderr)
                self.assertFalse(any(call[0] in ('systemctl','uv') for call in calls))
                self.assertEqual(any(call[0]=='sudo' for call in calls), mode=='off')


if __name__ == '__main__': unittest.main()
