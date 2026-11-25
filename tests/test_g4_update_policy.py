"""Mock-only VM package policy; never run sudo/systemctl on the test host."""
import json
import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest
ROOT=Path(__file__).resolve().parents[1]

FAKE=r'''import json,os,pathlib,sys
root=pathlib.Path(os.environ['FAKE_ROOT']);args=sys.argv[1:]
with (root/'calls').open('a') as f:f.write(json.dumps(args)+'\n')
state_path=root/'state.json';state=json.loads(state_path.read_text()) if state_path.exists() else {}
cmd=args.pop(0)
if cmd=='systemctl':
    action=args[0]
    if action=='show':
        unit=args[1];prop=next((x.split('=',1)[1] for x in args if x.startswith('--property=')),'')
        if prop=='ActiveState':print('active' if os.environ.get('FAKE_BUSY')==unit else state.get(unit,{}).get('active','inactive'))
        elif prop=='LoadState':print(state.get(unit,{}).get('load','loaded'))
        else:print('Type=exec\nRestart=no\nKillMode=control-group')
    elif action in ('disable','mask'):
        if os.environ.get('FAKE_MASK_FAIL') and action=='mask':sys.exit(1)
        for unit in args[1:]:
            if unit.startswith('-'):continue
            state.setdefault(unit,{})['active']='inactive'
            if action=='mask':state[unit]['load']='masked'
        state_path.write_text(json.dumps(state))
    else:raise RuntimeError(args)
elif cmd=='mkdir':
    for arg in args:
        if not arg.startswith('-'):(root/arg.lstrip('/')).mkdir(parents=True,exist_ok=True)
elif cmd=='tee':
    path=root/args[0].lstrip('/');path.write_text(sys.stdin.read());print(path.read_text(),end='')
else:raise RuntimeError(cmd)
'''


class UpdatePolicyTests(unittest.TestCase):
    def run_policy(self,root,**changes):
        root=Path(root);bin=root/'bin';bin.mkdir(exist_ok=True)
        for name,text in [('id','#!/bin/sh\necho yangxuan\n'),('hostname','#!/bin/sh\necho "${FAKE_HOST:-rtx6000pro}"\n'),
            ('apt-config','#!/bin/sh\nprintf \'%s\\n\' "APT::Periodic::Update-Package-Lists \\"${FAKE_APT_VALUE:-0}\\";" \'APT::Periodic::Unattended-Upgrade "0";\'\n'),
            ('sudo','#!'+sys.executable+'\n'+FAKE)]:
            p=bin/name;p.write_text(text);p.chmod(0o755)
        env=dict(os.environ,FAKE_ROOT=str(root),PATH=str(bin)+':'+os.environ['PATH'],**changes)
        return subprocess.run(['bash',str(ROOT/'scripts/g4/setup.sh'),'freeze-updates'],env=env,text=True,capture_output=True)

    def test_masks_persist_policy_needrestart_exception_and_idempotence(self):
        with TemporaryDirectory() as tmp:
            root=Path(tmp)
            for _ in range(2):
                result=self.run_policy(root);self.assertEqual(result.returncode,0,result.stderr)
            config=(root/'etc/apt/apt.conf.d/zz-cvdl-no-auto-updates').read_text()
            self.assertIn('Unattended-Upgrade "0"',config)
            need=(root/'etc/needrestart/conf.d/99-cvdl-job.conf').read_text()
            self.assertEqual(need,'$nrconf{override_rc}->{qr(^cvdl-g4-job\\.service$)} = 0;\n')
            state=json.loads((root/'state.json').read_text())
            self.assertEqual(set(state),{'apt-daily.timer','apt-daily-upgrade.timer','apt-daily.service','apt-daily-upgrade.service','unattended-upgrades.service'})
            self.assertTrue(all(x==dict(active='inactive',load='masked') for x in state.values()))
            calls=[json.loads(x) for x in (root/'calls').read_text().splitlines()]
            self.assertFalse(any('daemon-reexec' in x or 'restart' in x for x in calls))
            self.assertFalse(any(x[:2]==['systemctl','stop'] for x in calls))
            self.assertIn('security updates',result.stdout)

    def test_running_job_refused_without_any_policy_writes(self):
        with TemporaryDirectory() as tmp:
            result=self.run_policy(tmp,FAKE_BUSY='cvdl-g4-job.service')
            self.assertEqual(result.returncode,2);self.assertIn('not quiescent',result.stderr)
            self.assertFalse((Path(tmp)/'etc').exists());self.assertFalse((Path(tmp)/'state.json').exists())

    def test_busy_apt_stops_only_new_timers_not_the_transaction(self):
        for unit in ('apt-daily.service','apt-daily-upgrade.service'):
            with TemporaryDirectory() as tmp:
                result=self.run_policy(tmp,FAKE_BUSY=unit)
                self.assertEqual(result.returncode,2);self.assertIn('wait for it to finish',result.stderr)
                calls=[json.loads(x) for x in (Path(tmp)/'calls').read_text().splitlines()]
                self.assertTrue(any(x[:3]==['systemctl','disable','--now'] for x in calls))
                self.assertFalse(any('mask' in x or 'stop' in x for x in calls));self.assertFalse((Path(tmp)/'etc').exists())

    def test_wrong_host_and_overridden_apt_policy_refused(self):
        with TemporaryDirectory() as tmp:
            result=self.run_policy(tmp,FAKE_HOST='not-the-vm');self.assertEqual(result.returncode,2)
            self.assertFalse((Path(tmp)/'calls').exists())
        with TemporaryDirectory() as tmp:
            result=self.run_policy(tmp,FAKE_APT_VALUE='1');self.assertNotEqual(result.returncode,0)
            self.assertNotIn('Automatic package updates are disabled',result.stdout)

    def test_mask_failure_and_base_policy_order(self):
        with TemporaryDirectory() as tmp:
            result=self.run_policy(tmp,FAKE_MASK_FAIL='1');self.assertNotEqual(result.returncode,0)
            self.assertNotIn('Automatic package updates are disabled',result.stdout)
        source=(ROOT/'scripts/g4/setup.sh').read_text().split('    base)',1)[1]
        self.assertLess(source.index('freeze_updates'),source.index('sudo apt-get update'))
        unit=(ROOT/'scripts/g4/cvdl-g4-job.service').read_text()
        self.assertNotIn('Restart=always',unit);self.assertNotIn('RefuseManualStop=',unit)


if __name__=='__main__':unittest.main()
