"""Tiny-process lock checks and read-only fixture/runner tests; no model work."""
from contextlib import contextmanager, nullcontext, redirect_stderr
import io
import json
import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import time
import unittest
from unittest.mock import patch

from scripts import test_cpu as runner
from scripts.test_support import LOCK_PATH, check_fixtures, suite_lock

ROOT = Path(__file__).resolve().parents[1]
CHILD = '''
import sys, time
from pathlib import Path
sys.path.insert(0, sys.argv[1])
from scripts.test_support import suite_lock
with suite_lock(Path.cwd(), 'tiny child', path=Path(sys.argv[2]), timeout=0):
    print('held', flush=True)
    if sys.argv[3] == 'wait': sys.stdin.readline()
    else: time.sleep(.3)
'''


class LockTests(unittest.TestCase):
    def child(self, root, path, mode='wait'):
        process = subprocess.Popen([sys.executable, '-B', '-c', CHILD, str(ROOT), str(path), mode],
                                   cwd=root, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                   stderr=subprocess.PIPE, text=True)
        def cleanup():
            if process.poll() is None:process.kill()
            process.communicate(timeout=5)
        self.addCleanup(cleanup)
        self.assertEqual(process.stdout.readline().strip(), 'held')
        return process

    def test_default_is_fixed_container_path_and_refusal_shows_owner_across_checkouts(self):
        self.assertEqual(LOCK_PATH, Path('/tmp/cvdl-cpu-test-suite.lock'))
        with TemporaryDirectory() as tmp:
            root=Path(tmp);a=root/'one';b=root/'two';a.mkdir();b.mkdir();path=root/'lock'
            p=self.child(a,path)
            inode=path.stat().st_ino
            for timeout in (0., .05):
                output=io.StringIO();before=time.monotonic()
                with self.assertRaisesRegex(TimeoutError, 'owner'):
                    with suite_lock(b,'second',path=path,timeout=timeout,stream=output):
                        self.fail('overlap')
                self.assertLess(time.monotonic()-before,1.)
                self.assertIn(str(p.pid),output.getvalue());self.assertIn(str(a),output.getvalue())
                self.assertIn('tiny child',output.getvalue())
            p.stdin.write('\n');p.stdin.flush();p.communicate(timeout=5)
            with suite_lock(b,'after exit',path=path,timeout=0):
                self.assertEqual(json.loads(path.read_text())['selection'],'after exit')
            self.assertEqual(path.stat().st_ino,inode)

    def test_bounded_wait_succeeds_after_other_process_exits(self):
        with TemporaryDirectory() as tmp:
            root=Path(tmp);path=root/'lock';p=self.child(root,path,'sleep')
            output=io.StringIO()
            with suite_lock(root,'waiter',path=path,timeout=3,stream=output):
                self.assertIn('locked by',output.getvalue())
            p.communicate(timeout=5)
            self.assertEqual(p.returncode,0)

    def test_killed_owner_and_exception_release_without_unlink_or_pid_trust(self):
        with TemporaryDirectory() as tmp:
            root=Path(tmp);path=root/'lock';p=self.child(root,path);inode=path.stat().st_ino
            p.kill();p.communicate(timeout=5)
            # Owner text is stale; a live process must not be mistaken for a lock.
            path.write_text(json.dumps(dict(pid=os.getpid(),selection='stale')))
            with self.assertRaisesRegex(RuntimeError,'abort'):
                with suite_lock(root,'crash recovered',path=path,timeout=0):raise RuntimeError('abort')
            with suite_lock(root,'again',path=path,timeout=0):pass
            self.assertEqual(path.stat().st_ino,inode)

    def test_invalid_timeout_and_symlink_refused(self):
        with TemporaryDirectory() as tmp:
            root=Path(tmp);path=root/'lock'
            for timeout in (-1.,float('nan'),float('inf')):
                with self.assertRaises(ValueError):
                    with suite_lock(root,'bad',path=path,timeout=timeout):pass
            self.assertFalse(path.exists())
            target=root/'target';target.write_text('unchanged');path.symlink_to(target)
            with self.assertRaises(OSError):
                with suite_lock(root,'bad',path=path,timeout=0):pass
            self.assertEqual(target.read_text(),'unchanged')


class FixtureTests(unittest.TestCase):
    def fixture(self, root):
        subprocess.run(['git','init','-q',str(root)],check=True,capture_output=True)
        (root/'notebooks').mkdir();(root/'fixture').write_bytes(b'kept')

    def test_good_and_external_data_links_are_read_only(self):
        with TemporaryDirectory() as tmp:
            root=Path(tmp)/'repo';self.fixture(root)
            external=Path(tmp)/'external';external.write_text('linked data')
            (root/'data-link').symlink_to(external)
            before=subprocess.check_output(['git','status','--porcelain'],cwd=root)
            check_fixtures(root,required=('fixture','data-link'))
            self.assertEqual(subprocess.check_output(['git','status','--porcelain'],cwd=root),before)
            self.assertFalse((root/'notebooks/triton-test-cache').exists())
            self.assertEqual((root/'fixture').read_bytes(),b'kept')

    def test_missing_dangling_cache_escape_and_unusable_git_pointer_refuse(self):
        with TemporaryDirectory() as tmp:
            root=Path(tmp)/'repo';self.fixture(root)
            for name in ('missing','dangling'):
                if name=='dangling':(root/name).symlink_to(root/'absent')
                with self.assertRaisesRegex(ValueError,name):check_fixtures(root,required=(name,))
            external=Path(tmp)/'outside';external.mkdir()
            cache=root/'notebooks/triton-test-cache';cache.symlink_to(external,target_is_directory=True)
            with self.assertRaisesRegex(ValueError,'outside checkout'):check_fixtures(root,required=('fixture',))
            cache.unlink()
            # A valid external Git pointer is allowed; a host-only/missing one is not.
            (root/'.git').rename(Path(tmp)/'gitdir')
            (root/'.git').write_text('gitdir: '+str(Path(tmp)/'gitdir')+'\n')
            check_fixtures(root,required=('fixture',))
            (root/'.git').write_text('gitdir: /missing/container/gitdir\n')
            with self.assertRaisesRegex(ValueError,'Git pointer'):check_fixtures(root,required=('fixture',))


class RunnerLockTests(unittest.TestCase):
    def test_all_modes_lock_before_loading_and_long_modes_check_fixtures(self):
        for args in ([],['--level','fast'],['--tests','tests.example.Case']):
            events=[]
            @contextmanager
            def locked(*a,**kw):
                events.append(('lock',kw['timeout']));yield;events.append(('release',))
            def loaded(*a,**kw):events.append(('load',));return unittest.TestSuite()
            with patch.object(runner,'suite_lock',side_effect=locked), \
                 patch.object(runner,'check_fixtures',side_effect=lambda root:events.append(('fixtures',))) as fixtures, \
                 patch.object(runner.unittest.defaultTestLoader,'discover',side_effect=loaded) as discover, \
                 patch.object(runner.unittest.defaultTestLoader,'loadTestsFromNames',side_effect=loaded) as focus, \
                 patch.object(runner.unittest,'TextTestRunner') as text, \
                 patch.object(runner.os,'chdir'),patch.dict(os.environ), \
                 patch.object(sys,'path',sys.path.copy()),patch.object(sys,'dont_write_bytecode',False):
                text.return_value.run.return_value.wasSuccessful.return_value=True
                self.assertEqual(runner.main(args),0)
                self.assertEqual(events[0],('lock',120.));self.assertEqual(events[-1],('release',))
                self.assertEqual(fixtures.call_count,0 if '--tests' in args else 1)
                self.assertEqual(focus.call_count,1 if '--tests' in args else 0)
                self.assertEqual(discover.call_count,0 if '--tests' in args else 1)
                if fixtures.called:self.assertLess(events.index(('fixtures',)),events.index(('load',)))

    def test_lock_and_fixture_refusals_do_not_discover(self):
        for refusal in ('lock','fixture'):
            with patch.object(runner,'suite_lock',side_effect=TimeoutError('owner PID42') if refusal=='lock' else None,
                              return_value=nullcontext()) as lock, \
                 patch.object(runner,'check_fixtures',side_effect=ValueError('fixture missing')), \
                 patch.object(runner.unittest.defaultTestLoader,'discover') as discover, \
                 patch.object(runner.os,'chdir'),patch.dict(os.environ), \
                 patch.object(sys,'path',sys.path.copy()),patch.object(sys,'dont_write_bytecode',False), \
                 redirect_stderr(io.StringIO()) as output:
                self.assertEqual(runner.main(['--no-wait']),2)
                self.assertEqual(lock.call_args.kwargs['timeout'],0.)
                discover.assert_not_called()
                self.assertIn('owner PID42' if refusal=='lock' else 'fixture missing',output.getvalue())


if __name__=='__main__':unittest.main()
