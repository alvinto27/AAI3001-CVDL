"""Small clean-tree evidence and FULL-only historical replay policy checks."""
from contextlib import nullcontext, redirect_stdout
import io
import json
import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from scripts import test_cpu as runner, test_support as support, required_test_level as policy
REAL_RUNTIME_IDENTITY = support.runtime_identity


class EvidenceTests(unittest.TestCase):
    def setUp(self):
        self.temp=TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)
        self.git('init','-q');self.git('config','user.name','Suite evidence tests')
        self.git('config','user.email','tests@example.invalid')
        (self.root/'.gitignore').write_text('runs/\n')
        (self.root/'source.py').write_text('original\n')
        (self.root/'runs').mkdir();(self.root/'runs/fixture.txt').write_bytes(b'fixture')
        self.git('add','.');self.git('-c','core.hooksPath=/dev/null','commit','-qm','base')
        for stub in (patch.object(support,'REQUIRED_FIXTURES',('runs/fixture.txt',)),
                     patch.object(support,'runtime_identity',return_value=dict(python='test-python',torch='test-torch'))):
            stub.start();self.addCleanup(stub.stop)
        self.result=SimpleNamespace(testsRun=3,skipped=[],wasSuccessful=lambda:True)

    def git(self,*args):
        return subprocess.check_output(['git',*args],cwd=self.root,stderr=subprocess.PIPE,text=True).strip()

    def save(self):
        key=support.full_evidence_key(self.root)
        self.assertIsNotNone(key)
        self.assertTrue(support.record_full_pass(self.root,key,self.result))
        return key

    def test_clean_full_roundtrip_is_small_ignored_and_uses_tree_not_commit(self):
        self.save()
        self.assertTrue(support.matching_full_evidence(self.root))
        self.assertEqual(self.git('status','--porcelain'),'')
        evidence=json.loads(support.evidence_path(self.root).read_text())
        self.assertEqual(evidence['key']['tree'],self.git('rev-parse','HEAD^{tree}'))
        self.assertLess(support.evidence_path(self.root).stat().st_size,1500)
        self.git('-c','core.hooksPath=/dev/null','commit','--allow-empty','-qm','same tree')
        self.assertTrue(support.matching_full_evidence(self.root))

    def test_dirty_tracked_staged_and_untracked_never_write_or_match(self):
        self.save();before=support.evidence_path(self.root).read_bytes()
        key=support.full_evidence_key(self.root)
        for kind in ('modified','staged','untracked'):
            if kind=='untracked':(self.root/'extra.py').write_text('untracked')
            else:
                (self.root/'source.py').write_text('changed')
                if kind=='staged':self.git('add','source.py')
            self.assertIsNone(support.full_evidence_key(self.root))
            self.assertFalse(support.matching_full_evidence(self.root))
            self.assertFalse(support.record_full_pass(self.root,key,self.result))
            self.assertEqual(support.evidence_path(self.root).read_bytes(),before)
            self.git('reset','--hard','HEAD')
            if (self.root/'extra.py').exists():(self.root/'extra.py').unlink()
        self.assertFalse(support.record_full_pass(self.root,None,self.result))

    def test_changed_clean_tree_during_full_cannot_write(self):
        key=support.full_evidence_key(self.root)
        (self.root/'source.py').write_text('new tree')
        self.git('add','source.py');self.git('-c','core.hooksPath=/dev/null','commit','-qm','changed')
        self.assertFalse(support.record_full_pass(self.root,key,self.result))
        self.assertFalse(support.evidence_path(self.root).exists())

    def test_fixture_contents_not_size_mtime_and_runtime_and_list_invalidate(self):
        self.save();key=support.full_evidence_key(self.root)
        path=self.root/'runs/fixture.txt';stamp=path.stat()
        path.write_bytes(b'changed');os.utime(path,ns=(stamp.st_atime_ns,stamp.st_mtime_ns))
        self.assertNotEqual(key,support.full_evidence_key(self.root))
        self.assertFalse(support.matching_full_evidence(self.root))
        path.write_bytes(b'fixture')
        with patch.object(support,'runtime_identity',return_value=dict(python='other',torch='other')):
            self.assertFalse(support.matching_full_evidence(self.root))
        with patch.object(support,'REQUIRED_FIXTURES',('runs/fixture.txt','missing')):
            self.assertIsNone(support.full_evidence_key(self.root))
            self.assertFalse(support.matching_full_evidence(self.root))

    def test_failed_empty_missing_and_invalid_evidence_never_qualify(self):
        key=support.full_evidence_key(self.root)
        self.assertFalse(support.record_full_pass(self.root,key,SimpleNamespace(testsRun=1,wasSuccessful=lambda:False)))
        self.assertFalse(support.record_full_pass(self.root,key,SimpleNamespace(testsRun=0,wasSuccessful=lambda:True)))
        self.assertFalse(support.matching_full_evidence(self.root))
        self.save();path=support.evidence_path(self.root);good=json.loads(path.read_text())
        for value in ('{', 'null', '[]', '{}', json.dumps(dict(good,level='fast')),
                      json.dumps(dict(good,passed=False)),json.dumps(dict(good,version=2)),
                      json.dumps(dict(good,tests=0))):
            path.write_text(value)
            self.assertFalse(support.matching_full_evidence(self.root),value)

    def test_runtime_identity_reads_distribution_versions_without_torch_import(self):
        with patch.object(support,'version',return_value='torch-fixture') as lookup:
            identity=REAL_RUNTIME_IDENTITY()
        lookup.assert_called_once_with('torch')
        self.assertEqual(identity['python'],sys.version)
        self.assertEqual(identity['torch'],'torch-fixture')


class PolicyTests(unittest.TestCase):
    def test_four_methods_skip_only_at_fast_and_restore_no_coverage_removed(self):
        self.assertEqual(len(runner.FULL_ONLY_METHODS),4)
        cases=[]
        for module,name,method in sorted(runner.FULL_ONLY_METHODS):
            cls=type(name,(unittest.TestCase,),{'__module__':module,method:lambda self:None,
                                                'test_small':lambda self:None})
            cases.extend([cls(method),cls('test_small')])
        def suite():return unittest.TestSuite(type(test)(test._testMethodName) for test in cases)
        with runner.select_level(suite(),'fast'):
            result=unittest.TextTestRunner(stream=io.StringIO()).run(suite())
        self.assertEqual(result.testsRun,8);self.assertEqual(len(result.skipped),4)
        self.assertTrue(all(reason==runner.REPLAY_SKIP_REASON for _,reason in result.skipped))
        with runner.select_level(suite(),'full'):
            full=unittest.TextTestRunner(stream=io.StringIO()).run(suite())
        self.assertTrue(full.wasSuccessful());self.assertEqual(full.testsRun,8);self.assertFalse(full.skipped)
        for test in cases:self.assertNotIn('__unittest_skip__',getattr(type(test),test._testMethodName).__dict__)

    def test_direct_dependencies_and_unknown_paths_force_full(self):
        for path in policy.REPLAY_FULL_FILES:
            self.assertTrue(policy.decisions([path])[0].startswith('TESTS: FULL'),path)
        for path in ('tests/fixtures/new.txt','unknown/dependency.py','cvdl-other/new.py'):
            self.assertTrue(policy.decisions([path])[0].startswith('TESTS: FULL'),path)
        self.assertTrue(policy.decisions(['scripts/cell_eval.py'])[0].startswith('TESTS: FAST'))

    def test_classifier_message_only_discharges_fast_not_full(self):
        for path,expected in [('scripts/cell_eval.py',True),('cvdl/models.py',False),('docs/guide.md',False)]:
            with patch.object(policy,'changed_paths',return_value=[path]), \
                 patch.object(support,'matching_full_evidence',return_value=True) as matching, \
                 redirect_stdout(io.StringIO()) as output:
                self.assertEqual(policy.main([]),0)
            self.assertEqual('FAST satisfied by matching FULL evidence' in output.getvalue(),expected)
            self.assertEqual(matching.called,expected)
        with patch.object(policy,'changed_paths',return_value=['scripts/cell_eval.py']), \
             patch.object(support,'matching_full_evidence',return_value=False),redirect_stdout(io.StringIO()) as output:
            policy.main([])
        self.assertNotIn('satisfied',output.getvalue())

    def test_runner_records_only_complete_full_not_fast_or_focused(self):
        for args,should_record in [([],True),(['--level','fast'],False),(['--tests','example'],False)]:
            with patch.object(runner,'suite_lock',return_value=nullcontext()), \
                 patch.object(runner,'check_fixtures'),patch.object(runner,'full_evidence_key',return_value={'key':'start'}) as key, \
                 patch.object(runner,'record_full_pass',return_value=False) as record, \
                 patch.object(runner.unittest.defaultTestLoader,'discover',return_value=unittest.TestSuite()), \
                 patch.object(runner.unittest.defaultTestLoader,'loadTestsFromNames',return_value=unittest.TestSuite()), \
                 patch.object(runner.unittest,'TextTestRunner') as text, \
                 patch.object(runner.os,'chdir'),patch.dict(os.environ), \
                 patch.object(sys,'path',sys.path.copy()),patch.object(sys,'dont_write_bytecode',False):
                text.return_value.run.return_value.wasSuccessful.return_value=True
                self.assertEqual(runner.main(args),0)
                self.assertEqual(key.called,should_record);self.assertEqual(record.called,should_record)


if __name__=='__main__':unittest.main()
