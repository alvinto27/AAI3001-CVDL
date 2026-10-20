"""Fast checks for CPU-level selection and the read-only changed-path policy."""
from contextlib import redirect_stdout
import io
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import required_test_level as policy
from scripts import test_cpu as runner


class PathPolicyTests(unittest.TestCase):
    def test_documentation_only_and_root_markdown_need_no_tests(self):
        paths = ['docs/README.md', 'docs/nested/guide.md', 'README.md', 'AGENTS.md', 'CHANGELOG.md']
        lines = policy.decisions(paths)
        self.assertTrue(lines[0].startswith('TESTS: NONE'))
        self.assertTrue(lines[1].startswith('GOLDEN: NOT REQUIRED'))
        self.assertEqual(lines[2], policy.REMINDER)
        for path in paths:
            self.assertIn(repr(path), lines[0])

    def test_docs_plus_code_keeps_fast_full_and_golden_rules(self):
        for path, tests, golden in [
            ('scripts/test_cpu.py', 'FAST', 'NOT REQUIRED'),
            ('cvdl/models.py', 'FAST', 'REQUIRED'),
            ('LSB-Dataset-Generator/lsb_torch.py', 'FULL', 'REQUIRED'),
            ('unknown.txt', 'FULL', 'REQUIRED'),
        ]:
            with self.subTest(path=path):
                lines = policy.decisions(['docs/guide.md', path])
                self.assertTrue(lines[0].startswith('TESTS: ' + tests))
                self.assertTrue(lines[1].startswith('GOLDEN: ' + golden))

    def test_code_read_docs_and_nontext_docs_stay_fast(self):
        for path in ('docs/active/plan-architecture-tests.md', *policy.DOCUMENTATION_INPUTS,
                     'docs/figures/other.png', 'docs/guide.txt',
                     'docs/code.py', 'docs/guide.MD', 'scripts/README.md'):
            with self.subTest(path=path):
                lines = policy.decisions(['docs/guide.md', path])
                self.assertTrue(lines[0].startswith('TESTS: FAST'))
                self.assertTrue(lines[1].startswith('GOLDEN: NOT REQUIRED'))

    def test_no_changes_need_no_tests(self):
        lines = policy.decisions([])
        self.assertTrue(lines[0].startswith('TESTS: NONE'))
        self.assertIn('(no changed paths)', lines[0])
        self.assertTrue(lines[1].startswith('GOLDEN: NOT REQUIRED'))
        self.assertEqual(lines[2], policy.REMINDER)

    def test_full_loader_dependencies(self):
        for path in ('LSB-Dataset-Generator/lsb_torch.py', 'tests/test_lsb_torch.py',
                     'tests/test_lsb_source.py', 'configs/raise512-run1.yaml',
                     'configs/test-bpp1.yaml'):
            with self.subTest(path=path):
                self.assertIsNotNone(policy.path_reasons(path)[0])

    def test_known_areas_are_fast(self):
        for path in ('cvdl/models.py', 'scripts/test_cpu.py', 'notebooks/new-experiment.ipynb',
                     'configs/new-experiment.yaml', 'data/new-test/pairs.csv',
                     'tests/test_cpu_levels.py', 'docs/runbook.md', 'README.md',
                     'AGENTS.md', 'CHANGELOG.md'):
            with self.subTest(path=path):
                self.assertIsNone(policy.path_reasons(path)[0])

    def test_golden_inputs_and_shared_code_require_golden(self):
        for path in (*policy.GOLDEN_FILES, 'tests/golden/cpu-det.env',
                     'cvdl/models.py', 'LSB-Dataset-Generator/lsb_source.py',
                     'pyproject.toml', 'uv.lock', 'Containerfile.rocm', 'Dockerfile.cpu'):
            with self.subTest(path=path):
                self.assertIsNotNone(policy.path_reasons(path)[1])

    def test_new_configs_data_scripts_and_tests_do_not_require_golden(self):
        for path in ('configs/new-experiment.yaml', 'data/new-test/pairs.csv',
                     'scripts/test_cpu.py', 'scripts/required_test_level.py',
                     'tests/test_cpu_levels.py', 'docs/runbook.md'):
            with self.subTest(path=path):
                self.assertEqual(policy.path_reasons(path), (None, None))

    def test_unknown_area_fails_closed_without_prefix_lookalikes(self):
        for path in ('mystery/file.py', 'cvdl-other/file.py', 'README.md.backup',
                     '.github/workflows/check.yml', 'train_template.py'):
            self.assertTrue(all(policy.path_reasons(path)), path)

    def test_dependencies_require_both_levels(self):
        for path in ('pyproject.toml', 'uv.lock', 'Containerfile', 'Dockerfile.dev'):
            self.assertTrue(all(policy.path_reasons(path)), path)

    def test_rename_parser_preserves_both_paths_and_whitespace(self):
        self.assertEqual(policy.parse_name_status(
            b'R100\0scripts/old name.py\0LSB-Dataset-Generator/new\nname.py\0M\0docs/a.md\0'),
            {'scripts/old name.py', 'LSB-Dataset-Generator/new\nname.py', 'docs/a.md'})
        with self.assertRaises(ValueError):
            policy.parse_name_status(b'R100\0old\0')

    def test_decisions_are_independent_and_reminder_is_always_present(self):
        lines = policy.decisions(['cvdl/models.py'])
        self.assertTrue(lines[0].startswith('TESTS: FAST'))
        self.assertTrue(lines[1].startswith('GOLDEN: REQUIRED'))
        self.assertIn('cvdl/models.py', lines[1])
        self.assertEqual(lines[-1], policy.REMINDER)
        self.assertIn('GOLDEN: NOT REQUIRED', policy.decisions([])[1])


class GitChangeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.git('init', '-q')
        self.git('config', 'user.name', 'CPU policy tests')
        self.git('config', 'user.email', 'cpu-tests@example.invalid')
        self.put('scripts/original.py', 'unchanged fixture\n' * 10)
        self.put('docs/base.md', 'base\n')
        self.put('.gitignore', 'ignored/\n')
        self.git('add', '.')
        self.git('-c', 'core.hooksPath=/dev/null', 'commit', '-qm', 'base')
        self.base = self.git('rev-parse', 'HEAD').strip()

    def git(self, *args):
        return subprocess.check_output(['git', *args], cwd=self.root,
                                       stderr=subprocess.PIPE, text=True)

    def put(self, name, text):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)

    def test_committed_staged_unstaged_untracked_and_ignored(self):
        self.put('docs/base.md', 'committed\n')
        self.git('add', 'docs/base.md')
        self.git('-c', 'core.hooksPath=/dev/null', 'commit', '-qm', 'change')
        self.put('scripts/staged.py', 'staged\n')
        self.git('add', 'scripts/staged.py')
        self.put('scripts/original.py', 'unstaged\n')
        self.put('tests/untracked name.py', 'untracked\n')
        self.put('ignored/noise.py', 'ignored\n')
        before = self.git('status', '--porcelain=v1', '-z')
        self.assertEqual(set(policy.changed_paths(self.root, self.base)), {
            'docs/base.md', 'scripts/staged.py', 'scripts/original.py', 'tests/untracked name.py'})
        self.assertEqual(self.git('status', '--porcelain=v1', '-z'), before)

    def test_merge_base_includes_both_branch_histories_correctly(self):
        self.put('docs/head.md', 'head\n')
        self.git('add', '.')
        self.git('-c', 'core.hooksPath=/dev/null', 'commit', '-qm', 'head')
        head = self.git('rev-parse', 'HEAD').strip()
        self.git('checkout', '-qb', 'other', self.base)
        self.put('cvdl/base-only.py', 'other branch\n')
        self.git('add', '.')
        self.git('-c', 'core.hooksPath=/dev/null', 'commit', '-qm', 'other')
        other = self.git('rev-parse', 'HEAD').strip()
        self.git('checkout', '-q', head)
        self.assertEqual(policy.changed_paths(self.root, other), ['docs/head.md'])

    def test_staged_and_committed_renames_retain_both_sides(self):
        (self.root / 'LSB-Dataset-Generator').mkdir()
        self.git('mv', 'scripts/original.py', 'LSB-Dataset-Generator/moved.py')
        expected = ['LSB-Dataset-Generator/moved.py', 'scripts/original.py']
        self.assertEqual(policy.changed_paths(self.root, self.base), expected)
        self.git('-c', 'core.hooksPath=/dev/null', 'commit', '-qm', 'rename')
        self.assertEqual(policy.changed_paths(self.root, self.base), expected)
        self.assertTrue(policy.decisions(expected)[0].startswith('TESTS: FULL'))

    def test_unstaged_rename_keeps_deleted_and_untracked_paths(self):
        (self.root / 'tests').mkdir()
        (self.root / 'scripts/original.py').rename(self.root / 'tests/test_lsb_source.py')
        self.assertEqual(policy.changed_paths(self.root, self.base),
                         ['scripts/original.py', 'tests/test_lsb_source.py'])

    def test_missing_base_prints_both_required_and_reminder(self):
        with patch.object(policy, 'ROOT', self.root), redirect_stdout(io.StringIO()) as output:
            self.assertEqual(policy.main(['--base', 'missing-base']), 0)
        lines = output.getvalue().splitlines()
        self.assertEqual(len(lines), 3)
        self.assertTrue(lines[0].startswith('TESTS: FULL'))
        self.assertTrue(lines[1].startswith('GOLDEN: REQUIRED'))
        self.assertIn('fail closed', lines[1])
        self.assertEqual(lines[-1], policy.REMINDER)


class RunnerSelectionTests(unittest.TestCase):
    def case(self, module):
        return type('Example', (unittest.TestCase,), {
            '__module__': module,
            'setUpClass': classmethod(lambda cls: (_ for _ in ()).throw(AssertionError('fixture ran'))),
            'test_example': lambda self: None,
        })

    def test_fast_skips_only_exact_module_with_visible_reason_and_no_fixture(self):
        full = self.case('test_lsb_torch')
        small = type('Small', (unittest.TestCase,), {
            '__module__': 'test_lsb_torch_other', 'test_example': lambda self: None})
        suite = unittest.TestSuite([unittest.TestSuite([full('test_example')]), small('test_example')])
        with runner.select_level(suite, 'fast'):
            result = unittest.TextTestRunner(stream=io.StringIO(), verbosity=2).run(suite)
        self.assertTrue(result.wasSuccessful())
        self.assertEqual(result.testsRun, 2)
        self.assertEqual(len(result.skipped), 1)
        self.assertEqual(result.skipped[0][1], runner.SKIP_REASON)
        self.assertNotIn('__unittest_skip__', full.__dict__)

    def test_full_is_default_and_does_not_add_skips(self):
        with patch.object(runner.unittest.defaultTestLoader, 'discover', return_value=unittest.TestSuite()), \
             patch.object(runner.unittest, 'TextTestRunner') as text_runner, \
             patch.object(runner.os, 'chdir'), patch.dict(os.environ, {}, clear=True), \
             patch.object(sys, 'path', sys.path.copy()), patch.object(sys, 'dont_write_bytecode', False), \
             patch.object(runner, 'select_level', wraps=runner.select_level) as select:
            text_runner.return_value.run.return_value.wasSuccessful.return_value = True
            self.assertEqual(runner.main([]), 0)
            self.assertEqual(select.call_args.args[1], 'full')
            for key, value in runner.CPU_ENV.items():
                self.assertEqual(os.environ[key], value)
            self.assertTrue(sys.dont_write_bytecode)

    def test_existing_skip_attributes_restore_even_after_error(self):
        cls = self.case('test_lsb_torch')
        cls.__unittest_skip__ = True
        cls.__unittest_skip_why__ = 'original'
        suite = unittest.TestSuite([cls('test_example')])
        with self.assertRaises(RuntimeError):
            with runner.select_level(suite, 'fast'):
                self.assertEqual(cls.__unittest_skip_why__, runner.SKIP_REASON)
                raise RuntimeError('abort')
        self.assertTrue(cls.__unittest_skip__)
        self.assertEqual(cls.__unittest_skip_why__, 'original')

    def test_invalid_level_is_rejected(self):
        with self.assertRaises(ValueError):
            with runner.select_level(unittest.TestSuite(), 'invalid'):
                pass


if __name__ == '__main__':
    unittest.main()
