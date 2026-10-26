"""Fast checks for CPU-level selection and the read-only changed-path policy."""
from contextlib import nullcontext, redirect_stdout
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
        self.assertEqual(lines[1], policy.REMINDER)
        self.assertEqual(len(lines), 2)
        for path in paths:
            self.assertIn(repr(path), lines[0])

    def test_docs_plus_code_keeps_fast_full_rules(self):
        for path, tests in [
            ('scripts/test_cpu.py', 'FAST'), ('cvdl/models.py', 'FAST'),
            ('LSB-Dataset-Generator/lsb_torch.py', 'FULL'), ('unknown.txt', 'FULL'),
        ]:
            with self.subTest(path=path):
                lines = policy.decisions(['docs/guide.md', path])
                self.assertTrue(lines[0].startswith('TESTS: ' + tests))
                self.assertEqual(lines[1:], [policy.REMINDER])

    def test_code_read_docs_and_nontext_docs_keep_required_levels(self):
        for path in ('docs/active/plan-architecture-tests.md', *policy.DOCUMENTATION_INPUTS,
                     'docs/figures/other.png', 'docs/guide.txt',
                     'docs/code.py', 'docs/guide.MD', 'scripts/README.md'):
            with self.subTest(path=path):
                lines = policy.decisions(['docs/guide.md', path])
                level = 'FULL' if path == 'docs/figures/report/data-sample.png' else 'FAST'
                self.assertTrue(lines[0].startswith('TESTS: ' + level))
                self.assertEqual(lines[1:], [policy.REMINDER])

    def test_no_changes_need_no_tests(self):
        lines = policy.decisions([])
        self.assertTrue(lines[0].startswith('TESTS: NONE'))
        self.assertIn('(no changed paths)', lines[0])
        self.assertEqual(lines[1:], [policy.REMINDER])

    def test_full_loader_and_frozen_figure_dependencies(self):
        for path in ('LSB-Dataset-Generator/lsb_torch.py', 'tests/test_lsb_torch.py',
                     'tests/test_lsb_source.py', 'configs/raise512-run1.yaml',
                     'configs/test-bpp1.yaml', 'tests/test_data_figure.py',
                     'scripts/build_data_figure.py', 'docs/figures/report/data-sample.png',
                     'cvdl/standardtest.py', 'cvdl/validation.py', 'cvdl/__init__.py',
                     'data/raise/photo_manifest.csv', 'data/div2k/photo_manifest.csv',
                     'data/alaska/photo_manifest.csv', 'data/lsdir/photo_manifest.csv',
                     'data/test/standard-256-v1/pairs-standard.csv',
                     'data/hard-test/256-v1/summary.json'):
            with self.subTest(path=path):
                self.assertIsNotNone(policy.path_reasons(path))
                self.assertTrue(policy.decisions(['docs/guide.md', path])[0].startswith('TESTS: FULL'))

    def test_known_areas_are_fast(self):
        for path in ('cvdl/models.py', 'scripts/test_cpu.py', 'notebooks/new-experiment.ipynb',
                     'configs/new-experiment.yaml', 'data/new-test/pairs.csv',
                     'tests/test_cpu_levels.py', 'docs/runbook.md', 'README.md',
                     'AGENTS.md', 'CHANGELOG.md', 'scripts/required_test_level.py',
                     'cvdl/metrics.py', 'tests/test_cvdl_metrics.py',
                     'tests/test_notebook_compile.py', 'tests/test_notebook_diagnostics.py',
                     'notebooks/04_baseline_kv_xunet_v3.ipynb',
                     'docs/figures/report/data-sample.png.backup',
                     'data/test/standard-256-v1-other/pairs.csv',
                     'data/hard-test/256-v1-other/pairs.csv',
                     'data/raise/photo_manifest.csv.backup'):
            with self.subTest(path=path):
                self.assertIsNone(policy.path_reasons(path))

    def test_former_comparison_inputs_keep_independent_test_levels(self):
        for path in ('cvdl/models.py', 'scripts/run_notebook.py', 'scripts/speed_matrix.py',
                     'scripts/map_demo.py', 'scripts/host_monitor.py', 'scripts/pc_safety.py',
                     'scripts/g4/preflight.py', 'scripts/g4/jobs/archive/newdata-1.env',
                     'scripts/g4/jobs/archive/wide-1.env', 'scripts/g4/jobs/speed-1.env',
                     'tests/test_model_copy_parity.py', 'configs/train-k1to4.yaml',
                     'notebooks/05_xunet_v4_nonorm.ipynb',
                     'notebooks/04_baseline_kv_xunet_v3.ipynb',
                     'notebooks/archive/05_baseline_kv_xunet_v4_pre-cvdl.ipynb',
                     'notebooks/rocm_conv1_wgrad.py', 'scripts/archive/refactor_golden.py',
                     'tests/archive/golden/cpu-det.env',
                     'configs/comparison-constants-v1.json', 'tests/test_comparison_constants.py'):
            with self.subTest(path=path):
                self.assertTrue(policy.decisions([path])[0].startswith('TESTS: FAST'))
        for path in ('Datasets/originals/raise/NEF/a.NEF',
                     'runs/g4-v4-r2/checkpoints/model/best.ckpt'):
            self.assertIn('unknown area', policy.path_reasons(path))

    def test_retired_enforcement_has_no_attributes_or_output(self):
        self.assertFalse(hasattr(policy, 'GOLDEN_FILES'))
        self.assertFalse(hasattr(policy, 'GOLDEN_PATTERNS'))
        for paths in ([], ['docs/guide.md'], ['cvdl/models.py'], ['unknown.txt']):
            with self.subTest(paths=paths):
                lines = policy.decisions(paths)
                self.assertEqual(len(lines), 2)
                self.assertNotIn('GOLDEN:', '\n'.join(lines))

    def test_unknown_area_fails_closed_without_prefix_lookalikes(self):
        for path in ('mystery/file.py', 'cvdl-other/file.py', 'README.md.backup',
                     '.github/workflows/check.yml', 'train_template.py'):
            self.assertIsNotNone(policy.path_reasons(path), path)

    def test_dependencies_require_full(self):
        for path in ('pyproject.toml', 'uv.lock', 'Containerfile', 'Dockerfile.dev'):
            self.assertIsNotNone(policy.path_reasons(path), path)

    def test_rename_parser_preserves_both_paths_and_whitespace(self):
        self.assertEqual(policy.parse_name_status(
            b'R100\0scripts/old name.py\0LSB-Dataset-Generator/new\nname.py\0M\0docs/a.md\0'),
            {'scripts/old name.py', 'LSB-Dataset-Generator/new\nname.py', 'docs/a.md'})
        with self.assertRaises(ValueError):
            policy.parse_name_status(b'R100\0old\0')

    def test_decisions_show_inspected_paths_and_exact_reminder(self):
        lines = policy.decisions(['cvdl/models.py'])
        self.assertTrue(lines[0].startswith('TESTS: FAST'))
        self.assertIn('cvdl/models.py', lines[0])
        self.assertEqual(lines[1:], ['Before a G4 launch: always FULL tests.'])
        self.assertEqual(policy.decisions([])[1], policy.REMINDER)


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

    def test_missing_base_prints_full_and_reminder(self):
        with patch.object(policy, 'ROOT', self.root), redirect_stdout(io.StringIO()) as output:
            self.assertEqual(policy.main(['--base', 'missing-base']), 0)
        lines = output.getvalue().splitlines()
        self.assertEqual(len(lines), 2)
        self.assertTrue(lines[0].startswith('TESTS: FULL'))
        self.assertIn('fail closed', lines[0])
        self.assertNotIn('GOLDEN:', '\n'.join(lines))
        self.assertEqual(lines[-1], policy.REMINDER)


class RunnerSelectionTests(unittest.TestCase):
    def case(self, module, name='Example'):
        return type(name, (unittest.TestCase,), {
            '__module__': module,
            'setUpClass': classmethod(lambda cls: (_ for _ in ()).throw(AssertionError('fixture ran'))),
            'test_example': lambda self: None,
        })

    def test_fast_skips_only_exact_module_or_class_with_visible_reason_and_no_fixture(self):
        full = self.case('test_lsb_torch')
        replay = self.case('test_data_figure', 'DataFigureReplayTests')
        small = type('Small', (unittest.TestCase,), {
            '__module__': 'test_lsb_torch_other', 'test_example': lambda self: None})
        smoke = type('DataFigureTests', (unittest.TestCase,), {
            '__module__': 'test_data_figure', 'test_example': lambda self: None})
        lookalike = type('DataFigureReplayTests', (unittest.TestCase,), {
            '__module__': 'test_data_figure_other', 'test_example': lambda self: None})
        suite = unittest.TestSuite([unittest.TestSuite([full('test_example'), replay('test_example')]),
                                   small('test_example'), smoke('test_example'), lookalike('test_example')])
        with runner.select_level(suite, 'fast'):
            result = unittest.TextTestRunner(stream=io.StringIO(), verbosity=2).run(suite)
        self.assertTrue(result.wasSuccessful())
        self.assertEqual(result.testsRun, 5)
        self.assertEqual([reason for _, reason in result.skipped],
                         [runner.SKIP_REASON, runner.FIGURE_SKIP_REASON])
        for cls in (full, replay, small, smoke, lookalike):
            self.assertNotIn('__unittest_skip__', cls.__dict__)
            self.assertNotIn('__unittest_skip_why__', cls.__dict__)
        with runner.select_level(unittest.TestSuite([full('test_example'), replay('test_example')]), 'full'):
            self.assertFalse(getattr(full, '__unittest_skip__', False))
            self.assertFalse(getattr(replay, '__unittest_skip__', False))

    def test_full_is_default_and_does_not_add_skips(self):
        with patch.object(runner, 'suite_lock', return_value=nullcontext()), \
             patch.object(runner, 'check_fixtures'), \
             patch.object(runner.unittest.defaultTestLoader, 'discover', return_value=unittest.TestSuite()), \
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
        for module, name, reason in (('test_lsb_torch', 'Example', runner.SKIP_REASON),
                                     ('test_data_figure', 'DataFigureReplayTests', runner.FIGURE_SKIP_REASON)):
            with self.subTest(module=module, name=name):
                cls = self.case(module, name)
                cls.__unittest_skip__ = True
                cls.__unittest_skip_why__ = 'original'
                suite = unittest.TestSuite([cls('test_example')])
                with self.assertRaises(RuntimeError):
                    with runner.select_level(suite, 'fast'):
                        self.assertEqual(cls.__unittest_skip_why__, reason)
                        raise RuntimeError('abort')
                self.assertTrue(cls.__unittest_skip__)
                self.assertEqual(cls.__unittest_skip_why__, 'original')
                with runner.select_level(suite, 'full'):
                    self.assertEqual(cls.__unittest_skip_why__, 'original')

    def test_invalid_level_is_rejected(self):
        with self.assertRaises(ValueError):
            with runner.select_level(unittest.TestSuite(), 'invalid'):
                pass


if __name__ == '__main__':
    unittest.main()
