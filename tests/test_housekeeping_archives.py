"""Keep retired command paths out of active imports and preserve archive roots."""
import ast
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import required_test_level as policy

ARCHIVED = (
    'evaluate_gapcov_context.py',
    'evaluate_learn_srm_context.py',
    'evaluate_rgb_candidate_context.py',
    'g4/step_profile.py',
    'loader_speed_matrix.py',
    'measure_hard_noise_256.py',
    'build_new_source_manifests.py',
)


class HousekeepingArchiveTests(unittest.TestCase):
    def test_archived_commands_have_correct_roots_and_no_old_entry_points(self):
        for name in ARCHIVED:
            with self.subTest(name=name):
                self.assertFalse((ROOT / 'scripts' / name).exists())
                path = ROOT / 'scripts/archive' / name
                tree = ast.parse(path.read_text())
                assignment = next(node for node in tree.body
                                  if isinstance(node, ast.Assign)
                                  and any(isinstance(t, ast.Name) and t.id == 'ROOT'
                                          for t in node.targets))
                value = assignment.value
                self.assertIsInstance(value, ast.Subscript)
                self.assertEqual(value.value.attr, 'parents')
                self.assertEqual(path.resolve().parents[value.slice.value], ROOT)

    def test_no_active_python_imports_retired_commands(self):
        # Do not traverse runs, datasets, or other worktrees. Archive-only tests
        # may import archived scripts; active readers and loader tests may not.
        files = list(ROOT.glob('*.py'))
        for area in ('cvdl', 'scripts', 'tests', 'notebooks', 'LSB-Dataset-Generator'):
            files.extend((ROOT / area).rglob('*.py'))
        stems = {Path(name).stem for name in ARCHIVED}
        for path in files:
            if 'archive' in path.relative_to(ROOT).parts:
                continue
            for node in ast.walk(ast.parse(path.read_text())):
                if isinstance(node, ast.Import):
                    names = {a.name.rsplit('.', 1)[-1] for a in node.names}
                elif isinstance(node, ast.ImportFrom):
                    names = {(node.module or '').rsplit('.', 1)[-1]}
                    names.update(a.name for a in node.names)
                else:
                    continue
                with self.subTest(path=str(path.relative_to(ROOT)), line=node.lineno):
                    self.assertFalse(stems & names, stems & names)

    def test_retired_paths_are_not_full_replay_dependencies(self):
        for name in ARCHIVED:
            for path in ('scripts/' + name, 'scripts/archive/' + name):
                with self.subTest(path=path):
                    self.assertNotIn(path, policy.REPLAY_FULL_FILES)
                    self.assertIsNone(policy.path_reasons(path))


if __name__ == '__main__':
    unittest.main()
