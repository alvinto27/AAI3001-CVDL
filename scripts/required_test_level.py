#!/usr/bin/env python3
"""Report required checks from local Git changes. This command does not fetch or write."""
from __future__ import annotations

import argparse
from fnmatch import fnmatchcase
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
KNOWN_AREAS = ('cvdl/', 'scripts/', 'notebooks/', 'configs/', 'data/', 'tests/', 'docs/',
               'LSB-Dataset-Generator/')
KNOWN_FILES = frozenset({'README.md', 'AGENTS.md', 'CHANGELOG.md', 'pyproject.toml', 'uv.lock'})
FULL_FILES = frozenset({'tests/test_lsb_torch.py', 'tests/test_lsb_source.py',
                        'configs/raise512-run1.yaml', 'configs/test-bpp1.yaml'})
FIGURE_FULL_FILES = frozenset({
    'tests/test_data_figure.py', 'scripts/build_data_figure.py',
    'docs/figures/report/data-sample.png', 'cvdl/standardtest.py',
    'cvdl/validation.py', 'cvdl/__init__.py',
    'data/raise/photo_manifest.csv', 'data/div2k/photo_manifest.csv',
    'data/alaska/photo_manifest.csv', 'data/lsdir/photo_manifest.csv',
})
FIGURE_FULL_PREFIXES = ('data/test/standard-256-v1/', 'data/hard-test/256-v1/')
DEPENDENCY_FILES = frozenset({'pyproject.toml', 'uv.lock'})
# Keep code-read documentation in FAST, even when the file is Markdown.
# architecture_decision hashes its plan; reporting/figure checks read the PNG.
DOCUMENTATION_INPUTS = frozenset({
    'docs/active/plan-architecture-tests.md',
    'docs/figures/report/data-sample.png',
})
ROOT_MARKDOWN = frozenset({'README.md', 'AGENTS.md', 'CHANGELOG.md'})
# Explicit inputs to the current golden recipes, not all experiment configs/data.
GOLDEN_FILES = frozenset({
    'scripts/refactor_golden.py', 'scripts/run_notebook.py', 'scripts/speed_matrix.py',
    'scripts/map_demo.py', 'scripts/host_monitor.py', 'scripts/pc_safety.py',
    'scripts/g4/preflight.py',
    'scripts/g4/jobs/archive/newdata-1.env', 'scripts/g4/jobs/archive/wide-1.env',
    'scripts/g4/jobs/speed-1.env', 'tests/test_model_copy_parity.py',
    'notebooks/05_xunet_v4_nonorm.ipynb', 'notebooks/04_baseline_kv_xunet_v3.ipynb',
    'notebooks/archive/05_baseline_kv_xunet_v4_pre-cvdl.ipynb',
    'notebooks/rocm_conv1_wgrad.py',
    'configs/raise512-run1.yaml', 'configs/train-k1to4.yaml', 'configs/test-bpp1.yaml',
    'data/raise/photo_manifest.csv', 'data/div2k/photo_manifest.csv',
    'data/alaska/photo_manifest.csv', 'data/lsdir/photo_manifest.csv',
})
# Real pixels and model weights are golden inputs too. Other paths in these
# ignored output/source areas are unknown, so they fail closed below.
GOLDEN_PATTERNS = (
    'Datasets/originals/raise/NEF/*.NEF',
    'Datasets/originals/div2k/DIV2K_train_HR/*.png',
    'Datasets/originals/alaska/tiff_various_color/*.tif',
    'Datasets/originals/lsdir/hr/*/*.png',
    'runs/xunet*wide-w4_r2-*/checkpoints/*/best*.ckpt',
    'runs/xunet*wide-w4_mixr2-*/checkpoints/*/best*.ckpt',
    'runs/g4-v4-r2/checkpoints/*/best*.ckpt',
    'runs/g4-prime-1/checkpoints/*/best*.ckpt',
)
REMINDER = 'Before a G4 launch: always FULL tests and the golden compare.'


def dependency_path(path):
    return path in DEPENDENCY_FILES or path.startswith(('Containerfile', 'Dockerfile'))


def path_reasons(path):
    """Return independent full-test and golden reasons (None means not required)."""
    known = (path in KNOWN_FILES or path.startswith(KNOWN_AREAS) or dependency_path(path))
    if not known:
        return 'unknown area (fail closed)', 'unknown area (fail closed)'
    full = None
    if dependency_path(path):
        full = 'runtime dependencies or container build'
    elif path.startswith('LSB-Dataset-Generator/'):
        full = 'exhaustive loader dependency'
    elif path in FULL_FILES:
        full = 'exhaustive loader test or fixture dependency'
    elif path in FIGURE_FULL_FILES or path.startswith(FIGURE_FULL_PREFIXES):
        full = 'frozen figure replay dependency'
    golden = None
    if dependency_path(path):
        golden = 'runtime dependencies or container build'
    elif path.startswith(('cvdl/', 'LSB-Dataset-Generator/')):
        golden = 'shared golden implementation'
    elif (path in GOLDEN_FILES or path.startswith('tests/golden/')
          or any(fnmatchcase(path, pattern) for pattern in GOLDEN_PATTERNS)):
        golden = 'golden script input or executable dependency'
    return full, golden


def parse_name_status(data):
    """Read Git's NUL-separated name/status format; retain both rename paths."""
    fields = data.split(b'\0')
    if fields[-1] == b'':
        fields.pop()
    paths = set()
    index = 0
    while index < len(fields):
        status = fields[index].decode('ascii')
        index += 1
        if not status or status[0] not in 'ACDMRTUXB':
            raise ValueError('invalid Git change status')
        count = 2 if status[0] in 'RC' else 1
        if index + count > len(fields):
            raise ValueError('incomplete Git name/status output')
        for value in fields[index:index + count]:
            paths.add(value.decode('utf-8', errors='surrogateescape'))
        index += count
    return paths


def git(repo, *args):
    return subprocess.check_output(['git', *args], cwd=repo, stderr=subprocess.PIPE)


def changed_paths(repo, base):
    merge_base = git(repo, 'merge-base', base, 'HEAD').decode('ascii').strip()
    if not merge_base:
        raise ValueError('no merge base')
    paths = set()
    for revisions in ((merge_base, 'HEAD'), ('--cached',), ()):
        paths.update(parse_name_status(git(repo, 'diff', '--no-ext-diff', '--no-textconv',
                                          '--name-status', '-z', '--find-renames', *revisions, '--')))
    paths.update(value.decode('utf-8', errors='surrogateescape') for value in
                 git(repo, 'ls-files', '--others', '--exclude-standard', '-z').split(b'\0') if value)
    return sorted(paths)


def documentation_text(path):
    return (path not in DOCUMENTATION_INPUTS
            and (path in ROOT_MARKDOWN or (path.startswith('docs/') and path.endswith('.md'))))


def decisions(paths):
    paths = sorted(set(paths))
    full, golden = [], []
    for path in paths:
        test_reason, golden_reason = path_reasons(path)
        if test_reason:
            full.append(f'{path!r}: {test_reason}')
        if golden_reason:
            golden.append(f'{path!r}: {golden_reason}')
    # Include the inspected paths even when none matched a required-check rule.
    inspected = ', '.join(repr(path) for path in paths) or '(no changed paths)'
    if all(documentation_text(path) for path in paths):
        return ['TESTS: NONE — documentation text only; inspected: ' + inspected,
                'GOLDEN: NOT REQUIRED — no golden dependency changed; inspected: ' + inspected,
                REMINDER]
    return [
        'TESTS: FULL — ' + '; '.join(full) if full else
        'TESTS: FAST — no full-only dependency changed; inspected: ' + inspected,
        'GOLDEN: REQUIRED — ' + '; '.join(golden) if golden else
        'GOLDEN: NOT REQUIRED — no golden dependency changed; inspected: ' + inspected,
        REMINDER,
    ]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', default='origin/yx')
    args = parser.parse_args(argv)
    try:
        lines = decisions(changed_paths(ROOT, args.base))
    except (OSError, subprocess.CalledProcessError, ValueError, UnicodeError):
        reason = f'cannot inspect changes against {args.base!r} (fail closed)'
        lines = ['TESTS: FULL — ' + reason, 'GOLDEN: REQUIRED — ' + reason, REMINDER]
    print('\n'.join(lines))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
