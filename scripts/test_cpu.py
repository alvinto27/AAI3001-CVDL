#!/usr/bin/env python3
"""Run CPU checks. Full is the default; fast skips loader and figure replays."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import os
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.test_support import check_fixtures, suite_lock, full_evidence_key, record_full_pass

FULL_ONLY_MODULES = frozenset({'test_lsb_torch'})
FULL_ONLY_CLASSES = frozenset({('test_data_figure', 'DataFigureReplayTests')})
FULL_ONLY_METHODS = frozenset({
    ('test_paired_training', 'PairedTrainingTests', 'test_unpaired_loaders_historical_bytes_order_rng_workers_packing_dtypes'),
    ('test_curriculum50', 'Curriculum50Tests', 'test_first_seven_epochs_normal_pixels_labels_order_and_rng'),
    ('test_curriculum50', 'Curriculum50Tests', 'test_legacy_run10_11_exact_tensors_draws_records_and_identities'),
    ('test_wiser_b', 'ContextTests', 'test_frozen_full_panel_replay_workers_zero_four'),
})
REPLAY_SKIP_REASON = 'full level only: historical multiworker replay'
SKIP_REASON = 'full level only: exhaustive loader order, worker-process and resume checks'
FIGURE_SKIP_REASON = 'full level only: frozen real-pair and byte-identical figure replay'
CPU_ENV = {
    'CUDA_VISIBLE_DEVICES': '', 'HIP_VISIBLE_DEVICES': '',
    'PYTHONDONTWRITEBYTECODE': '1', 'OMP_NUM_THREADS': '1',
    'OPENBLAS_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1',
}


def iter_tests(suite):
    for item in suite:
        if isinstance(item, unittest.TestSuite):
            yield from iter_tests(item)
        else:
            yield item


@contextmanager
def select_level(suite, level):
    """Use unittest's normal skip handling, including skipping class fixtures."""
    if level not in {'fast', 'full'}:
        raise ValueError(f'unknown test level: {level}')
    saved = {}
    if level == 'fast':
        for test in iter_tests(suite):
            cls = type(test)
            module = cls.__module__.removeprefix('tests.')
            reason = (SKIP_REASON if module in FULL_ONLY_MODULES else
                      FIGURE_SKIP_REASON if (module, cls.__name__) in FULL_ONLY_CLASSES else None)
            target = cls
            if reason is None and (module, cls.__name__, test._testMethodName) in FULL_ONLY_METHODS:
                target = getattr(cls, test._testMethodName)
                reason = REPLAY_SKIP_REASON
            if reason is not None and target not in saved:
                saved[target] = {key: target.__dict__[key] for key in
                                 ('__unittest_skip__', '__unittest_skip_why__') if key in target.__dict__}
                target.__unittest_skip__ = True
                target.__unittest_skip_why__ = reason
    try:
        yield
    finally:
        for cls, attributes in saved.items():
            for key in ('__unittest_skip__', '__unittest_skip_why__'):
                if key in attributes:
                    setattr(cls, key, attributes[key])
                else:
                    delattr(cls, key)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument('--level', choices=('fast', 'full'), default='full')
    selection.add_argument('--tests', nargs='+', metavar='TEST',
                           help='focused dotted module/class/method names; no level skips')
    parser.add_argument('--lock-timeout', type=float, default=120., metavar='SECONDS',
                        help='bounded suite-lock wait (default: 120 seconds)')
    parser.add_argument('--no-wait', action='store_true', help='refuse immediately if a suite holds the lock')
    args = parser.parse_args(argv)
    # Set limits before discovery imports torch, NumPy or any test module.
    os.environ.update(CPU_ENV)
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(ROOT))
    os.chdir(ROOT)
    try:
        with suite_lock(ROOT, args.tests or args.level,
                        timeout=0. if args.no_wait else args.lock_timeout):
            before = None
            if args.tests:
                suite = unittest.defaultTestLoader.loadTestsFromNames(args.tests)
            else:
                check_fixtures(ROOT)
                if args.level == 'full':
                    before = full_evidence_key(ROOT)
                suite = unittest.defaultTestLoader.discover(str(ROOT / 'tests'))
            with select_level(suite, 'full' if args.tests else args.level):
                result = unittest.TextTestRunner(verbosity=2, durations=20).run(suite)
            if not args.tests and args.level == 'full' and record_full_pass(ROOT, before, result):
                print('FULL evidence saved for clean, unchanged tree/fixtures/runtime.')
        return 0 if result.wasSuccessful() else 1
    except (OSError, ValueError) as error:
        print(f'CPU suite refused: {error}', file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
