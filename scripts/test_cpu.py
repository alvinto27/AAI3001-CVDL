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
FULL_ONLY_MODULES = frozenset({'test_lsb_torch'})
FULL_ONLY_CLASSES = frozenset({('test_data_figure', 'DataFigureReplayTests')})
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
            reason = (SKIP_REASON if cls.__module__ in FULL_ONLY_MODULES else
                      FIGURE_SKIP_REASON if (cls.__module__, cls.__name__) in FULL_ONLY_CLASSES else None)
            if reason is not None and cls not in saved:
                saved[cls] = {key: cls.__dict__[key] for key in
                              ('__unittest_skip__', '__unittest_skip_why__') if key in cls.__dict__}
                cls.__unittest_skip__ = True
                cls.__unittest_skip_why__ = reason
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
    parser.add_argument('--level', choices=('fast', 'full'), default='full')
    args = parser.parse_args(argv)
    # Set limits before discovery imports torch, NumPy or any test module.
    os.environ.update(CPU_ENV)
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(ROOT))
    os.chdir(ROOT)
    suite = unittest.defaultTestLoader.discover(str(ROOT / 'tests'))
    with select_level(suite, args.level):
        result = unittest.TextTestRunner(verbosity=2, durations=20).run(suite)
    return 0 if result.wasSuccessful() else 1


if __name__ == '__main__':
    raise SystemExit(main())
