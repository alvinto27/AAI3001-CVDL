"""The real-data rebuild is optional on checkouts without frozen datasets."""
import os
from pathlib import Path
import subprocess
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]


class DataFigureTests(unittest.TestCase):
    def test_saved_figure_is_small_png(self):
        data = (ROOT/'docs/figures/report/data-sample.png').read_bytes()
        self.assertTrue(data.startswith(b'\x89PNG\r\n\x1a\n'))
        self.assertLess(len(data), 500_000)

    @unittest.skipUnless((ROOT/'data/hard-test/256-v1/pairs-alaska-seen.csv').is_file(),
                         'real frozen datasets are not installed')
    def test_real_pairs_equal_evaluator_and_png_rebuild_is_identical(self):
        result = subprocess.run([sys.executable, '-B', 'scripts/build_data_figure.py', '--check'],
            cwd=ROOT, env={**os.environ, 'CUDA_VISIBLE_DEVICES':'', 'HIP_VISIBLE_DEVICES':'',
                           'OMP_NUM_THREADS':'1', 'OPENBLAS_NUM_THREADS':'1'},
            capture_output=True, text=True, timeout=120)
        self.assertEqual(result.returncode, 0, result.stdout+result.stderr)
        self.assertIn('byte-identical', result.stdout)
        self.assertIn('all seven pairs equal evaluator pixels', result.stdout)
