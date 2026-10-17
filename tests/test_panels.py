"""Main-stage validation outputs and per-source cutoff tables (cvdl/panels.py)."""
import json
import tempfile
import unittest
from pathlib import Path
import numpy as np
import pandas as pd
from cvdl import panels
from cvdl.metrics import rates, threshold


def predictions(source, n, seed, k=1, separation=2.0):
    rng = np.random.default_rng(seed)
    clean = 1/(1+np.exp(-rng.normal(-separation, 1.5, n)))
    stego = 1/(1+np.exp(-rng.normal(separation, 1.5, n)))
    rows = []
    for i in range(n):
        for label, score in ((0, clean[i]), (1, stego[i])):
            rows.append(dict(cover_id=f'{source}{i}', source=source, k=k if i % 2 else 2, label=label, prob_stego=float(score)))
    return pd.DataFrame(rows)


class PanelTests(unittest.TestCase):
    def folder(self, root):
        root = Path(root)
        standard = pd.concat([predictions('raise', 40, 1), predictions('div2k', 10, 2)], ignore_index=True)
        files = {'.': standard, 'extra-test-bpp1': standard, 'extra-alaska-val': predictions('alaska', 20, 3),
                 'extra-alaska-test_seen': predictions('alaska', 15, 4), 'extra-alaska-test_unseen': predictions('alaska', 15, 5),
                 'extra-lsdir-val': predictions('lsdir', 25, 6), 'extra-lsdir-test': predictions('lsdir', 30, 7),
                 'extra-standard-val': standard}
        for directory, frame in files.items():
            (root/directory).mkdir(parents=True, exist_ok=True)
            frame.to_csv(root/directory/'test_predictions.csv', index=False)
        capped = pd.concat([predictions('raise', 30, 8), predictions('alaska', 10, 9)], ignore_index=True)
        panels.write_validation(root, capped, root=Path(__file__).resolve().parents[1])
        return root, standard

    def test_validation_outputs(self):
        with tempfile.TemporaryDirectory() as directory:
            root, _ = self.folder(directory)
            saved = json.loads((root/'validation_calibration.json').read_text())
            capped = pd.read_csv(root/'validation_predictions.csv')
            cutoff, fa = threshold(capped, 'pe')
            # The file holds the cutoff of the in-memory frame; a CSV re-read can differ in the last digit.
            self.assertAlmostEqual(saved['cutoff'], cutoff, places=12)
            self.assertEqual((saved['validation_fa'], saved['pairs']), (fa, 40))
            self.assertEqual(saved['criterion'], 'minimum P_E; conservative tie')
            self.assertEqual(set(saved), {'criterion', 'cutoff', 'validation_fa', 'pairs', 'code_commit'})

    def test_tables_use_validation_only_cutoffs_and_are_reproducible(self):
        with tempfile.TemporaryDirectory() as directory, tempfile.TemporaryDirectory() as out:
            root, standard = self.folder(directory)
            with self.assertRaisesRegex(ValueError, 'unexpected standard validation pair counts'):
                panels.tables(root, out, 'm')
            # check=False allows the small synthetic counts.
            import unittest.mock as mock
            with mock.patch.object(panels, 'select_best_checkpoint', create=True), \
                 mock.patch('cvdl.lit.select_best_checkpoint', return_value=str(root/'validation_predictions.csv')):
                first = panels.tables(root, out, 'm', check=False)
                second = panels.tables(root, out, 'm', check=False)
            self.assertEqual(first, second)
            self.assertEqual(first['validation_pairs'], {'raise': 40, 'div2k': 10, 'alaska': 20, 'lsdir': 25})
            by_panel = {row['panel']: row for row in first['source_fa5']}
            self.assertEqual(set(by_panel), set(panels.TEST_PANELS))
            for panel, (source, where) in panels.TEST_PANELS.items():
                path = panels.panel_files(root)[where]
                test = pd.read_csv(path).query('source==@source')
                val_path = panels.panel_files(root)['standard_val' if source in ('raise', 'div2k') else f'{source}_val']
                val = pd.read_csv(val_path).query('source==@source')
                cutoff, valfa = threshold(val, 'fa5')
                expected = rates(test, cutoff)
                self.assertEqual(by_panel[panel]['cutoff'], cutoff)
                self.assertEqual(by_panel[panel]['validation_fa'], valfa)
                self.assertEqual({k: by_panel[panel][k] for k in expected}, expected)
                self.assertLessEqual(valfa, .05 + 1e-12)
            capped = json.loads((root/'validation_calibration.json').read_text())
            self.assertEqual(first['panels']['standard']['at_capped_val_min_pe'], rates(standard, capped['cutoff']))
            table = pd.read_csv(Path(out)/'pc-comparison/m-thresholds.csv')
            self.assertEqual(set(table.calibration), {'source', 'shared', 'fixed'})
            self.assertEqual(set(table.panel), set(panels.TEST_PANELS))
            self.assertEqual(len(table), len(panels.TEST_PANELS)*5)

    def test_missing_files_are_named(self):
        with tempfile.TemporaryDirectory() as directory:
            root, _ = self.folder(directory)
            (root/'extra-lsdir-val/test_predictions.csv').unlink()
            self.assertIn('extra-lsdir-val', ' '.join(panels.missing_panels(root)))


class CliTests(unittest.TestCase):
    def test_check_b_cli_equals_build(self):
        import sys
        from unittest.mock import patch
        import scripts.check_b_thresholds as cb
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            files = {s: root/f'{s}.csv' for s in cb.SOURCES}
            for i, s in enumerate(cb.SOURCES):
                predictions(s, 30, i).to_csv(files[s], index=False)
            argv = ['x', '--model', 'm', '--output', str(root/'cli.csv')]
            for s, f in files.items():
                argv += ['--validation', f'{s}={f}', '--test', f'{s}t={s}:{f}']
            with patch.object(sys, 'argv', argv):
                cb.main()
            cb.build('m', {s: files[s] for s in cb.SOURCES}, {f'{s}t': (s, files[s]) for s in cb.SOURCES}, root/'api.csv', quiet=True)
            self.assertEqual((root/'cli.csv').read_text(), (root/'api.csv').read_text())

if __name__ == '__main__':
    unittest.main()
