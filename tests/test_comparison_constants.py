"""Approved input constants, not captured behavior or a training equivalence check.

Change an approved value and this list in the same commit. Large frozen pair
CSVs are verified by their dedicated tests/runtime loaders; here we check only
their published identities in small metadata files. No pixels or GPU work.
"""
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import unittest

from cvdl import deployment, hardtest, hardtest256, standardtest, validation
from cvdl.settings import SCHEMA, load, machine_profile, plain
import lsb_settings
from lsb_batch import read_config
from scripts.g4.preflight import top_values

ROOT = Path(__file__).resolve().parents[1]
# Resolve shared imports during discovery, not by stubbing real Settings.load().
from lsb_torch import RunSettings  # noqa: F401


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f'duplicate constants key: {key}')
        result[key] = value
    return result


class ComparisonConstantsTests(unittest.TestCase):
    def test_approved_constants(self):
        contract = json.loads((ROOT/'configs/comparison-constants-v1.json').read_text(),
                              object_pairs_hook=unique_object)
        self.assertEqual(set(contract), {'version', 'resolved_settings', 'declared_values', 'frozen_inputs'})
        self.assertEqual(contract['version'], 1)
        self.assertTrue(contract['resolved_settings'])
        for name, case in contract['resolved_settings'].items():
            with self.subTest(case=name):
                self.assertEqual(set(case), {'profile', 'notebook_values', 'notebook', 'expected', 'run_settings'})
                self.assertTrue(case['expected'])
                values = case['notebook_values']
                if case['notebook'] is not None:
                    self.assertEqual(values, {})
                    values = top_values(json.loads((ROOT/case['notebook']).read_text()))
                settings = load(environ={}, _profile=machine_profile(case['profile'], device='cpu',
                    rocm=False, precision_limits=('32',)), **values)
                # Default-only opt-ins can be deliberately omitted from legacy records.
                for key, expected in case['expected'].items():
                    with self.subTest(field=key):
                        self.assertIn(key, SCHEMA)
                        self.assertFalse(SCHEMA[key].machine)
                        actual = settings.values[key] if key in settings.values else SCHEMA[key].default
                        self.assertFalse(callable(actual))
                        self.assertEqual(plain(actual), expected)
                run = asdict(settings.to_run_settings())
                for key, expected in case['run_settings'].items():
                    self.assertEqual(plain(run[key]), expected, key)

        declared = {
            'evaluation_seed': lsb_settings.FIXED_EVALUATION_SEED,
            'otf_key_version': lsb_settings.OTF_KEY_VERSION,
            'purpose_codes': dict(lsb_settings.PURPOSE_CODES),
            'grid_codes': dict(lsb_settings.GRID_CODES),
            'carrier_codes': dict(lsb_settings.CARRIER_CODES),
            'stream_codes': dict(lsb_settings.STREAM_CODES),
            'standard_version': standardtest.VERSION,
            'standard_suite': standardtest.SUITE,
            'capped_validation_seeds': None,
            'train_matched_version': validation.VERSION,
            'train_matched_rounding': validation.ROUNDING,
            'deployment_policy': deployment.POLICY,
            'calibration_mapping_sha256': deployment.MAPPING_SHA,
            'hard_version': hardtest256.VERSION,
            'hard_ks': list(hardtest.KS),
            'hard_conditions': list(hardtest.CONDITIONS),
            'hard_combinations': [list(c) for c in hardtest.COMBINATIONS],
            'hard_pairs_per_cell': hardtest.PAIRS_PER_CELL,
            'hard_per_combination': hardtest.PER_COMBINATION,
        }
        # These seeds are literals inside a small helper, not public module values.
        # Read the declaration rather than generate/capture a sampled subset.
        import ast
        tree = ast.parse((ROOT/'LSB-Dataset-Generator/lsb_notebook.py').read_text())
        cap = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name=='capped_validation')
        groups = next(n.iter for n in cap.body if isinstance(n, ast.For))
        declared['capped_validation_seeds'] = {
            ast.literal_eval(row.elts[0]): ast.literal_eval(row.elts[2])
            for row in groups.elts if ast.literal_eval(row.elts[0]) != 'alaska'
        }
        self.assertEqual(set(contract['declared_values']), set(declared))
        for key, expected in contract['declared_values'].items():
            with self.subTest(declaration=key):
                self.assertEqual(declared[key], expected)

        frozen = contract['frozen_inputs']
        self.assertEqual(set(frozen), {'file_sha256', 'json_fields', 'embedding_config'})
        for name, expected in frozen['file_sha256'].items():
            with self.subTest(input_file=name):
                self.assertEqual(hashlib.sha256((ROOT/name).read_bytes()).hexdigest(), expected)
        for name, fields in frozen['json_fields'].items():
            with self.subTest(metadata=name):
                self.assertTrue(fields)
                actual = json.loads((ROOT/name).read_text())
                for key, expected in fields.items():
                    self.assertEqual(actual[key], expected, key)
        config = read_config(ROOT/'configs/test-256-v1.yaml').as_dict()
        self.assertTrue(frozen['embedding_config'])
        for key, expected in frozen['embedding_config'].items():
            with self.subTest(embedding_field=key):
                self.assertEqual(plain(config[key]), expected)


if __name__ == '__main__':
    unittest.main()
