"""Real production fake/meta runs plus tiny refusal boundaries; no GPU/data allocation."""
import json
from pathlib import Path
import unittest
from unittest.mock import patch

import torch
from torch._subclasses.fake_tensor import FakeTensorMode
from cvdl.settings import load, machine_profile, plain
from scripts.g4.preflight import top_values, preflight, resume_identity
from scripts.g4.tensor_limits import LIMIT, OutputRecorder, check_tensors

ROOT = Path(__file__).resolve().parents[1]


def production(notebook, **overrides):
    values = top_values(json.loads((ROOT / 'notebooks/experiments' / notebook).read_text()))
    values.update(overrides)
    return plain(load(environ={}, _profile=machine_profile('g4'), **values).values)


class TensorLimitTests(unittest.TestCase):
    def test_real_r1_uncapped_512_refuses_srm_output_and_cap64_accepts(self):
        values = production('srm-e50-paired.ipynb')
        self.assertEqual((values['batch_size'], values['train_tile_size'], values['tile_size']), (128, 256, 512))
        with self.assertRaisesRegex(ValueError, r'validation.*aten.convolution.*3019898880.*2147483647'):
            check_tensors(values)
        report = check_tensors(dict(values, val_batch_images=64))
        self.assertEqual(report['stages']['training']['batch_images'], 128)  # not 256
        self.assertEqual(report['stages']['validation']['batch_images'], 64)
        peak = report['stages']['validation']['largest'][0]
        self.assertEqual((peak['shape'], peak['numel']), ([64, 90, 512, 512], 1509949440))

    def test_real_covariance_learn_srm_and_b36_full_production_graphs(self):
        for notebook in ('srm-e50-paired-gapcov32.ipynb', 'srm-e50-paired-learn-srm.ipynb',
                         'srm-e50-paired-coloursep36.ipynb'):
            with self.subTest(notebook=notebook), patch('torch.compile', side_effect=AssertionError('compile forbidden')):
                before = torch.get_rng_state().clone()
                report = check_tensors(production(notebook))
                self.assertTrue(torch.equal(before, torch.get_rng_state()))
                self.assertEqual(report['limit'], LIMIT)
                for stage in ('training', 'validation'):
                    part = report['stages'][stage]
                    self.assertEqual(part['batch_images'], 128 if stage == 'training' else 64)
                    self.assertEqual(part['tile_size'], 256 if stage == 'training' else 512)
                    self.assertGreater(part['tensor_outputs'], 20)
                    self.assertTrue(all(row['numel'] <= LIMIT and row['op'] for row in part['largest']))

    def test_training_backward_and_train_matched_geometry_no_cap_or_larger_cap(self):
        values = production('srm-e50-paired-learn-srm.ipynb')
        values.update(checkpoint_validation='train_matched', val_batch_images=None)
        observed = []
        original = OutputRecorder.observe
        def trace(recorder, op, tensor):
            observed.append((recorder.stage, str(op)))
            return original(recorder, op, tensor)
        with patch.object(OutputRecorder, 'observe', trace):
            report = check_tensors(values)
        self.assertEqual(report['stages']['validation']['tile_size'], 256)
        self.assertEqual(report['stages']['validation']['batch_images'], 128)
        self.assertTrue(any(stage == 'training' and 'backward' in op for stage, op in observed))
        self.assertFalse(any(stage == 'validation' and 'backward' in op for stage, op in observed))
        self.assertEqual(check_tensors(dict(values, val_batch_images=256))['stages']['validation']['batch_images'], 128)

    def test_exact_limit_passes_next_element_refuses_without_large_allocation(self):
        with FakeTensorMode(allow_fallback_kernels=False):
            tiny = torch.empty(1)
            recorder = OutputRecorder('synthetic')
            with patch.object(recorder, 'numel', return_value=2**31 - 1):
                recorder.observe('synthetic.boundary', tiny)
            self.assertEqual(recorder.largest[0]['numel'], LIMIT)
            with patch.object(recorder, 'numel', return_value=2**31):
                with self.assertRaisesRegex(ValueError, 'synthetic.boundary.*2147483648.*exceeds 2147483647'):
                    recorder.observe('synthetic.boundary', tiny)

    def test_unsupported_forward_and_backward_name_the_op_and_fail_closed(self):
        class BadBackward(torch.autograd.Function):
            @staticmethod
            def forward(ctx, x):
                ctx.shape = x.shape
                return x.sum()
            @staticmethod
            def backward(ctx, grad):
                torch.nonzero(grad)  # unknown data-dependent shape: no fake kernel fallback
                return torch.empty(ctx.shape)
        class Unsupported(torch.nn.Module):
            def __init__(self, backward):
                super().__init__(); self.weight = torch.nn.Parameter(torch.empty(1)); self.backward = backward
            def forward(self, x):
                return BadBackward.apply(x * self.weight) if self.backward else torch.nonzero(x)
        values = dict(batch_size=1, train_tile_size=2, tile_size=2)
        for backward in (False, True):
            with patch('cvdl.models.model_from_hparams', side_effect=lambda h: Unsupported(backward)):
                with self.assertRaisesRegex(ValueError, r'REFUSED.*training.*aten.nonzero'):
                    check_tensors(values)

    def test_pinned_preflight_cli_policy_refuses_r1_but_accepts_covariance(self):
        with self.assertRaisesRegex(ValueError, r'REFUSED.*validation.*3019898880'):
            preflight('notebooks/experiments/srm-e50-paired.ipynb', 'HEAD')
        report = preflight('notebooks/experiments/srm-e50-paired-gapcov32.ipynb', 'HEAD')
        self.assertEqual(report['tensor_limits']['stages']['validation']['batch_images'], 64)
        self.assertEqual(report['largest_validation_srm_output_elements'], 1509949440)

    def test_resume_identity_stays_lightweight_and_does_not_run_gate(self):
        with patch('scripts.g4.tensor_limits.check_tensors', side_effect=AssertionError('not identity-only')):
            report = resume_identity('notebooks/experiments/srm-e50-paired.ipynb', 'HEAD')
        self.assertEqual(report['validation'], 'resume-identity-only')
        self.assertNotIn('tensor_limits', report)


if __name__ == '__main__': unittest.main()
