"""Shared AMP boundaries and opt-in memory layout; no training recipe changes."""
import copy
import json
from types import SimpleNamespace
import tempfile
import unittest

import lightning as L
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from cvdl import runtime
from cvdl.gbras import GBRASBatchNorm
from cvdl.lit import SteganalysisModule
from cvdl.models import clipped_residuals, model_from_hparams
from cvdl.settings import load, machine_profile

HP = dict(model='gbras', front_end='srm', norm='batch')


def settings(**kwargs):
    return load(environ={}, _profile=machine_profile('pc', device='cpu',
        precision_limits=('32', 'bf16-mixed')), **kwargs)


class PrecisionLayoutTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_srm_boundary_all_models_and_kv_unchanged_under_cpu_autocast(self):
        x = torch.randint(0, 256, (2, 3, 32, 32)).float()
        for model in ('xunet', 'xunet_wide', 'xunet_wide_delayed', 'gbras'):
            for auxiliary in (False, True) if model == 'xunet_wide' else (False,):
                hp = dict(model=model, front_end='srm', norm='batch' if model == 'gbras' else 'none')
                if auxiliary:
                    hp['cell_supervision'] = 'changed_cells_v1'
                net = model_from_hparams(hp).eval()
                seen = []
                first = net.adapter if model == 'gbras' else net.conv1
                hook = first.register_forward_pre_hook(lambda m, a: seen.append(a[0].detach().clone()))
                with torch.no_grad():
                    expected = net.kv(x).clamp(-3, 3)
                    with torch.autocast('cpu', dtype=torch.bfloat16):
                        logits = net(x)
                hook.remove()
                self.assertEqual(logits.dtype, torch.bfloat16)
                self.assertEqual(seen[0].dtype, torch.float32)
                self.assertTrue(torch.equal(seen[0], expected), (model, auxiliary))
        kv = model_from_hparams(dict(model='xunet', norm='none')).kv
        with torch.autocast('cpu', dtype=torch.bfloat16):
            self.assertTrue(torch.equal(clipped_residuals(kv, x, 3), kv(x).clamp(-3, 3)))
            self.assertEqual(clipped_residuals(kv, x, 3).dtype, torch.bfloat16)

    def test_unclamped_srm_and_float_input_restored_from_low_precision(self):
        net = model_from_hparams(dict(model='xunet', front_end='srm', norm='none'))
        x = torch.randint(0, 256, (2, 3, 32, 32)).to(torch.bfloat16)
        expected = net.kv(x.float())
        with torch.autocast('cpu', dtype=torch.bfloat16):
            actual = clipped_residuals(net.kv, x, None)
        self.assertEqual(actual.dtype, torch.float32)
        self.assertTrue(torch.equal(actual, expected))

    def test_loss_and_validation_probabilities_are_fp32_even_without_autocast(self):
        class LowPrecisionLogits(nn.Module):
            def forward(self, images):
                return torch.tensor([[.2, .4]], dtype=torch.bfloat16).repeat(len(images), 1).requires_grad_()
        module = SteganalysisModule(**HP)
        module.net = LowPrecisionLogits()
        module.log = lambda *a, **k: None
        x = torch.zeros(2, 3, 32, 32)
        labels = torch.tensor([0, 1])
        loss = module.training_step((x, labels, labels), 0)['loss']
        self.assertEqual(loss.dtype, torch.float32)
        loss.backward()
        module.validation_step((x, x, labels), 0)
        self.assertEqual(module.validation_scores[0].dtype, torch.float32)
        module.on_validation_epoch_end()  # numpy() must not receive BF16.

    def test_channels_last_transfer_for_unpaired_paired_and_nested_batches(self):
        module = SteganalysisModule(**HP, channels_last=True)
        x = torch.randn(2, 3, 32, 32)
        labels = torch.tensor([0, 1])
        for batch in ((x, labels, labels), [x, x, labels], {'image': x, 'tag': (labels,)}):
            actual = module.transfer_batch_to_device(batch, torch.device('cpu'), 0)
            self.assertIs(type(actual), type(batch))
            image = actual['image'] if isinstance(actual, dict) else actual[0]
            self.assertEqual(image.shape, x.shape)
            self.assertTrue(image.is_contiguous(memory_format=torch.channels_last))
            self.assertTrue(torch.equal(image, x))
        batch = module.transfer_batch_to_device((x, x, labels), torch.device('cpu'), 0)
        seen = []
        hook = module.net.register_forward_pre_hook(lambda m, a: seen.append(a[0]))
        module.pair_logits_targets(batch)
        hook.remove()
        self.assertEqual(tuple(seen[0].shape), (4, 3, 32, 32))
        self.assertTrue(seen[0].is_contiguous(memory_format=torch.channels_last))
        original = SteganalysisModule(**HP)
        self.assertIs(original.transfer_batch_to_device((x, labels, labels), torch.device('cpu'), 0)[0], x)

    def test_channels_last_checkpoint_keys_values_and_reload(self):
        for hp in (dict(model='xunet', norm='group'),
                   dict(model='xunet_wide', front_end='srm', norm='none'),
                   dict(model='xunet_wide_delayed', front_end='srm', norm='none'),
                   dict(model='xunet_wide', front_end='srm', norm='none', cell_supervision='changed_cells_v1'), HP):
            torch.manual_seed(42); regular = SteganalysisModule(**hp)
            torch.manual_seed(42); formatted = SteganalysisModule(**hp, channels_last=True)
            self.assertEqual(set(regular.state_dict()), set(formatted.state_dict()))
            for key, tensor in regular.state_dict().items():
                self.assertTrue(torch.equal(tensor, formatted.state_dict()[key]), key)
            for tensor in formatted.net.parameters():
                if tensor.ndim == 4:
                    self.assertTrue(tensor.is_contiguous(memory_format=torch.channels_last))
            restored = SteganalysisModule(**dict(formatted.hparams))
            restored.load_state_dict(formatted.state_dict(), strict=True)
            self.assertTrue(restored.channels_last)
            self.assertNotIn('channels_last', regular.hparams)

    def test_real_lightning_bf16_fit_and_validation_dtypes(self):
        module = SteganalysisModule(**HP, channels_last=True)
        x = torch.randint(0, 256, (2, 3, 32, 32)).float()
        labels = torch.tensor([0, 1]); seen = []
        hook = module.net.adapter.register_forward_hook(lambda m, a, out: seen.append((a[0].dtype, out.dtype, a[0].is_contiguous(memory_format=torch.channels_last))))
        score_dtypes = []
        class CaptureScores(L.Callback):
            def on_validation_batch_end(self, trainer, model, outputs, batch, batch_idx, dataloader_idx=0):
                score_dtypes.append(model.validation_scores[-1].dtype)
        trainer = L.Trainer(accelerator='cpu', devices=1, precision='bf16-mixed', max_epochs=1,
            callbacks=[CaptureScores()], logger=False, enable_checkpointing=False, enable_progress_bar=False,
            enable_model_summary=False, num_sanity_val_steps=0)
        trainer.fit(module, DataLoader(TensorDataset(x, labels, labels), batch_size=2),
                    DataLoader(TensorDataset(x, x, labels), batch_size=2))
        hook.remove()
        self.assertTrue(seen)
        self.assertTrue(all(row == (torch.float32, torch.bfloat16, True) for row in seen))
        for name, tensor in module.named_parameters():
            self.assertEqual(tensor.dtype, torch.float32, name)
        norms = [m for m in module.modules() if isinstance(m,GBRASBatchNorm)]
        self.assertEqual(len(norms), 13)
        for bn in norms:
            for tensor in (bn.weight, bn.bias, bn.running_mean, bn.running_var):
                self.assertEqual(tensor.dtype, torch.float32)
                self.assertTrue(torch.isfinite(tensor).all())
        for key, value in trainer.callback_metrics.items():
            self.assertEqual(value.dtype, torch.float32, key)
        self.assertEqual(score_dtypes, [torch.float32])

    def test_settings_default_validation_and_run_records(self):
        self.assertFalse(settings().channels_last)
        self.assertTrue(settings(overrides=['channels_last=on']).channels_last)
        s = load(environ={'V3_CHANNELS_LAST': 'true'}, _profile=machine_profile('pc', device='cpu'))
        self.assertTrue(s.channels_last)
        with self.assertRaises(ValueError): settings(overrides=['channels_last=maybe'])
        with self.assertRaisesRegex(ValueError, 'machine-owned'): settings(channels_last=True)
        with tempfile.TemporaryDirectory() as directory:
            s = settings(model='gbras', front_end='srm', norm='batch', data='png',
                         overrides=['channels_last=on', 'precision=bf16-mixed', f'output_root={directory}'])
            bundle = runtime.create_training(s, None, SimpleNamespace(train=[0,1], val=[0]))
            record = json.loads((bundle.checkpoint_dir/'model_run_settings.json').read_text())
            self.assertTrue(bundle.lightning_model.channels_last)
            self.assertTrue(record['channels_last'])
            self.assertEqual(record['memory_format'], 'channels_last')
            self.assertEqual(record['precision'], 'bf16-mixed')
            self.assertEqual(record['amp_boundaries']['srm_front_end_and_clamp'], 'float32_autocast_disabled')
            self.assertEqual(record['amp_boundaries']['cross_entropy'], 'float32')
            self.assertEqual(record['amp_boundaries']['validation_probabilities_and_metrics'], 'float32')
            self.assertEqual(record['compile_mode'], s.compile)
            self.assertEqual(record['tf32_matmul'], torch.backends.cuda.matmul.allow_tf32)
            self.assertEqual(record['tf32_cudnn'], torch.backends.cudnn.allow_tf32)
            saved = json.loads((bundle.checkpoint_dir/'run_settings.json').read_text())
            self.assertTrue(saved['values']['channels_last'])

    def test_cpu_bf16_aot_compile_preserves_boundary_and_keys(self):
        module = SteganalysisModule(**HP, channels_last=True).eval()
        compiled = copy.deepcopy(module)
        compiled.net.compile(backend='aot_eager', dynamic=False, fullgraph=False)
        x = torch.randn(2, 3, 32, 32).contiguous(memory_format=torch.channels_last)
        with torch.no_grad(), torch.autocast('cpu', dtype=torch.bfloat16):
            expected = module.net(x); actual = compiled.net(x)
        self.assertEqual(actual.dtype, torch.bfloat16)
        torch.testing.assert_close(actual, expected)
        self.assertEqual(set(module.state_dict()), set(compiled.state_dict()))
        self.assertFalse(any('_orig_mod' in k for k in compiled.state_dict()))

    @unittest.skipUnless(torch.cuda.is_available() and torch.cuda.is_bf16_supported(), 'small BF16 GPU correctness smoke only')
    def test_gpu_bf16_channels_last_smoke(self):
        module = SteganalysisModule(**HP, channels_last=True).cuda().train()
        x = torch.randint(0,256,(2,3,64,64)).float()
        batch = module.transfer_batch_to_device((x, torch.tensor([0,1]), torch.arange(2)), torch.device('cuda'), 0)
        module.log = lambda *a, **k: None
        with torch.autocast('cuda', dtype=torch.bfloat16):
            loss = module.training_step(batch, 0)['loss']
        self.assertEqual(loss.dtype, torch.float32)
        loss.backward(); torch.cuda.synchronize()
        for name, tensor in module.named_parameters():
            self.assertEqual(tensor.dtype, torch.float32, name)
            if tensor.requires_grad:
                self.assertIsNotNone(tensor.grad, name)
                self.assertTrue(torch.isfinite(tensor.grad).all(), name)
        self.assertTrue(torch.isfinite(loss))
        for bn in module.modules():
            if isinstance(bn, GBRASBatchNorm):
                for tensor in (bn.weight, bn.bias, bn.running_mean, bn.running_var):
                    self.assertEqual(tensor.dtype, torch.float32)
                    self.assertTrue(torch.isfinite(tensor).all())


if __name__ == '__main__':
    unittest.main()
