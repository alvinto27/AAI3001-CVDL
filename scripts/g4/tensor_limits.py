"""Production tensor-size gate using meta-backed fake CPU execution, not shape rules."""
import json

import torch
from torch._subclasses.fake_tensor import FakeTensor, FakeTensorMode
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils._pytree import tree_leaves

LIMIT = 2**31 - 1


class OutputRecorder(TorchDispatchMode):
    def __init__(self, stage):
        super().__init__()
        self.stage = stage
        self.outputs = 0
        self.largest = []
        self.last_op = 'none'

    def numel(self, tensor):
        return int(tensor.numel())

    def observe(self, op, tensor):
        if not isinstance(tensor, FakeTensor):
            raise ValueError(f'{self.stage}: {op} returned a non-fake tensor; refuse real allocation')
        item = dict(op=str(op), shape=list(tensor.shape), numel=self.numel(tensor))
        self.outputs += 1
        if item not in self.largest:
            self.largest = sorted([*self.largest, item], key=lambda row: row['numel'], reverse=True)[:5]
        if item['numel'] > LIMIT:
            raise ValueError(f"{self.stage}: {op} shape={item['shape']} numel={item['numel']} exceeds {LIMIT}")

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        self.last_op = str(func)
        try:
            output = func(*args, **(kwargs or {}))
        except Exception as error:
            raise ValueError(f'{self.stage}: meta/fake operation {func} failed: {error}') from error
        for tensor in tree_leaves(output):
            if isinstance(tensor, torch.Tensor):
                self.observe(func, tensor)
        return output

    def report(self):
        return dict(tensor_outputs=self.outputs, largest=self.largest)


def check_tensors(values):
    """Run the selected model's real forward/autograd with no data or GPU kernels.

    Fake CPU devices avoid unsupported autocast('meta') in the frozen SRM code;
    their storage is meta. Fallback to real kernels is explicitly disabled.
    """
    from cvdl.models import model_from_hparams
    hparams = dict(values)
    if hparams.get('model') == 'xunet_wide_gapcov32':
        from cvdl.cov_pool import SPEC
        hparams['cov_pool_identity'] = dict(SPEC)
    batch = int(values['batch_size'])  # Already total images, including paired runs.
    train_tile = int(values.get('train_tile_size') or values['tile_size'])
    val_tile = train_tile if values.get('checkpoint_validation') == 'train_matched' else int(values['tile_size'])
    cap = values.get('val_batch_images')
    val_batch = min(batch, int(cap)) if cap is not None else batch
    if min(batch, val_batch, train_tile, val_tile) <= 0:
        raise ValueError('production tensor check requires positive batches and tile sizes')
    report = dict(limit=LIMIT, execution='meta-backed fake CPU; no real-kernel fallback', stages={})
    trace = OutputRecorder('construction')
    try:
        with torch.random.fork_rng(devices=[]), FakeTensorMode(allow_fallback_kernels=False):
            with trace:
                model = model_from_hparams(hparams)
            report['stages']['construction'] = trace.report()
            for stage, images, tile in (('training', batch, train_tile), ('validation', val_batch, val_tile)):
                training = stage == 'training'
                trace = OutputRecorder(stage)
                model.train(training)
                with trace, torch.set_grad_enabled(training):
                    x = torch.empty(images, 3, tile, tile, device='cpu', dtype=torch.float32)
                    kwargs = dict(return_aux=True) if training and values.get('cell_supervision', 'off') != 'off' else {}
                    outputs = model(x, **kwargs)
                    if training:
                        # Sum every output so all used heads participate in autograd.
                        # Values are irrelevant: this is shape/size validation only.
                        sum(t.sum() for t in tree_leaves(outputs) if isinstance(t, torch.Tensor)).backward()
                report['stages'][stage] = dict(batch_images=images, tile_size=tile, **trace.report())
    except Exception as error:
        report['stages'][trace.stage] = trace.report()
        raise ValueError(f'production tensor check REFUSED ({trace.stage}, last op {trace.last_op}): {error}\n'
                         + json.dumps(report, sort_keys=True)) from error
    return report
