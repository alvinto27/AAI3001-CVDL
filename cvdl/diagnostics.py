"""Batch-dependence checks with explicit data, device and normalization choices."""
from copy import deepcopy

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from .data import PairDataset
from lsb_notebook import fixed_subset
from lsb_torch import PreloadedPairs, preload_pairs


def diagnostic_pairs(dataset, *, quick, data_mode, workers, data_root=None):
    """Select seed-0 pairs and preload OTF tiles once per photo (no RNG changes)."""
    count = 32 if quick else 1000
    if data_mode == 'png':
        if data_root is None:
            raise ValueError('PNG diagnostics require data_root')
        frame = dataset.frame.sample(n=count, random_state=0).reset_index(drop=True)
        return PairDataset(frame, data_root)
    if data_mode != 'otf':
        raise ValueError('data_mode must be otf/png')
    subset = fixed_subset(dataset, min(count, len(dataset)), 0)
    result = PreloadedPairs(preload_pairs(subset, workers))
    result.frame = subset.frame
    return result


def batch_dependence(network, dataset, *, device, norm, split_name, quick,
                     loader_workers=0):
    """Return notebook-05 checks a/c/e/f without changing the original network.

    Pass the already-selected/preloaded pairs from diagnostic_pairs. Each train
    mode uses a fresh network copy and the same NumPy samples, as in cell 31.
    """
    if split_name not in {'train', 'val'}:
        raise ValueError('split_name must be train/val')
    if len(dataset) < 32:
        raise ValueError('batch diagnostics require at least 32 pairs')
    check_batches = 2 if quick else 63
    loader = DataLoader(dataset, batch_size=16, shuffle=False, num_workers=loader_workers)
    eval_copy = deepcopy(network).to(device).eval()
    total_loss = correct = count = 0
    with torch.no_grad():
        for clean, stego, _ in loader:
            images = torch.cat([clean, stego]).to(device, dtype=torch.float32)
            targets = torch.cat([torch.zeros(len(clean), dtype=torch.long), torch.ones(len(stego), dtype=torch.long)]).to(device)
            logits = eval_copy(images)
            total_loss += float(F.cross_entropy(logits, targets, reduction='sum'))
            correct += int((logits.argmax(1) == targets).sum())
            count += len(targets)
    results = {'a': dict(accuracy=correct/count, loss=total_loss/count)}
    for mode in ('c', 'e', 'f'):
        check_model = deepcopy(network).to(device).train()
        rng = np.random.default_rng(200 + (split_name == 'val'))
        correct = count = prob_sum = 0
        max_mode_difference = 0.0
        with torch.no_grad():
            for _ in range(check_batches):
                indices = rng.choice(len(dataset), size=32, replace=False)
                mixed = rng.permutation(np.r_[np.zeros(16, dtype=int), np.ones(16, dtype=int)])
                targets_np = mixed if mode == 'c' else np.zeros(32, dtype=int) if mode == 'e' else np.ones(32, dtype=int)
                images = torch.stack([dataset[int(i)][int(label)] for i,label in zip(indices,targets_np)]).to(device,dtype=torch.float32)
                targets = torch.tensor(targets_np, dtype=torch.long, device=device)
                logits = check_model(images)
                if norm in {'none', 'group'}:
                    eval_logits = eval_copy(images)
                    difference = float((logits - eval_logits).abs().max())
                    max_mode_difference = max(max_mode_difference, difference)
                    assert torch.allclose(logits, eval_logits, atol=1e-5, rtol=1e-5)
                correct += int((logits.argmax(1) == targets).sum())
                prob_sum += float(logits.softmax(1)[:, 1].sum())
                count += len(targets)
        results[mode] = dict(accuracy=correct/count, mean_prob_stego=prob_sum/count)
        if norm in {'none', 'group'}:
            results[mode]['max_mode_difference'] = max_mode_difference
    return results
