"""Opt-in highest-cell pooling; the normal model/evaluator is unchanged.

Definition fixed before results in plan-diagnostic-checks.md: rank spatial cells
by stego-minus-clean logit margin, select ONE common set for the two classes,
and average the paired logits. Never maximize each class separately. The user's
amendment scores float64 pooled-logit margins; softmax is retained for parity
only because it loses ranking information when probabilities saturate at 1.0.
"""
import torch


def pool_logits(cell_logits, mode='mean'):
    """Pool an N x 2 x H x W tensor into N x 2 logits."""
    if cell_logits.ndim != 4 or cell_logits.shape[1] != 2 or min(cell_logits.shape[2:]) < 1:
        raise ValueError('expected N x 2 x H x W cell logits')
    flat = cell_logits.flatten(2)
    if mode == 'mean':
        return flat.mean(2)
    if mode not in ('max', 'top4'):
        raise ValueError('pooling must be mean, max or top4')
    count = 1 if mode == 'max' else 4
    if flat.shape[2] < count:
        raise ValueError('not enough spatial cells')
    indices = (flat[:, 1]-flat[:, 0]).topk(count, dim=1).indices
    paired = flat.gather(2, indices[:, None, :].expand(-1, 2, -1))
    return paired.mean(2)


def pooled_logits64(cell_logits):
    """User amendment: average paired logits in float64, preserving margin order.

    Score each returned pair as logits[:, 1]-logits[:, 0], in float64. Do NOT
    softmax these scores before calibration or ranking: even float64 softmax
    can tie sufficiently large positive margins at 1.0.
    """
    cells = cell_logits.to(dtype=torch.float64)
    return {mode: pool_logits(cells, mode) for mode in ('mean', 'max', 'top4')}


def pooled_probabilities(cell_logits):
    """Pool logits first; return three N-vector stego probabilities."""
    return {mode: torch.softmax(pool_logits(cell_logits, mode), dim=1)[:, 1]
            for mode in ('mean', 'max', 'top4')}
