"""Opt-in auxiliary loss. Targets never enter the image network."""
import torch
from torch.nn import functional as F


def cell_loss(logits, tags):
    """Per-image class-balanced BCE; sparse-positive weights do not cancel."""
    if logits.ndim != 4 or tuple(logits.shape[1:]) != (1,8,8) or tuple(tags.shape) != (len(logits),130):
        raise ValueError('cell loss requires [B,1,8,8] logits and [B,130] tags')
    counts=tags[:,2:66].to(logits.dtype)
    valid=tags[:,66:130]
    # The CPU target reducer owns value validation. Do not synchronise the GPU
    # with a bool(tensor) check in every training step.
    positive=counts>0
    negative=(valid==1)&~positive
    bce=F.binary_cross_entropy_with_logits(logits.flatten(1),positive.to(logits.dtype),reduction='none')
    np_=positive.sum(1);nn_=negative.sum(1)
    pos=(bce*positive*(counts/16).clamp(max=1)).sum(1)/np_.clamp(min=1)
    neg=(bce*negative).sum(1)/nn_.clamp(min=1)
    classes=(np_>0).to(logits.dtype)+(nn_>0).to(logits.dtype)
    loss=((pos+neg)/classes.clamp(min=1)).mean()
    return loss, dict(positive_cells=np_.float().mean(),negative_cells=nn_.float().mean(),
        occupancy=positive.float().mean(),empty_stego_fraction=((np_==0)&(tags[:,0]%2==1)).float().mean())
