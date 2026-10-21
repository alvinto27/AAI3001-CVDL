"""Lightweight, shared evaluator pixel boundary. Preserve RGB8 sample values."""
import numpy as np


def evaluation_tensor(image):
    """Contiguous CHW RGB8, with no resize, scaling or augmentation."""
    import torch
    return torch.from_numpy(np.ascontiguousarray(image)).permute(2, 0, 1).contiguous()
