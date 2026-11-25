"""Small module-level dataset for tests/test_pc_safety.py (spawned workers must be able to import it)."""
import torch
from torch.utils.data import IterableDataset


class Counting(IterableDataset):
    def __iter__(self):
        for i in range(40):
            yield torch.full((3, 8, 8), i, dtype=torch.uint8), i % 2, i
