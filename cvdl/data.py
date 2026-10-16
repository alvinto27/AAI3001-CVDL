"""Notebook data helpers with explicit inputs (no notebook globals or env reads).

OTF readers/seeds/samplers remain owned by the LSB library. This module wires
that library into experiments; it does not duplicate its decoding or seed code.
"""
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

from .models import KV_ARRAY
from lsb_notebook import NotebookData
from lsb_torch import PrimedTrainingLoader


def make_pairs(frame: pd.DataFrame) -> pd.DataFrame:
    """Merge each clean tile with its matching stego tile."""
    clean = frame.loc[frame["label"] == 0].copy().rename(columns={"file": "file_clean"})
    stego = frame.loc[frame["label"] == 1].copy().rename(columns={"file": "file_stego"})
    pairs = clean.merge(stego, on="cover_id", suffixes=("_clean", "_stego"), validate="one_to_one")
    for column in ["source_photo", "camera_model", "tile_row", "tile_col"]:
        left = pairs[f"{column}_clean"].astype(str)
        right = pairs[f"{column}_stego"].astype(str)
        assert left.equals(right), f"clean/stego mismatch in {column}"
        pairs[column] = pairs[f"{column}_stego"]
    for column in ["k", "bit_mode", "method", "placement", "payload_kind", "bpp_actual"]:
        pairs[column] = pairs[f"{column}_stego"]
    keep = [
        "cover_id", "file_clean", "file_stego", "source_photo", "camera_model",
        "tile_row", "tile_col", "k", "bit_mode", "method", "placement",
        "payload_kind", "bpp_actual",
    ]
    return pairs[keep].reset_index(drop=True)


def source_split(pair_frame: pd.DataFrame, path: Path, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create or reuse a camera-stratified file-to-split map."""
    file_info = pd.concat([
        pair_frame[["file_clean", "source_photo", "camera_model"]].rename(columns={"file_clean": "file"}),
        pair_frame[["file_stego", "source_photo", "camera_model"]].rename(columns={"file_stego": "file"}),
    ], ignore_index=True)
    assert not file_info["file"].duplicated().any()
    source_info = file_info[["source_photo", "camera_model"]].drop_duplicates()
    assert not source_info["source_photo"].duplicated().any()
    if path.exists():
        saved = pd.read_csv(path, dtype={"file": str, "split": str})
        print(f"Reusing existing split: {path}")
    else:
        rng = np.random.default_rng(seed)
        source_rows = []
        for camera, group in source_info.groupby("camera_model", sort=True):
            sources = group["source_photo"].to_numpy(copy=True)
            rng.shuffle(sources)
            n = len(sources)
            n_train = int(round(n * 0.70))
            n_val = int(round(n * 0.15))
            n_test = n - n_train - n_val
            assert n_test >= 0
            for split_name, values in zip(
                ["train", "val", "test"],
                [sources[:n_train], sources[n_train:n_train + n_val], sources[n_train + n_val:]],
            ):
                source_rows.extend({"source_photo": str(source), "split": split_name} for source in values)
        source_assignments = pd.DataFrame(source_rows, columns=["source_photo", "split"])
        saved = file_info[["file", "source_photo"]].merge(source_assignments, on="source_photo", validate="many_to_one")[
            ["file", "split"]
        ]
        path.parent.mkdir(parents=True, exist_ok=True)
        saved.to_csv(path, index=False)
        print(f"Wrote new split: {path}")
    assert list(saved.columns) == ["file", "split"]
    assert set(saved["file"]) == set(file_info["file"])
    assert not saved["file"].duplicated().any()
    assert set(saved["split"]) <= {"train", "val", "test"}
    file_with_source = file_info.merge(saved, on="file", validate="one_to_one")
    source_assignments = file_with_source[["source_photo", "split"]].drop_duplicates()
    assert not source_assignments["source_photo"].duplicated().any()
    assert set(source_assignments["source_photo"]) == set(source_info["source_photo"])
    return saved, source_assignments


def read_raw(path: Path) -> torch.Tensor:
    """Read an RGB PNG as CHW float32 values in the original 0-255 range."""
    with Image.open(path) as image:
        array = np.asarray(image.convert("RGB"), dtype=np.uint8).copy()
    return torch.from_numpy(array).permute(2, 0, 1).contiguous().float()


def kv_residual(image: torch.Tensor) -> torch.Tensor:
    """Apply the fixed KV filter to CHW or BCHW RGB float tensors."""
    kernel = torch.from_numpy(KV_ARRAY).to(dtype=image.dtype, device=image.device)
    weight = kernel.view(1, 1, 5, 5).repeat(3, 1, 1, 1)
    single = image.ndim == 3
    result = F.conv2d(image.unsqueeze(0) if single else image, weight, padding=2, groups=3)
    return result.squeeze(0) if single else result


class PairDataset(Dataset):
    """Load paired tiles for validation, testing, and optional paired training."""

    def __init__(self, frame: pd.DataFrame, data_root: Path, train: bool = False):
        self.data_root = Path(data_root)
        self.frame = frame.reset_index(drop=True)
        self.train = train

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int):
        row = self.frame.iloc[index]
        clean = read_raw(self.data_root / row["file_clean"])
        stego = read_raw(self.data_root / row["file_stego"])
        if self.train:
            clean = augment_image(clean)
            stego = augment_image(stego)
        return clean, stego, index


def augment_image(image: torch.Tensor) -> torch.Tensor:
    """Apply the baseline's random geometric augmentation to one image."""
    if bool(torch.rand(()) < 0.5):
        image = torch.flip(image, dims=(-1,))
    if bool(torch.rand(()) < 0.5):
        image = torch.flip(image, dims=(-2,))
    turns = int(torch.randint(0, 4, ()).item())
    return torch.rot90(image, turns, dims=(-2, -1))


class ImageDataset(Dataset):
    """Expose clean and stego tiles as separate shuffled training samples."""

    def __init__(self, frame: pd.DataFrame, data_root: Path, train: bool = False):
        self.data_root = Path(data_root)
        self.train = train
        self.samples = []
        for row in frame.itertuples(index=False):
            self.samples.append((row.file_clean, 0))
            self.samples.append((row.file_stego, 1))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        file_name, label = self.samples[index]
        image = read_raw(self.data_root / file_name)
        if self.train:
            image = augment_image(image)
        return image, label, index



@dataclass
class PairTables:
    pairs: pd.DataFrame
    pair_splits: pd.DataFrame
    run_pairs: pd.DataFrame
    source_assignments: pd.DataFrame


def load_labels(*, labels_path=None, otf_data=None):
    """Load PNG labels or expose both labels for the frozen OTF pair table."""
    if (labels_path is None) == (otf_data is None):
        raise ValueError("provide exactly one of labels_path or otf_data")
    if labels_path is not None:
        return pd.read_csv(labels_path)
    return pd.concat([otf_data.pairs.assign(label=0), otf_data.pairs.assign(label=1)], ignore_index=True)


def _check_source_overlap(assignments):
    sets = {name: set(assignments.loc[assignments['split'] == name, 'source_photo'])
            for name in ('train', 'val', 'test')}
    assert not (sets['train'] & sets['val'] or sets['train'] & sets['test'] or sets['val'] & sets['test'])


def png_pair_tables(labels, split_path, seed, *, quick_pairs=None):
    """Create/reuse source splits. quick_pairs=None keeps every pair."""
    pairs = make_pairs(labels)
    _, assignments = source_split(pairs, Path(split_path), seed)
    splits = pairs.merge(assignments, on='source_photo', how='left', validate='many_to_one')
    assert not splits['split'].isna().any()
    _check_source_overlap(assignments)
    if quick_pairs is None:
        run = splits.copy()
    else:
        frames = []
        for offset, name in enumerate(('train', 'val', 'test')):
            available = splits.loc[splits['split'] == name]
            frames.append(available.sample(n=min(quick_pairs[name], len(available)), random_state=seed + offset).copy())
        run = pd.concat(frames, ignore_index=True)
    return PairTables(pairs, splits, run, assignments)


def otf_pair_tables(data):
    """Use frozen OTF splits without making or rewriting a split file."""
    pairs = data.pairs.copy()
    assignments = pairs[['source_photo', 'source', 'camera_model', 'split']].drop_duplicates()
    _check_source_overlap(assignments)
    return PairTables(pairs, pairs.copy(), pairs.copy(), assignments)


def build_otf_data(root, settings, *, quick=False, profile='overlap_full_mix', eval_pairs=16,
                   checkpoint_validation='standard', val_cap_raise=5000, val_cap_lsdir=1500, train_grid='recorded',
                   train_tile=None):
    """Construct the existing LSB NotebookData adapter with explicit settings."""
    return NotebookData(root, settings, quick=quick, profile=profile, eval_pairs=eval_pairs,
                        checkpoint_validation=checkpoint_validation, val_cap_raise=val_cap_raise,
                        val_cap_lsdir=val_cap_lsdir, train_grid=train_grid, train_tile=train_tile)


@dataclass
class Loaders:
    train: object
    val: object
    test: object
    train_eval: object
    train_frame: pd.DataFrame
    val_frame: pd.DataFrame
    test_frame: pd.DataFrame


def build_otf_loaders(data, *, batch_size, workers, eval_workers, pin_memory,
                      prefetch=None, prefetch_images=128, eval_persistent_workers=False,
                      prime_next_epoch=False):
    train, val, test, train_eval = data.loaders(batch_size, workers, pin_memory, prefetch,
        eval_workers=eval_workers, prefetch_images=prefetch_images,
        eval_persistent_workers=eval_persistent_workers)
    if prime_next_epoch:
        train = PrimedTrainingLoader(train)
    return Loaders(train, val, test, train_eval, data.fixed_train.frame, data.val.frame, data.test.frame)


def build_png_loaders(run_pairs, data_root, *, batch_size, workers, eval_workers,
                      pin_memory, paired_batches, seed, eval_persistent_workers=False):
    frames = [run_pairs.loc[run_pairs['split'] == name].reset_index(drop=True)
              for name in ('train', 'val', 'test')]
    train_frame, val_frame, test_frame = frames
    train_dataset = (PairDataset(train_frame, data_root, train=True) if paired_batches
                     else ImageDataset(train_frame, data_root, train=True))
    kwargs = dict(num_workers=workers, pin_memory=pin_memory)
    if workers > 0:
        kwargs['persistent_workers'] = True
    eval_kwargs = dict(num_workers=eval_workers, pin_memory=pin_memory, persistent_workers=False)
    pair_batch = max(1, batch_size // 2)
    train = DataLoader(train_dataset, batch_size=pair_batch if paired_batches else batch_size,
                       shuffle=True, **kwargs)
    val = DataLoader(PairDataset(val_frame, data_root), batch_size=pair_batch, shuffle=False,
                     **{**eval_kwargs, 'persistent_workers': eval_persistent_workers})
    test = DataLoader(PairDataset(test_frame, data_root), batch_size=pair_batch,
                      shuffle=False, **eval_kwargs)
    diagnostic = train_frame.sample(n=min(1000, len(train_frame)), random_state=seed).reset_index(drop=True)
    train_eval = DataLoader(PairDataset(diagnostic, data_root), batch_size=pair_batch,
                            shuffle=False, **eval_kwargs)
    return Loaders(train, val, test, train_eval, train_frame, val_frame, test_frame)
