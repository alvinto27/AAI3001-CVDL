"""Same-process notebook 05 / package data parity, with small decoded fixtures."""
import contextlib
import io
import itertools
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from cvdl import data
from lsb_notebook import NotebookData
from lsb_torch import RunSettings, PrimedTrainingLoader
from test_lsb_source import record

CELLS = json.loads((ROOT / 'notebooks/archive/05_baseline_kv_xunet_v4_pre-cvdl.ipynb').read_text())['cells']


def source(index):
    return ''.join(CELLS[index]['source'])


def notebook_namespace(root, **values):
    ns = dict(np=np, pd=pd, Path=Path, torch=torch, F=F, Image=Image, Dataset=Dataset,
              DataLoader=DataLoader, os=os, DEVICE=torch.device('cpu'), DATA_ROOT=root,
              SEED=3, QUICK=False, QUICK_PAIRS={'train': 4, 'val': 2, 'test': 2},
              BATCH_SIZE=4, EVAL_WORKERS=0, EVAL_PERSISTENT_WORKERS=False,
              OTF_WORKERS=0, OTF_PREFETCH=None, OTF_PREFETCH_IMAGES=8, PRIME_NEXT_EPOCH=False,
              PrimedTrainingLoader=PrimedTrainingLoader)
    ns.update(values)
    # Run the actual reader/filter definitions; leave the example figure in the notebook.
    exec(source(10).split('example =')[0], ns)
    return ns


def run_cells(ns):
    with patch.dict(os.environ, {'V3_WORKERS': '0'}), contextlib.redirect_stdout(io.StringIO()):
        for i in (6, 8, 12):
            exec(compile(source(i), f'notebook05-cell-{i}', 'exec'), ns)


def assert_batches(test, old, new, n=2):
    torch.manual_seed(42)
    a = list(itertools.islice(old, n))
    torch.manual_seed(42)
    b = list(itertools.islice(new, n))
    test.assertEqual(len(a), len(b))
    test.assertTrue(a)
    for left, right in zip(a, b):
        for x, y in zip(left, right):
            test.assertTrue(torch.equal(x, y))


class DataParityTests(unittest.TestCase):
    def test_png_frames_datasets_augmentation_and_loaders(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            rows = []
            for i in range(20):
                for label in (0, 1):
                    name = f'{i}-{label}.png'
                    pixels = np.random.default_rng(i * 2 + label).integers(0, 256, (16, 16, 3), dtype=np.uint8)
                    Image.fromarray(pixels).save(root / name)
                    rows.append(dict(file=name, label=label, cover_id=f'cover{i}', source_photo=f'photo{i}',
                        camera_model='camera', tile_row=0, tile_col=0, k=1, bit_mode='lowest',
                        method='replace', placement='random', payload_kind='random_bytes', bpp_actual=.1))
            labels = pd.DataFrame(rows)
            labels.to_csv(root / 'labels.csv', index=False)
            for paired in (False, True):
                with self.subTest(paired=paired):
                    ns = notebook_namespace(root, V3_DATA='png', LABELS_PATH=root / 'labels.csv',
                        SPLIT_PATH=root / 'old-split.csv', PAIRED_BATCHES=paired, QUICK=True)
                    run_cells(ns)
                    tables = data.png_pair_tables(data.load_labels(labels_path=root / 'labels.csv'),
                        root / 'new-split.csv', 3, quick_pairs=ns['QUICK_PAIRS'])
                    for name in ('pairs', 'pair_splits', 'run_pairs', 'source_assignments'):
                        pd.testing.assert_frame_equal(ns[name], getattr(tables, name), check_exact=True)
                    loaders = data.build_png_loaders(tables.run_pairs, root, batch_size=4, workers=0,
                        eval_workers=0, pin_memory=False, paired_batches=paired, seed=3)
                    for name in ('train', 'val', 'test', 'train_eval'):
                        assert_batches(self, ns[name + '_loader'], getattr(loaders, name))
                    raw = ns['read_raw'](root / '0-0.png')
                    self.assertTrue(torch.equal(raw, data.read_raw(root / '0-0.png')))
                    self.assertTrue(torch.equal(ns['kv_residual'](raw), data.kv_residual(raw)))
                    self.assertTrue(torch.equal(data.kv_residual(raw), data.kv_residual(raw[None])[0]))

    def test_otf_decode_frames_and_epoch_batches(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / 'configs').mkdir()
            shutil.copyfile(ROOT / 'configs/raise512-run1.yaml', root / 'configs/raise512-run1.yaml')
            index = 0
            for source_name in ('raise', 'div2k', 'alaska', 'lsdir'):
                rows = []
                for split in ('train', 'val', 'test'):
                    index += 1
                    name = f'{source_name}-{split}.png'
                    pixels = np.random.default_rng(index).integers(0, 256, (512, 512, 3), dtype=np.uint8)
                    Image.fromarray(pixels).save(root / name)
                    row = {**record(512, 512), 'source': source_name, 'photo_id': f'photo{index}',
                           'photo_index': str(index), 'camera_model': '', 'master_path': name, 'split': split}
                    if source_name in {'alaska', 'lsdir'}:
                        row['grid_policy'] = 'trim512'
                    rows.append(row)
                path = root / 'data' / source_name / 'photo_manifest.csv'
                path.parent.mkdir(parents=True)
                pd.DataFrame(rows).to_csv(path, index=False)
            cases = [RunSettings(source='decode', tile_size=256, buffer_size=8),
                     RunSettings(source='decode', paired_batches=True, buffer_size=8),
                     RunSettings(source='decode', sources=('raise', 'lsdir', 'alaska'),
                         source_mix={'raise': .4, 'lsdir': .2, 'alaska': .4}, tiles_per_epoch=10, buffer_size=8)]
            for settings in cases:
                with self.subTest(settings=settings):
                    validation = 'capped' if settings.source_mix else 'standard'
                    kwargs = dict(quick=True, checkpoint_validation=validation, val_cap_raise=1, val_cap_lsdir=1)
                    original = NotebookData(root, settings, **kwargs)
                    shared = data.build_otf_data(root, settings, **kwargs)
                    ns = notebook_namespace(root, V3_DATA='otf', otf_data=original,
                                            PAIRED_BATCHES=settings.paired_batches, settings=settings)
                    run_cells(ns)
                    pd.testing.assert_frame_equal(ns['labels'], data.load_labels(otf_data=shared), check_exact=True)
                    tables = data.otf_pair_tables(shared)
                    for name in ('pairs', 'pair_splits', 'run_pairs', 'source_assignments'):
                        pd.testing.assert_frame_equal(ns[name], getattr(tables, name), check_exact=True)
                    loaders = data.build_otf_loaders(shared, batch_size=4, workers=0, eval_workers=0,
                                                     pin_memory=False, prefetch_images=8)
                    try:
                        for epoch in (0, 1):
                            original.train.set_epoch(epoch)
                            shared.train.set_epoch(epoch)
                            assert_batches(self, ns['train_loader'], loaders.train)
                        for name in ('val', 'test', 'train_eval'):
                            pd.testing.assert_frame_equal(ns[name + '_loader'].dataset.frame,
                                getattr(loaders, name).dataset.frame, check_exact=True)
                            assert_batches(self, ns[name + '_loader'], getattr(loaders, name))
                    finally:
                        ns['train_loader'].shutdown()
                        loaders.train.shutdown()


if __name__ == '__main__':
    unittest.main()
