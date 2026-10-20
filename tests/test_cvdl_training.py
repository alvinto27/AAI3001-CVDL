"""CPU checkpoints load/resume and retain earlier training/validation metrics."""
import contextlib
import glob
import io
import json
import os
from pathlib import Path
from types import SimpleNamespace
import sys
import tempfile
import unittest
from unittest.mock import patch

import lightning as L
import numpy as np
from lightning.pytorch.loggers import TensorBoardLogger
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from cvdl import callbacks as cb, data, models
from cvdl.lit import SteganalysisModule, select_best_checkpoint
from scripts.refactor_golden import CHECKPOINTS, ckpt_structure, read_csv_rows, read_env_file


def prepare_data_and_rng():
    """Reuse only pre-settings/probe cells until their later refactor phases.

    The notebook initializes two throwaway models (count and KV optimizer
    probe) before the real model. Replaying those allocations is necessary to
    preserve its RNG stream, not a change to the training algorithm.
    """
    notebook = json.loads((ROOT / 'notebooks/archive/05_baseline_kv_xunet_v4_pre-cvdl.ipynb').read_text())
    ns = {'__name__': '__main__'}
    with patch('lsb_notebook.NotebookData', side_effect=data.build_otf_data):
        exec(''.join(notebook['cells'][2]['source']), ns)
    otf = ns['otf_data']
    # Everything that supplies data/models to fit comes from cvdl.
    data.load_labels(otf_data=otf)
    data.otf_pair_tables(otf)
    loaders = data.build_otf_loaders(otf, batch_size=ns['BATCH_SIZE'], workers=ns['OTF_WORKERS'],
        eval_workers=ns['EVAL_WORKERS'], pin_memory=False, prefetch=ns['OTF_PREFETCH'],
        prefetch_images=ns['OTF_PREFETCH_IMAGES'], eval_persistent_workers=ns['EVAL_PERSISTENT_WORKERS'],
        prime_next_epoch=ns['PRIME_NEXT_EPOCH'])
    clean, stego, _ = otf.fixed_train[0]
    ns['clean_batch'], ns['stego_batch'] = clean.float()[None], stego.float()[None]
    next(iter(loaders.val))  # notebook cell 12's dtype check consumes a loader seed
    otf.check_epochs()
    ns.update(XuNet=models.XuNet, XuNetWide=models.XuNetWide)
    count_model = models.XuNet(clip_t=ns['CLIP_T'], norm=ns['NORM'])
    count_model(ns['clean_batch'][:1])
    exec(''.join(notebook['cells'][16]['source']), ns)
    return ns, otf, loaders


def make_trainer(folder, ns, otf, loaders, *, epochs, limit):
    logger = cb.AppendCSVLogger(save_dir=str(folder / 'logs'), name='csv', version='run', flush_logs_every_n_steps=1)
    loggers = [logger]
    try:
        import tensorboard
    except ImportError:
        tensorboard = None
    timing = cb.EpochTiming(folder / 'epoch_timing.csv')
    curves = cb.TrainingCurves(logger, folder, timing)
    scalar = None
    if tensorboard is not None:
        loggers.append(TensorBoardLogger(save_dir=str(folder / 'logs'), name='tb', version='run'))
        scalar = cb.PerEpochScalars(folder / 'logs/tb', timing, capped_validation=otf.checkpoint_validation == 'capped')
    checkpoint = cb.TimedModelCheckpoint(timing=timing, dirpath=str(folder),
        filename='best-epoch{epoch:02d}-val_pe{val_pe:.4f}-val_auc{val_auc:.4f}',
        auto_insert_metric_name=False, monitor='val_pe', mode='min', save_top_k=1, save_last=True,
        every_n_epochs=1, save_on_train_epoch_end=False)
    callbacks = [timing, curves, checkpoint,
        cb.V4LRTrace(folder / 'lr_trace.csv', len(loaders.train), ns['EPOCHS'], ns['WARMUP_EPOCHS'], ns['LR_DROPS'], timing)]
    if scalar is not None:
        callbacks.append(scalar)
    callbacks.append(cb.OTF_Epoch(otf, loaders.train, timing, prime_next_epoch=ns['PRIME_NEXT_EPOCH']))
    trainer = L.Trainer(max_epochs=epochs, limit_train_batches=limit, precision=ns['PRECISION'],
        gradient_clip_val=ns['GRAD_CLIP'] or None, gradient_clip_algorithm='norm' if ns['GRAD_CLIP'] else None,
        accelerator='cpu', devices=1, callbacks=callbacks, logger=loggers,
        default_root_dir=str(folder.parent), log_every_n_steps=50, enable_progress_bar=False, deterministic=False)
    return trainer, logger, checkpoint


class PackageTrainingTests(unittest.TestCase):
    @unittest.skipUnless((ROOT / 'Datasets/originals/raise/NEF').is_dir(), 'real photos not present')
    def test_cpu_checkpoint_resume_preserves_metrics(self):
        env = read_env_file(ROOT / 'tests/golden/cpu-det.env')
        before_threads = torch.get_num_threads()
        with tempfile.TemporaryDirectory() as temp, patch.dict(os.environ, env), patch('torch.cuda.is_available', return_value=False):
            folder = Path(temp)
            loaders = None
            try:
                with contextlib.redirect_stdout(io.StringIO()):
                    ns, otf, loaders = prepare_data_and_rng()
                    # Two batches finish epoch 0, including its validation and checkpoint.
                    trainer, logger, checkpoint = make_trainer(folder, ns, otf, loaders, epochs=1, limit=2)
                    model = SteganalysisModule(lr=ns['LR'], max_epochs=ns['EPOCHS'], paired_batches=ns['PAIRED_BATCHES'],
                        clip_t=ns['CLIP_T'], norm=ns['NORM'], steps_per_epoch=len(loaders.train),
                        warmup_epochs=ns['WARMUP_EPOCHS'], grad_clip=ns['GRAD_CLIP'], lr_drops=ns['LR_DROPS'],
                        batch_size=ns['BATCH_SIZE'], precision=ns['PRECISION'], model=ns['MODEL'], width=ns['WIDTH'],
                        validation_data=otf)
                    trainer.fit(model, loaders.train, loaders.val)
                    logger.save()
                metrics_path = folder / 'logs/csv/run/metrics.csv'
                last = folder / 'last.ckpt'
                self.assertEqual(trainer.global_step, 2)
                saved = ckpt_structure(last)
                self.assertEqual((saved['global_step'], saved['epoch']), (2, 0))
                self.assertTrue(Path(checkpoint.best_model_path).is_file())
                self.assertEqual(select_best_checkpoint(checkpoint.best_model_path, folder), checkpoint.best_model_path)
                metrics = read_csv_rows(metrics_path)
                old_rows = metrics['rows']
                val_column = metrics['kept_columns'].index('val_pe')
                self.assertTrue(any(row[val_column] is not None for row in old_rows),
                                'metrics prefix must include end-of-epoch validation')
                # Actually resume optimizer/callback/epoch state, not just a model load.
                with contextlib.redirect_stdout(io.StringIO()):
                    resumed = SteganalysisModule.load_from_checkpoint(last, map_location='cpu', strict=True,
                                                                     weights_only=False, validation_data=otf)
                    trainer2, logger2, _ = make_trainer(folder, ns, otf, loaders, epochs=2, limit=2)
                    trainer2.fit(resumed, loaders.train, loaders.val, ckpt_path=str(last))
                    logger2.save()
                self.assertEqual(trainer2.global_step, 4)
                self.assertEqual(ckpt_structure(last)['global_step'], 4)
                new_rows = read_csv_rows(metrics_path)['rows']
                self.assertGreater(len(new_rows), len(old_rows))
                self.assertEqual(new_rows[:len(old_rows)], old_rows)
            finally:
                if loaders is not None:
                    loaders.train.shutdown()
                torch.set_num_threads(before_threads)

    def test_capped_validation_runtime_dependency(self):
        def sample(sanity, names):
            logged = {}
            obj = SimpleNamespace(validation_data=SimpleNamespace(checkpoint_validation='capped'),
                trainer=SimpleNamespace(sanity_checking=sanity),
                validation_scores=[torch.tensor([.1] * len(names) + [.9] * len(names))],
                validation_labels=[torch.tensor([0] * len(names) + [1] * len(names))],
                validation_sources=[np.array(names * 2)], validation_losses=[torch.ones(2 * len(names))],
                log=lambda key, value, **kw: logged.update({key: value}))
            return obj, logged
        partial, logged = sample(True, ['raise'])
        SteganalysisModule.on_validation_epoch_end(partial)
        self.assertFalse(any(key.startswith('val_pe_') for key in logged))
        partial, _ = sample(False, ['raise'])
        with self.assertRaisesRegex(ValueError, 'no alaska examples'):
            SteganalysisModule.on_validation_epoch_end(partial)
        complete, logged = sample(False, ['raise', 'alaska', 'lsdir'])
        SteganalysisModule.on_validation_epoch_end(complete)
        self.assertEqual({key for key in logged if key.startswith(('val_pe_', 'val_loss_'))},
                         {f'val_{metric}_{name}' for name in ('raise', 'alaska', 'lsdir') for metric in ('pe', 'loss')})

    def test_four_old_checkpoints_load_strict_if_present(self):
        found = 0
        for name, pattern in CHECKPOINTS.items():
            for path in glob.glob(str(ROOT / pattern)):
                found += 1
                with self.subTest(checkpoint=name):
                    model = SteganalysisModule.load_from_checkpoint(path, map_location='cpu', strict=True, weights_only=False)
                    saved = torch.load(path, map_location='cpu', weights_only=False)
                    self.assertEqual(list(model.state_dict()), list(saved['state_dict']))
                    for key, value in model.state_dict().items():
                        self.assertTrue(torch.equal(value, saved['state_dict'][key]))
        if not found:
            self.skipTest('no local real checkpoints')


if __name__ == '__main__':
    unittest.main()
