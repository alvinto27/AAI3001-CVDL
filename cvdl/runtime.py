"""Explicit wiring for the shared training notebook; no environment readers.

The count-model and KV probe are retained to preserve the historical RNG stream.
Experiment choices belong to Settings, not to this module.
"""
from pathlib import Path
import json
import random
import time
from types import SimpleNamespace

import lightning as L
from lightning.pytorch.loggers import TensorBoardLogger
import numpy as np
import torch
from torch.nn import functional as F

from . import callbacks as cb, data, models
from .lit import SteganalysisModule, select_best_checkpoint


def configure(settings):
    """Apply machine limits/backend choice, then seed in the original order."""
    torch.set_num_threads(settings.main_threads)
    torch.backends.cudnn.benchmark = torch.version.cuda is not None
    random.seed(settings.seed)
    np.random.seed(settings.seed)
    torch.manual_seed(settings.seed)
    L.seed_everything(settings.seed, workers=True)


def conv1_factory(settings):
    if settings.rocm_conv1_fix and settings.rocm and settings.device == 'cuda':
        from notebooks.rocm_conv1_wgrad import UnfoldWgradConv2d
        return UnfoldWgradConv2d.from_conv
    return None


def create_data(s):
    """Build frozen split tables; do not advance a loader or model RNG here."""
    if s.data == 'otf':
        pilot = getattr(s, 'pilot', False)
        otf = data.build_otf_data(s.root, s.to_run_settings(), quick=False if pilot else s.quick,
            profile=s.eval_profile, eval_pairs=s.quick_eval_pairs,
            checkpoint_validation=s.checkpoint_validation,
            val_cap_raise=s.val_cap_raise, val_cap_lsdir=s.val_cap_lsdir, train_grid=s.train_grid,
            train_tile=s.train_tile_size,
            **({'validation_recipe':s.validation_recipe} if s.checkpoint_validation=='train_matched' else {}))
        if pilot:
            from .pilot import apply
            apply(otf, s)
        if getattr(s,'embedding_schedule',None) is not None:
            from .curriculum import CurriculumImages
            otf.train=CurriculumImages(otf.corpus,s.embedding_schedule,s.embedding_stage_epochs)
            otf.record['embedding_schedule']=dict(mode=s.embedding_schedule,stage_epochs=s.embedding_stage_epochs,
                original_fraction=.25 if s.embedding_schedule=='low_bpp' else 1.,assignment='exact per epoch before worker sharding; counts reflect consumed stegos')
        otf.record.update(compile_mode=s.compile, prime_next_epoch=s.prime_next_epoch)
        return otf, data.otf_pair_tables(otf)
    labels = data.load_labels(labels_path=s.labels_path)
    caps = {'train':s.quick_train_pairs, 'val':s.quick_eval_pairs, 'test':s.quick_eval_pairs}
    return None, data.png_pair_tables(labels, s.split_path, s.seed, quick_pairs=caps if s.quick else None)


def create_loaders(s, otf, tables):
    """Build loaders and retain the historical preview/epoch-check seed draws."""
    kwargs = dict(batch_size=s.batch_size, workers=s.workers, eval_workers=s.eval_workers,
        pin_memory=s.device == 'cuda', eval_persistent_workers=s.persistent_workers)
    if otf is not None:
        loaders = data.build_otf_loaders(otf, **kwargs, prefetch=s.prefetch,
            prefetch_images=s.prefetch_images, prime_next_epoch=s.prime_next_epoch,
            pack_blocks=s.pack_blocks)
        clean, stego, _ = otf.fixed_train[0]
        byte_images, _, _ = next(iter(loaders.val))
        assert byte_images.dtype == torch.uint8
        assert torch.equal(byte_images.to(s.device).float(), byte_images.float().to(s.device))
        if s.post_fit_eval == 'full':
            otf.check_epochs()
        otf.print_record()
    else:
        loaders = data.build_png_loaders(tables.run_pairs, s.data_root, **kwargs,
            paired_batches=s.paired_batches, seed=s.seed)
        clean, stego, _ = data.PairDataset(loaders.train_frame.head(1), s.data_root)[0]
        next(iter(loaders.train))  # Historical notebook preview consumes a shuffle seed.
    return loaders, clean.float()[None], stego.float()[None]


def model_checks(s, clean_batch, stego_batch):
    """Count parameters and check frozen KV weights, preserving both allocations."""
    def model():
        return models.model_from_hparams(dict(model=s.model, clip_t=s.clip_t, norm=s.norm,
                                             width=s.width, front_end=s.front_end))
    counted = model()
    print(f'Total parameters: {sum(p.numel() for p in counted.parameters()):,}')
    print(f'Output shape for one batch: {tuple(counted(clean_batch[:1]).shape)}')
    probe = model().to(s.device).train()
    optimizer = torch.optim.Adam(probe.parameters(), lr=1e-3)
    clean, stego = clean_batch.to(s.device), stego_batch.to(s.device)
    images = torch.cat([clean, stego], dim=0)
    targets = torch.cat([torch.zeros(len(clean), dtype=torch.long, device=s.device),
                         torch.ones(len(stego), dtype=torch.long, device=s.device)])
    before = probe.kv.conv.weight.detach().clone()
    optimizer.zero_grad(set_to_none=True)
    loss = F.cross_entropy(probe(images), targets)
    loss.backward()
    optimizer.step()
    assert torch.equal(before, probe.kv.conv.weight.detach())
    print(f'Probe loss: {loss.item():.5f}; KV weights unchanged: True')


def finish_training(s, bundle, loaders):
    """Release train workers and check the one-best-plus-final-last policy."""
    train_loader = loaders.train
    if s.data == 'otf':
        train_loader.shutdown()
        train_loader.loader._iterator = None
    else:
        iterator = getattr(train_loader, '_iterator', None)
        if iterator is not None:
            iterator._shutdown_workers()
            train_loader._iterator = None
    checkpoint = bundle.checkpoint_callback
    best = select_best_checkpoint(checkpoint.best_model_path, bundle.checkpoint_dir)
    saved = torch.load(checkpoint.last_model_path, map_location='cpu', weights_only=False)
    assert saved['epoch'] in {s.epochs - 1, s.epochs}
    assert saved['global_step'] == s.epochs * min(len(train_loader), s.limit_train_batches or len(train_loader))
    assert len(list(bundle.checkpoint_dir.glob('best-*.ckpt'))) == 1
    assert not list(bundle.checkpoint_dir.glob('epoch*.ckpt'))
    bundle.curves_callback.redraw(bundle.trainer, 'final')
    print(f"CHECKPOINT CHECK last epoch={saved['epoch']}, global_step={saved['global_step']}")
    print(f'Best checkpoint: {best}')
    return str(checkpoint.last_model_path) if getattr(s,'evaluation_checkpoint','best')=='last' else best


def create_training(s, otf_data, loaders):
    """Wire checkpoint-compatible callbacks, loggers, Trainer and module."""
    train_loader, val_loader = loaders.train, loaders.val
    checkpoint_dir = Path(s.output_root) / s.run_directory_name
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    starting=getattr(s,'init_checkpoint',None) is not None
    resuming=False; prior_model={}
    if starting:
        from .curriculum import check_start
        resuming=check_start(s,checkpoint_dir)
        if resuming:prior_model=json.loads((checkpoint_dir/'model_run_settings.json').read_text())
    model_run_settings = {"compile_mode": s.compile,
                          "compile_target": "net" if s.compile == "default" else None,
                          "compile_dynamic": False if s.compile == "default" else None,
                          "compile_fullgraph": False if s.compile == "default" else None,
                          "tf32_matmul": torch.backends.cuda.matmul.allow_tf32,
                          "tf32_cudnn": torch.backends.cudnn.allow_tf32,
                          "torch_version": str(torch.__version__),
                          "norm": s.norm, "clip_t": s.clip_t, "lr": s.lr,
                          "max_epochs": s.epochs, "warmup_epochs": s.warmup_epochs,
                          "steps_per_epoch": len(train_loader), "grad_clip_norm": s.grad_clip,
                          "weight_decay": 1e-4, "lr_drops": s.lr_drops,
                          "batch_size": s.batch_size, "precision": s.precision, "model": s.model, "width": s.width, "persistent_validation_workers": s.persistent_workers,
                          "train_yaml": s.train_yaml, "extra_tests": s.extra_tests}
    (checkpoint_dir / "model_run_settings.json").write_text(json.dumps(model_run_settings, indent=2) + "\n")
    print("Model run settings:", json.dumps(model_run_settings, indent=2))
    if s.data == "otf":
        otf_data.save_record(checkpoint_dir / "data_run_settings.json")
        otf_data.corpus.save_training_plan(checkpoint_dir)
        if otf_data.checkpoint_validation in {"capped", "train_matched"}:
            validation_file = checkpoint_dir / "validation_pairs.csv"
            if validation_file.exists() and validation_file.read_text() != otf_data.validation_pair_csv:
                raise ValueError("validation pair IDs changed in an existing run")
            validation_file.write_text(otf_data.validation_pair_csv)
    csv_logger = cb.AppendCSVLogger(save_dir=str(checkpoint_dir / "logs"), name="csv", version="run", flush_logs_every_n_steps=1)
    run_loggers = [csv_logger]
    try:
        import tensorboard
    except ImportError:
        print("WARNING: TensorBoard is unavailable; using CSV logging only.")
    else:
        tb_logger = TensorBoardLogger(save_dir=str(checkpoint_dir / "logs"), name="tb", version="run")
        run_loggers.append(tb_logger)
        custom_layout = {"Training": {
            "Loss": ["Multiline", ["^loss$"]],
            "Accuracy": ["Multiline", ["^acc$"]],
            "Errors at 0.5": ["Multiline", ["^fa$", "^md$"]],
            "Detection error P_E": ["Multiline", ["^pe$"]],
            "AUC": ["Multiline", ["^auc$"]],
        }}
        layout_marker = Path(tb_logger.log_dir) / "custom_scalars_layout.json"
        layout_text = json.dumps(custom_layout, sort_keys=True)
        if not layout_marker.exists() or layout_marker.read_text() != layout_text:
            tb_logger.experiment.add_custom_scalars(custom_layout)
            tb_logger.experiment.flush()
            layout_marker.write_text(layout_text)
    epoch_timing_callback = cb.EpochTiming(checkpoint_dir / "epoch_timing.csv")
    curves_callback = cb.TrainingCurves(csv_logger, checkpoint_dir, epoch_timing_callback)
    matched_sources = ({'validation_sources':tuple(otf_data.corpus.settings.sources)}
                       if otf_data is not None and otf_data.checkpoint_validation == 'train_matched' else {})
    per_epoch_callback = cb.PerEpochScalars(checkpoint_dir / "logs" / "tb", epoch_timing_callback,
        capped_validation=otf_data is not None and otf_data.checkpoint_validation == "capped",
        **matched_sources) if len(run_loggers) > 1 else None
    checkpoint_callback = cb.TimedModelCheckpoint(
        timing=epoch_timing_callback,
        dirpath=str(checkpoint_dir),
        filename="best-epoch{epoch:02d}-val_pe{val_pe:.4f}-val_auc{val_auc:.4f}",
        auto_insert_metric_name=False, monitor="val_pe", mode="min", save_top_k=1, save_last=True,
        every_n_epochs=1, save_on_train_epoch_end=False,
    )
    trainer = L.Trainer(
        max_epochs=s.epochs,
        **({"limit_train_batches": s.limit_train_batches} if s.limit_train_batches else {}),
        precision=s.precision,
        gradient_clip_val=s.grad_clip or None, gradient_clip_algorithm="norm" if s.grad_clip else None,
        accelerator="gpu" if torch.device(s.device).type == "cuda" else "cpu",
        devices=1,
        callbacks=[epoch_timing_callback, curves_callback, checkpoint_callback, cb.V4LRTrace(checkpoint_dir / "lr_trace.csv", len(train_loader), s.epochs, s.warmup_epochs, s.lr_drops, epoch_timing_callback)] + ([per_epoch_callback] if per_epoch_callback else []) + ([cb.OTF_Epoch(otf_data, train_loader, epoch_timing_callback, prime_next_epoch=s.prime_next_epoch)] if s.data == "otf" else []) + ([cb.QuietHeadlessProgress()] if s.headless else []) + curriculum_callbacks(s,checkpoint_dir),
        logger=run_loggers,
        default_root_dir=str(Path(s.output_root)),
        log_every_n_steps=50,
        enable_progress_bar=not s.headless,
        deterministic=False,
    )
    training_started = time.perf_counter()
    assert trainer.checkpoint_callbacks == [checkpoint_callback], "only one best-and-last checkpoint callback is allowed"
    lightning_model = SteganalysisModule(clip_t=s.clip_t, norm=s.norm, validation_data=otf_data, conv1_factory=conv1_factory(s), lr=s.lr, max_epochs=s.epochs, paired_batches=s.paired_batches, steps_per_epoch=len(train_loader), warmup_epochs=s.warmup_epochs, grad_clip=s.grad_clip, lr_drops=s.lr_drops, batch_size=s.batch_size, precision=s.precision, model=s.model, width=s.width, front_end=s.front_end)
    if starting:
        from .curriculum import initialize
        initialization=initialize(s,lightning_model,checkpoint_dir,resuming)
        model_run_settings['initialization_checkpoint_sha256']=initialization['checkpoint_sha256']
        model_run_settings['evaluation_checkpoint']=getattr(s,'evaluation_checkpoint','best')
    if s.name:
        import hashlib
        digest=hashlib.sha256()
        for name,tensor in lightning_model.net.state_dict().items():
            digest.update(name.encode());digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
        model_run_settings['initial_weights_sha256']=prior_model['initial_weights_sha256'] if resuming else digest.hexdigest()
        (checkpoint_dir/'model_run_settings.json').write_text(json.dumps(model_run_settings,indent=2)+'\n')
    if s.compile == "default":
        # In place, not torch.compile(net): no _orig_mod prefix in checkpoints.
        # Do not compile in __init__: checkpoint reload and deepcopy diagnostics stay eager.
        lightning_model.net.compile(mode="default", dynamic=False, fullgraph=False)
    s.write(checkpoint_dir)
    return SimpleNamespace(trainer=trainer, lightning_model=lightning_model, checkpoint_dir=checkpoint_dir,
        checkpoint_callback=checkpoint_callback, curves_callback=curves_callback)


def curriculum_callbacks(s,folder):
    if getattr(s,'embedding_schedule',None) is None:return []
    from .curriculum import CurriculumCounts
    return [CurriculumCounts(folder/'embedding_epoch_counts.jsonl',s.embedding_schedule,s.embedding_stage_epochs)]
