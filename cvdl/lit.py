"""Lightning training module with checkpoint-compatible names and explicit dependencies."""
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
import lightning as L
from .models import XuNet, XuNetWide
from .metrics import probability_error, auc_numpy


def v4_lr_factor(step: int, steps_per_epoch: int, max_epochs: int, warmup_epochs: int, lr_drops=None) -> float:
    """Factor applied BEFORE optimizer step `step`; scheduler advances after each step."""
    assert steps_per_epoch > 0 and max_epochs > 0 and 0 <= warmup_epochs <= max_epochs
    warmup_steps = warmup_epochs * steps_per_epoch
    factor = 1.0 if not warmup_steps else 0.01 + 0.99 * min(step / warmup_steps, 1.0)
    for epoch, multiplier in (lr_drops if lr_drops is not None else ((max_epochs // 2 + 1, 0.1),)):
        if step >= (epoch - 1) * steps_per_epoch:
            factor *= multiplier
    return factor


class SteganalysisModule(L.LightningModule):
    """Lightning wrapper for paired or unpaired image batches."""

    def __init__(self, lr: float = 3e-4, max_epochs: int = 20, weight_decay: float = 1e-4, paired_batches: bool = False, clip_t=3.0, norm="none", steps_per_epoch: int = 1, warmup_epochs: int = 1, grad_clip: float = 1.0, lr_drops=None, batch_size: int = 32, precision: str = "32", model: str = "xunet", width: int = 4, front_end: str = "kv", *, validation_data=None, conv1_factory=None):
        super().__init__()
        if lr_drops is None:
            lr_drops = ((max_epochs // 2 + 1, 0.1),)
        # front_end is saved only when it is not the default, so KV checkpoints keep their old hyper_parameters.
        self.save_hyperparameters(ignore=["validation_data", "conv1_factory", *(["front_end"] if front_end == "kv" else [])])
        self.validation_data = validation_data
        self.net = (XuNetWide(clip_t=clip_t, norm=norm, width=width, front_end=front_end) if model == "xunet_wide"
                    else XuNet(clip_t=clip_t, norm=norm, front_end=front_end))
        if conv1_factory is not None:
            self.net.conv1 = conv1_factory(self.net.conv1)
        self.validation_scores = []
        self.validation_labels = []
        self.validation_sources = []
        self.validation_losses = []

    def pair_logits_targets(self, batch):
        clean, stego, _ = batch
        images = torch.cat([clean, stego], dim=0).float()
        targets = torch.cat([
            torch.zeros(clean.shape[0], dtype=torch.long, device=clean.device),
            torch.ones(stego.shape[0], dtype=torch.long, device=stego.device),
        ])
        return self.net(images), targets

    def training_logits_targets(self, batch):
        if self.hparams.paired_batches:
            return self.pair_logits_targets(batch)
        images, targets, _ = batch
        return self.net(images.float()), targets

    def training_step(self, batch, batch_idx):
        logits, targets = self.training_logits_targets(batch)
        loss = F.cross_entropy(logits, targets)
        accuracy = (logits.argmax(dim=1) == targets).float().mean()
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=targets.numel())
        self.log("train_acc", accuracy, on_step=False, on_epoch=True, prog_bar=True, batch_size=targets.numel())
        return {"loss": loss, "accuracy": accuracy.detach(), "count": targets.numel()}

    def on_validation_epoch_start(self):
        self.validation_scores.clear()
        self.validation_labels.clear()
        self.validation_sources.clear()
        self.validation_losses.clear()

    def validation_step(self, batch, batch_idx):
        logits, targets = self.pair_logits_targets(batch)
        loss = F.cross_entropy(logits, targets)
        accuracy = (logits.argmax(dim=1) == targets).float().mean()
        self.validation_scores.append(logits.softmax(dim=1)[:, 1].detach().cpu())
        self.validation_labels.append(targets.detach().cpu())
        if self.validation_data is not None and self.validation_data.checkpoint_validation in {"capped", "train_matched"}:
            batch_sources = self.validation_data.val.frame.iloc[batch[2].tolist()]["source"].to_numpy()
            self.validation_sources.append(np.concatenate((batch_sources, batch_sources)))
            self.validation_losses.append(F.cross_entropy(logits, targets, reduction="none").detach().cpu())
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=targets.numel())
        self.log("val_acc", accuracy, on_step=False, on_epoch=True, prog_bar=True, batch_size=targets.numel())

    def on_validation_epoch_end(self):
        scores = torch.cat(self.validation_scores).numpy()
        labels_array = torch.cat(self.validation_labels).numpy()
        self.log("val_pe", probability_error(labels_array, scores), prog_bar=True)
        self.log("val_auc", auc_numpy(labels_array, scores), prog_bar=True)
        clean_scores = scores[labels_array == 0]
        stego_scores = scores[labels_array == 1]
        self.log("val_fa", float((clean_scores >= 0.5).mean()), prog_bar=True)
        self.log("val_md", float((stego_scores < 0.5).mean()), prog_bar=True)
        # Lightning's two-batch sanity pass sees only the first source in full validation.
        # Keep combined sanity metrics, but require every source only in real epochs.
        if self.validation_data is not None and self.validation_data.checkpoint_validation in {"capped", "train_matched"} and not self.trainer.sanity_checking:
            sources = np.concatenate(self.validation_sources)
            losses = torch.cat(self.validation_losses).numpy()
            names = (tuple(self.validation_data.corpus.settings.sources)
                     if self.validation_data.checkpoint_validation == 'train_matched' else ("raise", "alaska", "lsdir"))
            for name in names:
                selected = sources == name
                if not selected.any():
                    raise ValueError(f"capped validation has no {name} examples")
                self.log(f"val_pe_{name}", probability_error(labels_array[selected], scores[selected]))
                self.log(f"val_loss_{name}", float(losses[selected].mean()))
        self.validation_scores.clear()
        self.validation_labels.clear()
        self.validation_sources.clear()
        self.validation_losses.clear()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lr_lambda=lambda step: v4_lr_factor(
                step, self.hparams.steps_per_epoch, self.hparams.max_epochs, self.hparams.warmup_epochs, self.hparams.lr_drops
            )
        )
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "step"}}


def select_best_checkpoint(callback_path, run_folder):
    import re
    if callback_path and Path(callback_path).is_file():
        return str(callback_path)
    candidates = sorted(path for path in Path(run_folder).glob("best-*.ckpt") if path.is_file())
    if not candidates:
        raise FileNotFoundError(f"No best checkpoint found in {run_folder}: callback path {callback_path!r} is missing and no best-*.ckpt files exist.")
    if len(candidates) == 1:
        selected = candidates[0]
    else:
        def score(path):
            match = re.search(r"-val_pe([0-9]+(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?)", path.name)
            if match is None:
                raise ValueError(f"Cannot select the lowest val_pe: {path.name} has no val_pe in its name.")
            return float(match.group(1)), path.name
        selected = min(candidates, key=score)
    print(f"Best checkpoint fallback: callback path missing; using {selected}")
    return str(selected)

