"""Checkpoint-compatible notebook callbacks with explicit runtime dependencies."""
from pathlib import Path
import csv
import time
import pandas as pd
import lightning as L
from lightning.pytorch.loggers import CSVLogger
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg


class AppendCSVLogger(CSVLogger):
    """Keep existing CSV rows when Lightning opens a fixed-version run."""

    @property
    def experiment(self):
        if self._experiment is None:
            path = Path(self.log_dir) / "metrics.csv"
            previous = path.read_bytes() if path.exists() else None
            writer = super().experiment  # Lightning normally removes an existing CSV here.
            if previous is not None:
                path.write_bytes(previous)
                writer.metrics_keys = next(csv.reader(previous.decode().splitlines()))
        return super().experiment


class TrainingCurves(L.Callback):
    """Draw with an Agg canvas only; do not change the notebook's inline backend."""

    def __init__(self, csv_logger, run_dir, timing):
        self.timing = timing
        self.csv_logger = csv_logger
        self.run_dir = Path(run_dir)
        self.history = {}

    def state_dict(self):
        return {"history": self.history}

    def load_state_dict(self, state_dict):
        self.history = state_dict.get("history", {})

    def _redraw(self, trainer, phase):
        self.csv_logger.save()
        path = Path(self.csv_logger.log_dir) / "metrics.csv"
        if path.exists():
            frame = pd.read_csv(path)
            for _, row in frame.iterrows():
                if pd.isna(row.get("epoch")):
                    continue
                epoch = int(row["epoch"])
                values = self.history.setdefault(epoch, {})
                values.update({key: float(value) for key, value in row.items()
                               if key not in {"epoch", "step"} and pd.notna(value)})
        if phase == "validation":
            values = self.history.setdefault(trainer.current_epoch, {})
            for key in ("val_loss", "val_acc", "val_pe", "val_auc", "val_fa", "val_md"):
                if key in trainer.callback_metrics:
                    values[key] = float(trainer.callback_metrics[key].detach().cpu())
        if not self.history:
            return
        frame = pd.DataFrame.from_dict(self.history, orient="index").sort_index()
        fig = Figure(figsize=(10, 7), dpi=100)
        FigureCanvasAgg(fig)
        axes = fig.subplots(2, 2).flat
        for ax, keys, title in zip(axes,
                [("train_loss", "val_loss"), ("train_acc", "val_acc"),
                 ("val_pe", "val_auc"), ("val_fa", "val_md")],
                ["Loss", "Accuracy at 0.5", "Validation P_E and AUC", "Validation errors at 0.5"]):
            for key in keys:
                if key in frame:
                    ax.plot(frame.index + 1, frame[key], marker="o", label=key)
            ax.set(title=title, xlabel="Epoch")
            ax.grid(alpha=0.2)
            if ax.lines:
                ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(self.run_dir / "training_curves.png")
        print(f"CURVES updated after {phase}, epoch={trainer.current_epoch}", flush=True)

    def redraw(self, trainer, phase):
        start = time.perf_counter()
        try:
            self._redraw(trainer, phase)
        finally:
            self.timing.add(trainer.current_epoch, "curves_s", time.perf_counter() - start)

    def on_validation_end(self, trainer, pl_module):
        if not trainer.sanity_checking:
            self.redraw(trainer, "validation")

    def on_train_epoch_end(self, trainer, pl_module):
        self.redraw(trainer, "train epoch")

    def on_train_end(self, trainer, pl_module):
        self.redraw(trainer, "train end")


class PerEpochScalars(L.Callback):
    """Use separate train/val runs so matching tags get different colours."""

    def __init__(self, log_root, timing, *, capped_validation=False, validation_sources=None):
        self.timing = timing
        self.capped_validation = capped_validation
        self.validation_sources = validation_sources
        from torch.utils.tensorboard import SummaryWriter
        self.train_writer = SummaryWriter(str(Path(log_root) / "train"))
        self.val_writer = SummaryWriter(str(Path(log_root) / "val"))
        self.last_step = -1
        self.loss_sum = None
        self.accuracy_sum = None
        self.count = 0

    def state_dict(self):
        return {"last_step": self.last_step}

    def load_state_dict(self, state_dict):
        self.last_step = state_dict.get("last_step", -1)

    def on_train_epoch_start(self, trainer, pl_module):
        self.loss_sum = None
        self.accuracy_sum = None
        self.count = 0

    def _collect_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        count = outputs["count"]
        loss = outputs["loss"].detach() * count
        accuracy = outputs["accuracy"].detach() * count
        self.loss_sum = loss if self.loss_sum is None else self.loss_sum + loss
        self.accuracy_sum = accuracy if self.accuracy_sum is None else self.accuracy_sum + accuracy
        self.count += count

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        start = time.perf_counter()
        try:
            self._collect_train_batch_end(trainer, pl_module, outputs, batch, batch_idx)
        finally:
            self.timing.add(trainer.current_epoch, "scalars_s", time.perf_counter() - start)

    def _write_validation_end(self, trainer, pl_module):
        if trainer.sanity_checking or trainer.global_step <= self.last_step:
            return
        step = trainer.global_step
        if self.count:
            self.train_writer.add_scalar("loss", float(self.loss_sum / self.count), step)
            self.train_writer.add_scalar("acc", float(self.accuracy_sum / self.count), step)
        for tag, key in [("loss", "val_loss"), ("acc", "val_acc"), ("pe", "val_pe"),
                         ("auc", "val_auc"), ("fa", "val_fa"), ("md", "val_md")]:
            self.val_writer.add_scalar(tag, float(trainer.callback_metrics[key]), step)
        if self.capped_validation or self.validation_sources is not None:
            for name in self.validation_sources if self.validation_sources is not None else ("raise", "alaska", "lsdir"):
                for tag in ("pe", "loss"):
                    key = f"val_{tag}_{name}"
                    self.val_writer.add_scalar(f"{tag}/{name}", float(trainer.callback_metrics[key]), step)
        self.train_writer.flush()
        self.val_writer.flush()
        self.last_step = step

    def on_validation_end(self, trainer, pl_module):
        if trainer.sanity_checking or trainer.global_step <= self.last_step:
            return
        start = time.perf_counter()
        try:
            self._write_validation_end(trainer, pl_module)
        finally:
            self.timing.add(trainer.current_epoch, "scalars_s", time.perf_counter() - start)

    def on_train_end(self, trainer, pl_module):
        self.train_writer.close()
        self.val_writer.close()


class V4LRTrace(L.Callback):
    """Save the LR actually used at schedule boundaries, including after resume."""
    def __init__(self, path, steps_per_epoch, epochs, warmup_epochs, lr_drops, timing):
        self.timing = timing
        self.path = Path(path)
        warm = steps_per_epoch * warmup_epochs
        boundaries = (0, warm, *(steps_per_epoch * (epoch - 1) for epoch, _ in lr_drops))
        self.samples = {step for boundary in boundaries for step in (boundary - 1, boundary, boundary + 1) if 0 <= step < epochs * steps_per_epoch}

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        step = trainer.global_step
        if step not in self.samples:
            return
        start = time.perf_counter()
        try:
            self._write_lr(trainer, step)
        finally:
            self.timing.add(trainer.current_epoch, "lr_trace_s", time.perf_counter() - start)

    def _write_lr(self, trainer, step):
        lr = trainer.optimizers[0].param_groups[0]["lr"]
        with self.path.open("a", encoding="utf-8") as output:
            if output.tell() == 0:
                output.write("step,epoch,lr\n")
            output.write(f"{step},{trainer.current_epoch},{lr:.12g}\n")
        print(f"V4 LR step={step} epoch={trainer.current_epoch} lr={lr:.12g}", flush=True)


class EpochTiming(L.Callback):
    """Measure hook intervals without changing the model or loaders."""
    fields = ("epoch", "train_first_wait_s", "train_steps_s", "train_end_to_val_start_s",
              "val_loader_setup_s", "val_first_wait_s", "val_compute_s", "val_last_to_end_s",
              "val_end_to_next_first_s", "curves_s", "scalars_s", "lr_trace_s",
              "prime_s", "checkpoint_s")

    def __init__(self, path):
        self.path = Path(path)
        self.rows = {}
        self.clock = {}
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            with self.path.open("w", newline="", encoding="utf-8") as output:
                csv.writer(output).writerow(self.fields)

    def add(self, epoch, field, elapsed):
        row = self.rows.setdefault(int(epoch), {"epoch": int(epoch)})
        row[field] = row.get(field, 0.0) + elapsed

    def mark(self, epoch, field):
        self.clock[(int(epoch), field)] = time.perf_counter()

    def difference(self, epoch, output, start):
        before = self.clock.get((int(epoch), start))
        if before is not None:
            self.add(epoch, output, time.perf_counter() - before)

    def finish(self, epoch):
        row = self.rows.pop(int(epoch), None)
        if row is None:
            return
        with self.path.open("a", newline="", encoding="utf-8") as output:
            csv.writer(output).writerow([row.get(field, "") if field == "epoch"
                else f'{row[field]:.6f}' if field in row else "" for field in self.fields])
        print(f'EPOCH TIMING {int(epoch)} train={row.get("train_steps_s", 0):.2f}s '
              f'gap={row.get("train_end_to_val_start_s", 0):.2f}s '
              f'val={row.get("val_compute_s", 0):.2f}s '
              f'post={row.get("val_end_to_next_first_s", 0):.2f}s', flush=True)

    def on_train_epoch_start(self, trainer, pl_module):
        self.last_epoch = trainer.current_epoch
        self.mark(trainer.current_epoch, "train_epoch_start")

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        epoch = trainer.current_epoch
        if batch_idx == 0:
            self.difference(epoch, "train_first_wait_s", "train_epoch_start")
            self.mark(epoch, "train_first")
            if epoch > 0:
                self.difference(epoch - 1, "val_end_to_next_first_s", "val_end")
                self.finish(epoch - 1)

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if batch_idx == trainer.num_training_batches - 1:
            epoch = trainer.current_epoch
            self.mark(epoch, "train_last")
            self.difference(epoch, "train_steps_s", "train_first")

    def on_validation_start(self, trainer, pl_module):
        if not trainer.sanity_checking:
            epoch = trainer.current_epoch
            self.difference(epoch, "train_end_to_val_start_s", "train_last")
            self.mark(epoch, "val_start")

    def on_validation_epoch_start(self, trainer, pl_module):
        if not trainer.sanity_checking:
            self.difference(trainer.current_epoch, "val_loader_setup_s", "val_start")
            self.mark(trainer.current_epoch, "val_epoch_start")

    def on_validation_batch_start(self, trainer, pl_module, batch, batch_idx, dataloader_idx=0):
        if not trainer.sanity_checking and batch_idx == 0:
            self.difference(trainer.current_epoch, "val_first_wait_s", "val_epoch_start")
            self.mark(trainer.current_epoch, "val_first")

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        if not trainer.sanity_checking and batch_idx == trainer.num_val_batches[0] - 1:
            self.mark(trainer.current_epoch, "val_last")
            self.difference(trainer.current_epoch, "val_compute_s", "val_first")

    def on_validation_end(self, trainer, pl_module):
        if not trainer.sanity_checking:
            epoch = trainer.current_epoch
            self.difference(epoch, "val_last_to_end_s", "val_last")
            self.mark(epoch, "val_end")

    def on_fit_end(self, trainer, pl_module):
        self.finish(self.last_epoch)


class TimedModelCheckpoint(L.pytorch.callbacks.ModelCheckpoint):
    def __init__(self, *args, timing, **kwargs):
        self.timing = timing
        super().__init__(*args, **kwargs)

    def on_validation_end(self, trainer, pl_module):
        super().on_validation_end(trainer, pl_module)
        # Lightning 2.6 saves last only when top-1 improves. Force a fresh
        # last.ckpt after every full epoch, including non-improving ones.
        if (not self._should_skip_saving_checkpoint(trainer)
                and not self._should_save_on_train_epoch_end(trainer)
                and self._last_global_step_saved != trainer.global_step):
            self._save_last_checkpoint(trainer, self._monitor_candidates(trainer))

    def _save_checkpoint(self, trainer, filepath):
        start = time.perf_counter()
        try:
            return super()._save_checkpoint(trainer, filepath)
        finally:
            self.timing.add(trainer.current_epoch, "checkpoint_s", time.perf_counter() - start)


class QuietHeadlessProgress(L.Callback):
    """Print one progress line per 45 s and one after each training epoch."""
    def __init__(self):
        self.last_time = time.perf_counter()
        self.last_step = 0

    def report(self, trainer, label):
        now = time.perf_counter()
        elapsed = max(now - self.last_time, 1e-9)
        step = trainer.global_step
        rate = (step - self.last_step) / elapsed
        print(f"TRAIN {label} epoch={trainer.current_epoch + 1}/{trainer.max_epochs} step={step} steps/s={rate:.2f} it/s={rate:.2f}", flush=True)
        self.last_time, self.last_step = now, step

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if time.perf_counter() - self.last_time >= 45:
            self.report(trainer, "progress")

    def on_validation_end(self, trainer, pl_module):
        if not trainer.sanity_checking:
            self.report(trainer, "epoch-end")


class OTF_Epoch(L.Callback):
    def __init__(self, data, train_loader, timing, *, prime_next_epoch=False):
        self.data = data
        self.train_loader = train_loader
        self.timing = timing
        self.prime_next_epoch = prime_next_epoch

    def on_validation_start(self, trainer, pl_module):
        if self.prime_next_epoch and not trainer.sanity_checking and trainer.current_epoch + 1 < trainer.max_epochs:
            start = time.perf_counter()
            try:
                self.train_loader.prime_next(int(trainer.current_epoch) + 1)
            finally:
                self.timing.add(trainer.current_epoch, "prime_s", time.perf_counter() - start)

    def on_fit_start(self, trainer, pl_module):
        self.data.train.set_epoch(int(trainer.current_epoch))
    def on_train_epoch_start(self, trainer, pl_module):
        self.data.train.set_epoch(int(trainer.current_epoch))
        print(f"OTF EPOCH START {trainer.current_epoch}", flush=True)
        if self.data.corpus.settings.source_mix is not None:
            _, repeats = self.data.corpus.train_tile_schedule(int(trainer.current_epoch))
            print(f"OTF SOURCE MIX epoch={trainer.current_epoch} target={self.data.corpus.settings.tiles_per_epoch} weights={self.data.corpus.settings.source_mix} repeated_tiles={repeats}", flush=True)
    def on_train_epoch_end(self, trainer, pl_module):
        # Set the next epoch before Lightning creates/prefetches its iterator.
        self.data.train.set_epoch(int(trainer.current_epoch)+1)

