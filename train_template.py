"""Training template: saves after every epoch and resumes by itself.

Copy this file, edit the two EDIT sections, and run:
    uv run python my_train.py --run-name my-run --epochs 20

If the run stops (Spot VM preemption, crash, Ctrl+C), run the SAME command again.
It continues from checkpoints/<run-name>/last.ckpt. Use a new --run-name to start fresh.
"""

import argparse

import lightning as L
import torch
from lightning.pytorch.callbacks import ModelCheckpoint
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


# ---- EDIT 1: the model, the loss, and the optimizer -------------------------
class Model(L.LightningModule):
    def __init__(self, lr: float = 1e-3):
        super().__init__()
        self.save_hyperparameters()
        self.net = nn.Sequential(nn.Flatten(), nn.Linear(3 * 32 * 32, 256), nn.ReLU(), nn.Linear(256, 10))

    def training_step(self, batch, batch_idx):
        x, y = batch
        loss = nn.functional.cross_entropy(self.net(x), y)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self.net(x)
        self.log("val_loss", nn.functional.cross_entropy(logits, y), prog_bar=True)
        self.log("val_acc", (logits.argmax(1) == y).float().mean(), prog_bar=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)


# ---- EDIT 2: the data -------------------------------------------------------
def make_loaders(batch_size: int):
    # Random placeholder data. Replace with your real datasets.
    train = TensorDataset(torch.randn(2048, 3, 32, 32), torch.randint(0, 10, (2048,)))
    val = TensorDataset(torch.randn(512, 3, 32, 32), torch.randint(0, 10, (512,)))
    return (DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=2),
            DataLoader(val, batch_size=batch_size, num_workers=2))


# ---- Save and resume: do not edit below this line ---------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    L.seed_everything(42, workers=True)
    train_loader, val_loader = make_loaders(args.batch_size)
    checkpoint = ModelCheckpoint(
        dirpath=f"checkpoints/{args.run_name}",
        save_last=True,       # writes last.ckpt after every epoch
        save_top_k=1,         # also keeps the best epoch by val_loss
        monitor="val_loss",
    )
    trainer = L.Trainer(max_epochs=args.epochs, callbacks=[checkpoint], default_root_dir="checkpoints")
    # "last" loads checkpoints/<run-name>/last.ckpt if it exists; otherwise starts from scratch.
    trainer.fit(Model(), train_loader, val_loader, ckpt_path="last")


if __name__ == "__main__":
    main()
