"""Durable local checkpoint replacement shared by every Trainer checkpoint save."""
import os
from pathlib import Path
import tempfile
import torch
from lightning.fabric.plugins.io import TorchCheckpointIO


class DurableCheckpointIO(TorchCheckpointIO):
    def save_checkpoint(self, checkpoint, path, storage_options=None):
        if storage_options is not None:
            raise TypeError('local checkpoint storage_options are unsupported')
        if '://' in str(path):
            raise ValueError('durable checkpoints require a local path')
        path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
        fd,name=tempfile.mkstemp(prefix='.'+path.name+'.',suffix='.tmp',dir=path.parent)
        try:
            with os.fdopen(fd,'wb') as stream:
                torch.save(checkpoint,stream)
                stream.flush();os.fsync(stream.fileno())
            os.replace(name,path)
            directory=os.open(path.parent,os.O_RDONLY|os.O_DIRECTORY)
            try:os.fsync(directory)
            finally:os.close(directory)
        finally:
            Path(name).unlink(missing_ok=True)
