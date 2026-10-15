"""Small shared status, bounded log, and disk-space helpers for Pi jobs."""
from __future__ import annotations

import datetime as dt
import logging
import os
import shutil
from logging.handlers import RotatingFileHandler
from pathlib import Path

MIN_FREE = 50 * 1024**3


def setup(root: Path, name: str) -> logging.Logger:
    root.mkdir(parents=True, exist_ok=True)
    log = logging.getLogger(name)
    log.setLevel(logging.INFO)
    handler = RotatingFileHandler(root / f"{name}.log", maxBytes=5 * 1024**2, backupCount=3)
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    log.addHandler(handler)
    return log


def check_space(root: Path, incoming: int = 0) -> None:
    if shutil.disk_usage(root).free < MIN_FREE + incoming:
        raise RuntimeError(f"DISK_GUARD: need {incoming} incoming bytes plus 50 GiB reserve")


def status(root: Path, *, state: str, done: int, total: int, bytes_: int,
           failures: int, started: dt.datetime, note: str = "") -> None:
    now = dt.datetime.now(dt.timezone.utc)
    seconds = max((now - started).total_seconds(), 1)
    text = (f"state={state}\ndone={done}/{total}\nbytes={bytes_}\nfailures={failures}\n"
            f"last_update={now.isoformat()}\nrate_MB_s={bytes_ / 1e6 / seconds:.3f}\n"
            f"note={note}\n")
    temp = root / "STATUS.tmp"
    temp.write_text(text, encoding="utf-8")
    os.replace(temp, root / "STATUS")


def summary(root: Path, *, done: int, total: int, bytes_: int, failures: int,
            started: dt.datetime) -> None:
    now = dt.datetime.now(dt.timezone.utc)
    seconds = max((now - started).total_seconds(), 1)
    (root / "SUMMARY").write_text(
        f"finished={now.isoformat()}\ndone={done}/{total}\nbytes={bytes_}\n"
        f"failures={failures}\nwall_s={seconds:.1f}\nrate_MB_s={bytes_ / 1e6 / seconds:.3f}\n",
        encoding="utf-8",
    )
