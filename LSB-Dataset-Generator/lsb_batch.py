"""Batch orchestration shared by the desktop GUI and the command line."""
from __future__ import annotations

import bootstrap  # noqa: F401
import csv
import json
import logging
import threading
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Callable

import numpy as np
import yaml
from PIL import Image

from lsb_core import embed, load_rgb, payload_length

Progress = Callable[[dict], None]
LOGGER_NAME = "lsb.generator"
ARTIFACT_DIRS = ("clean", "stego", "masks")
METADATA_FIELDS = [
    "source_id", "source_file", "clean_file", "stego_file", "mask_file", "payload_rate",
]


@dataclass(frozen=True)
class Config:
    input_dir: str
    output_dir: str

    payload_rate: float = 0.4
    run_seed: int = 42

    payload_mode: str = "fixed"
    payload_min: float = 0.1
    payload_max: float = 0.9

    recursive: bool = True

    def validate(self) -> None:
        if not isinstance(self.input_dir, str) or not self.input_dir.strip():
            raise ValueError("Choose an input folder.")
        if not isinstance(self.output_dir, str) or not self.output_dir.strip():
            raise ValueError("Choose an output folder.")
        if type(self.run_seed) is not int or self.run_seed < 0:
            raise ValueError("Run seed must be a non-negative integer.")
        if self.payload_mode not in ("fixed", "range"):
            raise ValueError("Payload mode must be fixed or range.")
        for value in (self.payload_rate, self.payload_min, self.payload_max):
            if type(value) not in (float, int):
                raise ValueError("Payload rates must be numbers.")
            payload_length(value, 3)
        if self.payload_min > self.payload_max:
            raise ValueError("Minimum payload rate must not exceed maximum.")
        if type(self.recursive) is not bool:
            raise ValueError("Include subfolders must be true or false.")


def read_config(path: Path) -> Config:
    """Load a YAML configuration; relative folder paths resolve beside that file."""
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("Configuration must contain a YAML mapping.")
    data = dict(data)
    extras = set(data) - {field.name for field in fields(Config)}
    if extras:
        raise ValueError(f"Unknown configuration fields: {', '.join(sorted(extras))}")
    for key in ("input_dir", "output_dir"):
        if key in data and isinstance(data[key], str) and data[key].strip():
            data[key] = str((path.parent / data[key]).resolve())
    result = Config(**data)
    result.validate()
    return result


def write_config(path: Path, config: Config) -> None:
    config.validate()
    path.write_text(yaml.safe_dump(asdict(config), sort_keys=False), encoding="utf-8")


def generate(config: Config, progress: Progress | None = None,
             cancel: threading.Event | None = None) -> dict:
    """Generate a dataset for every valid source image found under the input folder.

    Invalid or failed sources are logged, counted as skipped and never stop the
    run. One run-level NumPy generator supplies the payload rates, payload bits
    and embedding locations, so the same input set, order, seed and
    configuration reproduce the same random sequence.
    """
    config.validate()
    source, target = Path(config.input_dir).resolve(), Path(config.output_dir).resolve()
    if not source.is_dir():
        raise ValueError("Input folder does not exist.")
    if target == source or target.is_relative_to(source) or source.is_relative_to(target):
        raise ValueError("Input and output folders must be separate, with neither inside the other.")
    if target.exists() and (not target.is_dir() or any(target.iterdir())):
        raise ValueError("Output folder must be new or empty. Existing output is never overwritten.")
    candidates = sorted((path for path in (source.rglob("*") if config.recursive else source.iterdir()) if path.is_file()),
                        key=lambda path: path.as_posix().casefold())
    target.mkdir(parents=True, exist_ok=True)
    config = Config(**{**asdict(config), "input_dir": str(source), "output_dir": str(target)})
    rng = np.random.default_rng(config.run_seed)
    summary = {
        "status": "running",
        "run_seed": config.run_seed,
        "payload_mode": config.payload_mode,
        "payload_min": config.payload_min,
        "payload_max": config.payload_max,
        "total_discovered": len(candidates),
        "total_generated": 0,
        "total_skipped": 0,
    }
    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    handler = logging.FileHandler(target / "generation.log", encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(handler)

    def notify(current: int, message: str) -> None:
        """Per-image GUI progress; not every successful image reaches the log file."""
        if progress:
            progress({"kind": "progress", "current": current, "total": len(candidates), "message": message})

    def report(message: str, current: int) -> None:
        logger.info(message)
        notify(current, message)

    try:
        for folder in ARTIFACT_DIRS:
            (target / folder).mkdir()
        write_config(target / "run_config.yaml", config)
        logger.info("Run started: %s source files discovered in %s", len(candidates), source)
        logger.info("Configuration: %s", asdict(config))
        if not candidates:
            logger.error("No source files were discovered; nothing was generated")
        notify(0, f"Generating from {len(candidates)} discovered source files")
        with (target / "metadata.csv").open("x", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=METADATA_FIELDS)
            writer.writeheader()
            for current, file in enumerate(candidates, 1):
                if cancel is not None and cancel.is_set():
                    summary["status"] = "cancelled"
                    break
                try:
                    if file.suffix.lower() != ".png":
                        raise ValueError("Unsupported extension; only PNG is accepted.")
                    clean = load_rgb(file)
                except Exception as exc:
                    summary["total_skipped"] += 1
                    report(f"Skipped {file.relative_to(source)}: {exc}", current)
                    continue
                source_id = f"{summary['total_generated'] + 1:06d}"
                if config.payload_mode == "fixed":
                    rate = float(config.payload_rate)
                else:
                    rate = float(rng.uniform(config.payload_min, config.payload_max))
                written: list[Path] = []
                try:
                    stego, mask, selected = embed(clean, rate, rng)
                    row = {
                        "source_id": source_id,
                        "source_file": file.relative_to(source).as_posix(),
                        "clean_file": f"clean/{source_id}.png",
                        "stego_file": f"stego/{source_id}.png",
                        "mask_file": f"masks/{source_id}.npy",
                        "payload_rate": rate,
                    }
                    for field, array in (("clean_file", clean), ("stego_file", stego),
                                         ("mask_file", mask)):
                        path = target / row[field]
                        with path.open("xb") as output:
                            written.append(path)
                            if path.suffix == ".png":
                                Image.fromarray(array).save(output, format="PNG", compress_level=6, optimize=False)
                            else:
                                np.save(output, array, allow_pickle=False)
                    writer.writerow(row)
                    stream.flush()
                except Exception as exc:
                    for path in written:
                        path.unlink(missing_ok=True)
                    summary["total_skipped"] += 1
                    report(f"Failed {file.relative_to(source)}: {exc}", current)
                    continue
                summary["total_generated"] += 1
                notify(current, f"{source_id}  {len(selected):,} locations selected, "
                                f"{int(mask.sum()):,} channels changed, rate {rate:.4f}")
        if summary["status"] == "running":
            summary["status"] = "completed" if summary["total_generated"] else "failed"
    except BaseException as exc:
        summary["status"] = "failed"
        summary["error"] = str(exc)
        logger.exception("Run failed")
        raise
    finally:
        if summary["status"] == "cancelled":
            logger.info("Run cancelled after %s generated and %s skipped samples",
                        summary["total_generated"], summary["total_skipped"])
        else:
            logger.info("Run %s: %s generated, %s skipped of %s discovered files",
                        summary["status"], summary["total_generated"], summary["total_skipped"],
                        summary["total_discovered"])
        try:
            (target / "run_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        except OSError:
            logger.exception("Could not write run_summary.json")
        handler.close()
        logger.removeHandler(handler)
    return summary
