"""Batch orchestration and saved-artifact verification shared by CLI and GUI."""
from __future__ import annotations

import bootstrap  # noqa: F401
import csv
import json
import logging
import platform
import threading
import uuid
from dataclasses import asdict, dataclass, fields
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import numpy as np
import PIL
import yaml
from PIL import Image

from lsb_core import VERSION, derive_seed, embed, load_rgb, payload_length, source_identity, verify_arrays

Progress = Callable[[dict], None]
METADATA_FIELDS = [
    "sample_id", "source_id", "label", "clean_file", "stego_file", "mask_file",
    "selected_locations_file", "generator_version", "payload_rate", "payload_length_bits",
    "random_seed", "width", "height", "channels", "input_format", "output_format",
    "source_file", "source_dataset", "conversion_applied", "selected_location_count",
    "actual_change_count", "zero_change", "numpy_version", "pillow_version",
]


@dataclass(frozen=True)
class Config:
    input_dir: str
    output_dir: str
    payload_rate: float = 0.4
    run_seed: int = 42
    source_dataset: str = ""
    payload_mode: str = "fixed"
    payload_min: float = 0.1
    payload_max: float = 0.5
    recursive: bool = True
    error_policy: str = "skip"

    def validate(self) -> None:
        if not isinstance(self.input_dir, str) or not self.input_dir.strip():
            raise ValueError("Choose an input folder.")
        if not isinstance(self.output_dir, str) or not self.output_dir.strip():
            raise ValueError("Choose an output folder.")
        if type(self.run_seed) is not int or not 0 <= self.run_seed < 2 ** 128:
            raise ValueError("Run seed must be an integer from 0 to 2^128 - 1.")
        if self.payload_mode not in ("fixed", "range"):
            raise ValueError("Payload mode must be fixed or range.")
        for value in (self.payload_rate, self.payload_min, self.payload_max):
            if type(value) not in (float, int):
                raise ValueError("Payload rates must be numbers.")
            payload_length(value, 3)
        if self.payload_min > self.payload_max:
            raise ValueError("Minimum payload rate must not exceed maximum.")
        if self.error_policy not in ("skip", "stop"):
            raise ValueError("Error policy must be skip or stop.")
        if type(self.recursive) is not bool or not isinstance(self.source_dataset, str):
            raise ValueError("Invalid recursive setting or dataset name.")


def read_config(path: Path) -> Config:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("Configuration must contain a YAML mapping.")
    data = dict(data)
    if "random" in data:
        random = data.pop("random")
        if not isinstance(random, dict) or set(random) != {"run_seed"} or "run_seed" in data:
            raise ValueError("Use one run_seed, either at top level or inside random.")
        data["run_seed"] = random["run_seed"]
    allowed = {field.name for field in fields(Config)}
    extras = set(data) - allowed
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


def rate_for(config: Config, identity: str) -> float:
    if config.payload_mode == "fixed":
        return float(config.payload_rate)
    if config.payload_min == config.payload_max:
        return float(config.payload_min)
    rng = np.random.Generator(np.random.PCG64(derive_seed(config.run_seed, identity, "rate")))
    return float(rng.uniform(config.payload_min, config.payload_max))


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _contained(root: Path, relative: str) -> Path:
    if not relative or Path(relative).is_absolute():
        raise ValueError("Artifact paths must be relative to the run folder.")
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError("Artifact path escapes the run folder.")
    return path


def verify_record(root: Path, row: dict, config: Config) -> None:
    if row["generator_version"] != VERSION:
        raise ValueError("Generator version mismatch; use the original generator version.")
    if row["numpy_version"] != np.__version__:
        raise ValueError("NumPy version mismatch; use the recorded dependency version.")
    if row["label"] != "stego" or row["channels"] != "RGB" or row["input_format"] != "PNG" or row["output_format"] != "PNG":
        raise ValueError("Unexpected sample label or representation metadata.")
    if str(row["conversion_applied"]).lower() != "false":
        raise ValueError("Conversion is unsupported by this generator.")
    clean = load_rgb(_contained(root, row["clean_file"]))
    stego = load_rgb(_contained(root, row["stego_file"]))
    mask = np.load(_contained(root, row["mask_file"]), allow_pickle=False)
    selected = np.load(_contained(root, row["selected_locations_file"]), allow_pickle=False)
    identity = source_identity(clean)
    seed = int(row["random_seed"])
    rate = float(row["payload_rate"])
    if identity != row["source_id"] or seed != derive_seed(config.run_seed, identity):
        raise ValueError("Source identity or derived seed mismatch.")
    if rate != rate_for(config, identity) or row["source_dataset"] != config.source_dataset:
        raise ValueError("Sample configuration does not match effective run configuration.")
    if int(row["width"]) != clean.shape[1] or int(row["height"]) != clean.shape[0]:
        raise ValueError("Metadata dimensions mismatch.")
    verify_arrays(clean, stego, mask, selected, rate, seed)
    if int(row["payload_length_bits"]) != len(selected) or int(row["selected_location_count"]) != len(selected):
        raise ValueError("Metadata selected/payload count mismatch.")
    if int(row["actual_change_count"]) != int(mask.sum()):
        raise ValueError("Metadata actual-change count mismatch.")
    if str(row["zero_change"]).lower() != str(not mask.any()).lower():
        raise ValueError("Zero-change flag mismatch.")


def verify_run(root: Path, progress: Progress | None = None,
               cancel: threading.Event | None = None) -> dict:
    config = read_config(root / "run_config.yaml")
    summary = json.loads((root / "run_summary.json").read_text(encoding="utf-8"))
    with (root / "metadata.csv").open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != METADATA_FIELDS:
            raise ValueError("Metadata schema mismatch.")
        rows = list(reader)
    if not rows:
        raise ValueError("This run has no generated samples to verify.")
    if summary["generator_version"] != VERSION or summary["total_generated"] != len(rows):
        raise ValueError("Summary version/count mismatch.")
    identities, sample_ids, paths = set(), set(), set()
    selected_total = changed_total = 0
    for i, row in enumerate(rows, 1):
        if cancel is not None and cancel.is_set():
            return {"status": "cancelled", "verified": i - 1, "total": len(rows)}
        if row["source_id"] in identities or row["sample_id"] in sample_ids:
            raise ValueError("Duplicate metadata identity/sample ID.")
        identities.add(row["source_id"])
        sample_ids.add(row["sample_id"])
        for field in ("clean_file", "stego_file", "mask_file", "selected_locations_file"):
            path = _contained(root, row[field])
            if path in paths:
                raise ValueError("More than one artifact refers to the same file.")
            paths.add(path)
        verify_record(root, row, config)
        selected_total += int(row["selected_location_count"])
        changed_total += int(row["actual_change_count"])
        if progress:
            progress({"kind": "progress", "current": i, "total": len(rows), "message": f"Verified {i}/{len(rows)}"})
    if selected_total != summary["total_selected_locations"] or changed_total != summary["total_actual_changes"]:
        raise ValueError("Summary change/selected totals mismatch.")
    return {"status": "verified", "verified": len(rows), "original_run_status": summary["status"]}


def generate(config: Config, progress: Progress | None = None,
             cancel: threading.Event | None = None) -> dict:
    config.validate()
    source, target = Path(config.input_dir).resolve(), Path(config.output_dir).resolve()
    if not source.is_dir():
        raise ValueError("Input folder does not exist.")
    if target == source or target.is_relative_to(source) or source.is_relative_to(target):
        raise ValueError("Input and output folders must be separate, with neither inside the other.")
    if target.exists() and (not target.is_dir() or any(target.iterdir())):
        raise ValueError("Output folder must be new or empty. Existing output is never overwritten.")
    candidates = sorted((p for p in (source.rglob("*") if config.recursive else source.iterdir()) if p.is_file()), key=lambda p: p.as_posix().casefold())
    target.mkdir(parents=True, exist_ok=True)
    # Exclusive creation protects against concurrent starts into the same empty directory.
    lock = target / ".generation.lock"
    lock.touch(exist_ok=False)
    run_id = uuid.uuid4().hex
    config = Config(**{**asdict(config), "input_dir": str(source), "output_dir": str(target)})
    summary = dict(generator_version=VERSION, run_id=run_id, run_seed=config.run_seed,
                   payload_rate=config.payload_rate if config.payload_mode == "fixed" else None,
                   payload_mode=config.payload_mode, payload_min=config.payload_min, payload_max=config.payload_max,
                   source_dataset=config.source_dataset, total_discovered=len(candidates), total_accepted=0,
                   total_rejected=0, total_generated=0, total_failed=0, total_duplicates=0,
                   total_selected_locations=0, total_actual_changes=0, total_zero_change=0,
                   total_unprocessed=len(candidates), output_directory=str(target), start_time=utc_now(),
                   end_time=None, status="running", python_version=platform.python_version(),
                   numpy_version=np.__version__, pillow_version=PIL.__version__)
    logger = logging.getLogger(f"lsb.{run_id}")
    logger.setLevel(logging.INFO)
    handler = logging.FileHandler(target / "generation.log", encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(handler)
    logger.propagate = False

    def emit(message: str, current: int = 0) -> None:
        logger.info(message)
        if progress:
            progress({"kind": "progress", "current": current, "total": len(candidates), "message": message})

    seen = set()
    try:
        for folder in ("clean", "stego", "masks", "selected_locations"):
            (target / folder).mkdir()
        write_config(target / "run_config.yaml", config)
        emit(f"Run {run_id}; generator {VERSION}; {len(candidates)} files; output {target}")
        logger.info("Effective configuration: %s", asdict(config))
        with (target / "metadata.csv").open("x", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=METADATA_FIELDS)
            writer.writeheader()
            for i, file in enumerate(candidates, 1):
                if cancel is not None and cancel.is_set():
                    summary["status"] = "cancelled"
                    break
                summary["total_unprocessed"] -= 1
                try:
                    if file.is_symlink() or not file.resolve().is_relative_to(source):
                        raise ValueError("Linked sources outside the input tree are unsupported.")
                    if file.suffix.lower() != ".png":
                        raise ValueError("Unsupported extension; only PNG accepted.")
                    clean = load_rgb(file)
                    identity = source_identity(clean)
                    if identity in seen:
                        summary["total_duplicates"] += 1
                        raise ValueError(f"Duplicate canonical source {identity}; already processed.")
                except Exception as exc:
                    summary["total_rejected"] += 1
                    emit(f"Rejected {file.relative_to(source)}: {exc}", i)
                    if config.error_policy == "stop":
                        summary["status"] = "failed"
                        break
                    continue
                summary["total_accepted"] += 1
                seen.add(identity)
                seed, rate = derive_seed(config.run_seed, identity), rate_for(config, identity)
                sample_id = f"{run_id}_{identity}"
                written = []
                try:
                    stego, mask, selected = embed(clean, rate, seed)
                    row = dict(sample_id=sample_id, source_id=identity, label="stego",
                               clean_file=f"clean/{identity}.png", stego_file=f"stego/{identity}.png",
                               mask_file=f"masks/{identity}.npy", selected_locations_file=f"selected_locations/{identity}.npy",
                               generator_version=VERSION, payload_rate=rate, payload_length_bits=len(selected),
                               random_seed=seed, width=clean.shape[1], height=clean.shape[0], channels="RGB",
                               input_format="PNG", output_format="PNG", source_file=file.relative_to(source).as_posix(),
                               source_dataset=config.source_dataset, conversion_applied=False,
                               selected_location_count=len(selected), actual_change_count=int(mask.sum()),
                               zero_change=not bool(mask.any()), numpy_version=np.__version__, pillow_version=PIL.__version__)
                    for field, array in (("clean_file", clean), ("stego_file", stego), ("mask_file", mask), ("selected_locations_file", selected)):
                        path = target / row[field]
                        with path.open("xb") as output:
                            written.append(path)
                            if field.endswith("file") and path.suffix == ".png":
                                Image.fromarray(array).save(output, format="PNG", compress_level=6, optimize=False)
                            else:
                                np.save(output, array, allow_pickle=False)
                    verify_record(target, row, config)
                    writer.writerow(row)
                    stream.flush()
                    summary["total_generated"] += 1
                    summary["total_selected_locations"] += len(selected)
                    summary["total_actual_changes"] += int(mask.sum())
                    summary["total_zero_change"] += int(not mask.any())
                    emit(f"Generated {file.relative_to(source)}: {len(selected):,} selected, {int(mask.sum()):,} changed" + (" — ZERO CHANGE; retain this flag for training review" if not mask.any() else ""), i)
                except Exception as exc:
                    for path in written:
                        path.unlink(missing_ok=True)
                    summary["total_failed"] += 1
                    emit(f"Failed {file.relative_to(source)}: {exc}", i)
                    if config.error_policy == "stop":
                        summary["status"] = "failed"
                        break
        if summary["status"] == "running":
            summary["status"] = "failed" if summary["total_failed"] or not summary["total_generated"] else ("completed_with_rejections" if summary["total_rejected"] else "completed")
    except BaseException as exc:
        summary["status"] = "failed"
        summary["fatal_error"] = str(exc)
        logger.exception("Run failed")
        raise
    finally:
        summary["end_time"] = utc_now()
        try:
            (target / "run_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
            logger.info("Run ended: %s", summary)
        finally:
            handler.close()
            logger.removeHandler(handler)
            lock.unlink(missing_ok=True)
    return summary
