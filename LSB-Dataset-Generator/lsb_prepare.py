"""Prepare fixed-size 8-bit RGB tiles from photo sources."""
from __future__ import annotations

import csv
import json
import multiprocessing as mp
import os
import time
import uuid
from concurrent.futures import Future, ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from threading import Event
from typing import Any, Callable

import numpy as np
import yaml
from PIL import Image, ImageOps

from lsb_raw import decode_raw_pixels, raw_settings_record

VERSION = "6.0.0"
RUN_FILE = "run_config.yaml"
LEDGER_FILE = "done.jsonl"
MANIFEST_FILE = "manifest.csv"
SUPPORTED_RASTER = {
    ".png": "png",
    ".jpg": "jpeg",
    ".jpeg": "jpeg",
    ".tif": "tiff",
    ".tiff": "tiff",
    ".bmp": "bmp",
    ".webp": "webp",
}
RAW_EXTENSIONS = {
    ".3fr",
    ".arw",
    ".cr2",
    ".cr3",
    ".dcr",
    ".dng",
    ".erf",
    ".kdc",
    ".mef",
    ".mos",
    ".mrw",
    ".nef",
    ".nrw",
    ".orf",
    ".pef",
    ".raf",
    ".raw",
    ".rw2",
    ".sr2",
    ".srf",
    ".srw",
    ".x3f",
}
SUPPORTED_EXTENSIONS = set(SUPPORTED_RASTER) | RAW_EXTENSIONS
MANIFEST_FIELDS = (
    "tile_file",
    "source_file",
    "source_format",
    "camera_make",
    "camera_model",
    "source_width",
    "source_height",
    "tile_row",
    "tile_col",
    "pixel_x",
    "pixel_y",
)
Progress = Callable[[dict[str, Any]], None]


@dataclass(frozen=True)
class PrepareConfig:
    """Validated preparation settings."""

    input: str
    output: str
    tile: int = 512
    max_tiles_per_image: int | None = None
    seed: int = 0
    png_compress_level: int = 1

    def normalized(self) -> PrepareConfig:
        """Return settings with absolute paths and validated values."""
        input_path = Path(self.input).expanduser().resolve()
        output_path = Path(self.output).expanduser().resolve()
        if not input_path.is_dir():
            raise ValueError("input folder does not exist")
        if input_path == output_path or output_path.is_relative_to(input_path):
            raise ValueError("output folder must be outside input folder")
        _validate_int(self.tile, "tile", 1)
        if self.max_tiles_per_image is not None:
            _validate_int(self.max_tiles_per_image, "max_tiles_per_image", 1)
        _validate_int(self.seed, "seed", 0)
        _validate_int(self.png_compress_level, "png_compress_level", 0, 9)
        return PrepareConfig(
            input=str(input_path),
            output=str(output_path),
            tile=int(self.tile),
            max_tiles_per_image=(
                None
                if self.max_tiles_per_image is None
                else int(self.max_tiles_per_image)
            ),
            seed=int(self.seed),
            png_compress_level=int(self.png_compress_level),
        )

    def as_dict(self) -> dict[str, Any]:
        """Return the canonical run settings."""
        config = self.normalized()
        result = asdict(config)
        result["version"] = VERSION
        result["rawpy_version"] = _rawpy_version()
        result["raw_settings"] = raw_settings_record()
        return result


Config = PrepareConfig


@dataclass(frozen=True)
class DecodedImage:
    """Decoded source pixels and manifest information."""

    pixels: np.ndarray
    source_format: str
    camera_make: str
    camera_model: str


class NoCompleteTiles(ValueError):
    """The source is smaller than one complete tile."""


def _validate_int(value: Any, name: str, minimum: int, maximum: int | None = None) -> None:
    """Validate one non-Boolean integer."""
    if type(value) is not int:
        raise ValueError(f"{name} must be an integer")
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    if maximum is not None and value > maximum:
        raise ValueError(f"{name} must be at most {maximum}")


def _rawpy_version() -> str | None:
    """Return the installed rawpy version, if available."""
    try:
        import rawpy
    except ImportError:
        return None
    return str(rawpy.__version__)


def _source_format(path: Path) -> str | None:
    """Return the supported source format for one path."""
    suffix = path.suffix.lower()
    if suffix in RAW_EXTENSIONS:
        return "raw"
    return SUPPORTED_RASTER.get(suffix)


def _png_bit_depth(path: Path) -> int | None:
    """Read a PNG bit depth before Pillow decodes it."""
    try:
        with path.open("rb") as stream:
            header = stream.read(26)
    except OSError as exc:
        raise ValueError(f"cannot read source: {exc}") from exc
    if len(header) < 26 or header[:8] != b"\x89PNG\r\n\x1a\n" or header[12:16] != b"IHDR":
        return None
    return int(header[24])


def _camera_info_from_exif(image: Image.Image) -> tuple[str, str]:
    """Read camera make and model from EXIF, if present."""
    exif = image.getexif()
    make = exif.get(271)
    model = exif.get(272)
    return (
        "" if make is None else str(make).strip(),
        "" if model is None else str(model).strip(),
    )


def _raw_camera_info(path: Path) -> tuple[str, str]:
    """Read RAW TIFF IFD0 camera tags without decoding RAW pixels."""
    try:
        with Image.open(path) as image:
            tags = image.tag_v2
            make = tags.get(271)
            model = tags.get(272)
            return (
                "" if make is None else str(make).strip(),
                "" if model is None else str(model).strip(),
            )
    except Exception:
        return "", ""


def _decode_raw(path: Path) -> DecodedImage:
    """Decode one RAW image with fixed LibRaw settings."""
    camera_make, camera_model = _raw_camera_info(path)
    pixels = decode_raw_pixels(path)
    return DecodedImage(pixels, "raw", camera_make, camera_model)


def _decode_raster(path: Path, source_format: str) -> DecodedImage:
    """Decode and validate one non-RAW image."""
    if source_format == "png":
        bit_depth = _png_bit_depth(path)
        if bit_depth is None or bit_depth > 8:
            raise ValueError("16-bit or invalid PNG is unsupported")
    try:
        with Image.open(path) as original:
            if getattr(original, "n_frames", 1) != 1:
                raise ValueError("animated images are unsupported")
            if source_format == "png" and original.mode in {"I;16", "I;16B", "I;16L"}:
                raise ValueError("16-bit images are unsupported")
            if source_format == "tiff":
                bits = original.tag_v2.get(258)
                if bits is not None and max(int(value) for value in (bits if isinstance(bits, tuple) else (bits,))) > 8:
                    raise ValueError("16-bit images are unsupported")
            if original.mode in {"1", "L", "LA", "I", "F", "I;16", "I;16B", "I;16L"}:
                raise ValueError("grayscale images are unsupported")
            if original.mode in {"P", "PA"} and "transparency" in original.info:
                raise ValueError("palette transparency is unsupported")
            if "transparency" in original.info and original.mode not in {"RGBA", "RGBa"}:
                raise ValueError("image transparency is unsupported")
            image = ImageOps.exif_transpose(original)
            camera_make, camera_model = _camera_info_from_exif(original)
            if image.mode in {"RGBA", "RGBa"}:
                image.load()
                alpha = np.asarray(image.getchannel("A"))
                if np.any(alpha < 255):
                    raise ValueError("RGBA images with transparency are unsupported")
            image = image.convert("RGB")
            image.load()
            pixels = np.asarray(image)
    except ValueError:
        raise
    except Exception as exc:
        raise RuntimeError(f"image decode failed: {exc}") from exc
    if pixels.dtype != np.uint8 or pixels.ndim != 3 or pixels.shape[2] != 3:
        raise ValueError("source did not convert to 8-bit RGB")
    return DecodedImage(np.ascontiguousarray(pixels), source_format, camera_make, camera_model)


def decode_source(path: Path) -> DecodedImage:
    """Decode one supported source or raise a classified error."""
    source_format = _source_format(path)
    if source_format is None:
        raise ValueError("unsupported file extension")
    if source_format == "raw":
        return _decode_raw(path)
    return _decode_raster(path, source_format)


def _positions(
    height: int,
    width: int,
    tile: int,
    maximum: int | None,
    seed: int,
    image_index: int,
) -> list[tuple[int, int]]:
    """Choose complete tile positions in stable row-major order."""
    rows, columns = height // tile, width // tile
    total = rows * columns
    if total < 1:
        raise NoCompleteTiles("source has no complete tile")
    if maximum is None or maximum >= total:
        selected = np.arange(total, dtype=np.intp)
    else:
        rng = np.random.default_rng(np.random.SeedSequence([seed, image_index]))
        selected = np.asarray(rng.choice(total, size=maximum, replace=False), dtype=np.intp)
        selected.sort()
    return [(int(index // columns), int(index % columns)) for index in selected]


def _save_tile_atomic(tile: np.ndarray, path: Path, compression: int) -> None:
    """Save one tile through a same-directory temporary file."""
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    try:
        Image.fromarray(tile, mode="RGB").save(
            temporary,
            format="PNG",
            compress_level=compression,
            optimize=False,
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _process_source(job: tuple[dict[str, Any], str, int]) -> dict[str, Any]:
    """Decode one source and atomically write its selected tiles."""
    raw_config, source_path_string, image_index = job
    config = PrepareConfig(**{key: raw_config[key] for key in (
        "input", "output", "tile", "max_tiles_per_image", "seed", "png_compress_level"
    )}).normalized()
    source_path = Path(source_path_string)
    source_file = source_path.relative_to(Path(config.input)).as_posix()
    try:
        decoded = decode_source(source_path)
    except ValueError as exc:
        return {
            "image_index": image_index,
            "source_file": source_file,
            "status": "skipped",
            "message": f"Skipped {source_file}: {exc}",
            "tiles": [],
        }
    except Exception as exc:
        return {
            "image_index": image_index,
            "source_file": source_file,
            "status": "retry",
            "message": f"Retry {source_file}: unexpected decode error: {exc}",
            "tiles": [],
        }
    pixels = decoded.pixels
    try:
        positions = _positions(
            int(pixels.shape[0]),
            int(pixels.shape[1]),
            config.tile,
            config.max_tiles_per_image,
            config.seed,
            image_index,
        )
        output_root = Path(config.output)
        written: list[Path] = []
        rows: list[dict[str, Any]] = []
        for row, column in positions:
            filename = f"{source_path.stem}_r{row}_c{column}.png"
            path = output_root / filename
            tile = pixels[
                row * config.tile : (row + 1) * config.tile,
                column * config.tile : (column + 1) * config.tile,
            ]
            _save_tile_atomic(tile, path, config.png_compress_level)
            written.append(path)
            rows.append(
                {
                    "tile_file": filename,
                    "source_file": source_file,
                    "source_format": decoded.source_format,
                    "camera_make": decoded.camera_make,
                    "camera_model": decoded.camera_model,
                    "source_width": int(pixels.shape[1]),
                    "source_height": int(pixels.shape[0]),
                    "tile_row": row,
                    "tile_col": column,
                    "pixel_x": column * config.tile,
                    "pixel_y": row * config.tile,
                }
            )
        return {
            "image_index": image_index,
            "source_file": source_file,
            "status": "done",
            "message": "",
            "tiles": rows,
        }
    except NoCompleteTiles as exc:
        return {
            "image_index": image_index,
            "source_file": source_file,
            "status": "skipped",
            "message": f"Skipped {source_file}: {exc}",
            "tiles": [],
        }
    except Exception as exc:
        for path in locals().get("written", []):
            path.unlink(missing_ok=True)
        return {
            "image_index": image_index,
            "source_file": source_file,
            "status": "retry",
            "message": f"Retry {source_file}: unexpected tile error: {exc}",
            "tiles": [],
        }


def _discover(input_root: Path) -> list[Path]:
    """Discover and sort every file recursively."""
    return sorted(
        (path for path in input_root.rglob("*") if path.is_file()),
        key=lambda path: path.relative_to(input_root).as_posix(),
    )


def _load_done(path: Path) -> dict[int, dict[str, Any]]:
    """Load the latest ledger record for each source."""
    records: dict[int, dict[str, Any]] = {}
    if not path.exists():
        return records
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            record = json.loads(line)
            records[int(record["image_index"])] = record
    return records


def _record_complete(record: dict[str, Any], output: Path) -> bool:
    """Check that all tiles in a deterministic record still exist."""
    if record.get("status") == "skipped":
        return True
    if record.get("status") != "done":
        return False
    return all((output / row["tile_file"]).is_file() for row in record.get("tiles", []))


def _append_done(path: Path, record: dict[str, Any]) -> None:
    """Append one deterministic source outcome."""
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n")
        stream.flush()


def _write_manifest(path: Path, records: dict[int, dict[str, Any]]) -> None:
    """Rebuild the tile manifest from the ledger."""
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        for index in sorted(records):
            for row in records[index].get("tiles", []):
                writer.writerow(row)


def _same_run(first: dict[str, Any], second: dict[str, Any]) -> bool:
    """Compare canonical preparation settings."""
    return json.dumps(first, sort_keys=True) == json.dumps(second, sort_keys=True)


def prepare(
    config: PrepareConfig,
    workers: int | None = None,
    progress: Progress | None = None,
    cancel: Event | None = None,
) -> dict[str, Any]:
    """Prepare all supported source photos into fixed-size tiles."""
    config = config.normalized()
    output = Path(config.output)
    output.mkdir(parents=True, exist_ok=True)
    run_path = output / RUN_FILE
    run_data = config.as_dict()
    if run_path.exists():
        try:
            existing = yaml.safe_load(run_path.read_text(encoding="utf-8"))
        except Exception as exc:
            raise ValueError(f"cannot read {RUN_FILE}: {exc}") from exc
        if not isinstance(existing, dict) or not _same_run(existing, run_data):
            raise ValueError("output already has a different run configuration")
    elif any(output.iterdir()):
        raise ValueError("output exists without a run configuration; refusing to overwrite it")
    else:
        run_path.write_text(yaml.safe_dump(run_data, sort_keys=False), encoding="utf-8")
    ledger_path = output / LEDGER_FILE
    records = _load_done(ledger_path)
    sources = _discover(Path(config.input))
    jobs: list[tuple[dict[str, Any], str, int]] = []
    raw_config = asdict(config)
    for image_index, source in enumerate(sources, 1):
        record = records.get(image_index)
        if record is not None and record.get("source_file") == source.relative_to(Path(config.input)).as_posix() and _record_complete(record, output):
            continue
        jobs.append((raw_config, str(source), image_index))
    log_path = output / "run.log"
    total = len(sources)
    completed = len(records)
    handled = len(records)
    retries = 0
    started = time.perf_counter()
    with log_path.open("a", encoding="utf-8") as log:
        log.write(f"Run started: {total} sources, {len(jobs)} remaining\n")
        if cancel is not None and cancel.is_set():
            _write_manifest(output / MANIFEST_FILE, records)
            log.write("Run cancelled before processing\n")
            return {"status": "cancelled", "sources": total, "tiles": sum(len(item.get("tiles", [])) for item in records.values())}
        worker_count = workers if workers is not None else (os.cpu_count() or 1)
        if worker_count < 1:
            raise ValueError("workers must be at least 1")
        pending: dict[int, dict[str, Any]] = {}
        next_index = min((job[2] for job in jobs), default=total + 1)
        while next_index in records:
            next_index += 1
        has_raw = any(_source_format(path) == "raw" for path in sources)
        process_context = mp.get_context("spawn") if has_raw else None
        executor = ProcessPoolExecutor(max_workers=worker_count, mp_context=process_context)
        cancelled = False
        try:
            futures: dict[Future[dict[str, Any]], int] = {
                executor.submit(_process_source, job): job[2] for job in jobs
            }
            for future in as_completed(futures):
                index = futures[future]
                if cancel is not None and cancel.is_set():
                    cancelled = True
                    break
                try:
                    result = future.result()
                except Exception as exc:
                    result = {
                        "image_index": index,
                        "source_file": sources[index - 1].relative_to(Path(config.input)).as_posix(),
                        "status": "retry",
                        "message": f"Retry source {index}: unexpected worker error: {exc}",
                        "tiles": [],
                    }
                pending[index] = result
                while next_index in records or next_index in pending:
                    if next_index in records:
                        next_index += 1
                        continue
                    record = pending.pop(next_index)
                    if record.get("status") in {"done", "skipped"}:
                        _append_done(ledger_path, record)
                        records[next_index] = record
                        completed += 1
                    else:
                        retries += 1
                    handled += 1
                    message = record.get("message", "")
                    if message:
                        log.write(str(message) + "\n")
                    elapsed = max(time.perf_counter() - started, 1e-9)
                    tile_count = sum(len(item.get("tiles", [])) for item in records.values())
                    photos_per_second = handled / elapsed
                    tiles_per_second = tile_count / elapsed
                    event = {
                        "kind": "progress",
                        "completed": handled,
                        "total": total,
                        "tiles": tile_count,
                        "photos_per_second": photos_per_second,
                        "tiles_per_second": tiles_per_second,
                    }
                    if progress is not None:
                        progress(event)
                    print(
                        f"Processed {handled}/{total} photos "
                        f"({photos_per_second:.2f} photos/s, {tiles_per_second:.2f} tiles/s)",
                        flush=True,
                    )
                    next_index += 1
        except KeyboardInterrupt:
            cancelled = True
        finally:
            executor.shutdown(wait=not cancelled, cancel_futures=cancelled)
        _write_manifest(output / MANIFEST_FILE, records)
        if cancelled:
            log.write(f"Run cancelled: {completed}/{total} sources complete\n")
            return {
                "status": "cancelled",
                "sources": total,
                "tiles": sum(len(item.get("tiles", [])) for item in records.values()),
            }
        if retries:
            log.write(f"Run incomplete: {retries} sources will be retried\n")
            return {
                "status": "failed",
                "sources": total,
                "tiles": sum(len(item.get("tiles", [])) for item in records.values()),
            }
        log.write(f"Run completed: {completed}/{total} sources complete\n")
    return {
        "status": "completed",
        "sources": total,
        "tiles": sum(len(item.get("tiles", [])) for item in records.values()),
    }


__all__ = [
    "Config",
    "MANIFEST_FIELDS",
    "PrepareConfig",
    "decode_source",
    "prepare",
]
