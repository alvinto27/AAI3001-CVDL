"""Deterministic v6 bulk generation and verification.

Each variant uses two independent streams: settings use
``SeedSequence([seed, cover_index, variant, 0])`` and embedding uses
``SeedSequence([seed, cover_index, variant, 1])``.  The embed stream is stored
in each stego PNG so replay needs no run configuration.
"""
from __future__ import annotations

import csv
import json
import math
import os
import time
import uuid
from concurrent.futures import Future, ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from threading import Event
from typing import Any, Callable, Mapping

import numpy as np
import yaml
from PIL import Image
from PIL.PngImagePlugin import PngInfo

from lsb_engine import EmbedResult, embed, extract

VERSION = "6.0.0"
ARTIFACT_DIRS = ("clean", "stego", "masks")
CONFIG_KEYS = {
    "seed",
    "input",
    "output",
    "png_compress_level",
    "variants_per_cover",
    "write_masks",
    "bit_mode",
    "k",
    "method",
    "placement",
    "stride",
    "payload",
    "start",
    "size",
}
JSON_FIELDS = (
    "label",
    "cover_id",
    "cover_index",
    "variant",
    "source_file",
    "size",
    "generator_version",
    "seed",
    "embed_seed",
    "bit_mode",
    "k",
    "method",
    "placement",
    "start",
    "start_random",
    "stride",
    "payload_kind",
    "payload_bytes",
    "embedded_bits",
    "units_used",
    "units_changed",
    "bpp_actual",
    "size_kind",
    "size_value",
)
CSV_FIELDS = (
    "label",
    "cover_id",
    "cover_index",
    "variant",
    "source_file",
    "height",
    "width",
    "channels",
    "generator_version",
    "seed",
    "embed_seed_seed",
    "embed_seed_cover_index",
    "embed_seed_variant",
    "embed_seed_stream",
    "bit_mode",
    "k",
    "method",
    "placement",
    "start",
    "start_random",
    "stride",
    "payload_kind",
    "payload_bytes",
    "embedded_bits",
    "units_used",
    "units_changed",
    "bpp_actual",
    "size_kind",
    "size_value",
    "file",
    "mask_file",
)

Progress = Callable[[dict[str, Any]], None]


@dataclass(frozen=True)
class Config:
    """Validated v6 run configuration."""

    seed: int
    input: str
    output: str
    png_compress_level: int = 1
    variants_per_cover: int = 1
    write_masks: bool = True
    bit_mode: str | list[str] = "lowest"
    k: int | list[int] = 1
    method: str | list[str] = "replace"
    placement: str | list[str] = "contiguous"
    stride: int | list[int] = 1
    payload: str | list[str] = "random_bytes"
    start: str | int = "random"
    size: dict[str, int | float | list[int] | list[float]] | int | float | list[int] | list[float] | None = None

    def normalized(self) -> Config:
        """Return a validated configuration with canonical size syntax."""
        _validate_scalar_int(self.seed, "seed", minimum=0)
        _validate_path_value(self.input, "input")
        _validate_path_value(self.output, "output")
        _validate_scalar_int(self.png_compress_level, "png_compress_level", minimum=0, maximum=9)
        _validate_scalar_int(self.variants_per_cover, "variants_per_cover", minimum=1)
        if type(self.write_masks) is not bool:
            raise ValueError("write_masks must be true or false")
        bit_modes = _normalize_choice(self.bit_mode, "bit_mode")
        if any(value not in {"lowest", "plane"} for value in bit_modes):
            raise ValueError("bit_mode must contain only 'lowest' or 'plane'")
        ks = _normalize_choice(self.k, "k")
        for value in ks:
            _validate_scalar_int(value, "k", minimum=1, maximum=8)
        methods = _normalize_choice(self.method, "method")
        if any(value not in {"replace", "match"} for value in methods):
            raise ValueError("method must contain only 'replace' or 'match'")
        placements = _normalize_choice(self.placement, "placement")
        if any(value not in {"contiguous", "stride", "random"} for value in placements):
            raise ValueError("placement must contain only contiguous, stride, or random")
        stride = _normalize_range(self.stride, "stride", integer=True, minimum=1)
        if "stride" not in placements and stride != [1] and stride != 1:
            raise ValueError("stride is only valid when placement can be stride")
        payloads = _normalize_choice(self.payload, "payload")
        if any(value not in {"random_bytes", "letters"} for value in payloads):
            raise ValueError("payload must contain only random_bytes or letters")
        if self.start != "random":
            _validate_scalar_int(self.start, "start", minimum=0)
        size = _normalize_size(self.size)
        return Config(
            seed=int(self.seed),
            input=str(self.input),
            output=str(self.output),
            png_compress_level=self.png_compress_level,
            variants_per_cover=int(self.variants_per_cover),
            write_masks=self.write_masks,
            bit_mode=_restore_choice(bit_modes, self.bit_mode),
            k=_restore_choice(ks, self.k),
            method=_restore_choice(methods, self.method),
            placement=_restore_choice(placements, self.placement),
            stride=_restore_range(stride, self.stride),
            payload=_restore_choice(payloads, self.payload),
            start=self.start,
            size=size,
        )

    def as_dict(self) -> dict[str, Any]:
        """Return the canonical YAML-compatible mapping."""
        result = asdict(self.normalized())
        return result


class SizeDoesNotFit(ValueError):
    """The minimum configured payload cannot fit one embedding."""


def _validate_scalar_int(
    value: Any,
    name: str,
    minimum: int | None = None,
    maximum: int | None = None,
) -> None:
    """Validate one non-Boolean integer."""
    if type(value) is not int:
        raise ValueError(f"{name} must be an integer")
    if minimum is not None and value < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    if maximum is not None and value > maximum:
        raise ValueError(f"{name} must be at most {maximum}")


def _validate_path_value(value: Any, name: str) -> None:
    """Validate one non-empty path string."""
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty path")


def _normalize_choice(value: Any, name: str) -> list[Any]:
    """Validate a fixed value or a non-empty list of choices."""
    values = value if isinstance(value, list) else [value]
    if not values or any(isinstance(item, (list, dict)) for item in values):
        raise ValueError(f"{name} must be one value or a non-empty list of values")
    return list(values)


def _restore_choice(values: list[Any], original: Any) -> Any:
    """Keep fixed choices scalar and choice sets as lists."""
    return values[0] if not isinstance(original, list) else values


def _normalize_range(
    value: Any,
    name: str,
    *,
    integer: bool,
    minimum: int | float,
) -> list[Any]:
    """Validate a scalar or inclusive two-value range."""
    if isinstance(value, list):
        if len(value) != 2:
            raise ValueError(f"{name} range must contain exactly two values")
        values = list(value)
    else:
        values = [value]
    for item in values:
        if integer:
            _validate_scalar_int(item, name, minimum=int(minimum))
        elif isinstance(item, (bool, np.bool_)) or not isinstance(item, (int, float)):
            raise ValueError(f"{name} must contain numbers")
        elif not math.isfinite(float(item)) or float(item) < float(minimum):
            raise ValueError(f"{name} must contain finite values at least {minimum}")
    if len(values) == 2 and values[0] > values[1]:
        raise ValueError(f"{name} minimum must not exceed maximum")
    return values


def _restore_range(values: list[Any], original: Any) -> Any:
    """Keep scalar ranges scalar and ranges as lists."""
    return values[0] if not isinstance(original, list) else values


def _normalize_size(value: Any) -> dict[str, Any]:
    """Normalize the exclusive bpp or payload_bytes size setting."""
    if isinstance(value, dict):
        if set(value) not in ({"bpp"}, {"payload_bytes"}):
            raise ValueError("size must contain exactly one of bpp or payload_bytes")
        kind = next(iter(value))
        raw = value[kind]
    elif isinstance(value, (bool, np.bool_)) or value is None:
        raise ValueError("size must contain bpp or payload_bytes")
    else:
        kind = "payload_bytes" if type(value) is int else "bpp"
        raw = value
    if kind == "payload_bytes":
        values = _normalize_range(raw, "payload_bytes", integer=True, minimum=1)
    else:
        values = _normalize_range(raw, "bpp", integer=False, minimum=0.0)
        if any(float(item) <= 0 for item in values):
            raise ValueError("bpp must be positive")
    normalized: int | float | list[int] | list[float]
    if len(values) == 1:
        normalized = values[0]
    else:
        normalized = values
    return {kind: normalized}


def read_config(path: Path) -> Config:
    """Read and strictly validate a YAML configuration."""
    path = Path(path)
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ValueError(f"Could not read configuration: {exc}") from exc
    if not isinstance(raw, dict):
        raise ValueError("Configuration must contain a YAML mapping")
    unknown = set(raw) - CONFIG_KEYS
    if unknown:
        raise ValueError(f"Unknown configuration keys: {', '.join(sorted(unknown))}")
    data = dict(raw)
    for key, default in (
        ("png_compress_level", 1),
        ("variants_per_cover", 1),
        ("write_masks", True),
        ("bit_mode", "lowest"),
        ("k", 1),
        ("method", "replace"),
        ("placement", "contiguous"),
        ("stride", 1),
        ("payload", "random_bytes"),
        ("start", "random"),
    ):
        data.setdefault(key, default)
    for key in ("input", "output"):
        if key not in data:
            raise ValueError(f"Missing required configuration key: {key}")
        value = data[key]
        if isinstance(value, str) and not Path(value).is_absolute():
            data[key] = str((path.parent / value).resolve())
    if "seed" not in data:
        raise ValueError("Missing required configuration key: seed")
    if "size" not in data:
        raise ValueError("Missing required configuration key: size")
    return Config(**data).normalized()


def write_config(path: Path, config: Config) -> None:
    """Write a canonical configuration mapping."""
    path.write_text(yaml.safe_dump(config.as_dict(), sort_keys=False), encoding="utf-8")


def _draw_choice(value: Any, rng: np.random.Generator) -> Any:
    """Draw one uniformly selected choice or return a fixed value."""
    if isinstance(value, list):
        return value[int(rng.integers(0, len(value)))]
    return value


def _draw_stride(value: int | list[int], rng: np.random.Generator) -> int:
    """Draw one integer uniformly from a stride range."""
    if isinstance(value, list):
        return int(rng.integers(value[0], value[1] + 1))
    return int(value)


def _seed_sequence(
    seed: int,
    cover_index: int,
    variant: int,
    stream: int,
) -> np.random.SeedSequence:
    """Build one stable settings or embedding seed sequence."""
    return np.random.SeedSequence([seed, cover_index, variant, stream])


def _bpp_payload_bytes(value: float, pixels: int) -> int:
    """Apply the engine's bpp-to-byte conversion."""
    return max(1, int(round(value * pixels / 8.0)))


def _maximum_payload_bytes(
    image: np.ndarray,
    bit_mode: str,
    k: int,
    placement: str,
    stride: int,
    start: str | int,
) -> int:
    """Find the largest byte payload for the drawn embedding settings."""
    total_units = int(image.size)
    bits_per_unit = k if bit_mode == "lowest" else 1
    if placement == "random":
        usable_units = total_units
    elif start == "random":
        usable_units = (total_units - 1) // stride + 1
    else:
        if int(start) >= total_units:
            return 0
        usable_units = (total_units - 1 - int(start)) // stride + 1
    return (usable_units * bits_per_unit) // 8


def _draw_size(
    size: dict[str, Any],
    image: np.ndarray,
    max_payload_bytes: int,
    rng: np.random.Generator,
) -> tuple[str, int | float]:
    """Draw a size and cap its upper bound at carrier capacity."""
    kind = next(iter(size))
    raw = size[kind]
    values = raw if isinstance(raw, list) else [raw]
    low = values[0]
    high = values[-1]
    if max_payload_bytes < 1:
        raise SizeDoesNotFit("carrier cannot hold the minimum one-byte payload")
    if kind == "payload_bytes":
        if low > max_payload_bytes:
            raise SizeDoesNotFit("minimum payload_bytes does not fit")
        high = min(high, max_payload_bytes)
        actual = int(rng.integers(low, high + 1)) if len(values) == 2 else int(low)
        return kind, actual
    pixels = int(image.shape[0] * image.shape[1])
    low_bytes = _bpp_payload_bytes(float(low), pixels)
    if low_bytes > max_payload_bytes:
        raise SizeDoesNotFit("minimum bpp does not fit")
    high = min(float(high), max_payload_bytes * 8.0 / pixels)
    actual = float(rng.uniform(float(low), high)) if len(values) == 2 else float(low)
    if _bpp_payload_bytes(actual, pixels) > max_payload_bytes:
        actual = max_payload_bytes * 8.0 / pixels
    return kind, actual


def _draw_settings(
    config: Config,
    image: np.ndarray,
    rng: np.random.Generator,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Draw one variant's settings and its metadata size value."""
    config = config.normalized()
    compression = config.png_compress_level
    bit_mode = str(_draw_choice(config.bit_mode, rng))
    k = int(_draw_choice(config.k, rng))
    method = str(_draw_choice(config.method, rng))
    placement = str(_draw_choice(config.placement, rng))
    stride = _draw_stride(config.stride, rng) if placement == "stride" else 1
    payload = str(_draw_choice(config.payload, rng))
    start: str | int = config.start
    maximum = _maximum_payload_bytes(image, bit_mode, k, placement, stride, start)
    size_kind, size_value = _draw_size(config.size, image, maximum, rng)
    kwargs: dict[str, Any] = {
        "bit_mode": bit_mode,
        "k": k,
        "method": method,
        "placement": placement,
        "rng": rng,
        "start": None if start == "random" else int(start),
        "stride": stride,
        "payload": payload,
    }
    kwargs[size_kind] = size_value
    draw_metadata = {
        "png_compress_level": compression,
        "size_kind": size_kind,
        "size_value": size_value,
    }
    return kwargs, draw_metadata


def _size_metadata_value(value: Any) -> Any:
    """Make a drawn size safe for JSON."""
    if isinstance(value, np.generic):
        return value.item()
    return value


def _size_descriptor(image: np.ndarray) -> list[int]:
    """Return the JSON image size descriptor."""
    return [int(image.shape[0]), int(image.shape[1]), int(image.shape[2])]


def _result_fields(result: EmbedResult) -> dict[str, Any]:
    """Copy all public EmbedResult fields into a JSON mapping."""
    return {
        "bit_mode": result.bit_mode,
        "k": result.k,
        "method": result.method,
        "placement": result.placement,
        "start": result.start,
        "stride": result.stride,
        "payload_kind": result.payload_kind,
        "payload_bytes": result.payload_bytes,
        "embedded_bits": result.embedded_bits,
        "units_used": result.units_used,
        "units_changed": result.units_changed,
        "bpp_actual": result.bpp_actual,
    }


def _clean_metadata(cover_id: str, source_file: str, image: np.ndarray) -> dict[str, Any]:
    """Build the required metadata stored in a clean PNG."""
    return {
        "label": "clean",
        "cover_id": cover_id,
        "source_file": source_file,
        "size": _size_descriptor(image),
    }


def _stego_metadata(
    cover_id: str,
    cover_index: int,
    variant: int,
    source_file: str,
    image: np.ndarray,
    config: Config,
    embed_sequence: np.random.SeedSequence,
    start_random: bool,
    draw_metadata: dict[str, Any],
    result: EmbedResult,
) -> dict[str, Any]:
    """Build metadata stored in a stego PNG."""
    fields = _result_fields(result)
    entropy = embed_sequence.entropy
    if isinstance(entropy, (int, np.integer)):
        entropy_value: int | list[int] = int(entropy)
    else:
        entropy_value = [int(value) for value in entropy]
    fields.update(
        {
            "label": "stego",
            "cover_id": cover_id,
            "cover_index": cover_index,
            "variant": variant,
            "source_file": source_file,
            "size": _size_descriptor(image),
            "generator_version": VERSION,
            "seed": config.seed,
            "embed_seed": {
                "entropy": entropy_value,
                "spawn_key": [int(value) for value in embed_sequence.spawn_key],
            },
            "start_random": start_random,
            "size_kind": draw_metadata["size_kind"],
            "size_value": _size_metadata_value(draw_metadata["size_value"]),
        }
    )
    return fields


def _save_png_atomic(
    array: np.ndarray,
    path: Path,
    compression: int,
    metadata: dict[str, Any] | None = None,
    mask: bool = False,
) -> None:
    """Save one PNG through a same-directory temporary file and rename."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    try:
        if mask:
            image = Image.fromarray((array.astype(np.uint8) * 255), mode="L").convert("1")
            image.save(temporary, format="PNG", compress_level=compression, optimize=False)
        else:
            image = Image.fromarray(array, mode="RGB")
            pnginfo = None
            if metadata is not None:
                pnginfo = PngInfo()
                pnginfo.add_text(
                    "stego",
                    json.dumps(metadata, sort_keys=True),
                )
            image.save(
                temporary,
                format="PNG",
                compress_level=compression,
                optimize=False,
                pnginfo=pnginfo,
            )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _read_png_header(path: Path) -> None:
    """Reject non-8-bit-RGB PNG files before decoding."""
    try:
        with path.open("rb") as stream:
            header = stream.read(26)
    except OSError as exc:
        raise ValueError(f"cannot read file: {exc}") from exc
    if len(header) < 26 or header[:8] != b"\x89PNG\r\n\x1a\n" or header[12:16] != b"IHDR":
        raise ValueError("not a PNG file")
    if header[24] != 8 or header[25] != 2:
        raise ValueError("only 8-bit RGB PNG is supported")


def load_rgb_png(path: Path) -> np.ndarray:
    """Check the PNG header, then decode one static RGB image."""
    if path.suffix.lower() != ".png":
        raise ValueError("unsupported extension; only PNG is accepted")
    _read_png_header(path)
    try:
        with Image.open(path) as image:
            if image.format != "PNG" or image.mode != "RGB":
                raise ValueError("decoded image is not RGB")
            if getattr(image, "n_frames", 1) != 1:
                raise ValueError("animated PNG is unsupported")
            if "transparency" in image.info:
                raise ValueError("PNG transparency is unsupported")
            image.load()
            array = np.asarray(image)
    except ValueError:
        raise
    except Exception as exc:
        raise ValueError(f"could not decode PNG: {exc}") from exc
    if array.dtype != np.uint8 or array.ndim != 3 or array.shape[2] != 3:
        raise ValueError("decoded image is not an 8-bit RGB array")
    return np.ascontiguousarray(array)


def _row_from_metadata(metadata: dict[str, Any], file_path: str, mask_file: str) -> dict[str, Any]:
    """Flatten one JSON metadata mapping into CSV-safe scalar fields."""
    size = metadata.get("size")
    size_values = size if isinstance(size, list) and len(size) == 3 else [None, None, None]
    seed = metadata.get("embed_seed")
    entropy = seed.get("entropy") if isinstance(seed, dict) else None
    entropy_values = entropy if isinstance(entropy, list) else [entropy]
    while len(entropy_values) < 4:
        entropy_values.append(None)
    cover_id = metadata.get("cover_id")
    cover_index = metadata.get("cover_index")
    if cover_index is None and isinstance(cover_id, str) and cover_id.isdigit():
        cover_index = int(cover_id)
    row = {
        "label": metadata.get("label"),
        "cover_id": cover_id,
        "cover_index": cover_index,
        "variant": metadata.get("variant"),
        "source_file": metadata.get("source_file"),
        "height": size_values[0],
        "width": size_values[1],
        "channels": size_values[2],
        "generator_version": metadata.get("generator_version"),
        "seed": metadata.get("seed"),
        "embed_seed_seed": entropy_values[0],
        "embed_seed_cover_index": entropy_values[1],
        "embed_seed_variant": entropy_values[2],
        "embed_seed_stream": entropy_values[3],
        "bit_mode": metadata.get("bit_mode"),
        "k": metadata.get("k"),
        "method": metadata.get("method"),
        "placement": metadata.get("placement"),
        "start": metadata.get("start"),
        "start_random": metadata.get("start_random"),
        "stride": metadata.get("stride"),
        "payload_kind": metadata.get("payload_kind"),
        "payload_bytes": metadata.get("payload_bytes"),
        "embedded_bits": metadata.get("embedded_bits"),
        "units_used": metadata.get("units_used"),
        "units_changed": metadata.get("units_changed"),
        "bpp_actual": metadata.get("bpp_actual"),
        "size_kind": metadata.get("size_kind"),
        "size_value": metadata.get("size_value"),
        "file": file_path,
        "mask_file": mask_file,
    }
    return row


def _process_cover(job: tuple[dict[str, Any], str, int]) -> dict[str, Any]:
    """Decode, generate, and atomically write one cover's outputs."""
    raw_config, source_path_string, cover_index = job
    config = Config(**raw_config).normalized()
    source_path = Path(source_path_string)
    input_root = Path(config.input)
    source_file = source_path.relative_to(input_root).as_posix()
    cover_id = f"{cover_index:06d}"
    try:
        image = load_rgb_png(source_path)
    except ValueError as exc:
        return {
            "cover_index": cover_index,
            "cover_id": cover_id,
            "status": "skipped",
            "message": f"Skipped {source_file}: {exc}",
            "rows": [],
        }
    except Exception as exc:
        return {
            "cover_index": cover_index,
            "cover_id": cover_id,
            "status": "retry",
            "message": f"Retry {source_file}: unexpected decode error: {exc}",
            "rows": [],
        }
    output_root = Path(config.output)
    clean_path = output_root / "clean" / f"{cover_id}.png"
    clean_meta = _clean_metadata(cover_id, source_file, image)
    written: list[Path] = []
    try:
        _save_png_atomic(image, clean_path, config.png_compress_level, clean_meta)
        written.append(clean_path)
        rows = [_row_from_metadata(clean_meta, f"clean/{cover_id}.png", "")]
        messages: list[str] = []
        for variant in range(1, config.variants_per_cover + 1):
            settings_sequence = _seed_sequence(config.seed, cover_index, variant, 0)
            settings_rng = np.random.default_rng(settings_sequence)
            try:
                kwargs, draw_metadata = _draw_settings(config, image, settings_rng)
            except SizeDoesNotFit as exc:
                messages.append(f"Skipped {cover_id}_{variant}: {exc}")
                continue
            embed_sequence = _seed_sequence(config.seed, cover_index, variant, 1)
            kwargs["rng"] = np.random.default_rng(embed_sequence)
            stego, mask, result = embed(image, **kwargs)
            stego_meta = _stego_metadata(
                cover_id,
                cover_index,
                variant,
                source_file,
                image,
                config,
                embed_sequence,
                config.start == "random",
                draw_metadata,
                result,
            )
            stego_relative = f"stego/{cover_id}_{variant}.png"
            stego_path = output_root / stego_relative
            mask_relative = f"masks/{cover_id}_{variant}.png" if config.write_masks else ""
            mask_path = output_root / mask_relative if mask_relative else None
            _save_png_atomic(stego, stego_path, config.png_compress_level, stego_meta)
            written.append(stego_path)
            if mask_path is not None:
                _save_png_atomic(mask, mask_path, config.png_compress_level, mask=True)
                written.append(mask_path)
            rows.append(_row_from_metadata(stego_meta, stego_relative, mask_relative))
        return {
            "cover_index": cover_index,
            "cover_id": cover_id,
            "status": "done",
            "message": messages,
            "rows": rows,
        }
    except Exception as exc:
        for path in written:
            path.unlink(missing_ok=True)
        return {
            "cover_index": cover_index,
            "cover_id": cover_id,
            "status": "retry",
            "message": f"Retry {source_file}: unexpected generation error: {exc}",
            "rows": [],
        }


def _load_done(path: Path) -> dict[int, dict[str, Any]]:
    """Read completed cover records from the append-only ledger."""
    records: dict[int, dict[str, Any]] = {}
    if not path.exists():
        return records
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            record = json.loads(line)
            records[int(record["cover_index"])] = record
    return records


def _append_done(path: Path, record: dict[str, Any]) -> None:
    """Append one completed cover record."""
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n")
        stream.flush()


def _write_metadata(path: Path, records: Mapping[int, dict[str, Any]]) -> None:
    """Rebuild metadata.csv from the done ledger."""
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for index in sorted(records):
            for row in records[index].get("rows", []):
                writer.writerow({field: _csv_value(row.get(field)) for field in CSV_FIELDS})


def _csv_value(value: Any) -> str | int | float:
    """Return one scalar metadata CSV cell."""
    if value is None:
        return ""
    if isinstance(value, (dict, list)):
        raise ValueError("metadata CSV rows must be flat")
    if isinstance(value, bool):
        return str(value).lower()
    return value


def _same_config(first: Config, second: Config) -> bool:
    """Compare canonical configurations."""
    return json.dumps(first.as_dict(), sort_keys=True) == json.dumps(second.as_dict(), sort_keys=True)


def _validate_paths(config: Config) -> tuple[Path, Path]:
    """Validate input and output folder relationships."""
    input_root = Path(config.input).resolve()
    output_root = Path(config.output).resolve()
    if not input_root.is_dir():
        raise ValueError("input folder does not exist")
    if input_root == output_root or output_root.is_relative_to(input_root):
        raise ValueError("output folder must be outside input folder")
    return input_root, output_root


def _discover(input_root: Path) -> list[Path]:
    """Discover and stably sort all files under the input folder."""
    return sorted(
        (path for path in input_root.rglob("*") if path.is_file()),
        key=lambda path: path.relative_to(input_root).as_posix(),
    )


def generate(
    config: Config,
    workers: int | None = None,
    progress: Progress | None = None,
    cancel: Event | None = None,
) -> dict[str, Any]:
    """Generate one clean image and configured stego variants per cover."""
    config = config.normalized()
    input_root, output_root = _validate_paths(config)
    output_root.mkdir(parents=True, exist_ok=True)
    run_config_path = output_root / "run_config.yaml"
    if run_config_path.exists():
        existing = read_config(run_config_path)
        if not _same_config(existing, config):
            raise ValueError("output already has a different run_config.yaml")
    elif any(output_root.iterdir()):
        raise ValueError("output exists without run_config.yaml; refusing to overwrite it")
    else:
        write_config(run_config_path, config)
    for folder in ARTIFACT_DIRS:
        (output_root / folder).mkdir(exist_ok=True)
    done_path = output_root / "done.jsonl"
    records = _load_done(done_path)
    candidates = _discover(input_root)
    jobs = [
        (config.as_dict(), str(path), index)
        for index, path in enumerate(candidates, 1)
        if index not in records
    ]
    log_path = output_root / "run.log"
    start_time = time.perf_counter()
    completed = len(records)
    handled = len(records)
    retry_count = 0
    total = len(candidates)
    with log_path.open("a", encoding="utf-8") as log:
        log.write(f"Run started: {total} files, {len(jobs)} remaining\n")
        if cancel is not None and cancel.is_set():
            _write_metadata(output_root / "metadata.csv", records)
            log.write("Run cancelled before processing\n")
            return {"status": "cancelled", "total": total, "completed": completed}
        worker_count = workers if workers is not None else (os.cpu_count() or 1)
        if worker_count < 1:
            raise ValueError("workers must be at least 1")
        future_map: dict[Future[dict[str, Any]], int] = {}
        pending_results: dict[int, dict[str, Any]] = {}
        next_to_write = min((job[2] for job in jobs), default=total + 1)
        while next_to_write in records:
            next_to_write += 1
        cancelled = False
        executor = ProcessPoolExecutor(max_workers=worker_count)
        try:
            for job in jobs:
                future_map[executor.submit(_process_cover, job)] = job[2]
            for future in as_completed(future_map):
                index = future_map[future]
                if cancel is not None and cancel.is_set():
                    cancelled = True
                    break
                try:
                    result = future.result()
                except Exception as exc:
                    result = {
                        "cover_index": index,
                        "cover_id": f"{index:06d}",
                        "status": "retry",
                        "message": f"Retry cover {index:06d}: unexpected worker error: {exc}",
                        "rows": [],
                    }
                pending_results[index] = result
                while next_to_write in records or next_to_write in pending_results:
                    if next_to_write in records:
                        next_to_write += 1
                        continue
                    record = pending_results.pop(next_to_write)
                    status = record.get("status")
                    if status in {"done", "skipped"}:
                        _append_done(done_path, record)
                        records[next_to_write] = record
                        completed += 1
                    else:
                        retry_count += 1
                    handled += 1
                    message = record.get("message", [])
                    if isinstance(message, str):
                        message = [message]
                    for item in message:
                        log.write(str(item) + "\n")
                    elapsed = max(time.perf_counter() - start_time, 1e-9)
                    images_per_second = handled / elapsed
                    output = {
                        "kind": "progress",
                        "completed": handled,
                        "total": total,
                        "images_per_second": images_per_second,
                    }
                    if progress is not None:
                        progress(output)
                    print(
                        f"Processed {handled}/{total} covers "
                        f"({images_per_second:.2f} images/s)",
                        flush=True,
                    )
                    if handled % 500 == 0:
                        _write_metadata(output_root / "metadata.csv", records)
                    next_to_write += 1
        except KeyboardInterrupt:
            cancelled = True
        finally:
            executor.shutdown(wait=not cancelled, cancel_futures=cancelled)
        _write_metadata(output_root / "metadata.csv", records)
        if cancelled:
            log.write(f"Run cancelled: {completed}/{total} covers complete\n")
            return {"status": "cancelled", "total": total, "completed": completed}
        if retry_count:
            log.write(f"Run incomplete: {retry_count} covers will be retried\n")
            return {"status": "failed", "total": total, "completed": completed}
        log.write(f"Run completed: {completed}/{total} covers complete\n")
    return {"status": "completed", "total": total, "completed": completed}


def _metadata_from_png(path: Path) -> dict[str, Any]:
    """Read the stego JSON text chunk from one PNG."""
    try:
        with Image.open(path) as image:
            text = image.text.get("stego")
    except Exception as exc:
        raise ValueError(f"cannot read metadata from {path.name}: {exc}") from exc
    if not text:
        raise ValueError(f"{path.name} has no stego metadata")
    metadata = json.loads(text)
    if not isinstance(metadata, dict) or metadata.get("label") != "stego":
        raise ValueError(f"{path.name} does not contain stego metadata")
    return metadata


def replay(
    clean: np.ndarray,
    metadata: Mapping[str, Any],
) -> tuple[np.ndarray, np.ndarray, EmbedResult]:
    """Replay one stego image from its clean pixels and PNG metadata."""
    seed_data = metadata.get("embed_seed")
    if not isinstance(seed_data, dict):
        raise ValueError("metadata has no embed seed")
    entropy = seed_data.get("entropy")
    spawn_key = tuple(int(value) for value in seed_data.get("spawn_key", []))
    sequence = np.random.SeedSequence(entropy=entropy, spawn_key=spawn_key)
    start_random = metadata.get("start_random")
    if type(start_random) is not bool:
        raise ValueError("metadata has no start mode")
    start = None if start_random else metadata.get("start")
    kwargs: dict[str, Any] = {
        "bit_mode": metadata["bit_mode"],
        "k": int(metadata["k"]),
        "method": metadata["method"],
        "placement": metadata["placement"],
        "rng": np.random.default_rng(sequence),
        "start": start,
        "stride": metadata.get("stride") or 1,
        "payload": metadata["payload_kind"],
        "payload_bytes": int(metadata["payload_bytes"]),
    }
    return embed(clean, **kwargs)


def regenerate_payload(
    stego: np.ndarray,
    metadata: Mapping[str, Any],
    config: Config | None = None,
    carrier: np.ndarray | None = None,
) -> tuple[bytes, EmbedResult]:
    """Regenerate one expected payload without using the run configuration."""
    del config
    source = np.zeros_like(stego) if carrier is None else carrier
    generated, _, result = replay(source, metadata)
    return extract(generated, result), result


def verify_output(output: Path) -> dict[str, int]:
    """Verify every stego PNG by replaying pixels and payload."""
    output = Path(output)
    stego_paths = sorted((output / "stego").glob("*.png"), key=lambda path: path.name)
    passed = 0
    failed = 0
    log_lines: list[str] = []
    for path in stego_paths:
        try:
            metadata = _metadata_from_png(path)
            image = load_rgb_png(path)
            clean_path = output / "clean" / f"{metadata['cover_id']}.png"
            clean = load_rgb_png(clean_path)
            replayed, _, result = replay(clean, metadata)
            if not np.array_equal(replayed, image):
                raise ValueError("stored stego pixels differ from replay")
            for field, value in _result_fields(result).items():
                if metadata.get(field) != value:
                    raise ValueError(f"metadata field {field} does not match")
            expected_payload = extract(replayed, result)
            actual_payload = extract(image, result)
            if actual_payload != expected_payload:
                raise ValueError("extracted payload differs from regenerated payload")
            if len(expected_payload) != int(metadata["payload_bytes"]):
                raise ValueError("payload length does not match metadata")
            passed += 1
        except Exception as exc:
            failed += 1
            log_lines.append(f"FAIL {path.name}: {exc}")
    report = {"passed": passed, "failed": failed, "total": len(stego_paths)}
    with (output / "run.log").open("a", encoding="utf-8") as stream:
        stream.write(
            f"Verify: {passed} passed, {failed} failed\n"
            + "\n".join(log_lines)
            + ("\n" if log_lines else "")
        )
    return report


__all__ = [
    "CSV_FIELDS",
    "Config",
    "VERSION",
    "generate",
    "load_rgb_png",
    "read_config",
    "regenerate_payload",
    "replay",
    "verify_output",
    "write_config",
]
