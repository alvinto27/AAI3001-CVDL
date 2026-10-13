"""Shared settings draws and versioned generation seeds.

Legacy helpers retain their original random calls and failure rules. Only
OTF callers opt into lower-to-fit size draws. No image embedding occurs here.
"""
from __future__ import annotations

from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Callable

import numpy as np

if TYPE_CHECKING:
    from lsb_batch import Config


class SizeDoesNotFit(ValueError):
    """The minimum configured payload cannot fit one embedding."""


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


def _seed_sequence(seed: int, cover_index: int, variant: int, stream: int) -> np.random.SeedSequence:
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
    *,
    size_draw: Callable = _draw_size,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Draw settings in unchanged legacy order; OTF may replace only size draw."""
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
    size_kind, size_value = size_draw(config.size, image, maximum, rng)
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


legacy_seed_sequence = _seed_sequence
draw_legacy_settings = _draw_settings
OTF_KEY_VERSION = 1
FIXED_EVALUATION_SEED = 1
PURPOSE_CODES = MappingProxyType({"train": 1, "val": 2, "test": 3})
GRID_CODES = MappingProxyType({"none": 0, "topleft_crop": 1, "overlap_last": 2})
CARRIER_CODES = MappingProxyType({"tile": 1, "photo": 2})
STREAM_CODES = MappingProxyType({"settings": 0, "embedding": 1})


def otf_seed_sequence(*, run_seed: int, purpose: str, epoch: int, photo_index: int,
                      grid_policy: str, tile_size: int, tile_row: int, tile_col: int,
                      variant: int, stream: str, carrier: str = "tile",
                      fixed_variants: bool = False) -> np.random.SeedSequence:
    """Explicit v1 key. Evaluation excludes run seed/epoch; photo keys omit grid.

    Order: version, purpose, seed, epoch, carrier, photo_index, grid_id,
    tile_size, tile_row, tile_col, variant, stream. Mode 2 uses carrier=photo.
    """
    if type(fixed_variants) is not bool:
        raise ValueError("fixed_variants must be Boolean")
    if purpose not in PURPOSE_CODES or stream not in STREAM_CODES or carrier not in CARRIER_CODES:
        raise ValueError("unknown purpose, stream, or carrier")
    seed = run_seed if purpose == "train" else FIXED_EVALUATION_SEED
    generation_epoch = epoch if purpose == "train" and not fixed_variants else 0
    if grid_policy not in GRID_CODES:
        raise ValueError("unknown grid policy")
    grid_id = GRID_CODES[grid_policy]
    if carrier == "photo":
        grid_id, tile_size, tile_row, tile_col = 0, 0, 0, 0
    elif (grid_policy == "none" or tile_size not in (256, 512)
          or (grid_policy == "topleft_crop" and tile_size != 512)):
        raise ValueError("tile keys require overlap 256/512 or top-left 512")
    values = [OTF_KEY_VERSION, PURPOSE_CODES[purpose], seed, generation_epoch,
              CARRIER_CODES[carrier], photo_index, grid_id, tile_size,
              tile_row, tile_col, variant, STREAM_CODES[stream]]
    if any(type(v) is not int or v < 0 for v in values) or variant < 1:
        raise ValueError("seed key parts must be non-negative integers; variant starts at 1")
    return np.random.SeedSequence(values)


def draw_otf_settings(config: Config, image: np.ndarray,
                      rng: np.random.Generator) -> tuple[dict[str, Any], dict[str, Any]]:
    """Draw requested size, round to bytes, then lower overflow to capacity.

    Unlike legacy upper-bound clipping, new ranges draw their requested value
    first. Record it separately from effective bytes/bpp. Below-byte requests
    retain v6's one-byte minimum; a carrier unable to hold one byte fails.
    """
    recorded = {}

    def lower_size(size, image, capacity, rng):
        if capacity < 1:
            raise SizeDoesNotFit("carrier cannot hold the minimum one-byte payload")
        kind = next(iter(size))
        value = size[kind]
        if isinstance(value, list):
            value = (int(rng.integers(value[0], value[1] + 1)) if kind == "payload_bytes"
                     else float(rng.uniform(value[0], value[1])))
        pixels = int(image.shape[0] * image.shape[1])
        requested_bytes = int(value) if kind == "payload_bytes" else _bpp_payload_bytes(float(value), pixels)
        effective_bytes = min(requested_bytes, capacity)
        requested_bpp = float(value) if kind == "bpp" else requested_bytes * 8.0 / pixels
        effective_bpp = effective_bytes * 8.0 / pixels
        recorded.update(requested_size_kind=kind, requested_size_value=value,
                        requested_bpp=requested_bpp, requested_payload_bytes=requested_bytes,
                        payload_bytes=effective_bytes, effective_bpp=effective_bpp,
                        capacity_payload_bytes=capacity, clipped=effective_bytes < requested_bytes)
        return kind, effective_bytes if kind == "payload_bytes" else effective_bpp

    kwargs, metadata = _draw_settings(config, image, rng, size_draw=lower_size)
    metadata.update(recorded)
    return kwargs, metadata
