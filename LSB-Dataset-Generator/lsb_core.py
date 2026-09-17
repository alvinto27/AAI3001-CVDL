"""Deterministic, strict RGB PNG LSB dataset generation. No UI dependencies."""
from __future__ import annotations

import hashlib
import struct
import warnings
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path

import numpy as np
from PIL import Image

VERSION = "2.0.0"
MAX_PIXELS = 16_000_000


def payload_length(rate: float, capacity: int) -> int:
    value = Decimal(str(rate))
    if not value.is_finite() or not 0 < value <= 1:
        raise ValueError("Payload rate must be greater than 0 and at most 1.")
    return int((value * capacity).quantize(Decimal("1"), rounding=ROUND_HALF_UP))


def validate_array(array: np.ndarray) -> None:
    if array.dtype != np.uint8 or array.ndim != 3 or array.shape[2] != 3:
        raise ValueError("Expected an H × W × 3 uint8 RGB array.")
    if min(array.shape[:2]) < 1 or array.shape[0] * array.shape[1] > MAX_PIXELS:
        raise ValueError(f"Image must contain 1 to {MAX_PIXELS:,} pixels.")


def load_rgb(path: Path) -> np.ndarray:
    with path.open("rb") as stream:
        header = stream.read(33)
    if len(header) != 33 or header[:8] != b"\x89PNG\r\n\x1a\n" or header[12:16] != b"IHDR":
        raise ValueError("Not a valid PNG header.")
    width, height = struct.unpack(">II", header[16:24])
    if header[24] != 8 or header[25] != 2:
        raise ValueError("Only 8-bit RGB PNG is supported; no alpha, palette or grayscale conversion.")
    if not 0 < width * height <= MAX_PIXELS or width == 0 or height == 0:
        raise ValueError(f"Image exceeds the {MAX_PIXELS:,}-pixel safety limit or has invalid dimensions.")
    with warnings.catch_warnings():
        warnings.simplefilter("error", Image.DecompressionBombWarning)
        with Image.open(path) as im:
            if im.format != "PNG" or im.mode != "RGB":
                raise ValueError("Expected RGB PNG.")
            if getattr(im, "n_frames", 1) != 1 or "transparency" in im.info:
                raise ValueError("Animated PNG and PNG transparency are unsupported.")
            im.verify()
        with Image.open(path) as im:
            im.load()
            result = np.array(im, dtype=np.uint8)
    validate_array(result)
    return result


def source_identity(clean: np.ndarray) -> str:
    validate_array(clean)
    digest = hashlib.sha256(b"lsb-rgb8-v2\0")
    digest.update(struct.pack(">II", clean.shape[1], clean.shape[0]))
    digest.update(clean.tobytes(order="C"))
    return digest.hexdigest()


def derive_seed(run_seed: int, source_id: str, purpose: str = "embedding") -> int:
    raw = f"lsb-{VERSION}|{purpose}|{run_seed}|{source_id}".encode("ascii")
    return int.from_bytes(hashlib.sha256(raw).digest()[:16], "big")


def embed(clean: np.ndarray, rate: float, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    validate_array(clean)
    count = payload_length(rate, clean.size)
    rng = np.random.Generator(np.random.PCG64(seed))
    payload = rng.integers(0, 2, size=count, dtype=np.uint8)
    selected = rng.choice(clean.size, size=count, replace=False).astype(np.int64)
    stego = clean.copy()
    flat = stego.reshape(-1)
    flat[selected] = (flat[selected] & np.uint8(254)) | payload
    return stego, clean != stego, selected


def verify_arrays(clean: np.ndarray, stego: np.ndarray, mask: np.ndarray,
                  selected: np.ndarray, rate: float, seed: int) -> None:
    validate_array(clean)
    validate_array(stego)
    if clean.shape != stego.shape or mask.shape != clean.shape or mask.dtype != np.bool_:
        raise ValueError("Image/mask shape or mask dtype mismatch.")
    count = payload_length(rate, clean.size)
    if selected.ndim != 1 or selected.dtype.kind not in "iu" or len(selected) != count:
        raise ValueError("Invalid selected-location representation/count.")
    if count and (selected.min() < 0 or selected.max() >= clean.size):
        raise ValueError("Selected index out of bounds.")
    if np.unique(selected).size != count:
        raise ValueError("Repeated embedding location.")
    difference = clean != stego
    if not np.array_equal(mask, difference):
        raise ValueError("Modification mask does not exactly match actual differences.")
    if np.any((clean ^ stego) & np.uint8(254)):
        raise ValueError("A higher-order bit changed.")
    if np.any(np.abs(clean.astype(np.int16) - stego.astype(np.int16)) > 1):
        raise ValueError("A channel changed by more than one.")
    eligible = np.zeros(clean.size, dtype=np.bool_)
    eligible[selected] = True
    if np.any(difference.reshape(-1) & ~eligible):
        raise ValueError("A change occurred outside selected locations.")
    replay = embed(clean, rate, seed)
    if not all(np.array_equal(actual, expected) for actual, expected in zip((stego, mask, selected), replay)):
        raise ValueError("Seed replay does not reproduce the saved sample.")
