"""Strict RGB PNG LSB dataset generation. No UI dependencies."""
from __future__ import annotations

import warnings
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

import numpy as np
from PIL import Image

VERSION = "3.0.0"
MAX_PIXELS = 16_000_000


def payload_length(rate: float, capacity: int) -> int:
    """Number of LSB positions used by ``rate`` of ``capacity``, rounded half up."""
    value = Decimal(str(rate))
    if not value.is_finite() or not 0 < value <= 1:
        raise ValueError("Payload rate must be greater than 0 and at most 1.")
    return int((value * capacity).quantize(Decimal("1"), rounding=ROUND_HALF_UP))


def validate_array(array: np.ndarray) -> None:
    """Require a decoded H × W × 3 uint8 RGB array."""
    if array.dtype != np.uint8 or array.ndim != 3 or array.shape[2] != 3:
        raise ValueError("Expected an H × W × 3 uint8 RGB array.")
    if min(array.shape[:2]) < 1 or array.shape[0] * array.shape[1] > MAX_PIXELS:
        raise ValueError(f"Image must contain 1 to {MAX_PIXELS:,} pixels.")


def require_8bit_rgb_png(path: Path) -> None:
    """Check the PNG signature and IHDR encoding before the image is decoded.

    Pillow reports a 16-bit RGB PNG as mode ``RGB`` and silently reduces it to
    8 bits, so the decoded array alone cannot prove that the source was 8-bit.
    """
    try:
        with path.open("rb") as stream:
            header = stream.read(26)
    except OSError as exc:
        raise ValueError(f"Could not read the file: {exc}") from exc
    if len(header) < 26 or header[:8] != b"\x89PNG\r\n\x1a\n" or header[12:16] != b"IHDR":
        raise ValueError("Not a valid PNG file.")
    if header[24] != 8 or header[25] != 2:
        raise ValueError("Only 8-bit RGB PNG is supported; no alpha, palette, grayscale or 16-bit input.")


def load_rgb(path: Path) -> np.ndarray:
    """Decode a static 8-bit RGB PNG. No format, mode or size conversion is applied."""
    require_8bit_rgb_png(path)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(path) as image:
                if image.format != "PNG" or image.mode != "RGB":
                    raise ValueError("Only static 8-bit RGB PNG images are supported.")
                width, height = image.size
                if width < 1 or height < 1 or width * height > MAX_PIXELS:
                    raise ValueError(f"Image must contain 1 to {MAX_PIXELS:,} pixels.")
                if getattr(image, "n_frames", 1) != 1 or "transparency" in image.info:
                    raise ValueError("Animated PNG and PNG transparency are unsupported.")
                image.load()
                array = np.asarray(image)
    except ValueError:
        raise
    except Exception as exc:  # unreadable, truncated or unsupported file
        raise ValueError(f"Could not read a static 8-bit RGB PNG: {exc}") from exc
    validate_array(array)
    return array


def embed(clean: np.ndarray, rate: float, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Replace LSBs at ``rate`` of the flat RGB positions with random payload bits.

    Returns the stego image, the exact H × W × 3 mask of channels that changed and
    the selected flat RGB-channel indices, which may include positions whose value
    did not change.
    """
    validate_array(clean)
    count = payload_length(rate, clean.size)
    payload = rng.integers(0, 2, size=count, dtype=np.uint8)
    selected = rng.choice(clean.size, size=count, replace=False).astype(np.int64)
    stego = clean.copy()
    flat = stego.reshape(-1)
    flat[selected] = (flat[selected] & np.uint8(254)) | payload
    return stego, clean != stego, selected
