"""Strict RGB PNG LSB dataset generation. No UI dependencies."""
from __future__ import annotations

import warnings
import string
from pathlib import Path

import numpy as np
from PIL import Image

VERSION = "5.0.0"
PAYLOAD_ENCODING = "ASCII"
MESSAGE_BIT_ORDER = "most_significant_bit_first"
TEXT_POLICY = "uniform_independent_A-Z_a-z"
LETTER_BYTES = np.frombuffer(string.ascii_letters.encode("ascii"), dtype=np.uint8)
PLACEMENT_POLICY = "sequential_random_start_no_wrap"
MAX_PIXELS = 16_000_000


def validate_character_count(character_count: int) -> None:
    """Require a positive whole-number count, excluding Boolean values."""
    if type(character_count) is not int or character_count < 1:
        raise ValueError("Character count must be a positive whole number.")


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


def embed(clean: np.ndarray, character_count: int,
          rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, int, int]:
    """Return stego, Boolean H × W selected-region mask, start channel and length.

    Selection is one nonwrapping interval in row-major, interleaved RGB order.
    The region mask includes selected pixels even when their values do not change.
    """
    validate_array(clean)
    validate_character_count(character_count)
    count = 8 * character_count
    if count > clean.size:
        raise ValueError(f"Requested {character_count} characters exceeds capacity "
                         f"of {clean.size // 8} characters ({clean.size} bits).")
    start = int(rng.integers(0, clean.size - count + 1))
    end = start + count
    letter_indices = rng.integers(0, len(LETTER_BYTES), size=character_count)
    message_bytes = LETTER_BYTES[letter_indices]
    payload = np.unpackbits(message_bytes, bitorder="big")
    stego = clean.copy()
    flat = stego.reshape(-1)
    flat[start:end] = (flat[start:end] & np.uint8(254)) | payload
    mask = np.zeros(clean.shape[:2], dtype=np.bool_)
    mask.reshape(-1)[start // 3:(end - 1) // 3 + 1] = True
    return stego, mask, start, count
