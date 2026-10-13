"""Full-photo sources and manifest-backed tile views; no cache or embedding."""
from __future__ import annotations

import os
import sys
import zlib
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Iterator, Mapping

import numpy as np
from PIL import Image

from lsb_raw import decode_raw_pixels


@dataclass(frozen=True)
class SourceTile:
    """One real tile view with a stable geometry identity, before augmentation."""
    pixels: np.ndarray
    photo_id: str
    grid_policy: str
    tile_size: int
    row: int
    column: int
    y: int
    x: int

    @property
    def identity(self) -> tuple:
        return (self.photo_id, self.grid_policy, self.tile_size,
                self.row, self.column, self.y, self.x)


def configure_source_worker() -> None:
    """Set OMP=1 before rawpy loads. Call in each newly spawned worker.

    Also export OMP_NUM_THREADS=1 in the launcher before Python starts.
    Setting it after rawpy/LibRaw loads cannot reconfigure OpenMP safely.
    """
    if "rawpy" in sys.modules and os.environ.get("OMP_NUM_THREADS") != "1":
        raise RuntimeError("rawpy already loaded; start a worker with OMP_NUM_THREADS=1")
    os.environ["OMP_NUM_THREADS"] = "1"


def _integer(record: Mapping, name: str) -> int:
    value = record[name]
    if isinstance(value, bool):
        raise ValueError(f"{name} must be an integer")
    result = int(value)
    if str(result) != str(value):
        raise ValueError(f"{name} must be a canonical integer")
    return result


def _shape(record: Mapping) -> tuple[int, int, int]:
    h, w = _integer(record, "source_height"), _integer(record, "source_width")
    if min(h, w) < 1:
        raise ValueError("source dimensions must be positive")
    return h, w, 3


def _validate_pixels(pixels: np.ndarray, record: Mapping) -> None:
    if pixels.dtype != np.uint8 or pixels.shape != _shape(record):
        raise ValueError(f"decoded RGB8 shape does not match manifest: {record['photo_id']}")


def _png8(path: Path) -> np.ndarray:
    with path.open("rb") as stream:
        header = stream.read(33)
    if (len(header) != 33 or header[:16] != b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR"
            or header[24:26] != bytes((8, 2))
            or zlib.crc32(header[12:29]) != int.from_bytes(header[29:33], "big")):
        raise ValueError("png8 requires an 8-bit RGB PNG header")
    with Image.open(path) as image:
        if image.format != "PNG" or image.mode != "RGB" or getattr(image, "n_frames", 1) != 1:
            raise ValueError("png8 requires a single RGB image; no mode conversion")
        if "transparency" in image.info:
            raise ValueError("png8 transparency is unsupported")
        # Preserve stored orientation, gamma and RGB values; no EXIF transpose.
        image.load()
        pixels = np.array(image, copy=True)
    return np.ascontiguousarray(pixels)


def load_photo(record: Mapping, repo_root: Path) -> np.ndarray:
    """Return full oriented uint8 (H,W,3), with strict source/size validation."""
    relative = PurePosixPath(record["master_path"])
    if relative.is_absolute() or ".." in relative.parts or "\\" in str(relative):
        raise ValueError("master_path must be repository-relative")
    path = Path(repo_root) / relative
    source_type = record["source_type"]
    if source_type == "nef":
        if record.get("noise") != "disabled":
            raise ValueError("RAW noise is not implemented")
        configure_source_worker()
        pixels = decode_raw_pixels(path)
    elif source_type == "png8":
        if record.get("noise") != "not_applicable":
            raise ValueError("png8 cannot use RAW noise")
        pixels = _png8(path)
    else:
        raise ValueError(f"unsupported source_type: {source_type}")
    _validate_pixels(pixels, record)
    return pixels


def overlap_starts(length: int, tile_size: int) -> tuple[int, ...]:
    """Full real-pixel coverage; move only the last tile to the photo edge."""
    if type(length) is not int or type(tile_size) is not int or tile_size < 1 or length < tile_size:
        raise ValueError("positive tile size and source side >= tile size required")
    count = (length + tile_size - 1) // tile_size
    return tuple(i * tile_size for i in range(count - 1)) + (length - tile_size,)


def grid_axes(record: Mapping, grid_policy: str, tile_size: int) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Use frozen overlap starts; validate all counts and legacy dimensions."""
    h, w, _ = _shape(record)
    if grid_policy == "overlap_last":
        if tile_size not in (512, 256) or record["grid_policy"] != "overlap_last":
            raise ValueError("overlap requires the recorded 512 or 256 grid")
        axes = []
        for axis, dimension, count_field in (("y", h, "rows"), ("x", w, "cols")):
            values = tuple(int(v) for v in record[f"overlap{tile_size}_{axis}"].split(";"))
            if (values != overlap_starts(dimension, tile_size)
                    or len(values) != _integer(record, f"overlap{tile_size}_{count_field}")):
                raise ValueError(f"invalid recorded overlap {axis} starts/count")
            axes.append(values)
        return axes[0], axes[1]
    if grid_policy == "topleft_crop":
        if tile_size != 512 or record["source"] != "raise" or record["legacy512_policy"] != "topleft_crop":
            raise ValueError("legacy grid is RAISE topleft_crop 512 only")
        rows, cols = _integer(record, "legacy512_rows"), _integer(record, "legacy512_cols")
        if (min(rows, cols) < 1 or (rows, cols) != (h // 512, w // 512)
                or _integer(record, "legacy512_height") != rows * 512
                or _integer(record, "legacy512_width") != cols * 512):
            raise ValueError("invalid legacy grid/crop dimensions")
        return tuple(r * 512 for r in range(rows)), tuple(c * 512 for c in range(cols))
    raise ValueError(f"unsupported grid_policy: {grid_policy}")


def topleft_crop(pixels: np.ndarray, record: Mapping) -> np.ndarray:
    """Legacy-only cropped view, with no resize or padding."""
    _validate_pixels(pixels, record)
    ys, xs = grid_axes(record, "topleft_crop", 512)
    return pixels[:ys[-1] + 512, :xs[-1] + 512]


def iter_tiles(pixels: np.ndarray, record: Mapping, grid_policy: str = "overlap_last",
               tile_size: int = 512) -> Iterator[SourceTile]:
    """Yield row-major tile views and identities. Do not mutate these views."""
    _validate_pixels(pixels, record)
    ys, xs = grid_axes(record, grid_policy, tile_size)
    for r, y in enumerate(ys):
        for c, x in enumerate(xs):
            yield SourceTile(pixels[y:y + tile_size, x:x + tile_size], record["photo_id"],
                             grid_policy, tile_size, r, c, y, x)
