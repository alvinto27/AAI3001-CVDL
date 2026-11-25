"""Mode 2: embed one full photo, then label each real overlap tile.

No source reader, writer, training loader, augmentation or photo model is here.
Call the shared source/cache reader first. Generation has no tile-size input.
"""
from __future__ import annotations

from typing import Mapping, Iterator

import numpy as np

from lsb_source import _validate_pixels, grid_axes, iter_tiles
from lsb_settings import otf_seed_sequence, draw_otf_settings
from lsb_engine import embed_fast
from lsb_mode1 import OTFSeed

ALGORITHM_VERSION = "mode2_full_v1"


def embed_photo(photo: np.ndarray, record: Mapping, config,
                seed: OTFSeed) -> tuple[np.ndarray, dict, np.ndarray]:
    """Return an owned stego photo, JSON-safe record and changed RGB indices.

    Indices refer to interleaved row-major RGB values, not selected values or
    distinct pixels. The clean input is never mutated. Evaluation ignores the
    caller's run seed/epoch, exactly as the shared Mode 1 OTF seed path does.
    """
    _validate_pixels(photo, record)
    if not isinstance(seed, OTFSeed):
        raise TypeError("Mode 2 requires OTFSeed")
    arguments = dict(run_seed=seed.run_seed, purpose=seed.purpose, epoch=seed.epoch,
                     photo_index=int(record["photo_index"]), grid_policy="none",
                     tile_size=0, tile_row=0, tile_col=0, variant=seed.variant,
                     carrier="photo", fixed_variants=seed.fixed_variants)
    settings_key = otf_seed_sequence(**arguments, stream="settings")
    embedding_key = otf_seed_sequence(**arguments, stream="embedding")
    kwargs, drawn = draw_otf_settings(config, photo, np.random.default_rng(settings_key))
    settings = {**{k: v for k, v in kwargs.items() if k != "rng"}, **drawn}
    kwargs["rng"] = np.random.default_rng(embedding_key)
    stego, embedded, changed = embed_fast(photo, **kwargs, return_changed_units=True)
    metadata = dict(mode=2, algorithm_version=ALGORITHM_VERSION, numpy_version=np.__version__, carrier="photo",
                    photo_id=record["photo_id"], photo_index=int(record["photo_index"]),
                    source=record["source"], source_type=record["source_type"],
                    noise=record["noise"], split=record["split"],
                    master_path=record["master_path"], camera_model=record.get("camera_model", ""),
                    source_height=int(photo.shape[0]), source_width=int(photo.shape[1]),
                    photo_pixels=int(photo.shape[0]*photo.shape[1]),
                    variant=seed.variant, purpose=seed.purpose,
                    variant_policy="fixed" if seed.purpose != "train" or seed.fixed_variants else "per_epoch",
                    generation_epoch=int(settings_key.entropy[3]),
                    generation_seed=int(settings_key.entropy[2]), seed_version="otf_v1", seed_scheme="otf_v1",
                    settings_seed_key=list(settings_key.entropy),
                    embedding_seed_key=list(embedding_key.entropy),
                    photo_embedded=True, actual_changed=bool(embedded["units_changed"]),
                    settings=settings, embedding=embedded)
    return stego, metadata, changed


def tile_change_counts(changed_units: np.ndarray, record: Mapping, tile_size: int,
                       *, block_size: int = 65536) -> tuple[np.ndarray, np.ndarray]:
    """Return row/column grids of changed-value and distinct changed-pixel counts.

    Accept strictly increasing unique RGB indices, as returned by embed_fast.
    Map bounded blocks to the last containing axis interval and its predecessor.
    A shared edge/corner pixel counts in every containing tile. Three changed
    channels count as three values but only one pixel, including block edges.
    No full-photo RGB comparison or difference mask is made.
    """
    if (not isinstance(changed_units, np.ndarray) or changed_units.ndim != 1
            or changed_units.dtype.kind not in "iu"):
        raise ValueError("changed units must be a one-dimensional integer array")
    if type(block_size) is not int or block_size < 1:
        raise ValueError("block size must be a positive integer")
    ys, xs = grid_axes(record, "overlap_last", tile_size)
    ys, xs = np.asarray(ys), np.asarray(xs)
    height, width = int(record["source_height"]), int(record["source_width"])
    total_units = height*width*3
    values = np.zeros((len(ys), len(xs)), dtype=np.int64)
    pixels = np.zeros_like(values)
    previous_unit = previous_pixel = -1
    for offset in range(0, len(changed_units), block_size):
        units = changed_units[offset:offset+block_size]
        if (int(units[0]) <= previous_unit or int(units[-1]) >= total_units
                or np.any(units[1:] <= units[:-1])):
            raise ValueError("changed units must be sorted, unique and inside the photo")
        p = units.astype(np.int64, copy=False)//3
        distinct = np.empty(len(p), dtype=bool)
        distinct[0] = int(p[0]) != previous_pixel
        distinct[1:] = p[1:] != p[:-1]
        y, x = p//width, p%width
        last_row = np.searchsorted(ys, y, side="right")-1
        last_col = np.searchsorted(xs, x, side="right")-1
        for row_shift in (0, 1):
            row = last_row-row_shift
            safe_row = np.maximum(row, 0)
            row_valid = (row >= 0) & (y < ys[safe_row]+tile_size)
            for col_shift in (0, 1):
                col = last_col-col_shift
                safe_col = np.maximum(col, 0)
                valid = row_valid & (col >= 0) & (x < xs[safe_col]+tile_size)
                indices = row*len(xs)+col
                values += np.bincount(indices[valid], minlength=values.size).reshape(values.shape)
                pixels += np.bincount(indices[valid & distinct], minlength=pixels.size).reshape(pixels.shape)
        previous_unit, previous_pixel = int(units[-1]), int(p[-1])
    return values, pixels


def iter_photo_tiles(clean: np.ndarray, stego: np.ndarray, record: Mapping,
                     metadata: Mapping, changed_units: np.ndarray,
                     tile_size: int = 512) -> Iterator[tuple[np.ndarray, dict]]:
    """Yield owned clean/stego RGB8 tiles, including unchanged stego-photo tiles.

    Tile IDs follow photo, mode, grid, actual start, variant and output role.
    Each overlap rectangle is a separate sample; equal pixels are not merged.
    Side records distinguish photo embedding from the actual-change tile label.
    """
    _validate_pixels(clean, record)
    _validate_pixels(stego, record)
    if (metadata.get("mode") != 2 or metadata.get("photo_id") != record["photo_id"]
            or metadata.get("photo_index") != int(record["photo_index"])):
        raise ValueError("Mode 2 photo metadata does not match the source record")
    changed_values, changed_pixels = tile_change_counts(changed_units, record, tile_size)
    for clean_tile, stego_tile in zip(iter_tiles(clean, record, tile_size=tile_size),
                                     iter_tiles(stego, record, tile_size=tile_size)):
        row, col = clean_tile.row, clean_tile.column
        for kind, tile in (("clean", clean_tile), ("stego", stego_tile)):
            value_count = int(changed_values[row, col]) if kind == "stego" else 0
            pixel_count = int(changed_pixels[row, col]) if kind == "stego" else 0
            tile_record = dict(metadata, grid_policy="overlap_last", tile_size=tile_size,
                               tile_row=row, tile_col=col, y=tile.y, x=tile.x, kind=kind,
                               photo_embedded=kind == "stego",
                               photo_actual_changed=kind == "stego" and bool(metadata["actual_changed"]),
                               actual_changed=bool(value_count),
                               label=int(value_count > 0), changed_values=value_count,
                               changed_pixels=pixel_count,
                               sample_id=f"{record['photo_id']}:mode2:overlap_last:{tile_size}:"
                                         f"{tile.y}:{tile.x}:v{metadata['variant']}:{kind}")
            yield tile.pixels.copy(), tile_record


__all__ = ["ALGORITHM_VERSION", "embed_photo", "tile_change_counts", "iter_photo_tiles"]
