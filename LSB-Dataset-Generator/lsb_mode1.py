"""Mode 1: validated tile -> shared settings -> mask-free embedding, no I/O."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np

from lsb_source import tile_at
from lsb_settings import (legacy_seed_sequence, otf_seed_sequence,
                          draw_legacy_settings, draw_otf_settings)
from lsb_engine import embed_fast


@dataclass(frozen=True)
class TileSpec:
    grid_policy: str
    tile_size: int
    row: int
    col: int


@dataclass(frozen=True)
class LegacySeed:
    seed: int
    cover_index: int
    variant: int = 1


@dataclass(frozen=True)
class OTFSeed:
    run_seed: int
    purpose: str
    epoch: int
    variant: int = 1
    fixed_variants: bool = False


def make_sample(photo: np.ndarray, record: Mapping, tile: TileSpec,
                config, seed: LegacySeed | OTFSeed, kind: str) -> tuple[np.ndarray, dict]:
    """Return an owned RGB8 sample and JSON-safe provenance/settings metadata.

    Clean samples draw no settings and have settings/embedding=None. Both seed
    keys identify the paired sample, even for clean. Effective entropy is used,
    so ignored evaluation run_seed/epoch do not change bytes OR metadata.
    Source views are never mutated; stego keeps no mask or changed-index array.
    """
    if kind not in ("clean", "stego"):
        raise ValueError("kind must be clean or stego")
    from lsb_train_grid import POLICIES, crop, pad_crop
    training_policy = next((k for k,v in POLICIES.items() if v == tile.grid_policy), None)
    pad = None
    if training_policy is not None:
        if not isinstance(seed, OTFSeed) or seed.purpose != 'train':
            raise ValueError('new grids are training-only')
        from lsb_source import SourceTile
        pixels, (y,x), pad = crop(photo, record, tile.tile_size, tile.row, tile.col, training_policy)
        view = SourceTile(pixels, record['photo_id'], tile.grid_policy, tile.tile_size, tile.row, tile.col, y, x)
    else:
        view = tile_at(photo, record, tile.grid_policy, tile.tile_size, tile.row, tile.col)
    if isinstance(seed, LegacySeed):
        if tile.grid_policy != "topleft_crop" or tile.tile_size != 512:
            raise ValueError("legacy keys require topleft_crop 512")
        if any(type(v) is not int or v < 0 for v in (seed.seed, seed.cover_index, seed.variant)) or seed.variant < 1:
            raise ValueError("legacy key parts must be non-negative integers; variant starts at 1")
        sequences = [legacy_seed_sequence(seed.seed, seed.cover_index, seed.variant, s) for s in (0, 1)]
        draw = draw_legacy_settings
        version = "legacy"
    elif isinstance(seed, OTFSeed):
        arguments = dict(run_seed=seed.run_seed, purpose=seed.purpose, epoch=seed.epoch,
                         photo_index=int(record["photo_index"]), grid_policy=tile.grid_policy,
                         tile_size=tile.tile_size, tile_row=tile.row, tile_col=tile.col,
                         variant=seed.variant, fixed_variants=seed.fixed_variants)
        sequences = [otf_seed_sequence(**arguments, stream=s) for s in ("settings", "embedding")]
        draw = draw_otf_settings
        version = "otf_v1"
    else:
        raise TypeError("seed must be LegacySeed or OTFSeed")
    metadata = dict(photo_id=record["photo_id"], photo_index=int(record["photo_index"]),
                    source=record["source"], split=record["split"], grid_policy=tile.grid_policy,
                    tile_size=tile.tile_size, tile_row=tile.row, tile_col=tile.col,
                    y=view.y, x=view.x, kind=kind, label=int(kind == "stego"),
                    seed_version=version, settings_seed_key=list(sequences[0].entropy),
                    embedding_seed_key=list(sequences[1].entropy), settings=None, embedding=None)
    if pad is not None:
        metadata.update(real_shape=list(view.pixels.shape[:2]), real_pixels=int(view.pixels.shape[0]*view.pixels.shape[1]), pad=list(pad))
    if kind == "clean":
        pixels = view.pixels.copy()
        return (pad_crop(pixels,pad) if pad is not None else pixels), metadata
    kwargs, drawn = draw(config, view.pixels, np.random.default_rng(sequences[0]))
    metadata["settings"] = {**{k: v for k, v in kwargs.items() if k != "rng"}, **drawn}
    kwargs["rng"] = np.random.default_rng(sequences[1])
    pixels, embedded, _ = embed_fast(view.pixels, **kwargs)
    metadata["embedding"] = embedded
    return (pad_crop(pixels,pad) if pad is not None else pixels), metadata


def replay_legacy(photo: np.ndarray, record: Mapping, legacy_row: Mapping,
                  config, kind: str, *, variant: int = 1) -> tuple[np.ndarray, dict]:
    """Thin frozen-manifest adapter; use run1 config.seed and legacy cover index."""
    if (legacy_row["photo_id"] != record["photo_id"] or legacy_row["source"] != record["source"]
            or legacy_row["split"] != record["split"]):
        raise ValueError("legacy row does not match photo manifest")
    pixels, metadata = make_sample(photo, record,
        TileSpec("topleft_crop", 512, int(legacy_row["tile_row"]), int(legacy_row["tile_col"])),
        config, LegacySeed(config.seed, int(legacy_row["cover_index"]), variant), kind)
    if (metadata["y"], metadata["x"]) != (int(legacy_row["pixel_y"]), int(legacy_row["pixel_x"])):
        raise ValueError("legacy row has inconsistent pixel coordinates")
    metadata.update(cover_id=legacy_row["cover_id"], cover_index=int(legacy_row["cover_index"]))
    return pixels, metadata


__all__ = ["TileSpec", "LegacySeed", "OTFSeed", "make_sample", "replay_legacy"]
