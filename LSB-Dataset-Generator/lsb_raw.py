"""Shared project RAW settings. Import rawpy only when decoding a RAW source."""
from __future__ import annotations

from pathlib import Path
from types import MappingProxyType

import numpy as np

RAW_SETTINGS = MappingProxyType({
    "use_camera_wb": True,
    "no_auto_bright": True,
    "output_color": "sRGB",
    "output_bps": 8,
    "demosaic_algorithm": "AHD",
    "orientation": "default",
})


def raw_settings_record() -> dict:
    """Return a writable copy for the unchanged v6 preparation record."""
    return dict(RAW_SETTINGS)


def raw_postprocess_kwargs(rawpy) -> dict:
    """Resolve enum names; leaving orientation unspecified keeps rawpy default."""
    settings = raw_settings_record()
    settings.pop("orientation")
    settings["output_color"] = getattr(rawpy.ColorSpace, settings["output_color"])
    settings["demosaic_algorithm"] = getattr(rawpy.DemosaicAlgorithm, settings["demosaic_algorithm"])
    return settings


def _disabled_raw_noise_hook(raw) -> None:
    """Reserved pre-demosaic hook. Do not read or modify sensor values now."""
    # A future enabled hook requires its own configuration and source version.
    pass


def decode_raw_pixels(path: Path) -> np.ndarray:
    """Decode full oriented RGB8 using the existing numerical path.

    For worker use, export OMP_NUM_THREADS=1 before any rawpy import. This
    shared function does not change the preparation caller's thread policy.
    """
    try:
        import rawpy
    except ImportError as exc:
        raise RuntimeError("rawpy is required to decode RAW files") from exc
    try:
        with rawpy.imread(str(path)) as raw:
            _disabled_raw_noise_hook(raw)
            pixels = raw.postprocess(**raw_postprocess_kwargs(rawpy))
    except Exception as exc:
        raise RuntimeError(f"RAW decode failed: {exc}") from exc
    if pixels.dtype != np.uint8 or pixels.ndim != 3 or pixels.shape[2] != 3:
        raise RuntimeError("RAW decoder did not return 8-bit RGB pixels")
    return np.ascontiguousarray(pixels)
