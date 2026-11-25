"""Changed RGB-unit indices -> small 256-grid training targets; no RNG or I/O."""
import numpy as np

VERSION = 'changed_cells_v1'
CELL_SIZE = 32
GRID = 8
TAG_COLUMNS = 130


def changed_cell_targets(changed_units, shape=(256, 256), pad=None):
    """Return unique changed-pixel counts and valid-negative bits (both 8x8).

    Indices refer to the unpadded RGB array passed to embed_fast. Padding is
    applied only to coordinates, exactly as Mode1 pads the resulting image.
    A negative is valid only if its entire clipped 48x48 receptive field is clean.
    """
    h, w = shape[:2]
    top, bottom, left, right = (0, 0, 0, 0) if pad is None else pad
    if h+top+bottom != 256 or w+left+right != 256 or min(h, w) <= 0:
        raise ValueError('cell targets require a padded 256x256 RGB tile')
    units = np.asarray(changed_units)
    if units.ndim != 1 or (units.size and (units.dtype.kind not in 'iu' or units.min() < 0 or units.max() >= h*w*3)):
        raise ValueError('invalid changed RGB-unit indices')
    pixels = np.unique(units.astype(np.int64, copy=False)//3)
    y, x = pixels//w+top, pixels%w+left
    rows, cols = y//32, x//32
    counts = np.bincount(rows*8+cols, minlength=64).reshape(8, 8).astype(np.uint16)
    # Each pixel can touch its own field and at most one neighbour per axis.
    dy = np.where(y%32 < 8, -1, np.where(y%32 >= 24, 1, 0))
    dx = np.where(x%32 < 8, -1, np.where(x%32 >= 24, 1, 0))
    touched = np.zeros((8, 8), dtype=bool)
    for rr, cc in ((rows, cols), (rows+dy, cols), (rows, cols+dx), (rows+dy, cols+dx)):
        inside = (rr >= 0) & (rr < 8) & (cc >= 0) & (cc < 8)
        touched[rr[inside], cc[inside]] = True
    return counts, ~touched
