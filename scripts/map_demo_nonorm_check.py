#!/usr/bin/env python3
"""Check that no-norm suspicion cells do not change in unaffected chunks.

Use the fixed-settings R1 checkpoint and the same seeds as map_demo.py.
Run in the GPU container; write only to runs-out/map-demo-r1/.
"""
import csv
from pathlib import Path

import numpy as np
import torch

from map_demo import (ROOT, RATES, XuNet, chunk_map, input_photos,
                      read_photo, starts, cell_mask)
from lsb_engine import embed_fast

OUTPUT = ROOT / 'runs-out/map-demo-r1'
CHECKPOINT = next((ROOT / 'runs-out/g4-v4-r1b/checkpoints').glob('*/best-*.ckpt'))


def owner_mask(changed, shape):
    """Mark cells whose source window has changed pixels, matching chunk_map ownership."""
    rows, cols = shape
    h, w = rows * 32, cols * 32
    ys, xs = starts(h), starts(w)
    mask = np.zeros(shape, dtype=bool)
    filled = np.zeros(shape, dtype=bool)
    for y in ys:
        for x in xs:
            y0 = 0 if y == 0 else (y + 64) // 32
            y1 = rows if y == ys[-1] else (y + 448) // 32
            x0 = 0 if x == 0 else (x + 64) // 32
            x1 = cols if x == xs[-1] else (x + 448) // 32
            available = ~filled[y0:y1, x0:x1]
            mask[y0:y1, x0:x1][available] = bool(changed[y:y + 512, x:x + 512].any())
            filled[y0:y1, x0:x1] = True
    assert filled.all()
    return mask


def main():
    torch.set_num_threads(1)
    device = torch.device('cuda')
    ckpt = torch.load(CHECKPOINT, map_location='cpu', weights_only=False)
    hp = ckpt['hyper_parameters']
    assert hp['norm'] == 'none'
    model = XuNet(clip_t=hp['clip_t'], norm=hp['norm']).to(device).eval()
    model.load_state_dict({key.removeprefix('net.'): val
                           for key, val in ckpt['state_dict'].items()}, strict=True)
    results = []
    for index, (source, photo_id, row, camera) in enumerate(input_photos()):
        cover = read_photo(source, row)
        clean = chunk_map(model, cover, device)
        for group, k in (('A', 1), ('B', 4)):
            base = dict(bit_mode='lowest', k=k, method='match', payload='random_bytes')
            for rate in RATES:
                rng = np.random.default_rng(np.random.SeedSequence([0, index, 0, int(rate * 1000), 102]))
                stego, _, _ = embed_fast(cover, **base, placement='contiguous', start=None,
                                         rng=rng, bpp=rate, return_changed_units=False)
                changed = np.any(cover != stego, axis=2)
                stego_map = chunk_map(model, stego, device)
                owned = owner_mask(changed, clean.shape)
                outside = ~owned
                delta = np.abs(stego_map - clean)
                result = dict(photo_id=photo_id, group=group, bpp=rate,
                              cells_outside_changed_chunk=int(outside.sum()),
                              outside_max_abs=float(delta[outside].max()) if outside.any() else None,
                              cells_inside_unchanged=int((owned & ~cell_mask(changed, clean.shape)).sum()),
                              inside_max_abs=float(delta[owned].max()) if owned.any() else None)
                results.append(result)
                print('CASE', result, flush=True)
        del cover
        torch.cuda.empty_cache()
    with (OUTPUT / 'nonorm_check.csv').open('w', newline='') as out:
        writer = csv.DictWriter(out, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)
    print('DONE', len(results), flush=True)


if __name__ == '__main__':
    main()
