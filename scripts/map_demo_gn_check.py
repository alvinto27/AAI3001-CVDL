#!/usr/bin/env python3
"""Diagnose nonlocal GroupNorm influence in map_demo's contiguous edits.

Run inside aai3001 after the GPU/kernel preflight. No training or checkpoint writes.
"""
from __future__ import annotations

import csv
import math
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from map_demo import (CHECKPOINT, OUTPUT, PLACEMENTS, RATES, ROOT, XuNet, as_tensor,
                      cell_mask, chunk_map, input_photos, read_photo, roc_auc, starts)
from lsb_batch import read_config
from lsb_engine import embed_fast
from lsb_settings import draw_otf_settings

# Every one of the five notebook 04 normalizers is GroupNorm(8, channels).
NORM_NAMES = ('bn1', 'bn2', 'bn3', 'bn4', 'bn5')


def geometry(image):
    h, w = ((size // 32) * 32 for size in image.shape[:2])
    ys, xs = starts(h), starts(w)
    return h // 32, w // 32, ys, xs


def placement(canvas, owner, window, y, x, ys, xs, affected=None):
    """Use exactly map_demo.chunk_map's first-valid-window ownership rule."""
    rows, cols = canvas.shape
    y0 = 0 if y == 0 else (y + 64) // 32
    y1 = rows if y == ys[-1] else (y + 448) // 32
    x0 = 0 if x == 0 else (x + 64) // 32
    x1 = cols if x == xs[-1] else (x + 448) // 32
    dest = canvas[y0:y1, x0:x1]
    src = window[y0 - y // 32:y1 - y // 32, x0 - x // 32:x1 - x // 32]
    available = np.isnan(dest)
    dest[available] = src[available]
    if owner is not None:
        owner[y0:y1, x0:x1][available] = bool(affected)


def clean_stats(model, image, device):
    """Capture each clean chunk's exact input mean/biased variance per GN group."""
    rows, cols, ys, xs = geometry(image)
    canvas = np.full((rows, cols), np.nan, dtype=np.float32)
    stats = {}
    modules = [getattr(model, name) for name in NORM_NAMES]
    for y in ys:
        for x in xs:
            captured = []
            handles = []
            def capture(module, inputs):
                tensor = inputs[0]
                n, channels, height, width = tensor.shape
                assert n == 1 and channels % module.num_groups == 0
                grouped = tensor.reshape(n, module.num_groups, channels // module.num_groups,
                                         height, width)
                variance, mean = torch.var_mean(grouped, dim=(2, 3, 4), unbiased=False)
                captured.append((mean.detach(), variance.detach()))
            try:
                handles = [module.register_forward_pre_hook(capture) for module in modules]
                with torch.inference_mode():
                    out = model.cell_logits(as_tensor(image[y:y + 512, x:x + 512], device))
                window = (out[:, 1] - out[:, 0])[0].cpu().numpy()
            finally:
                for handle in handles:
                    handle.remove()
            if len(captured) != len(modules):
                raise AssertionError(f'captured only {len(captured)} GN layers')
            stats[y, x] = captured
            placement(canvas, None, window, y, x, ys, xs)
    if not np.isfinite(canvas).all():
        raise AssertionError('unfilled clean cells')
    return canvas, stats


def frozen_stats_map(model, image, changed, stats, device):
    """Keep the clean *chunk's* per-group statistics at all five GN layers."""
    rows, cols, ys, xs = geometry(image)
    canvas = np.full((rows, cols), np.nan, dtype=np.float32)
    affected_owner = np.zeros((rows, cols), dtype=bool)
    modules = [getattr(model, name) for name in NORM_NAMES]
    for y in ys:
        for x in xs:
            clean = stats[y, x]
            handles = []
            def frozen(index):
                def hook(module, inputs, output):
                    data = inputs[0]
                    n, channels, height, width = data.shape
                    grouped = data.reshape(n, module.num_groups, channels // module.num_groups,
                                           height, width)
                    mean, variance = clean[index]
                    normalized = ((grouped - mean[:, :, None, None, None]) /
                                  torch.sqrt(variance[:, :, None, None, None] + module.eps))
                    normalized = normalized.reshape_as(data)
                    return (normalized * module.weight[None, :, None, None] +
                            module.bias[None, :, None, None])
                return hook
            try:
                handles = [module.register_forward_hook(frozen(index))
                           for index, module in enumerate(modules)]
                with torch.inference_mode():
                    out = model.cell_logits(as_tensor(image[y:y + 512, x:x + 512], device))
                window = (out[:, 1] - out[:, 0])[0].cpu().numpy()
            finally:
                for handle in handles:
                    handle.remove()
            affected = bool(changed[y:y + 512, x:x + 512].any())
            placement(canvas, affected_owner, window, y, x, ys, xs, affected)
    if not np.isfinite(canvas).all():
        raise AssertionError('unfilled frozen cells')
    return canvas, affected_owner


def figure(photo_id, clean, ordinary, frozen, rate):
    difference = ordinary - clean
    lo, hi = np.percentile(np.concatenate([clean.ravel(), ordinary.ravel(), frozen.ravel()]),
                           (2, 98))
    if hi <= lo:
        hi = lo + 1e-6
    lim = max(float(np.percentile(np.abs(difference), 99)), 1e-6)
    fig, axes = plt.subplots(1, 4, figsize=(17, 4), dpi=115)
    for ax, image, title, lower, upper in zip(
            axes, (clean, ordinary, difference, frozen),
            ('clean map', 'stego map / ordinary GN', 'stego - clean',
             'stego map / CLEAN GN statistics'),
            (lo, lo, -lim, lo), (hi, hi, lim, hi)):
        ax.imshow(image, cmap='coolwarm', vmin=lower, vmax=upper,
                  interpolation='nearest')
        ax.set_title(title)
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle(f'{photo_id} contiguous bpp={rate:g} | raw stego-clean logits')
    fig.tight_layout()
    path = OUTPUT / f'figure-gn-{photo_id}-contiguous-0p275.png'
    fig.savefig(path)
    plt.close(fig)
    print('FIGURE', path.relative_to(ROOT), flush=True)


def main():
    if not torch.cuda.is_available():
        raise RuntimeError('GPU required')
    torch.set_num_threads(1)
    device = torch.device('cuda')
    model = XuNet(clip_t=3.0, norm='group').eval().to(device)
    ckpt = torch.load(CHECKPOINT, map_location='cpu', weights_only=False)
    model.load_state_dict({key.removeprefix('net.'): value
                           for key, value in ckpt['state_dict'].items()}, strict=True)
    config = read_config(ROOT / 'configs/raise512-run1.yaml')
    with (OUTPUT / 'metrics.csv').open(newline='') as stream:
        baseline = {(r['photo_id'], r['placement'], r['requested_bpp']): r
                    for r in csv.DictReader(stream)}
    results = []
    for index, (source, photo_id, row, camera) in enumerate(input_photos()):
        cover = read_photo(source, row)
        clean, stats = clean_stats(model, cover, device)
        old_clean = baseline[photo_id, 'clean', '']
        assert abs(float(old_clean['mean_margin']) - clean.mean()) < 2e-4
        if index == 0:
            frozen_clean, _ = frozen_stats_map(model, cover,
                              np.zeros(cover.shape[:2], dtype=bool), stats, device)
            clean_equivalence = float(np.max(np.abs(clean - frozen_clean)))
            print('PROOF clean GN vs clean frozen-GN max abs', clean_equivalence, flush=True)
            if clean_equivalence > 0.01:
                raise AssertionError('frozen GN does not reproduce clean forward')
        print('PHOTO', photo_id, 'clean cells', clean.size, 'clean GN chunks', len(stats), flush=True)
        settings_rng = np.random.default_rng(np.random.SeedSequence([0, index, 101]))
        settings, _ = draw_otf_settings(config, cover, settings_rng)
        base = {key: settings[key] for key in ('bit_mode', 'k', 'method', 'payload')}
        for rate in RATES:
            rng = np.random.default_rng(np.random.SeedSequence([0, index, 0,
                                          int(rate * 1000), 102]))
            stego, meta, _ = embed_fast(cover, **base, placement='contiguous', start=None,
                                       rng=rng, bpp=rate, return_changed_units=False)
            changed = np.any(cover != stego, axis=2)
            ordinary = chunk_map(model, stego, device)
            frozen, owner = frozen_stats_map(model, stego, changed, stats, device)
            old = baseline[photo_id, 'contiguous', str(rate)]
            assert abs(float(old['mean_margin']) - ordinary.mean()) < 2e-4
            positive = cell_mask(changed, ordinary.shape)
            unchanged = ~positive
            inside = unchanged & owner
            outside = unchanged & ~owner
            difference = np.abs(ordinary - clean)
            frozen_difference = np.abs(frozen - clean)
            def masked_stats(values, selected):
                if not selected.any():
                    return None, None
                return float(values[selected].mean()), float(values[selected].max())
            inside_mean, inside_max = masked_stats(difference, inside)
            outside_mean, outside_max = masked_stats(difference, outside)
            frozen_inside_mean, frozen_inside_max = masked_stats(frozen_difference, inside)
            frozen_outside_mean, frozen_outside_max = masked_stats(frozen_difference, outside)
            result = dict(photo_id=photo_id, source=source, camera_model=camera,
                          requested_bpp=rate, changed_pixels=int(changed.sum()),
                          changed_cells=int(positive.sum()),
                          unchanged_in_changed_chunks=int(inside.sum()),
                          unchanged_in_unchanged_chunks=int(outside.sum()),
                          ordinary_inside_mean_abs=inside_mean,
                          ordinary_inside_max_abs=inside_max,
                          ordinary_outside_mean_abs=outside_mean,
                          ordinary_outside_max_abs=outside_max,
                          frozen_inside_mean_abs=frozen_inside_mean,
                          frozen_inside_max_abs=frozen_inside_max,
                          frozen_outside_mean_abs=frozen_outside_mean,
                          frozen_outside_max_abs=frozen_outside_max,
                          ordinary_cell_auc=roc_auc(ordinary, positive),
                          frozen_cell_auc=roc_auc(frozen, positive),
                          frozen_changed_mean=float(frozen[positive].mean()),
                          frozen_unchanged_mean=float(frozen[~positive].mean()),
                          frozen_top1pct=float(np.partition(frozen.ravel(),
                                               -max(1, math.ceil(frozen.size * .01)))[
                                               -max(1, math.ceil(frozen.size * .01)):].mean()))
            results.append(result)
            def formatted(value):
                return '-' if value is None else f'{value:.8f}'
            print(f"CASE {photo_id} {rate:g} inside {formatted(inside_mean)}/"
                  f"{formatted(inside_max)} outside {formatted(outside_mean)}/"
                  f"{formatted(outside_max)} frozen_auc "
                  f"{result['frozen_cell_auc']:.8f}", flush=True)
            if photo_id in ('r00679daet', 'div2k_0802') and rate == 0.275:
                figure(photo_id, clean, ordinary, frozen, rate)
            del stego, changed, ordinary, frozen
        del cover
        torch.cuda.empty_cache()
    path = OUTPUT / 'gn_check.csv'
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(results[0]))
        writer.writeheader(); writer.writerows(results)
    print('DONE', len(results), 'cases', path.relative_to(ROOT), flush=True)


if __name__ == '__main__':
    main()
