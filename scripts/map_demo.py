#!/usr/bin/env python3
"""Demo last-layer XuNet suspicion cells on full-photo LSB modifications.

Run only in the PC GPU container: podman exec -w /work aai3001 python scripts/map_demo.py
Pass --fixed-settings for comparable k=1/k=4 figures in runs/map-demo-v2/.
The default retains the original drawn-settings demo in runs/map-demo/.
This is inference, not training.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import cvdl  # exposes the flat LSB library
from lsb_batch import read_config
from lsb_engine import embed_fast
from lsb_source import load_photo, grid_axes
from lsb_settings import draw_otf_settings

OUTPUT = ROOT / 'runs/map-demo'
# Points at the g4-prime-1 result folder: map-demo no longer keeps its own copy of this checkpoint.
CHECKPOINT = ROOT / 'runs/g4-prime-1/checkpoints/g4-prime-1/best-epoch08-val_pe0.0147-val_auc0.9973.ckpt'
RATES = (0.05, 0.275, 0.5)  # low, midpoint, high of the training [0.05, 0.5] bpp range
PLACEMENTS = ('contiguous', 'random')
from cvdl.models import KV_ARRAY, KVFilter, make_norm, XuNet, XuNetWide, model_from_hparams
from cvdl.metrics import roc_auc, sigmoid


def as_tensor(image, device):
    return torch.from_numpy(np.ascontiguousarray(image)).permute(2, 0, 1).float().unsqueeze(0).to(device)


def margin_map(model, images, device):
    batch = torch.cat([as_tensor(p, device) for p in images], dim=0)
    logits = model.cell_logits(batch)
    return (logits[:, 1] - logits[:, 0]).detach().cpu().numpy()


def starts(length):
    if length < 512 or length % 32:
        raise ValueError('chunked input must be at least 512 and a multiple of 32')
    last = length - 512
    result = list(range(0, last + 1, 384))
    if result[-1] != last:
        result.append(last)
    return result


def chunk_map(model, image, device):
    """512 px windows, aligned to 32 px; keep 64 px margins except at edges.

    Each cell is taken once, from the first qualifying window. Overlap beyond
    the centre is resolved by retaining that first window's value.
    """
    h, w = tuple((s // 32) * 32 for s in image.shape[:2])
    rows, cols = h // 32, w // 32
    canvas = np.full((rows, cols), np.nan, dtype=np.float32)
    ys, xs = starts(h), starts(w)
    with torch.inference_mode():
        for y in ys:
            for x in xs:
                patch = image[y:y + 512, x:x + 512]
                current = margin_map(model, [patch], device)[0]
                y0 = 0 if y == 0 else (y + 64) // 32
                y1 = rows if y == ys[-1] else (y + 448) // 32
                x0 = 0 if x == 0 else (x + 64) // 32
                x1 = cols if x == xs[-1] else (x + 448) // 32
                dest = canvas[y0:y1, x0:x1]
                src = current[y0 - y // 32:y1 - y // 32, x0 - x // 32:x1 - x // 32]
                empty = np.isnan(dest)
                dest[empty] = src[empty]
    if not np.isfinite(canvas).all():
        raise AssertionError(f'unfilled cells: {np.isnan(canvas).sum()}')
    return canvas


def whole_map(model, image, device):
    h, w = tuple((s // 32) * 32 for s in image.shape[:2])
    with torch.inference_mode():
        return margin_map(model, [image[:h, :w]], device)[0]


def cell_mask(changed, shape):
    h, w = shape
    return changed[:h * 32, :w * 32].reshape(h, 32, w, 32).any(axis=(1, 3))




def input_photos():
    def rows(source):
        with (ROOT / f'data/{source}/photo_manifest.csv').open(newline='') as stream:
            return [r for r in csv.DictReader(stream) if r['split'] == 'test']
    raise_rows = rows('raise')
    raise_test = [next(r for r in raise_rows if r['camera_model'] == camera)
                  for camera in ('NIKON D7000', 'NIKON D90')]
    div2k_test = rows('div2k')[:2]
    with (ROOT / 'data/alaska/selection_v1.csv').open(newline='') as stream:
        alaska = list(csv.DictReader(stream))
    alaska_test = [next(r for r in alaska if r['camera_model'] == camera)
                   for camera in ('Google_Pixel-XL', 'Canon_EOS700D')]
    return [('raise', r['photo_id'], r, r['camera_model']) for r in raise_test] + [
        ('div2k', r['photo_id'], r, 'not_applicable') for r in div2k_test] + [
        ('alaska', f"alaska_{r['alaska_id']}", r, r['camera_model']) for r in alaska_test]


def read_photo(source, row):
    if source != 'alaska':
        return load_photo(row, ROOT)
    path = ROOT / 'Datasets/originals/alaska/tiff_various_color' / (row['alaska_id'] + '.tif')
    with Image.open(path) as image:
        if image.format != 'TIFF' or image.mode != 'RGB' or min(image.size) < 512:
            raise ValueError(f'invalid ALASKA cover: {path}')
        return np.asarray(image, dtype=np.uint8).copy()


def prove_32(model, photos, device):
    """32 *test* tiles: original forward, mean of cell logits, and linear(mean)."""
    test_tiles = []
    for source, _, row, _ in photos:
        if source == 'alaska':
            continue  # no ALASKA test split has been assigned
        photo = read_photo(source, row)
        ys, xs = grid_axes(row, 'overlap_last', 512)
        used = 0
        for y in ys:
            for x in xs:
                test_tiles.append(photo[y:y + 512, x:x + 512].copy())
                used += 1
                if used == 8:
                    break
            if used == 8:
                break
        if used != 8:
            raise AssertionError(f'only {used} tiles in test photo {row["photo_id"]}')
    if len(test_tiles) != 32:
        raise AssertionError(f'only {len(test_tiles)} test tiles')
    reference_diff = pooled_diff = 0.0
    with torch.inference_mode():
        for i in range(0, 32, 4):
            batch = torch.cat([as_tensor(x, device) for x in test_tiles[i:i + 4]])
            forward = model(batch)
            features = model.features(batch)
            linear_mean = model.classifier(features.mean((-2, -1)))
            cell_mean = model.cell_logits(batch).mean((-2, -1))
            reference_diff = max(reference_diff, (forward - linear_mean).abs().max().item())
            pooled_diff = max(pooled_diff, (forward - cell_mean).abs().max().item())
    print(f'PROOF test_tiles=32 forward_vs_meanpooled_max_abs={reference_diff:.9g} '
          f'forward_vs_cellmean_max_abs={pooled_diff:.9g}', flush=True)
    if max(reference_diff, pooled_diff) > 5e-5:
        raise AssertionError('cell classifier fails equivalence to original forward')
    return reference_diff, pooled_diff


def verdicts(scores):
    flat = scores.ravel()
    k = max(1, math.ceil(flat.size * 0.01))
    return float(flat.mean()), float(np.partition(flat, -k)[-k:].mean()), float(flat.max()), k


def normalise(scores):
    median = np.median(scores)
    mad = np.median(np.abs(scores - median))
    return (scores - median) / max(float(mad), 1e-6), float(median), float(mad)




def render_v2(photo_id, group, cover, cases):
    small = Image.fromarray(cover).copy()
    small.thumbnail((320, 320))
    fig, axes = plt.subplots(len(cases), 4, figsize=(17, 2.35 * len(cases)), dpi=110)
    prob_scale = TwoSlopeNorm(vmin=0, vcenter=0.5, vmax=1)
    norm_scale = TwoSlopeNorm(vmin=-5, vcenter=0, vmax=5)
    prob_artist = norm_artist = None
    for row, case in enumerate(cases):
        score = case['map']
        normalized, _, _ = normalise(score)
        ax = axes[row]
        ax[0].imshow(small)
        ax[0].set_ylabel(case['label'], fontsize=8, rotation=0,
                         ha='right', va='center', labelpad=8)
        ax[1].imshow(case['changed_mask_thumb'], cmap='gray', vmin=0, vmax=1,
                     interpolation='nearest')
        prob_artist = ax[2].imshow(sigmoid(score), cmap='bwr', norm=prob_scale,
                                   interpolation='nearest')
        norm_artist = ax[3].imshow(np.clip(normalized, -5, 5), cmap='bwr',
                                   norm=norm_scale, interpolation='nearest')
        for axis in ax:
            axis.set_xticks([]); axis.set_yticks([])
    for axis, title in zip(axes[0], ('photo', 'changed-pixel mask\nwhite = pixels changed',
                                      'per-cell P(stego)', 'photo-normalised cell score')):
        axis.set_title(title, fontsize=10)
    fig.suptitle(f'{photo_id} | {group}: lowest-bit k={1 if group == "A" else 4}, match, random bytes',
                 y=0.995, fontsize=12)
    fig.text(0.53, 0.967,
             'Each square = 32x32 pixels. Red = model thinks a message is here.',
             ha='center', va='center', fontsize=10)
    fig.subplots_adjust(left=0.24, right=0.90, top=0.93, bottom=0.035,
                        wspace=0.24, hspace=0.28)
    # Two dedicated, full-height legends remain readable for tall 7-row figures.
    bar = fig.colorbar(prob_artist, cax=fig.add_axes([0.725, 0.30, 0.010, 0.42]),
                       ticks=[0, 0.25, 0.5, 0.75, 1])
    bar.set_label('P(stego) per 32x32 cell')
    bar = fig.colorbar(norm_artist, cax=fig.add_axes([0.902, 0.30, 0.010, 0.42]),
                       ticks=[-5, -2.5, 0, 2.5, 5])
    bar.set_label("cell score vs this photo's normal (MADs)")
    dest = OUTPUT / f'figure-{photo_id}-{group}-k{1 if group == "A" else 4}.png'
    fig.savefig(dest)
    plt.close(fig)
    print('FIGURE', dest.relative_to(ROOT), flush=True)


def render(photo_id, cover, cases):
    maps = np.concatenate([case['map'].ravel() for case in cases])
    lo, hi = np.percentile(maps, [2, 98])
    if hi <= lo:
        hi = lo + 1e-5
    small = Image.fromarray(cover).copy()
    small.thumbnail((320, 320))
    fig, axes = plt.subplots(len(cases), 4, figsize=(14, 2.15 * len(cases)), dpi=105)
    for row, case in enumerate(cases):
        scores = case['map']
        norm, _, _ = normalise(scores)
        axs = axes[row]
        axs[0].imshow(small); axs[0].set_ylabel(case['label'], fontsize=9)
        axs[1].imshow(case['changed_mask_thumb'], cmap='gray', vmin=0, vmax=1,
                      interpolation='nearest')
        axs[2].imshow(scores, cmap='coolwarm', vmin=lo, vmax=hi, interpolation='nearest')
        axs[3].imshow(np.clip(norm, -5, 5), cmap='coolwarm', vmin=-5, vmax=5,
                      interpolation='nearest')
        for ax in axs:
            ax.set_xticks([]); ax.set_yticks([])
    for ax, label in zip(axes[0], ('photo', 'changed-pixel mask (thumbnail)',
                                   'raw stego-clean logit', '(logit - median) / MAD')):
        ax.set_title(label, fontsize=10)
    fig.suptitle(photo_id + ' | each row uses full-photo embedding; scores are 32px cells', fontsize=11)
    fig.tight_layout()
    dest = OUTPUT / f'figure-{photo_id}.png'
    fig.savefig(dest)
    plt.close(fig)
    print('FIGURE', dest.relative_to(ROOT), flush=True)


def main():
    global OUTPUT, CHECKPOINT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fixed-settings', action='store_true',
                        help='compare classic lowest-bit k=1 and k=4 match; default preserves drawn settings')
    parser.add_argument('--checkpoint', type=Path, default=CHECKPOINT)
    parser.add_argument('--output', type=Path, default=None)
    args = parser.parse_args()
    CHECKPOINT = args.checkpoint.resolve()
    if args.fixed_settings:
        OUTPUT = ROOT / 'runs/map-demo-v2'
    if args.output is not None:
        OUTPUT = args.output.resolve()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    if not torch.cuda.is_available():
        raise RuntimeError('GPU required: run in aai3001 after checking live kernels')
    torch.set_num_threads(1)
    device = torch.device('cuda')
    ckpt = torch.load(CHECKPOINT, map_location='cpu', weights_only=False)
    hparams = ckpt.get('hyper_parameters', {})
    norm = hparams.get('norm', 'group')  # Old checkpoints have no recorded norm.
    model = model_from_hparams(hparams).to(device).eval()
    state = {key.removeprefix('net.'): value for key, value in ckpt['state_dict'].items()}
    if len(state) != len(ckpt['state_dict']):
        raise ValueError('unexpected checkpoint key collision')
    model.load_state_dict(state, strict=True)
    print('CHECKPOINT', CHECKPOINT.name, 'epoch', ckpt['epoch'], 'norm', norm, 'strict=True',
          'state_keys', len(state), 'GPU', torch.cuda.get_device_name(0), flush=True)
    photos = input_photos()
    proof = prove_32(model, photos, device)
    config = read_config(ROOT / 'configs/raise512-run1.yaml')
    assert config.size == {'bpp': [0.05, 0.5]}
    results = []
    photo_records = []
    for index, (source, photo_id, row, camera) in enumerate(photos):
        cover = read_photo(source, row)
        print('PHOTO', photo_id, source, camera, cover.shape, flush=True)
        h, w = tuple((s // 32) * 32 for s in cover.shape[:2])
        # The changed mask uses the untrimmed full cover; maps cover complete
        # 32px cells. Record any unscored right/bottom strip in the CSV.
        if args.fixed_settings:
            groups = [('A', dict(bit_mode='lowest', k=1, method='match', payload='random_bytes')),
                      ('B', dict(bit_mode='lowest', k=4, method='match', payload='random_bytes'))]
        else:
            settings_rng = np.random.default_rng(np.random.SeedSequence([0, index, 101]))
            settings, _ = draw_otf_settings(config, cover, settings_rng)
            groups = [(None, {key: settings[key] for key in
                              ('bit_mode', 'k', 'method', 'payload')})]
        for group, base in groups:
            cases = [('clean', None, None, cover, np.zeros(cover.shape[:2], dtype=bool), None)]
            for placement in PLACEMENTS:
                for rate in RATES:
                    rng = np.random.default_rng(np.random.SeedSequence([0, index,
                                                  0 if placement == 'contiguous' else 1,
                                                  int(rate * 1000), 102]))
                    stego, meta, changed_units = embed_fast(
                        cover, **base, placement=placement, start=None,
                        rng=rng, bpp=rate, return_changed_units=True)
                    changed = np.any(cover != stego, axis=2)
                    assert int(changed.sum()) == len(np.unique(changed_units // 3))
                    label = f'{placement} bpp={rate:g}'
                    cases.append((label, placement, rate, stego, changed, meta))
            figure_cases = []
            for label, placement, rate, image, changed, meta in cases:
                scores = chunk_map(model, image, device)
                positive = cell_mask(changed, scores.shape)
                mask_thumbnail = Image.fromarray(changed.astype(np.uint8))
                mask_thumbnail.thumbnail((320, 320), resample=Image.Resampling.NEAREST)
                avg, topk, highest, k = verdicts(scores)
                _, median, mad = normalise(scores)
                changed_scores = scores[positive]
                unchanged_scores = scores[~positive]
                whole = whole_map(model, image, device) if source != 'raise' else None
                if whole is not None:
                    assert whole.shape == scores.shape
                    delta = float(np.max(np.abs(scores - whole)))
                    correlation = float(np.corrcoef(scores.ravel(), whole.ravel())[0, 1])
                else:
                    delta = correlation = None
                result = dict(source=source, photo_id=photo_id, camera_model=camera,
                              placement=placement or 'clean', requested_bpp=rate,
                              effective_bpp=meta['bpp_actual'] if meta else None,
                              bit_mode=base['bit_mode'], k=base['k'], method=base['method'],
                              payload=base['payload'], changed_pixels=int(changed.sum()),
                              cells=int(scores.size), changed_cells=int(positive.sum()),
                              unchanged_cells=int((~positive).sum()), cell_auc=roc_auc(scores, positive),
                              changed_score_mean=float(changed_scores.mean()) if len(changed_scores) else None,
                              unchanged_score_mean=float(unchanged_scores.mean()) if len(unchanged_scores) else None,
                              mean_margin=avg, top1pct_margin=topk, max_margin=highest, top_k_cells=k,
                              mean_verdict='stego' if avg > 0 else 'clean',
                              top1pct_verdict='stego' if topk > 0 else 'clean',
                              max_verdict='stego' if highest > 0 else 'clean',
                              photo_median=median, photo_MAD=mad,
                              chunk_whole_max_abs=delta, chunk_whole_correlation=correlation,
                              cropped_right_pixels=cover.shape[1] - w,
                              cropped_bottom_pixels=cover.shape[0] - h)
                if group is not None:
                    result['group'] = group
                results.append(result)
                if args.fixed_settings:
                    auc = 'n/a' if result['cell_auc'] is None else f"{result['cell_auc']:.3f}"
                    photo_probability = float(sigmoid(avg))  # sigmoid(mean of cell logits)
                    row_label = (f'{label}\n{base["bit_mode"]} k={base["k"]} {base["method"]} '
                                 f'{base["payload"]}\nP(stego)={photo_probability:.4f}; cell AUC={auc}')
                else:
                    row_label = label
                figure_cases.append({'label': row_label, 'map': scores,
                                     'changed_mask_thumb': np.asarray(mask_thumbnail)})
                if placement:
                    prefix = f'{group}-' if group else ''
                    mask_path = OUTPUT / f'mask-{prefix}{photo_id}-{placement}-{str(rate).replace(".", "p")}.npz'
                    np.savez_compressed(mask_path, packed=np.packbits(changed.reshape(-1)),
                                        shape=np.asarray(changed.shape, dtype=np.int32))
                print('CASE', group, photo_id, label, 'mean', f'{avg:.5f}',
                      'Pmean', f'{float(sigmoid(avg)):.6f}', 'top1%', f'{topk:.5f}',
                      'max', f'{highest:.5f}', 'AUC', result['cell_auc'],
                      'chunk/whole', delta, correlation, flush=True)
                del image, changed
            if args.fixed_settings:
                render_v2(photo_id, group, cover, figure_cases)
            else:
                render(photo_id, cover, figure_cases)
            photo_records.append(dict(photo_id=photo_id, source=source, camera=camera,
                                      width=cover.shape[1], height=cover.shape[0],
                                      settings=base, group=group))
        del cover
        torch.cuda.empty_cache()
    csv_name = 'metrics-v2.csv' if args.fixed_settings else 'metrics.csv'
    with (OUTPUT / csv_name).open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(results[0]))
        writer.writeheader(); writer.writerows(results)
    json_name = 'run-v2.json' if args.fixed_settings else 'run.json'
    (OUTPUT / json_name).write_text(json.dumps(dict(checkpoint=CHECKPOINT.name,
        proof_max_abs_forward_meanpool=proof[0], proof_max_abs_forward_cellmean=proof[1],
        bpp=list(RATES), photo_records=photo_records, score='stego_minus_clean_logit',
        photo_probability='sigmoid(mean(cell_margin))', fixed_settings=args.fixed_settings,
        chunk_size=512, chunk_margin=64, cell_size=32, cases=len(results)), indent=2) + '\n')
    print('DONE cases', len(results), 'figures', len(photo_records), 'metrics', OUTPUT / csv_name, flush=True)


if __name__ == '__main__':
    main()
