#!/usr/bin/env python3
"""Measure clean ALASKA tile texture and draw six clean-score comparisons.

Noise is the population std of the three-channel KV residual after the
checkpoint's input clip (what the model sees), excluding the two-pixel border. 'Flat' means a 16x16 luminance block
has std < 5 on the raw 0..255 scale. All bin edges come from validation clean
covers, never test scores. Four photo readers and one GPU feature pass.
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import cvdl  # exposes the flat LSB library
from lsb_source import load_photo
from cvdl.models import model_from_hparams
from cvdl.data import kv_residual
from cvdl.metrics import sigmoid


class Covers(Dataset):
    def __init__(self, frames: dict[str, pd.DataFrame], manifest: dict[str, dict]):
        self.manifest = manifest
        self.rows = [(split, row.cover_id, row.source_photo, int(row.x), int(row.y))
                     for split, frame in frames.items() for row in frame.itertuples(index=False)]
        if len({r[1] for r in self.rows}) != len(self.rows):
            raise ValueError('cover IDs must be unique across ALASKA splits')

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        split, cover_id, photo_id, x, y = self.rows[i]
        raw = load_photo(self.manifest[photo_id], ROOT)
        tile = raw[y:y + 512, x:x + 512]
        if tile.shape != (512, 512, 3):
            raise ValueError(f'invalid crop: {cover_id}')
        return torch.from_numpy(np.ascontiguousarray(tile)), cover_id


def draw_covers(clean: pd.DataFrame, manifest: dict[str, dict], model, dest: Path, device):
    """Compare high/low clean scores within the same seen/unseen camera."""
    cameras = [('test_seen', 'Sigma_SD1Merrill'), ('test_unseen', 'Sony_A7R'),
               ('test_unseen', 'Panasonic_DMC-TZ60')]
    selected = []
    for split, camera in cameras:
        f = clean[(clean.split == split) & (clean.camera_model == camera)]
        if len(f) < 2:
            raise ValueError(f'need high and low covers for {split}/{camera}')
        selected.extend([f.nlargest(1, 'prob_stego').iloc[0],
                         f.nsmallest(1, 'prob_stego').iloc[0]])
    fig, axes = plt.subplots(6, 3, figsize=(10.3, 18), dpi=120)
    for i, row in enumerate(selected):
        photo = load_photo(manifest[row.source_photo], ROOT)
        tile = np.ascontiguousarray(photo[int(row.y):int(row.y) + 512,
                                          int(row.x):int(row.x) + 512])
        x = torch.from_numpy(tile).permute(2, 0, 1).float()[None].to(device)
        with torch.inference_mode():
            residual = model.kv(x)[0].abs().mean(dim=0).cpu().numpy()
            cell = model.cell_logits(x)[0].cpu().numpy()
        prob_cells = sigmoid(cell[1] - cell[0])
        axes[i, 0].imshow(tile)
        axes[i, 1].imshow(np.clip(residual, 0, 10), cmap='magma', vmin=0, vmax=10)
        axes[i, 2].imshow(prob_cells, cmap='bwr', vmin=0, vmax=1, interpolation='nearest')
        for a in axes[i]:
            a.set_xticks([]); a.set_yticks([])
        axes[i, 0].set_ylabel(f'{row.camera_model}\n{row.split} | clean P={row.prob_stego:.4g}\n'
                              f'KV input std={row.kv_std:.2f}, flat={row.flat_share:.1%}', fontsize=8,
                              rotation=0, ha='right', va='center', labelpad=7)
    for a, title in zip(axes[0], ('clean tile', 'raw absolute KV residual (0..10)',
                                   '32px cell P(stego), clean tile')):
        a.set_title(title, fontsize=9)
    fig.suptitle('Matched-camera clean-score contrasts: highest then lowest clean tile per camera', fontsize=11)
    fig.subplots_adjust(left=.32, right=.98, top=.965, bottom=.01, wspace=.12, hspace=.18)
    dest.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(dest, bbox_inches='tight')
    plt.close(fig)
    print('six-cover figure:', dest)
    return selected


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--val', type=Path, required=True)
    ap.add_argument('--seen', type=Path, required=True)
    ap.add_argument('--unseen', type=Path, required=True)
    ap.add_argument('--checkpoint', type=Path, required=True)
    ap.add_argument('--thresholds', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--figure', type=Path, required=True)
    args = ap.parse_args()
    torch.set_num_threads(1)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ckpt = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    clip_t = ckpt['hyper_parameters'].get('clip_t', 3.0)
    if clip_t is None or clip_t <= 0:
        raise ValueError('this diagnostic requires a positive checkpoint KV input clip')
    threshold_rows = pd.read_csv(args.thresholds)
    cal = threshold_rows[(threshold_rows.source == 'alaska') &
                         (threshold_rows.calibration == 'source') &
                         (threshold_rows.rule == 'fa5')]
    if len(cal) != 2 or cal.threshold.nunique() != 1:
        raise ValueError('need seen/unseen validation-derived ALASKA threshold')
    cutoff = float(cal.threshold.iloc[0])
    with (ROOT / 'data/alaska/photo_manifest.csv').open(newline='') as stream:
        manifest = {r['photo_id']: r for r in csv.DictReader(stream)}
    provenance = pd.read_csv(ROOT / 'data/alaska/provenance.csv')
    frames = {}
    all_preds = []
    for split, path in (('val', args.val), ('test_seen', args.seen), ('test_unseen', args.unseen)):
        p = pd.read_csv(path)
        if set(p.label.unique()) != {0, 1} or set(p.source.unique()) != {'alaska'}:
            raise ValueError(f'not paired ALASKA: {path}')
        f = p[p.label == 0].copy()
        if f.cover_id.duplicated().any() or len(f) * 2 != len(p):
            raise ValueError(f'clean identities must be unique: {path}')
        frames[split] = f
        p['split'] = split
        all_preds.append(p)
    features = []
    loader = DataLoader(Covers(frames, manifest), batch_size=16, num_workers=4,
                        pin_memory=(device.type == 'cuda'), prefetch_factor=2)
    for i, (image, cover_ids) in enumerate(loader):
        x = image.permute(0, 3, 1, 2).to(device, dtype=torch.float32, non_blocking=True)
        with torch.inference_mode():
            kv = kv_residual(x)
            raw_std = kv[:, :, 2:-2, 2:-2].flatten(1).std(dim=1, unbiased=False)
            kv_std = kv[:, :, 2:-2, 2:-2].clamp(-clip_t, clip_t).flatten(1).std(dim=1, unbiased=False)
            clipped = ((x == 0) | (x == 255)).float().mean(dim=(1, 2, 3))
            gray = x[:, 0] * .299 + x[:, 1] * .587 + x[:, 2] * .114
            brightness = gray.mean(dim=(1, 2))
            mu = F.avg_pool2d(gray[:, None], 16, 16)
            mu2 = F.avg_pool2d(gray[:, None].square(), 16, 16)
            flat = ((mu2 - mu.square()).clamp_min(0).sqrt() < 5).float().mean(dim=(1, 2, 3))
        for cid, noise, raw, clip, mean, flat_share in zip(cover_ids, kv_std.tolist(), raw_std.tolist(),
                                                           clipped.tolist(), brightness.tolist(), flat.tolist()):
            features.append(dict(cover_id=cid, kv_std=noise, kv_raw_std=raw, clipped_share=clip,
                                 brightness=mean, flat_share=flat_share))
        if (i + 1) % 100 == 0:
            print('feature batches:', i + 1, '/', len(loader), flush=True)
    feat = pd.DataFrame(features)
    pred = pd.concat(all_preds, ignore_index=True)
    clean = pred[pred.label == 0].merge(feat, on='cover_id', validate='one_to_one').merge(
        provenance[['photo_id','in_v1','block','width','height','tiles512','bytes']],
        left_on='source_photo', right_on='photo_id', validate='many_to_one')
    if len(clean) != len(feat):
        raise AssertionError('failed to join feature rows')
    q = clean.loc[clean.split == 'val', 'kv_std'].quantile([.25, .5, .75]).to_numpy()
    if len(np.unique(q)) != 3:
        raise ValueError('validation noise quartile edges must be unique')
    labels = ['Q1 low', 'Q2', 'Q3', 'Q4 high']
    clean['noise_bin'] = pd.cut(clean.kv_std, [-np.inf, *q, np.inf], labels=labels).astype(str)
    # Same *clean* tile measurements for the matched k1 stego image.
    stego = pred[(pred.label == 1) & (pred.k == 1)].merge(
        clean[['cover_id','noise_bin','camera_model','split','in_v1','flat_share','clipped_share',
               'brightness','kv_std']], on='cover_id', suffixes=('','_clean'), validate='one_to_one')
    rows = []
    for measure in ('noise_bin','camera_model','in_v1'):
        for split in ('val', 'test_seen', 'test_unseen'):
            c = clean[clean.split == split]
            s = stego[stego.split == split]
            for category, g in c.groupby(measure, dropna=False, observed=True):
                k1 = s[s[measure] == category]
                rows.append(dict(split=split, measure=measure, value=category,
                                 clean_n=len(g), k1_n=len(k1), kv_std_median=g.kv_std.median(),
                                 flat_median=g.flat_share.median(), clipped_median=g.clipped_share.median(),
                                 brightness_median=g.brightness.median(),
                                 fa_0p5=(g.prob_stego >= .5).mean(),
                                 fa_val5=(g.prob_stego >= cutoff).mean(),
                                 k1_miss_0p5=(k1.prob_stego < .5).mean() if len(k1) else np.nan,
                                 k1_miss_val5=(k1.prob_stego < cutoff).mean() if len(k1) else np.nan))
    args.output.mkdir(parents=True, exist_ok=True)
    clean.to_csv(args.output / 'clean_features.csv', index=False)
    pd.DataFrame(rows).to_csv(args.output / 'bins_and_cameras.csv', index=False)
    (args.output / 'noise_bins.txt').write_text('validation clean KV std quartile edges: ' +
                                                  ', '.join(map(str, q)) + '\n')
    model = model_from_hparams(ckpt['hyper_parameters']).eval()
    model.load_state_dict({k.removeprefix('net.'): v for k, v in ckpt['state_dict'].items()})
    model.to(device)
    selected = draw_covers(clean, manifest, model, args.figure, device)
    pd.DataFrame(selected).to_csv(args.output / 'figure_tiles.csv', index=False)
    print('features:', len(clean), 'k1 stego:', len(stego), 'val5 cutoff:', cutoff,
          'val-only noise quartiles:', q, flush=True)


if __name__ == '__main__':
    main()
