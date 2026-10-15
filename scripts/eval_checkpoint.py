#!/usr/bin/env python3
"""Score a fixed evaluation split without changing the standard notebook test.

Run in the project GPU environment, for example:
  python scripts/eval_checkpoint.py --checkpoint runs-out/g4-prime-1/checkpoints/g4-prime-1/best-epoch08-val_pe0.0147-val_auc0.9973.ckpt --eval-yaml configs/test-bpp1.yaml --split test --output runs-out/eval-bpp1/v3c --workers 6
Bpp bins refer to REQUESTED bpp; stride capacity can reduce effective bpp.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'LSB-Dataset-Generator'))
from lsb_torch import Corpus, FixedPairs, OrderedPairLoader, RunSettings
from map_demo import model_from_hparams


def counts(labels, scores):
    labels = np.asarray(labels, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    order = np.argsort(-scores, kind='mergesort')
    ordered = scores[order]
    ends = np.r_[np.flatnonzero(ordered[1:] != ordered[:-1]) + 1, len(scores)]
    pos = np.cumsum(labels[order] == 1, dtype=np.int64)[ends - 1]
    neg = np.cumsum(labels[order] == 0, dtype=np.int64)[ends - 1]
    return pos, neg, int((labels == 1).sum()), int((labels == 0).sum())


def metrics(frame):
    if frame.empty:
        return dict(images=0, pairs=0, P_E=None, AUC=None, accuracy=None, false_alarm_rate=None, miss_rate=None)
    labels = frame['label'].to_numpy(dtype=np.int64)
    scores = frame['prob_stego'].to_numpy(dtype=np.float64)
    pos, neg, positives, negatives = counts(labels, scores)
    false_alarm = np.r_[0., neg / negatives, 1.] if negatives else np.zeros(len(pos) + 2)
    miss = np.r_[1., (positives - pos) / positives, 0.] if positives else np.zeros(len(pos) + 2)
    auc = float(np.trapezoid(np.r_[0., pos / positives, 1.], np.r_[0., neg / negatives, 1.])) if positives and negatives else 0.
    return dict(images=len(frame), pairs=positives, P_E=float(np.min((false_alarm + miss) / 2)), AUC=auc,
                accuracy=float(((scores >= .5) == labels).mean()),
                false_alarm_rate=float((scores[labels == 0] >= .5).mean()) if negatives else None,
                miss_rate=float((scores[labels == 1] < .5).mean()) if positives else None)


def summary_rows(frame):
    rows = []
    def add(category, value, subset):
        rows.append(dict(category=category, value=str(value), **metrics(subset)))
    add('overall', 'all', frame)
    for name in ('k', 'bit_mode/k', 'method', 'placement', 'source', 'bpp_bin'):
        if name == 'bit_mode/k':
            group = frame['bit_mode'] + '/' + frame['k'].astype(str)
        else:
            group = frame[name]
        for value in sorted(group.dropna().unique(), key=str):
            add(name, value, frame.loc[group == value])
    one = frame.loc[frame['k'] == 1]
    for (placement, bpp_bin), subset in one.groupby(['placement', 'bpp_bin'], observed=True, sort=True):
        add('k1/placement/bpp_bin', f'{placement}/{bpp_bin}', subset)
    return rows


def evaluate(checkpoint, eval_yaml=ROOT / 'configs/raise512-run1.yaml', *, split='test', output,
             workers=6, source=None, manifest_root=None):
    """Score one frozen panel and return the summary; default is the standard test."""
    if (split not in ('test','val','test_seen','test_unseen') or workers < 0
            or source not in (None,'alaska','lsdir')
            or (source is None and split not in ('test','val'))
            or (source=='lsdir' and split not in ('test','val'))
            or (source=='alaska' and split=='test')):
        raise ValueError('unsupported source/split or worker count')
    start = time.perf_counter()
    torch.set_num_threads(1)
    checkpoint = Path(checkpoint).resolve()
    evaluation_yaml = Path(eval_yaml).resolve()
    output = Path(output).resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError(f'output must be empty: {output}')
    output.mkdir(parents=True, exist_ok=True)
    ckpt = torch.load(checkpoint, map_location='cpu', weights_only=False)
    hparams = ckpt.get('hyper_parameters', {})
    norm = hparams.get('norm', 'group')
    model = model_from_hparams(hparams).eval()
    state = {key.removeprefix('net.'): value for key, value in ckpt['state_dict'].items()}
    if len(state) != len(ckpt['state_dict']):
        raise ValueError('duplicate checkpoint keys')
    model.load_state_dict(state, strict=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    settings = RunSettings(evaluation_yaml=str(evaluation_yaml),
                           sources=(source,) if source else ('raise','div2k'),
                           manifest_root=str(manifest_root) if manifest_root else None)
    dataset = FixedPairs(Corpus(ROOT, settings), split, sources=(source,) if source else ('raise','div2k'))
    if not len(dataset):
        raise ValueError('selected evaluation panel is empty')
    loader = OrderedPairLoader(dataset, batch_size=16, num_workers=workers,
                               pin_memory=device.type == 'cuda', prefetch_images=128)
    pairs = dataset.frame
    setting_fields = ['cover_id','k','bit_mode','method','placement','payload_kind','stride',
                      'requested_bpp','effective_bpp','clipped']
    setting_digest = hashlib.sha256(pairs[setting_fields].to_csv(index=False).encode()).hexdigest()
    rows = []
    with torch.inference_mode():
        for clean, stego, indices in loader:
            images = torch.cat([clean, stego], dim=0).to(device=device, dtype=torch.float32, non_blocking=True)
            probabilities = model(images).softmax(dim=1)[:, 1].cpu().numpy()
            n = len(clean)
            for j, index in enumerate(indices.tolist()):
                row = pairs.iloc[int(index)]
                common = {**row.to_dict(), 'data_mode': 'otf'}
                rows.append({**common, 'file': str(row['cover_id']) + ':clean', 'label': 0,
                             'prob_stego': float(probabilities[j])})
                rows.append({**common, 'file': str(row['cover_id']) + ':stego', 'label': 1,
                             'prob_stego': float(probabilities[n + j])})
    predictions = pd.DataFrame(rows)
    # Keep the same first fields as notebook 04; no request/embedding fields are lost.
    front = ['cover_id','source_photo','camera_model','k','bit_mode','method','placement',
             'payload_kind','bpp_actual']
    predictions = predictions[front + [x for x in predictions if x not in front]]
    predictions.to_csv(output / 'test_predictions.csv', index=False)
    bins = [0., .05, .15, .3, .5, .75, 1.]
    labels = ['<=0.05','0.05-0.15','0.15-0.30','0.30-0.50','0.50-0.75','0.75-1.0']
    predictions['bpp_bin'] = pd.cut(predictions['requested_bpp'], bins=bins,
                                     labels=labels, include_lowest=True, right=True)
    if predictions['bpp_bin'].isna().any():
        raise ValueError('requested bpp outside 0..1')
    summary = summary_rows(predictions)
    pd.DataFrame(summary).to_csv(output / 'summary.csv', index=False)
    clipped = (pairs.groupby('placement', sort=True)['clipped'].agg(['sum','count'])
               .reset_index().rename(columns={'sum':'clipped_pairs','count':'pairs'}))
    clipped.to_csv(output / 'clipped_by_placement.csv', index=False)
    info = dict(checkpoint=str(checkpoint), checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                norm=norm, clip_t=hparams.get('clip_t', 3.0), evaluation_yaml=str(evaluation_yaml),
                split=split, workers=workers, settings_sha256=setting_digest,
                source=source or 'standard', manifest_root=str(manifest_root) if manifest_root else None,
                pairs=len(dataset), elapsed_seconds=time.perf_counter() - start, overall=summary[0])
    (output / 'run.json').write_text(json.dumps(info, indent=2) + '\n')
    print(json.dumps(info, indent=2), flush=True)
    return info, pd.DataFrame(summary)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--eval-yaml', type=Path, default=ROOT / 'configs/raise512-run1.yaml')
    parser.add_argument('--split', choices=('test','val','test_seen','test_unseen'), default='test')
    parser.add_argument('--source', choices=('alaska','lsdir'))
    parser.add_argument('--manifest-root', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=6)
    args = parser.parse_args()
    evaluate(args.checkpoint, args.eval_yaml, split=args.split,
             output=args.output, workers=args.workers, source=args.source,
             manifest_root=args.manifest_root)


if __name__ == '__main__':
    main()
