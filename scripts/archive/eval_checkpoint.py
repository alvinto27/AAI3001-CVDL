#!/usr/bin/env python3
"""Score a fixed evaluation split without changing the standard notebook test.

Run in the project GPU environment, for example:
  python scripts/archive/eval_checkpoint.py --checkpoint runs/g4-prime-1/checkpoints/g4-prime-1/best-epoch08-val_pe0.0147-val_auc0.9973.ckpt --eval-yaml configs/test-bpp1.yaml --split test --output runs/eval-bpp1/v3c --workers 6
Bpp bins refer to REQUESTED bpp; stride capacity can reduce effective bpp.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from cvdl import evaluation
from cvdl.metrics import counts, metrics, summary_rows  # compatibility for analysis callers
from cvdl.settings import extra_panel_limit


def panel_dataset(dataset, *, quick=None):
    """Legacy env-aware boundary; the package itself never reads env values."""
    limit = extra_panel_limit(quick=quick)
    return evaluation.panel_dataset(dataset, quick=limit is not None)


def evaluate(checkpoint, eval_yaml=ROOT/'configs/raise512-run1.yaml', *, split='test',
             output, workers=6, source=None, manifest_root=None, quick=None):
    """Keep the standalone/notebook transition API; delegate all scoring."""
    torch.set_num_threads(1)
    limit = extra_panel_limit(quick=quick)
    return evaluation.evaluate(checkpoint, eval_yaml, root=ROOT,
        device='cuda' if torch.cuda.is_available() else 'cpu', split=split,
        output=output, workers=workers, source=source, manifest_root=manifest_root,
        quick=limit is not None)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--eval-yaml', type=Path, default=ROOT / 'configs/raise512-run1.yaml')
    parser.add_argument('--split', choices=('test','val','test_seen','test_unseen'), default='test')
    parser.add_argument('--source', choices=('alaska','lsdir'))
    parser.add_argument('--manifest-root', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=6)
    parser.add_argument('--quick', action=argparse.BooleanOptionalAction, default=None,
                        help='cap panel at 16 pairs; default follows explicit QUICK env (otherwise full)')
    args = parser.parse_args()
    evaluate(args.checkpoint, args.eval_yaml, split=args.split,
             output=args.output, workers=args.workers, source=args.source,
             manifest_root=args.manifest_root, quick=args.quick)


if __name__ == '__main__':
    main()
