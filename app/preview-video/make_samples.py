"""Build the clean and stego sample photos for the landing preview videos.

The stego photo is the clean photo with a random LSB payload embedded by the
repository's own engine into one rectangle, so the model has a localised
modification to find. Run from the repository root:

    .venv/Scripts/python app/preview-video/make_samples.py SOURCE.png
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'LSB-Dataset-Generator'))
from lsb_engine import embed_fast  # noqa: E402

OUT = Path(__file__).resolve().parent / 'samples'


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('source', type=Path)
    ap.add_argument('--box', type=int, nargs=4, metavar=('X0', 'Y0', 'X1', 'Y1'), default=(416, 96, 640, 320))
    ap.add_argument('--k', type=int, default=3)
    ap.add_argument('--bpp', type=float, default=3.0)
    ap.add_argument('--seed', type=int, default=7)
    args = ap.parse_args()

    clean = np.asarray(Image.open(args.source).convert('RGB'))
    x0, y0, x1, y1 = args.box
    region, meta, _ = embed_fast(
        np.ascontiguousarray(clean[y0:y1, x0:x1]), 'lowest', args.k, 'replace', 'random',
        np.random.default_rng(args.seed), bpp=args.bpp,
    )
    stego = clean.copy()
    stego[y0:y1, x0:x1] = region

    OUT.mkdir(exist_ok=True)
    stem = args.source.stem
    Image.fromarray(clean).save(OUT / f'{stem}-clean.png')
    Image.fromarray(stego).save(OUT / f'{stem}-stego.png')
    record = {
        'source': args.source.name,
        'box_xyxy': [x0, y0, x1, y1],
        'engine': 'LSB-Dataset-Generator/lsb_engine.embed_fast',
        'bit_mode': 'lowest', 'k': args.k, 'method': 'replace', 'placement': 'random',
        'bpp': args.bpp, 'seed': args.seed,
        'units_changed': int(meta['units_changed']),
    }
    (OUT / f'{stem}-embed.json').write_text(json.dumps(record, indent=2) + '\n')
    print(json.dumps(record))


if __name__ == '__main__':
    main()
