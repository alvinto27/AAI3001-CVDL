"""Build the shared report illustration from pinned frozen test pairs, on CPU only."""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
IMAGE = Path('docs/figures/report/data-sample.png')
# These are frozen choices, not selection rules. Never redraw embedding settings.
CHOICES = (
    ('RAISE', 'standard', '5045c1076ab3f540a61c/standard/raise/00429/k1'),
    ('DIV2K', 'standard', '5045c1076ab3f540a61c/standard/div2k/00123/k1'),
    ('ALASKA', 'alaska-seen', '5045c1076ab3f540a61c/alaska-seen/alaska/00139/k1'),
    ('LSDIR', 'lsdir', '5045c1076ab3f540a61c/lsdir/lsdir/00125/k1'),
    ('Random', 'standard', '5045c1076ab3f540a61c/standard/raise/00845/k1'),
    ('Stride', 'standard', '5045c1076ab3f540a61c/standard/raise/00299/k1'),
    ('Noisy ALASKA', 'hard', 'd6d91a5a98dae856d1ae/alaska-seen/k1/low_noisy/0002'),
)


def samples(root):
    """Replay saved draws and prove equality with the evaluator's dataset output."""
    import numpy as np
    import pandas as pd
    from cvdl.standardtest import StandardTestPairs
    from lsb_torch import Corpus, RunSettings
    from lsb_train_grid import crop
    from lsb_engine import embed_fast

    root = Path(root)
    rows = []
    for label, panel, identity in CHOICES:
        folder = root/('data/hard-test/256-v1' if panel == 'hard' else 'data/test/standard-256-v1')
        name = f'pairs-{"alaska-seen" if panel == "hard" else panel}.csv'
        raw = (folder/name).read_bytes()
        summary = json.loads((folder/'summary.json').read_text())
        if hashlib.sha256(raw).hexdigest() != summary['files_sha256'][name]:
            raise ValueError('frozen pair list hash mismatch')
        frame = pd.read_csv(io.BytesIO(raw), float_precision='round_trip')
        chosen = frame[frame.pair_id == identity]
        if len(chosen) != 1:
            raise ValueError('pinned pair is missing or ambiguous')
        rows.append((label, chosen.iloc[0]))
    records = []
    for source in ('raise', 'div2k', 'alaska', 'lsdir'):
        with (root/f'data/{source}/photo_manifest.csv').open() as stream:
            records.extend(r for r in csv.DictReader(stream) if r['photo_id'] in {row.source_photo for _, row in rows})
    corpus = Corpus(root, RunSettings(source='decode', sources=('raise', 'div2k', 'alaska', 'lsdir')), records=records)
    result = []
    for label, row in rows:
        record = corpus.by_id[row.source_photo]
        photo = corpus.photo(record)
        clean, (y, x), pad = crop(photo, record, 256, int(row.tile_row), int(row.tile_col), 'overlap_last')
        if any(pad) or (y, x) != (int(row.y), int(row.x)):
            raise ValueError('recorded tile geometry mismatch')
        clean = clean.copy()
        kwargs = json.loads(row.draw_kwargs)
        kwargs['rng'] = np.random.default_rng(np.random.SeedSequence(json.loads(row.embedding_seed_key)))
        stego, _, _ = embed_fast(clean.copy(), **kwargs)
        expected_clean, expected_stego, _ = StandardTestPairs(corpus, pd.DataFrame([row]))[0]
        for actual, expected in ((clean, expected_clean), (stego, expected_stego)):
            np.testing.assert_array_equal(actual, expected.permute(1, 2, 0).numpy())
        if int(row.k) != 1 or float(row.requested_bpp) > .15:
            raise ValueError('illustration needs low-bpp, k=1 pairs')
        result.append(dict(label=label, row=row, record=record, photo=photo, clean=clean, stego=stego))
    if result[-1]['row'].condition != 'low_noisy':
        raise ValueError('noisy sample must belong to the frozen low/noisy cell')
    return result


def render(items):
    """Deterministic PNG bytes, with no artifact identities in the figure."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    import numpy as np
    from PIL import Image
    from lsb_train_grid import axes

    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11, 'axes.titlesize': 12})
    fig = plt.figure(figsize=(13, 11), dpi=110, layout='constrained')
    grid = fig.add_gridspec(3, 1, height_ratios=[1.2, 1, 1])
    fig.suptitle('Real test photos and hidden-message changes', fontsize=17)
    top = grid[0].subgridspec(2, 4, height_ratios=[1, .12])
    for i, item in enumerate(items[:4]):
        ax = fig.add_subplot(top[0, i]); photo = item['photo']; h, w = photo.shape[:2]
        thumb = Image.fromarray(photo); thumb.thumbnail((330, 280), Image.Resampling.LANCZOS)
        ax.imshow(thumb, extent=(0, w, h, 0))
        ys, xs = axes(item['record'], 256, 'overlap_last')
        for y in sorted({v for p in ys for v in (p, p+256)}): ax.axhline(y, color='white', lw=.35, alpha=.65)
        for x in sorted({v for p in xs for v in (p, p+256)}): ax.axvline(x, color='white', lw=.35, alpha=.65)
        r = item['row']; ax.add_patch(Rectangle((r.x, r.y), 256, 256, fill=False, edgecolor='#ffcf00', lw=2))
        ax.set_title(item['label']); ax.set_xticks([]); ax.set_yticks([])
        ax.set_xlabel('256-pixel grid; yellow = chosen tile', fontsize=9)
    photo_note = fig.add_subplot(top[1, :]); photo_note.axis('off')
    photo_note.text(.5, .5, 'Last row and column overlap so every tile is full size.', ha='center', va='center', fontsize=10)
    middle = grid[1].subgridspec(2, 5, height_ratios=[1, .17], width_ratios=[1]*5)
    def tile(ax, pixels, title):
        ax.imshow(pixels, vmin=0, vmax=255, cmap='gray', interpolation='nearest'); ax.set_title(title)
        ax.set_xticks([]); ax.set_yticks([])
    first = items[0]
    tile(fig.add_subplot(middle[0, 0]), first['clean'], 'Clean tile\nRAISE tile 1')
    tile(fig.add_subplot(middle[0, 1]), first['stego'], 'With hidden message\nRAISE tile 1')
    ax = fig.axes[-1]; ax.set_xlabel('The two tiles look the same.', fontsize=10)
    maps = [first, items[4], items[5]]
    for i, (name, item) in enumerate(zip(('Contiguous', 'Random', 'Stride'), maps), 2):
        changed = np.any(item['clean'] != item['stego'], axis=2)
        ax = fig.add_subplot(middle[0, i]); tile(ax, changed.astype(np.uint8)*255, name+f'\nRAISE tile {i-1}')
        ax.set_xlabel(f'k = 1; {item["row"].requested_bpp:.3f} bits per pixel', fontsize=9)
    # A full-width caption slot avoids stretching one panel's layout bounds.
    map_note = fig.add_subplot(middle[1, :]); map_note.axis('off')
    map_note.text(.5, .5, 'Change maps: white = a pixel the hidden message changed. The real change is too small to see.',
        ha='center', va='center', fontsize=10)
    bottom = grid[2].subgridspec(1, 4)
    noisy = items[-1]
    tile(fig.add_subplot(bottom[0]), noisy['clean'], 'Noisy ALASKA tile')
    tile(fig.add_subplot(bottom[1]), noisy['stego'], 'With hidden message')
    tile(fig.add_subplot(bottom[2]), np.any(noisy['clean'] != noisy['stego'], axis=2).astype(np.uint8)*255, 'Amplified change map')
    note = fig.add_subplot(bottom[3]); note.axis('off')
    note.text(0, .7, 'Noise hides the changes.\n\nWhite marks changed pixels;\nthe changes themselves\nare invisible here.\n\nk = 1; contiguous message\n'+f'{noisy["row"].requested_bpp:.3f} bits per pixel', va='top', fontsize=11)
    buffer = io.BytesIO(); fig.savefig(buffer, format='png', dpi=110, bbox_inches='tight', metadata={'Software': 'Frozen test data illustration'})
    plt.close(fig)
    # One shared palette keeps the photographic PNG small without changing the
    # replay/equality checks above. Map contrast and labels stay sharp.
    image = Image.open(io.BytesIO(buffer.getvalue())).convert('RGB')
    image = image.quantize(colors=256, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE)
    compact = io.BytesIO(); image.save(compact, format='PNG', optimize=True)
    return compact.getvalue()


def build(root=ROOT):
    return render(samples(root))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true', help='rebuild and compare PNG bytes without writing')
    args = parser.parse_args()
    os.environ.update(CUDA_VISIBLE_DEVICES='', HIP_VISIBLE_DEVICES='', MPLBACKEND='Agg')
    sys.path[:0] = [str(ROOT), str(ROOT/'LSB-Dataset-Generator')]
    data = build(ROOT); target = ROOT/IMAGE
    if args.check:
        if not target.is_file() or target.read_bytes() != data:
            raise SystemExit('Data figure differs; rebuild with the recorded inputs and software.')
        print('Data figure is byte-identical; all seven pairs equal evaluator pixels.')
    else:
        target.parent.mkdir(parents=True, exist_ok=True); target.write_bytes(data)
        print(f'Wrote {IMAGE} ({len(data)} bytes); all seven pairs equal evaluator pixels.')


if __name__ == '__main__':
    main()
