"""Build the README report figures from tracked results and the shipped bundle.

    podman exec -w /work aai3001 python -B scripts/build_report_figures.py

Writes docs/figures/report/model-history.png (from
docs/results/model-history-2026-11-25.json) and
docs/figures/report/heatmap-example.png (DIV2K test photo, the shipped run 22
bundle through the ONNX engine, one generated message with a known mask).
Needs matplotlib, onnxruntime and the DIV2K originals.
"""
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'LSB-Dataset-Generator')]
OUT = ROOT / 'docs/figures/report'
HISTORY = ROOT / 'docs/results/model-history-2026-11-25.json'
PHOTO = ROOT / 'Datasets/originals/div2k/DIV2K_valid_HR/0829.png'  # DIV2K 0801-0900 = test split
SEED = 20261125


def model_history():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    data = json.loads(HISTORY.read_text())
    models = [m['label'] for m in data['models']]
    hard = {}
    for step in data['steps']:
        h = step['hard_k1_low_auc']['models']
        hard[step['frm']] = h['a']; hard[step['to']] = h['b']
    pe = [100 * m['standard_pe'] for m in data['models']]
    auc = [hard[m]['point'] for m in models]
    auc_err = np.array([[hard[m]['point'] - hard[m]['low'], hard[m]['high'] - hard[m]['point']] for m in models]).T
    pe_err = []
    for i, m in enumerate(models):
        step = next((s for s in data['steps'] if s['to'] == m), None) or next(s for s in data['steps'] if s['frm'] == m)
        side = 'b' if step['to'] == m else 'a'
        v = step['standard_pe']['models'][side]
        pe_err.append([100 * (v['point'] - v['low']), 100 * (v['high'] - v['point'])])
    pe_err = np.array(pe_err).T
    x = np.arange(len(models))
    fig, (a, b) = plt.subplots(2, 1, figsize=(10, 6.2), sharex=True)
    colours = ['#9e9e9e'] + ['#4c72b0'] * (len(models) - 2) + ['#c44e52']
    a.bar(x, pe, yerr=pe_err, color=colours, capsize=3)
    a.set_ylabel('Standard test P_E (%)\nlower is better')
    for i, v in enumerate(pe):
        a.text(i, v + pe_err[1][i] + .08, f'{v:.2f}', ha='center', fontsize=8)
    a.set_ylim(0, 7.0)
    b.errorbar(x, auc, yerr=auc_err, fmt='o-', color='#333333', ecolor='#777777', capsize=3)
    b.scatter([x[0], x[-1]], [auc[0], auc[-1]], color=[colours[0], colours[-1]], zorder=3, s=50)
    b.set_ylabel('Hard test AUC\n(k=1, low payload)\nhigher is better')
    for i, v in enumerate(auc):
        b.text(i, v + .012, f'{v:.3f}', ha='center', fontsize=8)
    b.set_ylim(.58, .84)
    b.set_xticks(x, models, rotation=25, ha='right')
    for ax in (a, b):
        ax.grid(axis='y', alpha=.3)
    fig.suptitle('Model history: each model scored on the same frozen tests (95% photo-bootstrap intervals)', fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT / 'model-history.png', dpi=130)
    plt.close(fig)


def heatmap_example():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from PIL import Image
    from lsb_engine import embed
    from cvdl.pipeline import load_pipeline
    clean = np.asarray(Image.open(PHOTO).convert('RGB')).copy()
    h, w = clean.shape[:2]
    units = h * w * 3
    rng = np.random.default_rng(SEED)
    # k=1 LSB replacement, contiguous: a band of rows starting 45% down the photo.
    stego, mask, result = embed(clean, 'lowest', 1, 'replace', 'contiguous', rng,
                                start=int(.45 * units), bpp=.25)
    pipeline = load_pipeline(engine='onnx')
    runs = {name: pipeline.analyze(image) for name, image in (('clean', clean), ('stego', stego))}
    fig, axes = plt.subplots(2, 3, figsize=(13, 6.4))
    for row, name in enumerate(('clean', 'stego')):
        r = runs[name]
        grid = np.array(r['heatmap']['values']); cut = r['heatmap']['cutoff']
        display = np.repeat(np.repeat(grid - cut, 32, 0), 32, 1)[:h, :w]
        image = clean if name == 'clean' else stego
        truth = np.zeros((h, w), bool) if name == 'clean' else mask
        axes[row, 0].imshow(image); axes[row, 0].set_title(f'{name.title()} photo (DIV2K 0829, {w}x{h})', fontsize=9)
        axes[row, 1].imshow(truth, cmap='gray', vmin=0, vmax=1)
        axes[row, 1].set_title('Ground truth: pixels that carry the message' if name == 'stego' else 'Ground truth: no message', fontsize=9)
        axes[row, 2].imshow(image)
        axes[row, 2].imshow(np.ma.masked_where(display < 0, display), cmap='autumn_r', alpha=.55, vmin=0, vmax=max(1., float(display.max())))
        verdict = 'message detected' if r['photo_verdict'] == 'message_detected' else 'not detected'
        flagged = int(np.array(r['heatmap']['flagged']).sum())
        axes[row, 2].set_title(f'Model: {verdict} (score {r["photo_rule"]["probability"]:.2f}); {flagged} cells flagged', fontsize=9)
        for ax in axes[row]:
            ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle(f'Run 22 (ONNX engine) on a test photo. Message: k=1 LSB replacement, contiguous, '
                 f'{result.payload_bytes:,} bytes ({result.bpp_actual:.2f} bpp). Orange = 32 px cells above the cell cut-off.', fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT / 'heatmap-example.png', dpi=110)
    plt.close(fig)
    return {k: (v['photo_verdict'], round(v['photo_rule']['probability'], 3)) for k, v in runs.items()}


if __name__ == '__main__':
    OUT.mkdir(parents=True, exist_ok=True)
    model_history()
    print(heatmap_example())
