"""Opt-in CPU CHECK 1. Positive control gates all main-cell work; no model inference.

Run --phase positive first. Frozen pixels come from StandardTestPairs, which is
also used by the hard256 evaluator. Input SHA-256 manifests bracket each phase.
Six photo-owned workers default; one decoded photo at a time in each worker.
"""
from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor
import csv
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT/'runs/srm-e50-20261017T111310Z-0214'
EVAL = RUN/'evaluation/standard-256-v1/35afd1c7938bae78968c'
OUT = ROOT/'runs/verification/diagnostics-2026-10-19/check1'
SEED = 20261019


def snapshot():
    files = set(RUN.rglob('*'))
    for name in ('data/test/standard-256-v1', 'data/hard-test/256-v1', 'data/val/60da4391043e8ce48ff2'):
        files.update((ROOT/name).rglob('*'))
    for source in ('raise', 'div2k', 'alaska', 'lsdir'):
        files.add(ROOT/f'data/{source}/photo_manifest.csv')
    def digest(path):
        h = hashlib.sha256()
        with path.open('rb') as stream:
            for block in iter(lambda: stream.read(1024*1024), b''): h.update(block)
        return h.hexdigest()
    return {str(p.relative_to(ROOT)): digest(p) for p in sorted(files) if p.is_file()}


def score_photo(task):
    import numpy as np
    import pandas as pd
    from cvdl.classic_lsb import scores
    from cvdl.standardtest import StandardTestPairs
    from lsb_torch import Corpus, RunSettings
    from lsb_train_grid import crop
    from lsb_engine import embed_fast
    record, rows = task
    frame = pd.DataFrame(rows)
    corpus = Corpus(ROOT, RunSettings(source='decode', sources=('raise', 'div2k', 'alaska', 'lsdir')), records=[record])
    dataset = StandardTestPairs(corpus, frame)
    result = []; clean_cache = {}; parity = []
    for index, row in frame.iterrows():
        clean_t, stego_t, _ = dataset[index]
        clean, stego = [t.permute(1, 2, 0).numpy() for t in (clean_t, stego_t)]
        if index == 0:
            # Independent replay against the evaluator dataset, one sample/photo.
            photo = corpus.photo(record)
            pixels, xy, pad = crop(photo, record, int(row.tile_size), int(row.tile_row), int(row.tile_col), 'overlap_last')
            assert xy == (int(row.y), int(row.x)) and not any(pad)
            kw = json.loads(row.draw_kwargs)
            kw['rng'] = np.random.default_rng(np.random.SeedSequence(json.loads(row.embedding_seed_key)))
            replay, _, _ = embed_fast(pixels.copy(), **kw)
            np.testing.assert_array_equal(clean, pixels)
            np.testing.assert_array_equal(stego, replay)
            parity.append(dict(pair_id=row.pair_id, clean_sha256=hashlib.sha256(clean.tobytes()).hexdigest(),
                               stego_sha256=hashlib.sha256(stego.tobytes()).hexdigest()))
        if row.cover_id not in clean_cache: clean_cache[row.cover_id] = scores(clean)
        for label, values in ((0, clean_cache[row.cover_id]), (1, scores(stego))):
            result.append(dict(**row.to_dict(), label=label, **values))
    return result, parity


def score(frame, workers, stem):
    import pandas as pd
    records = {}
    wanted = set(frame.source_photo)
    for source in ('raise', 'div2k', 'alaska', 'lsdir'):
        with (ROOT/f'data/{source}/photo_manifest.csv').open() as stream:
            records.update({r['photo_id']: r for r in csv.DictReader(stream) if r['photo_id'] in wanted})
    tasks = [(records[photo], group.to_dict('records')) for photo, group in frame.groupby('source_photo', sort=True)]
    rows = []; parity = []
    with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context('spawn')) as pool:
        for i, (values, checks) in enumerate(pool.map(score_photo, tasks, chunksize=1)):
            rows.extend(values); parity.extend(checks)
            if (i+1) % 50 == 0: print(f'{stem}: {i+1}/{len(tasks)} photos', flush=True)
    result = pd.DataFrame(rows)
    result.to_csv(OUT/f'{stem}-scores.csv', index=False, float_format='%.17g')
    (OUT/f'{stem}-pixel-parity.json').write_text(json.dumps(parity, indent=2)+'\n')
    return result


def positive(workers):
    import pandas as pd
    from cvdl.metrics import auc_numpy
    path = ROOT/'data/test/standard-256-v1/pairs-standard.csv'
    frame = pd.read_csv(path, float_precision='round_trip')
    frame = frame[(frame.k == 1) & (frame.method == 'replace') & (frame.requested_bpp >= .35)]
    scored = score(frame, workers, 'positive')
    result = dict(pairs=len(frame), photos=int(frame.source_photo.nunique()),
                  selection='standard-256 standard panel; k1 replacement; requested_bpp >= .35; all placements',
                  ws_auc=auc_numpy(scored.label.to_numpy(), scored.ws.to_numpy()),
                  spa_auc=auc_numpy(scored.label.to_numpy(), scored.spa.to_numpy()))
    result['pass'] = result['ws_auc'] >= .90
    (OUT/'positive.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result), flush=True)
    return result['pass']


def read_frozen(folder, name):
    import pandas as pd
    folder = ROOT/folder
    raw = (folder/name).read_bytes()
    meta = json.loads((folder/'summary.json').read_text())
    expected = meta.get('files_sha256', {}).get(name, meta.get('pairs_sha256'))
    if hashlib.sha256(raw).hexdigest() != expected:
        raise ValueError('frozen CSV hash mismatch')
    import io
    return pd.read_csv(io.BytesIO(raw), float_precision='round_trip')


def main_cells(workers):
    import numpy as np
    import pandas as pd
    frames = []
    for panel in ('alaska-seen', 'alaska-unseen', 'lsdir'):
        f = read_frozen('data/hard-test/256-v1', f'pairs-{panel}.csv')
        f = f[(f.k == 1) & f.condition.isin(['low', 'noisy', 'low_noisy'])].copy()
        f['sample'] = 'test'; frames.append(f)
    val = read_frozen('data/val/60da4391043e8ce48ff2', 'pairs.csv')
    val = val[val.method == 'replace'].sort_values('pair_id').reset_index(drop=True)
    n = len(val)
    if n < 2000: raise ValueError('need the pre-specified 2000 validation replacement pairs')
    indices = np.sort(np.random.default_rng(SEED).choice(n, 2000, replace=False))
    val = val.iloc[indices].copy(); val['sample'] = 'validation'; frames.append(val)
    (OUT/'selection.json').write_text(json.dumps(dict(seed=SEED, validation_eligible=n,
        validation_pairs=2000, validation_scope='all k, replacement, no test-driven filtering',
        bootstrap_replicates=2000, bootstrap_unit='source_photo; clean/stego and repeated tiles kept together',
        threshold_rule='score >= nextafter(clean order statistic, +inf); empirical validation FA <= target'), indent=2)+'\n')
    score(pd.concat(frames, ignore_index=True), workers, 'main')


def validation_cutoff(clean, target):
    import numpy as np
    clean = np.asarray(clean, dtype=float)
    if not len(clean) or not np.isfinite(clean).all(): raise ValueError('invalid validation scores')
    index = int(np.ceil((1-target)*len(clean)))-1
    return float(np.nextafter(np.sort(clean)[index], np.inf))


def attach_saved(frame, path):
    import pandas as pd
    saved = pd.read_csv(path, float_precision='round_trip')
    keys = ['pair_id', 'label']
    if saved.duplicated(keys).any() or frame.duplicated(keys).any(): raise ValueError('ambiguous same-pair join')
    expected = frame[keys+['source_photo']].merge(saved[keys+['source_photo', 'prob_stego']], on=keys,
        suffixes=('_ours', '_saved'), validate='one_to_one', how='left')
    if expected.prob_stego.isna().any() or not expected.source_photo_ours.equals(expected.source_photo_saved):
        raise ValueError('saved SRM scores do not match every requested pair')
    return frame.merge(saved[keys+['prob_stego']], on=keys, how='left', validate='one_to_one').rename(columns={'prob_stego':'srm'})


def auc_intervals(frame, names, replicates=2000):
    import numpy as np
    from cvdl.metrics import auc_numpy
    labels = frame.label.to_numpy(); values = frame[names].to_numpy()
    groups = [g.index.to_numpy() for _, g in frame.reset_index(drop=True).groupby('source_photo', sort=True)]
    estimates = np.array([auc_numpy(labels, values[:, j]) for j in range(len(names))])
    rng = np.random.default_rng(SEED); draws = np.empty((replicates, len(names)))
    for b in range(replicates):
        indices = np.concatenate([groups[i] for i in rng.integers(0, len(groups), len(groups))])
        for j in range(len(names)): draws[b, j] = auc_numpy(labels[indices], values[indices, j])
    intervals = np.quantile(draws, [.025, .975], axis=0)
    return {name: (float(estimates[j]), float(intervals[0,j]), float(intervals[1,j])) for j, name in enumerate(names)}


def descriptive_roc(labels, values):
    """Empirical ROC coordinates only; never select an operating threshold."""
    import numpy as np
    labels = np.asarray(labels); values = np.asarray(values)
    order = np.argsort(-values, kind='stable'); y = labels[order]; s = values[order]
    ends = np.r_[np.flatnonzero(s[:-1] != s[1:]), len(s)-1]
    true_positive = np.cumsum(y == 1)[ends]; false_positive = np.cumsum(y == 0)[ends]
    return np.r_[0, false_positive/(y == 0).sum()], np.r_[1, 1-true_positive/(y == 1).sum()]


def plot_rocs(curves):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    labels = {'alaska-seen':'ALASKA seen', 'alaska-unseen':'ALASKA unseen', 'lsdir':'LSDIR',
              'low':'Small message', 'noisy':'Noisy images', 'low_noisy':'Small + noisy'}
    for placement in ('contiguous', 'random', 'stride'):
        fig, axes = plt.subplots(3, 3, figsize=(12, 10), sharex=True, sharey=True)
        for i, panel in enumerate(('alaska-seen', 'alaska-unseen', 'lsdir')):
            for j, condition in enumerate(('low', 'noisy', 'low_noisy')):
                ax = axes[i, j]
                for name, label in (('ws','WS'), ('spa','SPA'), ('srm','SRM-e50')):
                    f = curves[(curves.method == 'replace') & (curves.placement == placement) &
                        (curves.panel == panel) & (curves.condition == condition) & (curves.detector == name)]
                    ax.plot(100*f.false_alarm, 100*f.miss, label=label)
                ax.set(title=labels[panel]+' — '+labels[condition], xlim=(0,10), ylim=(0,100)); ax.grid(alpha=.2)
                if i == 2: ax.set_xlabel('False alarms (%)')
                if j == 0: ax.set_ylabel('Misses (%)')
        fig.suptitle(f'Replacement, {placement}: descriptive; no threshold chosen from it')
        handles, names = axes[0,0].get_legend_handles_labels(); fig.legend(handles, names, loc='lower center', ncol=3)
        fig.tight_layout(rect=(0,.035,1,.96)); fig.savefig(OUT/f'descriptive-roc-{placement}.png', dpi=120)
        plt.close(fig)


def summarize():
    import numpy as np
    import pandas as pd
    scored = pd.read_csv(OUT/'main-scores.csv', float_precision='round_trip')
    val = attach_saved(scored[scored['sample'] == 'validation'].copy(), EVAL/'validation/predictions.csv')
    test = attach_saved(scored[scored['sample'] == 'test'].copy(), EVAL/'hard-test/hard_predictions.csv')
    names = ['ws', 'spa', 'srm', 'ws_r', 'ws_g', 'ws_b', 'spa_r', 'spa_g', 'spa_b']
    thresholds = {name: {target: validation_cutoff(val.loc[val.label == 0, name], target)
                        for target in (.01, .05, .10)} for name in names}
    (OUT/'thresholds.json').write_text(json.dumps(thresholds, indent=2)+'\n')
    val.to_csv(OUT/'validation-with-srm.csv', index=False, float_format='%.17g')
    test.to_csv(OUT/'test-with-srm.csv', index=False, float_format='%.17g')
    records = []; roc_rows = []
    for key, g in test.groupby(['method', 'panel', 'condition', 'placement'], sort=True):
        g = g.reset_index(drop=True); ci = auc_intervals(g, names)
        for name in names:
            auc, low, high = ci[name]
            result = dict(zip(['method', 'panel', 'condition', 'placement'], key))
            result.update(detector=name, pairs=int((g.label == 1).sum()), photos=int(g.source_photo.nunique()),
                          auc=auc, auc_low=low, auc_high=high)
            for target, cutoff in thresholds[name].items():
                suffix = str(round(100*target))
                result[f'validation_cutoff_{suffix}'] = cutoff
                result[f'validation_fa_{suffix}'] = float((val.loc[val.label == 0, name] >= cutoff).mean())
                result[f'test_fa_{suffix}'] = float((g.loc[g.label == 0, name] >= cutoff).mean())
                result[f'miss_{suffix}'] = float((g.loc[g.label == 1, name] < cutoff).mean())
            records.append(result)
            if name in ('ws', 'spa', 'srm'):
                fa, miss = descriptive_roc(g.label.to_numpy(), g[name].to_numpy())
                for false_alarm, missed in zip(fa, miss):
                    roc_rows.append(dict(zip(['method', 'panel', 'condition', 'placement'], key),
                        detector=name, false_alarm=float(false_alarm), miss=float(missed)))
        print('summarized', key, flush=True)
    result = pd.DataFrame(records); result.to_csv(OUT/'metrics.csv', index=False, float_format='%.17g')
    curves = pd.DataFrame(roc_rows); curves.to_csv(OUT/'descriptive-roc.csv', index=False, float_format='%.17g')
    plot_rocs(curves)
    lines = ['# CHECK 1: frozen-pair classic detectors', '',
        'Descriptive test ROC curves: descriptive-roc.csv and descriptive-roc-*.png. '
        'Descriptive; no threshold chosen from them.', '',
        'AUC [photo-bootstrap 95% interval]; miss at the replacement-validation 5%-FA threshold. '
        'Each threshold is validation-only; metrics.csv also gives actual test FA and misses '
        'at validation 1%, 5% and 10% FA. Two thousand shared photo-cluster replicates; seed 20261019.', '']
    for method in ('replace', 'match'):
        lines += ['## '+method, '', '| Panel | Condition | Placement | Pairs | WS AUC [95%] / miss | SPA AUC [95%] / miss | SRM-e50 AUC [95%] / miss |',
                  '| --- | --- | --- | ---: | --- | --- | --- |']
        subset = result[result.method == method]
        for key, g in subset.groupby(['panel', 'condition', 'placement'], sort=True):
            cells = []
            for name in ('ws', 'spa', 'srm'):
                r = g[g.detector == name].iloc[0]
                cells.append(f'{r.auc:.3f} [{r.auc_low:.3f}, {r.auc_high:.3f}] / {r.miss_5:.1%}')
            lines.append('| '+' | '.join([*key, str(int(g.iloc[0].pairs)), *cells])+' |')
        lines.append('')
    # Factual interval comparisons only; the pre-registered decision rules stay untouched.
    comparisons = []
    for key, g in result[(result.method == 'replace') & result.panel.str.startswith('alaska') &
                         result.condition.isin(['noisy', 'low_noisy'])].groupby(['panel', 'condition', 'placement']):
        srm = g[g.detector == 'srm'].iloc[0]
        for name in ('ws', 'spa'):
            r = g[g.detector == name].iloc[0]
            comparisons.append(dict(panel=key[0], condition=key[1], placement=key[2], detector=name,
                above_nonoverlap=bool(r.auc_low > srm.auc_high), below_nonoverlap=bool(r.auc_high < srm.auc_low)))
    (OUT/'interval-comparisons.json').write_text(json.dumps(comparisons, indent=2)+'\n')
    (OUT/'report.md').write_text('\n'.join(lines)+'\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=['positive', 'main', 'summarize'], required=True)
    parser.add_argument('--workers', type=int, default=6)
    args = parser.parse_args()
    if not 1 <= args.workers <= 12: parser.error('workers must be 1 through 12')
    os.environ.update(CUDA_VISIBLE_DEVICES='', HIP_VISIBLE_DEVICES='', PYTHONDONTWRITEBYTECODE='1',
                      OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
    sys.path[:0] = [str(ROOT), str(ROOT/'LSB-Dataset-Generator')]
    OUT.mkdir(parents=True, exist_ok=True)
    before = snapshot(); (OUT/f'{args.phase}-inputs-before.json').write_text(json.dumps(before, indent=2)+'\n')
    try:
        if args.phase == 'positive':
            passed = positive(args.workers)
        else:
            control = json.loads((OUT/'positive.json').read_text())
            if not control['pass'] or control['ws_auc'] < .90:
                raise RuntimeError('positive control failed; stop without main scoring')
            if args.phase == 'main': main_cells(args.workers)
            summarize()
            passed = True
    finally:
        after = snapshot(); (OUT/f'{args.phase}-inputs-after.json').write_text(json.dumps(after, indent=2)+'\n')
        if after != before: raise RuntimeError('input bytes changed')
    if not passed: raise SystemExit(2)


if __name__ == '__main__': main()
