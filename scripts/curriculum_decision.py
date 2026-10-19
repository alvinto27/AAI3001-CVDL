"""CPU-only curriculum decision evidence from saved, exactly joined predictions.

No inference and no threshold fitting. Photo-cluster percentile bootstrap, 2,000
replicates, seed 20261019. Each replicate uses identical photo draws for models.
Paired observations (including repeated clean tiles) are never deduplicated.
"""
from pathlib import Path
import hashlib
import json
import os
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
SEED = 20261019
REPLICATES = 2000
NAMES = ('srm_e50', 'curriculum', 'control')
TARGETS = ('alaska-seen', 'alaska-unseen', 'lsdir')
INPUTS = (
    ('srm-e50-20261017T111310Z-0214', '35afd1c7938bae78968c'),
    ('srm-low-bpp-curriculum-20261019T051356Z-9566', 'e9ef059bb12c9076a226'),
    ('srm-low-bpp-control-20261019T060708Z-9736', 'af6e9879061fd441462f'),
)
DIFFERENCES = ((1, 0), (1, 2), (2, 0))
OUT = ROOT / 'runs/verification/curriculum-decision-2026-10-19'


def join_predictions(paths):
    """Require the complete key sets and all non-score metadata to match."""
    import numpy as np
    import pandas as pd
    frames = []
    for path in paths:
        frame = pd.read_csv(path, float_precision='round_trip')
        if frame[['pair_id', 'label', 'source_photo']].isna().any().any():
            raise ValueError('missing join key or source photo')
        if frame.duplicated(['pair_id', 'label']).any():
            raise ValueError('duplicate pair_id + label')
        if not frame.label.isin([0, 1]).all() or not np.isfinite(frame.prob_stego).all():
            raise ValueError('invalid labels or scores')
        frame = frame.set_index(['pair_id', 'label']).sort_index()
        frames.append(frame)
    base = frames[0]
    for frame in frames[1:]:
        if not base.index.equals(frame.index):
            raise ValueError('pair_id + label mismatch')
        try:
            pd.testing.assert_frame_equal(base.drop(columns='prob_stego'),
                                          frame.drop(columns='prob_stego'))
        except AssertionError as exc:
            raise ValueError('prediction metadata mismatch') from exc
    result = base.drop(columns='prob_stego').reset_index()
    for name, frame in zip(NAMES, frames, strict=True):
        result[name] = frame.prob_stego.to_numpy()
    pairs = result.groupby('pair_id', sort=False)
    if not pairs.label.agg(list).map(lambda labels: sorted(labels) == [0, 1]).all():
        raise ValueError('each pair must contain one clean and one stego')
    if not (pairs.source_photo.nunique() == 1).all():
        raise ValueError('clean/stego photo mismatch')
    return result


def intervals(frame, *, metric='auc', cutoffs=None, replicates=REPLICATES, seed=SEED):
    """Resample sorted source-photo clusters jointly across all score columns."""
    import numpy as np
    from cvdl.metrics import auc_numpy
    frame = frame.reset_index(drop=True)
    labels = frame.label.to_numpy()
    values = frame[list(NAMES)].to_numpy()
    groups = [g.index.to_numpy() for _, g in frame.groupby('source_photo', sort=True)]
    if not groups:
        raise ValueError('empty bootstrap population')
    if metric == 'pe':
        errors = ((values >= np.asarray(cutoffs)) != labels[:, None])
    elif metric != 'auc':
        raise ValueError('unknown metric')

    def estimate(indices):
        if metric == 'auc':
            return np.array([auc_numpy(labels[indices], values[indices, j]) for j in range(3)])
        # Each sampled photo keeps both rows of every pair: error rate = P_E.
        return errors[indices].mean(axis=0)

    point = estimate(np.arange(len(frame)))
    rng = np.random.default_rng(seed)
    draws = np.empty((replicates, 3))
    for b in range(replicates):
        indices = np.concatenate([groups[i] for i in rng.integers(0, len(groups), len(groups))])
        draws[b] = estimate(indices)
    ci = np.quantile(draws, [.025, .975], axis=0)
    models = {name: dict(point=float(point[j]), low=float(ci[0, j]), high=float(ci[1, j]))
              for j, name in enumerate(NAMES)}
    differences = {}
    for a, b in DIFFERENCES:
        low, high = np.quantile(draws[:, a] - draws[:, b], [.025, .975])
        differences[NAMES[a] + '-' + NAMES[b]] = dict(
            point=float(point[a] - point[b]), low=float(low), high=float(high))
    return dict(pairs=int((labels == 1).sum()), photos=len(groups), models=models,
                differences=differences), draws


def main():
    import numpy as np
    import pandas as pd
    OUT.mkdir(parents=True, exist_ok=True)
    folders = [ROOT / 'runs' / run / 'evaluation/standard-256-v1' / eid for run, eid in INPUTS]
    hard_paths = [p / 'hard-test/hard_predictions.csv' for p in folders]
    standard_paths = [p / 'panels/standard/predictions.csv' for p in folders]
    summaries = [p / 'summary.json' for p in folders]
    all_paths = [*hard_paths, *standard_paths, *summaries]
    def snapshot():
        return {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in all_paths}
    before = snapshot()
    hard = join_predictions(hard_paths)
    standard = join_predictions(standard_paths)
    cutoffs = [json.loads(p.read_text())['validation_threshold']['cutoff'] for p in summaries]
    low = hard[hard.panel.isin(TARGETS) & (hard.k == 1) & (hard.condition == 'low')]
    if set(low.method) != {'replace', 'match'} or set(low.placement) != {'contiguous', 'random', 'stride'}:
        raise ValueError('target method/placement coverage mismatch')
    pooled, pooled_draws = intervals(low)
    guard, guard_draws = intervals(standard, metric='pe', cutoffs=cutoffs)
    contexts = {}
    draws = {'pooled_auc': pooled_draws, 'standard_pe': guard_draws}
    for field, values in (('placement', ('contiguous', 'random', 'stride')), ('panel', TARGETS)):
        for value in values:
            key = field + '/' + value
            contexts[key], draws[key.replace('/', '_')] = intervals(low[low[field] == value])
    if before != snapshot():
        raise RuntimeError('saved input files changed')
    checks = dict(
        curriculum_above_srm_nonoverlap=pooled['models']['curriculum']['low'] > pooled['models']['srm_e50']['high'],
        curriculum_above_control_nonoverlap=pooled['models']['curriculum']['low'] > pooled['models']['control']['high'],
        standard_point_guard=guard['differences']['curriculum-srm_e50']['point'] <= .005)
    result = dict(seed=SEED, replicates=REPLICATES, interval='percentile 2.5/97.5%',
        bootstrap_unit='source_photo; sorted clusters; pairs/repeated clean tiles retained; same draws for three models within each scope',
        input_sha256=before, model_order=NAMES, cutoffs=dict(zip(NAMES, cutoffs)), pooled_auc=pooled,
        standard_pe=guard, context_auc=contexts, literal_checks=checks)
    (OUT / 'evidence.json').write_text(json.dumps(result, indent=2) + '\n')
    np.savez_compressed(OUT / 'bootstrap-replicates.npz', **draws)
    rows = []
    for scope, item in [('pooled', pooled), *contexts.items()]:
        for name, record in item['models'].items():
            rows.append(dict(scope=scope, model=name, pairs=item['pairs'], photos=item['photos'], **record))
    pd.DataFrame(rows).to_csv(OUT / 'auc-intervals.csv', index=False, float_format='%.17g')
    lines = ['# Curriculum decision evidence', '',
        'Saved scores only; exact pair_id + label joins and all non-score metadata checked. '
        '2,000 photo-cluster bootstrap replicates; seed 20261019; shared model draws. '
        'Repeated clean observations retained. Percentile 95% intervals.', '',
        '## Pooled hard k=1 low AUC', '',
        '| Model | AUC | 95% interval |', '| --- | ---: | --- |']
    for name, r in pooled['models'].items():
        lines.append(f'| {name} | {r["point"]:.8f} | [{r["low"]:.8f}, {r["high"]:.8f}] |')
    lines += ['', '## Paired AUC differences', '', '| Difference | AUC difference | 95% interval |', '| --- | ---: | --- |']
    for name, r in pooled['differences'].items():
        lines.append(f'| {name} | {r["point"]:+.8f} | [{r["low"]:+.8f}, {r["high"]:+.8f}] |')
    lines += ['', '## Standard-256 guard (each model at its own validation cut-off)', '',
        '| Model | P_E (%) | 95% interval (%) |', '| --- | ---: | --- |']
    for name, r in guard['models'].items():
        lines.append(f'| {name} | {100*r["point"]:.6f} | [{100*r["low"]:.6f}, {100*r["high"]:.6f}] |')
    r = guard['differences']['curriculum-srm_e50']
    lines += ['', f'Curriculum minus SRM-e50: {100*r["point"]:+.6f} percentage points '
        f'[{100*r["low"]:+.6f}, {100*r["high"]:+.6f}]. The literal point guard is +0.5 points.', '',
        '## Context only: hard k=1 low AUC', '', '| Scope | Model | AUC | 95% interval |', '| --- | --- | ---: | --- |']
    for key, item in contexts.items():
        for name, r in item['models'].items():
            lines.append(f'| {key} | {name} | {r["point"]:.8f} | [{r["low"]:.8f}, {r["high"]:.8f}] |')
    lines += ['', '## Literal rule and boundary evidence', '',
        'The applicable rule row is quoted literally, not replaced by a paired-difference rule:', '',
        '| Otherwise | No gain; the k=1 low-bpp limit is not a training-order problem |', '',
        'The +0.5-point guard uses the point difference, not its interval. '
        'The interval is context only. Overlapping model AUC intervals must not be replaced '
        'by a significant paired difference. See the intervals above; the Orchestrator owns '
        'the final decision and any borderline interpretation.', '', '```json', json.dumps(checks, indent=2), '```']
    (OUT / 'report.md').write_text('\n'.join(lines) + '\n')
    print(json.dumps(dict(pooled_auc=pooled, standard_pe=guard, literal_checks=checks), indent=2))


if __name__ == '__main__':
    os.environ.update(CUDA_VISIBLE_DEVICES='', HIP_VISIBLE_DEVICES='', PYTHONDONTWRITEBYTECODE='1',
                      OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
    main()
