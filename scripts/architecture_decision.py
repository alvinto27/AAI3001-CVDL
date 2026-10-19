"""CPU-only arm-A evidence from saved scores; no inference or threshold fitting.

Same method as curriculum_decision.py: sorted source-photo clusters, retained
paired rows/repeated clean tiles, shared draws, percentile 95% intervals,
2,000 replicates and seed 20261019. The architecture rule uses paired AUC
DIFFERENCES and an AUC guard, not the curriculum's model-interval/PE rule.
"""
from pathlib import Path
import argparse
import hashlib
import json
import os
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
NAMES = ('srm30', 'delayed')
PANELS = ('standard', 'alaska-seen', 'alaska-unseen', 'lsdir')
TARGETS = PANELS[1:]
SEED = 20261019
REPLICATES = 2000
MINIMUM_GAIN = .03
BASELINE_RUN = 'srm-20261017T052857Z-4521'
BASELINE_EVALUATION = '11a82bf812e573c7c62d'
DELAYED_RUN = 'srm-delayed-20261019T100714Z-3282'
OUT = ROOT / 'runs/verification/architecture-decision-2026-10-19'


def join_predictions(paths):
    """Fail on any complete key-set, paired-row, or non-score metadata mismatch."""
    import numpy as np
    import pandas as pd
    if len(paths) != 2:
        raise ValueError('architecture comparison needs exactly two models')
    frames = []
    for path in paths:
        frame = pd.read_csv(path, float_precision='round_trip')
        if frame[['pair_id', 'label', 'source_photo']].isna().any().any():
            raise ValueError('missing join key or source photo')
        if frame.duplicated(['pair_id', 'label']).any():
            raise ValueError('duplicate pair_id + label')
        if not frame.label.isin([0, 1]).all() or not np.isfinite(frame.prob_stego).all():
            raise ValueError('invalid labels or scores')
        frames.append(frame.set_index(['pair_id', 'label']).sort_index())
    base, candidate = frames
    if not base.index.equals(candidate.index):
        raise ValueError('pair_id + label mismatch')
    try:
        pd.testing.assert_frame_equal(base.drop(columns='prob_stego'), candidate.drop(columns='prob_stego'))
    except AssertionError as error:
        raise ValueError('prediction metadata mismatch') from error
    result = base.drop(columns='prob_stego').reset_index()
    for name, frame in zip(NAMES, frames, strict=True):
        result[name] = frame.prob_stego.to_numpy()
    pairs = result.groupby('pair_id', sort=False)
    if not pairs.label.agg(list).map(lambda labels: sorted(labels) == [0, 1]).all():
        raise ValueError('each pair must contain one clean and one stego')
    if not (pairs.source_photo.nunique() == 1).all():
        raise ValueError('clean/stego photo mismatch')
    return result


def auc_intervals(frame, *, replicates=REPLICATES, seed=SEED):
    """One photo draw per replicate, applied to both models before differencing."""
    import numpy as np
    from cvdl.metrics import auc_numpy
    frame = frame.reset_index(drop=True)
    labels = frame.label.to_numpy()
    scores = frame[list(NAMES)].to_numpy()
    groups = [g.index.to_numpy() for _, g in frame.groupby('source_photo', sort=True)]
    if not groups:
        raise ValueError('empty bootstrap population')

    def estimate(indices):
        return np.array([auc_numpy(labels[indices], scores[indices, j]) for j in range(2)])

    point = estimate(np.arange(len(frame)))
    rng = np.random.default_rng(seed)
    draws = np.empty((replicates, 2))
    for b in range(replicates):
        indices = np.concatenate([groups[i] for i in rng.integers(0, len(groups), len(groups))])
        draws[b] = estimate(indices)
    low, high = np.quantile(draws, [.025, .975], axis=0)
    delta_low, delta_high = np.quantile(draws[:, 1] - draws[:, 0], [.025, .975])
    models = {name: dict(point=float(point[j]), low=float(low[j]), high=float(high[j]))
              for j, name in enumerate(NAMES)}
    return dict(pairs=int((labels == 1).sum()), photos=len(groups), models=models,
        difference=dict(point=float(point[1] - point[0]), low=float(delta_low), high=float(delta_high))), draws


def rates(frame, cutoffs):
    """Frozen validation cut-offs only; preserve the evaluator's >= tie rule."""
    clean = frame[frame.label == 0]
    stego = frame[frame.label == 1]
    result = {}
    for name in NAMES:
        cutoff = cutoffs[name]
        fa = float((clean[name] >= cutoff).mean())
        miss = float((stego[name] < cutoff).mean())
        result[name] = dict(cutoff=cutoff, fa=fa, miss=miss, pe=(fa + miss) / 2)
    return result


def literal_checks(main_difference, standard_difference):
    """Registered paired-interval checks, including exact boundary meanings."""
    return dict(pooled_difference_interval_above_zero=main_difference['low'] > 0,
        pooled_point_gain_at_least_003=main_difference['point'] >= MINIMUM_GAIN,
        standard_auc_difference_interval_not_wholly_below_zero=standard_difference['high'] >= 0)


def summarize(evaluation_id, output=OUT):
    import numpy as np
    import pandas as pd
    if len(evaluation_id) != 20 or any(c not in '0123456789abcdef' for c in evaluation_id):
        raise ValueError('invalid delayed evaluation id')
    output = Path(output)
    if output.resolve() != OUT.resolve():
        raise ValueError('architecture evidence must use its dedicated output folder')
    folders = [ROOT/'runs'/run/'evaluation/standard-256-v1'/eid for run, eid in
               ((BASELINE_RUN, BASELINE_EVALUATION), (DELAYED_RUN, evaluation_id))]
    hard_paths = [p/'hard-test/hard_predictions.csv' for p in folders]
    panel_paths = {panel: [p/f'panels/{panel}/predictions.csv' for p in folders] for panel in PANELS}
    summary_paths = [p/'summary.json' for p in folders]
    metadata_paths = [p/'evaluation_settings.json' for p in folders]
    status_paths = [ROOT/'runs'/run/'status.json' for run in (BASELINE_RUN, DELAYED_RUN)]
    plan = ROOT/'docs/active/plan-architecture-tests.md'
    paths = [*hard_paths, *[p for pair in panel_paths.values() for p in pair],
             *summary_paths, *metadata_paths, *status_paths, plan]

    def snapshot():
        return {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}

    before = snapshot()
    summaries = [json.loads(p.read_text()) for p in summary_paths]
    metadata = [json.loads(p.read_text()) for p in metadata_paths]
    thresholds = [s['validation_threshold'] for s in summaries]
    for folder, summary, meta, threshold in zip(folders, summaries, metadata, thresholds, strict=True):
        if (summary['evaluation_id'] != folder.name or meta['evaluation_id'] != folder.name or
                summary['suite'] != 'standard-256-v1' or summary.get('quick') or
                summary.get('hard_test', {}).get('status') != 'complete' or
                not threshold.get('deployment') or threshold.get('status') != 'complete'):
            raise ValueError('incomplete or non-deployment saved evaluation')
    if thresholds[0]['recipe_id'] != thresholds[1]['recipe_id']:
        raise ValueError('validation recipe mismatch')
    cutoffs = {name: t['cutoff'] for name, t in zip(NAMES, thresholds, strict=True)}
    hard = join_predictions(hard_paths)
    low = hard[hard.panel.isin(TARGETS) & (hard.k == 1) & (hard.condition == 'low')]
    if set(low.method) != {'replace', 'match'} or set(low.placement) != {'contiguous', 'random', 'stride'}:
        raise ValueError('target method/placement coverage mismatch')
    main, main_draws = auc_intervals(low)
    draws = {'pooled_hard_auc': main_draws}
    standard = {}
    for panel, pair in panel_paths.items():
        frame = join_predictions(pair)
        item, draws['standard_' + panel.replace('-', '_')] = auc_intervals(frame)
        item['rates'] = rates(frame, cutoffs)
        standard[panel] = item
    contexts = {}
    for column, choices in (('panel', TARGETS), ('placement', ('contiguous', 'random', 'stride'))):
        for choice in choices:
            key = column + '/' + choice
            contexts[key], draws[key.replace('/', '_').replace('-', '_')] = auc_intervals(low[low[column] == choice])
    checks = literal_checks(main['difference'], standard['standard']['difference'])
    training = {name: dict(notebook_minutes=json.loads(p.read_text())['elapsed_s'] / 60)
                for name, p in zip(NAMES, status_paths, strict=True)}
    after = snapshot()
    if before != after:
        raise RuntimeError('saved inputs changed during analysis')
    result = dict(seed=SEED, replicates=REPLICATES, interval='percentile 2.5/97.5%',
        model_order=NAMES, evaluations=dict(zip(NAMES, [BASELINE_EVALUATION, evaluation_id])),
        bootstrap_unit='sorted source_photo clusters; paired rows and repeated clean observations retained; same photo draws for both models within scope',
        input_sha256=before, input_sha256_after=after, inputs_unchanged=True,
        threshold_metadata=dict(zip(NAMES, thresholds)),
        pooled_hard_auc=main, standard_panels=standard, context_hard_auc=contexts,
        minimum_gain=MINIMUM_GAIN, literal_checks=checks, training=training)
    output.mkdir(parents=True, exist_ok=True)
    (output/'evidence.json').write_text(json.dumps(result, indent=2) + '\n')
    np.savez_compressed(output/'bootstrap-replicates.npz', **draws)
    records = []
    for scope, item in [('pooled', main), *contexts.items(), *[('standard/' + k, v) for k, v in standard.items()]]:
        for name, value in item['models'].items():
            records.append(dict(scope=scope, model=name, pairs=item['pairs'], photos=item['photos'], **value))
        records.append(dict(scope=scope, model='delayed-srm30', pairs=item['pairs'], photos=item['photos'], **item['difference']))
    pd.DataFrame(records).to_csv(output/'auc-intervals.csv', index=False, float_format='%.17g')
    panel_records = [dict(panel=panel, model=name, auc=item['models'][name]['point'], **value)
                     for panel, item in standard.items() for name, value in item['rates'].items()]
    pd.DataFrame(panel_records).to_csv(output/'panel-rates.csv', index=False, float_format='%.17g')
    write_report(result, output)
    print(json.dumps(dict(pooled_hard_auc=main, standard_auc=standard['standard'], literal_checks=checks), indent=2))
    return result


def write_report(result, output):
    main = result['pooled_hard_auc']
    guard = result['standard_panels']['standard']
    lines = ['# Delayed pooling: saved-score architecture decision evidence', '',
        'SRM-30 vs arm A. No inference or threshold fitting. Exact pair_id + label and all non-score metadata joins. '
        '2,000 photo-cluster replicates, seed 20261019, identical draws for both models. '
        'Repeated clean observations retained. All intervals are percentile 95%.', '',
        '## Registered measurements', '', '| Measure | SRM-30 AUC [95%] | Delayed AUC [95%] | Paired difference [95%] |',
        '| --- | --- | --- | --- |']
    def cell(r):
        return f'{r["point"]:.8f} [{r["low"]:.8f}, {r["high"]:.8f}]'
    for label, item in [('Pooled hard k=1 low', main), ('Standard-256 guard', guard)]:
        lines.append(f'| {label} | {cell(item["models"]["srm30"])} | {cell(item["models"]["delayed"])} | {cell(item["difference"])} |')
    lines += ['', '## Literal checks (all three required for "The arm helps")', '',
        'The guard is AUC, not P_E. The comparison uses paired difference intervals, not overlap of model intervals.', '',
        '```json', json.dumps(result['literal_checks'], indent=2), '```', '',
        '| Result | Meaning |', '| --- | --- |',
        '| Paired pooled AUC difference vs SRM-30 has a 95% interval above 0 **and** a point gain >= D6, **and** the standard-256 AUC difference interval is not wholly below 0 | **The arm helps** |',
        '| Interval above 0 but gain < D6 | Possible small gain; inside likely chance (no seed repeat): no claim |',
        '| Otherwise | No gain from this change |', '',
        'D6 = +0.03 AUC. No threshold or rule was changed.', '',
        '## Standard-256 four-panel operating points (context only)', '',
        'Each model uses its own saved validation-minimum-P_E cut-off. Rates below are percentages.', '',
        '| Panel | Model | Cut-off | P_E (%) | FA (%) | Miss (%) | AUC [95%] |',
        '| --- | --- | ---: | ---: | ---: | ---: | --- |']
    for panel, item in result['standard_panels'].items():
        for name, r in item['rates'].items():
            lines.append(f'| {panel} | {name} | {r["cutoff"]:.10f} | {100*r["pe"]:.6f} | {100*r["fa"]:.6f} | {100*r["miss"]:.6f} | {cell(item["models"][name])} |')
    lines += ['', '## Hard k=1 low: panel/placement context only', '',
        '| Scope | SRM-30 AUC [95%] | Delayed AUC [95%] | Paired difference [95%] |', '| --- | --- | --- | --- |']
    for scope, item in result['context_hard_auc'].items():
        lines.append(f'| {scope} | {cell(item["models"]["srm30"])} | {cell(item["models"]["delayed"])} | {cell(item["difference"])} |')
    lines += ['', '## Run time', '',
        f'Delayed pooling: **216 minutes** (saved notebook elapsed {result["training"]["delayed"]["notebook_minutes"]:.3f} min, including runner overhead). '
        f'SRM-30 saved notebook elapsed {result["training"]["srm30"]["notebook_minutes"]:.3f} min. '
        'Real cloud measurements, not the superseded PC synchronize-callback forecast.']
    (output/'report.md').write_text('\n'.join(lines) + '\n')


if __name__ == '__main__':
    os.environ.update(CUDA_VISIBLE_DEVICES='', HIP_VISIBLE_DEVICES='', PYTHONDONTWRITEBYTECODE='1',
                      OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evaluation-id', required=True, help='delayed run evaluation id')
    args = parser.parse_args()
    summarize(args.evaluation_id)
