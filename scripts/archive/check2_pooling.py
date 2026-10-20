"""Opt-in CHECK 2: one SRM-e50 pass, three margin-ranked cell poolings.

Inference uses the evaluator's photo-owned tasks and 16 pairs/32 images per
GPU batch, rocm_conv1_fix=off. No training/evaluation records are written.
Each pooling has one validation-minimum-P_E threshold with the existing tie rule.
User amendment: scores are float64 stego-minus-clean pooled-logit margins;
softmax probabilities are retained for parity only. The superseded float32 pass
is kept in the output directory, with no decision made from it.
Decision scope is fixed in docs/active/plan-diagnostic-checks.md before results.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT/'runs/srm-e50-20261017T111310Z-0214'
EVAL = RUN/'evaluation/standard-256-v1/35afd1c7938bae78968c'
OUT = ROOT/'runs/verification/diagnostics-2026-10-19/check2'
MODES = ('mean', 'max', 'top4')
TARGETS = ('alaska-seen', 'alaska-unseen', 'lsdir')
PARITY_ATOL = 1e-4  # pre-result probability tolerance; fail rather than tune
SEED = 20261019


def score(network, dataset, saved_path, name, workers):
    import numpy as np
    import pandas as pd
    import torch
    from torch.nn import functional as F
    from torch.utils.data import DataLoader
    from lsb_torch import PhotoPairTasks, initialize_worker, prefetch_for_images
    from cvdl.archive.cell_pooling import pooled_probabilities, pooled_logits64
    from scripts.archive.check1_classic import attach_saved
    tasks = PhotoPairTasks(dataset)
    kwargs = dict(batch_size=None, num_workers=workers, pin_memory=True, worker_init_fn=initialize_worker)
    if workers: kwargs.update(multiprocessing_context='spawn', prefetch_factor=prefetch_for_images(tasks.max_images, 128))
    rows = []; pixels = []; algebraic_max = 0.; forward_max = 0.; total = 0
    torch.cuda.synchronize(); start = time.perf_counter()
    with torch.inference_mode():
        for indices, clean, stego in DataLoader(tasks, **kwargs):
            # Sample proof that the photo-task loader emits evaluator pixels.
            if len(pixels) < 8:
                index = int(indices[0]); c, s, _ = dataset[index]
                assert torch.equal(clean[0], c) and torch.equal(stego[0], s)
                pixels.append(dict(pair_id=dataset.frame.iloc[index].pair_id,
                    layout='CHW uint8', clean_sha256=hashlib.sha256(c.numpy().tobytes()).hexdigest(),
                    stego_sha256=hashlib.sha256(s.numpy().tobytes()).hexdigest()))
            for lo in range(0, len(indices), 16):
                part = slice(lo, lo+16); count = len(clean[part])
                images = torch.cat([clean[part], stego[part]], 0).to('cuda', dtype=torch.float32)
                features = network.features(images)
                cells = F.conv2d(features, network.classifier.weight[:, :, None, None], network.classifier.bias)
                if tuple(cells.shape[1:]) != (2, 8, 8): raise ValueError('expected two logits per 8x8 cell grid')
                probabilities = pooled_probabilities(cells)
                logits64 = pooled_logits64(cells)
                margins = {mode: values[:,1]-values[:,0] for mode,values in logits64.items()}
                original = network.classifier(features.mean((-2,-1))).softmax(1)[:,1]
                algebraic_max = max(algebraic_max, float((original-probabilities['mean']).abs().max()))
                if not rows:
                    forward_max = float((network(images).softmax(1)[:,1]-probabilities['mean']).abs().max())
                arrays = {mode: values.cpu().numpy() for mode, values in margins.items()}
                pooled_arrays = {mode: values.cpu().numpy() for mode, values in logits64.items()}
                probability_arrays = {mode: values.cpu().numpy() for mode, values in probabilities.items()}
                sigmoid_mean = margins['mean'].sigmoid().cpu().numpy()
                for position, index in enumerate(indices[part].tolist()):
                    for label, offset in ((0, 0), (1, count)):
                        at = position+offset
                        extra = {}
                        for mode in MODES:
                            extra.update({f'{mode}_logit_clean': float(pooled_arrays[mode][at,0]),
                                f'{mode}_logit_stego': float(pooled_arrays[mode][at,1]),
                                f'{mode}_float32_probability': float(probability_arrays[mode][at])})
                        rows.append(dict(index=int(index), label=label, mean_sigmoid_probability=float(sigmoid_mean[at]),
                            **{mode: float(values[at]) for mode, values in arrays.items()}, **extra))
            total += len(indices)
            if total % 1000 < len(indices): print(name, total, '/', len(dataset), flush=True)
    torch.cuda.synchronize(); seconds = time.perf_counter()-start
    values = pd.DataFrame(rows).sort_values(['index','label']).reset_index(drop=True)
    meta = dataset.frame.iloc[values['index'].to_numpy()].reset_index(drop=True)
    result = pd.concat([meta, values.drop(columns='index')], axis=1)
    result = attach_saved(result, saved_path).rename(columns={'srm':'saved_prob_stego'})
    differences = np.abs(result.mean_sigmoid_probability-result.saved_prob_stego)
    parity = dict(name=name, pairs=len(dataset), images=len(result), seconds=seconds,
        saved_max_abs=float(differences.max()), saved_mean_abs=float(differences.mean()),
        score_kind='float64 pooled logit margin; sigmoid only for mean parity',
        saved_float32_max_abs=float(np.abs(result.mean_float32_probability-result.saved_prob_stego).max()),
        saved_p99_abs=float(np.quantile(differences, .99)), algebraic_max_abs=algebraic_max,
        original_forward_sample_max_abs=forward_max, probability_tolerance=PARITY_ATOL,
        worst_pair_id=str(result.iloc[int(differences.to_numpy().argmax())].pair_id))
    parity['pass'] = max(parity['saved_max_abs'], algebraic_max, forward_max) <= PARITY_ATOL
    result.to_csv(OUT/f'{name}-scores.csv', index=False, float_format='%.17g')
    (OUT/f'{name}-pixel-parity.json').write_text(json.dumps(pixels, indent=2)+'\n')
    (OUT/f'{name}-parity.json').write_text(json.dumps(parity, indent=2)+'\n')
    print(json.dumps(parity), flush=True)
    if not parity['pass']: raise ValueError('mean pooling parity failed; no analysis allowed')
    return parity


def infer(workers):
    import numpy as np
    import torch
    from cvdl import deployment, standardtest, hardtest256
    from cvdl.models import model_from_hparams
    from cvdl.lit import select_best_checkpoint
    from lsb_torch import Corpus, RunSettings
    torch.set_num_threads(1); torch.manual_seed(0); np.random.seed(0)
    torch.backends.cudnn.benchmark = False
    meta = json.loads((EVAL/'evaluation_settings.json').read_text())
    folder = RUN/'checkpoints'/RUN.name
    checkpoint = Path(select_best_checkpoint('', folder))
    checkpoint_sha = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    if checkpoint_sha != meta['checkpoint_sha256']: raise ValueError('checkpoint differs from saved evaluation')
    if meta['batch_size'] != 32 or meta['rocm_conv1_fix']: raise ValueError('unexpected evaluator settings')
    if not torch.cuda.is_available(): raise RuntimeError('granted GPU is unavailable')
    validation, _, _, _ = deployment.load_recipe(ROOT, '60da4391043e8ce48ff2')
    standard, _, _ = standardtest.load_frozen(ROOT)
    hard, _, _ = hardtest256.load_frozen(ROOT)
    deployment.assert_disjoint(validation, [*standard.values(), hard])
    corpus = Corpus(ROOT, RunSettings(sources=('raise','div2k','alaska','lsdir'), source='decode'))
    saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
    network = model_from_hparams(saved['hyper_parameters']).eval()
    network.load_state_dict({k.removeprefix('net.'): v for k,v in saved['state_dict'].items()}, strict=True)
    if network.front_end != 'srm': raise ValueError('expected SRM front end')
    network.to('cuda')
    (OUT/'inference-settings.json').write_text(json.dumps(dict(checkpoint=str(checkpoint), checkpoint_sha256=checkpoint_sha,
        rocm_conv1_fix=False, eval_batch_size=32, pairs_per_batch=16, workers=workers,
        torch=str(torch.__version__), numpy=np.__version__, device=torch.cuda.get_device_name(),
        pooling='cast cell logits to float64; rank stego-clean margins; average paired logits; subtract in float64',
        score_kind='float64 logit margin (user amendment); probabilities for parity only'), indent=2)+'\n')
    timings = [score(network, standardtest.StandardTestPairs(corpus, validation), EVAL/'validation/predictions.csv', 'validation', workers)]
    for name, frame in standard.items():
        panel = str(frame.panel.iloc[0])
        timings.append(score(network, standardtest.StandardTestPairs(corpus, frame), EVAL/f'panels/{panel}/predictions.csv', panel, workers))
    timings.append(score(network, standardtest.StandardTestPairs(corpus, hard), EVAL/'hard-test/hard_predictions.csv', 'hard', workers))
    (OUT/'parity.json').write_text(json.dumps(timings, indent=2)+'\n')


def scalar_metrics(frame, mode, cutoff):
    import numpy as np
    from cvdl.metrics import auc_numpy
    clean = frame[frame.label == 0]; stego = frame[frame.label == 1]
    fa = float((clean[mode] >= cutoff).mean()); miss = float((stego[mode] < cutoff).mean())
    return dict(pairs=len(stego), photos=int(frame.source_photo.nunique()), fa=fa, miss=miss, pe=(fa+miss)/2,
                auc=auc_numpy(frame.label.to_numpy(), frame[mode].to_numpy()))


def wilson_rate(events, n):
    from cvdl.hardtest import wilson
    lo, hi = wilson(events, n)
    return dict(events=int(events), n=int(n), rate=events/n, low=float(lo), high=float(hi))


def paired_change(frame, event, cutoff, modes, replicates=2000):
    """Candidate-minus-mean event rate, photo-cluster bootstrap (fixed seed)."""
    import numpy as np
    if event == 'miss': matrix = np.column_stack([(frame[m] < cutoff[m]).to_numpy() for m in modes])
    elif event == 'error': matrix = np.column_stack([((frame[m] >= cutoff[m]) != frame.label).to_numpy() for m in modes])
    else: matrix = np.column_stack([(frame[m] >= cutoff[m]).to_numpy() for m in modes])
    groups = [g.index.to_numpy() for _,g in frame.reset_index(drop=True).groupby('source_photo', sort=True)]
    rng = np.random.default_rng(SEED); draws = []
    for _ in range(replicates):
        indices = np.concatenate([groups[i] for i in rng.integers(0,len(groups),len(groups))])
        rates = matrix[indices].mean(0); draws.append(rates-rates[0])
    ci = np.quantile(draws, [.025,.975], axis=0); point = matrix.mean(0)-matrix[:,0].mean()
    return {m: dict(change=float(point[i]), low=float(ci[0,i]), high=float(ci[1,i])) for i,m in enumerate(modes)}


def margin_ties(frame, mode):
    """Exact float64 value ties; duplicate clean observations are not removed."""
    import numpy as np
    values = frame[mode].to_numpy(dtype=np.float64)
    if not np.isfinite(values).all(): raise ValueError('non-finite logit margins')
    unique, counts = np.unique(values, return_counts=True)
    cross_label = int((frame.groupby(mode).label.nunique() > 1).sum())
    return dict(images=len(values), unique_margins=len(unique),
        tied_groups=int((counts > 1).sum()), images_in_tied_groups=int(counts[counts > 1].sum()),
        duplicate_excess=int(len(values)-len(unique)), clean_stego_tied_groups=cross_label)


def context_auc(hard, panel_rows, replicates=2000):
    """Threshold-free context only; not part of the registered decision."""
    from scripts.archive.check1_classic import auc_intervals
    result = []
    for placement in ('contiguous', 'random'):
        frame = hard[hard.panel.isin(TARGETS) & (hard.k == 1) & (hard.condition == 'low') &
                     (hard.placement == placement)].reset_index(drop=True)
        intervals = auc_intervals(frame, list(MODES), replicates=replicates)
        for mode in MODES:
            auc, lo, hi = intervals[mode]
            result.append(dict(context='target-hard-low-'+placement, pooling=mode,
                pairs=int((frame.label == 1).sum()), photos=int(frame.source_photo.nunique()),
                auc=auc, auc_low=lo, auc_high=hi))
    for r in panel_rows:
        if r['panel'] in ('standard','alaska-seen'):
            result.append(dict(context='overall-'+r['panel'], pooling=r['pooling'],
                **{k:r[k] for k in ('pairs','photos','auc','auc_low','auc_high')}))
    return result


def summarize():
    import numpy as np
    import pandas as pd
    from cvdl.metrics import threshold
    from scripts.archive.check1_classic import auc_intervals
    parity = json.loads((OUT/'parity.json').read_text())
    if len(parity) != 6 or not all(p['pass'] for p in parity): raise ValueError('incomplete/failed parity')
    validation = pd.read_csv(OUT/'validation-scores.csv', float_precision='round_trip')
    cutoff = {}; val_rows = []; tie_rows = []
    for mode in MODES:
        f = validation.assign(prob_stego=validation[mode]); cutoff[mode], _ = threshold(f, 'pe')
        val_rows.append(dict(pooling=mode, cutoff=cutoff[mode], **scalar_metrics(validation, mode, cutoff[mode])))
        tie_rows.append(dict(dataset='validation', pooling=mode, **margin_ties(validation, mode)))
    pd.DataFrame(val_rows).to_csv(OUT/'validation-metrics.csv', index=False, float_format='%.17g')
    (OUT/'thresholds.json').write_text(json.dumps(cutoff, indent=2)+'\n')
    panel_rows = []; low_rows = []; noise_rows = []
    for panel in ('standard','alaska-seen','alaska-unseen','lsdir'):
        frame = pd.read_csv(OUT/f'{panel}-scores.csv', float_precision='round_trip')
        intervals = auc_intervals(frame, list(MODES))
        for mode in MODES:
            tie_rows.append(dict(dataset=panel, pooling=mode, **margin_ties(frame, mode)))
            _, lo, hi = intervals[mode]
            panel_rows.append(dict(panel=panel, pooling=mode, **scalar_metrics(frame, mode, cutoff[mode]), auc_low=lo, auc_high=hi))
        low = frame[(frame.label == 1) & (frame.k == 1) & (frame.requested_bpp <= .15)]
        for placement, g in low.groupby('placement'):
            for mode in MODES:
                low_rows.append(dict(suite='standard', panel=panel, placement=placement, pooling=mode,
                    **wilson_rate(int((g[mode] < cutoff[mode]).sum()), len(g))))
    hard = pd.read_csv(OUT/'hard-scores.csv', float_precision='round_trip')
    tie_rows.extend(dict(dataset='hard', pooling=mode, **margin_ties(hard, mode)) for mode in MODES)
    pd.DataFrame(tie_rows).to_csv(OUT/'margin-ties.csv', index=False)
    for panel, group in hard.groupby('panel'):
        low = group[(group.label == 1) & (group.k == 1) & (group.condition == 'low')]
        for placement, g in low.groupby('placement'):
            for mode in MODES:
                low_rows.append(dict(suite='hard', panel=panel, placement=placement, pooling=mode,
                    **wilson_rate(int((g[mode] < cutoff[mode]).sum()), len(g))))
        for (k, condition), g in group[(group.label == 0) & group.condition.isin(['noisy','low_noisy'])].groupby(['k','condition']):
            g = g.drop_duplicates('cover_id')
            for mode in MODES:
                noise_rows.append(dict(panel=panel, k=int(k), condition=condition, pooling=mode,
                    **wilson_rate(int((g[mode] >= cutoff[mode]).sum()), len(g))))
    pd.DataFrame(panel_rows).to_csv(OUT/'panel-metrics.csv', index=False, float_format='%.17g')
    pd.DataFrame(low_rows).to_csv(OUT/'low-bpp-misses.csv', index=False, float_format='%.17g')
    pd.DataFrame(noise_rows).to_csv(OUT/'noisy-clean-fa.csv', index=False, float_format='%.17g')
    context = context_auc(hard, panel_rows)
    pd.DataFrame(context).to_csv(OUT/'context-auc.csv', index=False, float_format='%.17g')
    # Pre-result decision scope: pooled three panels, with no single-panel miss increase.
    target_low = hard[(hard.label == 1) & hard.panel.isin(TARGETS) & (hard.k == 1) &
                      (hard.condition == 'low') & (hard.placement == 'contiguous')].reset_index(drop=True)
    target_noise = hard[(hard.label == 0) & hard.panel.isin(TARGETS) &
                       hard.condition.isin(['noisy','low_noisy'])].drop_duplicates('cover_id').reset_index(drop=True)
    changes = paired_change(target_low, 'miss', cutoff, MODES)
    fa_changes = paired_change(target_noise, 'fa', cutoff, MODES)
    validation_changes = paired_change(validation, 'error', cutoff, MODES)
    panel_intervals = {p: paired_change(g.reset_index(drop=True), 'miss', cutoff, MODES)
                       for p,g in target_low.groupby('panel')}
    aggregates = []; decisions = []
    for mode in MODES:
        aggregates.append(dict(pooling=mode, contiguous_miss=float((target_low[mode] < cutoff[mode]).mean()),
            contiguous_pairs=len(target_low), noisy_fa=float((target_noise[mode] >= cutoff[mode]).mean()),
            distinct_noisy_tiles=len(target_noise), miss_change=changes[mode], fa_change=fa_changes[mode],
            noisy_fa_wilson=wilson_rate(int((target_noise[mode] >= cutoff[mode]).sum()), len(target_noise)),
            validation_pe_change=validation_changes[mode]))
    for mode in MODES[1:]:
        panel_changes = {p: float((g[mode] < cutoff[mode]).mean()-(g['mean'] < cutoff['mean']).mean())
                         for p,g in target_low.groupby('panel')}
        cell_changes = []
        for panel in TARGETS:
            for k in (1,2):
                for condition in ('noisy','low_noisy'):
                    f = pd.DataFrame(noise_rows)
                    f = f[(f.panel == panel) & (f.k == k) & (f.condition == condition)].set_index('pooling')
                    delta = float(f.loc[mode,'rate']-f.loc['mean','rate'])
                    if delta > .05: cell_changes.append(dict(panel=panel,k=k,condition=condition,fa_increase=delta))
        checks = dict(pooled_miss_drop_10_points=changes[mode]['change'] <= -.10+1e-12,
            no_target_panel_worse=all(v <= 1e-12 for v in panel_changes.values()),
            validation_pe_no_worse=val_rows[MODES.index(mode)]['pe'] <= val_rows[0]['pe']+1e-12,
            pooled_noisy_fa_no_worse=fa_changes[mode]['change'] <= 1e-12)
        borderline = dict(miss_interval_crosses_10_points=changes[mode]['low'] <= -.10 <= changes[mode]['high'],
            fa_interval_crosses_zero=fa_changes[mode]['low'] < 0 < fa_changes[mode]['high'],
            validation_change_interval_crosses_zero=validation_changes[mode]['low'] < 0 < validation_changes[mode]['high'])
        decisions.append(dict(pooling=mode, checks=checks, panel_miss_changes=panel_changes,
            panel_miss_change_intervals={p:v[mode] for p,v in panel_intervals.items()},
            borderline_panels=[p for p,v in panel_intervals.items() if v[mode]['low'] < 0 < v[mode]['high']],
            noisy_cell_increases_over_5_points=cell_changes, borderline=borderline))
    (OUT/'decision-evidence.json').write_text(json.dumps(dict(aggregates=aggregates, candidates=decisions,
        bootstrap_seed=SEED, bootstrap_replicates=2000,
        clean_union_rule='distinct cover_id across all target noisy/low_noisy cells, both k; first recorded score retained'), indent=2)+'\n')
    write_report(val_rows, panel_rows, low_rows, noise_rows, aggregates, decisions, tie_rows, context)


def write_report(validation, panels, low, noise, aggregates, decisions, ties, context):
    lines = ['# CHECK 2: highest-cell pooling — amended float64-margin scores', '',
        'Scores are float64 stego-minus-clean pooled-logit margins, not probabilities. '
        'First-pass numbers remain in superseded-float32/report.md, marked superseded: float32 saturation. '
        'No decision was made from that first pass.', '',
        'Each pooling uses its own validation-minimum-P_E threshold. Test thresholds were not fitted. '
        'AUC intervals: 2,000 photo-cluster bootstrap replicates, seed 20261019. '
        'Rate intervals: Wilson 95% (tiles/pairs, not independent photos).', '']
    prior = OUT/'superseded-float32/validation-metrics.csv'
    if prior.exists():
        import pandas as pd
        old = pd.read_csv(prior, float_precision='round_trip')
        old_scores = pd.read_csv(prior.parent/'validation-scores.csv', float_precision='round_trip')
        lines += ['## Superseded: float32 saturation', '',
            'No decision was made from this first pass; full tables remain in superseded-float32/report.md.', '',
            '| Pooling | First-pass validation P_E | Validation probabilities exactly 1.0 |',
            '| --- | ---: | ---: |']
        for mode in MODES:
            pe = float(old.loc[old.pooling == mode, 'pe'].iloc[0])
            ones = float((old_scores[mode] == 1).mean())
            lines.append(f'| {mode} | {pe:.2%} | {ones:.2%} |')
        lines.append('')
    for mode in MODES:
        v = next(x for x in validation if x['pooling'] == mode)
        lines += [f'## {mode}', '', f'Validation: P_E {v["pe"]:.2%}; FA {v["fa"]:.2%}; miss {v["miss"]:.2%}; cut-off {v["cutoff"]:.9g}.', '',
            '| Panel | FA | Miss | P_E | AUC [95%] |', '| --- | ---: | ---: | ---: | --- |']
        for r in panels:
            if r['pooling'] == mode:
                lines.append(f'| {r["panel"]} | {r["fa"]:.2%} | {r["miss"]:.2%} | {r["pe"]:.2%} | {r["auc"]:.3f} [{r["auc_low"]:.3f}, {r["auc_high"]:.3f}] |')
        lines += ['', '| Suite | Panel | Placement | Pairs | k=1 low-bpp miss [Wilson 95%] |', '| --- | --- | --- | ---: | --- |']
        for r in low:
            if r['pooling'] == mode:
                lines.append(f'| {r["suite"]} | {r["panel"]} | {r["placement"]} | {r["n"]} | {r["rate"]:.1%} [{r["low"]:.1%}, {r["high"]:.1%}] |')
        lines += ['', '| Panel | k | Condition | Distinct clean tiles | FA [Wilson 95%] |', '| --- | ---: | --- | ---: | --- |']
        for r in noise:
            if r['pooling'] == mode:
                lines.append(f'| {r["panel"]} | {r["k"]} | {r["condition"]} | {r["n"]} | {r["rate"]:.1%} [{r["low"]:.1%}, {r["high"]:.1%}] |')
        lines.append('')
    lines += ['## Threshold-free AUC context (not decision criteria)', '',
        'Target low cells pool ALASKA seen, ALASKA unseen and LSDIR, k=1, both embedding methods. '
        'All clean/stego paired observations are retained; source-photo clusters are bootstrapped jointly.', '',
        '| Context | Pooling | Pairs | Photos | AUC [photo-bootstrap 95%] |',
        '| --- | --- | ---: | ---: | --- |']
    for r in context:
        lines.append(f'| {r["context"]} | {r["pooling"]} | {r["pairs"]} | {r["photos"]} | {r["auc"]:.3f} [{r["auc_low"]:.3f}, {r["auc_high"]:.3f}] |')
    lines += ['', '## Exact margin ties', '',
        'Exact float64 ties, including repeated observations of the same clean tile. '
        'Duplicate excess = images minus unique margin values. No softmax is used for scoring.', '',
        '| Dataset | Pooling | Images | Unique margins | Tied groups | Images in tied groups | Duplicate excess | Cross-label tied groups |',
        '| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    for r in ties:
        lines.append('| '+' | '.join(str(r[k]) for k in ('dataset','pooling','images','unique_margins','tied_groups','images_in_tied_groups','duplicate_excess','clean_stego_tied_groups'))+' |')
    lines += ['', '## Pre-registered decision evidence', '', 'No board changes are made by this script.', '',
              '```json', json.dumps(dict(aggregates=aggregates,candidates=decisions),indent=2), '```']
    (OUT/'report.md').write_text('\n'.join(lines)+'\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=['infer','summarize'], required=True)
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    if not 0 <= args.workers <= 12: parser.error('workers must be 0 through 12')
    sys.path.insert(0,str(ROOT))
    os.environ.update(PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
    if args.phase == 'summarize': os.environ.update(CUDA_VISIBLE_DEVICES='',HIP_VISIBLE_DEVICES='')
    from scripts.archive.check1_classic import snapshot
    OUT.mkdir(parents=True, exist_ok=True)
    before = snapshot(); (OUT/f'{args.phase}-inputs-before.json').write_text(json.dumps(before,indent=2)+'\n')
    started = time.perf_counter()
    try:
        if args.phase == 'infer': infer(args.workers)
        else: summarize()
    finally:
        after = snapshot(); (OUT/f'{args.phase}-inputs-after.json').write_text(json.dumps(after,indent=2)+'\n')
        (OUT/f'{args.phase}-timing.json').write_text(json.dumps(dict(seconds=time.perf_counter()-started),indent=2)+'\n')
        if before != after: raise RuntimeError('protected inputs changed')


if __name__ == '__main__': main()
