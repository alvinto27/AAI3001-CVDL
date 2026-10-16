"""Shared metrics; preserve each existing API and its numerical operation order."""
import math
import numpy as np
import pandas as pd

def _threshold_counts(labels_array: np.ndarray, scores_array: np.ndarray) -> tuple[np.ndarray, np.ndarray, int, int]:
    """Count positives and negatives at tied score thresholds."""
    labels_array = np.asarray(labels_array, dtype=np.int64)
    scores_array = np.asarray(scores_array, dtype=np.float64)
    if labels_array.size == 0:
        return np.array([], dtype=np.int64), np.array([], dtype=np.int64), 0, 0
    order = np.argsort(-scores_array, kind="mergesort")
    sorted_scores = scores_array[order]
    sorted_labels = labels_array[order]
    group_ends = np.flatnonzero(sorted_scores[1:] != sorted_scores[:-1]) + 1
    group_ends = np.r_[group_ends, len(sorted_scores)]
    cumulative_positive = np.cumsum(sorted_labels == 1, dtype=np.int64)
    cumulative_negative = np.cumsum(sorted_labels == 0, dtype=np.int64)
    return (
        cumulative_positive[group_ends - 1],
        cumulative_negative[group_ends - 1],
        int((labels_array == 1).sum()),
        int((labels_array == 0).sum()),
    )


def auc_numpy(labels_array: np.ndarray, scores_array: np.ndarray) -> float:
    """Compute ROC AUC with one sort and tied-score groups."""
    true_positive, false_positive, positives, negatives = _threshold_counts(labels_array, scores_array)
    if positives == 0 or negatives == 0:
        return 0.0
    denominator_positive = float(positives)
    denominator_negative = float(negatives)
    true_positive_rate = np.r_[0.0, true_positive / denominator_positive, true_positive[-1] / denominator_positive]
    false_positive_rate = np.r_[0.0, false_positive / denominator_negative, false_positive[-1] / denominator_negative]
    return float(np.trapezoid(true_positive_rate, false_positive_rate))


def probability_error(labels_array: np.ndarray, scores_array: np.ndarray) -> float:
    """Compute minimum P_E with one sort and tied-score groups."""
    true_positive, false_positive, positives, negatives = _threshold_counts(labels_array, scores_array)
    if positives == 0 and negatives == 0:
        return 0.0
    if negatives:
        false_alarm = np.r_[0.0, false_positive / negatives, false_positive[-1] / negatives]
    else:
        false_alarm = np.zeros(len(true_positive) + 2, dtype=np.float64)
    if positives:
        missed_detection = np.r_[1.0, (positives - true_positive) / positives, 0.0]
    else:
        missed_detection = np.zeros(len(true_positive) + 2, dtype=np.float64)
    return float(np.min((false_alarm + missed_detection) / 2.0))


def reference_auc(labels_array: np.ndarray, scores_array: np.ndarray) -> float:
    thresholds = np.r_[np.inf, np.sort(np.unique(scores_array))[::-1], -np.inf]
    positives = max(int((labels_array == 1).sum()), 1)
    negatives = max(int((labels_array == 0).sum()), 1)
    tpr = []
    fpr = []
    for threshold in thresholds:
        predicted = scores_array >= threshold
        tpr.append(float(((predicted == 1) & (labels_array == 1)).sum()) / positives)
        fpr.append(float(((predicted == 1) & (labels_array == 0)).sum()) / negatives)
    return float(np.trapezoid(np.asarray(tpr), np.asarray(fpr)))


def reference_probability_error(labels_array: np.ndarray, scores_array: np.ndarray) -> float:
    positives = labels_array == 1
    negatives = labels_array == 0
    thresholds = np.r_[np.inf, np.sort(np.unique(scores_array))[::-1], -np.inf]
    errors = []
    for threshold in thresholds:
        predicted = scores_array >= threshold
        p_fa = float(predicted[negatives].mean()) if negatives.any() else 0.0
        p_md = float((~predicted[positives]).mean()) if positives.any() else 0.0
        errors.append((p_fa + p_md) / 2.0)
    return float(min(errors))


counts = _threshold_counts
threshold_counts = _threshold_counts

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


def roc_auc(scores, positive):
    a = scores.ravel().astype(np.float64)
    b = positive.ravel()
    n1 = int(b.sum())
    n0 = len(b) - n1
    if not n0 or not n1:
        return None
    order = np.argsort(a, kind='mergesort')
    sorted_scores = a[order]
    ranks = np.empty(len(a), dtype=np.float64)
    ends = np.r_[0, np.flatnonzero(np.diff(sorted_scores)) + 1, len(a)]
    for lo, hi in zip(ends[:-1], ends[1:]):
        ranks[order[lo:hi]] = (lo + 1 + hi) / 2
    return float((ranks[b].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def sigmoid(scores):
    """Stable stego probability from a stego-minus-clean logit margin."""
    values = np.asarray(scores, dtype=np.float64)
    small = np.exp(-np.abs(values))
    return np.where(values >= 0, 1 / (1 + small), small / (1 + small))


def threshold(frame: pd.DataFrame, kind: str, target: float = .05) -> tuple[float, float]:
    """Return (validation-derived cutoff, validation FA), with score >= cutoff positive.

    For a 5% target, choose the highest empirical FA not exceeding target.
    Tied scores move together; a zero-FA cutoff above the max is allowed.
    At equal minimum P_E, prefer the more conservative threshold.
    """
    y = frame.label.to_numpy(dtype=np.int8)
    s = frame.prob_stego.to_numpy(dtype=np.float64)
    if len(y) == 0 or set(np.unique(y)) != {0, 1} or not np.isfinite(s).all():
        raise ValueError('expected finite scores with both labels')
    values, inv = np.unique(s, return_inverse=True)
    clean = np.bincount(inv[y == 0], minlength=len(values))
    stego = np.bincount(inv[y == 1], minlength=len(values))
    # For each score value, positive means >= that value; the final cutoff
    # is immediately above the largest score and predicts no positives.
    fa = np.r_[np.cumsum(clean[::-1])[::-1] / clean.sum(), 0.]
    miss = np.r_[np.cumsum(stego) - stego, stego.sum()] / stego.sum()
    cutoffs = np.r_[values, np.nextafter(values[-1], np.inf)]
    if kind == 'fa5':
        candidates = np.flatnonzero(fa <= target + 1e-12)
        best = candidates[np.argmax(fa[candidates])]
    elif kind == 'pe':
        error = (fa + miss) / 2
        candidates = np.flatnonzero(np.isclose(error, error.min(), rtol=0, atol=1e-12))
        best = candidates[np.argmin(fa[candidates])]
    else:
        raise ValueError(f'unknown threshold kind {kind}')
    return float(cutoffs[best]), float(fa[best])


def rates(frame: pd.DataFrame, cutoff: float) -> dict:
    clean = frame.loc[frame.label == 0, 'prob_stego']
    stego = frame.loc[frame.label == 1, 'prob_stego']
    k1 = frame.loc[(frame.label == 1) & (frame.k == 1), 'prob_stego']
    if not len(clean) or not len(stego):
        raise ValueError('test requires clean and stego rows')
    fa = float((clean >= cutoff).mean())
    miss = float((stego < cutoff).mean())
    return dict(pairs=len(stego), k1_pairs=len(k1), fa=fa, miss=miss,
                k1_miss=float((k1 < cutoff).mean()) if len(k1) else None,
                pe_at_cutoff=(fa + miss) / 2)


