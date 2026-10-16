#!/usr/bin/env python3
"""Select score cutoffs on validation only; score untouched test panels.

Example: --validation raise=standard-val.csv --validation div2k=standard-val.csv
         --validation alaska=alaska-val.csv --validation lsdir=lsdir-val.csv
         --test raise=raise:standard-test.csv --test alaska_seen=alaska:seen.csv

`P_E@cutoff` means (false-alarm rate + miss rate)/2 at this fixed cutoff,
not a test-optimized threshold. All rows contain both clean and stego labels.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd

SOURCES = ('raise', 'div2k', 'alaska', 'lsdir')


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


def keyed(spec: str) -> tuple[str, str]:
    name, sep, value = spec.partition('=')
    if not sep or not name or not value:
        raise ValueError(f'expected NAME=VALUE: {spec}')
    return name, value


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--model', required=True, help='a unique result label')
    ap.add_argument('--validation', action='append', required=True,
                    help='SOURCE=predictions.csv (filter by row source)')
    ap.add_argument('--test', action='append', required=True,
                    help='PANEL=SOURCE:path/to/predictions.csv')
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    paths = dict(map(keyed, args.validation))
    if set(paths) != set(SOURCES) or len(paths) != len(args.validation):
        ap.error(f'validation must include each source exactly once: {SOURCES}')
    val = {}
    for source, path in paths.items():
        frame = pd.read_csv(path)
        val[source] = frame.loc[frame.source == source].copy()
        if val[source].empty:
            raise ValueError(f'no {source} rows in {path}')
    # The common cutoff uses the actual pooled validation-pair frequency;
    # the old standard val has both RAISE and DIV2K. No test rows enter it.
    all_val = pd.concat(list(val.values()), ignore_index=True)
    selected = {}
    for kind in ('fa5', 'pe'):
        for source in (*SOURCES, 'shared'):
            frame = all_val if source == 'shared' else val[source]
            selected[source, kind] = threshold(frame, kind)
    tests = dict(map(keyed, args.test))
    if len(tests) != len(args.test):
        ap.error('duplicate test panel')
    rows = []
    for panel, src_path in tests.items():
        source, sep, path = src_path.partition(':')
        if not sep or source not in SOURCES:
            raise ValueError(f'expected PANEL=SOURCE:path/to/predictions.csv: {panel}')
        test = pd.read_csv(path)
        test = test.loc[test.source == source].copy()
        if test.empty:
            raise ValueError(f'no {source} rows in {path}')
        for cal in ('source', 'shared'):
            for kind in ('fa5', 'pe'):
                cutoff, val_fa = selected[source if cal == 'source' else 'shared', kind]
                rows.append(dict(model=args.model, panel=panel, source=source,
                                 calibration=cal, rule=kind, threshold=cutoff,
                                 val_fa=val_fa, fa_drift=rates(test, cutoff)['fa'] - (.05 if kind == 'fa5' else val_fa),
                                 **rates(test, cutoff)))
        rows.append(dict(model=args.model, panel=panel, source=source,
                         calibration='fixed', rule='0.5', threshold=.5, val_fa=np.nan,
                         fa_drift=np.nan, **rates(test, .5)))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.output, index=False)
    print(f'wrote {len(rows)} test rows to {args.output}')
    print('Validation thresholds (each selected without test labels):')
    for (source, kind), (cutoff, fa) in selected.items():
        print(f'{source:7} {kind:3} cutoff={cutoff:.9g} val_FA={fa:.5%} n={len(all_val if source == "shared" else val[source])//2}')


if __name__ == '__main__':
    main()
