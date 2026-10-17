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
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cvdl.metrics import threshold, rates
import numpy as np
import pandas as pd

SOURCES = ('raise', 'div2k', 'alaska', 'lsdir')


def keyed(spec: str) -> tuple[str, str]:
    name, sep, value = spec.partition('=')
    if not sep or not name or not value:
        raise ValueError(f'expected NAME=VALUE: {spec}')
    return name, value


def build(model: str, validation: dict, tests: dict, output: Path, *, quiet: bool = False):
    """Validation-only cutoffs applied to untouched test panels; writes the threshold CSV.

    validation: {source: predictions.csv} for every source; tests: {panel: (source, predictions.csv)}.
    """
    val = {}
    for source, path in validation.items():
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
    rows = []
    for panel, (source, path) in tests.items():
        if source not in SOURCES:
            raise ValueError(f'expected a known source for panel {panel}: {source}')
        test = pd.read_csv(path)
        test = test.loc[test.source == source].copy()
        if test.empty:
            raise ValueError(f'no {source} rows in {path}')
        for cal in ('source', 'shared'):
            for kind in ('fa5', 'pe'):
                cutoff, val_fa = selected[source if cal == 'source' else 'shared', kind]
                rows.append(dict(model=model, panel=panel, source=source,
                                 calibration=cal, rule=kind, threshold=cutoff,
                                 val_fa=val_fa, fa_drift=rates(test, cutoff)['fa'] - (.05 if kind == 'fa5' else val_fa),
                                 **rates(test, cutoff)))
        rows.append(dict(model=model, panel=panel, source=source,
                         calibration='fixed', rule='0.5', threshold=.5, val_fa=np.nan,
                         fa_drift=np.nan, **rates(test, .5)))
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output, index=False)
    if not quiet:
        print(f'wrote {len(rows)} test rows to {output}')
        print('Validation thresholds (each selected without test labels):')
        for (source, kind), (cutoff, fa) in selected.items():
            print(f'{source:7} {kind:3} cutoff={cutoff:.9g} val_FA={fa:.5%} n={len(all_val if source == "shared" else val[source])//2}')
    return selected, {s: len(v) // 2 for s, v in val.items()}


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
    raw = dict(map(keyed, args.test))
    if len(raw) != len(args.test):
        ap.error('duplicate test panel')
    tests = {}
    for panel, src_path in raw.items():
        source, sep, path = src_path.partition(':')
        if not sep or source not in SOURCES:
            raise ValueError(f'expected PANEL=SOURCE:path/to/predictions.csv: {panel}')
        tests[panel] = (source, path)
    build(args.model, paths, tests, args.output)


if __name__ == '__main__':
    main()
