"""Build the frozen standard 256 tests without decoding photos. Runtime never builds.

python -B scripts/build_standard_test.py [--check]
"""
from __future__ import annotations
import argparse
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from cvdl import standardtest as S


def publish(files, output, *, check=False):
    output = Path(output)
    if check:
        for name, text in files.items():
            if not (output/name).is_file() or (output/name).read_bytes() != text.encode():
                raise ValueError(f'REBUILD MISMATCH: {name}')
        print('REBUILD IDENTICAL: 8 pair lists, recipe and summary; SHA-256 equal')
    else:
        if output.exists():
            raise ValueError(f'refusing to overwrite a frozen suite: {output}')
        output.mkdir(parents=True)
        for name, text in files.items():
            (output/name).write_text(text)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--check', action='store_true')
    p.add_argument('--output', type=Path, default=ROOT/S.DATA_DIR)
    args = p.parse_args()
    before = time.perf_counter()
    files, summary = S.build(ROOT)
    publish(files, args.output, check=args.check)
    print(f"{S.SUITE}: recipe {summary['recipe_id']}; {time.perf_counter()-before:.3f} s")
    for name, sha in summary['files_sha256'].items():
        print(f'{name}: {summary["panels"][name]["pairs"]} pairs; {sha}')


if __name__ == '__main__':
    main()
