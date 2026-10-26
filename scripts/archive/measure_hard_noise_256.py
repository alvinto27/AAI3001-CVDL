"""CPU-only 256 hard-test KV noise: timing pilot, full scan, or frozen sample check."""
import os
# Set before torch/rawpy import, including spawn children.
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['HIP_VISIBLE_DEVICES'] = ''
for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'): os.environ[key] = '1'
import argparse
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from cvdl import hardnoise as N
from scripts.pc_safety import own_process_group, guarded


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--workers', type=int, default=8)
    p.add_argument('--pilot-per-source', type=int, default=0)
    p.add_argument('--sample-check', action='store_true')
    p.add_argument('--output', type=Path, default=ROOT/N.DATA_DIR)
    a = p.parse_args()
    if a.sample_check:
        N.sample_check(ROOT, a.output)
        return
    if not 1 <= a.workers <= 12 or a.pilot_per_source < 0:
        raise ValueError('workers must be1..12; pilot count must be nonnegative')
    if not a.pilot_per_source and any((a.output/n).exists() for n in ('noise-val.csv','noise-test.csv','noise-scan.json')):
        raise ValueError('refusing to overwrite frozen noise')
    rows, report = N.scan(ROOT, workers=a.workers, pilot_per_source=a.pilot_per_source)
    if not a.pilot_per_source: N.write_scan(ROOT, rows, report, a.output)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    own_process_group()
    guarded(main)
