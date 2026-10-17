"""Run the packed-loader speed matrix serially: pack 1 vs pack 16, 256 and 512 recipes, real and synthetic.

CPU only. One loader_profile case at a time, each under `timeout`, each its own process, so a failure cannot
leave workers behind. Prints one RESULT line per case plus a final summary table.
Cases are listed first with --list; nothing runs without --run. Needs a CPU lease on the shared PC.
"""
from __future__ import annotations
import argparse
import subprocess
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
RECIPES=((256,128),(512,32))
PACKS=(1,16)


def cases(workers_list, steps_real, steps_synthetic):
    for workers in workers_list:
        for synthetic in (True,False):
            for tile,batch in RECIPES:
                for pack in PACKS:
                    yield dict(tile=tile,batch=batch,workers=workers,pack=pack,synthetic=synthetic,
                               steps=steps_synthetic if synthetic else steps_real)


def command(case, timeout):
    args=['timeout','-k','5',str(timeout),sys.executable,'scripts/profile_loader.py','--tile',str(case['tile']),
          '--batch',str(case['batch']),'--workers',str(case['workers']),'--pack-blocks',str(case['pack']),
          '--steps',str(case['steps']),'--warmup','30']
    if case['synthetic']:args.append('--synthetic')
    return args


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--workers',default='6,12')
    p.add_argument('--steps-real',type=int,default=150)
    p.add_argument('--steps-synthetic',type=int,default=400)
    p.add_argument('--timeout',type=int,default=100,help='seconds per case')
    p.add_argument('--list',action='store_true')
    p.add_argument('--run',action='store_true')
    a=p.parse_args()
    from scripts.pc_safety import check_workers
    workers=[int(x) for x in a.workers.split(',')]
    plan=list(cases(workers,a.steps_real,a.steps_synthetic))
    if a.list or not a.run:
        for i,case in enumerate(plan,1):print(i,case)
        print(f'{len(plan)} cases; worst case {len(plan)*a.timeout} s if every case hits its timeout')
        return
    for w in workers:check_workers(w)
    rows=[]
    for i,case in enumerate(plan,1):
        check_workers(case['workers'])
        done=subprocess.run(command(case,a.timeout),cwd=ROOT,capture_output=True,text=True)
        line=next((l for l in done.stdout.splitlines() if l.startswith('RESULT')),None)
        print(f'[{i}/{len(plan)}] exit={done.returncode} '+(line or 'NO RESULT: '+(done.stderr.strip().splitlines() or ['?'])[-1][:200]),flush=True)
        rows.append((case,line))
    print('MATRIX DONE')


if __name__=='__main__':main()
