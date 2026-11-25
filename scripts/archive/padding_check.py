"""Run only the padding check on a finished, already evaluated run (PC).

Reads the run's best checkpoint, test_predictions.csv and validation_predictions.csv.
Writes <output>/padding-check/; never writes into the run folder unless --output is that folder.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(ROOT/'scripts'))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run_folder',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--padding-pairs',type=int,default=2000,
        help='cases per (pattern, level, real shape), chosen by a fixed hash from the standard test list; 0 = all')
    parser.add_argument('--workers',type=int,default=None,help='data workers (default: PC profile eval workers)')
    args=parser.parse_args()
    import pandas as pd
    from evaluate_run import checkpoint_folder, settings_for_run
    from cvdl import runtime
    from cvdl.lit import SteganalysisModule, select_best_checkpoint
    from cvdl.archive.padding import check
    folder=checkpoint_folder(args.run_folder);settings=settings_for_run(folder)
    checkpoint=Path(select_best_checkpoint('',folder))
    runtime.configure(settings)
    otf,_=runtime.create_data(settings)
    if otf is None:raise ValueError('padding diagnostic requires OTF data')
    model=SteganalysisModule.load_from_checkpoint(checkpoint,map_location=settings.device,weights_only=False)
    read=lambda name:pd.read_csv(folder/name,dtype={'cover_id':'string'})
    args.output.mkdir(parents=True,exist_ok=True)
    start=time.perf_counter()
    result=check(model.net,otf.test,read('test_predictions.csv'),read('validation_predictions.csv'),args.output,
                 device=settings.device,quick=settings.quick,limit=args.padding_pairs,
                 workers=settings.eval_workers if args.workers is None else args.workers)
    print(f'PADDING CHECK COMPLETE: {result["cases"]} cases in {time.perf_counter()-start:.1f} s; '
          f'pairs_sha256={result["pairs_sha256"]} padding_pairs_sha256={result["padding_pairs_sha256"]}',flush=True)


if __name__=='__main__':main()
