"""Generate and CPU-execute runs/<run>/report.ipynb from the shared template."""
import argparse
import json
import os
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def generate(root,run,evaluation_id=None,*,reference_run=None,reference_evaluation_id=None,overwrite=False):
    from cvdl.reporting import SavedReport,compatible
    import nbformat
    from nbclient import NotebookClient
    root=Path(root).resolve();primary=SavedReport(root,run,evaluation_id)
    if reference_evaluation_id and not reference_run:raise ValueError('--reference-evaluation-id requires --reference-run')
    reference=SavedReport(root,reference_run,reference_evaluation_id) if reference_run else None
    if reference:compatible(primary,reference)
    target=primary.run/'report.ipynb'
    if target.exists() and not overwrite:raise ValueError(f'refusing to replace author notes: {target}; use --overwrite explicitly')
    nb=nbformat.read(root/'notebooks/templates/report.ipynb',as_version=4)
    params=dict(ROOT_HINT=str(root),RUN=primary.run.name,EVALUATION_ID=primary.path.name,
                REFERENCE_RUN=reference.run.name if reference else None,REFERENCE_EVALUATION_ID=reference.path.name if reference else None)
    cells=[c for c in nb.cells if 'parameters' in c.metadata.get('tags',[])]
    if len(cells)!=1:raise ValueError('template must have one parameters cell')
    cells[0].source='\n'.join(f'{k} = {v!r}' for k,v in params.items())
    nb.metadata['report']=dict(parameters=params,template='notebooks/templates/report.ipynb',saved_outputs_only=True)
    env={**os.environ,'CUDA_VISIBLE_DEVICES':'','HIP_VISIBLE_DEVICES':'','PYTHONDONTWRITEBYTECODE':'1',
         'MPLBACKEND':'Agg','OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1'}
    NotebookClient(nb,timeout=180,kernel_name='python3',resources={'metadata':{'path':str(root)}}).execute(env=env)
    temporary=target.with_suffix('.ipynb.tmp')
    try:
        nbformat.write(nb,temporary);temporary.replace(target)
    finally:
        temporary.unlink(missing_ok=True)
    print(json.dumps(dict(report=str(target),evaluation_id=primary.path.name,reference=reference.path.name if reference else None)))
    return target


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('run')
    parser.add_argument('--evaluation-id');parser.add_argument('--reference-run');parser.add_argument('--reference-evaluation-id')
    parser.add_argument('--overwrite',action='store_true');args=parser.parse_args()
    generate(ROOT,args.run,args.evaluation_id,reference_run=args.reference_run,
             reference_evaluation_id=args.reference_evaluation_id,overwrite=args.overwrite)

if __name__=='__main__':main()
