"""Validation outputs and per-source cutoff tables that the main evaluation stage writes.

Layout (names and numbers match the PC wrapper scripts used for the tiling arms):
  <checkpoint folder>/validation_predictions.csv     capped checkpoint-selection validation, per pair
  <checkpoint folder>/validation_calibration.json    shared minimum-P_E cutoff on that capped set
  <checkpoint folder>/extra-standard-val/            full standard validation (RAISE + DIV2K), same files as a panel
  <out>/pc-comparison/<label>.json                   panels at the capped cutoff + per-source validation FA5 cutoffs
  <out>/pc-comparison/<label>-thresholds.csv         check-B table (scripts/check_b_thresholds.py)
"""
from __future__ import annotations
import hashlib
import json
import subprocess
from pathlib import Path
import pandas as pd

from .metrics import metrics, rates, threshold

PANEL_DIRS={'standard':'.','bpp1':'extra-test-bpp1','alaska_val':'extra-alaska-val','alaska_seen':'extra-alaska-test_seen',
            'alaska_unseen':'extra-alaska-test_unseen','lsdir_val':'extra-lsdir-val','lsdir':'extra-lsdir-test'}
STANDARD_VAL='extra-standard-val'
# Test panel -> (source subset, panel). Validation sources and which file holds their validation pairs.
TEST_PANELS={'raise':('raise','standard'),'div2k':('div2k','standard'),'alaska_seen':('alaska','alaska_seen'),
             'alaska_unseen':('alaska','alaska_unseen'),'lsdir':('lsdir','lsdir'),'bpp1_raise':('raise','bpp1'),
             'bpp1_div2k':('div2k','bpp1')}
VALIDATION_SOURCES={'raise':STANDARD_VAL,'div2k':STANDARD_VAL,'alaska':'alaska_val','lsdir':'lsdir_val'}
EXPECTED_VALIDATION_PAIRS={'raise':9782,'div2k':1456,'alaska':1535,'lsdir':3164}


def code_commit(root):
    try:
        return subprocess.check_output(['git','rev-parse','--short','HEAD'],cwd=root,text=True,stderr=subprocess.DEVNULL).strip()
    except (OSError,subprocess.CalledProcessError):
        return 'unknown'


def write_validation(folder, val_predictions, *, root):
    """validation_predictions.csv and validation_calibration.json from the main stage's capped-validation predictions."""
    folder=Path(folder)
    val_predictions.to_csv(folder/'validation_predictions.csv',index=False)
    cutoff,fa=threshold(val_predictions,'pe')
    (folder/'validation_calibration.json').write_text(json.dumps(dict(criterion='minimum P_E; conservative tie',cutoff=cutoff,
        validation_fa=fa,pairs=len(val_predictions)//2,code_commit=code_commit(root)),indent=2)+'\n')


def write_standard_validation(checkpoint, folder, settings):
    """Score the full standard validation (RAISE + DIV2K) with the frozen evaluation settings."""
    from . import evaluation
    return evaluation.evaluate(checkpoint,Path(settings.root)/'configs/raise512-run1.yaml',root=settings.root,device=settings.device,
        split='val',output=Path(folder)/STANDARD_VAL,workers=settings.eval_workers,quick=settings.quick,overwrite=True)


def panel_files(folder):
    folder=Path(folder)
    files={name:folder/directory/'test_predictions.csv' for name,directory in PANEL_DIRS.items()}
    files['standard_val']=folder/STANDARD_VAL/'test_predictions.csv'
    return files


def missing_panels(folder):
    return [str(p) for p in panel_files(folder).values() if not p.exists()]+(
        [] if (Path(folder)/'validation_calibration.json').exists() else [str(Path(folder)/'validation_calibration.json')])


def source_cutoffs(folder):
    """Per-source validation FA5 cut-off: the same values the standard tables apply to the tests."""
    folder=Path(folder)
    gone=missing_panels(folder)
    if gone:raise ValueError('cut-offs need these saved files: '+', '.join(gone))
    files=panel_files(folder)
    result={}
    for source,where in VALIDATION_SOURCES.items():
        path=files['standard_val'] if where==STANDARD_VAL else files[where]
        frame=pd.read_csv(path).query('source==@source')
        result[source]=float(threshold(frame,'fa5')[0])
    return result


def tables(folder, out, label, *, check=True):
    """Per-source validation cutoffs applied to untouched test panels; no test row enters a cutoff."""
    from scripts.check_b_thresholds import build
    from .lit import select_best_checkpoint
    folder=Path(folder);out=Path(out)/'pc-comparison';out.mkdir(parents=True,exist_ok=True)
    gone=missing_panels(folder)
    if gone:raise ValueError('tables need these saved files: '+', '.join(gone))
    files=panel_files(folder)
    data={name:pd.read_csv(files[name]) for name in PANEL_DIRS}
    stdval=files['standard_val']
    val_files={s:(stdval if where==STANDARD_VAL else files[where]) for s,where in VALIDATION_SOURCES.items()}
    sources={s:pd.read_csv(f).query('source==@s') for s,f in val_files.items()}
    counts={s:len(v)//2 for s,v in sources.items()}
    if check and counts!=EXPECTED_VALIDATION_PAIRS:
        raise ValueError(f'unexpected standard validation pair counts {counts}')
    capped=json.loads((folder/'validation_calibration.json').read_text())
    panels={n:{**metrics(f),'at_capped_val_min_pe':rates(f,capped['cutoff'])} for n,f in data.items()}
    tests={n:(s,files[where]) for n,(s,where) in TEST_PANELS.items()}
    build(label,val_files,tests,out/f'{label}-thresholds.csv',quiet=True)
    fa5=[]
    for n,(s,path) in tests.items():
        frame=pd.read_csv(path).query('source==@s');cutoff,valfa=threshold(sources[s],'fa5')
        fa5.append(dict(panel=n,source=s,cutoff=cutoff,validation_fa=valfa,**rates(frame,cutoff)))
    checkpoint=Path(select_best_checkpoint('',folder))
    result=dict(checkpoint=checkpoint.name,checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                validation_pairs=counts,capped_validation=capped,panels=panels,source_fa5=fa5)
    (out/f'{label}.json').write_text(json.dumps(result,indent=2)+'\n')
    return result
