"""Two exact source scopes, not an extension of the original run21 approval.

Historical commands remain pinned-only. The cell-aux version checks trusted Git
lineage AND the live executing checkout; old blobs alone never approve new code.
"""
import ast
import hashlib
import json
from pathlib import Path
import subprocess

ORIGINAL_REVISION='9425d07cbf474ae20a8f9bf0bddb6eb14b9c23cc'
ORIGINAL_MODEL_SHA='09d3e41599a7cd28c167dacf95a67ce91ec2bca95b1ecfb5e73a491289a432bb'
MATH_SHA='31cf8ecb4eb7f36c12b789f669f7c13f23ff51f0fdb8be46d6e33ecdaeb6ea28'
ORIGINAL_NOTEBOOK='notebooks/experiments/srm-e50-paired-gapcov32.ipynb'
ORIGINAL_NOTEBOOK_SHA='bb67c346c1886a94af2e8dee43c953747a25fc1c0ce8d772182c56253e825fcd'
CELL_NOTEBOOK='notebooks/experiments/srm-cell-aux-paired-gapcov32.ipynb'
CELL_MODEL_SHA='403023c538d217f243b7f5f6c1de8f567928e7d2ab572ac105429315acd63c5d'
CELL_NOTEBOOK_SHA='7200edda277584b2a734a40640eb445b209d4233a456428e59281e9c5f180429'
HISTORICAL='run21-original-v1'
CELL_AUX='cell-aux-v2'


def digest(content):return hashlib.sha256(content).hexdigest()


def historical_sources(root, *, revision=ORIGINAL_REVISION):
    """Check the full launch revision and actual Git blob bytes, without eval."""
    if revision!=ORIGINAL_REVISION:raise ValueError('gapcov historical revision mismatch')
    hashes={'cvdl/models.py':ORIGINAL_MODEL_SHA,'cvdl/cov_pool.py':MATH_SHA,
            ORIGINAL_NOTEBOOK:ORIGINAL_NOTEBOOK_SHA}
    blobs={}
    for path,expected in hashes.items():
        try:blob=subprocess.check_output(['git','show',revision+':'+path],cwd=root,stderr=subprocess.PIPE)
        except (OSError,subprocess.CalledProcessError) as error:
            raise ValueError('gapcov trusted historical source unavailable: '+path) from error
        if digest(blob)!=expected:raise ValueError('gapcov historical source bytes mismatch: '+path)
        blobs[path]=blob
    # SPEC is a dict of literals. Parse it, never execute a historical module.
    node=next(n.value for n in ast.parse(blobs['cvdl/cov_pool.py']).body
              if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='SPEC' for t in n.targets))
    spec={k.arg:ast.literal_eval(k.value) for k in node.keywords}
    return dict(revision=revision,files_sha256=hashes,spec=spec)


def validate_sources(root,version, *, revision=ORIGINAL_REVISION):
    """Fail closed on unknown version, wrong lineage, live bytes or live SPEC.

    V1 is for the pinned historical checkout only. V2 must be this executing
    checkout, not a directory of historical files supplied as a substitute.
    """
    root=Path(root).resolve()
    if version not in (HISTORICAL,CELL_AUX):raise ValueError('unknown gapcov source version')
    lineage=historical_sources(root,revision=revision)
    current=dict(lineage['files_sha256'])
    if version==CELL_AUX:
        from . import models,cov_pool
        if (root!=Path(__file__).resolve().parents[1] or
                Path(models.__file__).resolve()!=root/'cvdl/models.py' or
                Path(cov_pool.__file__).resolve()!=root/'cvdl/cov_pool.py'):
            raise ValueError('gapcov cell-aux provenance must check the executing checkout')
        current['cvdl/models.py']=CELL_MODEL_SHA
        current[CELL_NOTEBOOK]=CELL_NOTEBOOK_SHA
        if cov_pool.SPEC!=lineage['spec']:raise ValueError('gapcov live SPEC drift')
    for path,expected in current.items():
        try:actual=digest((root/path).read_bytes())
        except OSError as error:raise ValueError('gapcov current source unavailable: '+path) from error
        if actual!=expected:raise ValueError('gapcov current source drift: '+path)
    return dict(version=version,historical=lineage,current_files_sha256=current,
                spec_sha256=digest(json.dumps(lineage['spec'],sort_keys=True,separators=(',',':')).encode()))
