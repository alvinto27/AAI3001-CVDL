"""Checkpoint-bound saved cell scores; no inference or threshold fitting."""
import hashlib
import json
from pathlib import Path

FILES=('val-pairs.csv','test-pairs.csv','val-scores.npz','test-scores.npz','summary.json','manifest.json')


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def checkpoint_identity(path):
    import torch
    from .pilot import refuse_run
    path=Path(path);refuse_run(path.parent)
    saved=torch.load(path,map_location='cpu',weights_only=False)
    h=saved.get('hyper_parameters',{})
    if h.get('pilot') or h.get('quick'):raise ValueError('quick/pilot cell checkpoint refused')
    return dict(checkpoint_sha256=sha(path),model=h.get('model'),cell_supervision=h.get('cell_supervision','off'),
                source_provenance=h.get('source_provenance'))


def bind_scores(folder,identity,manifest):
    """Call after writing summary and arrays. Sidecar hashes include summary."""
    folder=Path(folder)
    (folder/'manifest.json').write_text(json.dumps(manifest,sort_keys=True,indent=2)+'\n')
    summary=json.loads((folder/'summary.json').read_text())
    summary['identity']=dict(identity,manifest_sha256=sha(folder/'manifest.json'),quick=False,pilot=False)
    (folder/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
    binding=dict(version='cell-scores-v2',identity=summary['identity'],files_sha256={name:sha(folder/name) for name in FILES})
    (folder/'artifact_identity.json').write_text(json.dumps(binding,indent=2)+'\n')
    return binding


def validate_scores(folder,*,checkpoint=None,expected_sha=None,require_v2=False):
    folder=Path(folder)
    binding=json.loads((folder/'artifact_identity.json').read_text())
    if binding.get('version')!='cell-scores-v2' or set(binding.get('files_sha256',{}))!=set(FILES):
        raise ValueError('cell artifact binding schema mismatch')
    if any(sha(folder/name)!=value for name,value in binding['files_sha256'].items()):
        raise ValueError('cell artifact bytes changed')
    summary=json.loads((folder/'summary.json').read_text());identity=summary.get('identity',{})
    if identity!=binding.get('identity') or identity.get('quick') is not False or identity.get('pilot') is not False:
        raise ValueError('cell artifact identity/quick/pilot mismatch')
    if identity.get('manifest_sha256')!=sha(folder/'manifest.json'):raise ValueError('cell manifest identity mismatch')
    if expected_sha and identity.get('checkpoint_sha256')!=expected_sha:raise ValueError('cell checkpoint identity mismatch')
    if checkpoint is not None:
        actual=checkpoint_identity(checkpoint)
        if any(identity.get(k)!=v for k,v in actual.items()):raise ValueError('cell checkpoint/source identity mismatch')
    if require_v2:
        from .gapcov_provenance import validate_sources,CELL_AUX
        root=Path(__file__).resolve().parents[1]
        if (identity.get('source_provenance')!=validate_sources(root,CELL_AUX) or
            identity.get('model')!='xunet_wide_gapcov32' or identity.get('cell_supervision')!='changed_cells_v1'):
            raise ValueError('cell candidate source/head identity mismatch')
    return summary


def validate_cutoffs(folder,summary,keys):
    import numpy as np
    from .cell_metrics import fa_cutoff
    with np.load(Path(folder)/'val-scores.npz',allow_pickle=False) as arrays:
        valid=arrays['valid_negative'].astype(bool)
        for key in keys:
            scores=arrays[key]
            if scores.shape!=(len(valid),2,8,8) or valid.shape[1:]!=(8,8) or not np.isfinite(scores).all():
                raise ValueError('invalid validation cell score geometry')
            negatives=np.concatenate((scores[:,0].ravel(),scores[:,1][valid]))
            cutoff,fa=fa_cutoff(negatives)
            item=summary['maps'][key];limits=summary['color_limits'][key]
            if (item['cutoff']!=cutoff or item['validation_FA']!=fa or item['validation_negative_cells']!=len(negatives)):
                raise ValueError('cell cutoff is not this model own validation 5% cutoff')
            expected=np.quantile(scores,[.01,.99]);lo,hi=expected
            if lo==hi:expected=np.array([lo-1,hi+1])
            if not np.array_equal(np.asarray(limits),expected):raise ValueError('cell limits are not validation-derived')
