"""Fail before overwriting an incompatible GAP+covariance run on resume."""
import json
from pathlib import Path
import torch
from .cov_pool import MODEL,SPEC


def check_start(settings,folder):
    from .curriculum import check_start as recipe_check
    if getattr(settings,'init_checkpoint',None) is not None or getattr(settings,'embedding_schedule',None) is not None:
        raise ValueError('gapcov32 starts from random weights without a curriculum')
    resuming=recipe_check(settings,folder)
    if not resuming:return False
    folder=Path(folder);record=json.loads((folder/'model_run_settings.json').read_text())
    fingerprint=record.get('initial_weights_sha256')
    if (record.get('model')!=MODEL or record.get('cov_pool_identity')!=SPEC or
            record.get('srm_trainable') is not False or not isinstance(fingerprint,str) or
            len(fingerprint)!=64 or any(c not in '0123456789abcdef' for c in fingerprint)):
        raise ValueError('gapcov32 resume model/head/initialization mismatch')
    saved=torch.load(folder/'last.ckpt',map_location='cpu',weights_only=False)
    h=saved.get('hyper_parameters',{})
    if h.get('model')!=MODEL or h.get('cov_pool_identity')!=SPEC:
        raise ValueError('gapcov32 resume checkpoint head identity mismatch')
    state=saved['state_dict']
    if state['net.classifier.weight'].shape!=(2,784) or state['net.cov_reduce.weight'].shape!=(32,256,1,1):
        raise ValueError('gapcov32 resume head shapes mismatch')
    from .models import SRM_ARRAY
    bank=torch.from_numpy(SRM_ARRAY).unsqueeze(1).repeat(3,1,1,1)
    if not torch.equal(state['net.kv.conv.weight'],bank):raise ValueError('gapcov32 resume SRM must remain frozen')
    return True
