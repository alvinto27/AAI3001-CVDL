"""Opt-in resume guard for the same-shaped frozen/learnable SRM models."""
import json
import re
from pathlib import Path
import torch


def check_start(settings,folder):
    from .curriculum import check_start as recipe_check
    resuming=recipe_check(settings,folder)
    if not resuming:return False
    folder=Path(folder)
    record=json.loads((folder/'model_run_settings.json').read_text())
    if record.get('model')!='xunet_wide_learn_srm' or record.get('srm_trainable') is not True:
        raise ValueError('learn_srm resume model/trainability record mismatch')
    if not re.fullmatch('[0-9a-f]{64}',record.get('initial_weights_sha256','')):
        raise ValueError('learn_srm resume missing initial weights fingerprint')
    saved=torch.load(folder/'last.ckpt',map_location='cpu',weights_only=False)
    h=saved.get('hyper_parameters',{})
    required=dict(model=settings.model,front_end='srm',norm='none',width=4,clip_t=3.,
        batch_size=settings.batch_size,precision=settings.precision,lr=settings.lr,weight_decay=1e-4,
        paired_batches=settings.paired_batches,max_epochs=settings.epochs,val_batch_images=settings.val_batch_images)
    if settings.val_batch_images is not None:
        required['effective_val_batch_images']=min(settings.batch_size,settings.val_batch_images)
    for key,value in required.items():
        if h.get(key)!=value:raise ValueError('learn_srm resume checkpoint identity mismatch: '+key)
    # Optimizer parameter0 is the original first module's SRM bank. A frozen R1
    # state has no Adam moments for it, even though its param group lists it.
    try:
        optimizer=saved['optimizer_states'][0]
        first=optimizer['param_groups'][0]['params'][0]
        state=optimizer['state'][first]
        for key in ('exp_avg','exp_avg_sq'):
            if state[key].shape!=(90,1,5,5) or not torch.isfinite(state[key]).all():raise ValueError('invalid SRM Adam moments')
        if float(state['step'])!=saved['global_step'] or saved['global_step']<=0:raise ValueError('SRM Adam step mismatch')
    except (KeyError,IndexError,TypeError) as error:
        raise ValueError('learn_srm resume lacks SRM optimizer state') from error
    return True
