"""Fail before overwriting an incompatible GAP+covariance run on resume."""
import json
from pathlib import Path
import re
import torch
from .cov_pool import MODEL,SPEC


def cell_start(settings):
    return getattr(settings,'cell_supervision','off')=='changed_cells_v1'


def check_start(settings,folder):
    from .curriculum import check_start as recipe_check
    cells=cell_start(settings)
    provenance=None
    if cells:
        from .gapcov_provenance import validate_sources,CELL_AUX
        provenance=validate_sources(settings.root,CELL_AUX)
        if (not getattr(settings,'init_checkpoint',None) or
                getattr(settings,'embedding_schedule',None)!='original' or
                not settings.paired_batches):
            raise ValueError('gapcov32 cell training needs paired original weights-only start')
    elif getattr(settings,'init_checkpoint',None) is not None or getattr(settings,'embedding_schedule',None) is not None:
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
    if h.get('cell_supervision','off')!=getattr(settings,'cell_supervision','off'):
        raise ValueError('gapcov32 resume cell supervision mismatch')
    state=saved['state_dict']
    if state['net.classifier.weight'].shape!=(2,784) or state['net.cov_reduce.weight'].shape!=(32,256,1,1):
        raise ValueError('gapcov32 resume head shapes mismatch')
    from .models import SRM_ARRAY
    bank=torch.from_numpy(SRM_ARRAY).unsqueeze(1).repeat(3,1,1,1)
    if not torch.equal(state['net.kv.conv.weight'],bank):raise ValueError('gapcov32 resume SRM must remain frozen')
    if cells:
        start=json.loads((folder/'training_start.json').read_text())
        if any(item.get('source_provenance')!=provenance for item in (start,record,h)):
            raise ValueError('gapcov32 cell resume source provenance mismatch')
        _check_cell_resume(settings,folder,record,saved)
    return True


def _check_cell_resume(settings,folder,record,saved):
    """New continuation identity; old scratch records and SPEC stay unchanged."""
    h=saved['hyper_parameters']
    start=json.loads((folder/'training_start.json').read_text())
    required=dict(cell_supervision='changed_cells_v1',cell_loss_weight=settings.cell_loss_weight,
        paired_batches=True,front_end='srm',norm='none',width=4,clip_t=3.,
        batch_size=settings.batch_size,precision=settings.precision,lr=settings.lr,
        weight_decay=1e-4,max_epochs=settings.epochs,warmup_epochs=settings.warmup_epochs,
        grad_clip=settings.grad_clip,lr_drops=list(map(list,settings.lr_drops)),
        channels_last=settings.channels_last,val_batch_images=settings.val_batch_images)
    for key,value in required.items():
        actual=h.get(key,False if key=='channels_last' else None)
        if json.dumps(actual)!=json.dumps(value):raise ValueError('gapcov32 cell resume checkpoint identity mismatch: '+key)
    if (record.get('initialization')!='weights_only' or start.get('weights_only') is not True or
            start.get('optimizer_restored') is not False or start.get('scheduler_restored') is not False or
            start.get('checkpoint_sha256')!=settings.init_checkpoint_sha256 or
            record.get('initialization_checkpoint_sha256')!=settings.init_checkpoint_sha256 or
            record.get('cell_supervision')!='changed_cells_v1' or
            record.get('cell_loss_weight')!=settings.cell_loss_weight):
        raise ValueError('gapcov32 cell resume initialization record mismatch')
    for key in ('initial_weights_sha256','initial_cell_head_sha256'):
        value=start.get(key)
        if (not isinstance(value,str) or not re.fullmatch('[0-9a-f]{64}',value) or record.get(key)!=value or h.get(key)!=value):
            raise ValueError('gapcov32 cell resume initialization fingerprint mismatch: '+key)
    if h.get('initialization_checkpoint_sha256')!=settings.init_checkpoint_sha256:
        raise ValueError('gapcov32 cell resume source checkpoint mismatch')
    from .models import model_from_hparams
    # Check exact keys and shapes without changing the training RNG stream.
    with torch.random.fork_rng(devices=[]):network=model_from_hparams(h)
    network.load_state_dict({k.removeprefix('net.'):v for k,v in saved['state_dict'].items()},strict=True)
    # A same-shaped checkpoint without trained-head Adam moments is not a resume.
    try:
        optimizer=saved['optimizer_states'][0]
        ids=optimizer['param_groups'][0]['params']
        names=list(dict(network.named_parameters()))
        if len(ids)!=len(names):raise ValueError('gapcov32 cell resume optimizer parameter count mismatch')
        for key in ('cell_head.weight','cell_head.bias'):
            moments=optimizer['state'][ids[names.index(key)]]
            for field in ('exp_avg','exp_avg_sq'):
                if moments[field].shape!=network.state_dict()[key].shape or not torch.isfinite(moments[field]).all():
                    raise ValueError('gapcov32 cell resume invalid head Adam moments')
            if saved['global_step']<=0 or float(moments['step'])!=saved['global_step']:
                raise ValueError('gapcov32 cell resume head Adam step mismatch')
        if len(saved['lr_schedulers'])!=1:raise ValueError('gapcov32 cell resume needs scheduler state')
    except (KeyError,IndexError,TypeError) as error:
        raise ValueError('gapcov32 cell resume lacks head optimizer/scheduler state') from error
