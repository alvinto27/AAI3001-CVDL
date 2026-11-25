"""Versioned deployment contract and strict, weights-only model loading."""
import json
import os
import pickle
from pathlib import Path

import torch

from ..cov_pool import MODEL as GAPCOV, SPEC  # noqa: F401  (contract.py keeps checked copies)
from ..models import model_from_hparams
from .contract import (BundleError, DEFAULT_BUNDLE, canonical, contract_hash, file_hash,  # noqa: F401
    validate_config)


def load_bundle(bundle_dir=None, *, device='cpu'):
    folder=Path(bundle_dir or os.environ.get('CVDL_BUNDLE') or DEFAULT_BUNDLE)
    try:
        config=json.loads((folder/'deploy.json').read_text())
        validate_config(config)
        weights=folder/'weights.pt'
        if file_hash(weights)!=config['weights_sha256']:raise BundleError('weights hash mismatch')
        payload=torch.load(weights,map_location='cpu',weights_only=True)
        if set(payload)!={'state_dict','contract_sha256'} or payload['contract_sha256']!=contract_hash(config):
            raise BundleError('weights/model/threshold/photo-rule identity mismatch')
        if not isinstance(payload['state_dict'],dict) or not all(isinstance(k,str) and isinstance(v,torch.Tensor) for k,v in payload['state_dict'].items()):
            raise BundleError('weights must contain network tensors only')
        if any(not torch.isfinite(v).all().item() for v in payload['state_dict'].values()):
            raise BundleError('nonfinite network tensor')
        network=model_from_hparams(config['architecture'])
        expected=network.state_dict()
        for key,value in payload['state_dict'].items():
            if key in expected and (value.dtype!=expected[key].dtype or value.layout!=torch.strided):
                raise BundleError('network tensor dtype/layout mismatch')
        network.load_state_dict(payload['state_dict'],strict=True)
        if config['map']['kind']=='trained_head' and not hasattr(network,'cell_head'):
            raise BundleError('missing trained cell head')
        if device=='auto':device='cuda' if torch.cuda.is_available() else 'cpu'
        if device not in ('cpu','cuda'):raise BundleError('device must be cpu, cuda or auto')
        if device=='cuda' and not torch.cuda.is_available():raise BundleError('requested GPU is unavailable')
        network.eval().to(device=device,dtype=torch.float32)
    except BundleError:
        raise
    except (OSError, ValueError, TypeError, RuntimeError, KeyError, EOFError, pickle.UnpicklingError) as exc:
        raise BundleError('cannot load deployment bundle: '+str(exc)) from exc
    return network,config,device
