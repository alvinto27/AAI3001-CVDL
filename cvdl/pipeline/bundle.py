"""Versioned deployment contract and strict, weights-only model loading."""
import hashlib
import json
import os
import pickle
from pathlib import Path

import numpy as np
import torch

from ..models import model_from_hparams
from ..photo_verdict import FEATURES


class BundleError(ValueError):
    """An unsupported or inconsistent deployment bundle."""


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def contract_hash(config):
    return hashlib.sha256(canonical({k:v for k,v in config.items() if k!='weights_sha256'}).encode()).hexdigest()


def file_hash(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def _finite(value):
    return not isinstance(value, bool) and isinstance(value, (float, int)) and np.isfinite(value)


def validate_config(config):
    """Only implemented geometry/feature/architecture contracts are accepted."""
    try:
        if set(config)!={'format_version','name','version','architecture','input','tile','map','photo_rule','source','weights_sha256'}:
            raise ValueError('invalid bundle fields')
        if type(config['format_version']) is not int or config['format_version']!=1:
            raise ValueError('unsupported bundle format version')
        if any(not isinstance(config[k],str) or not config[k] for k in ('name','version')):
            raise ValueError('bundle name/version required')
        architecture=config['architecture']
        if set(architecture)!={'model','clip_t','norm','front_end','width','cell_supervision'}:
            raise ValueError('invalid architecture fields')
        if architecture['model'] not in ('xunet','xunet_wide') or architecture['norm'] not in ('none','batch','group') or architecture['front_end'] not in ('kv','srm'):
            raise ValueError('unsupported architecture')
        if type(architecture['width']) is not int or architecture['width'] not in (1,2,4):
            raise ValueError('unsupported width')
        if architecture['clip_t'] is not None and (not _finite(architecture['clip_t']) or architecture['clip_t']<=0):
            raise ValueError('invalid clip')
        if architecture['cell_supervision'] not in ('off','changed_cells_v1'):
            raise ValueError('unsupported cell supervision')
        if config['input']!='rgb8_0_255':raise ValueError('unsupported preprocessing')
        tile=config['tile'];maps=config['map']
        if set(tile)!={'size','grid','cutoff'} or type(tile['size']) is not int or tile['size']!=256 or tile['grid']!='overlap_last':
            raise ValueError('only 256 overlap-last is supported')
        if not _finite(tile['cutoff']) or not 0<=tile['cutoff']<=1:raise ValueError('invalid tile cutoff')
        if set(maps)!={'kind','cell_size','score','cutoff'} or type(maps['cell_size']) is not int or maps['cell_size']!=32:
            raise ValueError('unsupported cell geometry')
        kinds={'trained_head':'logit','cam':'class1_minus_class0'}
        if maps['kind'] not in kinds or maps['score']!=kinds[maps['kind']] or not _finite(maps['cutoff']):
            raise ValueError('unsupported map contract')
        if maps['kind']=='trained_head' and architecture['cell_supervision']=='off':
            raise ValueError('trained-head map requires a cell head')
        rule=config['photo_rule']
        if set(rule)!={'type','features','model','cutoff'} or rule['type']!='logistic_v1' or rule['features']!=list(FEATURES):
            raise ValueError('unsupported photo rule/features')
        if not _finite(rule['cutoff']):raise ValueError('invalid photo cutoff')
        model=rule['model']
        if set(model)!={'mean','scale','coefficients','intercept'} or not _finite(model['intercept']):
            raise ValueError('invalid logistic model')
        for key in ('mean','scale','coefficients'):
            if not isinstance(model[key],list) or len(model[key])!=len(FEATURES) or not all(_finite(v) for v in model[key]):
                raise ValueError('invalid logistic vector')
        if min(model['scale'])<=0:raise ValueError('scales must be positive')
        source=config['source']
        if set(source)!={'checkpoint_sha256','evaluation_id','threshold_id','photo_fit_sha256'}:
            raise ValueError('invalid source identity')
        for value,length in ((source['checkpoint_sha256'],64),(source['photo_fit_sha256'],64),(config['weights_sha256'],64)):
            if not isinstance(value,str) or len(value)!=length or any(c not in '0123456789abcdef' for c in value):
                raise ValueError('invalid source/weights hash')
        if any(not isinstance(source[k],str) or not source[k] for k in ('evaluation_id','threshold_id')):
            raise ValueError('evaluation/threshold ID required')
        canonical(config)
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        raise BundleError(str(exc)) from exc


def load_bundle(bundle_dir=None, *, device='cpu'):
    folder=Path(bundle_dir or os.environ.get('CVDL_BUNDLE') or Path(__file__).resolve().parents[2]/'deploy/run13-rule-c-v1')
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
