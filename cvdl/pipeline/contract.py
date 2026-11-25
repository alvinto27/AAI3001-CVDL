"""Torch-free deployment contract: bundle fields, hashes and validation.

Shared by the PyTorch loader (bundle.py) and the ONNX Runtime engine (onnx_engine.py).
"""
import hashlib
import json
from pathlib import Path

import numpy as np

from ..photo_verdict import FEATURES

# Copies of cvdl.cov_pool.MODEL/SPEC so this module does not import torch.
# tests/test_pipeline_onnx.py asserts they equal the cov_pool originals.
GAPCOV='xunet_wide_gapcov32'
SPEC=dict(version='gapcov32-v1',gap_features=256,reduced_features=32,cov_features=528,
    covariance_divisor='M',ridge_relative=1e-4,ridge_absolute=1e-12,iterations=5,
    sqrt='trace_normalized_newton_schulz_with_compensation',triangle='upper_unweighted',
    reduction_bias=False,post_normalization='none',branch_scaling='none',
    classifier='default_Linear784_to_2',attribution='gap_cam_plus_M_centered_cov_gradient')
DEFAULT_BUNDLE=Path(__file__).resolve().parents[2]/'deploy/run22-rule-c-v1'


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
        fields={'model','clip_t','norm','front_end','width','cell_supervision'}
        if architecture.get('model')==GAPCOV:
            # Covariance pooling: the exact frozen head SPEC is part of the contract.
            if set(architecture)!=fields|{'cov_pool_identity'} or architecture['cov_pool_identity']!=SPEC:
                raise ValueError('gapcov32 requires its exact covariance head identity')
            if (architecture['front_end'],architecture['norm'],architecture['width'],architecture['clip_t'])!=('srm','none',4,3.0):
                raise ValueError('gapcov32 requires SRM, norm none, width 4 and clip 3')
        elif set(architecture)!=fields:
            raise ValueError('invalid architecture fields')
        if architecture['model'] not in ('xunet','xunet_wide',GAPCOV) or architecture['norm'] not in ('none','batch','group') or architecture['front_end'] not in ('kv','srm'):
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
        if architecture['model']==GAPCOV and maps['kind']!='trained_head':
            raise ValueError('gapcov32 bundles ship the trained head map only; CAM does not apply')
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
        fields={'checkpoint_sha256','evaluation_id','threshold_id','photo_fit_sha256'}
        if set(source) not in (fields,fields|{'identity_route'}):
            raise ValueError('invalid source identity')
        if 'identity_route' in source and source['identity_route'] not in ('legacy_run13','pipeline_native','yx_cutoff_sources'):
            raise ValueError('invalid source identity route')
        for value,length in ((source['checkpoint_sha256'],64),(source['photo_fit_sha256'],64),(config['weights_sha256'],64)):
            if not isinstance(value,str) or len(value)!=length or any(c not in '0123456789abcdef' for c in value):
                raise ValueError('invalid source/weights hash')
        if any(not isinstance(source[k],str) or not source[k] for k in ('evaluation_id','threshold_id')):
            raise ValueError('evaluation/threshold ID required')
        canonical(config)
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        raise BundleError(str(exc)) from exc
