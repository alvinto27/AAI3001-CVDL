"""ONNX Runtime CPU engine: the same bundle, served without PyTorch.

A bundle folder may add two files next to deploy.json and weights.pt:

- model.onnx: the network exported by export_onnx.py. Input `images`
  (float32 [N,3,256,256], 0..255). Outputs `tile_prob` [N] and
  `cell_logits` [N,8,8], the same values as cell_maps(include_primary=False).
- onnx.json: binds model.onnx to the bundle. The engine refuses the model
  when any hash differs, so a model can never pair with another bundle's
  cut-offs or photo rule.
"""
import json
import os
from pathlib import Path

import numpy as np

from .contract import BundleError, DEFAULT_BUNDLE, contract_hash, file_hash, validate_config

ONNX_FIELDS={'format_version','onnx_sha256','weights_sha256','contract_sha256','precision','opset','inputs','outputs','export'}


def validate_onnx_record(record,config):
    if not isinstance(record,dict) or set(record)!=ONNX_FIELDS or record['format_version']!=1:
        raise BundleError('invalid onnx.json fields')
    if record['weights_sha256']!=config['weights_sha256'] or record['contract_sha256']!=contract_hash(config):
        raise BundleError('onnx.json does not belong to this bundle')
    if record['precision']!='fp32':
        raise BundleError('only the fp32 ONNX model is supported')
    if record['inputs']!={'images':['N',3,256,256]} or record['outputs']!={'tile_prob':['N'],'cell_logits':['N',8,8]}:
        raise BundleError('unsupported ONNX signature')


class OnnxEngine:
    """One ONNX Runtime session. run_batch(float32 NCHW) -> (tile_prob, cell_logits)."""
    def __init__(self,path,threads=None):
        import onnxruntime as ort
        options=ort.SessionOptions()
        if threads is not None:options.intra_op_num_threads=threads
        options.inter_op_num_threads=1
        self.session=ort.InferenceSession(str(path),options,providers=['CPUExecutionProvider'])
        self.version=ort.__version__

    def __call__(self,images):
        tile,cells=self.session.run(['tile_prob','cell_logits'],{'images':np.ascontiguousarray(images,dtype=np.float32)})
        if not (np.isfinite(tile).all() and np.isfinite(cells).all()):raise ValueError('nonfinite engine output')
        return tile,cells


def load_onnx_bundle(bundle_dir=None,*,device='cpu',threads=None):
    """Same folder selection as load_bundle. CPU only."""
    folder=Path(bundle_dir or os.environ.get('CVDL_BUNDLE') or DEFAULT_BUNDLE)
    if device not in ('cpu','auto'):raise BundleError('the ONNX engine runs on CPU only')
    if threads is not None and (type(threads) is not int or threads<1):raise BundleError('threads must be a positive integer')
    try:
        config=json.loads((folder/'deploy.json').read_text())
        validate_config(config)
        if config['map']['kind']!='trained_head':raise BundleError('the ONNX engine serves trained-head maps only')
        record=json.loads((folder/'onnx.json').read_text())
        validate_onnx_record(record,config)
        model=folder/'model.onnx'
        if file_hash(model)!=record['onnx_sha256']:raise BundleError('ONNX model hash mismatch')
        engine=OnnxEngine(model,threads)
    except BundleError:
        raise
    except (OSError,ValueError,TypeError,KeyError,RuntimeError) as exc:
        raise BundleError('cannot load ONNX bundle: '+str(exc)) from exc
    except Exception as exc:  # onnxruntime raises its own exception types
        raise BundleError('cannot load ONNX model: '+type(exc).__name__) from exc
    return engine,config,'cpu'
