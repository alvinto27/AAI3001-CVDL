"""Export a bundle's network to ONNX (model.onnx + onnx.json). Needs torch.

    python -m cvdl.pipeline.export_onnx deploy/run22-rule-c-v1

Export is a build step: the served ONNX engine does not import torch. The
check below compares PyTorch and ONNX Runtime outputs on fixed synthetic
tiles and refuses to write onnx.json when they differ by more than TOLERANCE.
"""
import argparse
import json
import platform
from pathlib import Path

import numpy as np

from .contract import contract_hash, file_hash

OPSET=18
TOLERANCE=1e-3
INPUTS={'images':['N',3,256,256]}
OUTPUTS={'tile_prob':['N'],'cell_logits':['N',8,8]}


def check_tiles(count=32,seed=20261125):
    """Smooth photo-like tiles plus noise; half get random LSB flips."""
    rng=np.random.default_rng(seed)
    yy,xx=np.mgrid[0:256,0:256]/255.
    tiles=[]
    for i in range(count):
        base=np.stack([(a*yy+b*xx+c)%1 for a,b,c in rng.random((3,3))],0)*200+28
        tile=np.clip(base+rng.normal(0,4,base.shape),0,255).astype(np.uint8)
        if i%2:tile^=(rng.random(tile.shape)<.3).astype(np.uint8)
        tiles.append(tile)
    return np.stack(tiles).astype(np.float32)


def export(bundle_dir,*,threads=2):
    import torch
    import onnxruntime
    from .bundle import load_bundle
    from .onnx_engine import OnnxEngine, validate_onnx_record

    class TileHead(torch.nn.Module):
        """The served outputs, as in cell_maps(include_primary=False)."""
        def __init__(self,network):
            super().__init__();self.network=network

        def forward(self,images):
            logits,aux=self.network(images,return_aux=True)
            return logits.softmax(1)[:,1],aux[:,0]

    folder=Path(bundle_dir)
    network,config,_=load_bundle(folder,device='cpu')
    if config['map']['kind']!='trained_head':raise ValueError('only trained-head bundles export to ONNX')
    model=TileHead(network).eval()
    path=folder/'model.onnx'
    torch.onnx.export(model,(torch.zeros(8,3,256,256),),str(path),dynamo=True,input_names=list(INPUTS),
        output_names=list(OUTPUTS),dynamic_shapes=({0:torch.export.Dim('batch',min=1,max=64)},),
        opset_version=OPSET,external_data=False)
    tiles=check_tiles()
    torch.set_num_threads(threads)
    engine=OnnxEngine(path,threads)
    with torch.inference_mode():
        expected=[torch.cat(v).numpy() for v in zip(*[model(torch.from_numpy(tiles[i:i+8])) for i in range(0,len(tiles),8)])]
    got=[np.concatenate(v) for v in zip(*[engine(tiles[i:i+8]) for i in range(0,len(tiles),8)])]
    differences={name:float(np.abs(a-b).max()) for name,a,b in zip(OUTPUTS,expected,got)}
    if any(d>TOLERANCE for d in differences.values()):
        path.unlink()
        raise ValueError(f'ONNX outputs differ from PyTorch: {differences}')
    record=dict(format_version=1,onnx_sha256=file_hash(path),weights_sha256=config['weights_sha256'],
        contract_sha256=contract_hash(config),precision='fp32',opset=OPSET,inputs=INPUTS,outputs=OUTPUTS,
        export=dict(torch=str(torch.__version__).split('+')[0],onnxruntime=onnxruntime.__version__,python=platform.python_version(),
            check=dict(tiles=len(tiles),seed=20261125,tolerance=TOLERANCE,max_abs_difference=differences)))
    validate_onnx_record(record,config)
    (folder/'onnx.json').write_text(json.dumps(record,indent=2)+'\n')
    return record


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('bundle',nargs='+',type=Path)
    args=parser.parse_args(argv)
    for folder in args.bundle:
        record=export(folder)
        print(folder,record['onnx_sha256'][:12],record['export']['check']['max_abs_difference'])


if __name__=='__main__':main()
