"""python -m cvdl.pipeline PHOTO --output result.json [--overlay overlay.png]."""
import argparse
import base64
import json
from pathlib import Path
import sys

from . import load_pipeline


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('photo',type=Path)
    parser.add_argument('--bundle',type=Path)
    parser.add_argument('--device',choices=('cpu','cuda','auto'),default='cpu')
    parser.add_argument('--batch-size',type=int,default=8)
    parser.add_argument('--allow-rgba',action='store_true')
    parser.add_argument('--output',type=Path)
    parser.add_argument('--overlay',type=Path)
    parser.add_argument('--heatmap',type=Path)
    args=parser.parse_args(argv)
    try:
        for path in (args.output,args.overlay,args.heatmap):
            if path and path.resolve()==args.photo.resolve():raise ValueError('output must not overwrite the input photo')
        destinations=[p.resolve() for p in (args.output,args.overlay,args.heatmap) if p]
        if len(destinations)!=len(set(destinations)):raise ValueError('output paths must be distinct')
        pipeline=load_pipeline(args.bundle,device=args.device,batch_size=args.batch_size)
        result=pipeline.analyze(args.photo,allow_rgba=args.allow_rgba,include_images=bool(args.overlay or args.heatmap))
        images=result.pop('images',None)
        if images:
            result['images']=dict(content_type='image/png',display_scale=images['display_scale'])
            for key,path in (('overlay_png_base64',args.overlay),('heatmap_png_base64',args.heatmap)):
                if path:
                    path.write_bytes(base64.b64decode(images[key]))
                    result['images'][key.removesuffix('_base64')]=str(path)
        text=json.dumps(result,allow_nan=False)+'\n'
        if args.output:args.output.write_text(text)
        else:sys.stdout.write(text)
    except (ValueError,OSError,RuntimeError) as exc:
        print('analysis failed: '+str(exc),file=sys.stderr)
        return 2
    return 0


if __name__=='__main__':raise SystemExit(main())
