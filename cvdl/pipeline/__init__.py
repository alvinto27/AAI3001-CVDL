"""Small, inference-only photo API. Load once, then call pipeline.analyze(photo)."""
import base64
import io
import threading

import numpy as np
from PIL import Image

from .bundle import BundleError, load_bundle
from .inputs import InputError, InputTooLarge, UnsupportedInput, decode_photo
from ..photo_inference import cell_edges, cell_to_pixels, infer_photo
from ..photo_verdict import features, predict


class Pipeline:
    def __init__(self,network,config,device,batch_size):
        self._network=network
        self._config=config
        self.device=device
        self.batch_size=batch_size
        self._lock=threading.Lock()

    @property
    def identity(self):
        return dict(name=self._config['name'],version=self._config['version'],
            format_version=self._config['format_version'],weights_sha256=self._config['weights_sha256'])

    def analyze(self,photo,*,allow_rgba=False,include_images=False):
        """Stored-pixel RGB8 analysis; scores are not calibrated real-world risk."""
        with self._lock:
            return self._analyze(photo,allow_rgba=allow_rgba,include_images=include_images)

    def _analyze(self,photo,*,allow_rgba,include_images):
        pixels,metadata=decode_photo(photo,allow_rgba=allow_rgba)
        h,w=pixels.shape[:2];config=self._config
        record=dict(photo_id='input',source_height=h,source_width=w)
        origins,scores,maps=infer_photo(self._network,pixels,record,self.device,self.batch_size,
            map_kinds=(config['map']['kind'],))
        grid=maps[0]
        tile_cutoff=config['tile']['cutoff'];cell_cutoff=config['map']['cutoff']
        values,counts=features(scores,grid,tile_cutoff,cell_cutoff)
        rule=config['photo_rule']
        logit=float(predict(rule['model'],[[values[k] for k in rule['features']]])[0])
        probability=float(np.exp(-np.logaddexp(0.,-logit)))
        detected=logit>=rule['cutoff']
        result=dict(schema_version=1,bundle=self.identity,device=self.device,width=w,height=h,
            input=metadata,photo_verdict='message_detected' if detected else 'not_detected',
            photo_rule=dict(type=rule['type'],logit=logit,probability=probability,cutoff=rule['cutoff'],
                probability_note='sigmoid score, not a calibrated real-world probability'),
            features=values,counts=counts,tile_cutoff=tile_cutoff,
            tiles=[dict(box=[x,y,x+256,y+256],probability=float(s),flagged=bool(s>=tile_cutoff)) for (y,x),s in zip(origins,scores)],
            heatmap=dict(kind=config['map']['kind'],score=config['map']['score'],cell_size=32,
                shape=list(grid.shape),x_edges=cell_edges(w).tolist(),y_edges=cell_edges(h).tolist(),
                cutoff=cell_cutoff,values=grid.tolist(),flagged=(grid>=cell_cutoff).tolist()))
        if include_images:
            result['images']=_images(pixels,grid,cell_cutoff)
        return result


def _images(pixels,grid,cutoff):
    # Fixed display mapping: red above cutoff, blue below. Not a probability map.
    display=np.exp(-np.logaddexp(0.,-(grid-cutoff)))
    colors=np.stack((display,np.zeros_like(display),1-display),axis=-1)
    cell_rgb=np.rint(colors*255).astype(np.uint8)
    rgb=cell_to_pixels(cell_rgb.transpose(2,0,1),pixels.shape[:2]).transpose(1,2,0)
    heat=Image.fromarray(rgb)
    overlay=Image.blend(Image.fromarray(pixels),heat,.4)
    result=dict(content_type='image/png',display_scale='sigmoid(raw map - cell cutoff); display only')
    for name,image in (('heatmap_png_base64',heat),('overlay_png_base64',overlay)):
        buffer=io.BytesIO();image.save(buffer,format='PNG')
        result[name]=base64.b64encode(buffer.getvalue()).decode('ascii')
    return result


def load_pipeline(bundle_dir=None,*,device='cpu',batch_size=8):
    """Select explicit folder > CVDL_BUNDLE > shipped bundle. Never download/refit."""
    if type(batch_size) is not int or not 1<=batch_size<=8:
        raise ValueError('batch_size must be in 1..8')
    network,config,device=load_bundle(bundle_dir,device=device)
    return Pipeline(network,config,device,batch_size)


def analyze(photo,*,bundle_dir=None,device='cpu',batch_size=8,allow_rgba=False,include_images=False):
    """One-shot convenience call; services should keep a loaded pipeline instead."""
    return load_pipeline(bundle_dir,device=device,batch_size=batch_size).analyze(
        photo,allow_rgba=allow_rgba,include_images=include_images)


__all__=['load_pipeline','analyze','Pipeline','BundleError','InputError','InputTooLarge','UnsupportedInput']
