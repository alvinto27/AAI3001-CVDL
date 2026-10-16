"""Versioned, training-only geometry. Frozen evaluation grids are never changed."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import numpy as np

VERSION = 'train-grid-v1'
SUMMARIES = {512: 'summary.json', 256: 'summary-256.json'}
POLICIES = {'overlap_last': 'train_overlap_v1', 'centre_trim': 'centre_trim', 'centre_pad': 'centre_pad'}

def starts(length, size, policy):
    if type(length) is not int or type(size) is not int or size not in (256, 512) or length < size:
        raise ValueError('training grids require integer sides >= tile_size (256/512)')
    if policy == 'overlap_last':
        return tuple(min(i*size, length-size) for i in range((length+size-1)//size))
    if policy == 'centre_trim':
        return tuple((length % size)//2+i*size for i in range(length//size))
    if policy == 'centre_pad':
        n=(length+size-1)//size
        return tuple(-((n*size-length)//2)+i*size for i in range(n))
    raise ValueError(f'unknown training grid {policy!r}')

def axes(record, size, policy):
    return (starts(int(record['source_height']),size,policy),
            starts(int(record['source_width']),size,policy))

def crop(photo, record, size, row, col, policy):
    from lsb_source import _validate_pixels
    _validate_pixels(photo,record)
    ys,xs=axes(record,size,policy)
    if type(row) is not int or type(col) is not int or not 0<=row<len(ys) or not 0<=col<len(xs):
        raise ValueError('training tile outside grid')
    h,w=photo.shape[:2]
    if (h,w)!=(int(record['source_height']),int(record['source_width'])):
        raise ValueError('decoded photo dimensions differ from frozen record')
    y,x=ys[row],xs[col]
    real=photo[max(0,y):min(h,y+size),max(0,x):min(w,x+size)]
    pad=(max(0,-y),size-max(0,-y)-real.shape[0],max(0,-x),size-max(0,-x)-real.shape[1])
    return real,(y,x),pad

def pad_crop(pixels, pad):
    top,bottom,left,right=pad
    return np.pad(pixels,((top,bottom),(left,right),(0,0)),constant_values=0)

def plan(records, size, policy):
    """Canonical byte stream; no image I/O or random draws."""
    for r in sorted(records,key=lambda r:int(r['photo_index'])):
        if r['split']!='train':continue
        ys,xs=axes(r,size,policy)
        row=dict(source=r['source'],photo_id=r['photo_id'],photo_index=int(r['photo_index']),
                 H=int(r['source_height']),W=int(r['source_width']),rule=policy,version=VERSION,
                 T=size,y=list(ys),x=list(xs))
        yield json.dumps(row,sort_keys=True,separators=(',',':'))+'\n'

def summary(root, records, size, policy, *, verify=True):
    digest=hashlib.sha256();counts={}
    for row in plan(records,size,policy):
        digest.update(row.encode());value=json.loads(row)
        counts[value['source']]=counts.get(value['source'],0)+len(value['y'])*len(value['x'])
    result=dict(version=VERSION,tile_size=size,policy=policy,sha256=digest.hexdigest(),pool_counts=counts)
    if verify:
        if size not in SUMMARIES:raise ValueError('no approved training grid summary for this tile size')
        expected=json.loads((Path(root)/'data/train-grids/v1'/SUMMARIES[size]).read_text())
        sources=sorted({r['source'] for r in records if r['split']=='train'})
        # Approved summary covers the full three-source population, not QUICK subsets.
        if sources!=['alaska','lsdir','raise']:
            raise ValueError(f'approved training grid requires all three sources at {size}')
        for source in sources:
            actual=hashlib.sha256((Path(root)/f'data/{source}/photo_manifest.csv').read_bytes()).hexdigest()
            if actual!=expected['inputs'][source]:raise ValueError(f'{source} frozen manifest changed')
        if result!=expected['plans'][policy]:raise ValueError('training grid plan digest/count mismatch')
    return result
