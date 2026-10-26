"""Shared whole-photo inference and pixel-first overlap reduction (no training)."""
import numpy as np


def cell_edges(length):
    return np.r_[np.arange(0,length,32),length]


def cell_to_pixels(values,shape):
    """Nearest-neighbor 32px cells; clip only the final partial photo cells."""
    h,w=shape;values=np.asarray(values)
    if values.shape[-2:]!=((h+31)//32,(w+31)//32):raise ValueError('cell grid differs from photo shape')
    return np.repeat(np.repeat(values,32,axis=-2),32,axis=-1)[...,:h,:w]


def reduce_cells(pixels):
    """Photo-anchored cell means; partial edge bins use their actual area."""
    h,w=pixels.shape[-2:];ys=cell_edges(h);xs=cell_edges(w)
    total=np.add.reduceat(np.add.reduceat(pixels,ys[:-1],axis=-2,dtype=np.float64),xs[:-1],axis=-1,dtype=np.float64)
    return total/(np.diff(ys)[:,None]*np.diff(xs)[None,:])


def stitch_cells(shape,origins,maps):
    """Average overlapping tiles at pixels BEFORE taking photo cell means.

    Last-tile offsets need not be multiples of 32. Input [tiles,maps,8,8];
    output [maps,ceil(H/32),ceil(W/32)], including partial edge cells.
    """
    maps=np.asarray(maps);h,w=shape
    if maps.ndim!=4 or maps.shape[0]!=len(origins) or maps.shape[-2:]!=(8,8) or not np.isfinite(maps).all():
        raise ValueError('expected finite [tiles,maps,8,8] maps')
    total=np.zeros((maps.shape[1],h,w),np.float64);coverage=np.zeros((h,w),np.uint8)
    for (y,x),values in zip(origins,maps):
        if not 0<=y<=h-256 or not 0<=x<=w-256:raise ValueError('tile outside photo')
        total[:,y:y+256,x:x+256]+=cell_to_pixels(values,(256,256))
        coverage[y:y+256,x:x+256]+=1
    if not coverage.all():raise ValueError('uncovered photo pixels')
    total/=coverage[None]
    return reduce_cells(total)


def infer_photo(network,photo,record,device,batch_size,*,map_kinds=('trained_head','cam')):
    """Photo stitching with explicit trained-head, CAM or signed attribution maps."""
    import torch
    from lsb_train_grid import axes,crop
    from .pixels import evaluation_tensor
    from .cell_diagnostics import cell_maps
    if type(batch_size) is not int or batch_size<1:raise ValueError('batch size must be positive')
    if not map_kinds or any(k not in ('trained_head','cam','attribution') for k in map_kinds):
        raise ValueError('unsupported map kind')
    attribution=getattr(network,'cell_map_kind',None)=='attribution'
    if attribution and any(k!='attribution' for k in map_kinds):
        raise ValueError('this model requires attribution, not CAM or trained-head maps')
    if not attribution and 'attribution' in map_kinds:
        raise ValueError('attribution map requires a GAP+covariance model')
    if 'trained_head' in map_kinds and not hasattr(network,'cell_head'):
        raise ValueError('trained-head map requested from a headless model')
    ys,xs=axes(record,256,'overlap_last')
    positions=[(r,c) for r in range(len(ys)) for c in range(len(xs))]
    origins=[];scores=[];maps=[]
    for lo in range(0,len(positions),batch_size):
        tensors=[]
        for r,c in positions[lo:lo+batch_size]:
            pixels,origin,pad=crop(photo,record,256,r,c,'overlap_last')
            if any(pad):raise ValueError('photo grid must not pad real pixels')
            tensors.append(evaluation_tensor(pixels));origins.append(origin)
        images=torch.stack(tensors).to(device=device,dtype=torch.float32)
        cam,aux,tile=cell_maps(network,images)
        available={'trained_head':aux,'cam':cam,'attribution':cam if attribution else None}
        maps.append(np.stack([available[k].cpu().numpy() for k in map_kinds],axis=1))
        scores.extend(tile.cpu().numpy().astype(float))
    return origins,scores,stitch_cells(photo.shape[:2],origins,np.concatenate(maps))
