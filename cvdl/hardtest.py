"""Shared hard-test design constants, Wilson intervals and pair scoring."""
import math
import pandas as pd

KS=(1,2)
CONDITIONS=('all','low','noisy','low_noisy')
PAIRS_PER_CELL=1008                      # 24 balanced combinations x 42
# bit_mode x method x placement x payload: equal split, each combination exactly PAIRS_PER_CELL/24 times
COMBINATIONS=tuple((b,m,p,y) for b in ('lowest','plane') for m in ('replace','match')
                   for p in ('contiguous','stride','random') for y in ('random_bytes','letters'))
PER_COMBINATION=PAIRS_PER_CELL//len(COMBINATIONS)


def wilson(events, n, z=1.96):
    if n<=0:
        return (float('nan'),float('nan'))
    p=events/n
    centre=(p+z*z/(2*n))/(1+z*z/n)
    half=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/(1+z*z/n)
    return (centre-half,centre+half)


def score(network, dataset, *, device, batch_size=16, workers=0):
    """Score every pair: one row per pair and label with prob_stego.

    One task per photo, in arrival order. The standard OrderedPairLoader is not used: it holds results until
    the next pair in table order arrives, and a hard-test list is scattered over all photos, so it would buffer
    most of the list in memory. Order does not matter here: every row carries its pair index.
    """
    import torch
    from torch.utils.data import DataLoader
    from lsb_torch import PhotoPairTasks, initialize_worker, prefetch_for_images
    tasks=PhotoPairTasks(dataset)
    kwargs=dict(batch_size=None,num_workers=workers,pin_memory=device!='cpu',worker_init_fn=initialize_worker)
    if workers:kwargs.update(multiprocessing_context='spawn',prefetch_factor=prefetch_for_images(tasks.max_images,128))
    network.eval()
    rows=[]
    with torch.inference_mode():
        for indices,clean,stego in DataLoader(tasks,**kwargs):
            for lo in range(0,len(indices),batch_size):
                part=slice(lo,lo+batch_size)
                images=torch.cat([clean[part],stego[part]],dim=0).to(device,dtype=torch.float32)
                probability=torch.softmax(network(images),dim=1)[:,1].cpu().numpy()
                count=clean[part].shape[0]
                for position,index in enumerate(indices[part].tolist()):
                    rows.append((int(index),0,float(probability[position])));rows.append((int(index),1,float(probability[count+position])))
    out=pd.DataFrame(rows,columns=['index','label','prob_stego']).sort_values(['index','label']).reset_index(drop=True)
    meta=dataset.frame.iloc[out['index'].to_numpy()].reset_index(drop=True)
    return pd.concat([meta,out[['label','prob_stego']]],axis=1)
