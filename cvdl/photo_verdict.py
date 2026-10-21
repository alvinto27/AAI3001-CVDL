"""Frozen-photo aggregation: NumPy features/statistics and CPU-only logistic fit."""
import numpy as np
from .cell_metrics import fa_cutoff,interval

CELL_FEATURES=('cell_fraction','log_group_count','group_fraction','log_cell_count')
FEATURES=CELL_FEATURES+('max_tile','top3_tile','tile_fraction','log_tile_count')


def largest_group(flagged):
    flagged=np.asarray(flagged,dtype=bool)
    if flagged.ndim!=2:raise ValueError('expected a 2D cell grid')
    h,w=flagged.shape;seen=np.zeros_like(flagged);largest=0
    for row,col in np.argwhere(flagged):
        if seen[row,col]:continue
        stack=[(int(row),int(col))];seen[row,col]=True;count=0
        while stack:
            r,c=stack.pop();count+=1
            for y in range(max(0,r-1),min(h,r+2)):
                for x in range(max(0,c-1),min(w,c+2)):
                    if flagged[y,x] and not seen[y,x]:seen[y,x]=True;stack.append((y,x))
        largest=max(largest,count)
    return largest


def features(tile_scores,head_map,tile_cutoff,cell_cutoff):
    scores=np.asarray(tile_scores,dtype=np.float64);head=np.asarray(head_map,dtype=np.float64)
    if scores.ndim!=1 or not len(scores) or head.ndim!=2 or not head.size or not np.isfinite(scores).all() or not np.isfinite(head).all():
        raise ValueError('finite nonempty tile scores and head grid required')
    flag=head>=cell_cutoff;group=largest_group(flag);tiles_above=int((scores>=tile_cutoff).sum())
    values=(float(flag.mean()),float(np.log1p(group)),group/flag.size,float(np.log1p(flag.size)),
        float(scores.max()),float(np.sort(scores)[-3:].mean()),tiles_above/len(scores),float(np.log1p(len(scores))))
    return dict(zip(FEATURES,values)),dict(tiles_above=tiles_above,tiles=len(scores),cells=head.size,
        largest_group=group,flagged_cells=int(flag.sum()))


def fit_logistic(x,labels,weights):
    """One fixed CPU float64 L2=.01 fit; no CNN, search or test data."""
    import torch
    from torch.nn import functional as F
    x=np.asarray(x,dtype=np.float64);labels=np.asarray(labels,dtype=np.float64);weights=np.asarray(weights,dtype=np.float64)
    if x.ndim!=2 or len(x)!=len(labels) or len(x)!=len(weights) or not np.isfinite(x).all() or set(labels)!={0.,1.} or (weights<=0).any():
        raise ValueError('invalid logistic fit data')
    mean=x.mean(0);scale=x.std(0);scale[scale==0]=1
    design=torch.tensor((x-mean)/scale,dtype=torch.float64,device='cpu')
    y=torch.tensor(labels,dtype=torch.float64,device='cpu');weight=torch.tensor(weights,dtype=torch.float64,device='cpu')
    coefficient=torch.zeros(x.shape[1]+1,dtype=torch.float64,device='cpu',requires_grad=True)
    optimizer=torch.optim.LBFGS([coefficient],lr=1.,max_iter=1000,max_eval=1250,history_size=20,
        tolerance_grad=1e-9,tolerance_change=1e-12,line_search_fn='strong_wolfe')
    def closure():
        optimizer.zero_grad()
        loss=(F.binary_cross_entropy_with_logits(design@coefficient[:-1]+coefficient[-1],y,reduction='none')*weight).sum()/weight.sum()+.005*coefficient[:-1].square().sum()
        loss.backward();return loss
    optimizer.step(closure)
    loss=float(closure().detach());coef=coefficient.detach().numpy()
    if not np.isfinite(coef).all():raise ValueError('nonfinite logistic coefficients')
    return dict(mean=mean.tolist(),scale=scale.tolist(),coefficients=coef[:-1].tolist(),intercept=float(coef[-1]),
        l2=.01,penalty='0.5*l2*sum(coefficients**2); intercept unpenalized',iterations=int(optimizer.state[coefficient]['n_iter']),loss=loss)


def predict(model,x):
    return (np.asarray(x,dtype=np.float64)-model['mean'])/model['scale']@np.asarray(model['coefficients'])+model['intercept']


def calibrate_count(counts):
    cutoff,_=fa_cutoff(counts)
    n=int(np.floor(cutoff))+1 if cutoff==np.floor(cutoff) else int(np.ceil(cutoff))
    return n,float((np.asarray(counts)>=n).mean())


def evaluate_scope(rows,cutoffs,*,replicates=2000,seed=20261021,population=None):
    """Photo-balanced classes; paired, panel-stratified photo bootstrap."""
    names,photo=np.unique([r['photo_id'] for r in rows],return_inverse=True)
    y=np.asarray([r['label'] for r in rows],dtype=bool)
    values=np.asarray([[r[k] for k in ('a','b','c')] for r in rows],dtype=np.float64)
    positives=np.bincount(photo[y],minlength=len(names));negatives=np.bincount(photo[~y],minlength=len(names))
    if not len(names) or (positives==0).any() or not (negatives==1).all():raise ValueError('each scope photo needs one clean and at least one stego')
    base=np.ones(len(rows));base[y]=1/positives[photo[y]]
    panels=np.empty(len(names),object)
    for r,p in zip(rows,photo):panels[p]=r['panel']
    # Draw the same full TEST photo population for every scope. Filtering only
    # after resampling preserves variable hard-slice eligibility per panel.
    population = dict(zip(names,panels)) if population is None else population
    population_names=np.asarray(sorted(population))
    population_panels=np.asarray([population[p] for p in population_names])
    lookup={p:i for i,p in enumerate(population_names)}
    scope_indices=np.asarray([lookup[p] for p in names])
    groups=[np.flatnonzero(population_panels==p) for p in sorted(set(population_panels))]
    pred=values>=np.array([cutoffs[k] for k in ('a','b','c')])
    fa=np.stack([np.bincount(photo[~y],weights=pred[~y,j],minlength=len(names)) for j in range(3)],axis=1)
    miss=np.stack([np.bincount(photo[y],weights=(~pred[y,j])*base[y],minlength=len(names)) for j in range(3)],axis=1)
    prepared=[]
    for column in values.T:
        order=np.argsort(column,kind='stable');starts=np.r_[0,np.flatnonzero(np.diff(column[order]))+1]
        prepared.append((order,starts))
    def estimate(counts):
        error=((fa+miss)*counts[:,None]).sum(0)/(2*counts.sum());auc=[]
        w=base*counts[photo]
        for order,starts in prepared:
            p=np.add.reduceat(w[order]*y[order],starts);n=np.add.reduceat(w[order]*~y[order],starts)
            auc.append(float((p*(np.cumsum(n)-.5*n)).sum()/(p.sum()*n.sum())))
        return error,np.asarray(auc)
    pe,auc=estimate(np.ones(len(names)))
    rng=np.random.default_rng(seed);pe_draws=[];auc_draws=[]
    for _ in range(replicates):
        counts=np.zeros(len(population_names),int)
        for group in groups:counts[group]=np.bincount(rng.integers(len(group),size=len(group)),minlength=len(group))
        scope_counts=counts[scope_indices]
        if not scope_counts.sum():
            pe_draws.append(np.full(3,np.nan));auc_draws.append(np.full(3,np.nan));continue
        p,a=estimate(scope_counts);pe_draws.append(p);auc_draws.append(a)
    pe_draws=np.asarray(pe_draws);auc_draws=np.asarray(auc_draws)
    result=dict(photos=len(names),stego_versions=int(y.sum()),rules={})
    for j,key in enumerate(('a','b','c')):
        result['rules'][key]=dict(FA=float(fa[:,j].mean()),miss=float(miss[:,j].mean()),P_E=float(pe[j]),AUC=float(auc[j]),
            P_E_interval=interval(pe[j],pe_draws[:,j]),AUC_interval=interval(auc[j],auc_draws[:,j]))
    result['differences']={f'c-{key}':dict(P_E=interval(pe[2]-pe[j],pe_draws[:,2]-pe_draws[:,j]),
        AUC=interval(auc[2]-auc[j],auc_draws[:,2]-auc_draws[:,j])) for j,key in enumerate(('a','b'))}
    return result
