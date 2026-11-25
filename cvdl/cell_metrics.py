"""Small cell diagnostics and paired photo bootstrap; no model or I/O."""
import numpy as np

SEED=20261021
REPLICATES=2000


def fa_cutoff(negative_scores, target=.05):
    """Lowest float64 cutoff with <= target empirical FA, using score >= cutoff.

    Ties cannot split. Move just above the first excluded negative score; this
    also retains positives between adjacent negative scores. No test fitting.
    """
    scores=np.sort(np.asarray(negative_scores,dtype=np.float64).ravel())
    if not len(scores) or not np.isfinite(scores).all() or not 0<=target<1:
        raise ValueError('cutoff needs finite nonempty validation negatives and target in [0,1)')
    allowed=int(np.floor(target*len(scores)+1e-12))
    cutoff=float(np.nextafter(scores[len(scores)-allowed-1],np.inf))
    return cutoff,float((scores>=cutoff).mean())


def weighted_auc(labels, scores, weights=None):
    """Tied-score ROC AUC; missing classes are ineligible, never AUC zero."""
    labels=np.asarray(labels,dtype=bool);scores=np.asarray(scores,dtype=np.float64)
    weights=np.ones(len(labels)) if weights is None else np.asarray(weights,dtype=float)
    if labels.shape!=scores.shape or weights.shape!=scores.shape or not np.isfinite(scores).all():
        raise ValueError('invalid AUC arrays')
    order=np.argsort(scores,kind='stable');s=scores[order];y=labels[order];w=weights[order]
    starts=np.r_[0,np.flatnonzero(np.diff(s))+1]
    if not len(s):return None
    p=np.add.reduceat(w*y,starts);n=np.add.reduceat(w*~y,starts)
    if p.sum()==0 or n.sum()==0:return None
    return float(np.sum(p*(np.cumsum(n)-.5*n))/(p.sum()*n.sum()))


def photo_auc_bootstrap(labels, scores, photos, *, replicates=REPLICATES, seed=SEED):
    """Shared photo draws for all score columns. Keeps paired/repeated rows.

    scores is [rows,maps]. Sorting once makes the 2,000 resamples inexpensive.
    """
    y=np.asarray(labels,dtype=bool);scores=np.asarray(scores,dtype=float)
    if scores.ndim==1:scores=scores[:,None]
    photos=np.asarray(photos)
    if len(y)!=len(photos) or len(scores)!=len(y) or not len(y) or not np.isfinite(scores).all():
        raise ValueError('invalid bootstrap population')
    names,cluster=np.unique(photos,return_inverse=True)
    prepared=[]
    for s in scores.T:
        order=np.argsort(s,kind='stable');starts=np.r_[0,np.flatnonzero(np.diff(s[order]))+1]
        prepared.append((order,starts))
    def estimate(counts):
        w=counts[cluster];values=[]
        for order,starts in prepared:
            p=np.add.reduceat(w[order]*y[order],starts);n=np.add.reduceat(w[order]*~y[order],starts)
            values.append(np.sum(p*(np.cumsum(n)-.5*n))/(p.sum()*n.sum()) if p.sum() and n.sum() else np.nan)
        return np.asarray(values)
    point=estimate(np.ones(len(names)))
    rng=np.random.default_rng(seed);draws=[]
    for _ in range(replicates):
        counts=np.bincount(rng.integers(len(names),size=len(names)),minlength=len(names))
        draws.append(estimate(counts))
    draws=np.asarray(draws)
    return point,draws


def interval(point, draws):
    draws=np.asarray(draws);draws=draws[np.isfinite(draws)]
    if point is None or not np.isfinite(point) or not len(draws):return None
    lo,hi=np.quantile(draws,[.025,.975])
    return dict(point=float(point),low=float(lo),high=float(hi),valid_replicates=len(draws))


def localization(counts, valid_negative, scores, photos, *, replicates=REPLICATES,
                 difference_name='aux_minus_cam', draws_out=None):
    """Both true within-stego tile-mean AUC and pooled eligible-cell AUC.

    scores [tiles,maps,64]; only tiles with positive AND valid negative cells
    enter either summary. Clean partner scores never enter this calculation.
    """
    counts=np.asarray(counts).reshape(-1,64);neg=np.asarray(valid_negative).reshape(-1,64).astype(bool)
    pos=counts>0;eligible=pos.any(1)&neg.any(1)
    scores=np.asarray(scores).reshape(len(counts),-1,64)
    result=dict(tiles=len(counts),eligible_tiles=int(eligible.sum()),ineligible_tiles=int((~eligible).sum()),
        positive_cells=int(pos.sum()),valid_negative_cells=int(neg.sum()),ignored_cells=int((~pos&~neg).sum()),
        occupancy=float(pos.mean()) if len(pos) else None,maps=[])
    if not eligible.any():
        result['maps']=[dict(within_stego_auc=None,pooled_cell_auc=None) for _ in range(scores.shape[1])]
        return result
    p=pos[eligible];n=neg[eligible];s=scores[eligible];photos=np.asarray(photos)[eligible]
    within=np.array([[weighted_auc(pp[pp|nn],ss[pp|nn]) for ss in tile] for pp,nn,tile in zip(p,n,s)])
    names,clusters=np.unique(photos,return_inverse=True)
    rng=np.random.default_rng(SEED);draws=[]
    for _ in range(replicates):
        w=np.bincount(rng.integers(len(names),size=len(names)),minlength=len(names))[clusters]
        draws.append((within*w[:,None]).sum(0)/w.sum())
    draws=np.asarray(draws);point=within.mean(0)
    keep=p|n
    labels=p[keep];flat_scores=s.transpose(0,2,1)[keep]
    flat_photos=np.broadcast_to(photos[:,None],keep.shape)[keep]
    pooled,pooled_draws=photo_auc_bootstrap(labels,flat_scores,flat_photos,replicates=replicates)
    if draws_out is not None:
        draws_out.update(within_stego_auc=draws,pooled_cell_auc=pooled_draws)
    for i in range(s.shape[1]):
        result['maps'].append(dict(within_stego_auc=interval(point[i],draws[:,i]),pooled_cell_auc=interval(pooled[i],pooled_draws[:,i])))
    if s.shape[1]==2:
        result[difference_name]=dict(within_stego_auc=interval(point[1]-point[0],draws[:,1]-draws[:,0]),
            pooled_cell_auc=interval(pooled[1]-pooled[0],pooled_draws[:,1]-pooled_draws[:,0]))
    return result
