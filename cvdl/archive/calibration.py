"""Temperature scaling of the two-class logit difference. Pure numpy; existing outputs are never changed."""
import numpy as np
import pandas as pd
from cvdl.metrics import threshold, rates

TIE_EPS=1e-6


def sigmoid(x):
    x=np.asarray(x,dtype=np.float64)
    return np.where(x>=0,1/(1+np.exp(-np.abs(x))),np.exp(-np.abs(x))/(1+np.exp(-np.abs(x))))


def nll(logits,labels,T=1.0):
    z=np.asarray(logits,dtype=np.float64)/T
    y=np.asarray(labels)
    return float(np.mean(np.logaddexp(0,np.where(y==1,-z,z))))


def fit_temperature(logits,labels,low=1e-2,high=1e3,iterations=200):
    """Minimise NLL over T>0 (golden section on log T; NLL is unimodal in 1/T)."""
    a,b=np.log(low),np.log(high);g=(np.sqrt(5)-1)/2
    c,d=b-g*(b-a),a+g*(b-a)
    fc,fd=nll(logits,labels,np.exp(c)),nll(logits,labels,np.exp(d))
    for _ in range(iterations):
        if fc<fd:b,d,fd=d,c,fc;c=b-g*(b-a);fc=nll(logits,labels,np.exp(c))
        else:a,c,fc=c,d,fd;d=a+g*(b-a);fd=nll(logits,labels,np.exp(d))
    return float(np.exp((a+b)/2))


def ece(prob,labels,bins=15):
    """Expected calibration error of P(stego): equal-width bins, |mean label - mean prob| weighted."""
    p=np.asarray(prob,dtype=np.float64);y=np.asarray(labels,dtype=np.float64)
    edges=np.linspace(0,1,bins+1);index=np.clip(np.digitize(p,edges[1:-1]),0,bins-1)
    total=0.0
    for b in range(bins):
        m=index==b
        if m.any():total+=m.mean()*abs(y[m].mean()-p[m].mean())
    return float(total)


def tie_share(prob,labels):
    """Share of clean scores exactly 1.0 and above 1-1e-6."""
    clean=np.asarray(prob,dtype=np.float64)[np.asarray(labels)==0]
    return dict(clean=len(clean),at_one=float((clean==1.0).mean()),above_1_minus_1e6=float((clean>1-TIE_EPS).mean()))


def fa5(frame):
    """Validation 5% FA cutoff as the repository defines it: achieved FA and the miss rate there."""
    cutoff,fa=threshold(frame,'fa5')
    return dict(cutoff=cutoff,fa=fa,miss=rates(frame,cutoff)['miss'])


def logit_summary(logits,labels):
    z=np.asarray(logits,dtype=np.float64);clean=z[np.asarray(labels)==0]
    q=np.quantile(clean,[0,.5,.95,.99,1])
    return dict(clean_quantiles_0_50_95_99_100=[float(v) for v in q],abs_max=float(np.abs(z).max()),
                clean_above_17=float((clean>17).mean()),distinct_clean_logits=int(len(np.unique(clean))))


def summarise(frame,logits_col='logit',T=1.0):
    """Per-source before/after table for one split. `frame` has source, label, k, prob_stego (saved) and logit."""
    out={}
    for source,part in [('all',frame),*frame.groupby('source')]:
        y=part.label.to_numpy();z=part[logits_col].to_numpy(dtype=np.float64)
        before=part.prob_stego.to_numpy(dtype=np.float64);after=sigmoid(z/T)
        row=dict(pairs=int((y==1).sum()),nll=[nll(z,y,1.0),nll(z,y,T)],ece=[ece(before,y),ece(after,y)],
                 ties_before=tie_share(before,y),ties_after=tie_share(after,y),logits=logit_summary(z,y))
        if set(np.unique(y))=={0,1}:
            b=part.copy();b['prob_stego']=before;a=part.copy();a['prob_stego']=after
            row['fa5_before']=fa5(b);row['fa5_after']=fa5(a)
        out[str(source)]=row
    return out
