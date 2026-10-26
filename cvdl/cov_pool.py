"""GAP + regularized covariance head primitives; independent iSQRT-COV math.

Inspired by Deng et al. IH&MMSec2019, public implementation a1db8a6a.
Five Newton-Schulz iterations approximate, not exactly compute, a matrix root.
"""
import torch

MODEL='xunet_wide_gapcov32'
SPEC=dict(version='gapcov32-v1',gap_features=256,reduced_features=32,cov_features=528,
    covariance_divisor='M',ridge_relative=1e-4,ridge_absolute=1e-12,iterations=5,
    sqrt='trace_normalized_newton_schulz_with_compensation',triangle='upper_unweighted',
    reduction_bias=False,post_normalization='none',branch_scaling='none',
    classifier='default_Linear784_to_2',attribution='gap_cam_plus_M_centered_cov_gradient')


def regularized_covariance(h,*,ridge_relative=1e-4,ridge_absolute=1e-12):
    """Population covariance. Dtype-preserving core allows independent FP64 tests."""
    h=h.flatten(2);u=h-h.mean(-1,keepdim=True)
    c=u.bmm(u.transpose(1,2))/h.shape[-1]
    c=(c+c.transpose(1,2))*.5
    k=h.shape[1];eye=torch.eye(k,device=h.device,dtype=h.dtype)[None]
    ridge=ridge_relative*c.diagonal(dim1=-2,dim2=-1).sum(-1)/k+ridge_absolute
    return c+ridge[:,None,None]*eye


def sqrt_newton_schulz(c,iterations=5):
    """Fixed-step trace-normalized approximation with amplitude compensation."""
    trace=c.diagonal(dim1=-2,dim2=-1).sum(-1)[:,None,None]
    y=c/trace;z=torch.eye(c.shape[-1],device=c.device,dtype=c.dtype)[None].expand_as(c)
    eye=z
    for _ in range(iterations):
        t=.5*(3*eye-z.bmm(y))
        y,z=y.bmm(t),t.bmm(z)  # Both updates use the old y/z.
    s=y*trace.sqrt()
    return (s+s.transpose(1,2))*.5


def descriptor(h):
    c=regularized_covariance(h)
    s=sqrt_newton_schulz(c)
    indices=torch.triu_indices(s.shape[-1],s.shape[-1],device=s.device)
    return s[:,indices[0],indices[1]]


def branch_statistics(network,images):
    """Probe-only raw and batch-centred RMS, without changing any state/scales."""
    from torch.nn import functional as F
    with torch.no_grad():
        gap,cov=network.pooled_parts(network.features(images.float()))
        g=F.linear(gap,network.classifier.weight[:,:256])
        c=F.linear(cov,network.classifier.weight[:,256:])
        def rms(x):return float(x.square().mean().sqrt())
        def ratio(a,b):return max(a,b)/min(a,b) if min(a,b)>0 else (None if max(a,b)>0 else 1.)
        result=dict(gap_rms=rms(gap),cov_rms=rms(cov),gap_logit_rms=rms(g),cov_logit_rms=rms(c),
            gap_centered_logit_rms=rms(g-g.mean(0)),cov_centered_logit_rms=rms(c-c.mean(0)),
            classifier_bias_rms=rms(network.classifier.bias))
        result['feature_ratio']=ratio(result['gap_rms'],result['cov_rms'])
        result['logit_ratio']=ratio(result['gap_logit_rms'],result['cov_logit_rms'])
        result['ratio_above_10']=any(v is None or v>10 for v in (result['feature_ratio'],result['logit_ratio']))
        return result
