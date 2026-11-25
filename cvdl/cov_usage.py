"""Streaming block-logit RMS; descriptive validation context, no fitted scale."""
import numpy as np


class BlockMoments:
    def __init__(self):
        self.count=0;self.batch_sizes={}
        self.sum={k:np.zeros(2) for k in ('gap','covariance')}
        self.square={k:np.zeros(2) for k in self.sum}
        self.centred_square={k:np.zeros(2) for k in self.sum}
        self.margin_square={k:0. for k in self.sum}

    def add(self,gap,covariance):
        if gap.shape!=covariance.shape or gap.ndim!=2 or gap.shape[1]!=2 or not len(gap):raise ValueError('expected equal nonempty Bx2 block logits')
        n=len(gap);self.count+=n;self.batch_sizes[n]=self.batch_sizes.get(n,0)+1
        for k,tensor in (('gap',gap),('covariance',covariance)):
            x=tensor.detach().cpu().double().numpy()
            if not np.isfinite(x).all():raise ValueError('nonfinite block logits')
            self.sum[k]+=x.sum(0);self.square[k]+=(x*x).sum(0)
            self.centred_square[k]+=((x-x.mean(0))**2).sum(0)
            self.margin_square[k]+=float(((x[:,1]-x[:,0])**2).sum())

    def report(self):
        if not self.count:raise ValueError('no validation images')
        result={}
        for k in self.sum:
            result[k]=dict(raw_rms=float(np.sqrt(self.square[k].sum()/(2*self.count))),
                batch_centered_rms=float(np.sqrt(self.centred_square[k].sum()/(2*self.count))),
                dataset_centered_rms=float(np.sqrt(np.maximum(self.square[k]-self.sum[k]**2/self.count,0).sum()/(2*self.count))),
                margin_rms=float(np.sqrt(self.margin_square[k]/self.count)))
        return dict(images=self.count,batch_sizes=self.batch_sizes,blocks=result,
            definition='RMS over both classes and all images; bias excluded. Batch-centred subtracts each class mean within each actual batch, image-weighted. Dataset-centred subtracts each class full-list mean.')
