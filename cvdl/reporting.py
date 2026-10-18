"""Read saved evaluation/training artifacts for reports. No scoring or fitting."""
import hashlib
import json
import os
from pathlib import Path
import socket
import sys

import numpy as np
import pandas as pd

SUITE='standard-256-v1'
BINS=(.05,.10,.15,.35,.50)


def cpu_offline_guard():
    """Keep notebook networking local to Jupyter; deny external DNS/connections."""
    os.environ.update(CUDA_VISIBLE_DEVICES='',HIP_VISIBLE_DEVICES='',MPLBACKEND='Agg')
    def local(host):return host in ('localhost','127.0.0.1','::1',None)
    original=socket.getaddrinfo
    def resolve(host,*args,**kwargs):
        if not local(host):raise PermissionError('report notebook: external network disabled')
        return original(host,*args,**kwargs)
    connect=socket.socket.connect
    def guarded(sock,address):
        if isinstance(address,tuple) and not local(address[0]):raise PermissionError('report notebook: external network disabled')
        return connect(sock,address)
    socket.getaddrinfo=resolve;socket.socket.connect=guarded
    class NoInference:
        def find_spec(self,fullname,path=None,target=None):
            if fullname.split('.')[0] in {'torch','tensorflow','jax'} or fullname in {'cvdl.models','cvdl.runtime','cvdl.deployment','cvdl.metrics'}:
                raise ImportError('report notebook: inference and threshold fitting modules disabled')
    sys.meta_path.insert(0,NoInference())


def read_csv(path):
    if not Path(path).is_file():raise ValueError(f'missing saved output: {path}')
    return pd.read_csv(path,float_precision='round_trip')


def run_path(root,run):
    root=Path(root).resolve();path=Path(run)
    if not path.is_absolute():path=root/path if path.parts[0]=='runs' else root/'runs'/path
    path=path.resolve()
    if path.parent!=root/'runs' or not path.is_dir():raise ValueError(f'expected existing runs/<run>: {path}')
    return path


def evaluation_path(run,eid=None):
    base=Path(run)/'evaluation'/SUITE
    choices=sorted(p.name for p in base.iterdir() if p.is_dir()) if base.is_dir() else []
    if eid is None:
        if len(choices)!=1:raise ValueError(f'require --evaluation-id; available: {choices}')
        eid=choices[0]
    if not isinstance(eid,str) or len(eid)!=20 or any(c not in '0123456789abcdef' for c in eid):raise ValueError('invalid evaluation id')
    path=base/eid
    if not path.is_dir():raise ValueError(f'missing evaluation {eid}; available: {choices}')
    return path


class SavedReport:
    def __init__(self,root,run,eid=None):
        self.run=run_path(root,run);self.path=evaluation_path(self.run,eid)
        def js(relative):
            p=self.path/relative
            if not p.is_file():raise ValueError(f'missing or pending output: {p}')
            return json.loads(p.read_text())
        self.meta=js('evaluation_settings.json');self.summary=js('summary.json');self.calibration=js('validation/calibration.json')
        if self.meta.get('suite')!=SUITE or self.meta.get('evaluation_id')!=self.path.name or self.summary.get('evaluation_id')!=self.path.name:
            raise ValueError('evaluation identity mismatch')
        if self.meta.get('quick') or not self.calibration.get('deployment') or self.meta.get('validation_threshold',{}).get('status')!='complete':
            raise ValueError('missing/pending/non-deployment calibration')
        if self.summary.get('validation_threshold',{}).get('threshold_id')!=self.calibration['threshold_id']:
            raise ValueError('pending or mismatched summary calibration')
        raw=self.path/'validation/predictions.csv'
        if not raw.is_file() or hashlib.sha256(raw.read_bytes()).hexdigest()!=self.calibration['score_csv_sha256']:
            raise ValueError('saved validation score SHA mismatch')
        self.main=read_csv(self.path/'main_summary.csv');self.diagnostics=read_csv(self.path/'diagnostic_summary.csv')
        self.check_main(self.main)
        self.headline=self.main[self.main.category.isin(['overall','source'])].copy()
        self.reference_thresholds=read_csv(self.path/'validation/source_reference_thresholds.csv')
        if 'div2k' in set(self.reference_thresholds.source):raise ValueError('DIV2K reference is not permitted')
        self.source_reference=read_csv(self.path/'source_reference_summary.csv')
        self.predictions={}
        for panel in self.main.panel.unique():
            f=read_csv(self.path/'panels'/panel/'predictions.csv')
            if set(f['split']) & {'train','val'}:raise ValueError('non-test predictions')
            self.predictions[panel]=f
        self.hard=None
        if self.meta.get('hard_enabled'):
            if self.summary.get('hard_test',{}).get('status')!='complete':raise ValueError('pending hard test')
            self.hard=read_csv(self.path/'hard-test/hard_cells.csv');self.check_main(self.hard)
            if len(self.hard)!=32:raise ValueError('incomplete hard cells')
        self.higher=None
        if self.meta.get('higher_k'):
            self.higher=read_csv(self.path/'higher-k/main_summary.csv');self.check_main(self.higher)
        elif (self.path/'higher-k').exists():raise ValueError('unrequested/stale higher-k output')
        folders=list((self.run/'checkpoints').glob('*'))
        folders=[p for p in folders if p.is_dir()]
        if len(folders)!=1:raise ValueError('ambiguous or missing saved training records')
        self.training=folders[0]
        saved=self.training/'run_settings.json'
        self.training_values=json.loads(saved.read_text())['values'] if saved.exists() else {}
        old=self.training/'data_run_settings.json'
        if not old.is_file():raise ValueError('missing saved training data record')
        self.training_record=json.loads(old.read_text())
        metrics=list(self.training.glob('logs/**/metrics.csv'))
        if len(metrics)!=1:raise ValueError('missing or ambiguous epoch metrics CSV')
        self.epochs=read_csv(metrics[0]).dropna(subset=['epoch']).groupby('epoch',sort=True).last()
        self.lr=read_csv(self.training/'lr_trace.csv')
        if not (self.training/'training_curves.png').is_file():raise ValueError('missing saved training_curves.png')

    def check_main(self,frame):
        c=self.calibration
        if frame.empty or frame[['cutoff','FA','miss','P_E_at_validation_cutoff','AUC']].isna().any().any():raise ValueError('pending main metrics')
        if not ((frame.cutoff==c['cutoff']) & (frame.threshold_id==c['threshold_id']) & (frame.threshold_origin==c['threshold_origin'])).all():
            raise ValueError('main metrics do not use saved deployment threshold')

    @property
    def training_label(self):
        mode=self.training_values.get('checkpoint_validation',self.training_record.get('checkpoint_validation','standard'))
        if mode=='train_matched':
            tile=self.training_values.get('train_tile_size') or self.training_values.get('tile_size',512)
            return f'Original checkpoint-selection validation: train-matched, {tile}px, recipe {self.training_values.get("validation_recipe")}'
        tile=self.training_values.get('tile_size',self.training_record.get('tile_size',512))
        return f'Original checkpoint-selection validation: {mode}, {tile}px (NOT the new post-hoc calibration list)'

    def run_card(self):
        v=self.training_values or self.training_record.get('settings',{})
        model_path=self.training/'model_run_settings.json'
        model=json.loads(model_path.read_text()) if model_path.is_file() else {}
        card=dict(run=self.run.name,evaluation=self.path.name,checkpoint=Path(self.meta.get('checkpoint','not recorded')).name,
            model=v.get('model',model.get('model','not recorded')),front_end=v.get('front_end',model.get('front_end','not recorded')),
            norm=v.get('norm',model.get('norm','not recorded')),clip_t=v.get('clip_t',model.get('clip_t','not recorded')),
            epochs=v.get('epochs',model.get('max_epochs','not recorded')),lr=v.get('lr',model.get('lr','not recorded')),
            lr_drops=str(v.get('lr_drops',model.get('lr_drops','not recorded'))),checkpoint_sha256=self.calibration['checkpoint_sha256'],
            training_tile=v.get('train_tile_size') or v.get('tile_size',512),training_grid=v.get('train_grid','recorded'),
            source_mix=str(v.get('source_mix','full-source; no fixed mixture')),train_yaml=v.get('train_yaml',v.get('settings_yaml')),
            calibration_origin=self.calibration['threshold_origin'],calibration_recipe=self.calibration['recipe_id'],
            cutoff=self.calibration['cutoff'],validation_pairs=self.calibration['pairs'],test='256px overlap-last; k1-4 balanced')
        return pd.DataFrame([(k,str(value)) for k,value in card.items()],columns=['Field','Value'])

    def miss_tables(self):
        f=pd.concat([g[g.label==1].assign(panel=panel) for panel,g in self.predictions.items()],ignore_index=True)
        f['miss_event']=f.prob_stego.astype(np.float64)<np.float64(self.calibration['cutoff'])
        f['bpp_bin']=pd.cut(f.requested_bpp,bins=BINS,right=False,include_lowest=True)
        # The final bin includes exactly0.5; requested BPP, not effective BPP.
        labels=['[0.05,0.10)','[0.10,0.15)','[0.15,0.35)','[0.35,0.50]']
        code=np.searchsorted(BINS,f.requested_bpp,side='right')-1
        if not f.requested_bpp.between(.05,.5).all():raise ValueError('BPP outside frozen test range')
        f['bpp_bin']=[labels[min(int(i),3)] for i in code]
        result={}
        for name,keys in [('k',['panel','k']),('bpp',['panel','bpp_bin']),('k_bpp',['panel','k','bpp_bin'])]:
            result[name]=f.groupby(keys,observed=True,sort=True).miss_event.agg(pairs='size',miss='mean').reset_index()
        return result


def compatible(primary,reference):
    if primary.meta['recipe_sha256']!=reference.meta['recipe_sha256']:raise ValueError('reference has a different standard test recipe')
    if bool(primary.hard is not None)!=bool(reference.hard is not None):raise ValueError('reference hard-test coverage differs')
    if primary.hard is not None and primary.meta['hard_recipe_sha256']!=reference.meta['hard_recipe_sha256']:
        raise ValueError('reference has a different hard test recipe')


def show_table(frame):
    from IPython.display import display
    def base(c):return c.removesuffix('_primary').removesuffix('_reference')
    rates=[c for c in frame if base(c) in {'FA','miss','P_E_at_validation_cutoff','test_min_P_E','P_E_at_0_5','FA_at_0_5','miss_at_0_5'} or base(c).startswith(('FA_ci95_','miss_ci95_'))]
    display(frame.style.format({**{c:'{:.2%}' for c in rates},**{c:'{:.5f}' for c in frame if base(c)=='AUC'}}))


def show_figure(fig):
    import io
    import matplotlib.pyplot as plt
    from IPython.display import display,Image
    data=io.BytesIO();fig.savefig(data,format='png',bbox_inches='tight')
    display(Image(data=data.getvalue()));plt.close(fig)


def training_plots(report):
    import matplotlib.pyplot as plt
    from IPython.display import display,Image,Markdown
    e=report.epochs;fig,axes=plt.subplots(1,3,figsize=(15,4));x=e.index+1
    for name in ('train_loss','val_loss'):
        if name in e:axes[0].plot(x,e[name],label=name)
    for name in ('train_acc','val_acc'):
        if name in e:axes[1].plot(x,e[name],label=name)
    lr=report.lr.groupby('epoch').lr.last();axes[2].plot(lr.index+1,lr);axes[2].set_yscale('log')
    drops=lr.index[lr.lt(lr.shift()*(1-1e-8))]
    for ax,title in zip(axes,['Loss','Accuracy','Saved learning rate']):
        ax.set(title=title,xlabel='Epoch (1-based)');ax.grid(alpha=.2)
        for epoch in drops:ax.axvline(epoch+1,color='gray',ls='--',alpha=.6)
    axes[0].legend();axes[1].legend();fig.suptitle(report.training_label);fig.tight_layout();show_figure(fig)
    display(Markdown('**Original saved training plot (embedded copy):**'))
    display(Image(data=(report.training/'training_curves.png').read_bytes()))


def miss_plots(report):
    import matplotlib.pyplot as plt
    from IPython.display import display
    tables=report.miss_tables();show_table(tables['k']);show_table(tables['bpp'])
    panels=list(report.predictions);fig,axes=plt.subplots(1,len(panels),figsize=(5*len(panels),4),squeeze=False)
    for ax,panel in zip(axes[0],panels):
        g=tables['k_bpp'].query('panel == @panel').pivot(index='k',columns='bpp_bin',values='miss')
        im=ax.imshow(g,vmin=0,vmax=1,cmap='magma');ax.set(title=panel,xlabel='Requested bpp',ylabel='k')
        ax.set_xticks(range(len(g.columns)),g.columns,rotation=60,ha='right');ax.set_yticks(range(len(g.index)),g.index)
        for (i,j),v in np.ndenumerate(g.to_numpy()):ax.text(j,i,f'{v:.0%}',ha='center',va='center',color='white' if v<.6 else 'black')
    fig.colorbar(im,ax=axes.ravel().tolist(),label='Miss at saved validation cutoff');show_figure(fig)
    return tables


def hard_plots(report,reference=None):
    import matplotlib.pyplot as plt
    from IPython.display import display
    if report.hard is None:print('Hard test was disabled for this saved evaluation.');return
    for panel,g in report.hard.groupby('panel',sort=False):
        fig,ax=plt.subplots(figsize=(12,5));x=np.arange(len(g));width=.18 if reference else .32
        items=[(report.run.name,g,-width/2)]
        if reference:
            r=reference.hard.set_index('cell').loc[g.cell].reset_index();items=[(report.run.name,g,-1.5*width),(reference.run.name,r,.5*width)]
        for name,frame,offset in items:
            for j,(metric,color) in enumerate([('FA','tab:blue'),('miss','tab:orange')]):
                y=frame[metric].to_numpy();lo=frame[f'{metric}_ci95_low'].to_numpy();hi=frame[f'{metric}_ci95_high'].to_numpy()
                ax.bar(x+offset+j*width,y,width,label=f'{name}: {metric}',yerr=np.maximum(0,np.vstack([y-lo,hi-y])),capsize=2,alpha=.85 if name==report.run.name else .45,color=color)
        ax.set_xticks(x,[f'k{r.k} {r.condition}' for r in g.itertuples()],rotation=40,ha='right')
        ax.set(title=f'{panel}: FA and miss, Wilson 95% ranges',ylabel='Rate',ylim=(0,1));ax.legend(fontsize=8);fig.tight_layout();show_figure(fig)
