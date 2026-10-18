"""Human-facing views of saved results. No model or threshold fitting imports."""
import html
import io
import json
import textwrap

import numpy as np
import pandas as pd

PANELS={'standard':'Standard test', 'alaska-seen':'ALASKA (cameras seen in training)',
        'alaska-unseen':'ALASKA (cameras not seen in training)', 'lsdir':'LSDIR'}
SOURCES={'raise':'RAISE','div2k':'DIV2K','alaska':'ALASKA','lsdir':'LSDIR'}
CONDITIONS={'all':'All message sizes','low':'Small message','noisy':'Noisy images','low_noisy':'Small + noisy'}
ORIGINS={'train_matched':'Threshold set on its own validation',
         'post_hoc_256':'Threshold set after training',
         'if_source_known_reference':'Source-known reference only'}
COLUMNS={'panel':'Test panel','category':'Group','value':'Subset','source':'Source','k':'Embedding strength k',
    'condition':'Image / message group','bpp_bin':'Message size (bpp)','pairs':'Image pairs','images':'Images',
    'photos':'Photos','distinct_photos':'Distinct photos','distinct_clean_tiles':'Distinct clean tiles',
    'max_tile_use':'Maximum uses of a tile','FA':'False alarms %','miss':'Misses %',
    'P_E_at_validation_cutoff':'Overall error %','AUC':'AUC',
    'test_min_P_E':'Test-optimised error % (diagnostic)','P_E_at_0_5':'Error at cut-off 0.500 % (diagnostic)',
    'FA_at_0_5':'False alarms at cut-off 0.500 %','miss_at_0_5':'Misses at cut-off 0.500 %',
    'cutoff':'Validation cut-off','test_min_cutoff':'Test-optimised cut-off (diagnostic)',
    'FA_events':'False alarms (count)','miss_events':'Misses (count)',
    'overlap_standard_distinct_tiles':'Tiles shared with standard test',
    'overlap_standard_pairs':'Pairs shared with standard test'}
for metric,label in [('FA','False alarms'),('miss','Misses')]:
    for edge in ('low','high'):
        COLUMNS[f'{metric}_ci95_{edge}']=f'{label} 95% {edge} %'
        COLUMNS[f'{metric}_at_0_5_ci95_{edge}']=f'{label} at 0.500: 95% {edge} %'
    COLUMNS[f'{metric}_events_at_0_5']=f'{label} at 0.500 (count)'
RATES={'FA','miss','P_E_at_validation_cutoff','test_min_P_E','P_E_at_0_5','FA_at_0_5','miss_at_0_5'}
RATES.update(k for k in COLUMNS if '_ci95_' in k)


def deduplicate(frame):
    """Keep overall only when a panel has one source; retain standard breakdown."""
    if not {'panel','category','value'}<=set(frame):return frame.copy()
    keep=pd.Series(True,index=frame.index)
    for _,g in frame.groupby('panel',sort=False):
        sources=g[g.category=='source']
        if len(sources)==1 and (g.category=='overall').any():
            keep.loc[sources.index]=False
            if (g.category=='k').any():keep.loc[g[g.category.isin(['source/k','source_k'])].index]=False
    return frame.loc[keep].copy()


def plain(value):
    value=str(value)
    if value in PANELS:return PANELS[value]
    if value in SOURCES:return SOURCES[value]
    if value in CONDITIONS:return CONDITIONS[value]
    if value in ORIGINS:return ORIGINS[value]
    parts=value.split('/')
    if len(parts)==2 and parts[0] in SOURCES and parts[1].isdigit():return f'{SOURCES[parts[0]]}, k = {parts[1]}'
    groups={'overall':'All images','source':'Source','source_k':'Source and strength','source/k':'Source and strength','k':'Embedding strength',
            'all':'All images','bpp':'Message size'}
    return groups.get(value,value.replace('_',' ').replace('/',' / '))


def display_frame(frame):
    """Explicit column allowlist: artifact ids/paths never enter rendered tables."""
    frame=deduplicate(frame)
    result={}
    for col in frame:
        if col not in COLUMNS:continue
        if col in RATES:values=frame[col].map(lambda x:f'{100*x:.1f}%' if pd.notna(x) else 'Not available')
        elif col=='AUC':values=frame[col].map(lambda x:f'{x:.3f}' if pd.notna(x) else 'Not available')
        elif col in {'cutoff','test_min_cutoff'}:values=frame[col].map(lambda x:f'{x:.3f}' if pd.notna(x) else 'Not available')
        elif pd.api.types.is_numeric_dtype(frame[col]):values=frame[col].map(lambda x:f'{x:g}' if pd.notna(x) else 'Not available')
        else:values=frame[col].map(lambda x:plain(x) if pd.notna(x) else '')
        result[COLUMNS[col]]=values.to_numpy()
    return pd.DataFrame(result)


def show_table(frame):
    from IPython.display import display,HTML
    display(HTML(display_frame(frame).to_html(index=False,border=0)))


def show_figure(fig):
    import matplotlib.pyplot as plt
    from IPython.display import display,Image
    data=io.BytesIO();fig.savefig(data,format='png',dpi=120,bbox_inches='tight')
    display(Image(data=data.getvalue()));plt.close(fig)


def style():
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':11,'axes.titlesize':13,'axes.labelsize':11,
        'xtick.labelsize':10,'ytick.labelsize':10,'legend.fontsize':10})
    return plt


def text(message):
    from IPython.display import display,Markdown
    display(Markdown(message))


def panel_rows(report):return report.main[report.main.category=='overall'].copy()


def headline(report):
    rows=panel_rows(report).set_index('panel')
    def odds(rate):return 'no errors' if rate<=0 else f'about 1 in {max(1,round(1/rate))} wrong'
    lines=[f'- **{PANELS[p]}:** {r.P_E_at_validation_cutoff:.1%} overall error ({odds(r.P_E_at_validation_cutoff)}).' for p,r in rows.iterrows()]
    if 'standard' in rows.index:lines.append(f'- **Standard-test AUC:** {rows.loc["standard","AUC"]:.3f} (0 to 1).')
    text('\n'.join(lines));text(ORIGINS[report.calibration['threshold_origin']]+'. One saved cut-off is used throughout.')


def training_card(report):
    p=report.training/'model_run_settings.json';model=json.loads(p.read_text()) if p.is_file() else {}
    v={**model,**report.training_record.get('settings',{}),**report.training_values}
    sources=v.get('sources',[]);mix=v.get('source_mix')
    source_text=', '.join(f'{SOURCES.get(k,k)} {float(w):.0%}' for k,w in mix.items()) if isinstance(mix,dict) else ', '.join(SOURCES.get(k,k) for k in sources) or 'Not recorded'
    front={'kv':'Fixed image-residual filter (KV)','srm':'Rich residual filters (SRM)'}.get(v.get('front_end'),'Not recorded')
    model_name={'xunet_wide':'Wide XuNet','xunet_v4':'XuNet'}.get(v.get('model'),plain(v.get('model','Not recorded')))
    return pd.DataFrame([('Model',model_name),('Front end',front),('Epochs',str(v.get('epochs',v.get('max_epochs','Not recorded')))),
        ('Training tile',f'{v.get("train_tile_size") or v.get("tile_size",512)} pixels'),('Training sources',source_text),
        ('Learning rate',f'{v["lr"]:g}' if 'lr' in v else 'Not recorded'),
        ('Training grid',plain(v.get('train_grid',v.get('grid','Not recorded')))),
        ('Checkpoint-selection validation',report.training_label.split(': ',1)[1])],columns=['Field','Value'])


def run_card(report,reference=None):
    card=training_card(report)
    text('\n'.join(f'- **{r.Field}:** {r.Value}' for r in card.iloc[:5].itertuples()))
    if reference is None:text('No reference model supplied.')
    else:
        other=training_card(reference).set_index('Field').Value
        text(f'**Reference:** {other["Model"]}; {other["Front end"]}; {other["Epochs"]} epochs.')
        differences=[f'{r.Field}: {r.Value} (reference: {other[r.Field]})' for r in card.itertuples() if r.Value!=other[r.Field]]
        text('**Different from reference:** '+('; '.join(differences) if differences else 'No difference in the listed training fields.')+'.')


def training_plots(report):
    plt=style();e=report.epochs;x=e.index+1
    fig,axes=plt.subplots(1,2,figsize=(12,4.8))
    lr=report.lr.groupby('epoch').lr.last();drops=lr.index[lr.lt(lr.shift()*(1-1e-8))]+1
    for ax,metric,title,ylabel,scale in [(axes[0],'loss','Training and validation loss','Loss',1),
            (axes[1],'acc','Training and validation accuracy','Correct predictions (%)',100)]:
        count=0
        for prefix,label in [('train','Training'),('val','Original validation')]:
            key=f'{prefix}_{metric}'
            if key in e and e[key].notna().any():ax.plot(x,e[key]*scale,label=label);count+=1
        for drop in drops:ax.axvline(drop,color='gray',ls='--',alpha=.6)
        ax.set(title=title,xlabel='Epoch',ylabel=ylabel);ax.grid(alpha=.2)
        if count>1:ax.legend()
    fig.suptitle('How training changed over time')
    caption=report.training_label+'. Dashed lines mark learning-rate drops.'
    fig.text(.5,.02,textwrap.fill(caption,105),ha='center',va='bottom',fontsize=10)
    fig.tight_layout(rect=(0,.14,1,.94));show_figure(fig)


def panel_plot(report):
    plt=style();g=panel_rows(report);y=np.arange(len(g));fig,ax=plt.subplots(figsize=(11,5))
    for metric,label,offset,color in [('FA','False alarms',-.19,'#377eb8'),('miss','Misses',.19,'#e88824')]:
        bars=ax.barh(y+offset,100*g[metric],height=.36,label=label,color=color)
        ax.bar_label(bars,labels=[f'{v:.1%}' for v in g[metric]],padding=4,fontsize=10)
    ax.set_yticks(y,[textwrap.fill(PANELS[p],25) for p in g.panel]);ax.invert_yaxis()
    ax.set(xlim=(0,110),xlabel='Images in error (%)',ylabel='Test panel',title='False alarms and missed messages by test panel')
    ax.set_xticks(range(0,101,20));ax.legend(loc='lower right');fig.tight_layout();show_figure(fig)


def miss_plots(report):
    plt=style();tables=report.miss_tables();fig,ax=plt.subplots(figsize=(11,5.4))
    for panel,g in tables['k'].groupby('panel',sort=False):ax.plot(g.k,100*g['miss'],marker='o',label=PANELS[panel])
    ax.set(title='Missed hidden messages by embedding strength',xlabel='Embedding strength k',ylabel='Misses (%)',ylim=(0,100))
    ax.set_xticks(sorted(tables['k'].k.unique()));ax.grid(alpha=.2)
    if tables['k'].panel.nunique()>1:ax.legend(loc='upper center',bbox_to_anchor=(.5,-.18),ncol=2,frameon=False)
    fig.tight_layout();show_figure(fig)
    panels=list(report.predictions);fig,axes=plt.subplots(2,2,figsize=(12,9),layout='constrained')
    for ax,panel in zip(axes.flat,panels):
        g=tables['k_bpp'].query('panel == @panel').pivot(index='k',columns='bpp_bin',values='miss')
        im=ax.imshow(g,vmin=0,vmax=1,cmap='Blues',aspect='auto')
        ax.set(title=textwrap.fill(PANELS[panel],36),xlabel='Requested message size (bits per pixel)',ylabel='Embedding strength k')
        ax.set_xticks(range(len(g.columns)),[v.replace(',','–').strip('[]()') for v in g.columns],rotation=25,ha='right')
        ax.set_yticks(range(len(g.index)),g.index)
        for (i,j),v in np.ndenumerate(g.to_numpy()):
            if np.isfinite(v):ax.text(j,i,f'{v:.1%}',ha='center',va='center',color='white' if v>.6 else 'black',fontsize=10)
    for ax in list(axes.flat)[len(panels):]:ax.set_visible(False)
    fig.suptitle('Missed hidden messages by strength and size — darker means more misses')
    fig.colorbar(im,ax=list(axes.flat),label='Misses (%)',ticks=[0,.25,.5,.75,1],format=lambda x,pos:f'{100*x:.0f}')
    show_figure(fig);return tables


def hard_plots(report):
    if report.hard is None:text('Hard test was disabled for this saved evaluation.');return
    plt=style()
    for panel,g in report.hard.groupby('panel',sort=False):
        fig,ax=plt.subplots(figsize=(13,5.4));x=np.arange(len(g));width=.36
        for metric,label,offset,color in [('FA','False alarms',-.18,'#377eb8'),('miss','Misses',.18,'#e88824')]:
            y=100*g[metric].to_numpy();lo=100*g[f'{metric}_ci95_low'].to_numpy();hi=100*g[f'{metric}_ci95_high'].to_numpy()
            ax.bar(x+offset,y,width,label=label,color=color,yerr=np.maximum(0,np.vstack([y-lo,hi-y])),capsize=3)
            # Values are above the ranges, not on top of the range lines.
            for pos,value,upper in zip(x+offset,y,hi):ax.text(pos,upper+1.8,f'{value:.1f}',ha='center',va='bottom',fontsize=9)
        ax.axhline(50,color='#555555',ls='--',label='50% coin-flip level')
        ax.set_xticks(x,[f'k = {r.k}\n'+CONDITIONS[r.condition].replace('Small + noisy','Small +\nnoisy') for r in g.itertuples()])
        ax.set(title=f'Hard images: false alarms and misses — {PANELS[panel]}',ylabel='Images in error (%)',ylim=(0,118),xlabel='Embedding strength and image / message group')
        ax.set_yticks(range(0,101,20));ax.legend(loc='upper center',bbox_to_anchor=(.5,1),ncol=3,frameon=False)
        fig.tight_layout();show_figure(fig)


def comparison_rows(report,reference):
    keys=['panel','category','value']
    return report.headline.merge(reference.headline,on=keys,suffixes=('_primary','_reference'),validate='one_to_one')


def change_label(value):
    if value==0:return '0.0 points unchanged'
    direction='better' if value<0 else 'worse'
    if round(abs(value),1)==0:return f'<0.1 points {direction}'
    return f'{value:+.1f} points {direction}'


def comparison_plot(report,reference):
    if reference is None:text('No reference model supplied; comparison skipped.');return
    plt=style();g=comparison_rows(report,reference);g=g[g.category=='overall']
    delta=100*(g.P_E_at_validation_cutoff_primary-g.P_E_at_validation_cutoff_reference)
    fig,ax=plt.subplots(figsize=(11,5));y=np.arange(len(g));limit=max(1,float(abs(delta).max()))
    ax.barh(y,delta,color=['#278348' if v<0 else '#be3838' if v>0 else 'gray' for v in delta])
    for i,v in enumerate(delta):
        label=change_label(v)
        ax.text(v+(.04*limit if v>=0 else -.04*limit),i,label,ha='left' if v>=0 else 'right',va='center',fontsize=11)
    ax.axvline(0,color='black',lw=1);ax.set_yticks(y,[textwrap.fill(PANELS[p],25) for p in g.panel]);ax.invert_yaxis()
    ax.set(xlim=(-1.8*limit,1.8*limit),xlabel='Change in overall error (percentage points)',ylabel='Test panel',title='Change in overall error compared with the reference')
    fig.tight_layout();show_figure(fig)
    text(f'**This model:** {ORIGINS[report.calibration["threshold_origin"]]}. **Reference:** {ORIGINS[reference.calibration["threshold_origin"]]}. Each keeps its own saved cut-off.')


def source_lines(report):
    refs=report.source_reference
    if not {'panel','source','P_E_at_validation_cutoff'}<=set(refs):raise ValueError('incomplete saved source-known metrics')
    if 'div2k' in set(refs.source):raise ValueError('DIV2K reference is not permitted')
    if 'k' in refs:refs=refs[refs.k.astype(str)=='all']
    lines={}
    for r in refs.itertuples():
        match=report.main[(report.main.panel==r.panel)&(report.main.category=='source')&(report.main.value==r.source)]
        if match.empty and r.panel!='standard':match=panel_rows(report).query('panel == @r.panel')
        if len(match)!=1:raise ValueError('source-known reference does not match a saved source row')
        label=SOURCES.get(r.source,r.source)
        group={'alaska-seen':'cameras seen in training: ','alaska-unseen':'cameras not seen in training: '}.get(r.panel,'')
        lines.setdefault(label,[]).append(f'{group}{match.iloc[0].P_E_at_validation_cutoff:.1%} → {r.P_E_at_validation_cutoff:.1%}')
    text('\n'.join(f'- **{label}:** '+('; '.join(values))+' overall error if the source were known.' for label,values in lines.items()) if lines else 'No source-known reference was saved.')


def appendix(report,reference=None):
    from IPython.display import display,HTML
    blocks=[]
    for label,item in [('This model',report),('Reference model',reference)]:
        if item is None:continue
        tables=[('Deployment results',item.main),('Diagnostic results only',item.diagnostics)]
        tables.extend((f'Misses by {label}',f) for label,f in item.miss_tables().items())
        if item.hard is not None:tables.append(('Hard-image results',item.hard))
        if item.higher is not None:tables.append(('Higher-strength results (k 5–8)',item.higher))
        tables.extend([('Source-known reference results only',item.source_reference),('Source-known reference cut-offs',item.reference_thresholds)])
        body=''.join(f'<h4>{html.escape(plain(title))}</h4><div style="overflow-x:auto">{display_frame(frame).to_html(index=False,border=0)}</div>' for title,frame in tables)
        blocks.append(f'<details><summary>{label}: full tables</summary>{body}</details>')
    display(HTML('<details><summary>Appendix — full tables (click to expand)</summary>'+''.join(blocks)+'</details>'))
