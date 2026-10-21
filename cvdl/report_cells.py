"""Optional cell-report views: saved JSON and PNG only, never model inference."""
import json
import math
from pathlib import Path

import pandas as pd
from .report_charts import PANELS,SOURCES,text


def cell_tables(folder,reference_dir=None):
    folder=Path(folder)
    def read(path):
        if not path.is_file():raise ValueError(f'missing saved cell output: {path}')
        return json.loads(path.read_text())
    summary=read(folder/'cells/summary.json')
    paired=read(folder/'aux-minus-control-cam.json')['panels']
    tiles=read(folder/'tile-comparison.json')
    figure=folder/'cells/cells.png'
    if not figure.is_file():raise ValueError(f'missing saved cell figure: {figure}')
    localized={r['panel']:r for r in summary['localization'] if r['split']=='test' and r['placement']=='contiguous'}
    def auc(value):return f"{value['point']:.3f} [{value['low']:.3f}, {value['high']:.3f}]"
    def difference(value):return f"{value['point']:+.4f} [{value['low']:+.4f}, {value['high']:+.4f}]"
    rows=[]
    for panel,item in localized.items():
        rows.append({'Test panel':'All panels' if panel=='pooled' else SOURCES.get(panel,PANELS.get(panel,panel)),
            'This model: cell head':auc(item['maps']['aux']['within_stego_auc']),
            'This model: CAM':auc(item['maps']['cam']['within_stego_auc']),
            'Control: CAM':auc(paired[panel]['maps']['control_cam']['within_stego_auc'])})
    differences=pd.DataFrame([
        {'Comparison':'Cell head − control CAM','AUC change (95% interval)':difference(paired['pooled']['run13_aux_minus_control_cam']['within_stego_auc'])},
        {'Comparison':'Cell head − same-model CAM','AUC change (95% interval)':difference(localized['pooled']['aux_minus_cam']['within_stego_auc'])}])
    map_rows=[('This model: cell head',summary['maps']['aux']),('This model: CAM',summary['maps']['cam'])]
    if reference_dir is not None:
        reference=read(Path(reference_dir)/'summary.json')
        control=next(r for r in reference['localization'] if r['split']=='test' and r['panel']=='pooled' and r['placement']=='contiguous')
        if not math.isclose(control['maps']['cam']['within_stego_auc']['point'],paired['pooled']['maps']['control_cam']['within_stego_auc']['point'],rel_tol=1e-10,abs_tol=1e-12):
            raise ValueError('control cell summary differs from saved paired comparison')
        map_rows.append(('Control: CAM',reference['maps']['cam']))
    rates=[];recalls=[]
    for label,values in map_rows:
        test=next(r for r in values['test'] if r['panel']=='pooled' and r['placement']=='all')
        rates.append({'Map':label,'Cell cut-off':f"{values['cutoff']:.3f}",
            'Validation false alarms':f"{100*values['validation_FA']:.1f}%",
            'Test false alarms':f"{100*test['FA']:.1f}%",'Test recall':f"{100*test['recall']:.1f}%"})
        recalls.append(round(100*test['recall']))
    note=(f'At these strict validation cut-offs, recall is about {recalls[0]}% for every shown map.' if len(set(recalls))==1
        else 'Recall is the fraction of changed cells above the strict validation cut-off.')
    standard=tiles['standard'];hard=tiles['hard_target'];noise=tiles['noisy_clean']
    tile_rows=[{'Measure':'Standard overall error','This model':f"{100*standard['candidate']['P_E']:.1f}%",'Control':f"{100*standard['control']['P_E']:.1f}%",
        'Change':f"{100*standard['candidate_minus_control']:+.1f} percentage points"},
        {'Measure':'Hard small-message AUC','This model':f"{hard['candidate']['point']:.3f}",'Control':f"{hard['control']['point']:.3f}",
        'Change':difference(hard['candidate_minus_control'])},
        {'Measure':'Noisy clean-image false alarms','This model':f"{100*noise['candidate']:.1f}%",'Control':f"{100*noise['control']:.1f}%",
        'Change':f"{100*noise['candidate_minus_control']:+.1f} percentage points"}]
    return dict(figure=figure,auc=pd.DataFrame(rows),differences=differences,rates=pd.DataFrame(rates),tiles=pd.DataFrame(tile_rows),recall_note=note)


def show_cell_report(folder,reference_dir=None):
    from IPython.display import display,HTML,Image
    tables=cell_tables(folder,reference_dir)
    display(Image(data=tables['figure'].read_bytes()))
    def table(name):display(HTML(tables[name].to_html(index=False,border=0)))
    table('auc')
    text('AUC and 95% photo-bootstrap intervals compare changed and valid unchanged cells **within each message tile**, then average the tile AUCs. '
         'CAM applies the tile-scoring layer at each grid location; the cell head was trained with changed-cell targets. '
         'Each square summarises a 48×48-pixel patch, with 32 pixels between squares; these are not exact causal pixel maps.')
    table('differences')
    text('Paired intervals use the same photos for both maps: 2,000 resamples. Positive changes mean higher AUC.')
    table('rates');text(tables['recall_note'])
    text('Cell cut-offs use validation only, at at most 5% false alarms. Test rates include clean partners and valid unchanged message cells; '
         'recall includes both contiguous and scattered random messages.')
    table('tiles')
    text('Each model uses its own saved deployment cut-off. The hard target is contiguous k=1, 0.05–0.15 bits per pixel, '
         'on both ALASKA camera groups and LSDIR. Its change includes the paired 95% photo-bootstrap interval. '
         'Noisy-image false alarms count distinct clean tiles. Changes are calculated before rounding; AUC changes use four decimals.')
