"""Optional cell-report views: saved JSON and PNG only, never model inference."""
import json
import math
from pathlib import Path

import pandas as pd
from .report_charts import PANELS,SOURCES,text


def _read(path):
    if not path.is_file():raise ValueError(f'missing saved cell output: {path}')
    return json.loads(path.read_text())


def _map_label(summary,key):
    """Plain map name from the saved summary; never guessed from the model."""
    if key=='aux':return 'cell head'
    if key=='cam':return 'CAM'
    if key=='class_score':
        return 'attribution' if summary.get('map_kind')=='attribution' else 'class-score map'
    raise ValueError(f'unknown saved cell map: {key}')


def _geometry(summary,key):
    meta=summary.get('map_metadata',{}).get(key,{})
    size=meta.get('receptive_field',48)
    return f'{size}×{size}'+(' (whole tile)' if meta.get('global_context') else '')


def general_cell_tables(folder,reference_dir=None):
    """Any saved cell_eval output: its own maps, own cut-offs, optional reference.

    Reads summary.json and cells.png only. No new statistics: the reference
    columns are side by side, not a paired test.
    """
    folder=Path(folder)
    summary=_read(folder/'cells/summary.json')
    figure=folder/'cells/cells.png'
    if not figure.is_file():raise ValueError(f'missing saved cell figure: {figure}')
    reference=None
    if reference_dir is not None:
        reference_dir=Path(reference_dir)
        reference=_read(reference_dir/'summary.json' if (reference_dir/'summary.json').is_file() else reference_dir/'cells/summary.json')
        if reference.get('selection_counts')!=summary.get('selection_counts'):
            raise ValueError('reference cell summary uses a different frozen fixture')
    def auc(value):return '—' if value is None else f"{value['point']:.3f} [{value['low']:.3f}, {value['high']:.3f}]"
    def difference(value):return f"{value['point']:+.4f} [{value['low']:+.4f}, {value['high']:+.4f}]"
    def localized(saved):return {r['panel']:r for r in saved['localization'] if r['split']=='test' and r['placement']=='contiguous'}
    own=localized(summary);ref=localized(reference) if reference else {}
    own_maps=[k for k in ('aux','cam','class_score') if k in summary['maps']]
    ref_maps=[k for k in ('aux','cam','class_score') if reference and k in reference['maps']]
    columns=[('This model: '+_map_label(summary,k),own,k) for k in own_maps]+[('Reference: '+_map_label(reference,k),ref,k) for k in ref_maps]
    rows=[]
    for panel in own:
        row={'Test panel':'All panels' if panel=='pooled' else SOURCES.get(panel,PANELS.get(panel,panel))}
        for label,saved,key in columns:
            item=saved.get(panel)
            row[label]=auc(item['maps'][key]['within_stego_auc'] if item else None)
        rows.append(row)
    differences=[]
    for name,label in (('aux_minus_cam','Cell head − same-model CAM'),('aux_minus_class_score','Cell head − same-model attribution')):
        if name in own['pooled']:
            differences.append({'Comparison':label,'AUC change (95% interval)':difference(own['pooled'][name]['within_stego_auc'])})
    rates=[]
    for prefix,saved,keys in (('This model',summary,own_maps),('Reference',reference,ref_maps)):
        for key in keys:
            values=saved['maps'][key]
            test=next(r for r in values['test'] if r['panel']=='pooled' and r['placement']=='all')
            rates.append({'Map':f'{prefix}: {_map_label(saved,key)}','Patch seen':_geometry(saved,key),'Cell cut-off':f"{values['cutoff']:.3f}",
                'Validation false alarms':f"{100*values['validation_FA']:.1f}%",
                'Test false alarms':f"{100*test['FA']:.1f}%",'Test recall':f"{100*test['recall']:.1f}%"})
    return dict(figure=figure,auc=pd.DataFrame(rows),differences=pd.DataFrame(differences),rates=pd.DataFrame(rates),
        has_reference=reference is not None)


def cell_tables(folder,reference_dir=None):
    folder=Path(folder)
    read=_read
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


def show_general_cell_report(folder,reference_dir=None):
    from IPython.display import display,HTML,Image
    tables=general_cell_tables(folder,reference_dir)
    display(Image(data=tables['figure'].read_bytes()))
    def table(name):display(HTML(tables[name].to_html(index=False,border=0)))
    table('auc')
    text('AUC and 95% photo-bootstrap intervals compare changed and valid unchanged cells **within each contiguous message tile**, then average the tile AUCs. '
         'The cell head was trained with changed-cell targets and sees a 48×48-pixel patch per square. '
         'CAM applies the tile-scoring layer at each grid location. Attribution is worked out afterwards from the tile score and depends on the whole 256×256 tile. '
         'Squares are 32 pixels apart; these are not exact causal pixel maps.')
    if tables['has_reference']:
        text('Reference columns come from its own saved run on the same frozen tiles. They are shown side by side; this is not a paired test.')
    if len(tables['differences']):
        table('differences')
        text('Same-model differences use the same photos for both maps: 2,000 paired resamples. Positive changes favour the cell head.')
    table('rates')
    text('Each map uses its own validation cut-off, at at most 5% false alarms. Test rates include clean partners and valid unchanged message cells; '
         'recall is the fraction of changed cells above the cut-off and includes contiguous and scattered random messages.')


def show_cell_report(folder,reference_dir=None):
    from IPython.display import display,HTML,Image
    if not (Path(folder)/'aux-minus-control-cam.json').is_file():
        return show_general_cell_report(folder,reference_dir)
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
