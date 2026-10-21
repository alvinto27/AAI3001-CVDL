import base64
import hashlib
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import pandas as pd
from cvdl.reporting import SavedReport,evaluation_path,compatible
from cvdl.report_charts import display_frame,deduplicate,training_card
from scripts.build_report import generate

EID='1'*20
PNG=base64.b64decode('iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+a2ioAAAAASUVORK5CYII=')


def fixture(root,name='model',eid=EID):
    run=root/'runs'/name;p=run/'evaluation/standard-256-v1'/eid;p.mkdir(parents=True)
    train=run/'checkpoints'/name;train.mkdir(parents=True)
    (train/'run_settings.json').write_text(json.dumps({'values':dict(checkpoint_validation='capped',tile_size=512,train_tile_size=256)}))
    (train/'data_run_settings.json').write_text('{}')
    pd.DataFrame(dict(epoch=[0,1],train_loss=[.8,.5],val_loss=[.9,.6],train_acc=[.6,.8],val_acc=[.55,.7])).to_csv(train/'metrics.csv',index=False)
    (train/'logs').mkdir();(train/'metrics.csv').rename(train/'logs/metrics.csv')
    pd.DataFrame(dict(epoch=[0,1],step=[0,1],lr=[.001,.0001])).to_csv(train/'lr_trace.csv',index=False)
    (train/'training_curves.png').write_bytes(PNG)
    val=p/'validation';val.mkdir();(val/'predictions.csv').write_text('label,prob_stego\n0,0.2\n1,0.8\n')
    cal=dict(deployment=True,threshold_id='f'*20,cutoff=.450123456789,threshold_origin='post_hoc_256',recipe_id='a'*20,
             checkpoint_sha256='b'*64,pairs=18000,score_csv_sha256=hashlib.sha256((val/'predictions.csv').read_bytes()).hexdigest())
    (val/'calibration.json').write_text(json.dumps(cal))
    refs=pd.DataFrame([dict(panel='standard',source='raise',k='all',cutoff=.6,threshold_origin='if_source_known_reference',FA=.08,miss=.16,P_E_at_validation_cutoff=.12,AUC=.91)]);refs.to_csv(val/'source_reference_thresholds.csv',index=False)
    refs.to_csv(p/'source_reference_summary.csv',index=False)
    main=[];diag=[];hard=[]
    for panel in ('standard','alaska-seen','alaska-unseen','lsdir'):
        row=dict(panel=panel,category='overall',value='all',pairs=2,images=4,photos=1,AUC=.9123456,threshold_id=cal['threshold_id'],cutoff=cal['cutoff'],
                 threshold_origin='post_hoc_256',FA=.1,miss=.2,P_E_at_validation_cutoff=.15)
        main.append(row);diag.append(dict(panel=panel,category='overall',value='all',test_min_P_E=.1,P_E_at_0_5=.2,AUC=.9))
        if panel=='standard':main.extend({**row,'category':'source','value':source,'pairs':1,'images':2} for source in ('raise','div2k'))
        out=p/'panels'/panel;out.mkdir(parents=True)
        pd.DataFrame(dict(label=[0,1,0,1],prob_stego=[.1,.8,.2,.3],requested_bpp=[.05,.05,.5,.5],k=[1,1,2,2],split='test',source_photo='photo')).to_csv(out/'predictions.csv',index=False)
        for k in (1,2):
            for condition in ('all','low','noisy','low_noisy'):
                hard.append(dict(**row,cell=f'{panel}/{k}/{condition}',k=k,condition=condition,distinct_clean_tiles=2,
                    FA_ci95_low=.01,FA_ci95_high=.3,miss_ci95_low=.1,miss_ci95_high=.4))
    pd.DataFrame(main).to_csv(p/'main_summary.csv',index=False);pd.DataFrame(diag).to_csv(p/'diagnostic_summary.csv',index=False)
    (p/'hard-test').mkdir();pd.DataFrame(hard).to_csv(p/'hard-test/hard_cells.csv',index=False)
    meta=dict(suite='standard-256-v1',evaluation_id=eid,quick=False,validation_threshold=dict(status='complete'),
              hard_enabled=True,hard_recipe_sha256='d'*64,recipe_sha256='c'*64,higher_k=False)
    (p/'evaluation_settings.json').write_text(json.dumps(meta))
    (p/'summary.json').write_text(json.dumps(dict(evaluation_id=eid,validation_threshold=cal,hard_test=dict(status='complete'))))
    return run,p


def cell_fixture(root):
    folder=root/'cell-output';(folder/'cells').mkdir(parents=True)
    reference=root/'cell-reference';reference.mkdir()
    def estimate(point):return dict(point=point,low=point-.05,high=point+.05)
    def map_record(cutoff):return dict(cutoff=cutoff,validation_FA=.0499,test=[dict(panel='pooled',placement='all',FA=.038,recall=.16)])
    primary=estimate(.8);cam=estimate(.7);control=estimate(.6)
    differences=dict(within_stego_auc=estimate(.1))
    localized=[dict(split='test',panel=p,placement='contiguous',maps=dict(aux=dict(within_stego_auc=primary),cam=dict(within_stego_auc=cam)),
        aux_minus_cam=differences) for p in ('pooled','raise')]
    summary=dict(localization=localized,maps=dict(aux=map_record(.123456),cam=map_record(1.23456)))
    (folder/'cells/summary.json').write_text(json.dumps(summary));(folder/'cells/cells.png').write_bytes(PNG)
    paired={p:dict(maps=dict(control_cam=dict(within_stego_auc=control)),run13_aux_minus_control_cam=dict(within_stego_auc=estimate(.2))) for p in ('pooled','raise')}
    (folder/'aux-minus-control-cam.json').write_text(json.dumps(dict(panels=paired)))
    (reference/'summary.json').write_text(json.dumps(dict(localization=[dict(split='test',panel='pooled',placement='contiguous',maps=dict(cam=dict(within_stego_auc=control)))],maps=dict(cam=map_record(2.34567)))))
    tiles=dict(standard=dict(candidate=dict(P_E=.03),control=dict(P_E=.026),candidate_minus_control=.004),
        hard_target=dict(candidate=estimate(.844),control=estimate(.839),candidate_minus_control=dict(point=.0051,low=.0004,high=.0097)),
        noisy_clean=dict(candidate=.189,control=.207,candidate_minus_control=-.018))
    (folder/'tile-comparison.json').write_text(json.dumps(tiles))
    return folder,reference


class ReportingTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.root=Path(self.tmp.name)
        self.run,self.path=fixture(self.root)
        image=self.root/'docs/figures/report/data-sample.png';image.parent.mkdir(parents=True)
        shutil.copy(ROOT/'docs/figures/report/data-sample.png',image)

    def test_headline_values_exact_saved_csv_and_honest_training_label(self):
        r=SavedReport(self.root,'model',EID)
        pd.testing.assert_frame_equal(r.headline, pd.read_csv(self.path/'main_summary.csv',float_precision='round_trip'))
        self.assertIn('RAISE, ALASKA and LSDIR, size-limited per source, 512px',r.training_label);self.assertNotIn('capped',r.training_label);self.assertIn('not the later',r.training_label)
        (self.run/'checkpoints/model/run_settings.json').write_text(json.dumps({'values':dict(checkpoint_validation='train_matched',train_tile_size=256,validation_recipe='own')}))
        self.assertIn('matched to training, 256px',SavedReport(self.root,'model',EID).training_label)
        self.assertNotIn('own',SavedReport(self.root,'model',EID).training_label)

    def test_missing_pending_and_unfinished_hard_refused(self):
        p=self.path/'summary.json';saved=p.read_text();p.unlink()
        with self.assertRaisesRegex(ValueError,'missing or pending'):SavedReport(self.root,'model',EID)
        p.write_text(saved);data=json.loads(saved);data['hard_test']['status']='pending';p.write_text(json.dumps(data))
        with self.assertRaisesRegex(ValueError,'pending hard'):SavedReport(self.root,'model',EID)

    def test_missing_saved_predictions_refused(self):
        (self.path/'panels/standard/predictions.csv').unlink()
        with self.assertRaisesRegex(ValueError,'missing saved output'):SavedReport(self.root,'model',EID)

    def test_pending_calibration_and_mismatched_threshold_refused(self):
        p=self.path/'evaluation_settings.json';data=json.loads(p.read_text());data['validation_threshold']['status']='pending';p.write_text(json.dumps(data))
        with self.assertRaisesRegex(ValueError,'pending'):SavedReport(self.root,'model',EID)
        data['validation_threshold']['status']='complete';p.write_text(json.dumps(data))
        m=pd.read_csv(self.path/'main_summary.csv');m.loc[0,'cutoff']=.9;m.to_csv(self.path/'main_summary.csv',index=False)
        with self.assertRaisesRegex(ValueError,'saved deployment'):SavedReport(self.root,'model',EID)

    def test_ambiguous_id_no_silent_selection(self):
        (self.path.parent/('2'*20)).mkdir()
        with self.assertRaisesRegex(ValueError,'available:'):evaluation_path(self.run)
        self.assertEqual(evaluation_path(self.run,EID),self.path)
        with self.assertRaisesRegex(ValueError,'missing evaluation'):evaluation_path(self.run,'3'*20)

    def test_miss_bins_include_upper_edge_use_saved_cutoff(self):
        r=SavedReport(self.root,'model',EID);tables=r.miss_tables()
        self.assertEqual(set(tables['bpp'].bpp_bin),{'[0.05,0.10)','[0.35,0.50]'})
        self.assertTrue((tables['k'].query('k==1').miss==0).all())
        self.assertTrue((tables['k'].query('k==2').miss==1).all())

    def test_div2k_reference_and_incompatible_reference_refused(self):
        refs=self.path/'validation/source_reference_thresholds.csv';f=pd.read_csv(refs);f['source']='div2k';f.to_csv(refs,index=False)
        with self.assertRaisesRegex(ValueError,'DIV2K'):SavedReport(self.root,'model',EID)
        f['source']='raise';f.to_csv(refs,index=False)
        a=SavedReport(self.root,'model',EID);_,other=fixture(self.root,'reference','2'*20)
        b=SavedReport(self.root,'reference','2'*20);compatible(a,b);b.meta['recipe_sha256']='different'
        with self.assertRaisesRegex(ValueError,'different standard'):compatible(a,b)

    def test_generator_executes_cpu_offline_and_preserves_inputs(self):
        template=self.root/'notebooks/templates';template.mkdir(parents=True)
        shutil.copy(ROOT/'notebooks/templates/report.ipynb',template/'report.ipynb')
        (self.root/'cvdl').symlink_to(ROOT/'cvdl',target_is_directory=True)
        before={p:hashlib.sha256(p.read_bytes()).hexdigest() for p in self.run.rglob('*') if p.is_file()}
        target=generate(self.root,'model',EID)
        nb=json.loads(target.read_text());outputs=[o for c in nb['cells'] for o in c.get('outputs',[])]
        self.assertFalse(any(o['output_type']=='error' for o in outputs))
        self.assertGreaterEqual(sum('image/png' in o.get('data',{}) for o in outputs),7)
        self.check_presentation(nb)
        self.assertNotIn('CELL_DIR',nb['metadata']['report']['parameters'])
        self.assertFalse(any('Where in the tile?' in ''.join(c['source']) for c in nb['cells']))
        self.assertTrue(any('No reference model supplied' in str(o.get('data',{})) for o in outputs))
        self.assertEqual(before,{p:hashlib.sha256(p.read_bytes()).hexdigest() for p in before})
        with self.assertRaisesRegex(ValueError,'author notes'):generate(self.root,'model',EID)

    def test_generator_fills_optional_reference_cpu(self):
        template=self.root/'notebooks/templates';template.mkdir(parents=True)
        shutil.copy(ROOT/'notebooks/templates/report.ipynb',template/'report.ipynb');(self.root/'cvdl').symlink_to(ROOT/'cvdl',target_is_directory=True)
        fixture(self.root,'reference','2'*20)
        target=generate(self.root,'model',EID,reference_run='reference',reference_evaluation_id='2'*20)
        nb=json.loads(target.read_text());self.assertEqual(nb['metadata']['report']['parameters']['REFERENCE_RUN'],'reference')
        self.assertFalse(any(o['output_type']=='error' for c in nb['cells'] for o in c.get('outputs',[])))
        self.check_presentation(nb,reference=True)

    def test_cell_tables_saved_values_and_control_summary_check(self):
        from cvdl.report_cells import cell_tables
        folder,control=cell_fixture(self.root)
        tables=cell_tables(folder,control)
        self.assertEqual(tables['auc'].iloc[0]['This model: cell head'],'0.800 [0.750, 0.850]')
        self.assertEqual(tables['rates']['Cell cut-off'].tolist(),['0.123','1.235','2.346'])
        self.assertTrue(tables['rates']['Test recall'].eq('16.0%').all())
        self.assertIn('about 16%',tables['recall_note'])
        self.assertEqual(tables['tiles'].iloc[1]['Change'],'+0.0051 [+0.0004, +0.0097]')
        self.assertEqual(len(cell_tables(folder)['rates']),2)
        saved=json.loads((control/'summary.json').read_text());saved['localization'][0]['maps']['cam']['within_stego_auc']['point']=.9
        (control/'summary.json').write_text(json.dumps(saved))
        with self.assertRaisesRegex(ValueError,'control cell summary differs'):cell_tables(folder,control)

    def test_generator_optional_cell_section_executes_saved_outputs_only(self):
        template=self.root/'notebooks/templates';template.mkdir(parents=True)
        shutil.copy(ROOT/'notebooks/templates/report.ipynb',template/'report.ipynb')
        (self.root/'cvdl').symlink_to(ROOT/'cvdl',target_is_directory=True)
        folder,control=cell_fixture(self.root)
        before={p:p.read_bytes() for base in (folder,control) for p in base.rglob('*') if p.is_file()}
        target=generate(self.root,'model',EID,cell_dir=folder,cell_reference_dir=control)
        nb=json.loads(target.read_text());self.check_presentation(nb,cell_section=True)
        self.assertFalse(any(o['output_type']=='error' for c in nb['cells'] for o in c.get('outputs',[])))
        content='\n'.join(str(o.get('data',{}).get('text/html',''))+str(o.get('data',{}).get('text/markdown','')) for c in nb['cells'] for o in c.get('outputs',[]))
        for value in ('0.800 [0.750, 0.850]','Control: CAM','about 16%','+0.0051 [+0.0004, +0.0097]'):
            self.assertIn(value,content)
        self.assertEqual(before,{p:p.read_bytes() for p in before})
        with self.assertRaisesRegex(ValueError,'requires --cell-dir'):
            generate(self.root,'model',EID,cell_reference_dir=control)

    def check_presentation(self,nb,reference=False,cell_section=False):
        visible=[]
        for cell in nb['cells']:
            if cell['cell_type']=='markdown':visible.append(''.join(cell['source']))
            else:
                self.assertTrue(cell['metadata']['jupyter']['source_hidden'])
                self.assertFalse(cell['metadata']['jupyter']['outputs_hidden'])
                for output in cell.get('outputs',[]):
                    visible.append(''.join(output.get('text',[])))
                    visible.extend(''.join(value) for key,value in output.get('data',{}).items() if key!='image/png')
        visible='\n'.join(visible)
        for secret in (EID,'2'*20,'f'*20,'a'*20,'b'*64,'c'*64,'d'*64,str(self.root),'.450123456789'):
            self.assertNotIn(secret,visible)
        self.assertIn('<details>',visible);self.assertNotIn('threshold_id',visible)
        meta=nb['metadata']['report'];self.assertEqual(meta['primary']['evaluation_id'],EID)
        self.assertEqual(meta['primary']['threshold_id'],'f'*20);self.assertEqual(meta['primary']['calibration_recipe_id'],'a'*20)
        self.assertEqual(meta['primary']['checkpoint_sha256'],'b'*64)
        if reference:self.assertEqual(meta['reference']['evaluation_id'],'2'*20)
        headings=[c for c in nb['cells'] if c['cell_type']=='markdown' and ''.join(c['source']).startswith('## ')]
        self.assertEqual(len(headings),12 if cell_section else 11)
        self.assertIn('## What does the data look like?', ''.join(headings[0]['source']))
        saved=(ROOT/'docs/figures/report/data-sample.png').read_bytes()
        images=[base64.b64decode(''.join(o['data']['image/png'])) for c in nb['cells'] for o in c.get('outputs',[])
                if 'image/png' in o.get('data',{})]
        self.assertIn(saved,images)
        self.assertFalse(any('Look at' in ''.join(c['source']) for c in nb['cells']))

    def test_display_rounding_matches_saved_csv(self):
        r=SavedReport(self.root,'model',EID);raw=pd.read_csv(self.path/'main_summary.csv',float_precision='round_trip')
        shown=display_frame(r.headline)
        for column,label in [('FA','False alarms %'),('miss','Misses %'),('P_E_at_validation_cutoff','Overall error %')]:
            self.assertEqual(list(shown[label]),[f'{100*x:.1f}%' for x in raw[column]])
        self.assertEqual(list(shown.AUC),[f'{x:.3f}' for x in raw.AUC])
        self.assertEqual(set(shown['Validation cut-off']),{'0.450'})
        self.assertNotIn('recipe',str(training_card(r)).lower())

    def test_single_source_rows_not_duplicated_standard_breakdown_retained(self):
        r=SavedReport(self.root,'model',EID);f=r.main.query('category == "overall"').copy()
        for panel,source in [('standard','raise'),('standard','div2k'),('alaska-seen','alaska'),('lsdir','lsdir')]:
            row=f.query('panel == @panel').iloc[0].copy();row['category']='source';row['value']=source
            f=pd.concat([f,pd.DataFrame([row])],ignore_index=True)
        result=deduplicate(f)
        self.assertEqual(len(result),6)
        self.assertEqual(len(result.query('panel == "standard"')),3)
        self.assertEqual(len(result.query('panel == "alaska-seen"')),1)
        row=f.query('panel == "alaska-seen"').iloc[0]
        f=pd.concat([f,pd.DataFrame([{**row.to_dict(),'category':'k','value':'1'},
            {**row.to_dict(),'category':'source/k','value':'alaska/1'}])],ignore_index=True)
        self.assertEqual(len(deduplicate(f).query('panel == "alaska-seen"')),2)

    def test_chart_titles_labels_and_direction(self):
        from cvdl import report_charts as P
        import matplotlib.pyplot as plt
        primary=SavedReport(self.root,'model',EID);fixture(self.root,'reference','2'*20)
        reference=SavedReport(self.root,'reference','2'*20)
        reference.main.loc[reference.main.panel=='standard','P_E_at_validation_cutoff']=.2
        reference.headline=deduplicate(reference.main)
        figures=[]
        with patch.object(P,'show_figure',side_effect=figures.append):
            P.training_plots(primary);P.panel_plot(primary);P.miss_plots(primary);P.hard_plots(primary);P.comparison_plot(primary,reference)
        self.assertEqual(len(figures),9)
        for fig in figures:
            labels=[]
            for ax in fig.axes:
                if ax.get_label()=='<colorbar>':continue
                self.assertTrue(ax.get_title() or fig._suptitle)
                self.assertTrue(ax.get_xlabel());self.assertTrue(ax.get_ylabel())
                labels.extend([ax.get_title(),ax.get_xlabel(),ax.get_ylabel(),*[t.get_text() for t in ax.get_xticklabels()]])
            self.assertNotIn('_',' '.join(labels))
        self.assertEqual(figures[-1].axes[0].patches[0].get_facecolor()[:3],tuple(int(v,16)/255 for v in ('27','83','48')))
        heat=figures[3].axes[0].images[0].get_cmap()
        self.assertGreater(sum(heat(0.)[:3]),sum(heat(1.)[:3]))
        for fig in figures:plt.close(fig)

    def test_tiny_change_not_labelled_zero_points_worse(self):
        from cvdl.report_charts import change_label
        self.assertEqual(change_label(.04),'<0.1 points worse')
        self.assertEqual(change_label(-.04),'<0.1 points better')
        self.assertEqual(change_label(0),'0.0 points unchanged')
        self.assertEqual(change_label(1.94),'+1.9 points worse')

    def test_recipe_and_hard_mismatch_checks_still_apply(self):
        a=SavedReport(self.root,'model',EID);fixture(self.root,'reference','2'*20);b=SavedReport(self.root,'reference','2'*20)
        b.meta['hard_recipe_sha256']='other'
        with self.assertRaisesRegex(ValueError,'different hard'):compatible(a,b)
        b.hard=None
        with self.assertRaisesRegex(ValueError,'coverage differs'):compatible(a,b)

if __name__=='__main__':unittest.main()
