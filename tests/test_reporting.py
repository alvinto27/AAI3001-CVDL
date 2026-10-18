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
    cal=dict(deployment=True,threshold_id='frozen',cutoff=.45,threshold_origin='post_hoc_256',recipe_id='cal',
             checkpoint_sha256='checkpoint',pairs=18000,score_csv_sha256=hashlib.sha256((val/'predictions.csv').read_bytes()).hexdigest())
    (val/'calibration.json').write_text(json.dumps(cal))
    refs=pd.DataFrame([dict(source='raise',cutoff=.6,threshold_origin='if_source_known_reference')]);refs.to_csv(val/'source_reference_thresholds.csv',index=False)
    refs.to_csv(p/'source_reference_summary.csv',index=False)
    main=[];diag=[];hard=[]
    for panel in ('standard','alaska-seen','alaska-unseen','lsdir'):
        row=dict(panel=panel,category='overall',value='all',pairs=2,images=4,photos=1,AUC=.9,threshold_id='frozen',cutoff=.45,
                 threshold_origin='post_hoc_256',FA=.1,miss=.2,P_E_at_validation_cutoff=.15)
        main.append(row);diag.append(dict(panel=panel,category='overall',value='all',test_min_P_E=.1,P_E_at_0_5=.2,AUC=.9))
        out=p/'panels'/panel;out.mkdir(parents=True)
        pd.DataFrame(dict(label=[0,1,0,1],prob_stego=[.1,.8,.2,.3],requested_bpp=[.05,.05,.5,.5],k=[1,1,2,2],split='test',source_photo='photo')).to_csv(out/'predictions.csv',index=False)
        for k in (1,2):
            for condition in ('all','low','noisy','low_noisy'):
                hard.append(dict(**row,cell=f'{panel}/{k}/{condition}',k=k,condition=condition,distinct_clean_tiles=2,
                    FA_ci95_low=.01,FA_ci95_high=.3,miss_ci95_low=.1,miss_ci95_high=.4))
    pd.DataFrame(main).to_csv(p/'main_summary.csv',index=False);pd.DataFrame(diag).to_csv(p/'diagnostic_summary.csv',index=False)
    (p/'hard-test').mkdir();pd.DataFrame(hard).to_csv(p/'hard-test/hard_cells.csv',index=False)
    meta=dict(suite='standard-256-v1',evaluation_id=eid,quick=False,validation_threshold=dict(status='complete'),
              hard_enabled=True,hard_recipe_sha256='same-hard',recipe_sha256='same-standard',higher_k=False)
    (p/'evaluation_settings.json').write_text(json.dumps(meta))
    (p/'summary.json').write_text(json.dumps(dict(evaluation_id=eid,validation_threshold=cal,hard_test=dict(status='complete'))))
    return run,p


class ReportingTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.root=Path(self.tmp.name)
        self.run,self.path=fixture(self.root)

    def test_headline_values_exact_saved_csv_and_honest_training_label(self):
        r=SavedReport(self.root,'model',EID)
        pd.testing.assert_frame_equal(r.headline, pd.read_csv(self.path/'main_summary.csv',float_precision='round_trip'))
        self.assertIn('capped, 512px',r.training_label);self.assertIn('NOT the new',r.training_label)
        (self.run/'checkpoints/model/run_settings.json').write_text(json.dumps({'values':dict(checkpoint_validation='train_matched',train_tile_size=256,validation_recipe='own')}))
        self.assertIn('train-matched, 256px',SavedReport(self.root,'model',EID).training_label)

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
        self.assertTrue(any('external network disabled' in ''.join(o.get('text',[])) for o in outputs))
        self.assertTrue(any('No reference model supplied' in ''.join(o.get('text',[])) for o in outputs))
        self.assertEqual(before,{p:hashlib.sha256(p.read_bytes()).hexdigest() for p in before})
        with self.assertRaisesRegex(ValueError,'author notes'):generate(self.root,'model',EID)

    def test_generator_fills_optional_reference_cpu(self):
        template=self.root/'notebooks/templates';template.mkdir(parents=True)
        shutil.copy(ROOT/'notebooks/templates/report.ipynb',template/'report.ipynb');(self.root/'cvdl').symlink_to(ROOT/'cvdl',target_is_directory=True)
        fixture(self.root,'reference','2'*20)
        target=generate(self.root,'model',EID,reference_run='reference',reference_evaluation_id='2'*20)
        nb=json.loads(target.read_text());self.assertEqual(nb['metadata']['report']['parameters']['REFERENCE_RUN'],'reference')
        self.assertFalse(any(o['output_type']=='error' for c in nb['cells'] for o in c.get('outputs',[])))

if __name__=='__main__':unittest.main()
