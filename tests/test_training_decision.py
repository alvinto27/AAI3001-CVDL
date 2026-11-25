"""Frozen decision boundaries, shared PE draws, saved cell alignment and CLI gates."""
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd
import torch
from cvdl.cell_metrics import localization
from scripts.architecture_decision import auc_intervals, join_predictions
from scripts import training_decision as d
from scripts.cell_eval import report as cell_report
from tests.test_architecture_decision import fixture


def delta(point=0.,low=0.,high=0.):return dict(point=point,low=low,high=high)


def write_evaluation(root,identifier,offset=0.,epoch=7):
    folder=Path(root)/identifier;folder.mkdir()
    from cvdl.curriculum import scratch50_record
    checkpoint=Path(root)/f'best-{identifier}.ckpt'
    torch.save(dict(epoch=epoch,hyper_parameters={'embedding_schedule_identity':scratch50_record(180000)}),checkpoint)
    digest=hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    (folder/'validation').mkdir();(folder/'validation/predictions.csv').write_text('saved validation scores\n')
    t=dict(checkpoint_sha256=digest,recipe_id='recipe',recipe_sha256='recipe-hash',
        selected_pairs_sha256='full',full_pairs_sha256='full',mapping_sha256='mapping',
        geometry={'tile_size':256,'grid':'overlap_last'},policy='validation-min-pe-v1',deployment=True,
        quick=False,cutoff=.5,threshold_id='frozen',score_csv_sha256=d.sha(folder/'validation/predictions.csv'))
    (folder/'validation/calibration.json').write_text(json.dumps(t))
    rows=[]
    for panel in d.PANELS:
        for condition in ('low','noisy','low_noisy'):
            for i,placement in enumerate(('contiguous','random','stride')):
                for label in (0,1):
                    rows.append(dict(pair_id=f'{panel}/{condition}/{i}',source_photo=f'{panel}-{i}',
                        panel=panel,k=1,condition=condition,method='match' if i%2 else 'replace',
                        placement=placement,label=label,prob_stego=.3+label*.3+offset))
    hard=pd.DataFrame(rows);(folder/'hard-test').mkdir();hard.to_csv(folder/'hard-test/hard_predictions.csv',index=False)
    timings=[]
    for panel in d.PANELS:
        target=folder/f'panels/{panel}';target.mkdir(parents=True)
        frame=hard[(hard.panel==panel)&(hard.condition=='low')]
        frame.to_csv(target/'predictions.csv',index=False);timings.append(dict(panel=panel,pairs=3))
    (folder/'summary.json').write_text(json.dumps(dict(evaluation_id=identifier,suite='standard-256-v1',quick=False,
        timings=timings,validation_threshold=dict(t,status='complete'),hard_test=dict(status='complete',pairs=len(hard)//2,validation_threshold=t))))
    (folder/'evaluation_settings.json').write_text(json.dumps(dict(evaluation_id=identifier,suite='standard-256-v1',quick=False,
        calibration_inputs=t,checkpoint_sha256=digest,checkpoint=str(checkpoint),hard_recipe_sha256='hard-recipe',values={'embedding_schedule':'low_bpp50_v1','tiles_per_epoch':180000})))
    return folder


def write_cells(folder,*,aux=True,change=0.):
    folder.mkdir()
    pairs={};values={}
    for split in ('val','test'):
        rows=[dict(pair_id=f'{split}-{i}',source_photo=f'{split}-photo-{i//2}',
            split=('test','test_seen','test_unseen')[i%3] if split=='test' else 'val',
            placement='contiguous' if i<3 else 'random',diagnostic_panel='raise',method='match',
            bit_mode='lowest',payload_kind='random_bytes') for i in range(4)]
        counts=np.zeros((4,8,8),dtype=np.uint16);counts[:,0,0]=1
        valid=np.ones_like(counts,dtype=bool);valid[:,0,0]=False
        rng=np.random.default_rng(3 if split=='val' else 9)
        scores=rng.normal(size=(4,2,8,8)).astype(np.float32)
        scores[:,1,0,0]+=change
        arrays=dict(counts=counts,valid_negative=valid,cam=scores)
        if aux:arrays['aux']=scores.copy()
        pd.DataFrame(rows).to_csv(folder/f'{split}-pairs.csv',index=False)
        np.savez_compressed(folder/f'{split}-scores.npz',**arrays)
        pairs[split]=rows;values[split]=arrays
    summary=cell_report(pairs['val'],pairs['test'],values['val'],values['test'],replicates=11)
    (folder/'summary.json').write_text(json.dumps(summary))
    return folder


def write_arm_fixture(root):
    from scripts import architecture_decision as arm
    root=Path(root);eid='e'*20
    plan=root/'docs/active/plan-architecture-tests.md';plan.parent.mkdir(parents=True);plan.write_text('frozen arm-A fixture\n')
    for index,(run,evaluation) in enumerate(((arm.BASELINE_RUN,arm.BASELINE_EVALUATION),(arm.DELAYED_RUN,eid))):
        folder=root/'runs'/run/'evaluation/standard-256-v1'/evaluation;folder.mkdir(parents=True)
        (root/'runs'/run/'status.json').write_text(json.dumps(dict(elapsed_s=600+index*60)))
        threshold=dict(status='complete',deployment=True,recipe_id='frozen',cutoff=.5+index*.2)
        (folder/'summary.json').write_text(json.dumps(dict(evaluation_id=evaluation,suite='standard-256-v1',quick=False,
            hard_test={'status':'complete'},validation_threshold=threshold)))
        (folder/'evaluation_settings.json').write_text(json.dumps(dict(evaluation_id=evaluation)))
        frame=fixture();codes=frame.groupby('pair_id',sort=True).ngroup()
        frame['placement']=np.asarray(['contiguous','random','stride'])[codes%3]
        frame['method']=np.where(codes%2,'match','replace')
        frame['k']=1;frame['condition']='low'
        name=arm.NAMES[index]
        def predictions(data):return data.drop(columns=list(arm.NAMES)).assign(prob_stego=data[name])
        hard=pd.concat([frame.assign(pair_id=panel+'/'+frame.pair_id,panel=panel) for panel in arm.TARGETS],ignore_index=True)
        (folder/'hard-test').mkdir();predictions(hard).to_csv(folder/'hard-test/hard_predictions.csv',index=False)
        for panel in arm.PANELS:
            target=folder/f'panels/{panel}';target.mkdir(parents=True)
            predictions(frame).to_csv(target/'predictions.csv',index=False)
    return eid


def artifact_digests(folder):
    result={name:d.sha(folder/name) for name in ('evidence.json','report.md','auc-intervals.csv','panel-rates.csv')}
    digest=hashlib.sha256()
    with np.load(folder/'bootstrap-replicates.npz') as saved:
        for name in saved.files:digest.update(name.encode());digest.update(saved[name].tobytes())
    result['bootstrap_arrays']=digest.hexdigest()
    return result


class TrainingDecisionTests(unittest.TestCase):
    def test_r1_both_routes_and_literal_boundaries(self):
        h=delta(-.001,-.01,.02);p=delta(-.002,-.004,-1e-9)
        r=d.rule_checks('paired_r1',h,p);self.assertTrue(r['passed']);self.assertTrue(r['routes']['a']['passed']);self.assertFalse(r['routes']['b']['passed'])
        self.assertFalse(d.rule_checks('paired_r1',h,dict(p,high=0))['passed'])
        self.assertFalse(d.rule_checks('paired_r1',dict(h,low=-.01000001),p)['passed'])
        h=delta(.01,1e-9,.02);p=delta(.001,-.001,.0025)
        r=d.rule_checks('paired_r1',h,p);self.assertTrue(r['passed']);self.assertTrue(r['routes']['b']['passed'])
        for changed in (dict(h,point=.009999),dict(h,low=0)):
            self.assertFalse(d.rule_checks('paired_r1',changed,p)['passed'])
        self.assertFalse(d.rule_checks('paired_r1',h,dict(p,high=.00250001))['passed'])

    def test_curriculum_epoch_and_r2_head_guard(self):
        h=delta(.01,.001,.02);p=delta(.001,-.001,.0025)
        self.assertTrue(d.rule_checks('curriculum50',h,p,epoch=8)['passed'])
        self.assertFalse(d.rule_checks('curriculum50',h,p,epoch=7)['passed'])
        with self.assertRaises(ValueError):d.rule_checks('curriculum50',h,p)
        cells=dict(head_vs_head=True,primary={'difference':delta(.02,.001,.03)},operating_points={'candidate':{'test_fa':.07}})
        self.assertTrue(d.rule_checks('paired_r2',dict(h,low=-.01),p,cells=cells)['passed'])
        for change in (dict(cells,head_vs_head=False),None):
            with self.assertRaises(ValueError):d.rule_checks('paired_r2',h,p,cells=change)
        cells['operating_points']['candidate']['test_fa']=.0700001
        self.assertFalse(d.rule_checks('paired_r2',h,p,cells=cells)['passed'])
        cells['operating_points']['candidate']['test_fa']=.07
        for bad in (delta(.0199,.001,.03),delta(.02,0,.03),None):
            cells['primary']['difference']=bad
            self.assertFalse(d.rule_checks('paired_r2',h,p,cells=cells)['passed'])

    def test_pe_paired_photo_draws_and_own_cutoff_ge_ties(self):
        frame=fixture().rename(columns=dict(srm30='reference',delayed='candidate'))
        cutoffs={'reference':2/7,'candidate':3/8}
        item,draws=auc_intervals(frame,metric='pe',cutoffs=cutoffs,names=d.NAMES,replicates=31)
        rng=np.random.default_rng(d.SEED);groups=[g.index.to_numpy() for _,g in frame.groupby('source_photo',sort=True)]
        expected=[]
        for _ in range(31):
            selected=np.concatenate([groups[i] for i in rng.integers(0,len(groups),len(groups))])
            expected.append([float(((frame[n].to_numpy()[selected]>=cutoffs[n])!=frame.label.to_numpy()[selected]).mean()) for n in d.NAMES])
        np.testing.assert_array_equal(draws,expected)
        lo,hi=np.quantile(draws[:,1]-draws[:,0],[.025,.975])
        self.assertEqual(item['difference']['low'],lo);self.assertEqual(item['difference']['high'],hi)
        frame['candidate']=frame.reference;cutoffs['candidate']=cutoffs['reference']
        item,draws=auc_intervals(frame,metric='pe',cutoffs=cutoffs,names=d.NAMES,replicates=31)
        self.assertEqual(item['difference'],delta());self.assertEqual(item['pairs'],8)

    def test_architecture_default_outputs_are_frozen(self):
        # Golden values from the pre-change arm-A algorithm. Exact NP operation order.
        frame=fixture();result,samples=auc_intervals(frame,replicates=31)
        self.assertEqual(result['models']['srm30']['point'],.875)
        self.assertEqual(result['models']['delayed']['point'],.96875)
        self.assertEqual(result['difference']['point'],.09375)
        with TemporaryDirectory() as folder:
            paths=[]
            for name in ('srm30','delayed'):
                p=Path(folder)/f'{name}.csv';frame.drop(columns=['srm30','delayed']).assign(prob_stego=frame[name]).to_csv(p,index=False);paths.append(p)
            old=join_predictions(paths);new=join_predictions(paths,names=d.NAMES)
            pd.testing.assert_frame_equal(old.rename(columns={'srm30':'reference','delayed':'candidate'}),new)
            other,draws=auc_intervals(new,names=d.NAMES,replicates=31)
            np.testing.assert_array_equal(samples,draws)
            self.assertEqual(result['difference'],other['difference'])

    def test_arm_a_all_outputs_match_pre_change_golden(self):
        # Golden generated with f91f397's original architecture_decision.py.
        # Only output hashes are stored: no second join implementation ships.
        from scripts import architecture_decision as arm
        with TemporaryDirectory() as folder:
            root=Path(folder);eid=write_arm_fixture(root);out=root/'output'
            with patch.object(arm,'ROOT',root),patch.object(arm,'OUT',out):arm.summarize(eid,out)
            expected=json.loads((d.ROOT/'tests/fixtures/architecture-decision-golden.json').read_text())
            self.assertEqual(artifact_digests(out),expected)

    def test_evaluation_ambiguity_quick_frozen_cutoff_and_checkpoint_epoch(self):
        with TemporaryDirectory() as folder:
            root=Path(folder);a=write_evaluation(root,'a'*20)
            self.assertEqual(d.evaluation_folder(a),a.resolve())
            _,meta,_=d.validate_evaluation(a);epoch,path=d.selected_epoch(meta);self.assertEqual(epoch,8)
            with self.assertRaisesRegex(ValueError,'SHA'):d.selected_epoch(dict(meta,checkpoint_sha256='bad'))
            with self.assertRaisesRegex(ValueError,'versioned'):d.selected_epoch(dict(meta,values={}))
            bad=torch.load(path,map_location='cpu',weights_only=False)
            bad['hyper_parameters']['embedding_schedule_identity']['version']=99
            torch.save(bad,path)
            with self.assertRaisesRegex(ValueError,'schedule identity'):
                d.selected_epoch(dict(meta,checkpoint_sha256=d.sha(path)))
            meta['values']['evaluation_checkpoint']='last'
            with self.assertRaisesRegex(ValueError,'best'):d.selected_epoch(meta)
            summary=json.loads((a/'summary.json').read_text());summary['quick']=True;(a/'summary.json').write_text(json.dumps(summary))
            with self.assertRaisesRegex(ValueError,'quick'):d.validate_evaluation(a)
            with self.assertRaisesRegex(ValueError,'exactly one'):d.evaluation_folder(root)

    def test_end_to_end_named_rule_hashes_outputs_and_no_input_writes(self):
        with TemporaryDirectory() as folder:
            root=Path(folder);a=write_evaluation(root,'a'*20);b=write_evaluation(root,'b'*20)
            paths=d.evaluation_inputs(a)+d.evaluation_inputs(b);before={str(p):d.sha(p) for p in paths}
            result=d.summarize(a,b,root/'decision','curriculum50')
            self.assertFalse(result['passed']);self.assertEqual(result['selected_epoch_one_based'],8)
            self.assertEqual(result['standard_pe']['difference'],delta())
            self.assertTrue(result['inputs_unchanged']);self.assertEqual(result['input_sha256'],result['input_sha256_after'])
            self.assertEqual(before,{str(p):d.sha(p) for p in paths})
            saved=json.loads((root/'decision/evidence.json').read_text());self.assertEqual(saved['rule_set'],'curriculum50')
            self.assertIn('FAIL',(root/'decision/report.md').read_text())
            with self.assertRaisesRegex(ValueError,'empty'):d.summarize(a,b,root/'decision','paired_r1')
            with self.assertRaisesRegex(ValueError,'input folders'):d.summarize(a,b,a/'new-output','paired_r1')

    def test_complete_summary_cannot_hide_dropped_rows_in_both_models(self):
        with TemporaryDirectory() as folder:
            root=Path(folder);a=write_evaluation(root,'a'*20);b=write_evaluation(root,'b'*20)
            for p in (a,b):
                frame=pd.read_csv(p/'hard-test/hard_predictions.csv').iloc[2:]
                frame.to_csv(p/'hard-test/hard_predictions.csv',index=False)
            with self.assertRaisesRegex(ValueError,'prediction count'):d.summarize(a,b,root/'out','paired_r1')

    def test_cell_head_pairing_shared_draws_and_cam_is_context_only(self):
        with TemporaryDirectory() as folder:
            root=Path(folder);a=write_cells(root/'a');b=write_cells(root/'b')
            result,samples=d.compare_cells(a,b,replicates=31)
            self.assertTrue(result['head_vs_head']);self.assertEqual(result['primary']['difference']['point'],0.)
            np.testing.assert_array_equal(samples['within_stego_auc'][:,0],samples['within_stego_auc'][:,1])
            self.assertEqual(result['primary']['eligible_tiles'],3)
            # Row order can differ: arrays follow their own CSV before the exact join.
            for split in ('val','test'):
                frame=pd.read_csv(b/f'{split}-pairs.csv');frame.iloc[::-1].to_csv(b/f'{split}-pairs.csv',index=False)
                with np.load(b/f'{split}-scores.npz') as arrays:values={k:v[::-1] for k,v in arrays.items()}
                np.savez_compressed(b/f'{split}-scores.npz',**values)
            reordered,_=d.compare_cells(a,b,replicates=31)
            self.assertEqual(result,reordered)
            context,_=d.compare_cells(a,b,reference_map='cam',replicates=31)
            with self.assertRaisesRegex(ValueError,'aux head'):d.rule_checks('paired_r2',delta(),delta(),cells=context)
            c=write_cells(root/'cam',aux=False)
            with self.assertRaisesRegex(ValueError,'missing saved aux'):d.compare_cells(c,b,replicates=11)

    def test_cell_target_metadata_and_cutoff_mismatch_fail_closed(self):
        with TemporaryDirectory() as folder:
            root=Path(folder);a=write_cells(root/'a');b=write_cells(root/'b')
            frame=pd.read_csv(b/'test-pairs.csv');frame.loc[0,'source_photo']='other';frame.to_csv(b/'test-pairs.csv',index=False)
            with self.assertRaisesRegex(ValueError,'metadata'):d.compare_cells(a,b,replicates=7)
            frame.loc[0,'source_photo']='test-photo-0';frame.to_csv(b/'test-pairs.csv',index=False)
            with np.load(b/'test-scores.npz') as saved:arrays={k:v.copy() for k,v in saved.items()}
            arrays['counts'][0,0,0]=2;np.savez_compressed(b/'test-scores.npz',**arrays)
            with self.assertRaisesRegex(ValueError,'eligibility'):d.compare_cells(a,b,replicates=7)
            arrays['counts'][0,0,0]=1;np.savez_compressed(b/'test-scores.npz',**arrays)
            summary=json.loads((b/'summary.json').read_text());summary['maps']['aux']['cutoff']=-100;(b/'summary.json').write_text(json.dumps(summary))
            with self.assertRaisesRegex(ValueError,'5%'):d.compare_cells(a,b,replicates=7)

    def test_cell_metrics_default_output_unchanged_by_generic_difference_label(self):
        counts=np.zeros((2,8,8),int);counts[:,0,0]=1
        neg=counts==0;scores=np.random.default_rng(4).normal(size=(2,2,8,8))
        old=localization(counts,neg,scores,['a','b'],replicates=31)
        new=localization(counts,neg,scores,['a','b'],replicates=31,difference_name='candidate_minus_reference')
        new['aux_minus_cam']=new.pop('candidate_minus_reference');self.assertEqual(old,new)


if __name__=='__main__':unittest.main()
