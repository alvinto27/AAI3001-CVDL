"""Synthetic CPU map/identity/two-reference tests; no dataset inference or GPU."""
import copy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch
from cvdl.cell_artifacts import sha,bind_scores,validate_scores,checkpoint_identity,validate_cutoffs
from cvdl.gapcov_provenance import validate_sources,CELL_AUX
from cvdl.cell_diagnostics import cell_maps
from cvdl.models import model_from_hparams
from cvdl.cov_pool import SPEC
from cvdl.photo_inference import infer_photo
from scripts import cell_eval,photo_heatmap as photo,compare_run21_cells as compare,training_decision as d
from tests.test_gapcov_cell_training import ON,ROOT
from tests.test_training_decision import write_evaluation,write_cells


def own_validation(folder,origin,checkpoint):
    """Real small validation predictions and own fitted cutoff, not test rows."""
    from cvdl.metrics import threshold
    rows=[dict(pair_id=f'v{i}',cover_id=f'v{i}',source_photo=f'valphoto{i}',split='val',label=y,
               prob_stego=.1+.6*y+.01*i) for i in range(4) for y in (0,1)]
    frame=pd.DataFrame(rows);frame.to_csv(folder/'validation/predictions.csv',index=False)
    cutoff,_=threshold(pd.read_csv(folder/'validation/predictions.csv',float_precision='round_trip'),'pe')
    t=json.loads((folder/'validation/calibration.json').read_text())
    t.update(checkpoint_sha256=sha(checkpoint),threshold_origin=origin,cutoff=cutoff,score_csv_sha256=sha(folder/'validation/predictions.csv'))
    (folder/'validation/calibration.json').write_text(json.dumps(t))
    summary=json.loads((folder/'summary.json').read_text());summary['validation_threshold']=dict(t,status='complete');summary['hard_test']['validation_threshold']=t
    (folder/'summary.json').write_text(json.dumps(summary))
    meta=json.loads((folder/'evaluation_settings.json').read_text());meta.update(calibration_inputs=t,checkpoint=str(checkpoint),checkpoint_sha256=sha(checkpoint))
    (folder/'evaluation_settings.json').write_text(json.dumps(meta))
    return t


def fixtures(root):
    reference=write_evaluation(root,'a'*20);candidate=write_evaluation(root,'b'*20)
    refcp=d.checkpoint_path(json.loads((reference/'evaluation_settings.json').read_text()))
    network=model_from_hparams(dict(ON,cov_pool_identity=SPEC)).eval()
    cp=root/'last.ckpt';h=dict(ON,cov_pool_identity=SPEC,source_provenance=validate_sources(ROOT,CELL_AUX),
        initialization_checkpoint_sha256=sha(refcp))
    torch.save(dict(hyper_parameters=h,state_dict={'net.'+k:v for k,v in network.state_dict().items()}),cp)
    own_validation(reference,'post_hoc_256',refcp);cal=own_validation(candidate,'train_matched',cp)
    cells_ref=write_cells(root/'cells-ref');cells=write_cells(root/'cells-candidate',change=.1)
    values={};rows={}
    for split in ('val','test'):
        rows[split]=pd.read_csv(cells/f'{split}-pairs.csv').to_dict('records')
        with np.load(cells/f'{split}-scores.npz') as saved:values[split]={k:saved[k] for k in saved.files}
        values[split]['class_score']=values[split].pop('cam')
        np.savez_compressed(cells/f'{split}-scores.npz',**values[split])
    summary=cell_eval.report(rows['val'],rows['test'],values['val'],values['test'],replicates=11)
    cell_eval.describe_map(summary,network);(cells/'summary.json').write_text(json.dumps(summary))
    identity=checkpoint_identity(cp)
    manifest=json.loads((ROOT/'configs/cell-diagnostic-v1.json').read_text())
    bind_scores(cells,identity,manifest)
    pins=dict(detector=dict(evaluation=str(reference),checkpoint_sha256=sha(refcp),
        files_sha256={str(p.relative_to(reference)):sha(p) for p in d.evaluation_inputs(reference)}),
        heatmap=dict(folder=str(cells_ref),checkpoint_sha256='e'*64,
        files_sha256={p.name:sha(p) for p in cells_ref.iterdir()}))
    return reference,candidate,cells_ref,cells,cp,network,cal,pins


class MapWorkflowTests(unittest.TestCase):
    def test_map_specific_metadata_and_difference_label_preserve_cam(self):
        with TemporaryDirectory() as tmp:
            folder=write_cells(Path(tmp)/'scores')
            rows={s:pd.read_csv(folder/f'{s}-pairs.csv').to_dict('records') for s in ('val','test')}
            arrays={}
            for split in rows:
                with np.load(folder/f'{split}-scores.npz') as saved:arrays[split]={k:saved[k] for k in saved.files}
                arrays[split]['class_score']=arrays[split].pop('cam')
            summary=cell_eval.report(rows['val'],rows['test'],arrays['val'],arrays['test'],replicates=8)
            net=model_from_hparams(dict(ON,cov_pool_identity=SPEC))
            cell_eval.describe_map(summary,net)
            self.assertFalse(summary['context_only']);self.assertNotIn('auxiliary_target_mark',summary)
            self.assertTrue(summary['map_metadata']['class_score']['context_only'])
            self.assertFalse(summary['map_metadata']['aux']['context_only'])
            self.assertEqual(summary['map_metadata']['aux']['receptive_field'],48)
            self.assertEqual(summary['map_metadata']['class_score']['receptive_field'],256)
            self.assertIn('aux_minus_class_score',summary['localization'][0]);self.assertNotIn('aux_minus_cam',summary['localization'][0])
            old=json.loads((folder/'summary.json').read_text());self.assertIn('aux_minus_cam',old['localization'][0])

    def test_bound_artifacts_reject_stale_score_row_manifest_checkpoint_source(self):
        with TemporaryDirectory() as tmp:
            _,_,_,cells,cp,_,_,_=fixtures(Path(tmp))
            validate_scores(cells,checkpoint=cp,require_v2=True)
            for name in ('val-pairs.csv','test-scores.npz','manifest.json','summary.json'):
                path=cells/name;before=path.read_bytes();path.write_bytes(before+b' ')
                with self.subTest(name=name),self.assertRaisesRegex(ValueError,'bytes changed'):validate_scores(cells,checkpoint=cp)
                path.write_bytes(before)
            with self.assertRaisesRegex(ValueError,'checkpoint identity'):validate_scores(cells,expected_sha='0'*64)
            identity=checkpoint_identity(cp);identity['source_provenance']={}
            bind_scores(cells,identity,json.loads((ROOT/'configs/cell-diagnostic-v1.json').read_text()))
            with self.assertRaisesRegex(ValueError,'source identity'):validate_scores(cells,checkpoint=cp,require_v2=True)

    def test_photo_head_only_dual_and_identity_cutoff_refusals(self):
        with TemporaryDirectory() as tmp:
            root=Path(tmp);_,_,_,cells,cp,network,cal,_=fixtures(root)
            for selection,count in (('head-only',1),('head-attribution',2)):
                kinds,cutoffs,limits=photo.photo_configuration(cp,cal,cells/'summary.json',network,selection)
                with patch('torch.autograd.grad',side_effect=AssertionError('head only')) if count==1 else patch('builtins.print'):
                    origins,scores,maps=infer_photo(network,np.zeros((256,256,3),np.uint8),dict(source_height=256,source_width=256),'cpu',1,map_kinds=kinds)
                self.assertEqual(maps.shape,(count,8,8));self.assertEqual(len(scores),1)
                row=dict(shape=(256,256),thumbnail=np.zeros((256,256,3),np.uint8),truth=np.zeros((8,8)),maps=maps,
                    origins=origins,decision=False,max_score=scores[0],photo_id='toy',name='clean')
                photo.plot_rows([row],root/(selection+'.png'),limits,cutoffs,map_kinds=kinds)
                self.assertTrue((root/(selection+'.png')).is_file())
            with self.assertRaisesRegex(ValueError,'no CAM'):photo.photo_configuration(cp,cal,cells/'summary.json',network)
            for bad in (dict(cal,checkpoint_sha256='0'*64),dict(cal,quick=True),dict(cal,deployment=False)):
                with self.assertRaisesRegex(ValueError,'calibration mismatch'):photo.photo_configuration(cp,bad,cells/'summary.json',network,'head-only')
            summary=json.loads((cells/'summary.json').read_text());summary['maps']['aux']['cutoff']+=1
            (cells/'summary.json').write_text(json.dumps(summary));bind_scores(cells,checkpoint_identity(cp),json.loads((cells/'manifest.json').read_text()))
            with self.assertRaisesRegex(ValueError,'own validation'):photo.photo_configuration(cp,cal,cells/'summary.json',network,'head-only')

    def test_cell_eval_cli_writes_checkpoint_manifest_score_binding(self):
        with TemporaryDirectory() as tmp:
            root=Path(tmp);_,_,_,cells,cp,_,_,_=fixtures(root);out=root/'cell-eval'
            rows={s:pd.read_csv(cells/f'{s}-pairs.csv').to_dict('records') for s in ('val','test')}
            values={}
            for split in rows:
                with np.load(cells/f'{split}-scores.npz') as saved:values[split]={k:saved[k] for k in saved.files}
            argv=['cell_eval.py',str(cp),'--output',str(out),'--device','cpu','--workers','0','--replicates','8']
            with patch('sys.argv',argv),patch('scripts.pc_safety.own_process_group'),\
                 patch('scripts.pc_safety.available_gb',return_value=20),\
                 patch.object(cell_eval,'selected_rows',side_effect=lambda root,split,**kwargs:rows[split]),\
                 patch.object(cell_eval,'score',side_effect=[(values['val'],None),(values['test'],None)]):cell_eval.main()
            summary=validate_scores(out,checkpoint=cp,require_v2=True)
            self.assertEqual(summary['identity']['checkpoint_sha256'],sha(cp));self.assertFalse(summary['context_only'])
            self.assertTrue((out/'artifact_identity.json').is_file());validate_cutoffs(out,summary,['aux','class_score'])

    def test_photo_cli_explicit_head_only_and_dual_metadata(self):
        with TemporaryDirectory() as tmp:
            root=Path(tmp);_,_,_,cells,cp,network,cal,_=fixtures(root)
            record=dict(photo_id='toy',photo_index='1',source='raise',split='test',source_height=256,source_width=256)
            for selection,count in (('head-only',1),('head-attribution',2)):
                out=root/('cli-'+selection)
                argv=['photo_heatmap.py','--run','synthetic-candidate','--evaluation-id','b'*20,
                      '--cell-summary',str(cells/'summary.json'),'--output',str(out),'--device','cpu','--maps',selection]
                with patch('sys.argv',argv),patch('scripts.pc_safety.own_process_group'),patch('scripts.pc_safety.available_gb',return_value=20),\
                     patch.object(photo,'photo_selection',return_value=[record]),patch.object(photo,'plot_rows'),\
                     patch('scripts.evaluate_run.checkpoint_folder',return_value=cp.parent),\
                     patch('cvdl.lit.select_evaluation_checkpoint',return_value=cp),\
                     patch('cvdl.standardtest.load_network',return_value=network),\
                     patch('scripts.training_decision.validate_evaluation',return_value=({}, {}, cal)),\
                     patch('lsb_source.load_photo',return_value=np.zeros((256,256,3),np.uint8)):
                    if count==1:
                        with patch('torch.autograd.grad',side_effect=AssertionError('head-only gradients')):photo.main()
                    else:photo.main()
                data=json.loads((out/'photo-toy.json').read_text())
                self.assertEqual(len(data['map_kinds']),count);self.assertEqual(data['checkpoint_sha256'],sha(cp))
                self.assertEqual(data['map_geometry']['trained_head']['receptive_field'],48)
                if count==2:self.assertTrue(data['map_geometry']['attribution']['global_context'])
                self.assertIn('not a calibrated photo',data['photo_rule'])

    def test_legacy_photo_defaults_use_pinned_run13_own_inputs(self):
        cp=ROOT/'runs/srm-cell-aux-20261021T054704Z-6997/checkpoints/srm-cell-aux-20261021T054704Z-6997/last.ckpt'
        if not cp.exists():self.skipTest('run13 local checkpoint absent')
        cal=json.loads((ROOT/'runs'/photo.RUN/'evaluation/standard-256-v1'/photo.EID/'validation/calibration.json').read_text())
        summary=ROOT/'runs/verification/cell-run13-2026-10-21/cells/summary.json'
        network=model_from_hparams(dict(model='xunet_wide',front_end='srm',norm='none',cell_supervision='changed_cells_v1'))
        kinds,_,_=photo.photo_configuration(cp,cal,summary,network)
        self.assertEqual(kinds,('trained_head','cam'))

    def test_two_reference_numeric_comparison_own_cutoffs_pins_and_snapshot(self):
        with TemporaryDirectory() as tmp:
            root=Path(tmp);ref,cand,cellref,cells,cp,_,_,pins=fixtures(root)
            before={str(p):sha(p) for p in root.rglob('*') if p.is_file()}
            with patch.object(compare,'load_pins',return_value=pins):result=compare.summarize(cand,cells,root/'output')
            self.assertEqual(result['detector_reference']['evaluation'],str(ref));self.assertEqual(result['heatmap_reference']['folder'],str(cellref))
            self.assertNotIn('passed',result);self.assertIn('Pending',result['proposed_guards']['status'])
            self.assertTrue(result['inputs_unchanged']);self.assertEqual(result['cell_bootstrap']['replicates'],2000)
            for path,value in before.items():self.assertEqual(sha(path),value)
            path=ref/'summary.json';original=path.read_text();path.write_text(original+' ')
            with patch.object(compare,'load_pins',return_value=pins),self.assertRaisesRegex(ValueError,'pinned reference'):compare.summarize(cand,cells,root/'bad')
            path.write_text(original)
            summary=json.loads((cells/'summary.json').read_text());summary['maps']['aux']['cutoff']+=1
            (cells/'summary.json').write_text(json.dumps(summary));bind_scores(cells,checkpoint_identity(cp),json.loads((cells/'manifest.json').read_text()))
            with patch.object(compare,'load_pins',return_value=pins),self.assertRaisesRegex(ValueError,'own validation'):compare.summarize(cand,cells,root/'bad')

    def test_comparison_rejects_changed_inputs_during_processing_and_bad_rows(self):
        with TemporaryDirectory() as tmp:
            root=Path(tmp);_,cand,_,cells,cp,_,_,pins=fixtures(root)
            original=d.compare_cells
            def changed(*args,**kwargs):
                result=original(*args,**kwargs)
                with (cells/'test-pairs.csv').open('a') as stream:stream.write('\\n')
                return result
            with patch.object(compare,'load_pins',return_value=pins),patch.object(d,'compare_cells',side_effect=changed):
                with self.assertRaisesRegex(ValueError,'inputs changed'):compare.summarize(cand,cells,root/'output')
            self.assertFalse((root/'output').exists())
            # Rebind a duplicated row: strict row/target validation still refuses.
            frame=pd.read_csv(cells/'test-pairs.csv');frame.loc[1,'pair_id']=frame.loc[0,'pair_id'];frame.to_csv(cells/'test-pairs.csv',index=False)
            bind_scores(cells,checkpoint_identity(cp),json.loads((cells/'manifest.json').read_text()))
            with patch.object(compare,'load_pins',return_value=pins),self.assertRaisesRegex(ValueError,'invalid cell row keys'):
                compare.summarize(cand,cells,root/'bad-rows')

    def test_comparison_refuses_quick_pilot_origin_targets_manifest_and_identity(self):
        for mode in ('quick','pilot','origin','targets','manifest','identity'):
            with self.subTest(mode=mode),TemporaryDirectory() as tmp:
                root=Path(tmp);_,cand,_,cells,cp,_,_,pins=fixtures(root)
                meta_path=cand/'evaluation_settings.json';meta=json.loads(meta_path.read_text())
                if mode=='quick':meta['quick']=True
                if mode=='pilot':meta['values']['pilot']=True
                meta_path.write_text(json.dumps(meta))
                if mode=='origin':
                    own_validation(cand,'post_hoc_256',cp)
                if mode=='targets':
                    path=cells/'test-scores.npz'
                    with np.load(path) as saved:arrays={k:saved[k] for k in saved.files}
                    arrays['counts'][0,0,0]+=1;np.savez_compressed(path,**arrays)
                identity=checkpoint_identity(cp);manifest=json.loads((cells/'manifest.json').read_text())
                if mode=='manifest':manifest['seed']=0
                if mode=='identity':identity['checkpoint_sha256']='0'*64
                bind_scores(cells,identity,manifest)
                with patch.object(compare,'load_pins',return_value=pins),self.assertRaises(ValueError):compare.summarize(cand,cells,root/'bad')
                self.assertFalse((root/'bad').exists())

    def test_calibration_equivalence_explicit_origins_and_policy(self):
        base=dict(recipe_id='r',recipe_sha256='s',selected_pairs_sha256='p',full_pairs_sha256='p',geometry={'tile_size':256},policy='p',threshold_origin='post_hoc_256',cutoff=.2)
        candidate=dict(base,threshold_origin='train_matched',cutoff=.8)
        compare.compatible_calibration(base,candidate)
        for key in ('recipe_id','selected_pairs_sha256','geometry','policy','threshold_origin'):
            with self.subTest(key=key),self.assertRaises(ValueError):compare.compatible_calibration(base,dict(candidate,**{key:'wrong'}))

    def test_real_reference_artifact_pins_readonly(self):
        pins=compare.load_pins()
        for name,key in (('detector','evaluation'),('heatmap','folder')):
            folder=ROOT/pins[name][key]
            if not folder.exists():self.skipTest('local reference absent')
            compare.pinned(folder,pins[name]['files_sha256'])
            if name=='heatmap':validate_cutoffs(folder,json.loads((folder/'summary.json').read_text()),['aux'])
        self.assertEqual(len(pins['heatmap']['files_sha256']),7)


if __name__=='__main__':unittest.main()
