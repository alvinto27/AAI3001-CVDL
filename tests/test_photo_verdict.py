"""CPU checks for frozen whole-photo aggregation."""
import json
import hashlib
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from cvdl.photo_verdict import (features, largest_group, fit_logistic, predict,
                                calibrate_count, evaluate_scope)
from cvdl.cell_metrics import fa_cutoff


class PhotoVerdictTests(unittest.TestCase):
    def test_synthetic_features(self):
        grid=np.array([[1,0,0],[0,1,0]])
        result,counts=features([.1,.6,.8,.9],grid,.5,.5)
        self.assertEqual(result['cell_fraction'],2/6)
        self.assertEqual(result['group_fraction'],2/6)
        self.assertEqual(result['log_group_count'],np.log1p(2))
        self.assertEqual(result['log_cell_count'],np.log1p(6))
        self.assertEqual(result['max_tile'],.9)
        self.assertAlmostEqual(result['top3_tile'],(.6+.8+.9)/3)
        self.assertEqual(result['tile_fraction'],.75)
        self.assertEqual(result['log_tile_count'],np.log1p(4))
        self.assertEqual(counts['tiles_above'],3)
        result,_=features([.5],np.zeros((1,1)),.5,.5)
        self.assertEqual(result['top3_tile'],.5)
        self.assertEqual(result['log_group_count'],0)
        self.assertEqual(result['group_fraction'],0)

    def test_groups_diagonal_and_empty(self):
        self.assertEqual(largest_group(np.eye(4)),4)
        self.assertEqual(largest_group([[1,0,0,1],[0,1,0,1]]),2)
        self.assertEqual(largest_group(np.zeros((3,4))),0)
        self.assertEqual(largest_group(np.ones((2,3))),6)

    def test_cutoff_ties_and_smallest_integer_count(self):
        scores=np.r_[np.zeros(212),np.ones(12)]
        cutoff,fa=fa_cutoff(scores)
        self.assertEqual(cutoff,np.nextafter(1.,np.inf))
        self.assertEqual(fa,0)
        n,fa=calibrate_count(scores)
        self.assertEqual(n,2)
        self.assertEqual(fa,0)
        counts=np.r_[np.zeros(213),np.full(11,3)]
        n,fa=calibrate_count(counts)
        self.assertEqual(n,1)
        self.assertEqual(fa,11/224)
        self.assertGreater((counts>=n-1).mean(),.05)

    def test_logistic_toy_cpu(self):
        x=np.array([[-3,1],[-2,1],[-1,1],[1,1],[2,1],[3,1]],float)
        y=np.array([0,0,0,1,1,1])
        model=fit_logistic(x,y,np.ones(6))
        self.assertEqual(model['scale'][1],1)
        self.assertGreater(model['coefficients'][0],0)
        self.assertLess(abs(model['coefficients'][1]),1e-10)
        self.assertLess(abs(model['intercept']),1e-7)
        np.testing.assert_array_equal(predict(model,x)>=0,y)
        self.assertLessEqual(model['iterations'],1000)

    def test_photo_balancing_and_paired_differences(self):
        rows=[]
        for photo,variants in [('one',1),('two',3)]:
            rows.append(dict(photo_id=photo,panel='panel',label=0,a=0,b=0,c=0))
            for i in range(variants):
                rows.append(dict(photo_id=photo,panel='panel',label=1,a=i,b=i,c=2))
        result=evaluate_scope(rows,dict(a=1,b=1,c=1),replicates=20)
        self.assertEqual(result['rules']['a']['FA'],0)
        self.assertAlmostEqual(result['rules']['a']['miss'],2/3)
        self.assertEqual(result['rules']['c']['P_E'],0)
        self.assertEqual(result['rules']['c']['AUC'],1)
        self.assertEqual(result['differences']['c-a'],result['differences']['c-b'])

    def test_scope_bootstrap_draws_full_population(self):
        rows=[dict(photo_id='eligible',panel='p',label=0,a=0,b=0,c=0),
              dict(photo_id='eligible',panel='p',label=1,a=1,b=1,c=1)]
        result=evaluate_scope(rows,dict(a=.5,b=.5,c=.5),replicates=100,
            population={'eligible':'p','ineligible':'p'})
        # Some full-population samples contain no eligible partner. They must
        # be ineligible replicates, not replaced by draws from eligible only.
        self.assertLess(result['rules']['c']['P_E_interval']['valid_replicates'],100)
        self.assertEqual(result['rules']['c']['P_E'],0)

    def test_all_12_forced_draw_settings(self):
        from dataclasses import replace
        from lsb_batch import read_config
        from lsb_settings import draw_otf_settings
        root=Path(__file__).resolve().parents[1]
        config=read_config(root/'configs/train-k1to4.yaml')
        image=np.zeros((256,256,3),dtype=np.uint8)
        for k in range(1,5):
            for placement in ('contiguous','random','stride'):
                cfg=replace(config,k=k,placement=placement,stride=config.stride if placement=='stride' else 1)
                kwargs,metadata=draw_otf_settings(cfg,image,np.random.default_rng(k))
                self.assertEqual(kwargs['k'],k)
                self.assertEqual(kwargs['placement'],placement)
                self.assertTrue(2<=kwargs['stride']<=16 if placement=='stride' else kwargs['stride']==1)
                self.assertTrue(.05<=metadata['requested_bpp']<=.5)
                self.assertLessEqual(metadata['effective_bpp'],metadata['requested_bpp']+8/(256*256))

    def test_fit_freezes_without_opening_test(self):
        from scripts.photo_verdict_fit import fit
        from cvdl.photo_verdict import FEATURES
        rows=[]
        for photo in range(224):
            for variant in range(13):
                value=(photo%7)/100+bool(variant)
                rows.append(dict(label=int(bool(variant)),features={f:value for f in FEATURES},
                    counts=dict(tiles_above=int(bool(variant)))))
        with tempfile.TemporaryDirectory() as directory:
            output=Path(directory)
            (output/'photo-list.json').write_text(json.dumps(dict(tile_cutoff=.5,head_cutoff=.3)))
            (output/'cache-identity.json').write_text('{}')
            with patch('scripts.photo_verdict_fit.load_partition',return_value=(rows,{})) as loader:
                fit(output)
                self.assertEqual([call.args[2] for call in loader.call_args_list],['fit','calibration'])
                frozen=json.loads((output/'frozen-rules.json').read_text())
                self.assertEqual(frozen['rules']['a']['N'],1)
                self.assertEqual(frozen['rules']['c']['features'],list(FEATURES))
                self.assertEqual(frozen['fit_photos'],224)
                self.assertTrue(frozen['rules']['c']['calibration_clean_FA']<=.05)
                with self.assertRaises(ValueError):fit(output)

    def test_legacy_and_run18_selection_hashes_keep_every_photo_split_and_seed(self):
        from scripts.photo_verdict_cache import selection,TILE_CUTOFF,HEAD_CUTOFF
        legacy=selection()
        def digest(value):return hashlib.sha256((json.dumps(value,indent=2)+'\n').encode()).hexdigest()
        self.assertEqual(digest(legacy),'c0219972978362096e5759ce0cdb9b2eb720b6b9923e70beb6d672fc1b35a2da')
        expected=dict(legacy,tile_cutoff=.5094263553619385,head_cutoff=.1966136991977692)
        self.assertEqual(selection(tile_cutoff=expected['tile_cutoff'],head_cutoff=expected['head_cutoff']),expected)
        self.assertEqual(digest(expected),'ba66f7669f962fcf4d93d3f56f87c961a3ca02bcaf9cbc20ef2388ddc80c5b25')
        self.assertEqual((legacy['tile_cutoff'],legacy['head_cutoff']),(TILE_CUTOFF,HEAD_CUTOFF))
        for partition,count,tiles in (('fit',224,15802),('calibration',224,15993),('test',576,33023)):
            photos=[p for p in legacy['photos'] if p['partition']==partition]
            self.assertEqual((len(photos),sum(p['tiles'] for p in photos)),(count,tiles))

    def test_own_cutoff_sources_unrounded_checkpoint_bound_and_selected_map(self):
        from scripts.photo_verdict_cache import collection_inputs,cutoff_sources,file_hash
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);run=root/'own-run';run.mkdir()
            checkpoint=run/'last.ckpt';checkpoint.write_bytes(b'weights')
            evaluation=run/'evaluation/standard-256-v1/own-eval/validation';evaluation.mkdir(parents=True)
            tile=evaluation/'calibration.json';cell=root/'summary.json'
            data=dict(checkpoint_sha256=file_hash(checkpoint),deployment=True,quick=False,
                geometry=dict(tile_size=256,grid='overlap_last'),cutoff=.5094263553619385,threshold_id='own')
            aux=dict(cutoff=.1966136991977692,validation_FA=.049962105754095494,validation_negative_cells=17153)
            cam=dict(aux,cutoff=7.109206199645997)
            def write():
                tile.write_text(json.dumps(data));cell.write_text(json.dumps(dict(maps=dict(aux=aux,cam=cam))))
            def resolve(kind='trained_head',explicit=None):return collection_inputs(run,'own-eval',cell,kind,explicit)
            write()
            self.assertEqual(resolve(),(checkpoint,data['cutoff'],aux['cutoff']))
            self.assertEqual(resolve('cam',checkpoint),(checkpoint,data['cutoff'],cam['cutoff']))
            sources=cutoff_sources(run,'own-eval',cell)
            self.assertEqual(sources,dict(tile_threshold_id='own',
                tile_calibration=dict(path=str(tile),sha256=file_hash(tile)),
                cell_summary=dict(path=str(cell),sha256=file_hash(cell))))
            checkpoint.write_bytes(b'other weights')
            with self.assertRaisesRegex(ValueError,'checkpoint/calibration'):resolve(explicit=checkpoint)
            with self.assertRaisesRegex(ValueError,'no evaluated checkpoint'):resolve()
            checkpoint.write_bytes(b'weights')
            for field,bad in (('quick',True),('deployment',False),('geometry',dict(tile_size=512,grid='overlap_last'))):
                old=data[field];data[field]=bad;write()
                with self.assertRaisesRegex(ValueError,'deployment calibration'):resolve()
                data[field]=old
            write()
            for field,bad in (('validation_FA',.06),('validation_negative_cells',0)):
                old=aux[field];aux[field]=bad;write()
                with self.assertRaisesRegex(ValueError,'5%'):resolve()
                aux[field]=old
            for field,bad in (('cutoff',float('nan')),('cutoff',1.1)):
                old=data[field];data[field]=bad;write()
                with self.assertRaisesRegex(ValueError,'invalid tile/cell'):resolve()
                data[field]=old
            write()
            with self.assertRaisesRegex(ValueError,'unsupported map kind'):resolve('unknown')
            cell.write_text(json.dumps(dict(maps=dict(cam=dict(cutoff=cam['cutoff'])))))
            self.assertEqual(resolve('cam'),(checkpoint,data['cutoff'],cam['cutoff']))

    def test_collect_native_identity_additive_sources_cpu_validation_only_and_map(self):
        from types import SimpleNamespace
        from scripts.photo_verdict_cache import collect,dump,file_hash
        for kind in ('trained_head','cam'):
            with self.subTest(map_kind=kind),tempfile.TemporaryDirectory() as directory:
                output=Path(directory);checkpoint=output/'last.ckpt';checkpoint.write_bytes(b'weights')
                tile=output/'calibration.json';tile.write_text('{}')
                cell=output/'summary.json';cell.write_text('{}')
                sources=dict(tile_threshold_id='own',tile_calibration=dict(path=str(tile),sha256=file_hash(tile)),
                    cell_summary=dict(path=str(cell),sha256=file_hash(cell)))
                selected=dict(partition='fit',panel='unit',record=dict(photo_id='unit',camera_model='unit'),
                    versions=[dict(variant=0,k=0,placement='clean')])
                frozen=dict(tile_cutoff=.5094263553619385,head_cutoff=.1966136991977692,
                    photos=[selected,dict(selected,partition='test')])
                dump(output/'photo-list.json',frozen)
                network=SimpleNamespace(cell_head=object()) if kind=='trained_head' else SimpleNamespace()
                with patch('scripts.pc_safety.own_process_group'),patch('scripts.pc_safety.check_workers'), \
                     patch('lsb_source.load_photo',return_value=np.zeros((256,256,3),np.uint8)) as load, \
                     patch('cvdl.standardtest.load_network',return_value=network) as load_network, \
                     patch('scripts.photo_verdict_cache.infer_photo',return_value=([(0,0)],[.51],np.array([[[.2]]]))) as infer:
                    collect(output,frozen,checkpoint=checkpoint,evaluation_id='own-eval',map_kind=kind,
                        device='cpu',validation_only=True,sources=sources,run_name='own-run')
                load_network.assert_called_once_with(checkpoint,'cpu');load.assert_called_once()
                self.assertEqual(infer.call_args.args[-2:],('cpu',8))
                self.assertEqual(infer.call_args.kwargs,dict(map_kinds=(kind,)))
                self.assertFalse((output/'features/test').exists())
                row=json.loads((output/'features/fit/unit.json').read_text())['rows'][0]
                self.assertEqual(row['counts']['tiles_above'],1);self.assertEqual(row['features']['cell_fraction'],1.)
                identity=json.loads((output/'cache-identity.json').read_text())
                self.assertEqual((identity['batch_size'],identity['workers']),(8,0))
                self.assertEqual((identity['tile_cutoff'],identity['head_cutoff']),(frozen['tile_cutoff'],frozen['head_cutoff']))
                self.assertEqual((identity['evaluation_id'],identity['map_kind']),('own-eval',kind))
                self.assertEqual(identity['cutoff_sources'],sources);self.assertEqual(identity['run_name'],'own-run')

    def test_collect_rechecks_checkpoint_and_sources_without_rewriting_old_identity(self):
        from types import SimpleNamespace
        from scripts.photo_verdict_cache import collect,dump,file_hash
        for changed in ('checkpoint','tile_calibration','cell_summary','old_yx_identity'):
            with self.subTest(changed=changed),tempfile.TemporaryDirectory() as directory:
                output=Path(directory);checkpoint=output/'last.ckpt';checkpoint.write_bytes(b'weights')
                sources={}
                for name in ('tile_calibration','cell_summary'):
                    path=output/(name+'.json');path.write_text('{}')
                    sources[name]=dict(path=str(path),sha256=file_hash(path))
                frozen=dict(tile_cutoff=.5,head_cutoff=.2,photos=[]);dump(output/'photo-list.json',frozen)
                identity_path=output/'cache-identity.json'
                if changed=='old_yx_identity':
                    dump(identity_path,dict(selection_sha256=file_hash(output/'photo-list.json'),
                        checkpoint_sha256=file_hash(checkpoint),batch_size=8,workers=0,
                        tile_cutoff=.5,head_cutoff=.2,cutoff_sources=sources))
                    before=identity_path.read_bytes()
                def network(*args):
                    path=checkpoint if changed=='checkpoint' else Path(sources[changed]['path'])
                    path.write_text('changed')
                    return SimpleNamespace(cell_head=object())
                with patch('scripts.pc_safety.own_process_group'),patch('scripts.pc_safety.check_workers'), \
                     patch('cvdl.standardtest.load_network',side_effect=network) as load:
                    pattern='cache identity mismatch' if changed=='old_yx_identity' else 'changed during cache build'
                    with self.assertRaisesRegex(ValueError,pattern):
                        collect(output,frozen,checkpoint=checkpoint,evaluation_id='own',device='cpu',sources=sources)
                if changed=='old_yx_identity':
                    load.assert_not_called();self.assertEqual(identity_path.read_bytes(),before)

    def test_score_refuses_unfrozen_test(self):
        from scripts.photo_verdict_fit import score
        with tempfile.TemporaryDirectory() as directory:
            output=Path(directory)
            with patch('scripts.photo_verdict_fit.load_partition') as loader:
                with self.assertRaises(FileNotFoundError):score(output)
                loader.assert_not_called()


if __name__=='__main__':unittest.main()
