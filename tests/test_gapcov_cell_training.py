"""Focused synthetic CPU checks: run21 continuation, no dataset or GPU run."""
import copy
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch, Mock

import lightning as L
import numpy as np
import torch
from torch.nn import functional as F
from cvdl.cov_pool import MODEL, SPEC
from cvdl.curriculum import weights_only
from cvdl.cov_training import check_start
from cvdl.lit import SteganalysisModule
from cvdl.models import model_from_hparams, XuNetWideGAPCov32
from cvdl.cell_training import cell_loss
from cvdl.cell_diagnostics import cell_maps
from cvdl.photo_inference import infer_photo
from cvdl.settings import SCHEMA, load, machine_profile
from scripts.g4.preflight import top_values

ROOT=Path(__file__).resolve().parents[1]
BOOK=ROOT/'notebooks/experiments/srm-cell-aux-paired-gapcov32.ipynb'
H=dict(model=MODEL,front_end='srm',norm='none',width=4,clip_t=3.)
ON=dict(H,cell_supervision='changed_cells_v1',cell_loss_weight=.2,paired_batches=True,
        lr=3e-5,max_epochs=15,warmup_epochs=1,lr_drops=(),batch_size=128,val_batch_images=64)


def settings(**changes):
    values=top_values(json.loads(BOOK.read_text()))
    overrides=[f'{k}={v if isinstance(v,str) else json.dumps(v)}' for k,v in changes.items() if SCHEMA[k].machine]
    values.update({k:v for k,v in changes.items() if not SCHEMA[k].machine})
    return load(environ={},overrides=overrides,_profile=machine_profile('pc',device='cpu'),**values)


def checkpoint(module,**extra):
    return dict(state_dict=copy.deepcopy(module.state_dict()),hyper_parameters=dict(module.hparams),
                epoch=43,**{'pytorch-lightning_version':L.__version__},**extra)


def save(path,payload):
    torch.save(payload,path)
    return hashlib.sha256(path.read_bytes()).hexdigest()


class GAPCovCellTests(unittest.TestCase):
    def test_off_rng_state_outputs_attribution_and_one_pass_head(self):
        torch.manual_seed(81);off=XuNetWideGAPCov32().eval();rng=torch.get_rng_state()
        torch.manual_seed(81);on=model_from_hparams(dict(ON,cov_pool_identity=SPEC)).eval()
        self.assertTrue(torch.equal(rng,torch.get_rng_state()))
        for k,v in off.state_dict().items():self.assertTrue(torch.equal(v,on.state_dict()[k]),k)
        self.assertEqual(sum(p.numel() for p in on.cell_head.parameters()),257)
        self.assertEqual((on.cell_head.in_channels,on.classifier.in_features),(256,784))
        self.assertEqual((on.aux_receptive_field,on.cell_receptive_field),(48,256))
        x=torch.rand(1,3,256,256)
        with torch.inference_mode(),patch.object(on,'features',wraps=on.features) as features:
            logits,aux=on(x,return_aux=True)
            self.assertEqual(features.call_count,1);self.assertEqual(aux.shape,(1,1,8,8))
            torch.testing.assert_close(logits,off(x),rtol=0,atol=0)
        with torch.inference_mode():
            a,p=off.class_score_cells(x);b,q=on.class_score_cells(x)
            torch.testing.assert_close(a,b,rtol=0,atol=0);torch.testing.assert_close(p,q,rtol=0,atol=0)
        self.assertFalse(hasattr(off,'cell_head'))
        with self.assertRaises(ValueError):off(x,return_aux=True)

    def test_weights_only_exact_base_new_head_fresh_optimizer_and_refusals(self):
        with TemporaryDirectory() as tmp:
            path=Path(tmp)/'source.ckpt';source=SteganalysisModule(**H);target=SteganalysisModule(**ON)
            payload=checkpoint(source,optimizer_states=[{'not':'restored'}],lr_schedulers=[{'not':'restored'}])
            head={k:v.clone() for k,v in target.net.cell_head.state_dict().items()}
            sha=save(path,payload);record=weights_only(target,path,sha)
            self.assertEqual(record['source_epoch'],43)
            self.assertFalse(record['optimizer_restored']);self.assertFalse(record['scheduler_restored'])
            for k,v in source.net.state_dict().items():self.assertTrue(torch.equal(v,target.net.state_dict()[k]),k)
            for k,v in head.items():self.assertTrue(torch.equal(v,target.net.cell_head.state_dict()[k]))
            config=target.configure_optimizers();self.assertFalse(config['optimizer'].state)
            self.assertEqual(config['lr_scheduler']['scheduler'].last_epoch,0)
            with self.assertRaisesRegex(ValueError,'SHA-256'):weights_only(target,path,'0'*64)
            for mode in ('spec','bank','shape','missing','extra','prefix','source_head'):
                bad=copy.deepcopy(payload)
                if mode=='spec':bad['hyper_parameters']['cov_pool_identity']=dict(SPEC,iterations=6)
                if mode=='bank':bad['state_dict']['net.kv.conv.weight']+=.001
                if mode=='shape':bad['state_dict']['net.classifier.weight']=torch.zeros(2,256)
                if mode=='missing':del bad['state_dict']['net.conv1.bias']
                if mode=='extra':bad['state_dict']['net.unexpected']=torch.zeros(1)
                if mode=='prefix':bad['state_dict']['conv1.bias']=bad['state_dict'].pop('net.conv1.bias')
                if mode=='source_head':bad['hyper_parameters']['cell_supervision']='changed_cells_v1'
                before={k:v.clone() for k,v in target.state_dict().items()}
                with self.subTest(mode=mode),self.assertRaises(ValueError):weights_only(target,path,save(path,bad))
                for k,v in before.items():self.assertTrue(torch.equal(v,target.state_dict()[k]),k)
            target.net.kv.conv.weight.requires_grad_(True)
            with self.assertRaisesRegex(ValueError,'frozen'):weights_only(target,path,save(path,payload))
            off_path=Path(tmp)/'off.ckpt';save(off_path,payload)
            restored=SteganalysisModule.load_from_checkpoint(off_path,map_location='cpu',weights_only=False)
            x=torch.rand(1,3,64,64)
            with torch.no_grad():torch.testing.assert_close(restored.net(x),source.net(x),rtol=0,atol=0)

    def test_paired_ce_bce_backward_and_strict_reconstruction(self):
        module=SteganalysisModule(**ON);module.log=lambda *a,**k:None
        clean=torch.rand(1,3,256,256);stego=torch.rand_like(clean)
        tags=torch.zeros(1,2,130,dtype=torch.long);tags[:,:,66:]=1;tags[:,1,2]=16;tags[:,1,66]=0
        images=torch.cat((clean,stego));targets=torch.tensor([0,1]);flat=torch.cat((tags[:,0],tags[:,1]))
        tile,aux=module.net(images,return_aux=True)
        expected=F.cross_entropy(tile,targets)+.2*cell_loss(aux,flat)[0]
        with patch.object(module.net,'features',wraps=module.net.features) as features:
            result=module.training_step((clean,stego,tags),0)
            self.assertEqual(features.call_count,1)
        torch.testing.assert_close(result['loss'],expected,rtol=0,atol=0)
        self.assertEqual(result['count'],2);result['loss'].backward()
        for p in (module.net.conv1.weight,module.net.conv7.weight,module.net.cov_reduce.weight,module.net.classifier.weight,module.net.cell_head.weight):
            self.assertIsNotNone(p.grad);self.assertTrue(torch.isfinite(p.grad).all());self.assertGreater(float(p.grad.abs().max()),0)
        self.assertIsNone(module.net.kv.conv.weight.grad)
        with TemporaryDirectory() as tmp:
            path=Path(tmp)/'cell.ckpt';save(path,checkpoint(module))
            restored=SteganalysisModule.load_from_checkpoint(path,map_location='cpu',weights_only=False)
            with torch.no_grad():
                a,b=restored.net(images,return_aux=True)
                torch.testing.assert_close(a,tile,rtol=0,atol=0);torch.testing.assert_close(b,aux,rtol=0,atol=0)
            for key,value in (('cell_supervision','off'),('cell_loss_weight',.3),('cov_pool_identity',dict(SPEC,iterations=6))):
                bad=checkpoint(module);bad['hyper_parameters'][key]=value
                with self.subTest(key=key),self.assertRaises(ValueError):module.on_load_checkpoint(bad)

    def test_head_only_and_dual_photo_maps_one_pass_no_cam(self):
        net=model_from_hparams(dict(ON,cov_pool_identity=SPEC)).eval()
        photo=np.zeros((256,256,3),np.uint8);record=dict(source_height=256,source_width=256)
        with patch('torch.autograd.grad',side_effect=AssertionError('head-only requested gradients')),patch.object(net,'features',wraps=net.features) as features:
            _,scores,head=infer_photo(net,photo,record,'cpu',1,map_kinds=('trained_head',))
            self.assertEqual(features.call_count,1)
        with patch.object(net,'features',wraps=net.features) as features:
            _,dual_scores,dual=infer_photo(net,photo,record,'cpu',1,map_kinds=('trained_head','attribution'))
            self.assertEqual(features.call_count,1)
        np.testing.assert_allclose(head,dual[:1],rtol=0,atol=0)
        np.testing.assert_allclose(scores,dual_scores,rtol=1e-6,atol=1e-7)
        self.assertEqual(dual.shape,(2,8,8));self.assertTrue(np.isfinite(dual).all())
        with self.assertRaisesRegex(ValueError,'not CAM'):infer_photo(net,photo,record,'cpu',1)
        self.assertTrue(all(p.grad is None for p in net.parameters()))

    def test_runtime_warm_start_not_overwritten_and_resume_guard_before_writes(self):
        from cvdl.runtime import create_training
        with TemporaryDirectory() as tmp:
            root=Path(tmp);source=root/'source.ckpt';sha=save(source,checkpoint(SteganalysisModule(**H)))
            s=settings(init_checkpoint=str(source),init_checkpoint_sha256=sha,output_root=str(root),compile='off',headless=True)
            data=SimpleNamespace(checkpoint_validation='train_matched',validation_pair_csv='toy\n',
                save_record=Mock(),corpus=SimpleNamespace(save_training_plan=Mock(),settings=SimpleNamespace(sources=('raise','lsdir','alaska'))))
            loaders=SimpleNamespace(train=[None],val=[None])
            # Build the real CPU Trainer but do not fit, read data, or launch workers.
            with patch('cvdl.curriculum.initialize_scratch50',side_effect=AssertionError('warm start overwritten')):
                run=create_training(s,data,loaders,trainer_options={'enable_model_summary':False})
            folder=run.checkpoint_dir;start=json.loads((folder/'training_start.json').read_text());record=json.loads((folder/'model_run_settings.json').read_text())
            self.assertTrue(start['weights_only']);self.assertEqual(record['initialization'],'weights_only')
            self.assertEqual(start['initial_weights_sha256'],record['initial_weights_sha256'])
            self.assertEqual(start['initial_cell_head_sha256'],record['initial_cell_head_sha256'])
            from cvdl.gapcov_provenance import validate_sources,CELL_AUX
            self.assertEqual(start['source_provenance'],validate_sources(s.root,CELL_AUX))
            self.assertEqual(record['source_provenance'],start['source_provenance'])
            self.assertEqual(run.lightning_model.hparams.source_provenance,start['source_provenance'])
            module=run.lightning_model;module.log=lambda *a,**k:None
            optimizer=module.configure_optimizers();opt=optimizer['optimizer'];sched=optimizer['lr_scheduler']['scheduler']
            tags=torch.zeros(1,2,130,dtype=torch.long);tags[:,:,66:]=1
            module.training_step((torch.rand(1,3,256,256),torch.rand(1,3,256,256),tags),0)['loss'].backward();opt.step();sched.step()
            payload=checkpoint(module,global_step=1,optimizer_states=[opt.state_dict()],lr_schedulers=[sched.state_dict()])
            last=folder/'last.ckpt';save(last,payload)
            reconstructed=SteganalysisModule.load_from_checkpoint(last,map_location='cpu',weights_only=False)
            for key,value in module.state_dict().items():self.assertTrue(torch.equal(value,reconstructed.state_dict()[key]),key)
            self.assertTrue(check_start(s,folder))
            with patch('cvdl.curriculum.initialize_scratch50',side_effect=AssertionError('resume overwritten')):
                resumed=create_training(s,data,loaders,trainer_options={'enable_model_summary':False})
            self.assertEqual(json.loads((folder/'training_start.json').read_text()),start)
            self.assertEqual(resumed.lightning_model.hparams.initial_weights_sha256,start['initial_weights_sha256'])
            for mode in ('loss','head','spec','bank','init','recipe','optimizer','source'):
                bad=copy.deepcopy(payload)
                if mode=='loss':bad['hyper_parameters']['cell_loss_weight']=.3
                if mode=='head':bad['state_dict']['net.cell_head.weight']=torch.zeros(1,784,1,1)
                if mode=='spec':bad['hyper_parameters']['cov_pool_identity']=dict(SPEC,iterations=6)
                if mode=='bank':bad['state_dict']['net.kv.conv.weight']+=.001
                if mode=='init':bad['hyper_parameters']['initialization_checkpoint_sha256']='0'*64
                if mode=='recipe':bad['hyper_parameters']['lr']=3e-4
                if mode=='optimizer':bad['optimizer_states']=[]
                if mode=='source':bad['hyper_parameters']['source_provenance']['version']='run21-original-v1'
                save(last,bad)
                before={p:p.read_bytes() for p in folder.glob('*.json')}
                with self.subTest(mode=mode),self.assertRaises((ValueError,RuntimeError)):
                    create_training(s,data,loaders,trainer_options={'enable_model_summary':False})
                for p,content in before.items():self.assertEqual(p.read_bytes(),content,p.name)
            save(last,payload)
            for name in ('training_start.json','model_run_settings.json'):
                path=folder/name;original=path.read_text();wrong=json.loads(original)
                wrong['source_provenance']['historical']['revision']='0'*40
                path.write_text(json.dumps(wrong));before={p:p.read_bytes() for p in folder.glob('*.json')}
                with self.subTest(record=name),self.assertRaisesRegex(ValueError,'source provenance mismatch'):
                    create_training(s,data,loaders,trainer_options={'enable_model_summary':False})
                for p,content in before.items():self.assertEqual(p.read_bytes(),content,p.name)
                path.write_text(original)
            read=Path.read_bytes
            def changed(path):
                content=read(path)
                return content+b'\n# source drift\n' if path.resolve()==ROOT/'cvdl/models.py' else content
            before={p:p.read_bytes() for p in folder.glob('*.json')}
            with patch.object(Path,'read_bytes',changed):
                with self.assertRaisesRegex(ValueError,'current source drift'):
                    create_training(s,data,loaders,trainer_options={'enable_model_summary':False})
            for p,content in before.items():self.assertEqual(p.read_bytes(),content,p.name)

    def test_local_run21_best_readonly_load_and_exact_warm_start(self):
        native=ROOT/'runs/srm-e50-paired-gapcov32-20261026T115421Z-1917/checkpoints/srm-e50-paired-gapcov32-20261026T115421Z-1917/best-epoch43-val_pe0.0329-val_auc0.9946.ckpt'
        if not native.is_file():self.skipTest('local run21 BEST not present')
        before=hashlib.sha256(native.read_bytes()).hexdigest()
        self.assertEqual(before,'4ba2a4b59a535b53ab9c0235c54231476b45e2193982bf28f7b7f2171d9898cf')
        from cvdl.standardtest import load_network
        off=load_network(native,'cpu');on=SteganalysisModule(**ON)
        weights_only(on,native,before)
        for key,value in off.state_dict().items():self.assertTrue(torch.equal(value,on.net.state_dict()[key]),key)
        x=torch.rand(1,3,256,256)
        with torch.inference_mode():torch.testing.assert_close(off(x),on.net(x),rtol=0,atol=0)
        self.assertEqual(hashlib.sha256(native.read_bytes()).hexdigest(),before)

    def test_notebook_recipe_and_native_initializer_override(self):
        s=settings()
        self.assertEqual((s.epochs,s.lr,s.warmup_epochs,s.lr_drops),(15,3e-5,1,()))
        self.assertEqual((s.batch_size,s.val_batch_images,s.tiles_per_epoch),(128,64,180000))
        self.assertEqual((s.embedding_schedule,s.evaluation_checkpoint,s.validation_recipe),('original','last','60da4391043e8ce48ff2'))
        self.assertEqual(dict(s.source_mix),{'raise':.4,'lsdir':.2,'alaska':.4})
        native=str(ROOT/str(s.init_checkpoint).replace('runs-out/','runs/',1))
        self.assertEqual(settings(init_checkpoint=native).init_checkpoint,native)
        for cell in json.loads(BOOK.read_text())['cells']:
            if cell['cell_type']=='code':compile(''.join(cell['source']),str(BOOK),'exec')


if __name__=='__main__':unittest.main()
