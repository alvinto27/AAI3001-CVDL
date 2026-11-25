"""Small CPU checks for the opt-in cell-target path, not another model search."""
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np
import torch
import lightning as L
from torch.nn import functional as F
from cvdl.curriculum import CurriculumImages, weights_only
from cvdl.lit import SteganalysisModule
from cvdl.models import XuNetWide, model_from_hparams
from cvdl.cell_training import cell_loss
from cvdl.cell_diagnostics import MANIFEST, select_manifest, selected_rows, replay_cell_pair, cell_maps, render_cell_figure
from cvdl.settings import load, machine_profile
from lsb_batch import read_config
from lsb_cells import changed_cell_targets
from lsb_mode1 import TileSpec, OTFSeed, make_sample
from lsb_torch import Corpus, RunSettings, image_tensor, training_loader

ROOT=Path(__file__).resolve().parents[1]
ENABLED=dict(model='xunet_wide',width=4,front_end='srm',norm='none',cell_supervision='changed_cells_v1')


class CellToyCorpus(Corpus):
    def photo(self,record):
        return np.random.default_rng(int(record['photo_index'])).integers(0,256,(512,512,3),dtype=np.uint8)


def corpus():
    records=[dict(photo_id=f'{s}_toy',photo_index=str(i),source=s,split='train',camera_model='toy',
        source_height='512',source_width='512',channels='3',grid_policy='overlap_last')
        for i,s in enumerate(('raise','lsdir','alaska'))]
    return CellToyCorpus(ROOT,RunSettings(sources=('raise','lsdir','alaska'),tile_size=512,
        settings_yaml=str(ROOT/'configs/train-k1to4.yaml'),source_mix={'raise':.4,'lsdir':.2,'alaska':.4},
        tiles_per_epoch=20,buffer_size=10,tensor_dtype='uint8'),records=records,train_grid='overlap_last',train_tile=256)


class CellTrainingTests(unittest.TestCase):
    def test_changed_unit_mapping_halo_crop_padding_and_actual_changes(self):
        # Multiple changed channels still describe one changed spatial pixel.
        units=np.array([0,1,2,(31*256+31)*3,(32*256+32)*3,256*256*3-1])
        counts,valid=changed_cell_targets(units)
        mask=np.zeros((256,256),bool);pixels=units//3;mask[pixels//256,pixels%256]=True
        self.assertEqual(int(counts.sum()),4)
        for r in range(8):
            for c in range(8):
                self.assertEqual(counts[r,c],mask[r*32:r*32+32,c*32:c*32+32].sum())
                self.assertEqual(valid[r,c],not mask[max(0,r*32-8):min(256,r*32+40),max(0,c*32-8):min(256,c*32+40)].any())
        c=corpus();record=c.photos[0];photo=c.photo(record);config=c.config
        for grid in ('train_overlap_v1','centre_pad'):
            if grid=='centre_pad':
                photo=photo[:300,:300];record={**record,'source_height':'300','source_width':'300'}
            spec=TileSpec(grid,256,0,0);seed=OTFSeed(0,'train',0)
            clean,cm,ct=make_sample(photo,record,spec,config,seed,'clean',return_cell_targets=True)
            stego,sm,st=make_sample(photo,record,spec,config,seed,'stego',return_cell_targets=True)
            self.assertFalse(ct[0].any());self.assertTrue(ct[1].all())
            different=(clean!=stego).any(2)
            expected=np.array([[different[r*32:r*32+32,k*32:k*32+32].sum() for k in range(8)] for r in range(8)])
            np.testing.assert_array_equal(st[0],expected)
            # Defaults still return the same pixels, JSON-safe metadata and keys.
            off,metadata=make_sample(photo,record,spec,config,seed,'stego')
            np.testing.assert_array_equal(off,stego);self.assertEqual(metadata,sm);json.dumps(sm)
        empty,negative=changed_cell_targets([])
        self.assertFalse(empty.any());self.assertTrue(negative.all())

    def test_same_flips_rotations_align_images_and_cell_targets(self):
        image=np.zeros((256,256,3),np.uint8);image[31,27]=255
        counts,valid=changed_cell_targets(np.array([(31*256+27)*3]))
        for horizontal in (False,True):
            for vertical in (False,True):
                for turns in range(4):
                    with patch('lsb_torch.torch.rand',side_effect=[torch.tensor(0. if horizontal else 1.),torch.tensor(0. if vertical else 1.)]),patch('lsb_torch.torch.randint',return_value=torch.tensor(turns)):
                        pixels,(got,negative)=image_tensor(image,[1,2],1,True,torch.uint8,cell_targets=(counts,valid))
                    changed=torch.nonzero(pixels.any(0)).numpy()
                    units=(changed[:,0]*256+changed[:,1])*3
                    expected,expected_negative=changed_cell_targets(units)
                    np.testing.assert_array_equal(got,expected);np.testing.assert_array_equal(negative,expected_negative)

    def test_batch_tags_survive_mixed_sources_workers_packing_and_tail(self):
        snapshots=[]
        for workers,pack in ((0,1),(2,1),(2,4)):
            c=corpus();dataset=CurriculumImages(c,'original',5,cell_supervision='changed_cells_v1')
            loader=training_loader(dataset,batch_size=16,num_workers=workers,pin_memory=False,pack_blocks=pack)
            try:
                batches=list(loader);self.assertEqual([len(b[1]) for b in batches],[16,16,8])
                images,labels,tags=(torch.cat([b[i] for b in batches]) for i in range(3))
                self.assertEqual(tuple(tags.shape),(40,130));self.assertTrue((tags[:,1]==1).all())
                expected={int(tag[0]):(im,label,tag) for im,label,tag in list(dataset)}
                for im,label,tag in zip(images,labels,tags):
                    ei,el,et=expected[int(tag[0])]
                    self.assertTrue(torch.equal(im,ei));self.assertEqual(label,el);self.assertTrue(torch.equal(tag,et))
                    if int(label)==0:
                        self.assertFalse(tag[2:66].any());self.assertTrue((tag[66:]==1).all())
                snapshots.append((images,labels,tags))
            finally:loader.shutdown()
        for a,b in zip(snapshots[1],snapshots[2]):self.assertTrue(torch.equal(a,b))

    def test_cell_loss_missing_classes_sparse_weight_and_finite_gradients(self):
        tags=torch.zeros((3,130),dtype=torch.long)
        tags[0,66:]=1                         # clean / only negatives
        tags[1,2:66]=1                       # only sparse positives
        tags[2,2]=1;tags[2,67:]=1             # both classes
        logits=torch.zeros((3,1,8,8),requires_grad=True)
        loss,stats=cell_loss(logits,tags)
        expected=np.log(2)*(1+1/16+(1+1/16)/2)/3
        self.assertAlmostEqual(float(loss.detach()),expected,places=6)
        loss.backward();self.assertTrue(torch.isfinite(logits.grad).all())
        self.assertAlmostEqual(float(stats['positive_cells']),(0+64+1)/3,places=5)
        module=SteganalysisModule(**ENABLED)
        module.log=lambda *a,**k:None
        tag=tags[:1]
        out=module.training_step((torch.rand(1,3,256,256),torch.zeros(1,dtype=torch.long),tag),0)
        self.assertTrue(torch.isfinite(out['loss']));out['loss'].backward()
        self.assertTrue(torch.isfinite(module.net.cell_head.weight.grad).all())
        self.assertIsNotNone(module.net.conv1.weight.grad);self.assertIsNone(module.net.kv.conv.weight.grad)

    def test_warm_start_only_head_missing_and_strict_checkpoint_reconstruction(self):
        with TemporaryDirectory() as folder:
            source=SteganalysisModule(model='xunet_wide',front_end='srm',norm='none')
            target=SteganalysisModule(**ENABLED)
            path=Path(folder)/'start.ckpt'
            def save(state):
                torch.save(dict(state_dict=state,hyper_parameters=dict(source.hparams),epoch=43),path)
                return hashlib.sha256(path.read_bytes()).hexdigest()
            sha=save(source.state_dict());head=target.net.cell_head.weight.detach().clone()
            weights_only(target,path,sha)
            self.assertTrue(torch.equal(head,target.net.cell_head.weight))
            for k,v in source.net.state_dict().items():self.assertTrue(torch.equal(v,target.net.state_dict()[k]))
            for bad in ({k:v for k,v in source.state_dict().items() if k!='net.conv1.bias'},
                        {**source.state_dict(),'net.unexpected':torch.zeros(1)}):
                with self.assertRaisesRegex(ValueError,'architecture mismatch'):weights_only(target,path,save(bad))
            saved=Path(folder)/'cell.ckpt'
            torch.save(dict(state_dict=target.state_dict(),hyper_parameters=dict(target.hparams),
                **{'pytorch-lightning_version':L.__version__}),saved)
            restored=SteganalysisModule.load_from_checkpoint(saved,map_location='cpu',weights_only=False,strict=True)
            # This is the exact strict reconstruction used by the active evaluator.
            raw=torch.load(saved,map_location='cpu',weights_only=False)
            evaluated=model_from_hparams(raw['hyper_parameters'])
            evaluated.load_state_dict({k.removeprefix('net.'):v for k,v in raw['state_dict'].items()},strict=True)
            x=torch.rand(1,3,256,256)
            with torch.no_grad():
                tile,aux=restored.net(x,return_aux=True)
                self.assertEqual(tuple(aux.shape),(1,1,8,8));torch.testing.assert_close(tile,evaluated(x),rtol=0,atol=0)
            self.assertEqual(sum(p.numel() for p in restored.net.cell_head.parameters()),257)

    def test_compiled_forward_returns_aux_and_keeps_state_names(self):
        # AOT eager captures both forward and backward; no GPU/Inductor workers.
        net=model_from_hparams(ENABLED);keys=list(net.state_dict());net.compile(backend='aot_eager',fullgraph=True)
        x=torch.rand(1,3,256,256)
        tile,aux=net(x,return_aux=True)
        (tile.square().mean()+aux.square().mean()).backward()
        self.assertTrue(torch.isfinite(net.cell_head.weight.grad).all())
        torch.testing.assert_close(tile,net(x),rtol=1e-5,atol=1e-6)
        self.assertEqual(keys,list(net.state_dict()))
        net._compiled_call_impl=None
        torch._dynamo.reset()

    def test_feature_off_existing_training_path_pixels_order_rng_and_recipe(self):
        torch.manual_seed(17);reference=XuNetWide(norm='none',front_end='srm');rng=torch.get_rng_state()
        torch.manual_seed(17);module=SteganalysisModule(model='xunet_wide',norm='none',front_end='srm')
        self.assertTrue(torch.equal(rng,torch.get_rng_state()))
        for k,v in reference.state_dict().items():self.assertTrue(torch.equal(v,module.net.state_dict()[k]))
        self.assertNotIn('cell_supervision',module.hparams);self.assertNotIn('cell_loss_weight',module.hparams)
        self.assertFalse(hasattr(module.net,'cell_head'))
        torch.manual_seed(17);enabled=SteganalysisModule(**ENABLED)
        self.assertTrue(torch.equal(rng,torch.get_rng_state()))
        off=CurriculumImages(corpus(),'original',5);on=CurriculumImages(corpus(),'original',5,cell_supervision='changed_cells_v1')
        for a,b in zip(list(off),list(on),strict=True):
            self.assertTrue(torch.equal(a[0],b[0]));self.assertEqual(a[1],b[1]);self.assertTrue(torch.equal(a[2],b[2][:2]))
        from scripts.g4.preflight import top_values
        settings=[]
        for name in ('srm-low-bpp-control','srm-cell-aux'):
            book=json.loads((ROOT/f'notebooks/experiments/{name}.ipynb').read_text())
            s=load(environ={},_profile=machine_profile('g4',device='cpu',precision_limits=('32',)),**top_values(book))
            settings.append(dict(s.values))
        self.assertEqual({k for k in settings[1] if settings[1][k]!=settings[0].get(k)},
            {'name','run_name','cell_supervision','cell_loss_weight'})
        self.assertNotIn('cell_supervision',settings[0]);self.assertNotIn('cell_loss_weight',settings[0])
        x=torch.rand(1,3,256,256);y=torch.tensor([1]);module.log=lambda *a,**k:None
        self.assertTrue(torch.equal(module.training_step((x,y,torch.tensor([0])),0)['loss'],F.cross_entropy(reference(x),y)))

    def test_frozen_pair_ids_caps_and_label_replay(self):
        frozen=json.loads((ROOT/MANIFEST).read_text());self.assertEqual(frozen,select_manifest(ROOT))
        self.assertEqual(len(selected_rows(ROOT,'val')),192);self.assertEqual(len(selected_rows(ROOT,'test')),256)
        rows=selected_rows(ROOT,'test');selected=next(r for r in rows if r['pair_id']==frozen['figure'][0]['pair_id'])
        # Synthetic photo data uses the same replay path without a large dataset read.
        record=dict(photo_id=selected['source_photo'],source='raise',split='test',photo_index=selected['photo_index'],
            source_height=str(int(selected['y'])+256),source_width=str(int(selected['x'])+256),channels='3')
        class Stub:
            by_id={selected['source_photo']:record}
            def photo(self,record):return np.zeros((int(record['source_height']),int(record['source_width']),3),np.uint8)
        clean,stego,counts,valid=replay_cell_pair(Stub(),selected)
        self.assertEqual(int(counts.sum()),int((clean!=stego).any(2).sum()))
        self.assertEqual(valid.shape,(8,8))
        cam,aux,tile=cell_maps(model_from_hparams(ENABLED),torch.from_numpy(stego).permute(2,0,1)[None])
        self.assertEqual(tuple(cam.shape),(1,8,8));self.assertEqual(tuple(aux.shape),(1,8,8));self.assertEqual(tuple(tile.shape),(1,))
        # Render the required heatmap columns with tiny synthetic photo fixtures.
        by_pair={r['pair_id']:r for r in rows}
        examples=[{**by_pair[f['pair_id']],'y':'0','x':'0','tile_row':'0','tile_col':'0'} for f in frozen['figure']]
        class TinyCorpus:
            by_id={r['source_photo']:dict(record,source_height='256',source_width='256') for r in examples}
            def photo(self,record):return np.zeros((256,256,3),np.uint8)
        with TemporaryDirectory() as folder,patch('cvdl.cell_diagnostics.selected_rows',return_value=examples):
            output=Path(folder)/'cells.png'
            render_cell_figure(model_from_hparams(ENABLED),TinyCorpus(),ROOT,output,score_limits={'cam':(-1,1),'aux':(-1,1)})
            self.assertTrue(output.is_file());self.assertEqual(len(json.loads(output.with_suffix('.json').read_text())['tiles']),3)


if __name__=='__main__':unittest.main()
