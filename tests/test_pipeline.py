"""Focused CPU contract checks: pixels, bundle isolation, CAM and public JSON."""
import base64
import io
import json
import os
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import zlib

import numpy as np
from PIL import Image, TiffImagePlugin
import torch

from cvdl.pipeline import load_pipeline, BundleError, InputError, InputTooLarge, UnsupportedInput
from cvdl.pipeline.inputs import decode_photo
from cvdl.pipeline.bundle import contract_hash, file_hash
from cvdl.models import model_from_hparams

ROOT=Path(__file__).resolve().parents[1]
BUNDLE=ROOT/'deploy/run13-rule-c-v1'


def encoded(image,format='PNG',**kwargs):
    stream=io.BytesIO();image.save(stream,format=format,**kwargs);return stream.getvalue()


def toy_bundle(folder,*,broken_state=False):
    """A deliberately different headless model/rule; not benchmark evidence."""
    config=json.loads((BUNDLE/'deploy.json').read_text())
    config['name']='toy-headless';config['version']='2'
    config['architecture']=dict(model='xunet',clip_t=3.,norm='none',front_end='kv',width=1,cell_supervision='off')
    config['map']=dict(kind='cam',score='class1_minus_class0',cell_size=32,cutoff=5.)
    config['source']['checkpoint_sha256']='1'*64
    config['source']['photo_fit_sha256']='2'*64
    config['photo_rule']['model']['coefficients']=[0.]*8
    config['photo_rule']['model']['intercept']=2.
    folder.mkdir()
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(7)
        state=model_from_hparams(config['architecture']).state_dict()
    if broken_state:state['unexpected']=torch.tensor(1.)
    torch.save(dict(state_dict=state,contract_sha256=contract_hash(config)),folder/'weights.pt')
    config['weights_sha256']=file_hash(folder/'weights.pt')
    (folder/'deploy.json').write_text(json.dumps(config))
    return config


class PipelineInputTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.pixels=np.random.default_rng(10).integers(0,256,(257,289,3),dtype=np.uint8)

    def test_rgb_paths_bytes_pil_array_preserve_samples(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'photo.png';path.write_bytes(encoded(Image.fromarray(self.pixels)))
            for photo in (path,str(path),path.read_bytes(),Image.fromarray(self.pixels),self.pixels):
                result,metadata=decode_photo(photo)
                np.testing.assert_array_equal(result,self.pixels)
                self.assertFalse(np.shares_memory(result,self.pixels))
            with Image.open(path) as photo:
                result,_=decode_photo(photo)
                np.testing.assert_array_equal(result,self.pixels)

    def test_rgba_opt_in_preserves_rgb_and_ignores_alpha(self):
        rgba=np.concatenate((self.pixels,np.zeros((257,289,1),dtype=np.uint8)),axis=2)
        for photo in (rgba,Image.fromarray(rgba),encoded(Image.fromarray(rgba))):
            with self.assertRaises(UnsupportedInput):decode_photo(photo)
            result,metadata=decode_photo(photo,allow_rgba=True)
            np.testing.assert_array_equal(result,self.pixels)
            self.assertTrue(any('Alpha' in w for w in metadata['warnings']))
        with self.assertRaises(UnsupportedInput):decode_photo(Image.new('RGBa',(256,256)),allow_rgba=True)

    def test_reject_grey_palette_non_uint8_and_encoded_rgb16(self):
        for photo in (Image.new('L',(256,256)),Image.new('P',(256,256)),
                      np.zeros((256,256,3),dtype=np.float32),np.zeros((256,256,3),dtype=np.uint16),
                      np.zeros((3,256,256),dtype=np.uint8)):
            with self.assertRaises(UnsupportedInput):decode_photo(photo)
        def chunk(kind,payload):
            return struct.pack('>I',len(payload))+kind+payload+struct.pack('>I',zlib.crc32(kind+payload))
        png=(b'\x89PNG\r\n\x1a\n'+chunk(b'IHDR',struct.pack('>IIBBBBB',256,256,16,2,0,0,0))+
             chunk(b'IDAT',zlib.compress((b'\0'+b'\x12\x34'*256*3)*256))+chunk(b'IEND',b''))
        with Image.open(io.BytesIO(png)) as image:
            self.assertEqual(image.mode,'RGB')  # mode alone would silently accept RGB16
            with self.assertRaises(UnsupportedInput):decode_photo(image)
        with self.assertRaises(UnsupportedInput):decode_photo(png)

    def test_orientation_stored_coordinates_and_jpeg_warning(self):
        exif=Image.Exif();exif[274]=6
        for format in ('PNG','JPEG'):
            data=encoded(Image.fromarray(self.pixels),format,exif=exif)
            result,metadata=decode_photo(data)
            with Image.open(io.BytesIO(data)) as stored:
                np.testing.assert_array_equal(result,np.array(stored))
            self.assertEqual(result.shape,self.pixels.shape)
            self.assertEqual(metadata['orientation'],6)
            self.assertTrue(any('not applied' in w for w in metadata['warnings']))
            if format=='JPEG':self.assertTrue(any('DCT' in w for w in metadata['warnings']))
        with self.assertRaises(UnsupportedInput):decode_photo(encoded(Image.new('CMYK',(256,256)),'JPEG'))

    def test_tiff_strict_rgb8_before_orientation_decode(self):
        image=Image.fromarray(self.pixels)
        result,_=decode_photo(encoded(image,'TIFF',compression='raw'))
        np.testing.assert_array_equal(result,self.pixels)
        tags=TiffImagePlugin.ImageFileDirectory_v2();tags[274]=6
        for kwargs in (dict(compression='raw',tiffinfo=tags),dict(compression='tiff_lzw')):
            with self.assertRaises(UnsupportedInput):decode_photo(encoded(image,'TIFF',**kwargs))
        multi=encoded(image,'TIFF',save_all=True,append_images=[image])
        with self.assertRaises(UnsupportedInput):decode_photo(multi)

    def test_limits_tiny_corrupt_and_unsupported_formats(self):
        with self.assertRaises(InputError):decode_photo(np.zeros((255,256,3),np.uint8))
        with self.assertRaises(InputError):decode_photo(b'not an image')
        with self.assertRaises(InputError):decode_photo(encoded(Image.fromarray(self.pixels))[:50])
        with self.assertRaises(UnsupportedInput):decode_photo(encoded(Image.fromarray(self.pixels),'GIF'))
        with patch('cvdl.pipeline.inputs.MAX_ENCODED_BYTES',10):
            with self.assertRaises(InputTooLarge):decode_photo(b'x'*11)
        with patch('cvdl.pipeline.inputs.MAX_PIXELS',256*256):
            with self.assertRaises(InputTooLarge):decode_photo(Image.fromarray(self.pixels))


class PipelineBundleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.pipeline=load_pipeline()

    def test_smoke_json_flags_features_and_overlay(self):
        pixels=np.zeros((257,289,3),np.uint8)
        result=self.pipeline.analyze(pixels,include_images=True)
        json.dumps(result,allow_nan=False)
        self.assertEqual(result['heatmap']['shape'],[9,10])
        self.assertEqual([r['box'] for r in result['tiles']],[[0,0,256,256],[33,0,289,256],[0,1,256,257],[33,1,289,257]])
        self.assertEqual(result['tile_cutoff'],.5094263553619385)
        self.assertEqual(result['heatmap']['cutoff'],.1966136991977692)
        self.assertEqual(result['photo_rule']['cutoff'],-.05636374639150742)
        self.assertEqual(result['bundle']['name'],'run18-rule-c')
        grid=np.asarray(result['heatmap']['values'])
        np.testing.assert_array_equal(result['heatmap']['flagged'],grid>=result['heatmap']['cutoff'])
        self.assertEqual(result['features']['cell_fraction'],float((grid>=result['heatmap']['cutoff']).mean()))
        for key in ('heatmap_png_base64','overlay_png_base64'):
            with Image.open(io.BytesIO(base64.b64decode(result['images'][key]))) as image:self.assertEqual(image.size,(289,257))
        np.testing.assert_array_equal(pixels,0)

    def test_no_heavy_imports_in_fresh_real_analysis(self):
        program="""import sys,numpy as np
from cvdl.pipeline import load_pipeline
load_pipeline().analyze(np.zeros((256,256,3),dtype=np.uint8))
assert not set(('lightning','pytorch_lightning','pandas','yaml','torchvision')) & set(sys.modules)
"""
        env={**os.environ,'CUDA_VISIBLE_DEVICES':'','HIP_VISIBLE_DEVICES':'','OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','PYTHONDONTWRITEBYTECODE':'1'}
        subprocess.run([sys.executable,'-B','-c',program],cwd=ROOT,env=env,check=True,capture_output=True,timeout=60)

    def test_bundle_swap_and_headless_cam_routing(self):
        from cvdl.cell_diagnostics import cell_maps
        from cvdl.pixels import evaluation_tensor
        with tempfile.TemporaryDirectory() as directory:
            folder=Path(directory)/'alternate';toy_bundle(folder)
            with patch.dict(os.environ,{'CVDL_BUNDLE':str(folder)}):
                alternate=load_pipeline()
                self.assertEqual(load_pipeline(BUNDLE).identity['name'],'run13-rule-c')
                self.assertNotEqual(load_pipeline(BUNDLE).identity,self.pipeline.identity)
            pixels=np.zeros((256,256,3),np.uint8)
            result=alternate.analyze(pixels)
            cam,head,tile=cell_maps(alternate._network,evaluation_tensor(pixels).unsqueeze(0).float())
            self.assertIsNone(head)
            np.testing.assert_array_equal(result['heatmap']['values'],cam[0].numpy())
            self.assertEqual(result['photo_verdict'],'message_detected')
            self.assertEqual(result['photo_rule']['logit'],2.)
            self.assertEqual(result['bundle']['name'],'toy-headless')
            self.assertEqual(result['heatmap']['kind'],'cam')

    def test_mixed_rule_threshold_identity_and_bad_state_refused(self):
        with tempfile.TemporaryDirectory() as directory:
            folder=Path(directory)/'alternate';toy_bundle(folder)
            path=folder/'deploy.json';original=path.read_text()
            for mutate in (lambda c:c['tile'].update(cutoff=.1),
                           lambda c:c['source'].update(checkpoint_sha256='f'*64),
                           lambda c:c['photo_rule']['model'].update(intercept=-3.),
                           lambda c:c.update(format_version=999),
                           lambda c:c.update(weights_sha256='0'*64)):
                config=json.loads(original);mutate(config);path.write_text(json.dumps(config))
                with self.assertRaises(BundleError):load_pipeline(folder)
            bad=Path(directory)/'bad';toy_bundle(bad,broken_state=True)
            with self.assertRaises(BundleError):load_pipeline(bad)

    def test_invalid_batch_and_unavailable_explicit_gpu(self):
        for batch in (0,9,True):
            with self.assertRaises(ValueError):load_pipeline(batch_size=batch)
        with patch('torch.cuda.is_available',return_value=False):
            with self.assertRaises(BundleError):load_pipeline(device='cuda')

    def test_cli_json_and_png(self):
        with tempfile.TemporaryDirectory() as directory:
            folder=Path(directory);photo=folder/'input.png'
            photo.write_bytes(encoded(Image.new('RGB',(256,256))))
            env={**os.environ,'CUDA_VISIBLE_DEVICES':'','HIP_VISIBLE_DEVICES':'','OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','PYTHONDONTWRITEBYTECODE':'1'}
            command=[sys.executable,'-B','-m','cvdl.pipeline',str(photo),'--output',str(folder/'result.json'),'--overlay',str(folder/'overlay.png')]
            subprocess.run(command,cwd=ROOT,env=env,check=True,capture_output=True,timeout=60)
            result=json.loads((folder/'result.json').read_text())
            self.assertEqual(result['bundle']['name'],'run18-rule-c')
            with Image.open(folder/'overlay.png') as overlay:self.assertEqual(overlay.size,(256,256))

    def test_builder_headless_with_own_validation_and_fit(self):
        from scripts.build_deploy_bundle import build_bundle
        from scripts.photo_verdict_cache import collection_inputs
        with tempfile.TemporaryDirectory() as directory:
            folder=Path(directory);config=toy_bundle(folder/'reference')
            run=folder/'new-run';run.mkdir()
            model=model_from_hparams(config['architecture'])
            checkpoint=run/'model.ckpt'
            torch.save(dict(hyper_parameters=config['architecture'],state_dict={'net.'+k:v for k,v in model.state_dict().items()}),checkpoint)
            checkpoint_sha=file_hash(checkpoint)
            evaluation=run/'evaluation/standard-256-v1/new-evaluation'
            (evaluation/'validation').mkdir(parents=True)
            calibration=dict(checkpoint_sha256=checkpoint_sha,deployment=True,quick=False,
                geometry=dict(tile_size=256,grid='overlap_last'),cutoff=.4321,threshold_id='own-threshold')
            (evaluation/'validation/calibration.json').write_text(json.dumps(calibration))
            (evaluation/'evaluation_settings.json').write_text(json.dumps(dict(quick=False,checkpoint_sha256=checkpoint_sha,evaluation_id='new-evaluation')))
            identity=dict(checkpoint_sha256=checkpoint_sha,map_kind='cam',evaluation_id='new-evaluation',tile_cutoff=.4321,head_cutoff=5.)
            fit=dict(cache_identity=identity,tile_cutoff=.4321,head_cutoff=5.,rules={'c':config['photo_rule']})
            fit_path=folder/'fit.json';fit_path.write_text(json.dumps(fit))
            cells=folder/'cells.json';cells.write_text(json.dumps(dict(maps=dict(cam=dict(cutoff=5.)))))
            self.assertEqual(collection_inputs(run,'new-evaluation',cells,'cam'),(checkpoint,.4321,5.))
            build_bundle(run,'new-evaluation',fit_path,cells,folder/'export',map_kind='cam')
            exported=load_pipeline(folder/'export')
            self.assertEqual(exported.analyze(np.zeros((256,256,3),np.uint8))['photo_rule']['logit'],2.)
            self.assertEqual(exported._config['map']['kind'],'cam')
            self.assertEqual(exported._config['source']['identity_route'],'pipeline_native')
            with self.assertRaisesRegex(ValueError,'map/evaluation'):
                build_bundle(run,'new-evaluation',fit_path,cells,folder/'wrong',map_kind='trained_head')
            calibration['cutoff']=.123
            (evaluation/'validation/calibration.json').write_text(json.dumps(calibration))
            with self.assertRaisesRegex(ValueError,'thresholds'):
                build_bundle(run,'new-evaluation',fit_path,cells,folder/'wrong',map_kind='cam')

    def test_builder_yx_cutoff_sources_and_each_mismatch_existing_routes_unchanged(self):
        import copy
        from scripts import build_deploy_bundle as builder
        with tempfile.TemporaryDirectory() as directory:
            folder=Path(directory);config=json.loads((BUNDLE/'deploy.json').read_text())
            run=folder/'own-run';run.mkdir();model=model_from_hparams(config['architecture'])
            checkpoint=run/'last.ckpt'
            torch.save(dict(hyper_parameters=config['architecture'],state_dict={'net.'+k:v for k,v in model.state_dict().items()}),checkpoint)
            sha=file_hash(checkpoint);evaluation=run/'evaluation/standard-256-v1/own-eval'
            (evaluation/'validation').mkdir(parents=True)
            calibration=dict(checkpoint_sha256=sha,deployment=True,quick=False,
                geometry=dict(tile_size=256,grid='overlap_last'),cutoff=.43,threshold_id='own-threshold')
            cal_path=evaluation/'validation/calibration.json';cal_path.write_text(json.dumps(calibration))
            (evaluation/'evaluation_settings.json').write_text(json.dumps(dict(quick=False,checkpoint_sha256=sha,evaluation_id='own-eval')))
            cells=folder/'cells.json';cell_data=dict(maps=dict(aux=dict(cutoff=.2)))
            cells.write_text(json.dumps(cell_data))
            identity=dict(checkpoint_sha256=sha,tile_cutoff=.43,head_cutoff=.2,
                cutoff_sources=dict(tile_threshold_id='own-threshold',cell_summary=dict(sha256=file_hash(cells))))
            fit=dict(cache_identity=identity,tile_cutoff=.43,head_cutoff=.2,rules={'c':config['photo_rule']})
            fit_path=folder/'fit.json';fit_path.write_text(json.dumps(fit));original=fit_path.read_bytes()
            exported=builder.build_bundle(run,'own-eval',fit_path,cells,folder/'yx',map_kind='trained_head')
            self.assertEqual(exported['source']['identity_route'],'yx_cutoff_sources')
            self.assertEqual(load_pipeline(folder/'yx')._config,exported)
            self.assertEqual((folder/'yx').stat().st_mode&0o777,0o755)
            self.assertEqual(fit_path.read_bytes(),original)
            for mismatch in ('checkpoint','threshold_id','threshold_value','cells_hash','cell_cutoff','cam','partial_map','partial_eval'):
                with self.subTest(mismatch=mismatch):
                    changed=copy.deepcopy(fit);cal=copy.deepcopy(calibration);cs=copy.deepcopy(cell_data);kind='trained_head'
                    if mismatch=='checkpoint':changed['cache_identity']['checkpoint_sha256']='0'*64
                    elif mismatch=='threshold_id':changed['cache_identity']['cutoff_sources']['tile_threshold_id']='wrong'
                    elif mismatch=='threshold_value':cal['cutoff']=.44
                    elif mismatch=='cells_hash':changed['cache_identity']['cutoff_sources']['cell_summary']['sha256']='0'*64
                    elif mismatch=='cell_cutoff':cs['maps']['aux']['cutoff']=.21
                    elif mismatch=='cam':kind='cam'
                    elif mismatch=='partial_map':changed['cache_identity']['map_kind']='trained_head'
                    else:changed['cache_identity']['evaluation_id']='own-eval'
                    cells.write_text(json.dumps(cs));cal_path.write_text(json.dumps(cal))
                    if mismatch=='cell_cutoff':changed['cache_identity']['cutoff_sources']['cell_summary']['sha256']=file_hash(cells)
                    fit_path.write_text(json.dumps(changed));out=folder/('bad-'+mismatch)
                    with self.assertRaises(ValueError):builder.build_bundle(run,'own-eval',fit_path,cells,out,map_kind=kind)
                    self.assertFalse(out.exists())
            cells.write_text(json.dumps(cell_data));cal_path.write_text(json.dumps(calibration))
            native=copy.deepcopy(fit);native['cache_identity'].pop('cutoff_sources')
            native['cache_identity'].update(map_kind='trained_head',evaluation_id='own-eval');fit_path.write_text(json.dumps(native))
            self.assertEqual(builder.build_bundle(run,'own-eval',fit_path,cells,folder/'native')['source']['identity_route'],'pipeline_native')
            legacy=copy.deepcopy(fit);legacy['cache_identity'].pop('cutoff_sources');fit_path.write_text(json.dumps(legacy))
            with patch.object(builder,'LEGACY_RUN13',sha):
                self.assertEqual(builder.build_bundle(run,'own-eval',fit_path,cells,folder/'legacy')['source']['identity_route'],'legacy_run13')
                with self.assertRaisesRegex(ValueError,'trained-head only'):
                    builder.build_bundle(run,'own-eval',fit_path,cells,folder/'legacy-cam',map_kind='cam')

    def test_builder_rejects_other_checkpoint_rule_before_export(self):
        from scripts.build_deploy_bundle import build_bundle
        run=ROOT/'runs/srm-cell-aux-20261021T054704Z-6997'
        fit=ROOT/'runs/verification/photo-verdict-run13-2026-10-21/frozen-rules.json'
        cells=ROOT/'runs/verification/cell-run13-2026-10-21/cells/summary.json'
        if not fit.exists():self.skipTest('local accepted fit artifact absent; runtime tests use tracked bundle')
        with tempfile.TemporaryDirectory() as directory:
            bad=Path(directory)/'fit.json';data=json.loads(fit.read_text())
            data['cache_identity']['checkpoint_sha256']='0'*64;bad.write_text(json.dumps(data))
            with self.assertRaisesRegex(ValueError,'another checkpoint'):
                build_bundle(run,'ea32a4be0e2ce3597fbd',bad,cells,Path(directory)/'bundle')


if __name__=='__main__':unittest.main()
