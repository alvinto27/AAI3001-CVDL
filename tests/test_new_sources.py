"""Strict new readers, immutable selection records, mixed epochs and isolated panels."""
import csv
import hashlib
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest

import numpy as np
from PIL import Image, TiffImagePlugin

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'LSB-Dataset-Generator'))
sys.path.insert(0,str(ROOT/'scripts'))
from build_new_source_manifests import build
from lsb_source import load_photo,grid_axes
from lsb_torch import Corpus,RunSettings,TrainingImages,FixedPairs,PhotoPairTasks,PairSubset,training_loader
from lsb_notebook import capped_validation,fixed_subset,source_balanced_subset
from test_lsb_source import record as old_record


class NewSources(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        self.root=Path(self.tmp.name)
        (self.root/'configs').mkdir()
        shutil.copy(ROOT/'configs/raise512-run1.yaml',self.root/'configs/raise512-run1.yaml')
    def tearDown(self):self.tmp.cleanup()
    def image(self,name,size=(512,512),mode='RGB',**kwargs):
        path=self.root/name;path.parent.mkdir(parents=True,exist_ok=True)
        arr=np.full((size[1],size[0],4 if mode=='RGBA' else 3),53,dtype=np.uint8)
        if mode=='RGB':arr[:,:,1]=np.arange(size[0],dtype=np.uint8)
        Image.fromarray(arr,mode=mode).save(path,**kwargs)
        return path
    def new_record(self,source,index,split='train',size=(512,512)):
        name=f'{source}{index}.'+('tif' if source=='alaska' else 'png')
        path=self.image(name,size)
        h,w=size[1],size[0]
        return {**old_record(h,w), 'source':source,'source_type':'tiff_rgb8' if source=='alaska' else 'png8',
                'photo_id':f'{source}_{index}','photo_index':str((100000 if source=='alaska' else 300000)+index),
                'master_path':name,'camera_model':'Alpha' if source=='alaska' else '',
                'noise':'not_applicable','grid_policy':'trim512','split':split,
                'overlap512_rows':str(h//512),'overlap512_cols':str(w//512),
                'overlap512_y':';'.join(str(v*512) for v in range(h//512)),
                'overlap512_x':';'.join(str(v*512) for v in range(w//512))}
    def test_tiff_strict_and_png_strict(self):
        row=self.new_record('alaska',1,size=(640,720))
        pixels=load_photo(row,self.root)
        self.assertEqual(pixels.shape,(720,640,3))
        self.assertEqual(grid_axes(row,'trim512',512),((0,),(0,)))
        with self.assertRaises(ValueError):grid_axes(row,'overlap_last',512)
        tags=TiffImagePlugin.ImageFileDirectory_v2();tags[274]=6
        self.image('rotate.tif',tiffinfo=tags)
        with self.assertRaises(ValueError):load_photo({**row,'master_path':'rotate.tif','source_height':512,'source_width':512},self.root)
        self.image('compressed.tif',compression='tiff_lzw')
        with self.assertRaises(ValueError):load_photo({**row,'master_path':'compressed.tif','source_height':512,'source_width':512},self.root)
        self.image('alpha.tif',mode='RGBA')
        with self.assertRaises(ValueError):load_photo({**row,'master_path':'alpha.tif','source_height':512,'source_width':512},self.root)
        png=self.new_record('lsdir',2)
        self.assertEqual(load_photo(png,self.root).shape,(512,512,3))
        self.image('alpha.png',mode='RGBA')
        with self.assertRaises(ValueError):load_photo({**png,'master_path':'alpha.png'},self.root)
    def test_manifest_selection_provenance(self):
        alaska=self.image('Datasets/alaska/tiff_various_color/00001.tif')
        lsdir=self.image('Datasets/lsdir/hr/0001000/0000551.png')
        def metadata(path):return dict(bytes=path.stat().st_size,sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        cases=[('alaska',dict(alaska_id='00001',camera_group='Alpha',camera_model='Alpha',split='train',block='1',
                              width='512',height='512',tiles512='1',in_v1='1',url='https://example.org/1.tif',**metadata(alaska))),
               ('lsdir',dict(lsdir_id='0000551',shard='shard-00',path='0001000/0000551.png',split='test',
                             width='512',height='512',tiles512='1',**metadata(lsdir)))]
        for source,row in cases:
            path=self.root/f'{source}.csv'
            with path.open('w',newline='') as f:
                writer=csv.DictWriter(f,fieldnames=row);writer.writeheader();writer.writerow(row)
            report=build(source,path,self.root,self.root)
            self.assertEqual(report['photos'],1)
            record_path=self.root/'data'/source/'photo_manifest.csv'
            with record_path.open() as f:manifest=next(csv.DictReader(f))
            self.assertEqual(manifest['grid_policy'],'trim512')
            self.assertEqual(manifest['source_type'],'tiff_rgb8' if source=='alaska' else 'png8')
            self.assertEqual(load_photo(manifest,self.root).shape,(512,512,3))
            self.assertEqual(json.loads((record_path.parent/'build_record.json').read_text())['selection_sha256'],
                             hashlib.sha256(path.read_bytes()).hexdigest())
            with self.assertRaises(ValueError):build(source,path,self.root,self.root)
    def test_seeded_sampling_and_worker_parity(self):
        originals=[]
        for i in range(2):
            row={**old_record(512,512),'source':'raise','camera_model':'',
                 'photo_id':f'raise{i}','photo_index':str(10+i),
                 'master_path':f'raise{i}.png','split':'train'}
            self.image(row['master_path']);originals.append(row)
        records=originals+[self.new_record(source,i,size=(1024,512))
                            for source in ('lsdir','alaska') for i in range(2)]
        settings=RunSettings(sources=('raise','lsdir','alaska'),source_mix={'raise':.4,'lsdir':.2,'alaska':.4},
                             tiles_per_epoch=10,buffer_size=8,augment=False)
        corpus=Corpus(self.root,settings,records=records)
        a,repeats=corpus.train_tile_schedule(0);b,_=corpus.train_tile_schedule(0)
        def identity(groups):return [(r['photo_id'],row,col,variant)
                                     for r,tiles in groups for row,col,variant in tiles]
        self.assertEqual(identity(a),identity(b))
        self.assertEqual(len(identity(a)),10)
        self.assertEqual(len(set(identity(a))),8)
        self.assertEqual(repeats,{'raise':2,'lsdir':0,'alaska':0})
        self.assertEqual({s:sum(r['source']==s for r,tiles in a for _ in tiles)
                          for s in settings.sources},{'raise':4,'lsdir':2,'alaska':4})
        changed,_=corpus.train_tile_schedule(1)
        self.assertNotEqual(identity(a),identity(changed))
        dataset=TrainingImages(corpus)
        def collect(workers):
            batches=list(training_loader(dataset,batch_size=4,num_workers=workers,
                                         pin_memory=False,persistent_workers=False))
            return sum(len(labels) for _,labels,_ in batches),[
                (int(i),int(label),hashlib.sha256(img.numpy().tobytes()).hexdigest())
                for images,labels,indices in batches for img,label,i in zip(images,labels,indices)]
        single,items=collect(0)
        parallel,other=collect(2)
        self.assertEqual(single,parallel)
        self.assertEqual(dict((i,(label,digest)) for i,label,digest in items),
                         dict((i,(label,digest)) for i,label,digest in other))
        self.assertEqual(single,20)
        dataset.set_epoch(1)
        count1,one=collect(0);count2,two=collect(2)
        self.assertEqual(count1,count2)
        self.assertEqual(dict((i,(label,digest)) for i,label,digest in one),
                         dict((i,(label,digest)) for i,label,digest in two))
    def test_capped_validation_fixed_across_run_seeds_and_excludes_div2k(self):
        records=[]
        for source,count in [('raise',10),('div2k',2),('alaska',2),('lsdir',3)]:
            for i in range(count):
                if source in ('alaska','lsdir'):
                    row=self.new_record(source,i,'val')
                else:
                    row={**old_record(512,512),'photo_id':f'{source}_{i}',
                         'source':source,'photo_index':str(10+i+(100 if source=='div2k' else 0)),
                         'camera_model':'Alpha','master_path':f'{source}_{i}.png','split':'val'}
                    self.image(row['master_path'])
                records.append(row)
        ids=[]
        for seed in (0,923):
            cfg=RunSettings(sources=('raise','alaska','lsdir'),run_seed=seed)
            full=FixedPairs(Corpus(self.root,cfg,records=records),'val')
            selected=capped_validation(full,5,2)
            frame=selected.frame
            self.assertEqual(frame['source'].value_counts().to_dict(),{'raise':5,'alaska':2,'lsdir':2})
            self.assertNotIn('div2k',frame['source'].values)
            quick=fixed_subset(selected,8,0)
            self.assertIsInstance(quick,PairSubset)
            self.assertIsInstance(quick.dataset,FixedPairs)
            self.assertEqual(sum(len(group) for group in PhotoPairTasks(quick).groups),8)
            smoke=source_balanced_subset(selected,3,0)
            self.assertEqual(smoke.frame['source'].value_counts().to_dict(),
                             {'raise':1,'alaska':1,'lsdir':1})
            self.assertIsInstance(smoke.dataset,FixedPairs)
            ids.append(frame['cover_id'].tolist())
        self.assertEqual(ids[0],ids[1])
        with self.assertRaises(ValueError):capped_validation(full,0,2)

    def test_source_panels_do_not_change_standard_test(self):
        standard={**old_record(512,512),'source':'raise','camera_model':'',
                  'photo_id':'raise_test','photo_index':'10',
                  'master_path':'raise_test.png','split':'test'}
        self.image(standard['master_path'])
        new=[self.new_record('alaska',0,'test_seen'),self.new_record('alaska',1,'test_unseen'),
             self.new_record('lsdir',0,'test'),self.new_record('lsdir',1,'val')]
        cfg=RunSettings(sources=('raise','lsdir','alaska'))
        corpus=Corpus(self.root,cfg,records=[standard]+new)
        std=FixedPairs(corpus,'test',sources=('raise','div2k'))
        self.assertEqual(std.frame['source'].tolist(),['raise'])
        standard_val=FixedPairs(corpus,'val',sources=('raise','div2k'))
        combined_val=FixedPairs(corpus,'val')
        self.assertEqual(len(standard_val),0)
        self.assertEqual(combined_val.frame['source'].tolist(),['lsdir'])
        for purpose,source in [('test_seen','alaska'),('test_unseen','alaska'),('test','lsdir'),('val','lsdir')]:
            panel=FixedPairs(corpus,purpose,sources=(source,))
            self.assertEqual(len(panel),1)
            clean,stego,_=panel[0]
            self.assertFalse(np.array_equal(clean.numpy(),stego.numpy()))
            self.assertEqual(panel.frame['grid'].tolist(),['trim512'])

if __name__=='__main__':unittest.main()
