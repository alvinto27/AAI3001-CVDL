"""Archived ALASKA/LSDIR one-off builder provenance check; not default discovery."""
import csv
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'LSB-Dataset-Generator'))
from scripts.archive.build_new_source_manifests import build
from lsb_source import load_photo


class NewSourceManifestTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def image(self, name):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        arr = np.full((512, 512, 3), 53, dtype=np.uint8)
        arr[:, :, 1] = np.arange(512, dtype=np.uint8)
        Image.fromarray(arr, mode='RGB').save(path)
        return path

    def test_manifest_selection_provenance(self):
        alaska=self.image('Datasets/originals/alaska/tiff_various_color/00001.tif')
        lsdir=self.image('Datasets/originals/lsdir/hr/0001000/0000551.png')
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


if __name__ == '__main__':
    unittest.main()
