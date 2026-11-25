"""Atomic cache commits, resume, strict freshness and mmap safety; VM only."""
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
from PIL import Image
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"LSB-Dataset-Generator"))
import lsb_cache as cache
from test_lsb_source import record


class CacheTests(unittest.TestCase):
    def setUp(self):
        self.temporary=tempfile.TemporaryDirectory()
        self.root=Path(self.temporary.name)
        self.folder=self.root/"cache"
        self.row=record(512,512)
        self.pixels=np.random.default_rng(2).integers(0,256,size=(512,512,3),dtype=np.uint8)
        Image.fromarray(self.pixels).save(self.root/"example.png")
    def tearDown(self):
        self.temporary.cleanup()
    def build(self):
        return cache.build_entry(self.row,self.folder,self.root)
    def test_build_resume_and_mmap(self):
        self.assertEqual(self.build()["status"],"built")
        abandoned = [self.folder / ".example.npy-interrupted.tmp",
                     self.folder / ".example.json-interrupted.tmp"]
        for path in abandoned:
            path.write_bytes(b"interrupted")
        with patch.object(cache,"load_photo",side_effect=AssertionError("decoded on resume")):
            self.assertEqual(self.build()["status"],"skipped")
        self.assertTrue(all(not p.exists() for p in abandoned))
        mapped=cache.load_cached(self.row,self.folder,self.root,mmap=True)
        self.assertIsInstance(mapped,np.memmap)
        self.assertFalse(mapped.flags.writeable)
        np.testing.assert_array_equal(mapped,self.pixels)
        self.assertEqual(list(self.folder.glob("*.tmp")),[])
    def test_atomic_failure_cleanup_and_resume(self):
        with patch.object(cache.np,"save",side_effect=RuntimeError("interrupted")):
            with self.assertRaises(RuntimeError):
                self.build()
        self.assertFalse((self.folder/"example.npy").exists())
        self.assertFalse((self.folder/"example.json").exists())
        self.assertEqual(list(self.folder.glob(".*.tmp")),[])
        with self.assertRaises(cache.CacheError):
            cache.load_cached(self.row,self.folder,self.root)
        self.assertEqual(self.build()["status"],"built")
    def test_record_commit_failure_never_exposes_uncommitted_entry(self):
        real=cache._atomic_json
        def interrupt(path,value):
            if path.name=="example.json":
                raise RuntimeError("record commit interrupted")
            return real(path,value)
        with patch.object(cache,"_atomic_json",side_effect=interrupt):
            with self.assertRaises(RuntimeError):
                self.build()
        self.assertTrue((self.folder/"example.npy").exists())
        with self.assertRaises(cache.CacheError):
            cache.load_cached(self.row,self.folder,self.root)
        self.assertEqual(self.build()["status"],"built")
    def test_stale_master_and_source_version_rejected(self):
        self.build()
        path=self.folder/"example.json"
        saved=json.loads(path.read_text())
        changed={**saved,"decode":{**saved["decode"],"source_decode_version":999}}
        path.write_text(json.dumps(changed))
        with self.assertRaisesRegex(cache.CacheError,"decode"):
            cache.load_cached(self.row,self.folder,self.root)
        path.write_text(json.dumps(saved))
        with (self.root/"example.png").open("ab") as stream:
            stream.write(b"changed")
        with self.assertRaisesRegex(cache.CacheError,"master"):
            cache.load_cached(self.row,self.folder,self.root)
    def test_shape_hash_identity_and_cache_version_rejected(self):
        self.build()
        with self.assertRaisesRegex(cache.CacheError,"shape"):
            cache.load_cached({**self.row,"source_height":"513"},self.folder,self.root)
        np.save(self.folder/"example.npy",self.pixels[:511])
        with self.assertRaisesRegex(cache.CacheError,"shape"):
            cache.load_cached(self.row,self.folder,self.root)
        changed=self.pixels.copy()
        changed[0,0,0]^=1
        np.save(self.folder/"example.npy",changed)
        with self.assertRaisesRegex(cache.CacheError,"SHA-256"):
            cache.load_cached(self.row,self.folder,self.root)
        (self.folder/"cache.json").write_text('{"format_version":999}')
        with self.assertRaisesRegex(cache.CacheError,"format"):
            cache.load_cached(self.row,self.folder,self.root)
    def test_no_decode_fallback_or_unsafe_ids(self):
        cache.initialize_cache(self.folder)
        with patch.object(cache,"load_photo",side_effect=AssertionError("silent decode")):
            with self.assertRaises(cache.CacheError):
                cache.load_cached(self.row,self.folder,self.root)
        with self.assertRaisesRegex(cache.CacheError,"photo_id"):
            cache.load_cached({**self.row,"photo_id":"../unsafe"},self.folder,self.root)


if __name__=="__main__":
    unittest.main()
