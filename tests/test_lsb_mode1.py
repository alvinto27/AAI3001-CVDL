"""Mode 1 direct geometry, key provenance, clean ownership and replay checks."""
import json
import sys
import unittest
from pathlib import Path

import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "LSB-Dataset-Generator"))
from lsb_batch import Config
from lsb_engine import embed
from lsb_settings import draw_legacy_settings, legacy_seed_sequence
from lsb_source import iter_tiles, tile_at
from lsb_mode1 import TileSpec, LegacySeed, OTFSeed, make_sample, replay_legacy
from test_lsb_source import record


class Mode1Tests(unittest.TestCase):
    def setUp(self):
        self.record = {**record(649,777,kind="nef"), "photo_index":"5", "split":"train"}
        self.photo = np.random.default_rng(99).integers(0,256,size=(649,777,3),dtype=np.uint8)
        self.photo.setflags(write=False)
        self.cfg = Config(seed=1,input="unused",output="unused",bit_mode=["lowest","plane"],
                          k=[1,3,8],method=["replace","match"],placement=["random","contiguous","stride"],
                          stride=[1,10],size={"bpp":[0.05,0.5]})

    def test_direct_lookup_matches_iterator(self):
        for grid,size in (("overlap_last",512),("overlap_last",256),("topleft_crop",512)):
            for expected in iter_tiles(self.photo,self.record,grid,size):
                actual = tile_at(self.photo,self.record,grid,size,expected.row,expected.column)
                self.assertEqual(actual.identity,expected.identity)
                self.assertTrue(np.shares_memory(actual.pixels,self.photo))
                np.testing.assert_array_equal(actual.pixels,expected.pixels)
        for row,col in ((-1,0),(0,-1),(999,0),(True,0),(0,1.0)):
            with self.assertRaises(ValueError):
                tile_at(self.photo,self.record,"overlap_last",512,row,col)

    def test_legacy_against_original_functions(self):
        tile = TileSpec("topleft_crop",512,0,0)
        view = tile_at(self.photo,self.record,"topleft_crop",512,0,0).pixels
        kwargs, drawn = draw_legacy_settings(self.cfg,view,np.random.default_rng(legacy_seed_sequence(1,3,1,0)))
        kwargs["rng"] = np.random.default_rng(legacy_seed_sequence(1,3,1,1))
        expected, _, result = embed(view,**kwargs)
        actual, metadata = make_sample(self.photo,self.record,tile,self.cfg,LegacySeed(1,3),"stego")
        np.testing.assert_array_equal(actual,expected)
        self.assertEqual(metadata["embedding"],{k:v for k,v in vars(result).items() if not k.startswith("_")})
        self.assertEqual(metadata["settings_seed_key"],[1,3,1,0])
        self.assertEqual(metadata["embedding_seed_key"],[1,3,1,1])
        json.dumps(metadata)

    def test_clean_owned_and_no_settings_draw(self):
        sample, meta = make_sample(self.photo,self.record,TileSpec("overlap_last",512,1,1),
                                  None,OTFSeed(2,"train",0),"clean")
        self.assertFalse(np.shares_memory(sample,self.photo))
        self.assertEqual((meta["y"],meta["x"]),(137,265))
        self.assertIsNone(meta["settings"])
        self.assertIsNone(meta["embedding"])
        self.assertEqual(meta["label"],0)
        snapshot = self.photo.copy()
        sample[:] = 0
        np.testing.assert_array_equal(self.photo,snapshot)

    def test_otf_metadata_fixed_and_clipped(self):
        tile = TileSpec("overlap_last",256,2,3)
        for purpose in ("val","test"):
            a,ma = make_sample(self.photo,self.record,tile,self.cfg,OTFSeed(1,purpose,0),"stego")
            b,mb = make_sample(self.photo,self.record,tile,self.cfg,OTFSeed(999,purpose,100),"stego")
            np.testing.assert_array_equal(a,b)
            self.assertEqual(ma,mb)
        overflow = Config(seed=1,input="unused",output="unused",placement="stride",stride=1000,
                          bit_mode="plane",size={"bpp":10.0})
        _, meta = make_sample(self.photo,self.record,tile,overflow,OTFSeed(1,"train",0),"stego")
        self.assertTrue(meta["settings"]["clipped"])
        self.assertLess(meta["settings"]["effective_bpp"],meta["settings"]["requested_bpp"])
        self.assertEqual(meta["settings"]["effective_bpp"],meta["embedding"]["bpp_actual"])

    def test_replay_row_validation_and_kind(self):
        row = dict(photo_id="example",source="raise",split="train",tile_row="0",tile_col="0",
                   pixel_y="0",pixel_x="0",cover_id="000003",cover_index="3")
        _, meta = replay_legacy(self.photo,self.record,row,self.cfg,"clean")
        self.assertEqual(meta["cover_id"],"000003")
        for altered in ({**row,"photo_id":"wrong"},{**row,"pixel_x":"512"}):
            with self.assertRaises(ValueError):
                replay_legacy(self.photo,self.record,altered,self.cfg,"clean")
        with self.assertRaises(ValueError):
            make_sample(self.photo,self.record,TileSpec("overlap_last",512,0,0),self.cfg,OTFSeed(1,"train",0),"bad")
        with self.assertRaises(ValueError):
            make_sample(self.photo,self.record,TileSpec("overlap_last",512,0,0),self.cfg,LegacySeed(1,3),"stego")


if __name__ == "__main__":
    unittest.main()
