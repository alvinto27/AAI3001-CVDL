"""Full-photo generation and bounded changed-index overlap labels."""
import itertools
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "LSB-Dataset-Generator"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "LSB-Dataset-Generator" / "archive"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # tests/ (test_lsb_source helpers)
from lsb_batch import Config
from lsb_mode1 import OTFSeed
from lsb_mode2 import embed_photo, tile_change_counts, iter_photo_tiles
from lsb_settings import otf_seed_sequence
from lsb_source import iter_tiles, load_photo
from lsb_cache import build_entry, load_cached
from test_lsb_source import record


def source_record(h=600, w=620):
    return dict(record(h, w), photo_index="1000", split="val", camera_model="")


def configuration(**kwargs):
    return Config(seed=1, input="unused", output="unused", **kwargs)


class Mode2Tests(unittest.TestCase):
    def setUp(self):
        self.row = source_record()
        self.photo = np.random.default_rng(37).integers(0, 256, (600, 620, 3), dtype=np.uint8)

    def compare_counts(self, clean, stego, changed, row, size, block_size=65536):
        values, pixels = tile_change_counts(changed, row, size, block_size=block_size)
        for tile in iter_tiles(clean, row, tile_size=size):
            difference = stego[tile.y:tile.y+size, tile.x:tile.x+size] != tile.pixels
            self.assertEqual(int(values[tile.row, tile.column]), int(np.count_nonzero(difference)))
            self.assertEqual(int(pixels[tile.row, tile.column]), int(np.count_nonzero(np.any(difference, axis=2))))

    def test_all_settings_and_full_area_payload(self):
        original = self.photo.copy()
        cases = itertools.product(("lowest", "plane"), range(1, 9), ("replace", "match"),
                                  ("contiguous", "stride", "random"), ("random_bytes", "letters"), (0.00001, 0.8))
        for mode, k, method, placement, payload, bpp in cases:
            with self.subTest(mode=mode, k=k, method=method, placement=placement, payload=payload, bpp=bpp):
                cfg = configuration(bit_mode=mode, k=k, method=method, placement=placement,
                                    payload=payload, stride=16 if placement == "stride" else 1, size={"bpp": bpp})
                stego, meta, changed = embed_photo(self.photo, self.row, cfg, OTFSeed(9, "val", 70))
                expected = min(max(1, round(bpp*600*620/8)), meta["settings"]["capacity_payload_bytes"])
                self.assertEqual(meta["embedding"]["payload_bytes"], expected)
                self.assertEqual(meta["settings"]["effective_bpp"], expected*8/(600*620))
                self.assertEqual(len(changed), meta["embedding"]["units_changed"])
                np.testing.assert_array_equal(changed, np.flatnonzero(stego.reshape(-1) != self.photo.reshape(-1)))
                for size in (256, 512):
                    self.compare_counts(self.photo, stego, changed, self.row, size)
                json.dumps(meta)
        np.testing.assert_array_equal(original, self.photo)

    def test_one_two_four_overlap_and_block_boundary_pixel_counts(self):
        for size, corner in ((512, 500), (256, 400)):
            for y, x, expected in ((10, 10, 1), (10, corner, 2), (corner, corner, 4), (599, 619, 1)):
                clean = np.zeros_like(self.photo); stego = clean.copy()
                stego[y, x] = 1
                units = np.asarray([(y*620+x)*3+c for c in range(3)], dtype=np.intp)
                values, pixels = tile_change_counts(units, self.row, size, block_size=1)
                self.assertEqual(np.count_nonzero(values), expected)
                self.assertEqual(int(values.sum()), 3*expected)
                self.assertEqual(int(pixels.sum()), expected)
                self.compare_counts(clean, stego, units, self.row, size, block_size=1)

    def test_geometry_exact_axes_exclusive_ends_and_random_indices(self):
        for h, w in ((512, 512), (1024, 1024), (513, 649), (1100, 1600)):
            row = source_record(h, w); clean = np.zeros((h, w, 3), np.uint8); stego = clean.copy()
            units = np.unique(np.r_[0, h*w*3-1, np.random.default_rng(42).integers(0, h*w*3, 2000)])
            stego.reshape(-1)[units] = 1
            for size in (256, 512):self.compare_counts(clean, stego, units, row, size, block_size=19)

    def test_repeat_cross_size_evaluation_and_training_policy(self):
        cfg = configuration(bit_mode=["lowest", "plane"], k=list(range(1, 9)),
                            method=["replace", "match"], placement=["contiguous", "stride", "random"],
                            stride=[2, 16], payload=["random_bytes", "letters"], size={"bpp": [0.05, 0.5]})
        first, meta, changed = embed_photo(self.photo, self.row, cfg, OTFSeed(9, "val", 70))
        again, repeated, other = embed_photo(self.photo, self.row, cfg, OTFSeed(33, "val", 99))
        np.testing.assert_array_equal(first, again);self.assertEqual(meta, repeated)
        np.testing.assert_array_equal(changed, other)
        self.assertEqual(meta["settings_seed_key"][4], 2)
        self.assertEqual(meta["settings_seed_key"][6:10], [0, 0, 0, 0])
        for size in (256, 512):
            output = list(iter_photo_tiles(self.photo, first, self.row, meta, changed, size))
            self.assertEqual(len(output), 2*int(self.row[f"overlap{size}_rows"])*int(self.row[f"overlap{size}_cols"]))
            self.assertEqual(len({r["sample_id"] for _, r in output}), len(output))
            for image, r in output:
                self.assertFalse(np.shares_memory(image, self.photo));self.assertFalse(np.shares_memory(image, first))
                self.assertEqual(r["photo_embedded"], r["kind"] == "stego")
                if r["kind"] == "clean":self.assertEqual((r["label"], r["changed_values"], r["changed_pixels"]), (0, 0, 0))
                else:self.assertEqual(r["label"], int(r["changed_values"] > 0))
        fixed = [embed_photo(self.photo, self.row, cfg, OTFSeed(7, "train", e, fixed_variants=True)) for e in (0, 1)]
        np.testing.assert_array_equal(fixed[0][0], fixed[1][0]);self.assertEqual(fixed[0][1], fixed[1][1])
        epoch = embed_photo(self.photo, self.row, cfg, OTFSeed(7, "train", 1))
        self.assertFalse(np.array_equal(epoch[0], fixed[0][0]))
        variant = embed_photo(self.photo, self.row, cfg, OTFSeed(7, "train", 0, variant=2))
        self.assertFalse(np.array_equal(variant[0], fixed[0][0]))

    def test_selected_but_unchanged_photo_and_clean_labels(self):
        clean = np.zeros_like(self.photo)
        key = otf_seed_sequence(run_seed=1, purpose="val", epoch=0, photo_index=1000,
                                grid_policy="none", tile_size=0, tile_row=0, tile_col=0,
                                variant=1, stream="embedding", carrier="photo")
        payload = np.random.default_rng(key).integers(0, 256, size=1, dtype=np.uint8)
        clean.reshape(-1)[:8] = np.unpackbits(payload)
        cfg = configuration(bit_mode="lowest", k=1, method="replace", placement="contiguous", start=0, size={"payload_bytes": 1})
        stego, meta, changed = embed_photo(clean, self.row, cfg, OTFSeed(1, "val", 0))
        self.assertEqual(changed.size, 0);self.assertFalse(meta["actual_changed"])
        for size in (256, 512):
            self.assertTrue(all(r["label"] == 0 for _, r in iter_photo_tiles(clean, stego, self.row, meta, changed, size)))

    def test_lower_to_fit_fixed_range_and_validation(self):
        for size in ({"bpp": 100.0}, {"bpp": [100.0, 200.0]}, {"payload_bytes": 9000000}):
            _, meta, _ = embed_photo(self.photo, self.row, configuration(bit_mode="plane", k=8,
                                      placement="stride", stride=16, size=size), OTFSeed(1, "val", 0))
            self.assertTrue(meta["settings"]["clipped"])
            self.assertEqual(meta["embedding"]["payload_bytes"], meta["settings"]["capacity_payload_bytes"])
        for units in (np.array([0, 0]), np.array([1, 0]), np.array([-1]), np.array([600*620*3]), np.array([1.0]), np.zeros((1, 1), dtype=int)):
            with self.assertRaises(ValueError):tile_change_counts(units, self.row, 512)
        with self.assertRaises(ValueError):tile_change_counts(np.array([], dtype=int), self.row, 512, block_size=0)
        with self.assertRaises(ValueError):embed_photo(self.photo.astype(np.float32), self.row, configuration(), OTFSeed(1, "val", 0))
        with self.assertRaises(TypeError):embed_photo(self.photo, self.row, configuration(), None)

    def test_fixed_test_seed_training_run_seed_and_minimum_capacity(self):
        cfg = configuration(placement="random", size={"bpp": 0.1})
        row = dict(self.row, split="test")
        a = embed_photo(self.photo, row, cfg, OTFSeed(1, "test", 0))
        b = embed_photo(self.photo, row, cfg, OTFSeed(90, "test", 500))
        np.testing.assert_array_equal(a[0], b[0]);self.assertEqual(a[1], b[1])
        self.assertEqual(a[1]["generation_epoch"], 0)
        self.assertEqual(a[1]["generation_seed"], 1)
        a = embed_photo(self.photo, self.row, cfg, OTFSeed(1, "train", 0))
        b = embed_photo(self.photo, self.row, cfg, OTFSeed(90, "train", 0))
        self.assertFalse(np.array_equal(a[0], b[0]))
        tiny = dict(self.row, source_height="1", source_width="1")
        with self.assertRaisesRegex(ValueError, "one-byte"):
            embed_photo(np.zeros((1, 1, 3), np.uint8), tiny,
                        configuration(bit_mode="plane", size={"payload_bytes": 1}), OTFSeed(1, "val", 0))

    def test_png_decode_cache_embedding_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory);Image.fromarray(self.photo).save(root/"example.png")
            cfg = configuration(size={"bpp": 0.2}, placement="random")
            build_entry(self.row, root/"cache", root)
            clean = load_photo(self.row, root)
            direct = embed_photo(clean, self.row, cfg, OTFSeed(1, "val", 0))
            for mmap in (False, True):
                cached = load_cached(self.row, root/"cache", root, mmap=mmap)
                cache = embed_photo(cached, self.row, cfg, OTFSeed(1, "val", 0))
                np.testing.assert_array_equal(direct[0], cache[0]);np.testing.assert_array_equal(direct[2], cache[2]);self.assertEqual(direct[1], cache[1])


if __name__ == "__main__":unittest.main()
