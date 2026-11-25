"""Source and geometry checks; execute only on the team VM."""
from __future__ import annotations

import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "LSB-Dataset-Generator"))
import lsb_prepare
import lsb_raw
import lsb_source as source


def record(h=1100, w=1600, kind="png8"):
    row = dict(photo_id="example", source="raise" if kind == "nef" else "div2k",
               source_type=kind, source_height=str(h), source_width=str(w),
               master_path="example.png", noise="disabled" if kind == "nef" else "not_applicable",
               grid_policy="overlap_last", legacy512_policy="topleft_crop",
               legacy512_rows=str(h // 512), legacy512_cols=str(w // 512),
               legacy512_height=str(h // 512 * 512), legacy512_width=str(w // 512 * 512))
    for tile in (512, 256):
        ys, xs = source.overlap_starts(h, tile), source.overlap_starts(w, tile)
        row.update({f"overlap{tile}_y": ";".join(map(str, ys)),
                    f"overlap{tile}_x": ";".join(map(str, xs)),
                    f"overlap{tile}_rows": str(len(ys)), f"overlap{tile}_cols": str(len(xs))})
    return row


class SourceTests(unittest.TestCase):
    def test_coverage_no_padding_and_identity(self):
        for h, w in ((512, 512), (513, 649), (1024, 1024), (1100, 2040)):
            pixels = np.arange(h * w * 3, dtype=np.uint8).reshape(h, w, 3)
            row = record(h, w)
            for size in (256, 512):
                covered = np.zeros((h, w), dtype=bool)
                tiles = list(source.iter_tiles(pixels, row, tile_size=size))
                self.assertEqual(len(tiles), int(row[f"overlap{size}_rows"]) * int(row[f"overlap{size}_cols"]))
                for tile in tiles:
                    self.assertEqual(tile.pixels.shape, (size, size, 3))
                    self.assertTrue(np.shares_memory(tile.pixels, pixels))
                    np.testing.assert_array_equal(tile.pixels, pixels[tile.y:tile.y+size, tile.x:tile.x+size])
                    self.assertEqual(tile.identity, ("example", "overlap_last", size,
                                                     tile.row, tile.column, tile.y, tile.x))
                    covered[tile.y:tile.y+size, tile.x:tile.x+size] = True
                self.assertTrue(covered.all())
                self.assertEqual((tiles[-1].y + size, tiles[-1].x + size), (h, w))

    def test_exact_multiple_and_one_tile(self):
        self.assertEqual(source.overlap_starts(1024, 512), (0, 512))
        self.assertEqual(source.overlap_starts(512, 512), (0,))
        self.assertEqual(source.overlap_starts(1100, 512), (0, 512, 588))
        for length, tile in ((511, 512), (0, 512), (512, 0), (512, True)):
            with self.assertRaises(ValueError):
                source.overlap_starts(length, tile)

    def test_legacy_equals_original_positions(self):
        row = record(kind="nef")
        pixels = np.zeros((1100, 1600, 3), dtype=np.uint8)
        tiles = list(source.iter_tiles(pixels, row, "topleft_crop", 512))
        self.assertEqual([(t.row, t.column) for t in tiles],
                         lsb_prepare._positions(1100, 1600, 512, None, 0, 0))
        self.assertEqual(len(tiles), 6)
        self.assertEqual(source.topleft_crop(pixels, row).shape, (1024, 1536, 3))
        row["legacy512_cols"] = "4"
        with self.assertRaises(ValueError):
            list(source.iter_tiles(pixels, row, "topleft_crop"))

    def test_bad_manifest_grid_or_shape(self):
        row = record()
        pixels = np.zeros((1100, 1600, 3), dtype=np.uint8)
        row["overlap512_x"] = "0;512;1024"
        with self.assertRaises(ValueError):
            list(source.iter_tiles(pixels, row))
        with self.assertRaises(ValueError):
            list(source.iter_tiles(pixels.astype(np.uint16), record()))
        with self.assertRaises(ValueError):
            list(source.iter_tiles(pixels[:-1], record()))
        with self.assertRaises(ValueError):
            list(source.iter_tiles(pixels, record(), "topleft_crop"))
        with self.assertRaises(ValueError):
            source.grid_axes(record(), "unknown", 512)

    def test_png_rgb_pixels_size_and_modes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pixels = np.arange(512 * 512 * 3, dtype=np.uint8).reshape(512, 512, 3)
            Image.fromarray(pixels).save(root / "example.png")
            row = record(512, 512)
            actual = source.load_photo(row, root)
            self.assertEqual(actual.dtype, np.uint8)
            self.assertTrue(actual.flags.c_contiguous)
            np.testing.assert_array_equal(actual, pixels)
            row["source_height"] = "513"
            with self.assertRaises(ValueError):
                source.load_photo(row, root)
            for mode in ("L", "P", "RGBA", "I;16"):
                Image.new(mode, (512, 512)).save(root / "example.png")
                with self.assertRaises(ValueError):
                    source.load_photo(record(512, 512), root)

    def test_no_raw_path_for_png(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            Image.new("RGB", (512, 512)).save(root / "example.png")
            with patch.object(source, "decode_raw_pixels", side_effect=AssertionError("RAW called")), \
                 patch.object(source, "configure_source_worker", side_effect=AssertionError("RAW hook called")):
                source.load_photo(record(512, 512), root)
            row = record(512, 512)
            row["noise"] = "disabled"
            with self.assertRaises(ValueError):
                source.load_photo(row, root)

    def test_shared_settings_and_disabled_hook_order(self):
        events = []
        class Raw:
            def __enter__(self):
                return self
            def __exit__(self, *args):
                pass
            def postprocess(self, **kwargs):
                events.append("postprocess")
                self.kwargs = kwargs
                return np.zeros((2, 3, 3), dtype=np.uint8)
        raw = Raw()
        fake = types.SimpleNamespace(__version__="test", imread=lambda _: raw,
                                     ColorSpace=types.SimpleNamespace(sRGB="colour_enum"),
                                     DemosaicAlgorithm=types.SimpleNamespace(AHD="ahd_enum"))
        with patch.dict(sys.modules, {"rawpy": fake}), \
             patch.object(lsb_raw, "_disabled_raw_noise_hook", side_effect=lambda _: events.append("hook")):
            lsb_raw.decode_raw_pixels(Path("test.NEF"))
        self.assertEqual(events, ["hook", "postprocess"])
        self.assertEqual(raw.kwargs, dict(use_camera_wb=True, no_auto_bright=True,
                                         output_color="colour_enum", output_bps=8,
                                         demosaic_algorithm="ahd_enum"))
        self.assertIs(lsb_prepare.raw_settings_record, lsb_raw.raw_settings_record)
        self.assertIs(lsb_prepare.decode_raw_pixels, lsb_raw.decode_raw_pixels)
        settings = lsb_raw.raw_settings_record()
        settings["output_bps"] = 16
        self.assertEqual(lsb_raw.RAW_SETTINGS["output_bps"], 8)

    def test_preparation_raw_record_is_unchanged(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "input").mkdir()
            config = lsb_prepare.PrepareConfig(str(root / "input"), str(root / "output"))
            with patch.dict(sys.modules, {"rawpy": types.SimpleNamespace(__version__="test")}):
                actual = config.as_dict()
            self.assertEqual(actual["raw_settings"], {
                "use_camera_wb": True, "no_auto_bright": True, "output_color": "sRGB",
                "output_bps": 8, "demosaic_algorithm": "AHD", "orientation": "default",
            })
            self.assertEqual(actual["version"], "6.0.0")

    def test_worker_thread_guard(self):
        with patch.dict(sys.modules, {"rawpy": object()}), patch.dict(os.environ, {"OMP_NUM_THREADS": "8"}):
            with self.assertRaises(RuntimeError):
                source.configure_source_worker()
        with patch.dict(os.environ, {"OMP_NUM_THREADS": "1"}):
            source.configure_source_worker()
            self.assertEqual(os.environ["OMP_NUM_THREADS"], "1")


if __name__ == "__main__":
    unittest.main()
