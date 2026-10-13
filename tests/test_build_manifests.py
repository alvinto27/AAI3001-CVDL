"""Small record-builder tests. Run with unittest on the VM, not the PC."""
from __future__ import annotations

import contextlib
import importlib.util
import io
import struct
import tempfile
import unittest
import zlib
from pathlib import Path

import numpy as np

SPEC = importlib.util.spec_from_file_location(
    "build_manifests", Path(__file__).resolve().parents[1] / "scripts/build_manifests.py")
builder = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(builder)


class ManifestTests(unittest.TestCase):
    def test_overlap_starts_cover_every_pixel(self):
        for length in (512, 513, 648, 1024, 1100, 2040, 4948):
            for tile in (256, 512):
                values = builder.starts(length, tile)
                self.assertEqual(values[0], 0)
                self.assertEqual(values[-1] + tile, length)
                self.assertTrue(all(a < b <= a + tile for a, b in zip(values, values[1:])))
                self.assertTrue(all(0 <= start <= length - tile for start in values))
        with self.assertRaises(ValueError):
            builder.starts(511, 512)

    def test_div2k_seed_matches_approved_draw(self):
        expected = {int(i) + 1 for i in np.random.default_rng(0).permutation(800)[:680]}
        self.assertEqual(builder.div2k_train_ids(), expected)
        self.assertEqual(len(expected), 680)
        self.assertEqual(builder.div2k_train_ids(), builder.div2k_train_ids())

    def test_png_header_strict_rgb8(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "image.png"
            def header(depth=8, colour=2):
                data = b"IHDR" + struct.pack(">IIBBBBB", 2040, 1404, depth, colour, 0, 0, 0)
                return b"\x89PNG\r\n\x1a\n" + struct.pack(">I", 13) + data + struct.pack(">I", zlib.crc32(data))
            path.write_bytes(header())
            self.assertEqual(builder.png_dimensions(path), (1404, 2040))
            for depth, colour in ((16, 2), (8, 3), (8, 0), (8, 6)):
                path.write_bytes(header(depth, colour))
                with self.assertRaises(ValueError):
                    builder.png_dimensions(path)
            path.write_bytes(header()[:-1] + b"\xff")
            with self.assertRaises(ValueError):
                builder.png_dimensions(path)
            path.write_bytes(b"invalid")
            with self.assertRaises(ValueError):
                builder.png_dimensions(path)

    def test_identity_and_source_fields(self):
        row = builder.photo("div2k", "div2k_0001", 1000,
                            "ImageSets/DIV2K_train_HR/0001.png", 1404, 2040, "train")
        self.assertEqual((row["overlap512_rows"], row["overlap512_cols"]), (3, 4))
        self.assertEqual(row["legacy512_rows"], "")
        self.assertEqual(row["noise"], "not_applicable")
        self.assertEqual(row["camera_model"], "")
        for unsafe in ("../outside.png", "/absolute.png", "x\\y.png"):
            with self.assertRaises(ValueError):
                builder.safe_relative(unsafe)

    def test_duplicate_keys_fail(self):
        with self.assertRaises(ValueError):
            builder.unique([{"id": "a"}, {"id": "a"}], "id", "test")

    def test_existing_destination_prevents_all_writes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            existing = root / "old.csv"
            existing.write_bytes(b"keep me\n")
            with self.assertRaises(ValueError):
                builder.write_outputs(root, {
                    "new.csv": (("id",), [{"id": 1}]),
                    "old.csv": (("id",), [{"id": 2}]),
                })
            self.assertEqual(existing.read_bytes(), b"keep me\n")
            self.assertFalse((root / "new.csv").exists())

    def test_source_checkout_and_symlink_outputs_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, out = root / "source", root / "out"
            (source / "data").mkdir(parents=True)
            out.mkdir()
            for destination in (source / "outputs", out):
                if destination == out:
                    (out / "data").symlink_to(source / "data", target_is_directory=True)
                with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                    builder.main(["--repo-root", str(source), "--output-root", str(destination)])
            self.assertEqual(list((source / "data").iterdir()), [])

    def test_malformed_csv_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "rows.csv"
            for data in ("other\\nx\\n", "id,path\\na\\n", "id,path\\na,b,extra\\n", "id,path\\n"):
                path.write_text(data, encoding="utf-8")
                with self.assertRaises(ValueError):
                    builder.read_csv(path, {"id", "path"})

    def test_csv_bytes_repeat_and_use_lf(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            outputs = {"rows.csv": (("id", "path"), [{"id": "000003", "path": "clean/000003.png"}])}
            builder.write_outputs(root / "one", outputs)
            builder.write_outputs(root / "two", outputs)
            first = (root / "one/rows.csv").read_bytes()
            self.assertEqual(first, (root / "two/rows.csv").read_bytes())
            self.assertEqual(first, b"id,path\n000003,clean/000003.png\n")


if __name__ == "__main__":
    unittest.main()
