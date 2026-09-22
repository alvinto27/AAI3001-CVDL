"""Core embedding and input-validation tests."""
import struct
import tempfile
import unittest
import zlib
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image

import lsb_core
from lsb_core import embed, load_rgb, payload_length, validate_array


def embed_with_seed(clean: np.ndarray, rate: float = 0.4, seed: int = 42):
    """One embedding step taken from a fresh run-level RNG."""
    return embed(clean, rate, np.random.default_rng(seed))


def write_16bit_rgb_png(path: Path, width: int, height: int) -> None:
    """Build a real 16-bit RGB PNG, which Pillow still reports as mode RGB."""
    rows = b"".join(b"\x00" + bytes([0, 1] * 3 * width) for _ in range(height))

    def chunk(tag: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)

    header = struct.pack(">IIBBBBB", width, height, 16, 2, 0, 0, 0)
    path.write_bytes(b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", header)
                     + chunk(b"IDAT", zlib.compress(rows)) + chunk(b"IEND", b""))


class PayloadLengthTests(unittest.TestCase):
    def test_rounds_half_up(self):
        self.assertEqual(payload_length(0.5, 3), 2)
        self.assertEqual(payload_length(0.0001, 3), 0)
        self.assertEqual(payload_length(1, 231), 231)
        self.assertEqual(payload_length(0.1, 231), 23)

    def test_rejects_invalid_rates(self):
        for rate in (0, -1, 1.01, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                payload_length(rate, 3)


class EmbedTests(unittest.TestCase):
    def setUp(self):
        self.clean = np.tile(np.array([0, 255, 154, 155, 1, 254], dtype=np.uint8), 200).reshape(20, 20, 3)

    def test_only_existing_lsb_values_change(self):
        for rate in (0.00001, 0.1, 0.5, 1):
            stego, mask, _ = embed_with_seed(self.clean, rate)
            self.assertTrue(np.array_equal(self.clean >> 1, stego >> 1))
            self.assertFalse(np.any((self.clean ^ stego) & np.uint8(254)))
            self.assertLessEqual(int(np.abs(self.clean.astype(np.int16) - stego.astype(np.int16)).max()), 1)
            self.assertEqual(mask.shape, self.clean.shape)

    def test_mask_equals_actual_differences(self):
        for rate in (0.00001, 0.1, 0.5, 1):
            stego, mask, _ = embed_with_seed(self.clean, rate)
            self.assertEqual(mask.dtype, np.bool_)
            self.assertTrue(np.array_equal(mask, self.clean != stego))

    def test_selected_location_count_matches_the_rate(self):
        for rate in (0.00001, 0.1, 0.5, 1):
            _, _, selected = embed_with_seed(self.clean, rate)
            self.assertEqual(selected.ndim, 1)
            self.assertEqual(selected.dtype.kind, "i")
            self.assertEqual(len(selected), payload_length(rate, self.clean.size))

    def test_selected_locations_are_unique_and_valid_indices(self):
        for rate in (0.00001, 0.1, 0.5, 1):
            _, _, selected = embed_with_seed(self.clean, rate)
            self.assertEqual(np.unique(selected).size, len(selected))
            if len(selected):
                self.assertGreaterEqual(int(selected.min()), 0)
                self.assertLess(int(selected.max()), self.clean.size)

    def test_changes_are_contained_in_selected_locations(self):
        for rate in (0.00001, 0.1, 0.5, 1):
            stego, mask, selected = embed_with_seed(self.clean, rate)
            changed = np.flatnonzero((self.clean != stego).reshape(-1))
            self.assertTrue(set(changed.tolist()).issubset(set(selected.tolist())))
            self.assertLessEqual(int(mask.sum()), len(selected))

    def test_every_rgb_channel_is_eligible(self):
        for seed in (42, 7):
            _, _, selected = embed_with_seed(self.clean, 0.5, seed)
            self.assertEqual({int(index) % 3 for index in selected}, {0, 1, 2})

    def test_clean_input_is_not_mutated(self):
        original = self.clean.copy()
        embed_with_seed(self.clean, 0.5)
        self.assertTrue(np.array_equal(original, self.clean))

    def test_same_seed_is_deterministic_and_other_seeds_differ(self):
        expected = embed_with_seed(self.clean, 0.4, 7)
        repeated = embed_with_seed(self.clean, 0.4, 7)
        other = embed_with_seed(self.clean, 0.4, 8)
        for actual, reference in zip(repeated, expected):
            self.assertTrue(np.array_equal(actual, reference))
        self.assertFalse(np.array_equal(expected[0], other[0]))
        self.assertFalse(np.array_equal(expected[2], other[2]))

    def test_one_run_rng_reproduces_a_sequence_in_processing_order(self):
        images = [self.clean, np.full((9, 9, 3), 7, dtype=np.uint8), np.tile(self.clean, (2, 1, 1))]

        def run(seed: int):
            rng = np.random.default_rng(seed)
            samples = []
            for image in images:
                rate = float(rng.uniform(0.1, 0.6))
                samples.append((rate, *embed(image, rate, rng)))
            return samples

        first, repeated, other = run(5), run(5), run(6)
        for actual, reference in zip(first, repeated):
            self.assertEqual(actual[0], reference[0])
            for array, expected in zip(actual[1:], reference[1:]):
                self.assertTrue(np.array_equal(array, expected))
        self.assertTrue(any(not np.array_equal(a[1], b[1]) for a, b in zip(first, other)))

    def test_invalid_arrays_are_rejected(self):
        for array in (np.zeros((4, 4, 3), dtype=np.uint16), np.zeros((4, 4), dtype=np.uint8),
                      np.zeros((4, 4, 4), dtype=np.uint8), np.zeros((0, 4, 3), dtype=np.uint8),
                      np.zeros((4, 4, 3), dtype=np.float32)):
            with self.assertRaises(ValueError):
                validate_array(array)
            with self.assertRaises(ValueError):
                embed(array, 0.5, np.random.default_rng(0))


class LoadRgbTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def test_decodes_a_static_rgb_png_unchanged(self):
        pixels = np.arange(7 * 9 * 3, dtype=np.uint8).reshape(9, 7, 3)
        Image.fromarray(pixels).save(self.root / "sample.png")
        loaded = load_rgb(self.root / "sample.png")
        self.assertEqual(loaded.dtype, np.uint8)
        self.assertEqual(loaded.shape, (9, 7, 3))
        self.assertTrue(np.array_equal(loaded, pixels))

    def test_rejects_formats_and_modes_that_would_need_conversion(self):
        rgb = Image.new("RGB", (7, 9), (1, 2, 3))
        rgb.save(self.root / "rgba.png", format="PNG")
        for mode in ("RGBA", "L", "P", "I;16"):
            Image.new(mode, (7, 9)).save(self.root / f"{mode.replace(';', '')}.png")
        rgb.save(self.root / "photo.png", format="JPEG")
        write_16bit_rgb_png(self.root / "deep.png", 7, 9)
        (self.root / "broken.png").write_bytes(b"broken")
        (self.root / "truncated.png").write_bytes((self.root / "rgba.png").read_bytes()[:40])
        rgb.save(self.root / "transparent.png", transparency=(1, 2, 3))
        for name in ("rgba.png", "RGBA.png", "L.png", "P.png", "I16.png", "photo.png",
                     "deep.png", "broken.png", "truncated.png", "transparent.png", "missing.png"):
            with self.subTest(name=name), self.assertRaises(ValueError):
                load_rgb(self.root / name)

    def test_enforces_the_pixel_safety_limit(self):
        Image.new("RGB", (7, 9), (1, 2, 3)).save(self.root / "small.png")
        with patch.object(lsb_core, "MAX_PIXELS", 10):
            with self.assertRaises(ValueError):
                load_rgb(self.root / "small.png")


if __name__ == "__main__":
    unittest.main()
