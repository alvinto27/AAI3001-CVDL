import struct
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from lsb_core import derive_seed, embed, load_rgb, payload_length, source_identity, verify_arrays


class CoreTests(unittest.TestCase):
    def test_rounding_and_invalid_rates(self):
        self.assertEqual(payload_length(0.5, 3), 2)
        self.assertEqual(payload_length(0.0001, 3), 0)
        for rate in (0, -1, 1.01, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                payload_length(rate, 3)

    def test_integrity_extremes_and_replay(self):
        clean = np.tile(np.array([0, 255, 154, 155, 1, 254], dtype=np.uint8), 200).reshape(20, 20, 3)
        for rate in (0.00001, 0.1, 0.5, 1):
            stego, mask, selected = embed(clean, rate, 42)
            verify_arrays(clean, stego, mask, selected, rate, 42)
            self.assertTrue(np.array_equal(clean >> 1, stego >> 1))
            self.assertTrue(np.array_equal(mask, clean != stego))
            self.assertEqual(len(selected), len(set(selected)))
            self.assertLessEqual(int(mask.sum()), len(selected))
        self.assertLess(int(mask.sum()), len(selected))
        self.assertEqual(set(selected % 3), {0, 1, 2})

    def test_seed_variation_and_input_not_mutated(self):
        clean = np.zeros((25, 25, 3), dtype=np.uint8)
        first, _, one = embed(clean, 0.4, 1)
        second, _, two = embed(clean, 0.4, 2)
        self.assertFalse(np.array_equal(first, second))
        self.assertFalse(np.array_equal(one, two))
        self.assertFalse(clean.any())
        identity = source_identity(clean)
        self.assertEqual(derive_seed(42, identity), derive_seed(42, identity))
        self.assertNotEqual(derive_seed(42, identity), derive_seed(42, identity, "rate"))

    def test_tampering_detected(self):
        clean = np.zeros((5, 6, 3), dtype=np.uint8)
        stego, mask, selected = embed(clean, 0.4, 42)
        stego[0, 0, 0] ^= 2
        with self.assertRaises(ValueError):
            verify_arrays(clean, stego, mask, selected, 0.4, 42)

    def test_strict_png_and_stable_identity(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            rgb = Image.new("RGB", (7, 9), (1, 2, 3))
            for filename in ("a.png", "renamed.png"):
                rgb.save(root / filename)
            self.assertEqual(source_identity(load_rgb(root / "a.png")), source_identity(load_rgb(root / "renamed.png")))
            for mode in ("RGBA", "L", "P", "I;16"):
                file = root / f"bad-{mode.replace(';', '')}.png"
                Image.new(mode, (7, 9)).save(file)
                with self.assertRaises(ValueError):
                    load_rgb(file)
            rgb.save(root / "jpeg.png", format="JPEG")
            with self.assertRaises(ValueError):
                load_rgb(root / "jpeg.png")
            raw = bytearray((root / "a.png").read_bytes())
            raw[24] = 16
            (root / "fake16.png").write_bytes(raw)
            with self.assertRaises(ValueError):
                load_rgb(root / "fake16.png")
            (root / "broken.png").write_bytes(b"broken")
            with self.assertRaises(ValueError):
                load_rgb(root / "broken.png")
            rgb.save(root / "transparent.png", transparency=(1, 2, 3))
            with self.assertRaises(ValueError):
                load_rgb(root / "transparent.png")


if __name__ == "__main__":
    unittest.main()
