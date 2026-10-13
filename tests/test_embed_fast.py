"""Fast-path output options, clean-array safety, and exception parity."""
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "LSB-Dataset-Generator"))
from lsb_engine import embed, embed_fast, extract


class FastTests(unittest.TestCase):
    def test_view_readonly_safety_and_metadata(self):
        source = np.arange(37*43*3, dtype=np.uint8).reshape(37,43,3)
        carrier = source[::2, ::-1]
        carrier.setflags(write=False)
        snapshot = source.copy()
        args = dict(bit_mode="lowest", k=3, method="match", placement="random", payload_bytes=50)
        a, mask, record = embed(carrier, rng=np.random.default_rng(4), **args)
        b, metadata, changes = embed_fast(carrier, rng=np.random.default_rng(4), return_changed_units=True, **args)
        np.testing.assert_array_equal(a, b)
        np.testing.assert_array_equal(source, snapshot)
        np.testing.assert_array_equal(changes, np.flatnonzero(carrier.reshape(-1) != b.reshape(-1)))
        self.assertEqual(metadata, {k:v for k,v in vars(record).items() if not k.startswith("_")})
        self.assertEqual(len(extract(a, record)), 50)
        self.assertFalse(changes.flags.writeable)

    def test_mask_absent_and_indices_not_requested(self):
        carrier = np.zeros((32,32,3), dtype=np.uint8)
        rng = np.random.default_rng(1)
        with patch("lsb_engine.np.zeros", side_effect=AssertionError("mask allocation")):
            _, metadata, changes = embed_fast(carrier, "plane", 1, "replace", "contiguous",
                                              rng, payload_bytes=1)
        self.assertIsNone(changes)
        self.assertNotIn("_selected_units", metadata)
        self.assertIn("units_changed", metadata)

    def test_validation_exception_and_rng_parity(self):
        carrier = np.zeros((16,16,3), dtype=np.uint8)
        base = dict(bit_mode="lowest", k=1, method="replace", placement="contiguous", payload_bytes=1)
        for change in (dict(k=0), dict(k=True), dict(method="bad"), dict(payload="bad"),
                       dict(placement="bad"), dict(start=1000), dict(payload_bytes=10000),
                       dict(bpp=0.1), dict(payload_bytes=None, bpp=float("nan"))):
            errors = []
            for function in (embed, embed_fast):
                rng = np.random.default_rng(7)
                before = rng.bit_generator.state
                try:
                    function(carrier, rng=rng, **{**base, **change})
                except Exception as error:
                    errors.append((type(error), str(error)))
                else:
                    self.fail("invalid request accepted")
                self.assertEqual(before, rng.bit_generator.state)
            self.assertEqual(errors[0], errors[1])


if __name__ == "__main__":
    unittest.main()
