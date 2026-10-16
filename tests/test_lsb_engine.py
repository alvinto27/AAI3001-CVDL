"""Tests for the pure NumPy v6 embedding engine."""

import unittest

import numpy as np

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "LSB-Dataset-Generator"))
import lsb_engine


class EngineTests(unittest.TestCase):
    """Exercise embedding, extraction, validation, and edge values."""

    def setUp(self) -> None:
        self.carrier = np.arange(8 * 9 * 3, dtype=np.uint8).reshape(8, 9, 3)

    @staticmethod
    def _expected_payload(seed: int, kind: str, count: int) -> bytes:
        """Reproduce the payload's first RNG draw for an independent check."""
        rng = np.random.default_rng(seed)
        if kind == "letters":
            alphabet = np.frombuffer(
                b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz", dtype=np.uint8
            )
            return bytes(alphabet[rng.integers(0, alphabet.size, size=count)])
        return rng.integers(0, 256, size=count, dtype=np.uint8).tobytes()

    @staticmethod
    def _selected(result: lsb_engine.EmbedResult) -> np.ndarray:
        """Resolve indices for assertions without relying on engine internals."""
        if result.placement == "random":
            assert result.units is not None
            return result.units
        stride = result.stride if result.placement == "stride" else 1
        return result.start + np.arange(result.units_used, dtype=np.intp) * stride

    def test_all_modes_methods_placements_round_trip(self) -> None:
        """Every k, method, and placement reads back the generated payload."""
        for bit_mode in ("lowest", "plane"):
            for k in range(1, 9):
                for method in ("replace", "match"):
                    for placement in ("contiguous", "stride", "random"):
                        seed = 1000 + k
                        kwargs = {
                            "bit_mode": bit_mode,
                            "k": k,
                            "method": method,
                            "placement": placement,
                            "rng": np.random.default_rng(seed),
                            "payload": "random_bytes",
                            "payload_bytes": 4,
                        }
                        if placement == "stride":
                            kwargs["stride"] = 2
                        stego, _, result = lsb_engine.embed(self.carrier, **kwargs)
                        expected = self._expected_payload(seed, "random_bytes", 4)
                        self.assertEqual(lsb_engine.extract(stego, result), expected)
                        self.assertEqual(result.units_used * (k if bit_mode == "lowest" else 1), result.embedded_bits)

    def test_replace_preserves_bits_outside_lowest_target(self) -> None:
        """Lowest-k replacement does not alter any higher bit."""
        for k in range(1, 9):
            stego, _, result = lsb_engine.embed(
                self.carrier,
                bit_mode="lowest",
                k=k,
                method="replace",
                placement="random",
                rng=np.random.default_rng(k),
                payload_bytes=5,
            )
            selected = self._selected(result)
            changed = self.carrier.reshape(-1) != stego.reshape(-1)
            outside = np.ones(self.carrier.size, dtype=bool)
            outside[selected] = False
            self.assertFalse(np.any(changed[outside]))
            field_mask = (1 << k) - 1
            self.assertTrue(
                np.array_equal(
                    self.carrier.reshape(-1)[selected] & (~field_mask & 0xFF),
                    stego.reshape(-1)[selected] & (~field_mask & 0xFF),
                )
            )

    def test_plane_replace_changes_only_requested_bit(self) -> None:
        """Plane replacement preserves all seven non-target bits."""
        for k in range(1, 9):
            stego, _, result = lsb_engine.embed(
                self.carrier,
                bit_mode="plane",
                k=k,
                method="replace",
                placement="contiguous",
                start=3,
                rng=np.random.default_rng(k),
                payload_bytes=2,
            )
            selected = self._selected(result)
            before = self.carrier.reshape(-1)
            after = stego.reshape(-1)
            target_mask = np.uint8(1 << (k - 1))
            self.assertTrue(np.all((before[selected] ^ after[selected]) & ~target_mask == 0))
            changed = before != after
            outside = np.ones(before.size, dtype=bool)
            outside[selected] = False
            self.assertFalse(np.any(changed[outside]))

    def test_match_uses_valid_nearest_values_and_bounds(self) -> None:
        """Match stays in range and chooses one of the nearest valid neighbours."""
        values = np.resize(np.array([0, 255], dtype=np.uint8), self.carrier.size)
        carrier = values.reshape(self.carrier.shape)
        for bit_mode in ("lowest", "plane"):
            for k in range(1, 9):
                stego, _, result = lsb_engine.embed(
                    carrier,
                    bit_mode=bit_mode,
                    k=k,
                    method="match",
                    placement="contiguous",
                    start=0,
                    rng=np.random.default_rng(200 + k),
                    payload_bytes=8,
                )
                selected = self._selected(result)
                before = carrier.reshape(-1)[selected].astype(np.int16)
                after = stego.reshape(-1)[selected].astype(np.int16)
                self.assertTrue(np.all((0 <= after) & (after <= 255)))
                changed = before != after
                if bit_mode == "lowest":
                    mask = (1 << k) - 1
                    target = after & mask
                    if k == 8:
                        self.assertTrue(np.array_equal(after, target))
                    else:
                        step = 1 << k
                        below = target + np.floor_divide(before - target, step) * step
                        above = below + step
                        valid = ((after == before) | (after == below) | (after == above))
                        self.assertTrue(np.all(valid))
                        self.assertTrue(np.all(after[changed] >= 0))
                        self.assertTrue(np.all(after[changed] <= 255))
                else:
                    step = 1 << (k - 1)
                    self.assertTrue(np.all(np.abs(after[changed] - before[changed]) == step))

    def test_match_lsb_changes_by_one(self) -> None:
        """Lowest-bit matching has only the standard plus/minus-one change."""
        carrier = np.resize(np.array([0, 1, 254, 255], dtype=np.uint8), self.carrier.size).reshape(self.carrier.shape)
        stego, _, result = lsb_engine.embed(
            carrier,
            bit_mode="lowest",
            k=1,
            method="match",
            placement="random",
            rng=np.random.default_rng(77),
            payload_bytes=8,
        )
        before = carrier.reshape(-1)[result.units].astype(np.int16)
        after = stego.reshape(-1)[result.units].astype(np.int16)
        changed = before != after
        self.assertTrue(np.all(np.abs(after[changed] - before[changed]) == 1))
        self.assertTrue(np.all((0 <= after) & (after <= 255)))

    def test_mask_marks_selected_pixels_even_when_unchanged(self) -> None:
        """The mask follows selected units, not only numerical differences."""
        carrier = np.zeros((3, 4, 3), dtype=np.uint8)
        stego, mask, result = lsb_engine.embed(
            carrier,
            bit_mode="lowest",
            k=8,
            method="replace",
            placement="contiguous",
            start=1,
            rng=np.random.default_rng(4),
            payload_bytes=1,
        )
        expected = np.zeros(carrier.shape[:2], dtype=bool)
        expected.reshape(-1)[self._selected(result) // 3] = True
        self.assertTrue(np.array_equal(mask, expected))
        self.assertEqual(mask.dtype, np.bool_)
        self.assertIsNot(mask, carrier)
        self.assertFalse(np.shares_memory(stego, carrier))

    def test_bpp_converts_to_byte_count_and_reports_actual_rate(self) -> None:
        """Bpp size uses rounded bytes, with a minimum one-byte payload."""
        stego, _, result = lsb_engine.embed(
            self.carrier,
            bit_mode="plane",
            k=8,
            method="replace",
            placement="contiguous",
            rng=np.random.default_rng(11),
            bpp=0.4,
        )
        expected_bytes = max(1, round(0.4 * self.carrier.shape[0] * self.carrier.shape[1] / 8))
        self.assertEqual(result.payload_bytes, expected_bytes)
        self.assertAlmostEqual(result.bpp_actual, expected_bytes * 8 / (8 * 9))
        self.assertEqual(len(lsb_engine.extract(stego, result)), expected_bytes)

    def test_bpp_validation_and_minimum_payload(self) -> None:
        """Reject invalid bpp requests and enforce one-byte minimum payloads."""
        options = dict(
            carrier=self.carrier,
            bit_mode="plane",
            k=1,
            method="replace",
            placement="contiguous",
            rng=np.random.default_rng(20),
        )
        for value in (0, -1, float("nan"), float("inf"), -float("inf")):
            with self.subTest(bpp=value), self.assertRaises(ValueError):
                lsb_engine.embed(**options, bpp=value)
        for value in (True, False, np.bool_(True)):
            with self.subTest(bpp=value), self.assertRaises(TypeError):
                lsb_engine.embed(**options, bpp=value)

        with self.assertRaises(ValueError):
            lsb_engine.embed(**options, payload_bytes=1, bpp=0.1)
        with self.assertRaises(ValueError):
            lsb_engine.embed(**options)

        tiny_options = dict(options)
        tiny_options["rng"] = np.random.default_rng(21)
        _, _, result = lsb_engine.embed(**tiny_options, bpp=1e-6)
        self.assertEqual(result.payload_bytes, 1)

    def test_invalid_parameters_raise(self) -> None:
        """Reject unsupported modes, methods, placements, payloads, and k values."""
        options = dict(
            carrier=self.carrier,
            bit_mode="plane",
            k=1,
            method="replace",
            placement="contiguous",
            rng=np.random.default_rng(22),
            payload_bytes=1,
        )
        for value in (0, 9):
            invalid_options = dict(options)
            invalid_options["k"] = value
            with self.subTest(k=value), self.assertRaises(ValueError):
                lsb_engine.embed(**invalid_options)
        for name, value in (
            ("bit_mode", "unknown"),
            ("method", "unknown"),
            ("placement", "unknown"),
            ("payload", "unknown"),
        ):
            invalid_options = dict(options)
            invalid_options[name] = value
            with self.subTest(parameter=name), self.assertRaises(ValueError):
                lsb_engine.embed(**invalid_options)

    def test_start_boundaries_and_random_starts(self) -> None:
        """Accept the largest legal start and reject the next index."""
        total_units = self.carrier.size
        units_used = 8
        for placement, stride in (("contiguous", 1), ("stride", 3)):
            maximum = total_units - 1 - (units_used - 1) * stride
            common = dict(
                carrier=self.carrier,
                bit_mode="plane",
                k=1,
                method="replace",
                placement=placement,
                stride=stride,
                payload_bytes=1,
            )
            _, _, result = lsb_engine.embed(
                **common,
                start=maximum,
                rng=np.random.default_rng(30 + stride),
            )
            self.assertEqual(result.start, maximum)
            with self.assertRaises(ValueError):
                lsb_engine.embed(
                    **common,
                    start=maximum + 1,
                    rng=np.random.default_rng(40 + stride),
                )

            for seed in range(50):
                _, _, drawn = lsb_engine.embed(
                    **common,
                    start=None,
                    rng=np.random.default_rng(seed),
                )
                self.assertIsNotNone(drawn.start)
                self.assertGreaterEqual(drawn.start, 0)
                self.assertLessEqual(drawn.start, maximum)

    def test_plane_match_preserves_lower_bits(self) -> None:
        """Plane matching leaves every bit below the selected plane unchanged."""
        before = self.carrier.reshape(-1)
        for k in range(2, 9):
            stego, _, _ = lsb_engine.embed(
                self.carrier,
                bit_mode="plane",
                k=k,
                method="match",
                placement="random",
                rng=np.random.default_rng(50 + k),
                payload_bytes=8,
            )
            lower_mask = (1 << (k - 1)) - 1
            self.assertTrue(
                np.array_equal(
                    before & lower_mask,
                    stego.reshape(-1) & lower_mask,
                )
            )

    def test_match_direction_is_fair(self) -> None:
        """Valid match directions occur with near-equal frequency."""
        flat = np.tile(np.arange(128, dtype=np.uint8), 300)
        carrier = flat.reshape(100, 128, 3)
        for bit_mode, k in (("lowest", 1), ("plane", 3)):
            stego, _, _ = lsb_engine.embed(
                carrier,
                bit_mode=bit_mode,
                k=k,
                method="match",
                placement="contiguous",
                start=0,
                rng=np.random.default_rng(60 + k),
                payload_bytes=4800,
            )
            before = carrier.reshape(-1)
            after = stego.reshape(-1)
            changed = before != after
            self.assertGreater(np.count_nonzero(changed), 10000)
            up_moves = np.count_nonzero(after[changed] > before[changed])
            share = up_moves / np.count_nonzero(changed)
            self.assertGreater(share, 0.45)
            self.assertLess(share, 0.55)

    def test_larger_bpp_round_trip_for_each_placement(self) -> None:
        """A larger bpp payload extracts correctly for every placement."""
        carrier = np.random.default_rng(70).integers(
            0,
            256,
            size=(256, 256, 3),
            dtype=np.uint8,
        )
        for index, placement in enumerate(("contiguous", "stride", "random")):
            seed = 80 + index
            kwargs = dict(
                bit_mode="lowest",
                k=3,
                method="match",
                placement=placement,
                rng=np.random.default_rng(seed),
                bpp=0.4,
            )
            if placement == "stride":
                kwargs["stride"] = 3
            stego, _, result = lsb_engine.embed(carrier, **kwargs)
            expected = self._expected_payload(seed, "random_bytes", result.payload_bytes)
            self.assertEqual(lsb_engine.extract(stego, result), expected)

    def test_capacity_failure_does_not_consume_rng(self) -> None:
        """Capacity validation leaves the random generator untouched."""
        seed = 90
        rng = np.random.default_rng(seed)
        with self.assertRaises(ValueError):
            lsb_engine.embed(
                self.carrier,
                bit_mode="plane",
                k=1,
                method="replace",
                placement="random",
                rng=rng,
                payload_bytes=100,
            )
        fresh = np.random.default_rng(seed)
        self.assertEqual(rng.integers(0, 2**32), fresh.integers(0, 2**32))

    def test_letters_payload_is_ascii_letters(self) -> None:
        """The letters payload never emits punctuation or digits."""
        stego, _, result = lsb_engine.embed(
            self.carrier,
            bit_mode="lowest",
            k=3,
            method="replace",
            placement="stride",
            stride=2,
            rng=np.random.default_rng(12),
            payload="letters",
            payload_bytes=20,
        )
        payload = lsb_engine.extract(stego, result)
        self.assertEqual(len(payload), 20)
        self.assertTrue(all(byte in b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz" for byte in payload))

    def test_invalid_start_and_capacity_raise(self) -> None:
        """Illegal non-wrapping starts and over-capacity payloads fail clearly."""
        with self.assertRaises(ValueError):
            lsb_engine.embed(
                self.carrier,
                bit_mode="plane",
                k=1,
                method="replace",
                placement="contiguous",
                start=self.carrier.size,
                rng=np.random.default_rng(1),
                payload_bytes=1,
            )
        with self.assertRaises(ValueError):
            lsb_engine.embed(
                self.carrier,
                bit_mode="plane",
                k=1,
                method="replace",
                placement="stride",
                stride=3,
                start=1,
                rng=np.random.default_rng(1),
                payload_bytes=10,
            )
        with self.assertRaises(ValueError):
            lsb_engine.embed(
                self.carrier,
                bit_mode="plane",
                k=1,
                method="replace",
                placement="random",
                rng=np.random.default_rng(1),
                payload_bytes=30,
            )

    def test_same_seed_is_identical(self) -> None:
        """All random choices are controlled by the passed generator."""
        kwargs = dict(
            bit_mode="lowest",
            k=5,
            method="match",
            placement="random",
            payload="letters",
            payload_bytes=7,
        )
        first = lsb_engine.embed(self.carrier, rng=np.random.default_rng(9), **kwargs)
        second = lsb_engine.embed(self.carrier, rng=np.random.default_rng(9), **kwargs)
        self.assertTrue(np.array_equal(first[0], second[0]))
        self.assertTrue(np.array_equal(first[1], second[1]))
        self.assertEqual(first[2], second[2])
        self.assertTrue(np.array_equal(first[2].units, second[2].units))


if __name__ == "__main__":
    unittest.main()
