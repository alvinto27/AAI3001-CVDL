"""Shared sampler and seed invariants; execute on the VM."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "LSB-Dataset-Generator"))
import lsb_settings as settings
from lsb_batch import Config, SizeDoesNotFit, _draw_settings


def key(**changes):
    args = dict(run_seed=7, purpose="train", epoch=2, photo_index=1000,
                grid_policy="overlap_last", tile_size=512, tile_row=0, tile_col=0,
                variant=1, stream="settings")
    args.update(changes)
    return settings.otf_seed_sequence(**args)


def state(sequence):
    return sequence.generate_state(16).tolist()


class SettingsTests(unittest.TestCase):
    def config(self, **changes):
        values = dict(seed=1, input="unused", output="unused-out", bit_mode="plane", k=1,
                      placement="stride", stride=16, size={"bpp": 0.5})
        values.update(changes)
        return Config(**values)

    def test_legacy_formula_and_aliases(self):
        self.assertEqual(state(settings.legacy_seed_sequence(1, 3, 1, 0)),
                         state(np.random.SeedSequence([1, 3, 1, 0])))
        self.assertIs(_draw_settings, settings.draw_legacy_settings)
        self.assertIs(SizeDoesNotFit, settings.SizeDoesNotFit)

    def test_repeat_settings_and_embedding_stream(self):
        cfg = self.config(size={"bpp": [0.05, 0.5]}, k=[1, 2, 3], method=["replace", "match"])
        image = np.zeros((512, 512, 3), np.uint8)
        a, am = settings.draw_otf_settings(cfg, image, np.random.default_rng(key()))
        b, bm = settings.draw_otf_settings(cfg, image, np.random.default_rng(key()))
        self.assertEqual({k:v for k,v in a.items() if k != "rng"}, {k:v for k,v in b.items() if k != "rng"})
        self.assertEqual(am, bm)
        self.assertEqual(state(key(stream="embedding")), state(key(stream="embedding")))
        self.assertNotEqual(state(key()), state(key(stream="embedding")))

    def test_key_parts_change_stream(self):
        for change in (dict(tile_row=1), dict(tile_col=1), dict(epoch=3), dict(variant=2),
                       dict(photo_index=1001), dict(tile_size=256), dict(run_seed=8)):
            self.assertNotEqual(state(key()), state(key(**change)))

    def test_fixed_evaluation_and_fixed_training(self):
        for purpose in ("val", "test"):
            self.assertEqual(state(key(purpose=purpose)), state(key(purpose=purpose, run_seed=999, epoch=999)))
        self.assertNotEqual(state(key(purpose="val")), state(key(purpose="test")))
        self.assertEqual(state(key(fixed_variants=True)), state(key(fixed_variants=True, epoch=999)))

    def test_photo_seed_excludes_extraction_grid(self):
        self.assertEqual(state(key(carrier="photo")),
                         state(key(carrier="photo", tile_size=256, tile_row=13, tile_col=19)))
        self.assertNotEqual(state(key()), state(key(carrier="photo")))

    def test_capacity_only_changes_new_path(self):
        image = np.zeros((512, 512, 3), np.uint8)
        for size in ({"bpp": 0.5}, {"bpp": [0.4, 0.5]},
                     {"payload_bytes": 100000}, {"payload_bytes": [100000, 120000]}):
            cfg = self.config(size=size)
            with self.assertRaises(SizeDoesNotFit):
                settings.draw_legacy_settings(cfg, image, np.random.default_rng(1))
            kwargs, metadata = settings.draw_otf_settings(cfg, image, np.random.default_rng(key()))
            self.assertEqual(metadata["payload_bytes"], 6144)
            self.assertEqual(metadata["effective_bpp"], 0.1875)
            self.assertGreater(metadata["requested_bpp"], metadata["effective_bpp"])
            self.assertTrue(metadata["clipped"])
            self.assertEqual(kwargs[metadata["size_kind"]], metadata["size_value"])

    def test_rounding_and_impossible_carrier(self):
        image = np.zeros((512, 512, 3), np.uint8)
        _, meta = settings.draw_otf_settings(self.config(size={"bpp": 0.05}), image, np.random.default_rng(1))
        self.assertEqual(meta["payload_bytes"], 1638)
        self.assertEqual(meta["effective_bpp"], 1638 * 8 / (512 * 512))
        self.assertFalse(meta["clipped"])
        with self.assertRaises(SizeDoesNotFit):
            settings.draw_otf_settings(self.config(), np.zeros((1, 1, 3), np.uint8), np.random.default_rng(1))

    def test_bad_key_parts(self):
        for changes in (dict(purpose="other"), dict(stream="other"), dict(variant=0),
                        dict(tile_row=-1), dict(epoch=True), dict(photo_index=-1),
                        dict(grid_policy="topleft_crop", tile_size=256)):
            with self.assertRaises(ValueError):
                key(**changes)


if __name__ == "__main__":
    unittest.main()
