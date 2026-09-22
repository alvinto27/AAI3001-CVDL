"""Batch generation, output contract and command-line tests."""
import csv
import json
import subprocess
import sys
import tempfile
import threading
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image

from lsb_batch import Config, METADATA_FIELDS, generate, read_config, write_config
from lsb_core import VERSION, load_rgb, payload_length

GENERATOR = Path(__file__).resolve().parents[1] / "generate.py"


def cli_summary(stdout: str) -> dict:
    return json.loads(stdout[stdout.index("{"):])


class BatchTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.source = self.root / "input"
        self.source.mkdir()
        Image.new("RGB", (11, 7), (10, 20, 31)).save(self.source / "one.png")
        Image.new("RGB", (5, 8), (42, 57, 91)).save(self.source / "two.PNG")
        self.config = Config(str(self.source), str(self.root / "out"))

    def tearDown(self):
        self.temp.cleanup()

    @property
    def out(self) -> Path:
        return self.root / "out"

    def rows(self, path=None):
        with ((path or self.out) / "metadata.csv").open(newline="", encoding="utf-8") as file:
            return list(csv.DictReader(file))

    def test_successful_run_writes_the_documented_structure(self):
        result = generate(self.config)
        self.assertEqual(result["status"], "completed")
        self.assertEqual((result["total_discovered"], result["total_generated"], result["total_skipped"]), (2, 2, 0))
        for folder in ("clean", "stego", "masks", "selected_locations"):
            self.assertTrue((self.out / folder).is_dir(), folder)
        for artifact in ("metadata.csv", "run_config.yaml", "run_summary.json", "generation.log"):
            self.assertTrue((self.out / artifact).is_file(), artifact)
        self.assertEqual(sorted(path.name for path in (self.out / "clean").iterdir()), ["000001.png", "000002.png"])
        self.assertEqual(sorted(path.name for path in (self.out / "stego").iterdir()), ["000001.png", "000002.png"])
        self.assertEqual(sorted(path.name for path in (self.out / "masks").iterdir()), ["000001.npy", "000002.npy"])
        self.assertEqual(sorted(path.name for path in (self.out / "selected_locations").iterdir()),
                         ["000001.npy", "000002.npy"])
        summary = json.loads((self.out / "run_summary.json").read_text(encoding="utf-8"))
        self.assertEqual(summary, {"status": "completed", "run_seed": 42, "payload_mode": "fixed",
                                   "payload_min": 0.1, "payload_max": 0.9, "total_discovered": 2,
                                   "total_generated": 2, "total_skipped": 0})

    def test_metadata_schema_sequential_ids_and_real_artifact_paths(self):
        generate(self.config)
        with (self.out / "metadata.csv").open(newline="", encoding="utf-8") as file:
            self.assertEqual(next(csv.reader(file)), METADATA_FIELDS)
        rows = self.rows()
        self.assertEqual([row["source_id"] for row in rows], ["000001", "000002"])
        self.assertEqual([row["source_file"] for row in rows], ["one.png", "two.PNG"])
        for row in rows:
            self.assertEqual(set(row), set(METADATA_FIELDS))
            for field in ("clean_file", "stego_file", "mask_file", "selected_locations_file"):
                self.assertTrue((self.out / row[field]).is_file(), row[field])
        self.assertEqual(read_config(self.out / "run_config.yaml"),
                         replace(self.config, input_dir=str(self.source), output_dir=str(self.out)))

    def test_saved_artifacts_match_masks_locations_and_rates(self):
        generate(replace(self.config, payload_mode="range", payload_min=0.1, payload_max=0.8))
        rows = self.rows()
        self.assertEqual(len(rows), 2)
        for row in rows:
            clean = load_rgb(self.out / row["clean_file"])
            stego = load_rgb(self.out / row["stego_file"])
            mask = np.load(self.out / row["mask_file"], allow_pickle=False)
            selected = np.load(self.out / row["selected_locations_file"], allow_pickle=False)
            self.assertEqual(mask.dtype, np.bool_)
            self.assertEqual(mask.shape, clean.shape)
            self.assertTrue(np.array_equal(mask, clean != stego))
            self.assertFalse(np.any((clean ^ stego) & np.uint8(254)))
            self.assertEqual(len(selected), payload_length(float(row["payload_rate"]), clean.size))
            self.assertEqual(np.unique(selected).size, len(selected))
            self.assertTrue(set(np.flatnonzero((clean != stego).reshape(-1)).tolist())
                            .issubset(set(int(index) for index in selected)))

    def test_fixed_rate_applies_to_every_sample(self):
        generate(replace(self.config, payload_rate=0.35))
        self.assertEqual({row["payload_rate"] for row in self.rows()}, {"0.35"})

    def test_configurable_range_rates_stay_inside_their_band(self):
        for low, high in ((0.1, 0.8), (0.2, 0.6), (0.05, 0.4), (0.4, 0.9)):
            out = self.root / f"range-{low}-{high}"
            result = generate(replace(self.config, output_dir=str(out), payload_mode="range",
                                      payload_min=low, payload_max=high))
            self.assertEqual(result["status"], "completed")
            rates = [float(row["payload_rate"]) for row in self.rows(out)]
            self.assertEqual(len(rates), 2)
            for rate in rates:
                self.assertGreaterEqual(rate, low)
                self.assertLessEqual(rate, high)

    def test_equal_range_bounds_behave_like_one_fixed_rate(self):
        generate(replace(self.config, payload_mode="range", payload_min=0.25, payload_max=0.25))
        self.assertEqual({row["payload_rate"] for row in self.rows()}, {"0.25"})

    def test_same_seed_and_order_reproduce_the_random_sequence(self):
        runs = {}
        for name, seed in (("run-a", 42), ("run-b", 42), ("run-c", 43)):
            generate(replace(self.config, output_dir=str(self.root / name), payload_mode="range",
                             payload_min=0.2, payload_max=0.7, run_seed=seed))
            runs[name] = self.rows(self.root / name)
        self.assertEqual([row["payload_rate"] for row in runs["run-a"]],
                         [row["payload_rate"] for row in runs["run-b"]])
        for first, second in zip(runs["run-a"], runs["run-b"]):
            self.assertEqual((self.root / "run-a" / first["stego_file"]).read_bytes(),
                             (self.root / "run-b" / second["stego_file"]).read_bytes())
        self.assertNotEqual([row["payload_rate"] for row in runs["run-a"]],
                            [row["payload_rate"] for row in runs["run-c"]])

    def test_subfolder_discovery_follows_the_recursive_setting(self):
        nested = self.source / "nested"
        nested.mkdir()
        Image.new("RGB", (6, 6), (3, 4, 5)).save(nested / "three.png")
        result = generate(replace(self.config, output_dir=str(self.root / "deep")))
        self.assertEqual(result["total_discovered"], 3)
        self.assertEqual([row["source_file"] for row in self.rows(self.root / "deep")],
                         ["nested/three.png", "one.png", "two.PNG"])
        result = generate(replace(self.config, output_dir=str(self.root / "flat"), recursive=False))
        self.assertEqual((result["total_discovered"], result["total_generated"]), (2, 2))

    def test_invalid_images_are_skipped_and_the_run_continues(self):
        (self.source / "bad.png").write_bytes(b"corrupt")
        (self.source / "photo.jpg").write_bytes(b"jpeg")
        Image.new("RGBA", (4, 4)).save(self.source / "alpha.png")
        result = generate(self.config)
        self.assertEqual(result["status"], "completed")
        self.assertEqual((result["total_discovered"], result["total_generated"], result["total_skipped"]), (5, 2, 3))
        self.assertEqual([row["source_id"] for row in self.rows()], ["000001", "000002"])

    def test_existing_output_is_never_overwritten(self):
        generate(self.config)
        original = (self.out / "metadata.csv").read_bytes()
        with self.assertRaises(ValueError):
            generate(self.config)
        self.assertEqual(original, (self.out / "metadata.csv").read_bytes())
        for output in (self.source, self.source / "nested", self.root):
            with self.assertRaises(ValueError):
                generate(replace(self.config, output_dir=str(output)))

    def test_cancellation_stops_cleanly(self):
        stopped = threading.Event()
        stopped.set()
        result = generate(replace(self.config, output_dir=str(self.root / "before")), cancel=stopped)
        self.assertEqual(result["status"], "cancelled")
        self.assertEqual(result["total_generated"], 0)

        stopping = threading.Event()

        def progress(event: dict) -> None:
            if event["kind"] == "progress" and event["current"] >= 1:
                stopping.set()

        result = generate(replace(self.config, output_dir=str(self.root / "during")), progress, stopping)
        self.assertEqual(result["status"], "cancelled")
        self.assertEqual((result["total_generated"], result["total_skipped"]), (1, 0))
        self.assertEqual(len(self.rows(self.root / "during")), 1)
        self.assertTrue((self.root / "during" / "clean" / "000001.png").is_file())
        self.assertFalse((self.root / "during" / "clean" / "000002.png").exists())

    def test_partially_written_samples_are_cleaned_up(self):
        with patch("lsb_batch.np.save", side_effect=OSError("disk full")):
            result = generate(self.config)
        self.assertEqual((result["total_generated"], result["total_skipped"]), (0, 2))
        self.assertEqual(result["status"], "failed")
        self.assertEqual(self.rows(), [])
        for folder in ("clean", "stego", "masks", "selected_locations"):
            self.assertEqual(list((self.out / folder).iterdir()), [], folder)
        self.assertIn("disk full", (self.out / "generation.log").read_text(encoding="utf-8"))

    def test_empty_input_folder_generates_nothing(self):
        empty = self.root / "empty"
        empty.mkdir()
        result = generate(replace(self.config, input_dir=str(empty), output_dir=str(self.root / "emptyout")))
        self.assertEqual(result["status"], "failed")
        self.assertEqual((result["total_discovered"], result["total_generated"]), (0, 0))

    def test_log_records_skips_and_summary_without_every_success(self):
        (self.source / "bad.png").write_bytes(b"corrupt")
        generate(self.config)
        log = (self.out / "generation.log").read_text(encoding="utf-8")
        self.assertIn("Run started", log)
        self.assertIn("Configuration:", log)
        self.assertIn("Skipped bad.png", log)
        self.assertIn("Run completed: 2 generated, 1 skipped of 3 discovered files", log)
        self.assertNotIn("000001", log)

    def test_config_round_trip_validation_and_cli(self):
        path = self.root / "config.yaml"
        write_config(path, self.config)
        self.assertEqual(read_config(path), self.config)
        for invalid in ("input_dir: input\noutput_dir: out\npayload_min: 0.8\npayload_max: 0.2\n",
                        "input_dir: input\noutput_dir: out\npayload_rate: 0\n",
                        "input_dir: input\noutput_dir: out\nsource_dataset: old\n",
                        "input_dir: input\noutput_dir: out\nrun_seed: -1\n"):
            path.write_text(invalid, encoding="utf-8")
            with self.assertRaises(ValueError):
                read_config(path)
        (self.root / "invalid.yaml").write_text("input_dir: input\noutput_dir: out\nrun_seed: -1\n", encoding="utf-8")
        result = subprocess.run([sys.executable, str(GENERATOR), "--config", str(self.root / "invalid.yaml")],
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 1)

        write_config(path, self.config)
        result = subprocess.run([sys.executable, str(GENERATOR), "--config", str(path)], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(cli_summary(result.stdout)["total_generated"], 2)
        result = subprocess.run([sys.executable, str(GENERATOR), "--input", str(self.source),
                                 "--output", str(self.root / "cli"), "--payload-rate", "0.3", "--run-seed", "9"],
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual({row["payload_rate"] for row in self.rows(self.root / "cli")}, {"0.3"})
        self.assertEqual(subprocess.run([sys.executable, str(GENERATOR), "--input", str(self.source),
                                         "--output", str(self.root / "cli")], capture_output=True).returncode, 1)
        self.assertEqual(subprocess.run([sys.executable, str(GENERATOR), "--verify", str(self.out)],
                                        capture_output=True).returncode, 2)
        version = subprocess.run([sys.executable, str(GENERATOR), "--version"], capture_output=True, text=True)
        self.assertEqual(version.stdout.strip(), VERSION)


if __name__ == "__main__":
    unittest.main()
