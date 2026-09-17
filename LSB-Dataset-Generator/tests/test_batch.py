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

from lsb_batch import Config, generate, read_config, verify_run, write_config


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

    def rows(self, path=None):
        with ((path or self.root / "out") / "metadata.csv").open(newline="", encoding="utf-8") as file:
            return list(csv.DictReader(file))

    def test_roundtrip_and_relocation(self):
        result = generate(self.config)
        self.assertEqual(result["total_generated"], 2)
        self.assertEqual(verify_run(self.root / "out")["verified"], 2)
        (self.root / "out").rename(self.root / "moved")
        self.assertEqual(verify_run(self.root / "moved")["verified"], 2)
        self.assertNotIn("split", self.rows(self.root / "moved")[0])

    def test_duplicate_rejection_and_continue(self):
        (self.source / "copy.png").write_bytes((self.source / "one.png").read_bytes())
        (self.source / "bad.png").write_bytes(b"corrupt")
        (self.source / "photo.jpg").write_bytes(b"jpeg")
        result = generate(self.config)
        self.assertEqual(result["total_generated"], 2)
        self.assertEqual(result["total_rejected"], 3)
        self.assertEqual(result["total_duplicates"], 1)
        self.assertEqual(result["status"], "completed_with_rejections")

    def test_output_safety(self):
        generate(self.config)
        original = (self.root / "out" / "metadata.csv").read_bytes()
        with self.assertRaises(ValueError):
            generate(self.config)
        self.assertEqual(original, (self.root / "out" / "metadata.csv").read_bytes())
        for output in (self.source, self.source / "nested", self.root):
            with self.assertRaises(ValueError):
                generate(replace(self.config, output_dir=str(output)))

    def test_range_order_independence(self):
        config = replace(self.config, payload_mode="range", payload_min=0.2, payload_max=0.8)
        generate(config)
        old = {row["source_id"]: row for row in self.rows()}
        (self.source / "one.png").rename(self.source / "zzz.png")
        generate(replace(config, output_dir=str(self.root / "out2")))
        for row in self.rows(self.root / "out2"):
            before = old[row["source_id"]]
            self.assertEqual(before["random_seed"], row["random_seed"])
            self.assertEqual(before["payload_rate"], row["payload_rate"])
            self.assertTrue(0.2 <= float(row["payload_rate"]) <= 0.8)
            self.assertEqual((self.root / "out" / before["stego_file"]).read_bytes(), (self.root / "out2" / row["stego_file"]).read_bytes())

    def test_tampered_mask_metadata_and_path(self):
        generate(self.config)
        rows = self.rows()
        path = self.root / "out" / rows[0]["mask_file"]
        mask = np.load(path)
        mask[0, 0, 0] ^= True
        np.save(path, mask)
        with self.assertRaises(ValueError):
            verify_run(self.root / "out")
        mask[0, 0, 0] ^= True
        np.save(path, mask)
        rows[0]["clean_file"] = "../../outside.png"
        with (self.root / "out" / "metadata.csv").open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
        with self.assertRaises(ValueError):
            verify_run(self.root / "out")

    def test_stop_cancel_and_empty(self):
        (self.source / "0bad.png").write_bytes(b"bad")
        result = generate(replace(self.config, error_policy="stop"))
        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["total_unprocessed"], 2)
        event = threading.Event()
        event.set()
        result = generate(replace(self.config, output_dir=str(self.root / "cancel")), cancel=event)
        self.assertEqual(result["status"], "cancelled")
        empty = self.root / "empty"
        empty.mkdir()
        result = generate(replace(self.config, input_dir=str(empty), output_dir=str(self.root / "emptyout")))
        self.assertEqual(result["status"], "failed")

    def test_failed_sample_cleanup(self):
        with patch("lsb_batch.verify_record", side_effect=ValueError("injected failure")):
            result = generate(self.config)
        self.assertEqual(result["total_failed"], 2)
        self.assertEqual(result["status"], "failed")
        self.assertFalse(list((self.root / "out" / "stego").iterdir()))
        self.assertEqual(self.rows(), [])

    def test_config_and_cli(self):
        path = self.root / "config.yaml"
        write_config(path, self.config)
        self.assertEqual(read_config(path), self.config)
        command = [sys.executable, str(Path(__file__).resolve().parents[1] / "generate.py")]
        result = subprocess.run(command + ["--config", str(path)], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(subprocess.run(command + ["--verify", self.config.output_dir], capture_output=True).returncode, 0)
        self.assertNotEqual(subprocess.run(command + ["--config", str(path)], capture_output=True).returncode, 0)

    def test_zero_change_and_equal_range(self):
        result = generate(replace(self.config, payload_mode="range", payload_min=0.000001, payload_max=0.000001))
        self.assertEqual(result["total_zero_change"], 2)
        self.assertEqual(result["total_selected_locations"], 0)
        self.assertEqual(verify_run(self.root / "out")["verified"], 2)


if __name__ == "__main__":
    unittest.main()
