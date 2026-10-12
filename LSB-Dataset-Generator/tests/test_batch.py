"""Tests for deterministic v6 bulk generation."""
from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image
from PIL.PngImagePlugin import PngInfo

from lsb_batch import Config, generate, read_config, replay, verify_output


class BatchTests(unittest.TestCase):
    """Exercise configuration, generation, resume, and verification."""

    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.source = self.root / "input"
        self.source.mkdir()
        (self.source / "nested").mkdir()
        Image.new("RGB", (32, 32), (10, 20, 30)).save(self.source / "cover.png")
        Image.new("RGB", (32, 32), (40, 50, 60)).save(self.source / "nested" / "other.png")

    def tearDown(self) -> None:
        self.temp.cleanup()

    def config(self, output: Path, **changes: object) -> Config:
        """Build a small deterministic test configuration."""
        values: dict[str, object] = {
            "seed": 7,
            "input": str(self.source),
            "output": str(output),
            "variants_per_cover": 2,
            "bit_mode": ["lowest", "plane"],
            "k": [1, 2],
            "method": ["replace", "match"],
            "placement": ["contiguous", "stride", "random"],
            "stride": [1, 3],
            "payload": ["random_bytes", "letters"],
            "size": {"payload_bytes": [1, 20]},
        }
        values.update(changes)
        return Config(**values)

    def files(self, output: Path) -> dict[str, bytes]:
        """Read deterministic output files for comparison."""
        names = ("clean", "stego", "masks", "metadata.csv", "done.jsonl")
        result: dict[str, bytes] = {}
        for name in names:
            path = output / name
            if path.is_dir():
                for file in sorted(path.rglob("*")):
                    if file.is_file():
                        result[str(file.relative_to(output))] = file.read_bytes()
            elif path.is_file():
                result[name] = path.read_bytes()
        return result

    def test_config_validation_rejects_unknown_and_invalid_values(self) -> None:
        """Reject unknown keys and malformed strict settings."""
        cases = (
            {"unknown": 1},
            {"size": {"bpp": [0.1, 0.2], "payload_bytes": [1, 2]}},
            {"size": {"bpp": [0, 0.2]}},
            {"k": 9},
            {"placement": "contiguous", "stride": 2},
            {"size": {"payload_bytes": [4, 1]}},
            {"png_compress_level": [1, 9]},
        )
        for index, changes in enumerate(cases):
            path = self.root / f"invalid-{index}.yaml"
            data = self.config(self.root / f"out-{index}").as_dict()
            data.update(changes)
            path.write_text(json.dumps(data), encoding="utf-8")
            with self.subTest(index=index), self.assertRaises(ValueError):
                read_config(path)

    def test_same_output_with_one_and_two_workers(self) -> None:
        """Per-image seeds make worker count irrelevant."""
        first = self.root / "one"
        second = self.root / "two"
        generate(self.config(first), workers=1)
        generate(self.config(second), workers=2)
        self.assertEqual(self.files(first), self.files(second))

    def test_resume_after_partial_run_matches_full_run(self) -> None:
        """A ledger-backed resume produces the same deterministic artifacts."""
        full = self.root / "full"
        partial = self.root / "partial"
        full_config = self.config(full)
        partial_config = self.config(partial)
        generate(full_config, workers=1)
        generate(partial_config, workers=1)
        done = partial / "done.jsonl"
        lines = done.read_text(encoding="utf-8").splitlines()
        done.write_text("\n".join(lines[:-1]) + "\n", encoding="utf-8")
        generate(partial_config, workers=2)
        self.assertEqual(self.files(full), self.files(partial))

    def test_resume_refuses_changed_config(self) -> None:
        """Resume refuses a run with a different configuration."""
        output = self.root / "run"
        config = self.config(output)
        generate(config, workers=1)
        with self.assertRaises(ValueError):
            generate(self.config(output, seed=8), workers=1)

    def test_non_png_and_rgba_inputs_are_skipped(self) -> None:
        """Unsupported extensions and alpha PNGs are logged and skipped."""
        (self.source / "notes.txt").write_text("not an image", encoding="utf-8")
        Image.new("RGBA", (8, 8), (1, 2, 3, 4)).save(self.source / "alpha.png")
        output = self.root / "run"
        result = generate(self.config(output, variants_per_cover=1), workers=1)
        self.assertEqual(result["completed"], 4)
        self.assertFalse(list((output / "stego").glob("*.png")) == [])
        log = (output / "run.log").read_text(encoding="utf-8")
        self.assertIn("unsupported extension", log)
        self.assertIn("only 8-bit RGB PNG is supported", log)

    def test_json_text_and_verify_pass_then_fail_after_payload_flip(self) -> None:
        """PNG metadata parses and verification catches a changed payload bit."""
        output = self.root / "run"
        config = self.config(
            output,
            variants_per_cover=1,
            bit_mode="lowest",
            k=1,
            method="replace",
            placement="contiguous",
            start=0,
            payload="random_bytes",
            stride=1,
            size={"payload_bytes": 4},
        )
        generate(config, workers=1)
        clean_path = output / "clean" / "000001.png"
        stego_path = output / "stego" / "000001_1.png"
        with Image.open(clean_path) as clean:
            clean_meta = json.loads(clean.text["stego"])
        with Image.open(stego_path) as stego:
            stego_meta = json.loads(stego.text["stego"])
        self.assertEqual(clean_meta["label"], "clean")
        self.assertEqual(set(clean_meta), {"label", "cover_id", "source_file", "size"})
        self.assertEqual(stego_meta["label"], "stego")
        self.assertEqual(stego_meta["generator_version"], "6.0.0")
        self.assertIn("embed_seed", stego_meta)
        with (output / "metadata.csv").open(newline="", encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
        self.assertNotIn("size", rows[0])
        self.assertNotIn("embed_seed", rows[0])
        self.assertTrue(all("[" not in value and "{" not in value for row in rows for value in row.values()))
        (output / "run_config.yaml").unlink()
        serial_report = verify_output(output, workers=1)
        parallel_report = verify_output(output, workers=2)
        self.assertEqual(serial_report, {"passed": 2, "failed": 0, "total": 2})
        self.assertEqual(serial_report, parallel_report)

        with Image.open(stego_path) as original:
            array = np.asarray(original).copy()
            raw_text = original.text["stego"]
        array.reshape(-1)[0] ^= np.uint8(1)
        info = PngInfo()
        info.add_text("stego", raw_text)
        Image.fromarray(array).save(stego_path, format="PNG", pnginfo=info)
        serial_report = verify_output(output, workers=1)
        serial_failure = [
            line for line in (output / "run.log").read_text(encoding="utf-8").splitlines()
            if line.startswith("FAIL ")
        ][-1]
        parallel_report = verify_output(output, workers=2)
        parallel_failure = [
            line for line in (output / "run.log").read_text(encoding="utf-8").splitlines()
            if line.startswith("FAIL ")
        ][-1]
        self.assertEqual(serial_report, {"passed": 1, "failed": 1, "total": 2})
        self.assertEqual(serial_report, parallel_report)
        self.assertEqual(serial_failure, parallel_failure)

        with Image.open(clean_path) as clean:
            clean_array = np.asarray(clean).copy()
        replayed, _, _ = replay(clean_array, stego_meta)
        info = PngInfo()
        info.add_text("stego", raw_text)
        array = replayed.copy()
        array.reshape(-1)[100] ^= np.uint8(1)
        Image.fromarray(array).save(stego_path, format="PNG", pnginfo=info)
        report = verify_output(output, workers=2)
        self.assertEqual(report, {"passed": 1, "failed": 1, "total": 2})

    def test_verify_workers_match_with_broken_pair(self) -> None:
        """Serial and parallel verification keep reports and failure details."""
        output = self.root / "verify-workers"
        config = self.config(
            output,
            variants_per_cover=1,
            bit_mode="lowest",
            k=1,
            method="replace",
            placement="contiguous",
            start=0,
            payload="random_bytes",
            stride=1,
            size={"payload_bytes": 4},
        )
        generate(config, workers=1)
        stego_path = output / "stego" / "000001_1.png"
        with Image.open(stego_path) as original:
            array = np.asarray(original).copy()
            raw_text = original.text["stego"]
        array.reshape(-1)[0] ^= np.uint8(1)
        info = PngInfo()
        info.add_text("stego", raw_text)
        Image.fromarray(array).save(stego_path, format="PNG", pnginfo=info)

        serial_report = verify_output(output, workers=1)
        serial_failure = [
            line for line in (output / "run.log").read_text(encoding="utf-8").splitlines()
            if line.startswith("FAIL ")
        ][-1]
        parallel_report = verify_output(output, workers=2)
        parallel_failure = [
            line for line in (output / "run.log").read_text(encoding="utf-8").splitlines()
            if line.startswith("FAIL ")
        ][-1]
        self.assertEqual(serial_report, {"passed": 1, "failed": 1, "total": 2})
        self.assertEqual(serial_report, parallel_report)
        self.assertEqual(serial_failure, parallel_failure)

    def test_unexpected_write_failure_is_retried_on_resume(self) -> None:
        """Unexpected worker failures are not written to the done ledger."""
        output = self.root / "retry"
        config = self.config(output, variants_per_cover=1, stride=1)
        with patch("lsb_batch._save_png_atomic", side_effect=OSError("disk full")):
            result = generate(config, workers=1)
        self.assertEqual(result["status"], "failed")
        done = output / "done.jsonl"
        self.assertFalse(done.exists())
        result = generate(config, workers=1)
        self.assertEqual(result["status"], "completed")
        self.assertEqual(len(done.read_text(encoding="utf-8").splitlines()), 2)

    def test_size_range_caps_to_capacity_and_skips_when_minimum_is_too_large(self) -> None:
        """Size ranges cap at capacity and skip variants whose minimum cannot fit."""
        small = self.root / "small"
        Image.new("RGB", (4, 4), (1, 2, 3)).save(self.source / "tiny.png")
        config = self.config(
            small,
            variants_per_cover=1,
            bit_mode="plane",
            k=1,
            placement="contiguous",
            stride=1,
            size={"payload_bytes": [1, 100]},
        )
        generate(config, workers=1)
        with (small / "metadata.csv").open(newline="", encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
        stego_rows = [row for row in rows if row["label"] == "stego" and row["source_file"] == "tiny.png"]
        self.assertEqual(len(stego_rows), 1)
        self.assertLessEqual(int(stego_rows[0]["payload_bytes"]), 6)

        skipped = self.root / "skipped"
        generate(
            self.config(
                skipped,
                variants_per_cover=1,
                input=str(self.source),
                bit_mode="plane",
                k=1,
                stride=1,
                size={"payload_bytes": [7, 100]},
            ),
            workers=1,
        )
        self.assertNotIn("000003_1.png", {path.name for path in (skipped / "stego").iterdir()})
        self.assertIn("minimum payload_bytes does not fit", (skipped / "run.log").read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
