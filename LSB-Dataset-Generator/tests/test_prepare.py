"""Tests for deterministic photo preparation."""
from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image
import lsb_batch
from lsb_prepare import PrepareConfig, _raw_camera_info, prepare


class PrepareTests(unittest.TestCase):
    """Exercise decoding, tiling, resume, and downstream compatibility."""

    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.source = self.root / "source"
        self.source.mkdir()
        pixels = np.arange(100 * 90 * 3, dtype=np.uint8).reshape(100, 90, 3)
        Image.fromarray(pixels).save(self.source / "photo.png")

    def tearDown(self) -> None:
        self.temp.cleanup()

    def config(self, output: Path, **changes: object) -> PrepareConfig:
        """Build a small preparation configuration."""
        values: dict[str, object] = {
            "input": str(self.source),
            "output": str(output),
            "tile": 32,
            "seed": 17,
            "png_compress_level": 1,
        }
        values.update(changes)
        return PrepareConfig(**values)

    def manifest(self, output: Path) -> list[dict[str, str]]:
        """Read the tile manifest."""
        with (output / "manifest.csv").open(newline="", encoding="utf-8") as stream:
            return list(csv.DictReader(stream))

    def test_tile_count_positions_and_partial_edges(self) -> None:
        """Drop partial edges and record row, column, and pixel positions."""
        output = self.root / "tiles"
        result = prepare(self.config(output), workers=1)
        self.assertEqual(result, {"status": "completed", "sources": 1, "tiles": 6})
        rows = self.manifest(output)
        self.assertEqual(len(rows), 6)
        self.assertEqual(
            [(int(row["tile_row"]), int(row["tile_col"])) for row in rows],
            [(row, col) for row in range(3) for col in range(2)],
        )
        self.assertEqual(
            [(int(row["pixel_x"]), int(row["pixel_y"])) for row in rows[:2]],
            [(0, 0), (32, 0)],
        )
        self.assertTrue(all(row["source_file"] == "photo.png" for row in rows))
        self.assertTrue(all(row["source_format"] == "png" for row in rows))
        self.assertEqual({path.name for path in output.glob("*.png")}, {
            f"photo_r{row}_c{col}.png" for row in range(3) for col in range(2)
        })

    def test_max_tiles_is_deterministic_across_worker_counts(self) -> None:
        """Per-image tile choices do not depend on worker count."""
        first = self.root / "one"
        second = self.root / "two"
        config = self.config(first, tile=16, max_tiles_per_image=5)
        prepare(config, workers=1)
        prepare(self.config(second, tile=16, max_tiles_per_image=5), workers=2)
        self.assertEqual((first / "manifest.csv").read_bytes(), (second / "manifest.csv").read_bytes())
        for first_tile in first.glob("*.png"):
            self.assertEqual(first_tile.read_bytes(), (second / first_tile.name).read_bytes())

    def test_exif_orientation_is_applied(self) -> None:
        """EXIF orientation changes pixels before tiling."""
        array = np.array(
            [
                [[255, 0, 0], [0, 255, 0], [0, 0, 255]],
                [[10, 20, 30], [40, 50, 60], [70, 80, 90]],
            ],
            dtype=np.uint8,
        )
        path = self.source / "rotated.jpg"
        image = Image.fromarray(array)
        exif = image.getexif()
        exif[274] = 6
        image.save(path, exif=exif)
        output = self.root / "rotated"
        prepare(self.config(output, tile=2), workers=1)
        rows = self.manifest(output)
        row = next(row for row in rows if row["source_file"] == "rotated.jpg")
        self.assertEqual((int(row["source_width"]), int(row["source_height"])), (2, 3))
        with Image.open(output / row["tile_file"]) as tile:
            self.assertEqual(tile.size, (2, 2))

    def test_rejects_grayscale_transparent_and_16bit(self) -> None:
        """Reject source modes that cannot safely become RGB."""
        Image.new("L", (40, 40), 12).save(self.source / "gray.png")
        Image.new("RGBA", (40, 40), (1, 2, 3, 127)).save(self.source / "alpha.png")
        sixteen = np.arange(40 * 40, dtype=np.uint16).reshape(40, 40)
        Image.fromarray(sixteen, mode="I;16").save(self.source / "deep.png")
        (self.source / "notes.txt").write_text("not an image", encoding="utf-8")
        output = self.root / "rejected"
        result = prepare(self.config(output, tile=32), workers=1)
        self.assertEqual(result["sources"], 5)
        self.assertEqual(len(self.manifest(output)), 6)
        log = (output / "run.log").read_text(encoding="utf-8")
        self.assertIn("grayscale", log)
        self.assertIn("transparency", log)
        self.assertIn("16-bit", log)
        self.assertIn("unsupported", log)

    def test_resume_skips_completed_sources(self) -> None:
        """A complete source ledger entry avoids decoding it again."""
        output = self.root / "resume"
        prepare(self.config(output), workers=1)
        original = (output / "photo_r0_c0.png").read_bytes()
        Image.new("RGB", (100, 90), (250, 1, 2)).save(self.source / "photo.png")
        result = prepare(self.config(output), workers=2)
        self.assertEqual(result["tiles"], 6)
        self.assertEqual((output / "photo_r0_c0.png").read_bytes(), original)

    def test_manifest_and_generator_compatibility(self) -> None:
        """Prepared tiles are accepted by the bulk generator."""
        output = self.root / "tiles"
        prepare(self.config(output), workers=1)
        rows = self.manifest(output)
        self.assertEqual(set(rows[0]), {
            "tile_file", "source_file", "source_format", "camera_make", "camera_model",
            "source_width", "source_height", "tile_row", "tile_col", "pixel_x", "pixel_y",
        })
        generated = self.root / "generated"
        config = lsb_batch.Config(
            seed=3,
            input=str(output),
            output=str(generated),
            bit_mode="lowest",
            k=1,
            method="replace",
            placement="contiguous",
            stride=1,
            payload="random_bytes",
            start="random",
            size={"payload_bytes": 1},
        )
        result = lsb_batch.generate(config, workers=1)
        self.assertEqual(result["status"], "completed")
        self.assertEqual(len(list((generated / "stego").glob("*.png"))), 6)

    def test_raw_camera_tags_are_read_without_raw_decode(self) -> None:
        """Read Make and Model from RAW TIFF IFD0 tags."""
        path = self.source / "camera.NEF"
        Image.new("RGB", (8, 8), (1, 2, 3)).save(
            path,
            format="TIFF",
            tiffinfo={271: "TEST MAKE", 272: "TEST MODEL"},
        )
        self.assertEqual(_raw_camera_info(path), ("TEST MAKE", "TEST MODEL"))

    def test_unexpected_tile_failure_is_retried(self) -> None:
        """Unexpected tile errors are not permanently recorded."""
        output = self.root / "retry"
        with patch("lsb_prepare._save_tile_atomic", side_effect=OSError("disk full")):
            result = prepare(self.config(output), workers=1)
        self.assertEqual(result["status"], "failed")
        self.assertFalse((output / "done.jsonl").exists())
        self.assertEqual(prepare(self.config(output), workers=1)["tiles"], 6)


if __name__ == "__main__":
    unittest.main()
