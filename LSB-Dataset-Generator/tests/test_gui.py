"""Native Tk tests; require a desktop session (not a headless runner)."""
import csv
import gc
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image

from gui import App
from lsb_batch import Config


class GuiTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.source = self.root / "source"
        self.source.mkdir()
        Image.new("RGB", (30, 20), (50, 60, 70)).save(self.source / "one.png")
        self.app = App()
        self.app.update()
        self.app.withdraw()
        self.app.apply_config(Config(str(self.source), str(self.root / "out")))
        self.errors = []
        self.app.show_error = self.errors.append

    def tearDown(self):
        if self.app.busy:
            self.app.cancel()
            self.wait_done()
        self.app.destroy()
        # Release the Tcl interpreter here, in the main thread. If the test case
        # keeps the window alive, the interpreter is torn down at process exit and
        # the run aborts with "Tcl_AsyncDelete: async handler deleted by the wrong
        # thread" even though every test passed.
        self.app = None
        gc.collect()
        self.temp.cleanup()

    def wait_done(self):
        deadline = time.monotonic() + 15
        while self.app.busy and time.monotonic() < deadline:
            self.app.update()
            time.sleep(0.01)
        self.app.update()
        self.assertFalse(self.app.busy)

    def rows(self, name: str = "out"):
        with (self.root / name / "metadata.csv").open(newline="", encoding="utf-8") as file:
            return list(csv.DictReader(file))

    def test_generation_produces_the_documented_structure(self):
        self.app.generate_button.invoke()
        self.assertTrue(self.app.busy)
        self.assertTrue(self.app.generate_button.instate(["disabled"]))
        self.wait_done()
        self.assertEqual(self.app.last_result["status"], "completed")
        self.assertEqual(self.app.last_result["total_generated"], 1)
        self.assertEqual(self.app.status.get(), "Dataset ready")
        self.assertFalse(self.app.generate_button.instate(["disabled"]))
        self.assertFalse(self.app.open_button.instate(["disabled"]))
        self.assertEqual(float(self.app.progress["value"]), float(self.app.progress["maximum"]))
        for folder in ("clean", "stego", "masks"):
            self.assertTrue((self.root / "out" / folder).is_dir(), folder)
        self.assertFalse((self.root / "out" / "selected_locations").exists())
        self.assertTrue((self.root / "out" / "metadata.csv").is_file())
        self.assertEqual([row["source_id"] for row in self.rows()], ["000001"])

    def test_existing_output_reports_an_error_and_recovers(self):
        self.app.generate_button.invoke()
        self.wait_done()
        self.app.generate_button.invoke()
        self.wait_done()
        self.assertTrue(self.errors)
        self.assertFalse(self.app.generate_button.instate(["disabled"]))
        self.assertEqual(len(self.rows()), 1)

    def test_fixed_and_range_controls_accept_a_custom_range(self):
        self.assertFalse(self.app.rate_entry.instate(["disabled"]))
        self.assertTrue(self.app.min_entry.instate(["disabled"]))
        self.app.vars["payload_mode"].set("range")
        self.app.update_mode()
        self.assertTrue(self.app.rate_entry.instate(["disabled"]))
        self.assertFalse(self.app.min_entry.instate(["disabled"]))
        self.app.vars["payload_min"].set("0.1")
        self.app.vars["payload_max"].set("0.8")
        config = self.app.get_config()
        self.assertEqual((config.payload_min, config.payload_max), (0.1, 0.8))
        self.app.generate_button.invoke()
        self.wait_done()
        rates = [float(row["payload_rate"]) for row in self.rows()]
        self.assertEqual(len(rates), 1)
        self.assertTrue(0.1 <= rates[0] <= 0.8)

    def test_config_save_load_and_invalid_configs(self):
        config_file = self.root / "settings.yaml"
        self.app.vars["payload_mode"].set("range")
        self.app.vars["payload_max"].set("0.8")
        with patch("gui.filedialog.asksaveasfilename", return_value=str(config_file)):
            self.app.save_settings()
        self.app.vars["run_seed"].set("999")
        self.app.vars["payload_max"].set("0.2")
        with patch("gui.filedialog.askopenfilename", return_value=str(config_file)):
            self.app.load_settings()
        self.assertEqual(self.app.vars["run_seed"].get(), "42")
        self.assertEqual(self.app.vars["payload_max"].get(), "0.8")
        self.app.vars["payload_min"].set("0")
        self.app.generate_button.invoke()
        self.assertFalse(self.app.busy)
        self.assertTrue(self.errors)
        broken = self.root / "broken.yaml"
        broken.write_text("output_dir: out\npayload_mode: sideways\n", encoding="utf-8")
        with patch("gui.filedialog.askopenfilename", return_value=str(broken)):
            self.app.load_settings()
        self.assertEqual(len(self.errors), 2)

    def test_removed_controls_are_absent_and_required_controls_remain(self):
        texts = []

        def collect(widget):
            texts.append(str(widget.cget("text")) if "text" in widget.keys() else "")
            for child in widget.winfo_children():
                collect(child)

        collect(self.app)
        joined = " | ".join(texts)
        for removed in ("Verify", "Dataset name", "New seed", "Stop run", "Skip and continue"):
            self.assertNotIn(removed, joined)
        for required in ("Generate dataset", "Cancel", "Open output folder", "Load config", "Save config",
                         "Fixed rate", "Random range", "Include subfolders", "Run seed", "Min", "Max"):
            self.assertIn(required, joined)
        self.assertEqual(set(self.app.vars), {"input_dir", "output_dir", "payload_rate", "run_seed",
                                              "payload_mode", "payload_min", "payload_max", "recursive"})

    def test_cancellation_and_output_folder_suggestion(self):
        with patch("gui.filedialog.askdirectory", return_value=str(self.root)):
            self.app.pick_output()
        self.assertTrue(Path(self.app.vars["output_dir"].get()).name.startswith("lsb-run-"))
        self.app.generate_button.invoke()
        self.app.cancel_button.invoke()
        self.wait_done()
        self.assertIn(self.app.last_result["status"], ("cancelled", "completed"))


if __name__ == "__main__":
    unittest.main()
