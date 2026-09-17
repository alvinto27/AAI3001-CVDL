"""Native Tk tests; require a desktop session (not a headless runner)."""
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
        self.temp.cleanup()

    def wait_done(self):
        deadline = time.monotonic() + 15
        while self.app.busy and time.monotonic() < deadline:
            self.app.update()
            time.sleep(0.01)
        self.app.update()
        self.assertFalse(self.app.busy)

    def test_generate_verify_and_error_recovery(self):
        self.app.generate_button.invoke()
        self.assertTrue(self.app.busy)
        self.assertTrue(self.app.generate_button.instate(["disabled"]))
        self.wait_done()
        self.assertEqual(self.app.last_result["total_generated"], 1)
        self.assertEqual(self.app.status.get(), "Dataset ready")
        self.assertFalse(self.app.open_button.instate(["disabled"]))
        self.app.start_verification(self.root / "out")
        self.wait_done()
        self.assertEqual(self.app.last_result["verified"], 1)
        self.app.generate_button.invoke()
        self.wait_done()
        self.assertTrue(self.errors)
        self.assertFalse(self.app.generate_button.instate(["disabled"]))

    def test_config_mode_and_validation(self):
        self.app.vars["payload_mode"].set("range")
        self.app.update_mode()
        self.assertTrue(self.app.rate_entry.instate(["disabled"]))
        self.assertFalse(self.app.min_entry.instate(["disabled"]))
        config_file = self.root / "settings.yaml"
        with patch("gui.filedialog.asksaveasfilename", return_value=str(config_file)):
            self.app.save_settings()
        self.app.vars["run_seed"].set("999")
        with patch("gui.filedialog.askopenfilename", return_value=str(config_file)):
            self.app.load_settings()
        self.assertEqual(self.app.vars["run_seed"].get(), "42")
        self.app.vars["payload_min"].set("0")
        self.app.generate_button.invoke()
        self.assertFalse(self.app.busy)
        self.assertTrue(self.errors)

    def test_cancellation_and_folder_controls(self):
        with patch("gui.filedialog.askdirectory", return_value=str(self.root)):
            self.app.pick_output()
        self.assertTrue(Path(self.app.vars["output_dir"].get()).name.startswith("lsb-run-"))
        self.app.generate_button.invoke()
        self.app.cancel_button.invoke()
        self.wait_done()
        self.assertIn(self.app.last_result["status"], ("cancelled", "completed"))


if __name__ == "__main__":
    unittest.main()
