"""pack_blocks loader setting: default on, machine-owned, recorded in run_settings.json."""
import json
import tempfile
import unittest
from pathlib import Path
from cvdl.settings import load
from test_cvdl_settings import profile


class PackBlocksSettingTests(unittest.TestCase):
    def test_default_is_on_and_machine_owned_and_recorded(self):
        s = load(environ={}, _profile=profile())
        self.assertEqual(s.pack_blocks, 16)
        self.assertEqual(load(environ={'V3_PACK_BLOCKS': '1'}, _profile=profile()).pack_blocks, 1)
        self.assertEqual(load(environ={}, _profile=profile(), overrides=['pack_blocks=4']).pack_blocks, 4)
        with self.assertRaises(ValueError):
            load(environ={'V3_PACK_BLOCKS': '0'}, _profile=profile())
        with self.assertRaisesRegex(ValueError, 'machine-owned'):
            load(environ={}, _profile=profile(), pack_blocks=1)   # a loader setting, not an experiment value
        with tempfile.TemporaryDirectory() as folder:
            s.write(Path(folder))
            record = json.loads((Path(folder) / 'run_settings.json').read_text())
            self.assertEqual(record['values']['pack_blocks'], 16)
            self.assertEqual(record['sources']['pack_blocks'], 'default')


if __name__ == '__main__':
    unittest.main()
