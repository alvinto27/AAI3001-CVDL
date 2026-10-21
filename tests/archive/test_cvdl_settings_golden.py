"""Archived captured settings comparison; manual check only."""
import json
from pathlib import Path
import sys
import unittest
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/'tests')]
from cvdl.settings import load
from test_cvdl_settings import profile
from scripts.archive.refactor_golden import jsonable, settings_dict

class GoldenSettingsTests(unittest.TestCase):
    def test_all_golden_env_settings_and_names(self):
        golden = json.loads((ROOT/'tests/archive/golden/refactor-golden.json').read_text())['settings']
        checked = []
        for name, expected in golden.items():
            if name == 'standard_test':
                continue
            env = expected['env_settings']
            device = expected.get('globals',{}).get('DEVICE','cuda')
            actual = load(environ=env, _profile=profile(device))
            with self.subTest(env=name):
                self.assertEqual(actual.run_name, expected['run_name'])
                if 'globals' in expected:
                    # Absolute checkout paths are profile-owned; the capture root was /work.
                    def local_paths(value):
                        if isinstance(value,dict): return {k:local_paths(v) for k,v in value.items()}
                        if isinstance(value,list): return [local_paths(v) for v in value]
                        return str(ROOT)+value[5:] if isinstance(value,str) and value.startswith('/work/') else value
                    self.assertEqual(actual.legacy_globals(), local_paths(expected['globals']))
                    self.assertEqual(jsonable(settings_dict(actual.to_run_settings())), local_paths(expected['run_settings']))
                else:
                    self.assertEqual(jsonable(actual.speed_configuration()), expected['effective_pairs_steps_warmup_workers'])
                for key in env:
                    from cvdl.settings import canonical
                    self.assertEqual(actual.provenance[canonical(key)],'env')
            checked.append(name)
        self.assertEqual(sorted(checked), ['cpu-det','newdata-1','quick-gpu','speed-1','wide-1'])

if __name__=="__main__":unittest.main()
