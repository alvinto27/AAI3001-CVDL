"""Thin-notebook wiring and context provenance; no historical output capture."""
import ast
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from cvdl.settings import load, runner_context
from scripts.run_notebook import settings_file
from test_cvdl_settings import profile

ROOT=Path(__file__).resolve().parents[1]


class ThinNotebookTests(unittest.TestCase):
    def test_short_notebook_driver(self):
        after=json.loads((ROOT/'notebooks/05_xunet_v4_nonorm.ipynb').read_text())
        new=[c for c in after['cells'] if c['cell_type']=='code']
        self.assertEqual(len(new),10)
        self.assertLess(sum(len(''.join(c['source'])) for c in new),5000)
        self.assertTrue(all(c['outputs']==[] and c['execution_count'] is None for c in new))
        source='\n'.join(''.join(c['source']) for c in new)
        from scripts.g4.preflight import top_values
        self.assertEqual(top_values(after),{'post_fit_eval':'none'})
        self.assertNotIn('os.environ',source)
        self.assertFalse(any(isinstance(n,(ast.FunctionDef,ast.ClassDef)) for n in ast.walk(ast.parse(source))))
        self.assertTrue(after['metadata']['cvdl_package'])
        assigned={n.id for n in ast.walk(ast.parse(source)) if isinstance(n,ast.Name) and isinstance(n.ctx,ast.Store)}
        self.assertTrue({'RUN_NAME','QUICK','V3_DATA','EPOCHS','BATCH_SIZE','COMPILE_MODE','settings',
                         'otf_data','train_loader','val_loader','test_loader'} <= assigned)

    def test_context_native_overrides_and_reset(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'job.env';path.write_text('QUICK=true\n')
            env=settings_file(path,['seed=9'],allow_native=True)
            self.assertNotIn('seed',env)
        with runner_context(env,['seed=9','lr=.001']), patch('cvdl.settings.automatic_profile',return_value=profile()):
            settings=load()
            self.assertEqual(settings.seed,9)
            self.assertEqual(settings.provenance['seed'],'override')
            self.assertEqual(settings.provenance['lr'],'override')
            self.assertEqual(load(overrides=['seed=10']).seed,10)
            with self.assertRaisesRegex(ValueError,'conflicts.*--set'):
                with runner_context({'V3_LR':'.002'},[]):load(lr=.001)
        settings=load(environ={},_profile=profile())
        self.assertEqual(settings.seed,0)
        self.assertEqual(settings.provenance['seed'],'default')


if __name__=='__main__':unittest.main()
