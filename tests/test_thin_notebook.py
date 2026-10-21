"""Thin-notebook wiring, context provenance and strict transition allow-list."""
import ast
from contextlib import redirect_stdout
from copy import deepcopy
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd

from cvdl.settings import load, runner_context
from scripts import refactor_golden as golden
from scripts.run_notebook import settings_file
from test_cvdl_settings import profile
from test_cvdl_evaluation import TinyPairs
from cvdl.metrics import summary_rows
from lsb_notebook import fixed_subset

ROOT=Path(__file__).resolve().parents[1]


class ThinNotebookTests(unittest.TestCase):
    def test_frozen_reference_and_short_new_notebook(self):
        before=json.loads((ROOT/'notebooks/archive/05_baseline_kv_xunet_v4_pre-cvdl.ipynb').read_text())
        after=json.loads(golden.NOTEBOOK.read_text())
        old=[c for c in before['cells'] if c['cell_type']=='code']
        new=[c for c in after['cells'] if c['cell_type']=='code']
        self.assertEqual((len(old),sum(len(''.join(c['source'])) for c in old)),(16,81097))
        self.assertEqual(len(new),10)
        self.assertLess(sum(len(''.join(c['source'])) for c in new),5000)
        self.assertTrue(all(c['outputs']==[] and c['execution_count'] is None for c in old+new))
        source='\n'.join(''.join(c['source']) for c in new)
        from scripts.g4.preflight import top_values
        self.assertEqual(top_values(after),{'post_fit_eval':'none'})
        self.assertNotIn('os.environ',source)
        self.assertFalse(any(isinstance(n,(ast.FunctionDef,ast.ClassDef)) for n in ast.walk(ast.parse(source))))
        self.assertTrue(after['metadata']['cvdl_package'])
        self.assertEqual(golden.NOTEBOOK.name,'05_xunet_v4_nonorm.ipynb')
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


class GoldenAllowListTests(unittest.TestCase):
    def test_only_named_new_file_is_allowed(self):
        before={'folder':{'files':['last.ckpt']}}
        after={'folder':{'files':['last.ckpt','run_settings.json']}}
        with redirect_stdout(io.StringIO()) as output:
            self.assertEqual(golden.compare_part('cpu',before,after),[])
        self.assertIn('[allowed run-settings-file]',output.getvalue())
        after['folder']['files'].append('unexpected.txt')
        with redirect_stdout(io.StringIO()):
            self.assertTrue(golden.compare_part('cpu',before,after))
        after['folder']['files'].remove('unexpected.txt')
        after['losses']=[123]
        with redirect_stdout(io.StringIO()):
            self.assertTrue(golden.compare_part('cpu',before,after))

    def test_only_completed_expected_cell_layout_is_allowed(self):
        before={'cells_run':16,'cell_states':['done']*16}
        after={'cells_run':10,'cell_states':['done']*10}
        with redirect_stdout(io.StringIO()) as output:
            self.assertEqual(golden.compare_part('gpu',before,after),[])
        self.assertIn('[allowed thin-cell-layout]',output.getvalue())
        after['cell_states'][-1]='failed'
        self.assertTrue(golden.compare_part('gpu',before,after))
        before={'folder':{'validation_pairs.csv':{'sha256':'old'}}}
        after={'folder':{'validation_pairs.csv':{'sha256':'new'}}}
        self.assertTrue(golden.compare_part('gpu',before,after))

    def test_cap_requires_exact_subset_hash_schema_and_every_group_count(self):
        full=TinyPairs();selected=fixed_subset(full,16,seed=1)
        frame=pd.concat([selected.frame.assign(label=0,prob_stego=0.),selected.frame.assign(label=1,prob_stego=0.)])
        frame['bpp_bin']=pd.cut(frame.requested_bpp,bins=[0.,.05,.15,.3,.5,.75,1.],
            labels=['<=0.05','0.05-0.15','0.15-0.30','0.30-0.50','0.50-0.75','0.75-1.0'],include_lowest=True,right=True)
        table=pd.DataFrame(summary_rows(frame))
        columns=list(table.columns)
        entry=dict(pairs=16,settings_sha256=golden.frame_sha(selected.frame,golden.SETTING_FIELDS),
            summary=dict(columns=columns,kept_columns=columns,rows=table.astype(object).values.tolist()))
        name='checkpoints/example/extra-test-bpp1'
        before={'extra_panels':{name:dict(pairs=40,settings_sha256='full',summary={})}}
        after={'extra_panels':{name:entry}}
        with patch('lsb_torch.Corpus',return_value=None),patch('lsb_torch.FixedPairs',return_value=full):
            with redirect_stdout(io.StringIO()) as output:
                self.assertEqual(golden.compare_part('gpu',before,after),[])
            self.assertIn('[allowed quick-extra-panel-cap]',output.getvalue())
            broken=deepcopy(after);broken['extra_panels'][name]['summary']['rows'][0][3]=15
            self.assertTrue(golden.compare_part('gpu',before,broken))
            broken=deepcopy(after);broken['extra_panels'][name]['settings_sha256']='wrong'
            self.assertTrue(golden.compare_part('gpu',before,broken))


if __name__=='__main__':unittest.main()
