"""Archived thin-notebook golden allowances; manual checks only."""
from contextlib import redirect_stdout
from copy import deepcopy
import io
from pathlib import Path
import sys
import unittest
from unittest.mock import patch
import pandas as pd
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/'tests')]
from scripts.archive import refactor_golden as golden
from test_cvdl_evaluation import TinyPairs
from cvdl.metrics import summary_rows
from lsb_notebook import fixed_subset

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

if __name__=="__main__":unittest.main()
