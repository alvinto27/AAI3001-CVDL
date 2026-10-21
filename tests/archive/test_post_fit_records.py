"""Post-fit evaluation owns its diagnostics; training artifacts remain immutable."""
from contextlib import ExitStack, redirect_stdout
import hashlib
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import pandas as pd

from cvdl.archive import post_fit
from lsb_notebook import NotebookData
from scripts.archive import evaluate_legacy_512 as evaluate_run


class PostFitRecordTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.folder=Path(self.temp.name)
        self.settings=SimpleNamespace(device='cpu',data='otf',root=self.folder,
            extra_tests='',eval_workers=0,quick=False,manifest_root='manifests',
            labels_path=None,diag_workers=0,data_root=self.folder,norm='none',eval_profile='test',
            batch_size=2,workers=0,prefetch=None,prefetch_images=128,
            values={},provenance={},profile='test')
        self.loaders=SimpleNamespace(train_eval='train',val='val',test='test')
        self.otf=NotebookData.__new__(NotebookData)
        self.otf.record={'training_identity':'fixed','diagnostic_pair_ids':{'old':['training-id']}}
        self.otf.fixed_train=object();self.otf.corpus=object()
        self.selected=[SimpleNamespace(frame=pd.DataFrame({'cover_id':[f'{split}-a',f'{split}-b']}))
                       for split in ('train','val')]
        self.checkpoint=self.folder/'best-fixture.ckpt'
        # Odd whitespace and unrelated content catch accidental JSON reserialization.
        for name,content in {'best-fixture.ckpt':b'checkpoint bytes',
            'last.ckpt':b'last checkpoint bytes',
            'data_run_settings.json':b'{ "training_identity": "fixed", "diagnostic_pair_ids": {"old":["training-id"]} }\n',
            'model_run_settings.json':b'{"model":"fixture"}\n',
            'run_settings.json':b'{"values":{"seed":17}}\n',
            'validation_pairs.csv':b'cover_id,source\noriginal,raise\n',
            'metrics.csv':b'epoch,loss\n0,0.3\n'}.items():
            (self.folder/name).write_bytes(content)

    def hashes(self):
        return {p.name:hashlib.sha256(p.read_bytes()).hexdigest()
                for p in self.folder.iterdir() if p.is_file()}

    def evaluate(self,full=False):
        model=SimpleNamespace(net=object(),to=lambda device:None)
        predictions=pd.DataFrame({'label':[0,1],'prob_stego':[.1,.9]})
        with ExitStack() as stack:
            for target,kwargs in [
                ('cvdl.archive.post_fit.SteganalysisModule.load_from_checkpoint',{'return_value':model}),
                ('cvdl.archive.post_fit.evaluation.predict_pairs',{'return_value':predictions}),
                ('cvdl.archive.post_fit.evaluation.per_setting_table',{'return_value':pd.DataFrame()}),
                ('cvdl.archive.post_fit.evaluation.extra_panels',{'return_value':{}}),
                ('cvdl.archive.post_fit.evaluation.breakdown',{'return_value':None}),
                ('cvdl.archive.post_fit.FixedPairs',{'return_value':object()}),
                ('cvdl.archive.post_fit.diagnostics.diagnostic_pairs',{'side_effect':self.selected}),
                ('cvdl.archive.post_fit.diagnostics.batch_dependence',{'return_value':{'loss':.25}}),
            ]:stack.enter_context(patch(target,**kwargs))
            stack.enter_context(redirect_stdout(io.StringIO()))
            if full:
                self.otf.validation_pair_csv=(self.folder/'validation_pairs.csv').read_text()
                self.later_stage_record=None
                def read_diagnostics(*args,**kwargs):
                    self.later_stage_record=json.loads((self.folder/'evaluation_data_run_settings.json').read_text())
                for target,kwargs in [
                    ('scripts.archive.evaluate_legacy_512.settings_for_evaluation',{'return_value':(self.settings,{},None)}),
                    ('cvdl.lit.select_best_checkpoint',{'return_value':str(self.checkpoint)}),
                    ('cvdl.runtime.configure',{}),
                    ('cvdl.runtime.create_data',{'return_value':(self.otf,None)}),
                    ('cvdl.data.build_otf_loaders',{'return_value':self.loaders}),
                    ('cvdl.archive.panels.write_validation',{'side_effect':read_diagnostics}),
                    ('cvdl.archive.panels.write_standard_validation',{}),
                    ('cvdl.archive.panels.tables',{}),
                ]:stack.enter_context(patch(target,**kwargs))
                return evaluate_run.evaluate(self.folder)
            return post_fit.run(self.settings,self.otf,None,self.loaders,self.checkpoint,self.folder)

    def test_full_post_fit_preserves_every_training_file_and_saves_readable_diagnostics(self):
        before=self.hashes()
        result=self.evaluate(full=True)
        after=self.hashes()
        self.assertEqual(before,{name:after[name] for name in before})
        # A later stage reads diagnostics from the evaluation-owned file, not training inputs.
        saved=json.loads((self.folder/'evaluation_data_run_settings.json').read_text())
        for split,selected in zip(('train','val'),self.selected):
            self.assertEqual(saved['diagnostic_pair_ids'][split],selected.frame.cover_id.tolist())
        self.assertEqual(saved['training_identity'],'fixed')
        self.assertEqual(self.later_stage_record,saved)
        self.assertEqual(json.loads((self.folder/'evaluation_results.json').read_text())['diagnostics'],result['diagnostics'])
        self.assertEqual(json.loads((self.folder/'data_run_settings.json').read_text())['diagnostic_pair_ids'],{'old':['training-id']})

    def test_missing_training_record_is_not_fabricated(self):
        (self.folder/'data_run_settings.json').unlink()
        before=self.hashes();self.evaluate();after=self.hashes()
        self.assertFalse((self.folder/'data_run_settings.json').exists())
        self.assertEqual(before,{name:after[name] for name in before})
        self.assertIn('val',json.loads((self.folder/'evaluation_data_run_settings.json').read_text())['diagnostic_pair_ids'])

    def test_later_evaluation_updates_only_its_own_diagnostic_record(self):
        original=self.hashes();self.evaluate()
        self.selected=[SimpleNamespace(frame=pd.DataFrame({'cover_id':[f'new-{split}']})) for split in ('train','val')]
        self.evaluate();after=self.hashes()
        self.assertEqual(original,{name:after[name] for name in original})
        saved=json.loads((self.folder/'evaluation_data_run_settings.json').read_text())
        self.assertEqual(saved['diagnostic_pair_ids']['train'],['new-train'])
        self.assertEqual(saved['diagnostic_pair_ids']['val'],['new-val'])


if __name__=='__main__':unittest.main()
