"""Exact historical/current source scopes; no inference or large-panel replay."""
from pathlib import Path
from tempfile import TemporaryDirectory
import subprocess
import unittest
from unittest.mock import patch

from cvdl import cov_pool
from cvdl.gapcov_provenance import (validate_sources,historical_sources,ORIGINAL_REVISION,
    ORIGINAL_MODEL_SHA,MATH_SHA,ORIGINAL_NOTEBOOK_SHA,ORIGINAL_NOTEBOOK,CELL_NOTEBOOK,
    HISTORICAL,CELL_AUX)
from tests.frozen_sources import pinned_source_tree

ROOT=Path(__file__).resolve().parents[1]


class SourceProvenanceTests(unittest.TestCase):
    def test_original_approvals_and_pinned_bytes_unchanged(self):
        from scripts.training_decision import COV_APPROVAL,COV_NOTEBOOK_SHA
        from cvdl.gapcov_context import MODEL_SHA,MATH_SHA as CONTEXT_MATH_SHA
        lineage=historical_sources(ROOT)
        self.assertEqual(lineage['revision'],ORIGINAL_REVISION)
        self.assertEqual(lineage['spec'],cov_pool.SPEC)
        self.assertEqual(COV_APPROVAL['model_source_sha256'],ORIGINAL_MODEL_SHA)
        self.assertEqual(MODEL_SHA,ORIGINAL_MODEL_SHA)
        self.assertEqual(COV_APPROVAL['math_source_sha256'],MATH_SHA)
        self.assertEqual(CONTEXT_MATH_SHA,MATH_SHA)
        self.assertEqual(COV_NOTEBOOK_SHA,ORIGINAL_NOTEBOOK_SHA)
        with pinned_source_tree(ORIGINAL_REVISION) as root:
            record=validate_sources(root,HISTORICAL)
            self.assertEqual(record['current_files_sha256'],lineage['files_sha256'])
            # Approved old files cannot substitute for the executing v2 runtime.
            with self.assertRaisesRegex(ValueError,'executing checkout'):validate_sources(root,CELL_AUX)

    def test_live_v1_decision_and_context_refused_v2_explicit(self):
        from scripts.training_decision import cov_pool_identity
        from cvdl.gapcov_context import validate_original
        with self.assertRaisesRegex(ValueError,'current source drift'):validate_sources(ROOT,HISTORICAL)
        with self.assertRaisesRegex(ValueError,'current source drift'):cov_pool_identity([],[])
        with self.assertRaisesRegex(ValueError,'current source drift'):validate_original(ROOT,ROOT/'unused-panel')
        record=validate_sources(ROOT,CELL_AUX)
        self.assertEqual(record['version'],CELL_AUX)
        self.assertNotEqual(record['current_files_sha256']['cvdl/models.py'],ORIGINAL_MODEL_SHA)
        self.assertEqual(record['historical']['files_sha256']['cvdl/models.py'],ORIGINAL_MODEL_SHA)
        self.assertIn(CELL_NOTEBOOK,record['current_files_sha256'])

    def test_current_model_math_original_and_cell_notebook_drift_rejected(self):
        read=Path.read_bytes
        for relative in ('cvdl/models.py','cvdl/cov_pool.py',ORIGINAL_NOTEBOOK,CELL_NOTEBOOK):
            target=(ROOT/relative).resolve()
            def changed(path):
                content=read(path)
                return content+b'\n# unapproved drift\n' if path.resolve()==target else content
            with self.subTest(path=relative),patch.object(Path,'read_bytes',changed):
                with self.assertRaisesRegex(ValueError,'current source drift: '+relative):validate_sources(ROOT,CELL_AUX)
        with pinned_source_tree(ORIGINAL_REVISION) as root:
            (root/'cvdl/cov_pool.py').write_bytes(b'not approved')
            with self.assertRaisesRegex(ValueError,'current source drift'):validate_sources(root,HISTORICAL)

    def test_wrong_historical_revision_blob_and_missing_git_refused(self):
        for revision in ('9425d07','HEAD','cb0d68ae75f938fc1e3efd1e95f4c45bbb1e482f'):
            with self.subTest(revision=revision),self.assertRaisesRegex(ValueError,'historical revision mismatch'):
                validate_sources(ROOT,CELL_AUX,revision=revision)
        original=subprocess.check_output
        for relative in ('cvdl/models.py','cvdl/cov_pool.py',ORIGINAL_NOTEBOOK):
            def changed(args,**kwargs):
                blob=original(args,**kwargs)
                return blob+b'\n' if args[-1]==ORIGINAL_REVISION+':'+relative else blob
            with self.subTest(path=relative),patch('cvdl.gapcov_provenance.subprocess.check_output',changed):
                with self.assertRaisesRegex(ValueError,'historical source bytes mismatch'):validate_sources(ROOT,CELL_AUX)
        with patch('cvdl.gapcov_provenance.subprocess.check_output',side_effect=subprocess.CalledProcessError(128,['git'])):
            with self.assertRaisesRegex(ValueError,'historical source unavailable'):validate_sources(ROOT,CELL_AUX)

    def test_new_start_current_drift_refused_before_records(self):
        from tests.test_gapcov_cell_training import settings
        from cvdl.cov_training import check_start
        read=Path.read_bytes;target=ROOT/'cvdl/models.py'
        def changed(path):
            data=read(path)
            return data+b'\n# drift\n' if path.resolve()==target else data
        with TemporaryDirectory() as tmp:
            folder=Path(tmp)
            with patch.object(Path,'read_bytes',changed):
                with self.assertRaisesRegex(ValueError,'current source drift'):check_start(settings(),folder)
            self.assertFalse(list(folder.iterdir()))

    def test_live_spec_and_unknown_version_refused(self):
        with patch.object(cov_pool,'SPEC',dict(cov_pool.SPEC,iterations=6)):
            with self.assertRaisesRegex(ValueError,'live SPEC drift'):validate_sources(ROOT,CELL_AUX)
        with self.assertRaisesRegex(ValueError,'unknown gapcov source version'):validate_sources(ROOT,'future-v3')


if __name__=='__main__':unittest.main()
