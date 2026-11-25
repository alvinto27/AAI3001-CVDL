"""Bounded probe contracts and real CPU Lightning save/restore; no GPU here."""
import copy
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
import numpy as np
import torch
from cvdl.lit import SteganalysisModule
from cvdl.gapcov_provenance import validate_sources,CELL_AUX
from scripts import pilot_run21_cells as p
from tests.test_gapcov_cell_training import ON,ROOT


def batch(pairs=1):
    from lsb_cells import changed_cell_targets
    clean=torch.zeros(pairs,3,256,256,dtype=torch.uint8);stego=clean.clone();stego[:,0,100,100]=1
    tags=torch.zeros(pairs,2,130,dtype=torch.long);tags[:,:,1]=1;tags[:,0,66:]=1
    counts,valid=changed_cell_targets(np.array([(100*256+100)*3]))
    for i in range(pairs):
        tags[i,0,0]=i*2;tags[i,1,0]=i*2+1
        tags[i,1,2:66]=torch.from_numpy(counts.flatten().astype(np.int64))
        tags[i,1,66:]=torch.from_numpy(valid.flatten().astype(np.int64))
    return clean,stego,tags


class PilotTests(unittest.TestCase):
    def test_settings_are_bounded_real_paired_weights_only_with_rocm_fix(self):
        with TemporaryDirectory() as tmp:
            s=p.settings(Path(tmp))
            self.assertTrue(s.pilot);self.assertTrue(s.paired_batches);self.assertTrue(s.rocm_conv1_fix)
            self.assertEqual((s.batch_size,s.epochs,s.lr,s.warmup_epochs),(16,15,3e-5,1))
            self.assertEqual((s.workers,s.eval_workers,s.prime_next_epoch,s.post_fit_eval),(4,0,False,'none'))
            self.assertEqual(s.init_checkpoint,str(p.SOURCE));self.assertEqual(s.init_checkpoint_sha256,p.SOURCE_SHA)
            self.assertEqual((s.checkpoint_validation,s.validation_recipe),('train_matched','60da4391043e8ce48ff2'))
            self.assertEqual((p.REPLAY_STEPS,p.FRESH_STEPS,p.MAX_SECONDS),(32,8,600))

    def test_actual_changed_pixels_align_with_tags_and_bad_pairs_refused(self):
        value=batch(8);self.assertEqual(p.aligned_batch(value)['images'],16)
        bad=list(copy.deepcopy(value));bad[2][0,1,2:66]=0
        with self.assertRaisesRegex(ValueError,'pixel changes'):p.aligned_batch(bad)
        bad=list(copy.deepcopy(value));bad[2][0,1,0]+=1
        with self.assertRaisesRegex(ValueError,'paired IDs'):p.aligned_batch(bad)

    def test_compiled_tolerance_formula_and_finite_gradient_contract(self):
        self.assertEqual(p.relative_error(torch.ones(2),torch.ones(2)),0.)
        with self.assertRaises(ValueError):p.relative_error(torch.ones(2)*1.01,torch.ones(2))
        module=SteganalysisModule(**ON);module.log=lambda *a,**k:None
        module.training_step(batch(),0)['loss'].backward();observed=p.gradients(module)
        self.assertGreater(observed['cell_head.weight'],0);self.assertIsNone(module.net.kv.conv.weight.grad)

    def test_frozen_comparison_normalizes_saved_device_and_cpu_teardown(self):
        class SavedDeviceCopy:
            # CPU-only stand-in for a saved GPU tensor; old torch.equal would
            # receive this object without normalization and fail immediately.
            def __init__(self,value):self.value=value;self.copied=False
            def detach(self):return self
            def cpu(self):self.copied=True;return self.value
        saved=SavedDeviceCopy(torch.tensor([1.,2.]))
        self.assertTrue(p.frozen_equal(saved,torch.tensor([1.,2.])))
        self.assertTrue(saved.copied)
        self.assertFalse(p.frozen_equal(saved,torch.tensor([1.,3.])))

    def test_real_lightning_checkpoint_resume_next_step_exact(self):
        provenance=validate_sources(ROOT,CELL_AUX)
        def rebuild():
            module=SteganalysisModule(**dict(ON,steps_per_epoch=512))
            module.hparams.update(source_provenance=provenance,pilot=True)
            return module
        with TemporaryDirectory() as tmp:
            result=p.continuation_check(rebuild(),batch(),Path(tmp),rebuild,device='cpu')
            self.assertEqual(result['resumed_global_step'],2)
            self.assertTrue(result['weights_optimizer_scheduler_bit_exact']);self.assertTrue(result['pilot_checkpoint'])


if __name__=='__main__':unittest.main()
