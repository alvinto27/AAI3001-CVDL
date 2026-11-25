"""Benchmark gates, arithmetic, budget decisions and tiny CPU notebook flow."""
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

import numpy as np
import torch

from cvdl.gbras_bench import (AuditedLoader, CONFIGS, GATES, compare_numerics,
                              projection, run)
from cvdl.settings import load, machine_profile
from scripts.g4.preflight import top_values

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = 'notebooks/experiments/gbras-speed-bench.ipynb'


def recipe():
    return top_values(json.loads((ROOT/NOTEBOOK).read_text()))


def settings(g4=False):
    return load(environ={}, _profile=machine_profile('g4' if g4 else 'pc',device='cuda' if g4 else 'cpu'), **recipe())


class BenchLogicTests(unittest.TestCase):
    def setUp(self):
        self.reference = dict(initial_sha256='init', batch_sha256='batches',lr=[.0003]*200,
                              losses=[.5]*200,gradient_norms=[1.]*200,bn_finite=True)
        self.logits = np.array([[.2,.4],[.4,.1]],dtype=np.float32)
    def gate(self, candidate=None, mixed=None):
        return compare_numerics(self.reference,candidate or self.reference,self.logits,self.logits,
                                self.logits if mixed is None else mixed)

    def test_frozen_gates_and_configuration_order(self):
        self.assertEqual(GATES,dict(loss_window=50,absolute_loss_gap=.02,relative_loss_gap=.05,probability_p99_gap=.01))
        self.assertEqual([x[0] for x in CONFIGS],['fp32-nchw','bf16-nchw','bf16-channels-last','fp32-channels-last'])
        values=recipe()
        self.assertEqual(values['batch_size'],128)
        self.assertEqual(values['tiles_per_epoch'],180000)
        self.assertEqual(values['epochs'],30)
        self.assertEqual(values['validation_recipe'],'60da4391043e8ce48ff2')
        self.assertFalse(values['paired_batches'])
        self.assertNotIn('pilot',values)

    def test_numerics_identical_pass_and_symmetric_window_limits(self):
        result=self.gate()
        self.assertTrue(result['passed']);self.assertEqual(len(result['loss_windows']),4)
        for loss,passed in ((.524,True),(.526,False),(.474,False)):
            candidate=copy.deepcopy(self.reference);candidate['losses']=[loss]*200
            self.assertEqual(self.gate(candidate)['passed'],passed)
        self.reference['losses']=[.1]*200
        candidate=copy.deepcopy(self.reference);candidate['losses']=[.119]*200
        self.assertTrue(self.gate(candidate)['passed'])
        candidate['losses']=[.121]*200
        self.assertFalse(self.gate(candidate)['passed'])

    def test_probability_gate_and_identity_fail_closed(self):
        self.assertFalse(self.gate(mixed=self.logits+np.array([[0,.2],[0,.2]]))['passed'])
        for field,value in (('initial_sha256','different'),('batch_sha256','different'),('lr',[1.]*200),('bn_finite',False),('losses',[float('nan')]*200),('gradient_norms',[float('inf')]*200)):
            candidate=copy.deepcopy(self.reference);candidate[field]=value
            self.assertFalse(self.gate(candidate)['passed'],field)
        result=compare_numerics(self.reference,self.reference,self.logits,self.logits[:1],self.logits[:1])
        self.assertFalse(result['passed'])

    def test_projection_training_validation_compile_and_operational_overhead(self):
        p=projection(.1,100,10,compile_seconds=60)
        self.assertEqual(p['training_minutes'],5)
        self.assertEqual(p['validation_checkpoint_minutes'],5)
        self.assertEqual(p['total_minutes'],26)
        self.assertAlmostEqual(p['dollars'],26/60*1.63)
        self.assertTrue(projection(6000,1,3900,epochs=1)['fits_180_minutes'])
        self.assertFalse(projection(6000,1,3900.01,epochs=1)['fits_180_minutes'])
        for value in (-1,float('nan'),float('inf'),0):
            with self.assertRaises(ValueError): projection(value,100,10)

    def test_input_hash_ignores_unconsumed_prefetch_and_detects_pixels(self):
        batches=[(torch.full((2,3,4,4),i,dtype=torch.uint8),torch.arange(2),torch.arange(2)) for i in range(3)]
        a=AuditedLoader(batches,2);list(a)
        changed=list(batches);changed[2]=(torch.ones(2,3,4,4,dtype=torch.uint8),torch.arange(2),torch.arange(2))
        b=AuditedLoader(changed,2);list(b)
        self.assertEqual(a.count,2);self.assertEqual(a.digest.hexdigest(),b.digest.hexdigest())
        changed[0]=changed[2];c=AuditedLoader(changed,2);list(c)
        self.assertNotEqual(a.digest.hexdigest(),c.digest.hexdigest())

    def test_pc_timing_smoke_on_g4_and_budget_extension_refused(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ValueError): run(settings(),directory)
            with self.assertRaises(ValueError): run(settings(True),directory,smoke=True)
            with self.assertRaises(ValueError): run(settings(),directory,smoke=True,budget_seconds=2401)

    def test_expired_budget_writes_outputs_without_launch(self):
        with tempfile.TemporaryDirectory() as directory, patch('cvdl.gbras_bench.subprocess.Popen') as popen:
            summary=run(settings(),directory,smoke=True,budget_seconds=1,started_at=time.monotonic()-5)
            popen.assert_not_called()
            self.assertEqual(summary['state'],'budget_exhausted')
            self.assertIsNone(summary['projection'])
            self.assertTrue((Path(directory)/'benchmark-summary.json').is_file())
            self.assertFalse(json.loads((Path(directory)/'numerics.json').read_text())['passed'])

    def test_configuration_timeout_preserves_partial_and_skips_rest(self):
        class FakeProcess:
            pid=987654321
            def wait(self,timeout=None):
                self.timeout=timeout
                raise subprocess.TimeoutExpired('benchmark',timeout)
        process=FakeProcess()
        def start(command,**kwargs):
            request=json.loads(Path(command[-1]).read_text())
            (Path(request['output'])/'progress.json').write_text(json.dumps({'completed_updates':125}))
            return process
        with tempfile.TemporaryDirectory() as directory, patch('cvdl.gbras_bench.subprocess.Popen',side_effect=start) as popen, patch('cvdl.gbras_bench._stop_process'):
            summary=run(settings(),directory,smoke=True,budget_seconds=900)
            self.assertEqual(popen.call_count,1)
            self.assertEqual(process.timeout,480)
            self.assertEqual(summary['state'],'budget_exhausted')
            self.assertEqual(summary['trials'][0]['partial']['completed_updates'],125)
            self.assertTrue((Path(directory)/'numerics.json').is_file())

    def test_worker_failure_aborts_rest(self):
        class FakeProcess:
            pid=987654321
            def wait(self,timeout=None): return 1
        with tempfile.TemporaryDirectory() as directory, patch('cvdl.gbras_bench.subprocess.Popen',return_value=FakeProcess()) as popen, patch('cvdl.gbras_bench._stop_process'):
            summary=run(settings(),directory,smoke=True)
            self.assertEqual(popen.call_count,1)
            self.assertEqual(summary['state'],'failed')
            self.assertIsNone(summary['winner'])

    def test_notebook_cpu_end_to_end_fresh_children_and_no_hardware_timings(self):
        # Fresh ignored output, execute both real notebook code cells via normal runner.
        scratch=ROOT/'runs/.scratch'
        scratch.mkdir(parents=True,exist_ok=True)
        with tempfile.TemporaryDirectory(dir=scratch,prefix='bench-cpu-test-') as directory:
            env={**os.environ,'CVDL_BENCH_SMOKE':'1','PYTHONDONTWRITEBYTECODE':'1',
                 'CUDA_VISIBLE_DEVICES':'','HIP_VISIBLE_DEVICES':'','OMP_NUM_THREADS':'1'}
            command=[sys.executable,'-B','scripts/run_notebook.py','--notebook',NOTEBOOK,
                     '--env-file','/dev/null','--output',str(Path(directory)/'output')]
            completed=subprocess.run(command,cwd=ROOT,env=env,capture_output=True,text=True,timeout=120)
            self.assertEqual(completed.returncode,0,completed.stdout+'\n'+completed.stderr)
            folder=next((Path(directory)/'output/checkpoints').iterdir())
            summary=json.loads((folder/'benchmark-summary.json').read_text())
            self.assertEqual(summary['state'],'smoke-complete',summary)
            self.assertEqual(len(summary['trials']),4)
            self.assertEqual(len({row['worker_pid'] for row in [*summary['trials'],summary['overhead']]}),5)
            self.assertIsNone(summary['projection'])
            for row in summary['trials']:
                self.assertEqual(row['state'],'done')
                self.assertFalse(row['hardware_timing_valid'])
                self.assertIsNone(row['steps_per_second'])
                self.assertEqual(row['audited_batches'],4)
                self.assertEqual(len(row['bn']),13)
                self.assertTrue(all(bn['num_batches_tracked']==4 for bn in row['bn'].values()))
                self.assertEqual(row['dtype_layout']['srm_clamp_dtype'],'torch.float32')
                self.assertEqual(row['dtype_layout']['loss_dtype'],'torch.float32')
                checkpoint=torch.load(row['checkpoint'],map_location='cpu',weights_only=False)
                self.assertFalse(checkpoint['benchmark']['deployment_eligible'])
            reference=summary['trials'][0]
            for row in summary['trials'][1:]:
                self.assertEqual(row['initial_sha256'],reference['initial_sha256'])
                self.assertEqual(row['batch_sha256'],reference['batch_sha256'])
                self.assertEqual(row['lr'],reference['lr'])
            self.assertEqual(summary['overhead']['state'],'done')
            self.assertEqual(summary['overhead']['full_validation_passes'],1)
            self.assertEqual(summary['overhead']['validation_warmed_image_shapes'],[4,2])
            self.assertEqual(summary['overhead']['validation_probability_dtype'],'torch.float32')
            self.assertTrue(Path(summary['overhead']['checkpoint']).is_file())
            numerics=json.loads((folder/'numerics.json').read_text())
            self.assertIn('passed',numerics)
            self.assertTrue(numerics['comparisons']['fp32-nchw']['passed'])


if __name__=='__main__': unittest.main()
