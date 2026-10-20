"""tanh saturation script: hooks read the real tanh inputs without changing the network."""
import unittest
import numpy as np
import torch
from cvdl.models import XuNetWide
from cvdl.saturation import measure, fixed_pairs, Stats


class TanhSaturationTests(unittest.TestCase):
    def test_hooks_match_direct_computation_and_leave_network_unchanged(self):
        torch.manual_seed(0)
        net=XuNetWide(clip_t=3.0,norm='none',width=1).eval()
        images=torch.randint(0,256,(6,3,32,32),dtype=torch.uint8);labels=torch.tensor([0,0,0,1,1,1])
        before=net(images.float()).detach().clone()
        table=measure(net,[(images[:3],labels[:3]),(images[3:],labels[3:])])
        self.assertTrue(torch.equal(net(images.float()),before))  # hooks removed, output unchanged
        self.assertEqual(len(net.bn1._forward_hooks),0)
        x=net.kv(images.float());x=torch.clamp(x,-3,3);t1=net.bn1(torch.abs(net.conv1(x)))
        for label,rows in ((0,slice(0,3)),(1,slice(3,6))):
            a=t1[rows].abs()
            r=table['tanh1'][str(label)]
            self.assertEqual(r['values'],a.numel())
            self.assertAlmostEqual(r['mean_abs'],float(a.detach().mean()),places=4)
            self.assertAlmostEqual(r['share_abs_gt2'],float((a>2).float().mean()),places=6)
            self.assertAlmostEqual(r['share_abs_gt3'],float((a>3).float().mean()),places=6)
            self.assertAlmostEqual(r['share_tanh_out_gt_0.99'],float((torch.tanh(a)>0.99).float().mean()),places=6)
            self.assertAlmostEqual(r['p50_abs'],float(a.detach().flatten().median()),delta=.01)
        self.assertEqual(set(table),{'tanh1','tanh2','final_logit'})
        d=(before[:,1]-before[:,0]).abs()
        self.assertAlmostEqual(table['final_logit']['1']['max_abs'],float(d[3:].max()),places=4)

    def test_pair_choice_is_fixed_and_stats_overflow(self):
        self.assertEqual(fixed_pairs(10,0,100),fixed_pairs(10,0,100));self.assertNotEqual(fixed_pairs(10,0,100),fixed_pairs(10,1,100))
        self.assertEqual(len(set(fixed_pairs(10,0,100))),10);self.assertEqual(len(fixed_pairs(500,0,100)),100)
        s=Stats();s.add(torch.tensor([0.5,1.5,100.,-2000.]))
        self.assertEqual(s.max,2000.);self.assertEqual(s.result(False)['share_abs_gt3'],.5);self.assertEqual(s.quantile(.99),float('inf'))

if __name__=='__main__':unittest.main()


class StageSelectionHelper:
    """Shared mocked evaluator helper; no discovered tests."""
    def run_main(self,argv,folder,existing=()):
        import sys, tempfile
        from pathlib import Path
        from unittest.mock import patch
        import scripts.evaluate_run as er
        for name in existing:
            path=Path(folder)/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_text('x')
        calls=[]
        with patch.object(er,'checkpoint_folder',return_value=Path(folder)), \
             patch.object(er,'evaluate',side_effect=lambda *a,**k:calls.append(('main',k))), \
             patch.object(er,'optional_stages',side_effect=lambda f,stages,**k:calls.append(('optional',sorted(stages)))), \
             patch.object(er,'calibrate',side_effect=lambda *a,**k:calls.append(('temperature',))), \
             patch.object(sys,'argv',['evaluate_run.py',str(folder),'--test-suite','legacy-512',*argv]):
            er.main()
        return calls


class StageSelectionTests(StageSelectionHelper, unittest.TestCase):
    """evaluate_run.py stage selection and no-overwrite rule (all work mocked)."""
    def test_only_and_skip_main_never_run_main(self):
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            calls=self.run_main(['--only','padding,temperature,tanh'],d)
            self.assertEqual([c[0] for c in calls],['optional','temperature']);self.assertEqual(calls[0][1],['padding','tanh'])
            calls=self.run_main(['--tanh-saturation','--skip-main'],d)
            self.assertEqual(calls,[('optional',['tanh'])])
            calls=self.run_main(['--padding-check','--skip-main'],d)
            self.assertEqual(calls,[('optional',['padding'])])
            calls=self.run_main(['--temperature'],d)  # legacy: temperature alone has no main
            self.assertEqual(calls,[('temperature',)])

    def test_legacy_padding_check_runs_main_with_inline_padding(self):
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            calls=self.run_main(['--padding-check','--tanh-saturation'],d)
            self.assertEqual([c[0] for c in calls],['main']);self.assertTrue(calls[0][1]['padding_check'])
            self.assertEqual(calls[0][1]['tanh'],dict(pairs=2000,train_batches=0))
            self.assertEqual([c[0] for c in self.run_main([],d)],['main'])

    def test_refuses_existing_output_unless_overwrite(self):
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            for argv,files in ((['--only','tanh'],['tanh_saturation.json']),
                               (['--only','padding'],['padding-check/summary.json']),
                               (['--only','temperature'],['calibration.json']),
                               (['--padding-check'],['test_predictions.csv'])):
                with self.assertRaisesRegex(ValueError,'refusing to overwrite'):self.run_main(argv,d,existing=files)
                calls=self.run_main([*argv,'--overwrite'],d,existing=files);self.assertTrue(calls)
        # A stage that is not selected does not care about other stages' files.
        with tempfile.TemporaryDirectory() as d:
            self.assertTrue(self.run_main(['--only','tanh'],d,existing=['test_predictions.csv','calibration.json']))

    def test_unknown_stage_and_empty_selection(self):
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(ValueError):self.run_main(['--only','bogus'],d)
            with self.assertRaisesRegex(ValueError,'nothing'):self.run_main(['--skip-main'],d)


class TablesStageTests(StageSelectionHelper, unittest.TestCase):
    def test_tables_stage_and_main_refuse_existing_tables(self):
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            # default label comes from the folder name; build the same name here
            import scripts.evaluate_run as er
            from pathlib import Path
            label=er.default_label(Path(d))
            files=[f'pc-comparison/{label}.json']
            with self.assertRaisesRegex(ValueError,'refusing to overwrite'):self.run_main(['--only','tables'],d,existing=files)
            with self.assertRaisesRegex(ValueError,'refusing to overwrite'):self.run_main([],d,existing=files)
            from unittest.mock import patch
            with patch('cvdl.panels.tables') as tables:
                self.run_main(['--only','tables','--overwrite'],d,existing=files)
            self.assertEqual(tables.call_count,1)
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaisesRegex(ValueError,'refusing to overwrite'):
                self.run_main([],d,existing=['validation_predictions.csv'])
            with self.assertRaisesRegex(ValueError,'refusing to overwrite'):
                self.run_main([],d,existing=['validation_calibration.json'])

    def test_default_label(self):
        from pathlib import Path
        import scripts.evaluate_run as er
        label=er.default_label(Path('/x/tiling-overlap-256-20261017T022212Z-9778/checkpoints/tiling-overlap-256-20261017T022212Z-9778'))
        self.assertEqual(label,'tilingoverlap256')
