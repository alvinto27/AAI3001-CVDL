"""Same-process CPU/notebook proofs; saved files are read-only optional fixtures."""
import ast
from contextlib import redirect_stdout
from copy import deepcopy
import io
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

from cvdl import evaluation as ev, diagnostics as diag
from cvdl.metrics import global_summary, auc_numpy, probability_error, summary_rows
from lsb_notebook import complete_bpp_bins, fixed_subset
from lsb_torch import PreloadedPairs

ROOT = Path(__file__).resolve().parents[1]


def code(name, index):
    notebook = json.loads((ROOT/'notebooks'/name).read_text())
    return ''.join(notebook['cells'][index]['source'])


def functions(source, namespace):
    tree = ast.parse(source)
    tree.body = [node for node in tree.body if isinstance(node, ast.FunctionDef)]
    exec(compile(tree, '<notebook functions>', 'exec'), namespace)


class TinyPairs(Dataset):
    def __init__(self):
        self.frame = pd.DataFrame([dict(cover_id=f'{i:06}', pair_id=f'p{i}',
            source_photo=f'photo{i//2}', source='raise' if i%2 else 'div2k', camera_model='A',
            k=i%4+1, bit_mode='lowest' if i%2 else 'plane', method='replace',
            placement='stride' if i%3 else 'random', stride=2, payload_kind='random',
            bpp_actual=.1 if i%2 else .3, requested_bpp=.1 if i%2 else .3,
            effective_bpp=.1 if i%2 else .3, clipped=False) for i in range(40)])
        generator=torch.Generator().manual_seed(77)
        self.images=torch.randint(0, 16, (40,2,3,8,8), dtype=torch.uint8, generator=generator)

    def __len__(self): return len(self.frame)
    def __getitem__(self, i): return self.images[i,0], self.images[i,1], i


class SmallModel(nn.Module):
    def __init__(self, norm='none'):
        super().__init__()
        normalization={'none':nn.Identity, 'batch':lambda:nn.BatchNorm2d(4),
                       'group':lambda:nn.GroupNorm(2,4)}[norm]()
        self.net=nn.Sequential(nn.Conv2d(3,4,1), normalization, nn.ReLU(),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(4,2))


class EvaluationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads=torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls): torch.set_num_threads(cls.threads)

    def test_predict_csv_and_setting_tables_match_notebook(self):
        torch.manual_seed(3)
        model=SmallModel()
        loader=DataLoader(fixed_subset(TinyPairs(),32,0),batch_size=8,shuffle=False)
        for mode in ('otf','png'):
            namespace=dict(np=np,pd=pd,torch=torch,SteganalysisModule=SmallModel,
                           DataLoader=DataLoader,DEVICE='cpu',V3_DATA=mode,
                           complete_bpp_bins=complete_bpp_bins)
            functions(code('05_xunet_v4_nonorm.ipynb',21),namespace)
            expected=namespace['predict_pairs'](model,loader)
            actual=ev.predict_pairs(model,loader,device='cpu',data_mode=mode)
            pd.testing.assert_frame_equal(actual,expected,check_exact=True)
            self.assertEqual(ev.prediction_accuracy(actual),namespace['prediction_accuracy'](expected))
            namespace['predictions']=expected
            with redirect_stdout(io.StringIO()):
                exec(code('05_xunet_v4_nonorm.ipynb',25),namespace)
            table=ev.per_setting_table(actual,data_mode=mode)
            pd.testing.assert_frame_equal(table,namespace['per_setting'],check_exact=True)
            with tempfile.TemporaryDirectory() as tmp:
                directory=Path(tmp)
                namespace.update(checkpoint_dir=directory)
                with redirect_stdout(io.StringIO()):exec(code('05_xunet_v4_nonorm.ipynb',27),namespace)
                path=ev.save_predictions(actual,directory/'package.csv')
                self.assertEqual(path.read_bytes(),namespace['predictions_path'].read_bytes())

    def test_saved_real_prediction_tables_summaries_and_heatmaps(self):
        paths=sorted((ROOT/'runs/g4-v4-r2/checkpoints').glob('*/test_predictions.csv'))
        if not paths: self.skipTest('saved g4-v4-r2 predictions unavailable')
        self.compare_breakdown(paths[0])
        # Notebook-05 tables from the same saved real scores, exact CSV bytes.
        predictions=pd.read_csv(paths[0],float_precision='round_trip')
        namespace=dict(predictions=predictions, np=np,pd=pd,V3_DATA='otf',
            complete_bpp_bins=complete_bpp_bins, probability_error=probability_error)
        with redirect_stdout(io.StringIO()):exec(code('05_xunet_v4_nonorm.ipynb',25),namespace)
        actual=ev.per_setting_table(predictions,data_mode='otf')
        self.assertEqual(actual.to_csv(index=False),namespace['per_setting'].to_csv(index=False))

    def compare_breakdown(self, original, labels_path=None):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); reference=root/'reference'; package=root/'package'
            reference.mkdir();package.mkdir()
            a=reference/'test_predictions.csv';b=package/'test_predictions.csv'
            a.write_bytes(Path(original).read_bytes());b.write_bytes(a.read_bytes())
            namespace=dict(np=np,pd=pd,plt=plt,Path=Path,os=os)
            # Execute the original notebook with its standalone PRED_PATH interface.
            with patch.dict(os.environ,{'PRED_PATH':str(a)}), patch.object(Path,'cwd',return_value=ROOT), \
                 patch.object(plt,'show'), redirect_stdout(io.StringIO()):
                setup=code('02_eval_breakdown.ipynb',2)
                if labels_path is not None:
                    setup=setup.replace('LABELS_PATH = DATA_ROOT / "labels.csv"',f'LABELS_PATH = Path({str(labels_path)!r})')
                exec(setup,namespace)
                for index in (4,6,8):exec(code('02_eval_breakdown.ipynb',index),namespace)
                result,table,cells,heatmaps=ev.breakdown(b,labels_path=labels_path)
            pd.testing.assert_frame_equal(table,namespace['per_setting'],check_exact=True)
            pd.testing.assert_frame_equal(cells,namespace['cell_table'],check_exact=True)
            self.assertEqual((package/'test_predictions_per_setting.csv').read_bytes(),
                             (reference/'test_predictions_per_setting.csv').read_bytes())
            for key, old in [('threshold','GLOBAL_THRESHOLD'),('P_E','GLOBAL_P_E'),
                             ('false_alarm_rate','GLOBAL_P_FA'),('miss_rate','GLOBAL_P_MD'),
                             ('detection_rate','GLOBAL_DETECTION_RATE')]:
                self.assertEqual(result[key],namespace[old])
            self.assertEqual(result['AUC'],namespace['auc_numpy'](namespace['labels_array'],namespace['scores_array']))
            self.assertEqual(len(heatmaps),4)
            for image in heatmaps:
                self.assertEqual(image.read_bytes(),(reference/image.name).read_bytes())

    def test_png_breakdown_still_uses_label_metadata(self):
        data=TinyPairs()
        predictions=ev.predict_pairs(SmallModel(),DataLoader(data,batch_size=16),device='cpu',data_mode='png')
        labels=predictions.copy()
        labels['stride']=2
        for column in ('start_random','payload_bytes','units_changed','size_value'):labels[column]=1
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'pred.csv';label_path=Path(tmp)/'labels.csv'
            predictions.to_csv(path,index=False);labels.to_csv(label_path,index=False)
            self.compare_breakdown(path,label_path)

    def test_checkpoint_scorer_outputs_and_quick_cap(self):
        torch.manual_seed(9)
        model=SmallModel()
        dataset=TinyPairs()
        namespace=dict(np=np,pd=pd,torch=torch,SteganalysisModule=SmallModel,
                       DataLoader=DataLoader,DEVICE='cpu',V3_DATA='otf')
        functions(code('05_xunet_v4_nonorm.ipynb',21),namespace)
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);checkpoint=root/'model.ckpt'
            torch.save(dict(hyper_parameters={'norm':'none'},state_dict=model.state_dict()),checkpoint)
            for quick in (False,True):
                output=root/str(quick)
                with patch.object(ev,'model_from_hparams',return_value=deepcopy(model.net)), \
                     patch.object(ev,'Corpus',return_value=None),patch.object(ev,'FixedPairs',return_value=dataset), \
                     patch.object(ev,'OrderedPairLoader',side_effect=lambda ds,**kw:DataLoader(ds,batch_size=16)), \
                     patch.dict(os.environ,{'QUICK':'true'}),redirect_stdout(io.StringIO()):
                    info,table=ev.evaluate(checkpoint,ROOT/'configs/test-bpp1.yaml',root=ROOT,device='cpu',
                                           output=output,workers=0,quick=quick)
                selected=fixed_subset(dataset,16,1) if quick else dataset
                expected=namespace['predict_pairs'](model,DataLoader(selected,batch_size=16))
                self.assertEqual((output/'test_predictions.csv').read_text(),expected.to_csv(index=False))
                expected['bpp_bin']=pd.cut(expected['requested_bpp'],bins=[0.,.05,.15,.3,.5,.75,1.],
                    labels=['<=0.05','0.05-0.15','0.15-0.30','0.30-0.50','0.50-0.75','0.75-1.0'],
                    include_lowest=True,right=True)
                expected_table=pd.DataFrame(summary_rows(expected))
                pd.testing.assert_frame_equal(table,expected_table,check_exact=True)
                self.assertEqual((output/'summary.csv').read_text(),expected_table.to_csv(index=False))
                self.assertEqual(info['pairs'],16 if quick else 40)
                self.assertEqual(info['quick_pair_cap'],16 if quick else None)
                self.assertEqual(info['overall'],summary_rows(expected)[0])
                clipped=(selected.frame.groupby('placement',sort=True)['clipped'].agg(['sum','count'])
                    .reset_index().rename(columns={'sum':'clipped_pairs','count':'pairs'}))
                self.assertEqual((output/'clipped_by_placement.csv').read_text(),clipped.to_csv(index=False))
                self.assertEqual(json.loads((output/'run.json').read_text()),info)

    def test_global_threshold_ties_and_single_class_validation(self):
        frame=pd.DataFrame({'label':[0,1,0,1], 'prob_stego':[.2,.2,.8,.8]})
        summary=global_summary(frame)
        self.assertEqual(summary['threshold'],np.inf)
        self.assertEqual(summary['P_E'],.5)
        with self.assertRaisesRegex(ValueError,'both labels'):
            global_summary(frame.loc[frame.label==0])

    def test_extra_panels_forward_explicit_quick_device_and_sources(self):
        with patch.object(ev,'evaluate',return_value=({},pd.DataFrame())) as score:
            result=ev.extra_panels('model.ckpt',root=ROOT,output=ROOT/'runs/fake',
                panels='test-bpp1,source-panels',workers=0,device='cpu',quick=True,
                data_mode='otf',manifest_root='manifests',source_panels=['alaska-test_seen','lsdir-val'])
        self.assertEqual(list(result),['test-bpp1','alaska-test_seen','lsdir-val'])
        self.assertEqual(score.call_count,3)
        for call in score.call_args_list:
            self.assertEqual(call.kwargs['device'],'cpu')
            self.assertIs(call.kwargs['quick'],True)
        self.assertEqual(score.call_args_list[1].kwargs['source'],'alaska')
        self.assertEqual(score.call_args_list[1].kwargs['split'],'test_seen')
        with patch.dict(os.environ,{'QUICK':'true'}):
            dataset=TinyPairs()
            self.assertIs(ev.panel_dataset(dataset),dataset)
            self.assertEqual(len(ev.panel_dataset(dataset,quick=True)),16)


class DiagnosticsTests(unittest.TestCase):
    def test_batch_numbers_match_original_cell_for_all_normalizations(self):
        threads=torch.get_num_threads();torch.set_num_threads(1)
        try:
            for norm in ('none','group','batch'):
                torch.manual_seed(4)
                model=SmallModel(norm); base=TinyPairs();captured={}
                record={}
                def preload(subset,workers): return [subset[i] for i in range(len(subset))]
                data=SimpleNamespace(fixed_train=base,corpus=None,record=record,save_record=lambda *a:None)
                namespace=dict(np=np,pd=pd,torch=torch,F=F,DataLoader=DataLoader,
                    QUICK=True,V3_DATA='otf',EVAL_WORKERS=0,DEVICE='cpu',NORM=norm,
                    os=os,PreloadedPairs=PreloadedPairs,preload_pairs=preload,
                    fixed_subset=fixed_subset,FixedPairs=lambda *a:base,EVAL_PROFILE='overlap_full_mix',
                    otf_data=data,best_model=model,checkpoint_dir=ROOT/'runs',captured=captured)
                source=code('05_xunet_v4_nonorm.ipynb',31)
                # Observe full-precision intermediate numbers, not rounded printed strings.
                source=source.replace('    print(f"CHECK {split_name} a',
                    '    captured[split_name] = {"a": dict(accuracy=correct/count, loss=total_loss/count)}\n    print(f"CHECK {split_name} a')
                source=source.replace('        print(f"CHECK {split_name} {mode}',
                    '        captured[split_name][mode] = dict(accuracy=correct/count, mean_prob_stego=prob_sum/count)\n        if NORM in {"none", "group"}: captured[split_name][mode]["max_mode_difference"] = max_mode_difference\n        print(f"CHECK {split_name} {mode}')
                rng=torch.get_rng_state()
                with patch.dict(os.environ,{'V3_DIAG_WORKERS':'0'}),redirect_stdout(io.StringIO()):exec(source,namespace)
                torch.set_rng_state(rng)
                original=deepcopy(model.state_dict())
                for split in ('train','val'):
                    with patch.object(diag,'preload_pairs',side_effect=preload):
                        dataset=diag.diagnostic_pairs(base,quick=True,data_mode='otf',workers=0)
                    self.assertEqual(dataset.frame.cover_id.tolist(),record['diagnostic_pair_ids'][split])
                    actual=diag.batch_dependence(model.net,dataset,device='cpu',norm=norm,split_name=split,quick=True)
                    self.assertEqual(actual,captured[split])
                for key,value in original.items():self.assertTrue(torch.equal(value,model.state_dict()[key]))
        finally:torch.set_num_threads(threads)


if __name__=='__main__':unittest.main()
