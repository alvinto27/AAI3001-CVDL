import unittest
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch
from cvdl.archive.cell_pooling import pool_logits, pooled_probabilities, pooled_logits64


class CellPoolingTests(unittest.TestCase):
    def test_mean_max_and_top_four_values(self):
        clean = torch.tensor([100., 0, 0, 0, 0, 0])
        margin = torch.tensor([-10., 1, 2, 3, 4, 5])
        cells = torch.stack([clean, clean+margin]).reshape(1, 2, 2, 3)
        torch.testing.assert_close(pool_logits(cells, 'mean'), torch.tensor([[100/6, 105/6]]))
        torch.testing.assert_close(pool_logits(cells, 'max'), torch.tensor([[0., 5.]]))
        torch.testing.assert_close(pool_logits(cells, 'top4'), torch.tensor([[0., 3.5]]))
        # Independently maximizing classes would select the wrong clean cell.
        self.assertNotEqual(pool_logits(cells, 'max')[0, 0], cells[:, 0].max())

    def test_shared_cell_offsets_do_not_change_probabilities(self):
        cells = torch.randn(3, 2, 8, 8, generator=torch.Generator().manual_seed(81))
        shifted = cells + torch.arange(64).reshape(1, 1, 8, 8)
        for mode, probability in pooled_probabilities(cells).items():
            torch.testing.assert_close(probability, pooled_probabilities(shifted)[mode], atol=3e-6, rtol=3e-6)

    def test_linear_cell_mean_equals_original_global_mean_forward(self):
        features = torch.randn(3, 7, 8, 8, generator=torch.Generator().manual_seed(12), dtype=torch.float64)
        classifier = torch.nn.Linear(7, 2).double()
        cells = torch.nn.functional.conv2d(features, classifier.weight[:, :, None, None], classifier.bias)
        expected = classifier(features.mean((-2, -1)))
        torch.testing.assert_close(pool_logits(cells), expected, atol=1e-14, rtol=1e-14)
        torch.testing.assert_close(pooled_probabilities(cells)['mean'], expected.softmax(1)[:, 1])

    def test_pool_logits_not_probabilities(self):
        cells = torch.tensor([[[[0., 0., 0., 0.]], [[-10., 0., 0., 0.]]]])
        actual = pooled_probabilities(cells)['mean']
        averaged_probabilities = cells.softmax(1)[:, 1].mean((-2, -1))
        self.assertGreater(float((actual-averaged_probabilities).abs()), .1)

    def test_float64_pooling_avoids_reduction_and_subtraction_loss(self):
        cells = torch.tensor([[[[1e8, 1, 1, 1]], [[1e8, 3, 3, 3]]]], dtype=torch.float32)
        values = pooled_logits64(cells)
        self.assertEqual(values['mean'].dtype, torch.float64)
        self.assertEqual(float(values['mean'][0,1]-values['mean'][0,0]), 1.5)
        self.assertEqual(float(values['top4'][0,1]-values['top4'][0,0]), 1.5)
        self.assertEqual(float(values['max'][0,1]-values['max'][0,0]), 2)

    def test_margins_preserve_order_when_softmax_is_exactly_one(self):
        cells = torch.zeros(2, 2, 8, 8); cells[0,1] = 100; cells[1,1] = 200
        for mode, logits in pooled_logits64(cells).items():
            margins = logits[:,1]-logits[:,0]
            torch.testing.assert_close(margins, torch.tensor([100.,200.], dtype=torch.float64))
            self.assertTrue(torch.all(pooled_probabilities(cells)[mode] == 1))

    def test_exact_margin_tie_counts(self):
        import pandas as pd
        from scripts.archive.check2_pooling import margin_ties
        frame = pd.DataFrame(dict(label=[0,0,1,1,0], mean=[1.,1.,1.,2.,3.]))
        self.assertEqual(margin_ties(frame, 'mean'), dict(images=5, unique_margins=3,
            tied_groups=1, images_in_tied_groups=3, duplicate_excess=2, clean_stego_tied_groups=1))

    def test_paired_photo_bootstrap_error_changes(self):
        import pandas as pd
        from scripts.archive.check2_pooling import paired_change
        frame = pd.DataFrame(dict(source_photo=['a','a','b','b'], label=[0,1,0,1],
            mean=[.1,.2,.6,.9], max=[.1,.8,.1,.9], top4=[.1,.2,.6,.9]))
        cutoff = dict(mean=.5, max=.5, top4=.5)
        result = paired_change(frame, 'error', cutoff, ('mean','max','top4'), replicates=30)
        self.assertEqual(result['max'], dict(change=-.5, low=-.5, high=-.5))
        self.assertEqual(result['top4'], dict(change=0., low=0., high=0.))
        self.assertEqual(result, paired_change(frame, 'error', cutoff, ('mean','max','top4'), replicates=30))

    def test_fixed_cutoff_miss_and_false_alarm_changes(self):
        import pandas as pd
        from scripts.archive.check2_pooling import paired_change
        frame = pd.DataFrame(dict(source_photo=['a','b'], label=[1,1], mean=[.1,.2], max=[.8,.9]))
        cutoff = dict(mean=.5, max=.5)
        misses = paired_change(frame, 'miss', cutoff, ('mean','max'), replicates=10)
        alarms = paired_change(frame.assign(label=0), 'fa', cutoff, ('mean','max'), replicates=10)
        self.assertEqual(misses['max']['change'], -1)
        self.assertEqual(alarms['max']['change'], 1)

    def test_context_auc_uses_exact_registered_low_cells_and_both_labels(self):
        import pandas as pd
        from scripts.archive.check2_pooling import context_auc, TARGETS, MODES
        rows = []
        for panel in (*TARGETS, 'standard'):
            for placement in ('contiguous','random','stride'):
                for k in (1,2):
                    for condition in ('low','noisy'):
                        wanted = panel in TARGETS and k == 1 and condition == 'low' and placement != 'stride'
                        for label in (0,1):
                            rows.append(dict(panel=panel, placement=placement, k=k, condition=condition,
                                source_photo=f'{panel}/{placement}/{k}/{condition}', label=label,
                                **{m:float(label if wanted else 1-label) for m in MODES}))
        panels = [dict(panel=p,pooling=m,pairs=9,photos=5,auc=.8,auc_low=.7,auc_high=.9)
                  for p in ('standard','alaska-seen','lsdir') for m in MODES]
        result = context_auc(pd.DataFrame(rows), panels, replicates=10)
        self.assertEqual(len(result), 12)
        for r in result[:6]:
            self.assertEqual(r['pairs'], 3)
            self.assertEqual((r['auc'],r['auc_low'],r['auc_high']), (1,1,1))
        self.assertEqual({r['context'] for r in result[6:]}, {'overall-standard','overall-alaska-seen'})

    def test_invalid_shape_mode_and_cell_count(self):
        with self.assertRaises(ValueError): pool_logits(torch.zeros(2, 3, 8, 8))
        with self.assertRaises(ValueError): pool_logits(torch.zeros(2, 2, 1, 1), 'top4')
        with self.assertRaises(ValueError): pool_logits(torch.zeros(2, 2, 8, 8), 'invalid')
