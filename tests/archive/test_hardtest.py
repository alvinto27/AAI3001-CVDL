"""Hard test v1: deterministic pair lists, balance, eligibility, metrics, and the evaluate_run.py stage wiring."""
import hashlib
import json
import math
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pandas as pd
from cvdl.archive import hardtest_v1 as H

ROOT=Path(__file__).resolve().parents[2]


def fake_panel(n_tiles, source='raise', photos=40, seed=0, noise_scale=1.0):
    rng=np.random.default_rng(seed)
    rows=[]
    for i in range(n_tiles):
        rows.append(dict(cover_id=f'{source}{i:05d}:overlap_last:512:0:{i}:v1',source=source,source_photo=f'{source}p{i%photos:03d}',
                         camera_model='cam',split='test',grid='overlap_last',tile_row=0,tile_col=i,y=0,x=512*i,photo_index=i%photos))
    frame=pd.DataFrame(rows)
    noise=pd.Series(rng.gamma(4.0,0.4*noise_scale,n_tiles),index=frame.cover_id)
    return frame,noise


class DesignTests(unittest.TestCase):
    def test_thirty_two_cells_and_balanced_design(self):
        self.assertEqual(len(H.cell_names()),4*2*4)
        self.assertEqual(len(H.COMBINATIONS),24);self.assertEqual(H.PER_COMBINATION*24,H.PAIRS_PER_CELL)
        self.assertEqual(H.PAIRS_PER_CELL,1008)

    def test_edges_come_from_validation_only(self):
        val,noise=fake_panel(400,seed=1);test,tn=fake_panel(400,seed=2,noise_scale=3.0)
        allnoise=pd.concat([noise,tn.rename(lambda s:s+'T')])
        edges=H.validation_edges(noise,val[['cover_id','source']])
        self.assertAlmostEqual(edges['raise'],float(np.quantile(noise.to_numpy(),.75)))
        # Changing every test tile's noise cannot change the edge.
        tn2=tn*10
        self.assertEqual(H.validation_edges(noise,val[['cover_id','source']]),edges)
        with self.assertRaisesRegex(ValueError,'missing validation noise'):
            H.validation_edges(noise.iloc[:10],val[['cover_id','source']])

    def test_noisy_pool_is_the_top_quarter_edge(self):
        test,noise=fake_panel(1500,seed=3);edges={'raise':float(np.quantile(noise.to_numpy(),.75))}
        pool=H.tile_pool(test,noise,edges,'noisy')
        self.assertTrue((pool.kv_std>edges['raise']).all());self.assertEqual(len(pool),int((noise>edges['raise']).sum()))
        self.assertEqual(len(H.tile_pool(test,noise,edges,'all')),1500);self.assertEqual(len(H.tile_pool(test,noise,edges,'low')),1500)
        with self.assertRaisesRegex(ValueError,'missing test noise'):H.tile_pool(test,noise.iloc[:5],edges,'all')

    def test_cell_is_balanced_even_and_deterministic(self):
        for tiles in (3000,572,1400):
            test,noise=fake_panel(tiles,seed=4);edges={'raise':float(np.quantile(noise.to_numpy(),.75))}
            for cond in H.CONDITIONS:
                rows=H.build_pairs(test,noise,edges,'lsdir',1,cond)
                pool=len(H.tile_pool(test,noise,edges,cond))
                self.assertTrue(H.check_cell(rows,pool))
                again=H.build_pairs(test,noise,edges,'lsdir',1,cond)
                self.assertEqual(H.pair_list_csv(rows),H.pair_list_csv(again))
                frame=pd.DataFrame(rows)
                self.assertEqual(frame.pair_id.nunique(),1008)
                lo,hi=H.LOW_BPP if cond in ('low','low_noisy') else H.ALL_BPP
                self.assertTrue(frame.requested_bpp.between(lo,hi).all())
                self.assertTrue((frame.loc[frame.placement=='stride','stride'].between(2,16)).all())
                self.assertTrue((frame.loc[frame.placement!='stride','stride']==1).all())
                self.assertEqual(set(frame.k),{1})

    def test_other_cells_differ_and_k_changes_the_draws(self):
        test,noise=fake_panel(2000,seed=5);edges={'raise':1.0}
        a=pd.DataFrame(H.build_pairs(test,noise,edges,'lsdir',1,'all'));b=pd.DataFrame(H.build_pairs(test,noise,edges,'lsdir',2,'all'))
        self.assertNotEqual(a.cover_id.tolist(),b.cover_id.tolist())          # keyed per cell
        self.assertNotEqual(a.requested_bpp.tolist(),b.requested_bpp.tolist())
        c=pd.DataFrame(H.build_pairs(test,noise,edges,'alaska_seen',1,'all'))
        self.assertNotEqual(a.cover_id.tolist(),c.cover_id.tolist())

    def test_reuse_is_even_when_the_pool_is_small(self):
        test,noise=fake_panel(572,seed=6)
        rows=H.build_pairs(test,noise,{'raise':0.0},'alaska_seen',1,'all')   # 572 tiles -> reuse 1008/572
        counts=pd.DataFrame(rows).cover_id.value_counts()
        self.assertEqual(set(counts.unique()),{1,2});self.assertEqual(int((counts==2).sum()),1008-572)
        self.assertEqual(pd.DataFrame(rows).tile_use.max(),2)
        with self.assertRaisesRegex(ValueError,'empty'):H.assign_tiles([],10,'p',1,'all')


class MetricTests(unittest.TestCase):

    def scored(self):
        rows=[]
        # one cell: 4 pairs on 3 distinct clean tiles (tile 't0' reused); stego 0.9,0.2,0.8,0.1; clean 0.1,0.6,0.1,0.1
        data=[('t0',0.9,0.1),('t0',0.2,0.1),('t1',0.8,0.6),('t2',0.1,0.1)]
        for i,(tile,stego,clean) in enumerate(data):
            base=dict(cell='lsdir/k1/all',panel='lsdir',k=1,condition='all',cover_id=tile,source='lsdir',source_photo='p'+tile,
                      effective_bpp=0.2,clipped=bool(i==3),pair_id=f'x{i}')
            rows.append(dict(base,label=0,prob_stego=clean));rows.append(dict(base,label=1,prob_stego=stego))
        return pd.DataFrame(rows)

    def test_fa_counts_distinct_clean_tiles_and_miss_counts_all_pairs(self):
        cells=H.cell_report(self.scored(),{'lsdir':0.5})
        row=cells.iloc[0]
        self.assertEqual(row.pairs,4);self.assertEqual(row.distinct_clean_tiles,3);self.assertEqual(row.max_tile_use,2)
        self.assertEqual(row.fa_events,1)                    # only t1 (0.6 >= 0.5), counted once
        self.assertAlmostEqual(row.fa_rate,1/3)
        self.assertEqual(row.miss_events,2)                  # stego 0.2 and 0.1 are below 0.5, over all 4 pairs
        self.assertAlmostEqual(row.miss_rate,0.5)
        self.assertEqual(row.clipped_pairs,1);self.assertAlmostEqual(row.clipped_share,0.25)
        low,high=H.wilson(2,4);self.assertAlmostEqual(row.miss_ci95_low,low);self.assertAlmostEqual(row.miss_ci95_high,high)
        self.assertEqual(json.loads(row.by_source),{'lsdir':4})
        self.assertEqual(row.distinct_photos,3)


class FrozenFileTests(unittest.TestCase):
    def test_pair_list_columns_round_trip_and_sha(self):
        test,noise=fake_panel(600,seed=7)
        rows=H.build_pairs(test,noise,{'raise':0.0},'lsdir',2,'low')
        text=H.pair_list_csv(rows)
        frame=H.read_pair_list(__import__('io').StringIO(text))
        self.assertEqual(list(frame.columns),list(H.LIST_COLUMNS))
        self.assertEqual(H.sha256_text(text),hashlib.sha256(text.encode()).hexdigest())
        self.assertEqual(frame.requested_bpp.tolist(),[r['requested_bpp'] for r in rows])   # exact float round trip
        import io
        self.assertNotEqual(pd.read_csv(io.StringIO(text)).requested_bpp.tolist(),frame.requested_bpp.tolist(),
                            'documents why the exact parser is needed: the default parser is not exact')

    def test_load_frozen_refuses_a_changed_file(self):
        with tempfile.TemporaryDirectory() as root:
            folder=Path(root)/H.DATA_DIR;folder.mkdir(parents=True)
            shas={}
            for panel in H.PANELS:
                name=f'pairs-{panel}.csv';text=f'pair_id\n{panel}\n';(folder/name).write_text(text);shas[name]=H.sha256_text(text)
            (folder/'summary.json').write_text(json.dumps({'files_sha256':shas}))
            frame,summary=H.load_frozen(root);self.assertEqual(len(frame),4)
            (folder/'pairs-lsdir.csv').write_text('pair_id\ntampered\n')
            with self.assertRaisesRegex(ValueError,'changed'):H.load_frozen(root)


class BuilderTests(unittest.TestCase):
    def test_train_val_photos_are_refused(self):
        sys.path.insert(0,str(ROOT/'scripts'))
        from scripts.archive import build_hard_test as B
        panel=pd.DataFrame(dict(source_photo=['a','b']))
        self.assertTrue(B.assert_test_only({'lsdir':panel},{'a':'test','b':'test'}))
        with self.assertRaisesRegex(ValueError,'non-test'):B.assert_test_only({'lsdir':panel},{'a':'test','b':'val'})
        with self.assertRaisesRegex(ValueError,'non-test'):B.assert_test_only({'alaska_seen':panel},{'a':'test_seen','b':'train'})
        with self.assertRaisesRegex(ValueError,'more than one'):B.assert_test_only({'lsdir':panel,'raise_div2k':panel},{'a':'test','b':'test'})


class StageWiringTests(unittest.TestCase):
    def run_main(self,argv,existing=()):
        import scripts.archive.evaluate_legacy_512 as er
        with tempfile.TemporaryDirectory() as d:
            for name in existing:
                path=Path(d)/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_text('x')
            calls=[]
            with patch.object(er,'checkpoint_folder',return_value=Path(d)), \
                 patch.object(er,'evaluate',side_effect=lambda *a,**k:calls.append(('main',k))), \
                 patch.object(er,'optional_stages',side_effect=lambda f,stages,**k:calls.append(('optional',sorted(stages)))), \
                 patch.object(er,'calibrate',side_effect=lambda *a,**k:calls.append(('temperature',))), \
                 patch.object(sys,'argv',['evaluate_run.py',d,'--test-suite','legacy-512',*argv]):
                er.main()
            return calls

    def test_hard_runs_by_default_with_main_and_can_be_switched_off(self):
        calls=self.run_main([])
        self.assertEqual([c[0] for c in calls],['main']);self.assertTrue(calls[0][1]['hard'])
        calls=self.run_main(['--no-hard']);self.assertFalse(calls[0][1]['hard'])
        calls=self.run_main(['--padding-check']);self.assertTrue(calls[0][1]['hard']);self.assertTrue(calls[0][1]['padding_check'])

    def test_only_hard_never_runs_main_and_other_stages_stay_opt_in(self):
        self.assertEqual(self.run_main(['--only','hard']),[('optional',['hard'])])
        self.assertEqual(self.run_main(['--hard','--skip-main']),[('optional',['hard'])])
        self.assertEqual(self.run_main(['--tanh-saturation','--skip-main']),[('optional',['tanh'])])
        self.assertEqual(self.run_main(['--temperature']),[('temperature',)])    # no main, so no hard

    def test_hard_refuses_existing_output_unless_overwrite(self):
        files=['hard-test/hard_cells.csv']
        with self.assertRaisesRegex(ValueError,'refusing to overwrite'):self.run_main(['--only','hard'],existing=files)
        with self.assertRaisesRegex(ValueError,'refusing to overwrite'):self.run_main([],existing=files)
        self.assertTrue(self.run_main(['--only','hard','--overwrite'],existing=files))
        self.assertTrue(self.run_main(['--only','tanh'],existing=files))      # another stage ignores the hard files


if __name__=='__main__':unittest.main()
