# G4 run queue

The Orchestrator keeps this list current. Add a run when the user approves it.
Change its state in the same step as the launch, the end, or the evaluation.
One G4 run per boot. Since 2026-10-17 the boot limit is 180 min: use `--timeout-min 165 --max-run-minutes 180` for long runs (shorter runs may keep timeout 105). Exception: run 12 used 300 min (user); the GCE max run duration is still 300 min until the user sets it back.

All runs were rescored on the standard-256-v1 suite + hard-256 test on 2026-10-18 (see the [2026-10-18 journal](../journal/2026-10-18/README.md)); the 256 evaluation id of each run is in its Result cell. Older Result text (512 tests) is kept as written.

States: `queued` (approved, waiting) · `preparing` (code or notebook not ready)
· `running` · `delivered` (on the Pi, not evaluated) · `evaluated` · `failed`.

| # | Run (notebook) | Purpose | Commit | State | Job ID | Result |
| ---: | --- | --- | --- | --- | --- | --- |
| 1 | `experiments/tiling-overlap.ipynb` | Tiling arm 1: overlap-last, 512 | `8fff50d` | evaluated (14:00-14:40Z; 39m49s; best epoch 29, val P_E .0586) | `tiling-overlap-20261016T135926Z-8501` | `runs/<job>/pc-comparison/` (standard P_E 2.80%; padding: misses about double at 50% pad); 256 eval `dedd015621676d05ccf5` |
| 2 | `experiments/tiling-trim.ipynb` | Tiling arm 2: centre trim, 512 | `8fff50d` | evaluated (14:41-15:22Z; 39m53s notebook time) | `tiling-trim-20261016T144117Z-3222` | evaluated: standard P_E 4.07%; 256 eval `711f76f8d35d5c5172d3` |
| 3 | `experiments/tiling-pad.ipynb` | Tiling arm 3: centre zero-pad, 512 | `8fff50d` | evaluated (01:31-02:11Z; 40m05s; best epoch 27, val P_E .0668; 54.65 steps/s, GPU-bound) | `tiling-pad-20261017T013055Z-7210` | evaluated: standard P_E 4.32% (overlap 2.80, trim 4.07); padding-trained model cuts padding misses about 60%; 256 eval `a848547ce1af4d579e4d` |
| 4 | `experiments/tiling-overlap-256.ipynb` | Tiling arm 1 at 256: overlap-last (batch 128, 180,000 tiles/epoch, LR 3e-4) | `30bac1f` | evaluated (02:22-03:18Z; 55m36s; 32-33 steps/s, loader-bound) | `tiling-overlap-256-20261017T022212Z-9778` | evaluated: standard P_E 2.25% (512 overlap 2.80); best so far; 256 eval `c144162a404460293797` |
| 5 | `experiments/tiling-trim-256.ipynb` | Tiling arm 2 at 256: centre trim (same 256 budget) | `30bac1f` | evaluated (03:20-04:16Z; 55m38s; best epoch 27, val P_E .0630) | `tiling-trim-256-20261017T032025Z-8579` | evaluated: standard P_E 3.55% (256 overlap 2.25); 256 eval `a2bb943a222ef5aaafc4` |
| 6 | `experiments/tiling-pad-256.ipynb` | Tiling arm 3 at 256: centre zero-pad (same 256 budget) | `30bac1f` | evaluated (04:31-05:27Z; 54m56s; best epoch 26, val P_E .0641; 33.1 steps/s, loader-bound) | `tiling-pad-256-20261017T043145Z-9721` | evaluated: standard P_E 3.60%; 256 eval `98426d6dd3ea63288791` |
| 7 | `experiments/srm.ipynb` | SRM front end (30 fixed filters per channel) with the 256 overlap recipe | `0d851b6` | evaluated (05:29-07:03Z; 93.1 min; 18.4 steps/s, GPU-bound; best epoch 29, val P_E .0466 vs 256 KV .0543) | `srm-20261017T052857Z-4521` | evaluated: standard P_E 1.67% (256 KV 2.25); better on all 5 tests and all sources; `runs/<job>/post-training/`; 256 eval `11a82bf812e573c7c62d` |
| 8 | `experiments/srm-e50.ipynb` | SRM, 50 epochs, LR drops scaled to epochs 8 and 33 (user 2026-10-17); boot limit 180 min, timeout 165 | `2841fdf` | done (11:13-13:47Z; 153.1 min; best epoch 43, val P_E .0432, AUC .9917); evaluated (standard + hard v1) | `srm-e50-20261017T111310Z-0214` | 256 eval `35afd1c7938bae78968c`: standard-256 P_E 2.54% (best model) |
| 9 | `experiments/srm-hard.ipynb` | SRM hard training: per epoch RAISE/ALASKA/LSDIR 72,000 each (216,000), k weights 3:1:1:1, bpp 0.5 x U(0.05,0.15) + 0.5 x U(0.05,0.5) (`configs/train-hard-v1.yaml`); 30 epochs; ~112 min. Checkpoint validation: train-matched recipe `bf7ede374cab01f73f5d` (21,600 pairs, 256). Timeout 165, boot limit 180 | `a0e72cd` | **evaluated** (04:42:59-06:28:44Z, 105.6 min, exit 0, first attempt; warm epoch ~205 s: train ~183 s, val ~22 s; best epoch 28, val P_E .1595, AUC .9325 on its own train-matched validation; Pi = PC SHA-256 verified; ~$2.9) | `srm-hard-20261018T043854Z-8467` | 256 eval `a08cade1bcae87ff827c`: standard-256 P_E 3.10% (SRM-30 3.07); mostly moved the threshold |
| 10 | `experiments/srm-low-bpp-curriculum.ipynb` | Low-bpp curriculum (ideas #9; [plan](plan-low-bpp-curriculum.md)): from SRM-e50 best weights (SHA `bf64848a...f076`), 3 stages x 5 epochs (bpp 0.05-0.35 / 0.25 / 0.15; k=1 weight 1/2/3), exactly 25% of stegos per epoch with the original `train-k1to4.yaml` mix; LR 3e-5; train-matched validation `60da...`; last epoch evaluated; ~50 min | `c501603` (job commit `beb4ad9`) | **evaluated** (eval `e9ef059bb12c9076a226`, last.ckpt, cut-off .5296; standard-256 P_E 3.10% vs SRM-e50 2.54%, AUC .9964 both; report notebook built) (05:16:02-06:03:19Z, 47.3 min, exit 0; SRM-e50 checkpoint staged on the VM by maintenance boot 05:10:50-05:12:21Z; all 15 epochs exactly 45,000 original + 135,000 stage stegos; start SHA bf64848a...f076, optimiser not restored; last epoch 14 val P_E .1093, AUC .9673 on `60da...` (SRM-e50 on the same list: .1029 / .9694); Pi = PC SHA-256; last.ckpt `ae1a521c...`) | `srm-low-bpp-curriculum-20261019T051356Z-9566` | `e9ef059bb12c9076a226`: small real gain (+.018 pooled k=1 low AUC vs SRM-e50), rule not met, not adopted |
| 11 | `experiments/srm-low-bpp-control.ipynb` | Control for run 10: same start, 15 epochs, LR and settings; 100% original embedding mix (differs only in `embedding_schedule`); ~50 min | `c501603` (job commit `beb4ad9`) | **evaluated** (eval `af6e9879061fd441462f`, last.ckpt `3d792d35...`, cut-off .4875; standard-256 P_E 2.65%) (Run_Supervisor; launched 06:07:05Z, no stockout; notebook ended 06:55:13Z, 2831 s, exit 0; VM TERMINATED 06:57Z; Pi = PC SHA-256) | `srm-low-bpp-control-20261019T060708Z-9736` | `af6e9879061fd441462f`: control; no gain vs SRM-e50 (-.001) |
| 12 | `experiments/srm-delayed.ipynb` | Architecture arm A ([plan](plan-architecture-tests.md)): SRM-30 recipe with the first two pools removed (`xunet_wide_delayed`); train-matched validation `60da...`; best checkpoint. PC pilot: 2.87x slower than SRM-30, predicted ~267 min. **Boot limit raised for this run (user): GCE max run 300 min, `--timeout-min 285 --max-run-minutes 300`**; stop early if a warm epoch exceeds ~9 min | `1bb7e20` (job commit `90243d6`) | done (VM 10:07:37-13:45:43Z; all 10 cells pass in 12,958 s, about 216 min; delivered to the Pi; best epoch 26, val P_E .1221, AUC .9558); evaluated | `srm-delayed-20261019T100714Z-3282` | `9915836bfe1b229e5259`: no gain (pooled hard k=1 low AUC -.060 vs SRM-30) |
| 13 | `experiments/srm-cell-aux.ipynb` | Cell-level training ([plan](plan-cell-training-256.md)): run 11's recipe (SRM-e50 start, 15 epochs, LR 3e-5, `60da` validation, last checkpoint) plus a 1x1 cell head trained on changed-cell targets (CE + 0.2 x cell BCE); compare with run 11 (2.65%) | `731a27c` (job commit `56aa917`) | evaluated | `srm-cell-aux-20261021T054704Z-6997` | Eval `ea32a4be0e2ce3597fbd` (last, train-matched 60da, eval batch 32): standard P_E **2.97%** (control 2.65%, +0.32 pts: guard missed), AUC .9962; hard k1-low-contiguous AUC .844 vs .839 (+.005 [+.0004, +.0097]; bonus not met); noisy-clean FA 18.9% vs 20.7%. Cell map (primary within-tile AUC): aux **.863** vs own CAM .806, control CAM .793; aux - control CAM +.070 [+.053, +.088]. Recall at 5% val cell FA about 16% for all maps. Figure `runs/verification/cell-run13-2026-10-21/cells/cells.png` |

## Next (approved in principle, not queued)

- None. (The SRM front end became runs 7-9.)

## PC task queue (free, no G4)

| Task | State | Note |
| --- | --- | --- |
| SRM notebook build | done (`632dd22`, `97487b8`, `0d851b6`) | run 7 |
| Padding check fix | done (`970d337`) | worker DataLoader, 2,000 pairs per shape and level (18,000 cases) |
| Temperature scaling | done (`f0bb81d`) | opt-in stage; helps only wide-R2 (ties) and slightly 256 overlap / SRM |
| tanh saturation | done (`0e05dcb`) | opt-in stage; only SRM tanh1 saturates (20.8% / 26.4%) |
| 256 loader speed | done (`863d9fb`, `efde4df`) | `pack_blocks=16` default; batches identical; G4 gain not measured (SRM GPU-bound) |
| SRM weakness breakdown | done | `runs/srm-20261017T052857Z-4521/post-training/breakdown/` |
| Hard test v1 | done (`d74d716`) | `data/hard-test/v1/`; ~~default evaluation stage~~ retired 2026-10-21 with the legacy-512 evaluation ([legacy-512 retirement](legacy-512-retirement.md)) |
| `srm-hard` notebook | done (`795fd93`, updated `a0e72cd`) | run 9, done |
| Standard-256 / hard-256 / threshold / rescoring / report notebook | done (`bd36403`, `a56c275`, `68e4f0b`, `dd9867a`, `716b2cd`) | all runs scored |
| Diagnostic checks 1, 2, 2b | done (`ed52419`, `9563540`, `d3ae6b8`) | [plan](plan-diagnostic-checks.md) |
| PC pilot mode | done (`aec69d0`) | `--set pilot=true` |
| Photo-level decision | deferred by the user | — |

## After the runs

- Copy each delivered run to the PC (check checkpoint SHA-256 Pi = PC), evaluate it with `scripts/evaluate_run.py` (standard-256-v1 + hard-256; SRM models on the PC: `--set rocm_conv1_fix=off --set eval_batch_size=32`), set its state to `evaluated` and put the evaluation id and result in its row.
- ~~Reports: 3-arm tiling comparison at 512 and 256~~ done: [training tiling result](../results/result-training-tiling.md).
