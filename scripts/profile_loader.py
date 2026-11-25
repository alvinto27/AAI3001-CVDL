"""Time the OTF training loader alone (no model): items/s, steps/s, main-process cost.

CPU only. Builds the same loader as a training run (mixed sources, overlap_last, k 1-4 recipe),
then runs a warm-up and a timed loop. Optional: cProfile of the main process, and the
main process / one worker PID for py-spy.
Example: python scripts/profile_loader.py --tile 256 --batch 128 --tiles 180000 --steps 400
"""
from __future__ import annotations
import argparse
import cProfile
import io
import os
import pstats
import sys
import torch
import threading
import time
import types
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'LSB-Dataset-Generator'))


class SyntheticImages(torch.utils.data.IterableDataset):
    """Cheap workers that send the same IPC volume as the real stream (uint8 CHW image, int label, int index).

    Isolates the main-process cost of receiving and joining groups from decode/embed cost in the workers.
    """
    def __init__(self,tile,sharing):
        super().__init__()
        self.tile,self.sharing=tile,sharing
        self.corpus=types.SimpleNamespace(settings=types.SimpleNamespace(paired_batches=False))
    def __len__(self):return 10**9
    def __iter__(self):
        if self.sharing:torch.multiprocessing.set_sharing_strategy(self.sharing)
        pool=[torch.randint(0,256,(3,self.tile,self.tile),dtype=torch.uint8) for _ in range(16)]
        i=0
        while True:
            yield pool[i%16],i%2,i;i+=1


def build(tile,batch,tiles,workers,group_images,pin_memory,synthetic=False,sharing=None,pack_blocks=1):
    if synthetic:
        import lsb_torch
        dataset=SyntheticImages(tile,sharing)
        loader=lsb_torch.training_loader(dataset,batch_size=batch,num_workers=workers,pin_memory=pin_memory,
                                         prefetch_images=128,group_images=group_images,pack_blocks=pack_blocks)
        return None,types.SimpleNamespace(train=dataset),loader
    from cvdl.settings import load
    from cvdl import data
    import lsb_torch
    extra={'train_tile_size':256} if tile==256 else {}
    s=load(environ={'QUICK':'false','V3_WORKERS':str(workers)},name='profile-loader',train_grid='overlap_last',**extra,post_fit_eval='none',
           model='xunet_wide',train_yaml='configs/train-k1to4.yaml',source_mix={'raise':.4,'lsdir':.2,'alaska':.4},
           tiles_per_epoch=tiles,checkpoint_validation='capped',batch_size=batch)
    otf=data.build_otf_data(s.root,s.to_run_settings(),quick=False,checkpoint_validation='capped',
                            train_grid='overlap_last',train_tile=256 if tile==256 else None)
    loader=lsb_torch.training_loader(otf.train,batch_size=batch,num_workers=workers,pin_memory=pin_memory,
                                     prefetch_images=s.prefetch_images,group_images=group_images,pack_blocks=pack_blocks)
    return s,otf,loader


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--tile',type=int,choices=(256,512),required=True)
    p.add_argument('--batch',type=int,required=True,help='images per step')
    p.add_argument('--tiles',type=int,default=None,help='tiles_per_epoch (default: same pixels as 45000 at 512)')
    p.add_argument('--workers',type=int,default=6)
    p.add_argument('--group-images',type=int,default=4,help='images per worker task (the loader default is 4)')
    p.add_argument('--warmup',type=int,default=30)
    p.add_argument('--steps',type=int,default=300)
    p.add_argument('--cprofile',type=int,default=0,metavar='TOP',help='print the top N main-process costs')
    p.add_argument('--pack-blocks',type=int,default=1,help='4-image blocks per worker message (1 = today)')
    p.add_argument('--pin-memory',action='store_true',help='needs the GPU: ask first')
    p.add_argument('--synthetic',action='store_true',help='cheap workers sending the real IPC volume: main-side ceiling')
    p.add_argument('--sharing',choices=('file_descriptor','file_system'),default=None,help='tensor sharing strategy (synthetic only)')
    p.add_argument('--report-pids',action='store_true',help='print main and worker PIDs once the loop starts (for py-spy)')
    a=p.parse_args()
    from scripts.pc_safety import check_workers, own_process_group
    print(f'available RAM {check_workers(a.workers):.1f} GB; workers {a.workers}',flush=True)
    own_process_group()
    tiles=a.tiles or 45000*(512//a.tile)**2
    s,otf,loader=build(a.tile,a.batch,tiles,a.workers,a.group_images,a.pin_memory,a.synthetic,a.sharing,a.pack_blocks)
    if hasattr(otf.train,'set_epoch'):otf.train.set_epoch(0)
    it=iter(loader);images=0;t0=None;profile=cProfile.Profile() if a.cprofile else None
    for step in range(a.warmup+a.steps):
        if step==a.warmup:
            t0=time.perf_counter();c0=time.process_time();images=0
            if a.report_pids:
                kids=[l.split()[0] for l in os.popen(f'ps -o pid=,cmd= --ppid {os.getpid()}')]
                print(f'PIDS main={os.getpid()} children={",".join(kids)}',flush=True)
            if profile:profile.enable()
        x,y,idx=next(it);images+=len(y)
    if profile:profile.disable()
    wall=time.perf_counter()-t0;cpu=time.process_time()-c0
    print(f'RESULT {"synthetic " if a.synthetic else ""}tile={a.tile} batch={a.batch} workers={a.workers} group_images={a.group_images} pack_blocks={a.pack_blocks} sharing={a.sharing} steps={a.steps} '
          f'steps/s={a.steps/wall:.1f} images/s={images/wall:.0f} groups/s={images/a.group_images/wall:.0f} '
          f'Mpx/s={images*a.tile*a.tile/wall/1e6:.0f} main_cpu_share={cpu/wall:.2f} main_ms_per_step={1000*cpu/a.steps:.1f}',flush=True)
    if profile:
        out=io.StringIO();pstats.Stats(profile,stream=out).sort_stats('tottime').print_stats(a.cprofile)
        print(out.getvalue()[:6000])
    if hasattr(loader,'shutdown'):loader.shutdown()   # joins the DataLoader workers; then return normally (exit 0)


if __name__=='__main__':
    from scripts.pc_safety import guarded
    guarded(main)
