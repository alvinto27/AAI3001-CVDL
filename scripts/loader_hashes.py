"""Hash every training batch of full epochs: images, labels, pair indices. Used to prove the packed loader.

CPU only. Prints one line per (epoch, workers): number of batches and a SHA-256 over all batch bytes in
order, so two runs (for example --pack-blocks 1 and 16) can be compared by eye or with diff.
Example: python scripts/loader_hashes.py --tile 256 --batch 128 --workers 6 --epochs 0,1 --pack-blocks 16
"""
from __future__ import annotations
import argparse
import hashlib
import sys
import time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'LSB-Dataset-Generator'))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--tile',type=int,choices=(256,512),required=True)
    p.add_argument('--batch',type=int,required=True)
    p.add_argument('--workers',type=int,required=True)
    p.add_argument('--epochs',default='0,1')
    p.add_argument('--pack-blocks',type=int,default=1)
    p.add_argument('--tiles',type=int,default=None,help='tiles_per_epoch (default: the full recipe)')
    p.add_argument('--max-batches',type=int,default=0,help='stop after N batches per epoch (0 = full epoch)')
    a=p.parse_args()
    from scripts.pc_safety import check_workers, own_process_group
    print(f'available RAM {check_workers(a.workers):.1f} GB; workers {a.workers}',flush=True)
    own_process_group()
    from scripts.profile_loader import build
    tiles=a.tiles or 45000*(512//a.tile)**2
    s,otf,loader=build(a.tile,a.batch,tiles,a.workers,4,False,pack_blocks=a.pack_blocks)
    try:
        for epoch in map(int,a.epochs.split(',')):
            otf.train.set_epoch(epoch)
            digest=hashlib.sha256();batches=0;images=0;start=time.perf_counter()
            for x,y,idx in loader:
                digest.update(x.numpy().tobytes());digest.update(y.numpy().tobytes());digest.update(idx.numpy().tobytes())
                batches+=1;images+=len(y)
                if a.max_batches and batches>=a.max_batches:break
            print(f'HASH tile={a.tile} batch={a.batch} workers={a.workers} epoch={epoch} batches={batches} images={images} '
                  f'sha256={digest.hexdigest()} seconds={time.perf_counter()-start:.0f}',flush=True)
    finally:
        loader.shutdown()   # joins the DataLoader workers; the process then returns normally (exit 0)


if __name__=='__main__':
    from scripts.pc_safety import guarded
    guarded(main)
