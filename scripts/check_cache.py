"""CPU-only cache decode scaling, integrity/replay and storage read checks."""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import multiprocessing as mp
import os
import sys
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

os.environ["OMP_NUM_THREADS"]="1"
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"LSB-Dataset-Generator"))
import numpy as np
from PIL import Image
from build_cache import inventory
from lsb_source import load_photo, configure_source_worker, grid_axes
from lsb_cache import load_cached, array_sha256
from lsb_batch import Config, read_config
from lsb_mode1 import replay_legacy


def check(ok,message):
    if not ok:
        raise ValueError(message)


def decode_one(job):
    root,row=job
    start=time.perf_counter()
    a=load_photo(row,Path(root))
    return time.perf_counter()-start,a.nbytes


def decode_scaling(root,photos):
    results=[]
    for camera in ("NIKON D7000","NIKON D90","NIKON D40"):
        selected=[r for r in photos if r["camera_model"]==camera][:8]
        for workers in (1,4,8):
            start=time.perf_counter()
            with ProcessPoolExecutor(max_workers=workers,mp_context=mp.get_context("spawn"),
                                     initializer=configure_source_worker) as pool:
                timings=list(pool.map(decode_one,[(str(root),r) for r in selected]))
            wall=time.perf_counter()-start
            results.append(dict(camera=camera,workers=workers,photos=len(selected),wall_seconds=wall,
                                wall_seconds_per_photo=wall/len(selected),photos_per_second=len(selected)/wall,
                                mean_decode_seconds=float(np.mean([t[0] for t in timings])),
                                photo_ids=[r["photo_id"] for r in selected]))
            print(json.dumps(results[-1]),flush=True)
    return dict(status="PASS",gate="decode",omp="1",results=results)


def verify_one(job):
    root,cache,row=job
    start=time.perf_counter()
    a=load_cached(row,cache,root,mmap=True,verify_hash=True,verify_master_hash=True)
    return row["photo_id"],a.nbytes,time.perf_counter()-start


def verify_all(root,cache,photos):
    start=time.perf_counter()
    with ProcessPoolExecutor(max_workers=8,mp_context=mp.get_context("spawn"),initializer=configure_source_worker) as pool:
        results=list(pool.map(verify_one,[(str(root),str(cache),r) for r in photos]))
    return dict(status="PASS",gate="verify",photos=len(results),hash_mismatches=0,
                array_bytes=sum(r[1] for r in results),wall_seconds=time.perf_counter()-start,
                master_hashes_checked=True,workers=8)


def replay_one(job):
    root,cache,row,tiles,configuration,reference=job
    root=Path(root)
    cfg=Config(**configuration)
    start=time.perf_counter()
    photo=load_cached(row,cache,root,mmap=False,verify_hash=False,verify_master_hash=False)
    read=time.perf_counter()-start
    generation=comparison=0.0
    digest=hashlib.sha256()
    for tile in tiles:
        start=time.perf_counter()
        clean,_=replay_legacy(photo,row,tile,cfg,"clean")
        stego,_=replay_legacy(photo,row,tile,cfg,"stego")
        digest.update(memoryview(stego).cast("B"))
        generation+=time.perf_counter()-start
        if reference:
            start=time.perf_counter()
            for kind,a in (("clean",clean),("stego",stego)):
                with Image.open(root/"Datasets/generated/raise512-run1-records"/tile[f"{kind}_path"]) as image:
                    check(np.array_equal(a,np.asarray(image)),f"cache {kind} mismatch {tile['cover_id']}")
            comparison+=time.perf_counter()-start
    return dict(photo_id=row["photo_id"],pairs=len(tiles),read=read,generation=generation,
                comparison=comparison,stego_digest=digest.hexdigest())


def replay_all(root,cache,photos):
    with (root/"data/raise/legacy_tile_manifest.csv").open(newline="",encoding="utf-8") as stream:
        tiles=defaultdict(list)
        for row in csv.DictReader(stream):
            tiles[row["photo_id"]].append(row)
    cfg=read_config(root/"configs/raise512-run1.yaml")
    report={}
    previous=None
    for name,reference in (("pixel_validation",True),("cache_read_cut_embed",False)):
        start=time.perf_counter()
        results=[]
        with ProcessPoolExecutor(max_workers=8,mp_context=mp.get_context("spawn"),initializer=configure_source_worker) as pool:
            jobs=[pool.submit(replay_one,(str(root),str(cache),r,tiles[r["photo_id"]],cfg.as_dict(),reference))
                  for r in photos if r["source"]=="raise"]
            for future in as_completed(jobs):
                results.append(future.result())
                if len(results)%100==0:
                    print(f"{name} {len(results)}/1000",flush=True)
        wall=time.perf_counter()-start
        fingerprints={r["photo_id"]:r["stego_digest"] for r in results}
        if previous is not None:
            check(previous==fingerprints,"cache-only repeat changed replay bytes")
        previous=fingerprints
        count=sum(r["pairs"] for r in results)
        check(count==49796,"wrong replay pair count")
        report[name]=dict(pairs=count,photos=len(results),mismatches=0,wall_seconds=wall,
                         summed_worker_seconds={f:sum(r[f] for r in results) for f in ("read","generation","comparison")},
                         reference_png_reads=reference,stego_digest_matches=True)
        print(json.dumps(report[name]),flush=True)
    return dict(status="PASS",gate="replay",workers=8,omp="1",nef_decodes=0,**report)


def io_bytes():
    try:
        return dict((a,int(b)) for a,b in (line.split(":") for line in Path("/proc/self/io").read_text().splitlines()))["read_bytes"]
    except (OSError,KeyError,ValueError):
        return None


def memory_info():
    result={}
    for line in Path("/proc/meminfo").read_text().splitlines():
        key,value=line.split(":",1)
        if key in ("MemTotal","MemAvailable","Cached","Buffers","Shmem"):
            result[key]=int(value.split()[0])*1024
    return result


def read_speeds(cache,photos):
    order=list(np.random.default_rng(719).permutation(len(photos)))
    results=[]
    for route in ("normal","mmap_copy"):
        for number in (1,2):
            memory=memory_info()
            initial=io_bytes()
            start=time.perf_counter()
            logical=0
            for i in order:
                path=cache/f"{photos[int(i)]['photo_id']}.npy"
                if route=="normal":
                    a=np.load(path,allow_pickle=False)
                else:
                    mapped=np.load(path,mmap_mode="r",allow_pickle=False)
                    a=mapped.copy()
                    del mapped
                logical+=a.nbytes
                del a
            elapsed=time.perf_counter()-start
            end=io_bytes()
            actual=None if initial is None or end is None else end-initial
            results.append(dict(route=route,pass_number=number,photos=len(order),seconds=elapsed,
                                logical_bytes=logical,logical_MB_per_second=logical/elapsed/1e6,
                                photos_per_second=len(order)/elapsed,read_bytes=actual,
                                physical_MB_per_second=None if actual is None else actual/elapsed/1e6,
                                likely_cached_fraction=None if actual is None else max(0,1-actual/logical),
                                memory_before=memory,memory_after=memory_info()))
            print(json.dumps(results[-1]),flush=True)
    rng=np.random.default_rng(818)
    initial=io_bytes()
    memory=memory_info()
    start=time.perf_counter()
    logical=0
    for i in rng.permutation(len(photos)):
        row=photos[int(i)]
        a=np.load(cache/f"{row['photo_id']}.npy",mmap_mode="r",allow_pickle=False)
        ys,xs=grid_axes(row,"overlap_last",512)
        y,x=ys[int(rng.integers(len(ys)))],xs[int(rng.integers(len(xs)))]
        tile=a[y:y+512,x:x+512].copy()  # force real page accesses, not just map creation
        logical+=tile.nbytes
        del tile,a
    elapsed=time.perf_counter()-start
    end=io_bytes()
    actual=None if initial is None or end is None else end-initial
    tiled=dict(tiles=len(photos),seconds=elapsed,tiles_per_second=len(photos)/elapsed,
               logical_bytes=logical,logical_MB_per_second=logical/elapsed/1e6,read_bytes=actual,
               physical_MB_per_second=None if actual is None else actual/elapsed/1e6,
               memory_before=memory,memory_after=memory_info())
    print(json.dumps(tiled),flush=True)
    # Large stream through all arrays exceeds RAM; read() not np.load.
    initial=io_bytes()
    start=time.perf_counter()
    sequential=0
    for row in photos:
        with (cache/f"{row['photo_id']}.npy").open("rb") as stream:
            for block in iter(lambda:stream.read(8*1024*1024),b""):
                sequential+=len(block)
    elapsed=time.perf_counter()-start
    end=io_bytes()
    actual=None if initial is None or end is None else end-initial
    disk=dict(file_bytes=sequential,seconds=elapsed,logical_MB_per_second=sequential/elapsed/1e6,
              read_bytes=actual,physical_MB_per_second=None if actual is None else actual/elapsed/1e6)
    return dict(status="PASS",gate="read",full_photo=results,random_tile=tiled,sequential=disk,
                note="No drop_caches. read_bytes is kernel-attributed storage IO, includes readahead; cached fractions are estimates, not controlled cold-cache results.")


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo-root",type=Path,required=True)
    p.add_argument("--cache-root",type=Path,required=True)
    p.add_argument("--gate",choices=["decode","verify","replay","read"],required=True)
    args=p.parse_args()
    root,cache=args.repo_root.resolve(),args.cache_root.resolve()
    photos=inventory(root)
    if args.gate=="decode":
        report=decode_scaling(root,photos)
    elif args.gate=="verify":
        report=verify_all(root,cache,photos)
    elif args.gate=="replay":
        report=replay_all(root,cache,photos)
    else:
        os.sched_setaffinity(0,{min(os.sched_getaffinity(0))})
        report=read_speeds(cache,photos)
    print(json.dumps(report,indent=2),flush=True)


if __name__=="__main__":
    main()
