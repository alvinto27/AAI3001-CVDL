"""Build/resume the full-photo session cache outside the checkout; CPU only."""
from __future__ import annotations
import argparse
import csv
import json
import multiprocessing as mp
import os
import shutil
import sys
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

os.environ["OMP_NUM_THREADS"] = "1"
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"LSB-Dataset-Generator"))
from lsb_cache import build_entry, initialize_cache
from lsb_source import configure_source_worker


def inventory(root):
    result = []
    for source in ("raise","div2k"):
        with (root/f"data/{source}/photo_manifest.csv").open(newline="",encoding="utf-8") as stream:
            result.extend(csv.DictReader(stream))
    if len(result)!=1900 or len({r["photo_id"] for r in result})!=1900:
        raise ValueError("expected 1,900 unique photos")
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo-root",type=Path,required=True)
    p.add_argument("--cache-root",type=Path,required=True)
    p.add_argument("--workers",type=int,default=8)
    args=p.parse_args()
    root,cache=args.repo_root.resolve(),args.cache_root.resolve()
    if cache.is_relative_to(root) or not 1<=args.workers<=8:
        raise ValueError("cache must be outside checkout; workers 1..8")
    os.umask(0o002)
    initialize_cache(cache)
    photos=inventory(root)
    expected=sum(int(r["source_height"])*int(r["source_width"])*3+128 for r in photos)
    existing=sum(p.stat().st_size for p in cache.glob("*.npy"))
    free=shutil.disk_usage(cache).free
    if free < max(0,expected-existing)+2*1024**3:
        raise ValueError("insufficient free space for remaining cache +2GiB reserve")
    start=time.perf_counter()
    results=[]
    with ProcessPoolExecutor(max_workers=args.workers,mp_context=mp.get_context("spawn"),
                             initializer=configure_source_worker) as pool:
        jobs=[pool.submit(build_entry,r,cache,root) for r in photos]
        for job in as_completed(jobs):
            results.append(job.result())
            if len(results)%100==0:
                print(f"Cache verified/built {len(results)}/1900",flush=True)
    size=sum(p.stat().st_size for p in cache.rglob("*") if p.is_file())
    allocated=sum(p.stat().st_blocks*512 for p in cache.rglob("*") if p.is_file())
    print(json.dumps(dict(status="PASS",photos=len(results),counts=dict(Counter(r["status"] for r in results)),
                         workers=args.workers,omp="1",wall_seconds=time.perf_counter()-start,
                         cache_root=str(cache),array_bytes=sum(r["bytes"] for r in results),
                         file_bytes=size,allocated_bytes=allocated,
                         free_bytes=shutil.disk_usage(cache).free),indent=2),flush=True)


if __name__=="__main__":
    main()
