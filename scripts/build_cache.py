"""Build/resume the full-photo session cache outside the checkout; CPU only."""
from __future__ import annotations
import argparse
import csv
import json
import multiprocessing as mp
import os
import shutil
import subprocess
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


def validate_cache_location(root, cache):
    """Allow an in-checkout cache only when Git confirms it is ignored."""
    root,cache=Path(root).resolve(),Path(cache).resolve()
    if not cache.is_relative_to(root):
        return
    try:
        result=subprocess.run(["git","check-ignore","-q","--",str(cache)],cwd=root,
                              check=False,stdout=subprocess.DEVNULL,stderr=subprocess.PIPE,
                              text=True)
    except FileNotFoundError as error:
        raise ValueError("cache is inside the checkout, but git is unavailable; refusing because ignored status cannot be verified (workers must be 1..8)") from error
    if result.returncode==0:
        return
    if result.returncode==1:
        raise ValueError("cache inside the checkout must be ignored by git (workers must be 1..8)")
    detail=result.stderr.strip() or f"git check-ignore exited {result.returncode}"
    raise ValueError(f"could not verify that the in-checkout cache is ignored: {detail} (workers must be 1..8)")


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo-root",type=Path,required=True)
    p.add_argument("--cache-root",type=Path,required=True)
    p.add_argument("--workers",type=int,default=8)
    args=p.parse_args()
    root,cache=args.repo_root.resolve(),args.cache_root.resolve()
    if not 1<=args.workers<=8:
        raise ValueError("workers must be 1..8")
    validate_cache_location(root,cache)
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
