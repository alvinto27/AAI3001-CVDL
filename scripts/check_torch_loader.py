"""VM-only loader parity/counts, two-minute dummy consumption and idle L4 check."""
from __future__ import annotations
import argparse
import ast
import dataclasses
import hashlib
import json
import os
import sys
import threading
import time
from pathlib import Path
from subprocess import check_output

os.environ["OMP_NUM_THREADS"]="1"
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"LSB-Dataset-Generator"))
import numpy as np
import psutil
from PIL import Image
from lsb_torch import (torch,Corpus,RunSettings,TrainingImages,FixedPairs,training_loader)
from build_cache import inventory


def check(ok,message):
    if not ok:
        raise ValueError(message)


def fingerprint(dataset,workers=0):
    result={}
    loader=training_loader(dataset,batch_size=32,num_workers=workers,pin_memory=False,persistent_workers=False)
    for images,labels,indices in loader:
        for a,l,i in zip(images,labels,indices):
            result[int(i)]=(int(l),hashlib.sha256(a.numpy().tobytes()).hexdigest())
    return result


def selected_photos(root):
    chosen={}
    photos=inventory(root)
    for r in photos:
        if r["source"]=="raise" and r["split"]=="train":
            orient="landscape" if int(r["source_width"])>int(r["source_height"]) else "portrait"
            chosen.setdefault((r["camera_model"],orient),r)
    return list(chosen.values())+[r for r in photos if r["source"]=="div2k" and r["split"]=="train"][:2]


def parity(root,cache):
    photos=selected_photos(root)
    report=[]
    for size in (512,256):
        settings=RunSettings(tile_size=size,buffer_size=8,augment=False)
        decode=Corpus(root,settings,records=photos)
        cached=Corpus(root,dataclasses.replace(settings,source="cache",cache_root=str(cache)),records=photos)
        a=fingerprint(TrainingImages(decode))
        b=fingerprint(TrainingImages(cached))
        check(a==b,"decode/cache train mismatch")
        # Workers change scheduling only; raw and keyed-augmentation tensors agree.
        c=fingerprint(TrainingImages(cached),4)
        check(b==c,"worker count changed raw samples")
        val_photos=[{**r,"split":"val"} for r in photos]
        vd=FixedPairs(Corpus(root,settings,records=val_photos),"val")
        vc=FixedPairs(Corpus(root,dataclasses.replace(settings,source="cache",cache_root=str(cache)),records=val_photos),"val")
        check(vd.frame.equals(vc.frame),"decode/cache frame mismatch")
        for i in range(len(vd)):
            da,db,_=vd[i]
            ca,cb,_=vc[i]
            check(torch.equal(da,ca) and torch.equal(db,cb),f"val pixels mismatch {i}")
        report.append(dict(tile_size=size,photos=len(photos),train_images=len(a),val_pairs=len(vd),
                           decode_cache_mismatches=0,worker_count_mismatches=0))
    return dict(status="PASS",gate="parity",results=report)


def legacy(root,cache):
    settings=RunSettings(source="cache",cache_root=str(cache),augment=False)
    dataset=FixedPairs(Corpus(root,settings),"all","legacy_v6")
    chosen=np.linspace(0,len(dataset)-1,2000,dtype=int)
    for i in chosen:
        clean,stego,_=dataset[int(i)]
        original=dataset.entries[int(i)][4]
        for kind,a in (("clean",clean),("stego",stego)):
            with Image.open(root/"Datasets/generated/raise512-run1-records"/original[f"{kind}_path"]) as image:
                check(np.array_equal(a.permute(1,2,0).numpy(),np.asarray(image)),f"legacy mismatch {i}/{kind}")
    return dict(status="PASS",gate="legacy",profile_pairs=len(dataset),sample_pairs=len(chosen),mismatches=0)


def counts(root):
    results=[]
    expected={512:{"train":107900,"val":22476,"test":21912},
              256:{"train":405144,"val":84408,"test":82176}}
    for size in (512,256):
        corpus=Corpus(root,RunSettings(tile_size=size))
        actual={"train":len(TrainingImages(corpus))}
        for purpose in ("val","test"):
            actual[purpose]=len(FixedPairs(corpus,purpose))*2
        check(actual==expected[size],f"count mismatch {size}/{actual}")
        doubled=Corpus(root,RunSettings(tile_size=size,variants_per_tile=2))
        check(len(TrainingImages(doubled))==actual["train"]*2,"variant count mismatch")
        results.append(dict(tile_size=size,images=actual,variants2_train=len(TrainingImages(doubled)),
                            selected_train_photos=len(corpus.selected)))
    return dict(status="PASS",gate="counts",results=results)


class BenchStream(TrainingImages):
    """Bench-only bounded photo set and natural producer stop, not run semantics."""
    def __init__(self,corpus,photo_limit=750):
        corpus.selected=corpus.train_schedule(0)[:photo_limit]
        super().__init__(corpus)
        self._stop_production=torch.zeros(1,dtype=torch.int64).share_memory_()
    def stop_production(self):
        self._stop_production[0]=1
    def _items(self,schedule,epoch):
        for item in super()._items(schedule,epoch):
            if int(self._stop_production.item()):
                return
            yield item


class Monitor:
    """Whole process tree CPU/read counters and peak RSS/PSS, sampled each second."""
    def __init__(self,gpu=False):
        self.stop=threading.Event()
        self.samples=[]
        self.gpu=gpu
        self.process=psutil.Process()
        self.thread=threading.Thread(target=self._run,daemon=True)
    def _run(self):
        while not self.stop.is_set():
            cpu=read=rss=pss=main_cpu=0
            for proc in [self.process]+self.process.children(recursive=True):
                try:
                    t=proc.cpu_times()
                    cpu+=t.user+t.system
                    if proc.pid==self.process.pid:
                        main_cpu=t.user+t.system
                    read+=proc.io_counters().read_bytes
                    rss+=proc.memory_info().rss
                    pss+=getattr(proc.memory_full_info(),"pss",0)
                except (psutil.NoSuchProcess,psutil.AccessDenied):
                    pass
            sample=dict(time=time.perf_counter(),cpu=cpu,main_cpu=main_cpu,read=read,rss=rss,pss=pss)
            if self.gpu:
                try:
                    sample["gpu_util"]=float(check_output(["nvidia-smi","--query-gpu=utilization.gpu",
                                                          "--format=csv,noheader,nounits"],text=True).strip())
                except Exception:
                    pass
            self.samples.append(sample)
            self.stop.wait(1)
    def start(self):
        self.thread.start()
    def finish(self,start,end):
        self.stop.set()
        self.thread.join()
        points=[s for s in self.samples if start<=s["time"]<=end]
        if len(points)<2:
            return {}
        first,last=points[0],points[-1]
        duration=last["time"]-first["time"]
        return dict(cpu_percent=max(0,(last["cpu"]-first["cpu"])/duration*100),
                    main_cpu_percent=max(0,(last["main_cpu"]-first["main_cpu"])/duration*100),
                    disk_read_MB_per_second=max(0,(last["read"]-first["read"])/duration/1e6),
                    peak_rss_bytes=max(s["rss"] for s in points),peak_pss_bytes=max(s["pss"] for s in points),
                    gpu_util_mean=float(np.mean([s["gpu_util"] for s in points if "gpu_util" in s])) if self.gpu else None)


def shutdown(loader,iterator=None):
    # Complete the CPU iterator before shutdown: forced early-stop teardown of
    # pinned/shared-tensor queues can abort a worker in upstream torch.
    if iterator is not None:
        for _ in iterator:
            pass
    underlying=getattr(loader,"loader",loader)
    active=getattr(underlying,"_iterator",None)
    if active is not None:
        active._shutdown_workers()
        codes=[worker.exitcode for worker in active._workers]
        check(all(code==0 for code in codes),f"worker teardown failed: {codes}")
        return codes
    return []


def speed(root,cache,source,size,workers,seconds,prefetch,in_order):
    settings=RunSettings(source=source,cache_root=str(cache) if source=="cache" else None,tile_size=size)
    corpus=Corpus(root,settings)
    dataset=BenchStream(corpus,photo_limit=len(corpus.selected))
    loader=training_loader(dataset,batch_size=32,num_workers=workers,pin_memory=True,
                           prefetch_factor=prefetch,in_order=in_order)
    monitor=Monitor()
    monitor.start()
    initial=time.perf_counter()
    iterator=iter(loader)
    for _ in range(16):
        next(iterator)
    filled=time.perf_counter()
    images=batches=0
    for batch,labels,indices in iterator:
        images+=len(batch)
        batches+=1
        check(batch.dtype==torch.uint8 and batch.is_pinned(),"dummy consumer contract/pin mismatch")
        if time.perf_counter()-filled>=seconds:
            break
    end=time.perf_counter()
    resource=monitor.finish(filled,end)
    dataset.stop_production()
    cleanup_start=time.perf_counter()
    exit_codes=shutdown(loader,iterator)
    cleanup_seconds=time.perf_counter()-cleanup_start
    return dict(status="PASS",worker_exit_codes=exit_codes,source=source,tile_size=size,workers=workers,
                prefetch_factor=prefetch,in_order=in_order,bounded_photos=len(dataset.corpus.selected),
                cleanup_seconds=cleanup_seconds,images=images,batches=batches,
                startup_and_16batch_warmup_seconds=filled-initial,seconds=end-filled,
                images_per_second=images/(end-filled),buffer_payload_per_worker=dataset.corpus.run_record["buffer_payload_bytes"],
                epoch_images=len(dataset),**resource)


def idle_gpu():
    active=check_output(["nvidia-smi","--query-compute-apps=pid,process_name,used_memory",
                         "--format=csv,noheader,nounits"],text=True).strip()
    gpu=check_output(["nvidia-smi","--query-gpu=name,utilization.gpu","--format=csv,noheader,nounits"],text=True).strip()
    check(not active,f"GPU not idle: {active}")
    check("L4" in gpu and int(gpu.rsplit(",",1)[1])==0,f"GPU not idle L4: {gpu}")
    return gpu


def gpu_joint(root,cache,workers,notebook,prefetch,in_order,steps,gpu_only=False):
    before=idle_gpu()
    torch.manual_seed(0)
    torch.backends.cudnn.benchmark=True
    cells=json.loads(notebook.read_text())["cells"]
    namespace=dict(np=np,torch=torch,nn=torch.nn,F=torch.nn.functional,CLIP_T=3.0,NORM="group")
    # Execute only notebook definitions/constant assignments, never training/logging cells.
    for index in (10,14):
        parsed=ast.parse("".join(cells[index]["source"]))
        nodes=[n for n in parsed.body if
               (isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id in
                  {"KV_ARRAY","KV_TENSOR"} for t in n.targets)) or
               (index==14 and isinstance(n,(ast.FunctionDef,ast.ClassDef)))]
        exec(compile(ast.Module(body=nodes,type_ignores=[]),f"notebook-cell-{index}","exec"),namespace)
    net=namespace["XuNet"](clip_t=3.0,norm="group").to("cuda").train()
    optimizer=torch.optim.AdamW(net.parameters(),lr=1e-3,weight_decay=1e-4)
    if gpu_only:
        fixed=torch.randint(0,256,(32,3,512,512),dtype=torch.uint8,device="cuda")
        fixed_targets=torch.arange(32,device="cuda",dtype=torch.long)%2
        torch.cuda.synchronize()
    else:
        dataset=BenchStream(Corpus(root,RunSettings(source="cache",cache_root=str(cache))))
        loader=training_loader(dataset,batch_size=32,num_workers=workers,pin_memory=True,
                               prefetch_factor=prefetch,in_order=in_order)
        iterator=iter(loader)
    monitor=Monitor(gpu=True)
    monitor.start()
    start=time.perf_counter()
    first_wait=None
    waits=[]
    for step in range(steps):
        wait=time.perf_counter()
        if gpu_only:
            images,targets=fixed,fixed_targets
        else:
            images,targets,_=next(iterator)
        waits.append(0.0 if gpu_only else time.perf_counter()-wait)
        if step==0:
            first_wait=waits[-1]
        optimizer.zero_grad(set_to_none=True)
        images=images.to("cuda",non_blocking=True).float()
        targets=targets.to("cuda",non_blocking=True)
        loss=torch.nn.functional.cross_entropy(net(images),targets)
        if step%50==0:
            check(bool(torch.isfinite(loss)),"nonfinite joint loss")
        loss.backward()
        optimizer.step()
        if step==19:
            torch.cuda.synchronize()
            steady_start=time.perf_counter()
        if (step+1)%50==0:
            print(f"Joint step {step+1}/{steps}",flush=True)
    torch.cuda.synchronize()
    end=time.perf_counter()
    resource=monitor.finish(steady_start,end)
    if gpu_only:
        exit_codes=[]
    else:
        dataset.stop_production()
        exit_codes=shutdown(loader,iterator)
    return dict(status="PASS",gpu_only=gpu_only,worker_exit_codes=exit_codes,gate="gpu",idle_before=before,steps=steps,batch_size=32,precision="32-true",
                clip_t=3,norm="group",workers=workers,prefetch_factor=prefetch,in_order=in_order,total_seconds=end-start,
                first_batch_wait=first_wait,steady_steps=steps-20,steady_seconds=end-steady_start,
                images_per_second=(steps-20)*32/(end-steady_start),
                mean_loader_wait_after_warmup=float(np.mean(waits[20:])),
                loader_wait_fraction=sum(waits[20:])/(end-steady_start),**resource)


def mixing(root,cache,workers):
    reports=[]
    for size in (512,256):
        snapshots={}
        for group in (0,4):
            dataset=BenchStream(Corpus(root,RunSettings(source="cache",cache_root=str(cache),tile_size=size)))
            loader=training_loader(dataset,batch_size=32,num_workers=workers,pin_memory=True,
                                   prefetch_factor=4,group_images=group)
            offsets=sorted((v,k) for k,v in dataset.corpus.offsets.items())
            starts=np.array([v for v,k in offsets])
            worker_by_photo={r["photo_id"]:w for w in range(workers)
                             for r in dataset.corpus.train_schedule(0)[w::workers]}
            photos,worker_counts,hashes_by_key=[],[],{}
            iterator=iter(loader)
            for batch_number in range(200):
                images,labels,indices=next(iterator)
                names=[offsets[int(np.searchsorted(starts,int(i)//2,side="right")-1)][1] for i in indices]
                photos.append(len(set(names)))
                worker_counts.append(len({worker_by_photo[n] for n in names}))
                for image,index in zip(images,indices):
                    hashes_by_key[int(index)]=hashlib.sha256(image.numpy().tobytes()).hexdigest()
            dataset.stop_production()
            codes=shutdown(loader,iterator)
            snapshots[group]=hashes_by_key
            reports.append(dict(tile_size=size,group_images=group,batches=200,
                                distinct_photos_mean=float(np.mean(photos)),distinct_photos_min=min(photos),
                                distinct_workers_mean=float(np.mean(worker_counts)),distinct_workers_min=min(worker_counts),
                                worker_exit_codes=codes))
        common=set(snapshots[0])&set(snapshots[4])
        check(all(snapshots[0][key]==snapshots[4][key] for key in common),"mixing changed keyed pixels")
        reports[-1]["common_keys_checked"]=len(common)
        reports[-1]["keyed_mismatches"]=0
    return dict(status="PASS",gate="mixing",results=reports)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo-root",type=Path,required=True)
    p.add_argument("--cache-root",type=Path,required=True)
    p.add_argument("--gate",choices=["parity","legacy","counts","speed","gpu","mixing"],required=True)
    p.add_argument("--source",choices=["cache","decode"],default="cache")
    p.add_argument("--tile-size",type=int,default=512)
    p.add_argument("--workers",type=int,default=8)
    p.add_argument("--seconds",type=float,default=120)
    p.add_argument("--notebook",type=Path)
    p.add_argument("--prefetch",type=int,default=1)
    p.add_argument("--unordered",action="store_true")
    p.add_argument("--steps",type=int,default=300)
    p.add_argument("--gpu-only",action="store_true")
    args=p.parse_args()
    torch.set_num_threads(1)
    root,cache=args.repo_root.resolve(),args.cache_root.resolve()
    start=time.perf_counter()
    if args.gate=="parity":
        report=parity(root,cache)
    elif args.gate=="legacy":
        report=legacy(root,cache)
    elif args.gate=="counts":
        report=counts(root)
    elif args.gate=="speed":
        check(args.seconds>=120,"at least 120seconds requested")
        report=speed(root,cache,args.source,args.tile_size,args.workers,args.seconds,args.prefetch,not args.unordered)
    elif args.gate=="mixing":
        report=mixing(root,cache,args.workers)
    else:
        check(args.notebook is not None,"notebook path required")
        check(args.steps>=30,"at least 30 joint steps")
        report=gpu_joint(root,cache,args.workers,args.notebook,args.prefetch,not args.unordered,args.steps,args.gpu_only)
    report["wall_seconds"]=time.perf_counter()-start
    print(json.dumps(report,indent=2),flush=True)


if __name__=="__main__":
    main()
