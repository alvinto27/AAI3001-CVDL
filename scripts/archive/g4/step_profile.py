"""Compare notebook-04 steady training steps: raw, Lightning, fixed data and decode.

Reads XuNet and SteganalysisModule class definitions directly from notebook 04
(AST extraction, no training/evaluation cells executed). One process, one GPU;
JSON is replaced after each variant, including on interruption. Durations exclude
startup, compilation, loader creation, warm-up and end-of-epoch hooks. The full
G4 recipe is documented in --help and reported to the orchestrator separately.
"""
from __future__ import annotations
import argparse
import ast
import json
import os
from pathlib import Path
import resource
import sys
import time

os.environ.setdefault("TORCHINDUCTOR_COMPILE_THREADS", "4")
os.environ.setdefault("OMP_NUM_THREADS", "1")
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "LSB-Dataset-Generator"))
import lightning as L
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import IterableDataset, DataLoader
from lsb_torch import RunSettings, Corpus, TrainingImages, training_loader


def notebook_classes():
    """Extract the production model, loss and logging methods, not a model copy."""
    notebook = json.loads((ROOT / "notebooks/04_baseline_kv_xunet_v3.ipynb").read_text())
    classes = []
    for cell, names in ((14, {"KVFilter", "XuNet", "make_norm"}),
                        (19, {"SteganalysisModule"})):
        tree = ast.parse("".join(notebook["cells"][cell]["source"]))
        found = {node.name: node for node in tree.body
                 if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name in names}
        if set(found) != names:
            raise RuntimeError(f"Notebook cell {cell} model definitions changed: {set(found)}")
        classes.extend(found.values())
    # Do not duplicate weights: use the notebook's literal KV_ARRAY expression.
    kv_tree = ast.parse("".join(notebook["cells"][10]["source"]))
    kv_node = next(n for n in kv_tree.body if isinstance(n, ast.Assign) and
                   any(isinstance(t, ast.Name) and t.id == "KV_ARRAY" for t in n.targets))
    rocm_fix = torch.version.hip is not None
    if rocm_fix:
        sys.path.insert(0, str(ROOT / "notebooks"))
        from rocm_conv1_wgrad import UnfoldWgradConv2d
    else:
        UnfoldWgradConv2d = None
    ns = {"torch": torch, "np": np, "nn": nn, "F": F, "L": L,
          "CLIP_T": 3.0, "NORM": "group", "ROCM_CONV1_FIX_ACTIVE": rocm_fix,
          "KV_ARRAY": eval(compile(ast.Expression(kv_node.value), "notebook-04", "eval"), {"np": np}),
          "UnfoldWgradConv2d": UnfoldWgradConv2d}
    ns["KV_TENSOR"] = torch.tensor(ns["KV_ARRAY"], dtype=torch.float32)
    module = ast.fix_missing_locations(ast.Module(body=classes, type_ignores=[]))
    exec(compile(module, "notebook-04-cells-14-and-19", "exec"), ns)
    return ns["SteganalysisModule"]


class FixedGPU(IterableDataset):
    """Lightning's fixed batch is already on the GPU; no loader/transfer work."""
    def __init__(self, batch, length):
        self.batch, self.length = batch, length

    def __iter__(self):
        for _ in range(self.length):
            yield self.batch

    def __len__(self):
        return self.length


class Window(L.Callback):
    def __init__(self, warmup, steps):
        self.warmup, self.steps = warmup, steps
        self.start = self.end = self.cpu_start = self.cpu_end = None

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if batch_idx == self.warmup - 1:
            torch.cuda.synchronize()
            self.start, self.cpu_start = time.perf_counter(), time.process_time()
        if batch_idx == self.warmup + self.steps - 1:
            torch.cuda.synchronize()
            self.end, self.cpu_end = time.perf_counter(), time.process_time()


def measure(label, started, ended, cpu_started, cpu_ended, steps, extra=None):
    seconds = ended - started
    return dict(variant=label, steps=steps, seconds=seconds,
                steps_per_second=steps/seconds, ms_per_step=1000*seconds/steps,
                main_cpu_percent=100*(cpu_ended-cpu_started)/seconds, **(extra or {}))


def make_loader(dataset, workers, batch_size, prefetch):
    return training_loader(dataset,batch_size=batch_size,num_workers=workers,
                           pin_memory=True,prefetch_images=prefetch)


def run_raw(name, Module, dataset, args, fixed_batch):
    loader = None
    model = Module(max_epochs=10).cuda().train()
    if args.compile:
        model.net.compile(mode="default", dynamic=False, fullgraph=False)
    optim = (torch.optim.Adam(model.parameters(),lr=1e-3) if name=="raw_speed_baseline"
             else torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4))
    try:
        if name == "raw_speed_baseline":
            fixed_batch=(fixed_batch[0].float(),fixed_batch[1],fixed_batch[2])
        if name == "raw_decode":
            loader = make_loader(dataset,args.workers,args.batch_size,args.prefetch_images)
            iterator = iter(loader)
        else:
            iterator = iter([fixed_batch] * (args.warmup + args.steps))
        started = cpu_started = None
        for i in range(args.warmup + args.steps):
            if i == args.warmup:
                torch.cuda.synchronize()
                started, cpu_started = time.perf_counter(), time.process_time()
            image, label, index = next(iterator)
            if name == "raw_decode":
                image = image.cuda(non_blocking=True)
                label = label.cuda(non_blocking=True)
            optim.zero_grad(set_to_none=True)
            logits, target = model.training_logits_targets((image,label,index))
            loss = F.cross_entropy(logits,target)
            loss.backward()
            optim.step()
        torch.cuda.synchronize()
        return measure(name,started,time.perf_counter(),cpu_started,time.process_time(),args.steps)
    finally:
        if loader is not None:
            loader.shutdown()
            loader.loader._iterator = None
        del optim, model
        torch.cuda.empty_cache()


def run_lightning(name, Module, dataset, args, fixed_batch, *, progress, logging, callbacks):
    use_decode = name.startswith("lightning_decode")
    loader = (make_loader(dataset,args.workers,args.batch_size,args.prefetch_images) if use_decode
              else DataLoader(FixedGPU(fixed_batch,args.warmup+args.steps),batch_size=None,num_workers=0))
    # Keep the exact module class when logging is enabled; override only self.log
    # in the logging-off variant, preserving forward, loss, optimizer and scheduler.
    if not logging:
        class NoLog(Module):
            def log(self, *args, **kwargs):
                pass
        module_cls = NoLog
    else:
        module_cls = Module
    model = module_cls(max_epochs=10)
    if args.compile:
        model.net.compile(mode="default",dynamic=False,fullgraph=False)
    clock = Window(args.warmup,args.steps)
    cb = []
    if callbacks:
        class NotebookBatchAccumulator(L.Callback):
            """Notebook PerEpochScalars.on_train_batch_end, without epoch I/O."""
            def on_train_epoch_start(self,trainer,pl_module):
                self.loss_sum=self.accuracy_sum=None
                self.count=0
            def on_train_batch_end(self,trainer,pl_module,outputs,batch,batch_idx):
                count=outputs["count"]
                loss=outputs["loss"].detach()*count
                accuracy=outputs["accuracy"].detach()*count
                self.loss_sum=loss if self.loss_sum is None else self.loss_sum+loss
                self.accuracy_sum=accuracy if self.accuracy_sum is None else self.accuracy_sum+accuracy
                self.count+=count
        cb.append(NotebookBatchAccumulator())
    cb.append(clock)  # Timestamp after the real batch-end callback, when present.
    trainer=L.Trainer(max_epochs=1, accelerator="gpu", devices=1, callbacks=cb,
                      enable_progress_bar=progress, logger=False, enable_checkpointing=False,
                      enable_model_summary=False, num_sanity_val_steps=0,
                      limit_train_batches=args.warmup+args.steps, log_every_n_steps=50,
                      deterministic=False)
    try:
        trainer.fit(model,loader)
        if clock.start is None or clock.end is None:
            raise RuntimeError("No steady measurement window")
        return measure(name,clock.start,clock.end,clock.cpu_start,clock.cpu_end,args.steps,
                       {"progress":progress,"logging":logging,"batch_callback":callbacks})
    finally:
        if use_decode:
            loader.shutdown()
            loader.loader._iterator=None
        del model, trainer
        torch.cuda.empty_cache()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out",type=Path,required=True)
    p.add_argument("--source",choices=("decode","cache"),default="decode")
    p.add_argument("--cache-root",type=Path)
    p.add_argument("--workers",type=int,default=32)
    p.add_argument("--batch-size",type=int,default=32)
    p.add_argument("--prefetch-images",type=int,default=128)
    p.add_argument("--warmup",type=int,default=15)
    p.add_argument("--steps",type=int,default=120)
    p.add_argument("--compile",action="store_true",help="Compile the net in place as notebook 04 does")
    p.add_argument("--budget-seconds",type=float,default=660,
                   help="Stop between variants before G4 exceeds this wall-clock budget")
    p.add_argument("--variants",nargs="+",default=["raw_speed_baseline","raw_fixed","raw_decode",
        "lightning_fixed_off","lightning_fixed_progress","lightning_fixed_callback",
        "lightning_fixed_on","lightning_fixed_nolog","lightning_decode_off",
        "lightning_decode_progress","lightning_decode_callback","lightning_decode_on",
        "lightning_decode_nolog"],
        choices=["raw_speed_baseline","raw_fixed","raw_decode","lightning_fixed_off","lightning_fixed_progress",
                 "lightning_fixed_callback","lightning_fixed_on","lightning_fixed_nolog",
                 "lightning_decode_off","lightning_decode_progress","lightning_decode_callback",
                 "lightning_decode_on","lightning_decode_nolog"])
    args=p.parse_args()
    if args.workers<0 or min(args.steps,args.warmup,args.batch_size,args.prefetch_images)<1 or args.budget_seconds<=0:
        p.error("workers must be nonnegative, other counts positive")
    if args.source=="cache" and args.cache_root is None:
        p.error("--cache-root required for cache source")
    if not torch.cuda.is_available():
        p.error("CUDA/ROCm GPU unavailable")
    soft,hard=resource.getrlimit(resource.RLIMIT_NOFILE)
    if soft<65536:
        resource.setrlimit(resource.RLIMIT_NOFILE,(min(65536,hard),hard))
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=True
    torch.backends.cudnn.benchmark=True
    Module=notebook_classes()
    settings=RunSettings(source=args.source,cache_root=str(args.cache_root) if args.cache_root else None,
                         train_photo_fraction=1.0,paired_batches=False,tensor_dtype="uint8")
    train=TrainingImages(Corpus(ROOT,settings))
    if len(train)<args.warmup+args.steps:
        p.error("not enough training batches")
    fixed=(torch.randint(0,256,(args.batch_size,3,512,512),device="cuda",dtype=torch.uint8),
           torch.arange(args.batch_size,device="cuda")%2,
           torch.arange(args.batch_size,device="cuda"))
    report=dict(status="running",source=args.source,workers=args.workers,
                device=torch.cuda.get_device_name(0),compile=args.compile,
                warmup=args.warmup,steps=args.steps,batch_size=args.batch_size,
                model="AST-extracted XuNet and SteganalysisModule from notebook 04 cells 14/19",
                note="Window synchronized on both ends; includes next-batch fetch, excludes startup/compile and epoch hooks. Main CPU percentage: 100% is one core. Fixed GPU data skips transfer. raw_speed_baseline uses fixed float32 GPU inputs and Adam as speed_test.py; raw_fixed uses uint8-to-float and AdamW. No validation/checkpoints; logger=False (Lightning self.log still aggregates unless *_nolog); notebook batch accumulator reproduced for *_callback and *_on, but no end-of-epoch file writes. Progress *_progress and *_on writes to stdout; redirect to a file as on G4.",
                variants=[])
    args.out.parent.mkdir(parents=True,exist_ok=True)
    def save():
        temp=args.out.with_suffix(args.out.suffix+".part")
        temp.write_text(json.dumps(report,indent=2)+"\n")
        temp.replace(args.out)
    save()
    started=time.monotonic()
    try:
        for name in args.variants:
            if time.monotonic()-started>args.budget_seconds:
                report["status"]="budget_exceeded"
                print("Budget exceeded between variants; stop",flush=True)
                break
            print(f"BEGIN {name}",flush=True)
            if name.startswith("raw"):
                row=run_raw(name,Module,train,args,fixed)
            else:
                row=run_lightning(name,Module,train,args,fixed,
                                  progress=name.endswith(("_on","_progress")),
                                  logging=not name.endswith("_nolog"),
                                  callbacks=name.endswith(("_on","_callback")))
            print(json.dumps(row),flush=True)
            report["variants"].append(row)
            save()
        if report["status"]=="running":
            report["status"]="done"
    except BaseException as error:
        report["status"]="failed"
        report["error"]=repr(error)
        raise
    finally:
        save()

if __name__=="__main__":
    main()
