"""Bounded first-pass cache sample and notebook-04 loader/model speed comparison."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import sys
import time

os.environ.setdefault("TORCHINDUCTOR_COMPILE_THREADS", "1")
os.environ["OMP_NUM_THREADS"] = "1"
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "LSB-Dataset-Generator"))
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from lsb_torch import Corpus, RunSettings, TrainingImages, training_loader


def io_bytes():
    values = dict(line.split(": ", 1) for line in Path("/proc/self/io").read_text().splitlines())
    return int(values["read_bytes"])


def save(path, report):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".part")
    temp.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    temp.replace(path)
    print(json.dumps(report.get("measurements", [])[-1] if report.get("measurements") else report["cold_sample"]), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--sample-photos", type=int, default=48)
    parser.add_argument("--warmup", type=int, default=12)
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument("--budget-seconds", type=int, default=310)
    parser.add_argument("--workers", type=int, nargs="+", default=[4, 8, 16])
    parser.add_argument("--modes", nargs="+", choices=["eager", "compiled"], default=["eager", "compiled"])
    args = parser.parse_args()
    assert args.sample_photos > 0 and args.steps > 0 and args.warmup >= 0
    cache = args.cache_root.resolve()
    report = {"cache_root": str(cache), "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none",
              "note": "Run after sync and drop_caches; first-pass sample, not a full cold epoch.",
              "measurements": [], "status": "running"}
    if not torch.cuda.is_available():
        raise SystemExit("GPU unavailable")
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    started = time.monotonic()
    settings = RunSettings(source="cache", cache_root=str(cache), train_photo_fraction=1.0,
                           paired_batches=False, tensor_dtype="uint8")
    corpus = Corpus(ROOT, settings)
    train = TrainingImages(corpus)
    schedule = corpus.train_schedule(0)
    train_bytes = sum((cache / f"{row['photo_id']}.npy").stat().st_size for row in schedule)
    report["train_photos"] = len(schedule)
    report["train_bytes"] = train_bytes
    report["train_images"] = len(train)
    # Measure real file reads once, immediately after setup.sh drops the page cache.
    sample = schedule[:args.sample_photos]
    before = io_bytes()
    begin = time.monotonic()
    logical = 0
    for row in sample:
        with (cache / f"{row['photo_id']}.npy").open("rb") as stream:
            while data := stream.read(8 * 1024 * 1024):
                logical += len(data)
    elapsed = time.monotonic() - begin
    physical = io_bytes() - before
    rate = logical / elapsed / 1e6
    report["cold_sample"] = {"photos": len(sample), "bytes": logical, "physical_read_bytes": physical,
                             "likely_cold": physical >= 0.8 * logical,
                             "seconds": elapsed, "logical_MB_per_second": rate,
                             "estimated_first_epoch_read_seconds": train_bytes / (rate * 1e6)}
    if not report["cold_sample"]["likely_cold"]:
        report["note"] += " WARNING: physical reads <80% of sample; do not treat this as cold."
    save(args.out, report)

    # Use the exact model definition from notebook 04, not a second model copy.
    notebook = json.loads((ROOT / "notebooks/04_baseline_kv_xunet_v3.ipynb").read_text())
    source = "".join(notebook["cells"][14]["source"])
    if "model_for_count = XuNet()" not in source:
        raise RuntimeError("Notebook-04 model cell changed; update the speed test")
    kv_array = np.array([[-1, 2, -2, 2, -1], [2, -6, 8, -6, 2],
                         [-2, 8, -12, 8, -2], [2, -6, 8, -6, 2],
                         [-1, 2, -2, 2, -1]], dtype=np.float32) / 12
    namespace = {"np": np, "torch": torch, "nn": nn, "F": F,
                 "KV_ARRAY": kv_array, "CLIP_T": 3.0, "NORM": "group"}
    exec(source.split("model_for_count = XuNet()")[0], namespace)
    for mode in args.modes:
        for workers in args.workers:
            if time.monotonic() - started > args.budget_seconds:
                report["status"] = "budget_exceeded"
                save(args.out, report)
                return
            print(f"Benchmark {mode}, cache workers={workers}", flush=True)
            loader = training_loader(train, batch_size=32, num_workers=workers, pin_memory=True,
                                     prefetch_images=128)
            model = namespace["XuNet"]().cuda().train()
            if mode == "compiled":
                model.compile(mode="default", dynamic=False, fullgraph=False)
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
            begin = None
            fetched = 0
            try:
                for index, batch in enumerate(loader):
                    if index == args.warmup:
                        torch.cuda.synchronize()
                        begin = time.monotonic()
                    images, labels, _ = batch
                    images = images.cuda(non_blocking=True).float()
                    labels = labels.cuda(non_blocking=True)
                    optimizer.zero_grad(set_to_none=True)
                    loss = F.cross_entropy(model(images), labels)
                    loss.backward()
                    optimizer.step()
                    if index >= args.warmup:
                        fetched += images.shape[0]
                    if index + 1 >= args.warmup + args.steps:
                        torch.cuda.synchronize()
                        break
                if begin is None or fetched == 0:
                    raise RuntimeError("Loader did not provide enough batches")
                duration = time.monotonic() - begin
                report["measurements"].append({"mode": mode, "workers": workers,
                    "warmup_steps": args.warmup, "steady_steps": args.steps,
                    "steady_images": fetched, "steady_seconds": duration,
                    "steady_images_per_second": fetched / duration,
                    "estimated_epoch_seconds": report["train_images"] / (fetched / duration)})
                save(args.out, report)
            finally:
                loader.shutdown()
                loader.loader._iterator = None
                del optimizer, model, loader
                torch.cuda.empty_cache()
    fastest = max(report["measurements"], key=lambda row: row["steady_images_per_second"])
    report["best"] = fastest
    report["cold_read_to_best_epoch_ratio"] = report["cold_sample"]["estimated_first_epoch_read_seconds"] / fastest["estimated_epoch_seconds"]
    report["status"] = "done"
    report["elapsed_seconds"] = time.monotonic() - started
    save(args.out, report)


if __name__ == "__main__":
    main()
