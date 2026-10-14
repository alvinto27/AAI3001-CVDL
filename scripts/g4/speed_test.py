"""Cold input samples, loader capacity, and notebook-04 GPU speed on G4."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import resource
import subprocess
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
    print(f"Report updated: {path}, state={report['status']}", flush=True)


def read_sample(paths, train_bytes):
    before = io_bytes()
    begin = time.monotonic()
    logical = 0
    for path in paths:
        with path.open("rb") as stream:
            while data := stream.read(8 * 1024 * 1024):
                logical += len(data)
    elapsed = time.monotonic() - begin
    physical = io_bytes() - before
    rate = logical / elapsed / 1e6
    return {"photos": len(paths), "bytes": logical, "physical_read_bytes": physical,
            "likely_cold": physical >= 0.8 * logical,
            "seconds": elapsed, "logical_MB_per_second": rate,
            "estimated_first_epoch_read_seconds": train_bytes / (rate * 1e6)}


def cpu_ticks(pid):
    try:
        # The command name is in parentheses and can itself contain spaces.
        fields = Path(f"/proc/{pid}/stat").read_text().rsplit(") ", 1)[1].split()
        return int(fields[11]) + int(fields[12])  # fields 14 and 15: user + system
    except (OSError, IndexError, ValueError):
        return 0


def loader_capacity(train, workers, seconds, warmup, source):
    loader = training_loader(train, batch_size=32, num_workers=workers,
                             pin_memory=source == "cache", prefetch_images=128)
    fetched = 0
    try:
        iterator = iter(loader)
        for _ in range(warmup):
            next(iterator)
        worker_pids = [worker.pid for worker in loader.loader._iterator._workers]
        before = sum(cpu_ticks(pid) for pid in worker_pids)
        main_before = time.process_time()
        begin = time.monotonic()
        exhausted = False
        while time.monotonic() - begin < seconds:
            try:
                images, _, _ = next(iterator)
            except StopIteration:
                exhausted = True  # The whole epoch was faster than the timing window.
                break
            fetched += images.shape[0]
        duration = time.monotonic() - begin
        cpu_seconds = ((sum(cpu_ticks(pid) for pid in worker_pids) - before) /
                       os.sysconf("SC_CLK_TCK") + time.process_time() - main_before)
        result = {"source": source, "workers": workers, "warmup_batches": warmup,
                  "timed_images": fetched, "timed_seconds": duration,
                  "epoch_exhausted": exhausted,
                  "loader_images_per_second": fetched / duration,
                  "cpu_core_equivalents": cpu_seconds / duration,
                  "cpu_percent_of_machine": 100 * cpu_seconds / duration / (os.cpu_count() or 1)}
        print(json.dumps(result), flush=True)
        return result
    finally:
        loader.shutdown()
        loader.loader._iterator = None


def model_class():
    # Use the model definition from notebook 04, not a second model copy.
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
    return namespace["XuNet"]


def train_measure(train, XuNet, mode, workers, warmup, steps, gpu_only=False):
    loader = None
    if not gpu_only:
        loader = training_loader(train, batch_size=32, num_workers=workers,
                                 pin_memory=True, prefetch_images=128)
    model = XuNet().cuda().train()
    if mode == "compiled":
        model.compile(mode="default", dynamic=False, fullgraph=False)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    begin = None
    fetched = 0
    try:
        if gpu_only:
            fixed_images = torch.rand(32, 3, 512, 512, device="cuda") * 255
            fixed_labels = torch.arange(32, device="cuda") % 2
        else:
            iterator = iter(loader)
        for index in range(warmup + steps):
            if index == warmup:
                torch.cuda.synchronize()
                begin = time.monotonic()
            if gpu_only:
                images, labels = fixed_images, fixed_labels
            else:
                byte_images, byte_labels, _ = next(iterator)
                images = byte_images.cuda(non_blocking=True).float()
                labels = byte_labels.cuda(non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(images), labels)
            loss.backward()
            optimizer.step()
            if index >= warmup:
                fetched += images.shape[0]
        torch.cuda.synchronize()
        if begin is None or fetched == 0:
            raise RuntimeError("Loader did not provide enough batches")
        duration = time.monotonic() - begin
        result = {"mode": mode, "workers": workers if not gpu_only else None,
                  "gpu_only": gpu_only, "warmup_steps": warmup, "steady_steps": steps,
                  "steady_images": fetched, "steady_seconds": duration,
                  "steady_images_per_second": fetched / duration}
        print(json.dumps(result), flush=True)
        return result
    finally:
        if loader is not None:
            loader.shutdown()
            loader.loader._iterator = None
        del optimizer, model, loader
        torch.cuda.empty_cache()


def raise_open_file_limit():
    """Many loader workers share tensors through file descriptors; 1024 is too few."""
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    wanted = hard if hard != resource.RLIM_INFINITY else 1048576
    if soft < min(wanted, 65536):
        resource.setrlimit(resource.RLIMIT_NOFILE, (min(wanted, 65536), hard))
    print("Open-file limit:", resource.getrlimit(resource.RLIMIT_NOFILE), flush=True)


def main():
    raise_open_file_limit()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--sample-photos", type=int, default=48)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--steps", type=int, default=150)
    parser.add_argument("--budget-seconds", type=int, default=720)
    parser.add_argument("--workers", type=int, nargs="+", default=[4, 8, 16])
    parser.add_argument("--decode-workers", type=int, nargs="+", default=[16, 24, 32, 40])
    parser.add_argument("--decode-seconds", type=float, default=40)
    parser.add_argument("--cache-loader-seconds", type=float, default=15)
    parser.add_argument("--loader-warmup", type=int, default=2)
    parser.add_argument("--modes", nargs="+", choices=["eager", "compiled"], default=["eager", "compiled"])
    parser.add_argument("--skip-master-cache-drop", action="store_true",
                        help="PC smoke tests only: master sample will NOT be cold")
    args = parser.parse_args()
    if args.sample_photos < 1 or args.steps < 1 or args.warmup < 0 or args.loader_warmup < 1 or \
       args.decode_seconds <= 0 or args.cache_loader_seconds <= 0 or min(*args.workers, *args.decode_workers) < 1:
        parser.error("positive counts and durations are required")
    cache = args.cache_root.resolve()
    report = {"cache_root": str(cache), "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none",
              "note": "First-pass samples, not full cold epochs; epoch estimates are lower bounds if disk, CPU and GPU overlap.",
              "measurements": [], "gpu_only": [], "cache_loader_only": [], "decode_loader_only": [], "status": "running"}
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
    masters = [ROOT / row["master_path"] for row in schedule]
    master_bytes = sum(path.stat().st_size for path in masters)
    report.update(train_photos=len(schedule), train_cache_bytes=train_bytes,
                  train_master_bytes=master_bytes, train_images=len(train))
    report["cache_first_pass"] = read_sample(
        [cache / f"{row['photo_id']}.npy" for row in schedule[:args.sample_photos]], train_bytes)
    print("Cache first-pass:", json.dumps(report["cache_first_pass"]), flush=True)
    save(args.out, report)

    if not args.skip_master_cache_drop:
        print("Drop the page cache again before reading master files", flush=True)
        subprocess.run(["sudo", "-n", "sh", "-c", "sync; echo 3 > /proc/sys/vm/drop_caches"], check=True)
    else:
        report["note"] += " PC smoke mode: master page cache was not dropped."
    report["master_first_pass"] = read_sample(masters[:args.sample_photos], master_bytes)
    print("Master first-pass:", json.dumps(report["master_first_pass"]), flush=True)
    if not report["cache_first_pass"]["likely_cold"] or not report["master_first_pass"]["likely_cold"]:
        report["note"] += " WARNING: physical reads <80% of sample; cold-time estimates are not trustworthy."
    save(args.out, report)

    decode_settings = RunSettings(source="decode", train_photo_fraction=1.0,
                                  paired_batches=False, tensor_dtype="uint8")
    decode_corpus = Corpus(ROOT, decode_settings)
    if [row["photo_id"] for row in decode_corpus.selected] != [row["photo_id"] for row in corpus.selected]:
        raise RuntimeError("Cache and decode training photo selections differ")
    decode_train = TrainingImages(decode_corpus)
    for source, dataset, worker_counts, seconds, name in (
            ("cache", train, args.workers, args.cache_loader_seconds, "cache_loader_only"),
            ("decode", decode_train, args.decode_workers, args.decode_seconds, "decode_loader_only")):
        for workers in worker_counts:
            if time.monotonic() - started > args.budget_seconds:
                report["status"] = "budget_exceeded"
                save(args.out, report)
                return
            report[name].append(loader_capacity(dataset, workers, seconds, args.loader_warmup, source))
            save(args.out, report)
    XuNet = model_class()
    for mode in args.modes:
        for workers in args.workers:
            if time.monotonic() - started > args.budget_seconds:
                report["status"] = "budget_exceeded"
                save(args.out, report)
                return
            report["measurements"].append(train_measure(train, XuNet, mode, workers, args.warmup, args.steps))
            save(args.out, report)
        report["gpu_only"].append(train_measure(train, XuNet, mode, 0, args.warmup, args.steps, gpu_only=True))
        save(args.out, report)
    cache_best = max(report["cache_loader_only"], key=lambda row: row["loader_images_per_second"])
    decode_best = max(report["decode_loader_only"], key=lambda row: row["loader_images_per_second"])
    gpu_best = max(report["gpu_only"], key=lambda row: row["steady_images_per_second"])
    combined_best = max(report["measurements"], key=lambda row: row["steady_images_per_second"])
    gpu_rate = gpu_best["steady_images_per_second"]
    report["summary"] = {
        "gpu_only_best": gpu_best, "loader_plus_gpu_best": combined_best,
        "steady_gpu_epoch_seconds": report["train_images"] / gpu_rate,
        "cache": {"best_loader": cache_best,
                  "loader_to_gpu_rate": cache_best["loader_images_per_second"] / gpu_rate,
                  "cold_read_seconds": report["cache_first_pass"]["estimated_first_epoch_read_seconds"],
                  "cpu_loader_seconds": report["train_images"] / cache_best["loader_images_per_second"],
                  "estimated_first_epoch_lower_bound_seconds": max(
                      report["cache_first_pass"]["estimated_first_epoch_read_seconds"],
                      report["train_images"] / cache_best["loader_images_per_second"],
                      report["train_images"] / gpu_rate)},
        "decode": {"best_loader": decode_best,
                   "loader_to_gpu_rate": decode_best["loader_images_per_second"] / gpu_rate,
                   "cold_read_seconds": report["master_first_pass"]["estimated_first_epoch_read_seconds"],
                   "cpu_loader_seconds": report["train_images"] / decode_best["loader_images_per_second"],
                   "estimated_first_epoch_lower_bound_seconds": max(
                       report["master_first_pass"]["estimated_first_epoch_read_seconds"],
                       report["train_images"] / decode_best["loader_images_per_second"],
                       report["train_images"] / gpu_rate)}}
    report["status"] = "done"
    report["elapsed_seconds"] = time.monotonic() - started
    save(args.out, report)
    print("Comparison:", json.dumps(report["summary"], indent=2), flush=True)


if __name__ == "__main__":
    main()
