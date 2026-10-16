#!/usr/bin/env python3
"""Train short independent XuNet configurations; report steady speed, no checkpoints.

Run directly for local tests, or via run_notebook.py with V3_SPEED_MATRIX=on on the G4.
No validation, test, diagnostics, or checkpoint is performed. Priming is
configured in the job env, but cannot start before the first epoch boundary.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'notebooks')]
import cvdl  # exposes the flat LSB library
from lsb_torch import Corpus, RunSettings, TrainingImages, training_loader
from cvdl.models import XuNet


def telemetry(stop, rows, phase):
    if not shutil_which('nvidia-smi'):
        return
    while not stop.is_set():
        try:
            result = subprocess.run(['nvidia-smi', '--query-gpu=utilization.gpu,memory.used,power.draw',
                                     '--format=csv,noheader,nounits'], capture_output=True, text=True, timeout=5)
        except (OSError, subprocess.TimeoutExpired):
            stop.wait(2)
            continue
        if result.returncode == 0:
            try:
                util, memory, watts = (float(s.strip()) for s in result.stdout.splitlines()[0].split(','))
                rows.append(dict(time=time.time(), phase=phase[0], util_pct=util, memory_mib=memory, power_w=watts))
            except (ValueError, IndexError):
                pass
        stop.wait(2)


def shutil_which(name):
    from shutil import which
    return which(name)


def settings():
    def integer(key, default):
        value = int(os.environ.get(key, str(default)))
        if value < 1:
            raise ValueError(f'{key} must be positive')
        return value
    batches = tuple(int(part) for part in os.environ.get('V3_SPEED_BATCHES', '32,64,128').split(','))
    precisions = tuple(os.environ.get('V3_SPEED_PRECISIONS', '32,bf16-mixed').split(','))
    if not batches or not set(batches) <= {32, 64, 128} or len(batches) != len(set(batches)):
        raise ValueError('V3_SPEED_BATCHES must use distinct values from 32,64,128')
    if not precisions or not set(precisions) <= {'32', 'bf16-mixed'} or len(precisions) != len(set(precisions)):
        raise ValueError('V3_SPEED_PRECISIONS must use 32,bf16-mixed')
    if os.environ.get('V3_SOURCE', 'decode') != 'decode' or os.environ.get('V3_DATA', 'otf') != 'otf':
        raise ValueError('speed matrix needs V3_DATA=otf V3_SOURCE=decode')
    if os.environ.get('V3_COMPILE', 'default') not in {'off', 'default'}:
        raise ValueError('V3_COMPILE must be default or off')
    pairs = [(batch, precision) for batch in batches for precision in precisions]
    if os.environ.get('V3_SPEED_PAIRS'):
        pairs = []
        for item in os.environ['V3_SPEED_PAIRS'].split(','):
            parts = item.split(':', 1)
            if len(parts) != 2 or not parts[0].isdigit():
                raise ValueError('V3_SPEED_PAIRS needs batch:precision pairs')
            pair = int(parts[0]), parts[1]
            if pair[0] not in {32, 64, 128} or pair[1] not in {'32', 'bf16-mixed'} or pair in pairs:
                raise ValueError('invalid or repeated V3_SPEED_PAIRS item')
            pairs.append(pair)
    if any(precision == 'bf16-mixed' for _, precision in pairs) and not torch.cuda.is_bf16_supported():
        raise RuntimeError('GPU does not support bf16-mixed')
    return pairs, integer('V3_SPEED_STEPS', 600), integer('V3_SPEED_WARMUP', 50), integer('V3_WORKERS', 32)


def run(output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if not torch.cuda.is_available():
        raise RuntimeError('speed matrix requires a CUDA device (ROCm uses the cuda PyTorch API)')
    pairs, steps, warmup, workers = settings()
    torch.set_num_threads(1)
    torch.backends.cudnn.benchmark = torch.version.cuda is not None
    profile = os.environ.get('V3_TRAIN_YAML')
    corpus = Corpus(ROOT, RunSettings(source='decode', settings_yaml=str((ROOT / profile).resolve()) if profile else None))
    device = torch.device('cuda')
    results = []
    config_start = time.perf_counter()
    for batch_size, precision in pairs:
            label = f'b{batch_size}-{precision}'
            print('CONFIG START', label, flush=True)
            dataset = TrainingImages(corpus)
            loader = training_loader(dataset, batch_size=batch_size, num_workers=workers,
                                     pin_memory=True, persistent_workers=True, prefetch_images=128)
            if len(loader) < warmup + steps:
                raise ValueError(f'{label}: need {warmup + steps} steps but loader has {len(loader)}')
            model = XuNet(clip_t=3.0, norm='none').to(device).train()
            if torch.version.hip is not None:
                from rocm_conv1_wgrad import UnfoldWgradConv2d
                model.conv1 = UnfoldWgradConv2d.from_conv(model.conv1)
            if os.environ.get('V3_COMPILE', 'default') == 'default':
                model.compile(mode='default', dynamic=False, fullgraph=False)
            optimizer = torch.optim.AdamW(model.parameters(), lr=float(os.environ.get('V3_LR','3e-4')),
                                          weight_decay=1e-4)
            wait_s = step_s = 0.
            samples = []
            phase = ['warmup']
            stop = threading.Event()
            monitor = threading.Thread(target=telemetry, args=(stop, samples, phase), daemon=True)
            monitor.start()
            config_wall = time.perf_counter()
            try:
                iterator = iter(loader)
                for step in range(warmup + steps):
                    read_at = time.perf_counter()
                    images, labels, _ = next(iterator)
                    torch.cuda.synchronize()
                    after_read = time.perf_counter()
                    images = images.to(device, non_blocking=True).float()
                    labels = labels.to(device, non_blocking=True)
                    optimizer.zero_grad(set_to_none=True)
                    with torch.autocast('cuda', dtype=torch.bfloat16, enabled=precision == 'bf16-mixed'):
                        logits = model(images)
                        loss = F.cross_entropy(logits, labels)
                    if not torch.isfinite(loss).item():
                        raise FloatingPointError(f'{label} nonfinite loss at step {step}')
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    torch.cuda.synchronize()
                    after_step = time.perf_counter()
                    if step >= warmup:
                        phase[0] = 'measure'
                        wait_s += after_read - read_at
                        step_s += after_step - after_read
                    if step == warmup - 1:
                        torch.cuda.reset_peak_memory_stats()
                    if (step + 1) % 50 == 0:
                        print(label, 'step', step + 1, 'loss', f'{float(loss.detach()):.5f}', flush=True)
                total = wait_s + step_s
                steady_samples = [sample for sample in samples if sample['phase'] == 'measure']
                row = dict(config=label, batch_size=batch_size, precision=precision,
                           compile=os.environ.get('V3_COMPILE', 'default'), workers=workers,
                           warmup_steps=warmup, measured_steps=steps, total_elapsed_s=time.perf_counter()-config_wall,
                           steady_elapsed_s=total, steady_steps_per_s=steps/total,
                           steady_images_per_s=steps*batch_size/total,
                           loader_wait_ms_per_step=wait_s*1000/steps,
                           gpu_and_transfer_ms_per_step=step_s*1000/steps,
                           torch_peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,
                           nvidia_samples=len(steady_samples), nvidia_util_mean_pct=(sum(x['util_pct'] for x in steady_samples)/len(steady_samples) if steady_samples else None),
                           nvidia_memory_max_mib=(max(x['memory_mib'] for x in steady_samples) if steady_samples else None))
                results.append(row)
                print('RESULT',json.dumps(row),flush=True)
            finally:
                stop.set(); monitor.join(timeout=7)
                if samples:
                    with (output / f'{label}-nvidia.csv').open('w',newline='') as f:
                        writer=csv.DictWriter(f,fieldnames=list(samples[0]))
                        writer.writeheader();writer.writerows(samples)
                loader.shutdown()
                del iterator, loader, dataset, optimizer, model
                torch.cuda.empty_cache()
                (output/'speed-matrix.json').write_text(json.dumps(dict(results=results,
                    total_wall_s=time.perf_counter()-config_start, prime_next_epoch_requested=os.environ.get('V3_PRIME_NEXT_EPOCH','off'),
                    prime_exercised=False, note='Each short configuration stays within epoch zero; priming does not run.'),indent=2)+'\n')
    print('MATRIX PASS',len(results),'configs',flush=True)
    return results


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    run(args.output)

if __name__=='__main__':
    main()
