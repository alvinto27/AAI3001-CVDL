"""Bounded sequential GBRAS benchmark, NOT a training/architecture result.

Every trial is a fresh process using runtime.create_training and Lightning's
optimizer/precision/transfer path. No custom training loop. G4 production bounds
and numerical gates below are frozen; the CPU fixture only tests control flow.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time
import traceback
from types import SimpleNamespace

import lightning as L
from lightning.pytorch.profilers import SimpleProfiler
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from . import runtime
from .lit import SteganalysisModule
from .settings import SCHEMA, MachineProfile, load, plain

MARKER = {'kind': 'benchmark, not a result', 'deployment_eligible': False}
GATES = {'loss_window': 50, 'absolute_loss_gap': .02, 'relative_loss_gap': .05,
         'probability_p99_gap': .01}
CONFIGS = (('fp32-nchw', '32', False), ('bf16-nchw', 'bf16-mixed', False),
           ('bf16-channels-last', 'bf16-mixed', True), ('fp32-channels-last', '32', True))
BUDGET_SECONDS = 40 * 60
RATE = 1.63


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.writing')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temp.replace(path)


def projection(step_seconds, steps_per_epoch, validation_checkpoint_seconds, *,
               epochs=30, fixed_overhead_seconds=900, compile_seconds=0):
    values = (step_seconds, steps_per_epoch, validation_checkpoint_seconds, epochs,
              fixed_overhead_seconds, compile_seconds)
    if any(not math.isfinite(v) or v < 0 for v in values) or min(step_seconds, steps_per_epoch, epochs) <= 0:
        raise ValueError('invalid projection inputs')
    train = epochs * steps_per_epoch * step_seconds
    validation = epochs * validation_checkpoint_seconds
    total = train + validation + fixed_overhead_seconds + compile_seconds
    return dict(training_minutes=train/60, validation_checkpoint_minutes=validation/60,
                fixed_overhead_minutes=fixed_overhead_seconds/60, compile_minutes=compile_seconds/60,
                total_minutes=total/60, dollars=total/3600*RATE, rate_per_hour=RATE,
                fits_180_minutes=total <= 180*60,
                assumptions='One measured full validation+checkpoint per epoch; 15min boot/data/delivery allowance; measured compile excess added separately.')


def compare_numerics(reference, candidate, reference_logits, candidate_logits, mixed_logits, *, window=50):
    """Symmetric loss closeness; same-weight mixed-vs-FP32 probability gate."""
    identity = all(reference[k] == candidate[k] for k in ('initial_sha256', 'batch_sha256', 'lr'))
    a, b = np.asarray(reference['losses']), np.asarray(candidate['losses'])
    arrays = [a, b, np.asarray(reference['gradient_norms']), np.asarray(candidate['gradient_norms']),
              reference_logits, candidate_logits, mixed_logits]
    finite = all(np.isfinite(v).all() for v in arrays) and reference['bn_finite'] and candidate['bn_finite']
    shapes = [np.asarray(z).shape for z in (reference_logits,candidate_logits,mixed_logits)]
    complete = (len(a) == len(b) and len(a) > 0 and shapes[0] == shapes[1] == shapes[2]
                and len(shapes[0]) == 2 and shapes[0][0] > 0 and shapes[0][1] == 2)
    windows = []
    if finite and complete:
        for start in range(0, len(a), window):
            mean = float(a[start:start+window].mean())
            gap = float(abs(b[start:start+window].mean()-mean))
            limit = max(GATES['absolute_loss_gap'], GATES['relative_loss_gap']*abs(mean))
            windows.append(dict(start=start, count=len(a[start:start+window]), fp32_mean=mean, gap=gap, limit=limit, passed=gap <= limit))
    def probability(logits):
        z = np.asarray(logits, dtype=np.float64)
        z = np.exp(z-z.max(axis=1, keepdims=True))
        return z[:, 1]/z.sum(axis=1)
    diagnostics = {}
    if finite and complete:
        delta = np.asarray(candidate_logits)-np.asarray(reference_logits)
        margins = lambda z: np.asarray(z)[:,1]-np.asarray(z)[:,0]
        p = np.abs(probability(mixed_logits)-probability(candidate_logits))
        diagnostics = dict(fp32_checkpoint_logit_rms=float(np.sqrt(np.mean(delta**2))),
            fp32_checkpoint_logit_abs_max=float(np.abs(delta).max()),
            fp32_checkpoint_logit_p99=float(np.quantile(np.abs(delta), .99)),
            fp32_checkpoint_margin_rms=float(np.sqrt(np.mean((margins(candidate_logits)-margins(reference_logits))**2))),
            fp32_checkpoint_margin_abs_max=float(np.abs(margins(candidate_logits)-margins(reference_logits)).max()),
            fp32_checkpoint_margin_p99=float(np.quantile(np.abs(margins(candidate_logits)-margins(reference_logits)),.99)),
            same_weight_probability_p99=float(np.quantile(p, .99)))
    passed = bool(identity and finite and complete and windows and all(w['passed'] for w in windows)
                  and diagnostics['same_weight_probability_p99'] <= GATES['probability_p99_gap'])
    return dict(**MARKER, passed=passed, identity_matched=identity, finite=bool(finite), complete=complete,
                gates=GATES, window_used=window, loss_windows=windows, **diagnostics)


def _tensor_hash(digest, tensor):
    value = tensor.detach().cpu().contiguous()
    digest.update(str((str(value.dtype), tuple(value.shape))).encode())
    digest.update(memoryview(value.numpy()))


def state_hash(module):
    digest = hashlib.sha256()
    for name, value in module.state_dict().items():
        digest.update(name.encode()); _tensor_hash(digest, value)
    return digest.hexdigest()


class AuditedLoader:
    """Hash actual CPU images/labels/tags before transfer, never GPU copies."""
    def __init__(self, loader, limit):
        self.loader, self.limit, self.count = loader, limit, 0
        self.digest = hashlib.sha256()
        self.hash_seconds = 0.
    def __len__(self): return len(self.loader)
    def __getattr__(self, name): return getattr(self.loader, name)
    def __iter__(self):
        for batch in self.loader:
            if self.count < self.limit:
                began = time.perf_counter()
                for tensor in batch: _tensor_hash(self.digest, tensor)
                self.hash_seconds += time.perf_counter()-began
                self.count += 1
            yield batch


def bn_record(module):
    result = {}
    for name, layer in module.named_modules():
        if isinstance(layer, torch.nn.BatchNorm2d):
            row = {k: getattr(layer,k).detach().float().cpu().tolist()
                   for k in ('running_mean','running_var','weight','bias') if getattr(layer,k) is not None}
            row['dtypes'] = {k: str(getattr(layer,k).dtype) for k in ('running_mean','running_var','weight','bias') if getattr(layer,k) is not None}
            row['num_batches_tracked'] = int(layer.num_batches_tracked.cpu())
            result[name] = row
    return result


def bn_finite(record):
    return all(np.isfinite(row[k]).all() for row in record.values()
               for k in ('running_mean','running_var','weight','bias') if k in row)


class Measurement(L.Callback):
    def __init__(self, warm, measured, updates, output, smoke):
        self.warm, self.measured, self.updates = warm, measured, updates
        self.output, self.smoke = output, smoke
        self.losses, self.norms, self.lr, self.events = [], [], [], []
        self.window_seconds = self.first_step_seconds = None
        self.layout = {}
    def sync(self, module):
        if module.device.type == 'cuda': torch.cuda.synchronize()
    def on_train_batch_start(self, trainer, module, batch, batch_idx):
        if batch_idx == 0:
            self.sync(module); self.first_at = time.perf_counter()
            self.layout = dict(shape=list(batch[0].shape), input_dtype=str(batch[0].dtype),
                input_channels_last=batch[0].is_contiguous(memory_format=torch.channels_last),
                input_stride=list(batch[0].stride()),
                weight_channels_last=module.net.body.conv4.weight.is_contiguous(memory_format=torch.channels_last),
                conv4_weight_stride=list(module.net.body.conv4.weight.stride()))
        if batch_idx == self.warm:
            self.sync(module)
            if module.device.type == 'cuda': torch.cuda.reset_peak_memory_stats()
            self.window_at = time.perf_counter()
        if self.warm <= batch_idx < self.warm+self.measured and module.device.type == 'cuda':
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record(); self.events.append((start,end))
        self.lr.append(trainer.optimizers[0].param_groups[0]['lr'])
    def on_before_optimizer_step(self, trainer, module, optimizer):
        gradients = [p.grad for p in module.parameters() if p.grad is not None]
        self.norms.append(torch.nn.utils.get_total_norm(gradients, foreach=True).detach())
    def on_train_batch_end(self, trainer, module, outputs, batch, batch_idx):
        self.losses.append(outputs['loss'].detach().clone())
        self.layout['loss_dtype'] = str(outputs['loss'].dtype)
        if self.warm <= batch_idx < self.warm+self.measured and module.device.type == 'cuda': self.events[-1][1].record()
        if batch_idx == 0:
            self.sync(module); self.first_step_seconds = time.perf_counter()-self.first_at
        if batch_idx == self.warm+self.measured-1:
            self.sync(module); self.window_seconds = time.perf_counter()-self.window_at
        # Finite checks only at compilation/warmup/window/end edges: no per-step GPU sync.
        if batch_idx in {0,self.warm-1,self.warm+self.measured-1,self.updates-1}:
            self.sync(module)
            finite = torch.isfinite(torch.stack(self.losses)).all().item() and torch.isfinite(torch.stack(self.norms)).all().item()
            write_json(self.output/'progress.json',dict(**MARKER,completed_updates=batch_idx+1,finite=finite,
                measured_steps=self.measured if self.window_seconds is not None else 0,
                window_seconds=None if self.smoke else self.window_seconds,dtype_layout=self.layout))
            if not finite: raise FloatingPointError('Nonfinite loss/gradient norm; abort benchmark')
    def on_save_checkpoint(self, trainer, module, checkpoint): checkpoint['benchmark'] = MARKER


class OverheadStart(L.Callback):
    def __init__(self, fixed, partial_pairs):
        self.fixed, self.partial_pairs = fixed, partial_pairs
        self.sanity_seconds = 0.; self.seconds = None; self.score_dtype = None; self.validation_passes = 0
    def on_sanity_check_start(self, trainer, module): self.sanity_at = time.perf_counter()
    def on_sanity_check_end(self, trainer, module):
        # Also warm the short final validation batch: otherwise one cold shape
        # compile in the full pass would be projected thirty times.
        if self.partial_pairs:
            images = self.fixed[:1].repeat(self.partial_pairs,1,1,1)
            batch = module.transfer_batch_to_device((images,images,torch.arange(len(images))),module.device,0)
            prior_mode = module.net.training
            module.net.eval()
            try:
                with torch.inference_mode(), trainer.precision_plugin.forward_context():
                    module.pair_logits_targets(batch)
            finally: module.net.train(prior_mode)
        if module.device.type == 'cuda': torch.cuda.synchronize()
        self.sanity_seconds = time.perf_counter()-self.sanity_at
    def on_validation_start(self, trainer, module):
        if not trainer.sanity_checking:
            if module.device.type == 'cuda': torch.cuda.synchronize()
            self.started = time.perf_counter(); self.validation_passes += 1
    def on_validation_batch_end(self, trainer, module, outputs, batch, batch_idx, dataloader_idx=0):
        if not trainer.sanity_checking: self.score_dtype = str(module.validation_scores[-1].dtype)
    def on_save_checkpoint(self, trainer, module, checkpoint): checkpoint['benchmark'] = MARKER


class OverheadEnd(L.Callback):
    def __init__(self, start): self.start = start
    def on_train_epoch_end(self, trainer, module):
        # All validation callbacks/checkpoints have completed even if Lightning
        # reorders ModelCheckpoint callbacks to the end of its callback list.
        if hasattr(self.start,'started'):
            if module.device.type == 'cuda': torch.cuda.synchronize()
            self.start.seconds = time.perf_counter()-self.start.started


def _settings(request):
    values = request['settings']
    machine = {k:v for k,v in values.items() if SCHEMA[k].machine}
    experiment = {k:v for k,v in values.items() if not SCHEMA[k].machine}
    overrides = dict(precision=request['precision'],channels_last=request['channels_last'],
                     output_root=f"{request['output']}/checkpoints",headless=True,
                     job_id=f"{values['job_id'] or values['name']+'-bench'}-{request['label']}")
    if request['smoke']:
        machine.update(device='cpu', rocm=False, precision_limits=['32','bf16-mixed'])
        overrides.update(workers=0,eval_workers=0,diag_workers=0,compile='off',batch_size=4,
                         data='png',train_yaml=None,train_grid='recorded',train_tile_size=None,
                         checkpoint_validation='standard',validation_recipe=None,source_mix=None,tiles_per_epoch=None)
    else: overrides.update(compile='default')
    return load(environ={}, _profile=MachineProfile(request['profile'], machine), overrides=overrides, **experiment)


def _data(s, smoke, updates):
    if not smoke:
        otf, tables = runtime.create_data(s)
        loaders, _, _ = runtime.create_loaders(s, otf, tables)
        # Fixed balanced source/k pairs: only this tiny audit, NEVER the overhead pass.
        indices = otf.val.frame.groupby(['source','k'], sort=True).head(2).index.tolist()
        pairs = [otf.val[i] for i in indices]
        fixed = torch.stack([x for clean,stego,_ in pairs for x in (clean,stego)])
        return otf, loaders, fixed, otf.val.frame.iloc[indices].pair_id.tolist()
    gen = torch.Generator().manual_seed(20261022)
    images = torch.randint(0,256,(4*(updates+2),3,32,32),generator=gen,dtype=torch.uint8)
    labels = torch.arange(len(images))%2
    train = DataLoader(TensorDataset(images,labels,torch.arange(len(images))),batch_size=4)
    val = DataLoader(TensorDataset(images[:5],images[5:10],torch.arange(5)),batch_size=2)
    return None, SimpleNamespace(train=train,val=val), images[:8], ['synthetic-smoke-only']


def _shutdown(loaders):
    for name in ('train','val','test','train_eval'):
        loader = getattr(loaders,name,None)
        if loader is None: continue
        if hasattr(loader,'shutdown'): loader.shutdown()
        iterator = getattr(loader,'_iterator',None)
        if iterator is not None: iterator._shutdown_workers()


def _inference(checkpoint, fixed, precision, device):
    # Reload both saved checkpoints eagerly in FP32: isolate training drift.
    with torch.random.fork_rng(devices=[]):
        eager = SteganalysisModule.load_from_checkpoint(checkpoint,map_location='cpu',weights_only=False)
    eager.to(device).eval()
    x = eager.transfer_batch_to_device((fixed,torch.zeros(len(fixed)),torch.arange(len(fixed))),torch.device(device),0)[0].float()
    evidence = {}
    hook = eager.net.adapter.register_forward_hook(lambda m,args,out: evidence.update(
        srm_clamp_dtype=str(args[0].dtype),adapter_output_dtype=str(out.dtype)))
    with torch.inference_mode(), torch.autocast(torch.device(device).type, enabled=False): fp32 = eager.net(x).float().cpu().numpy()
    with torch.inference_mode(), torch.autocast(torch.device(device).type,dtype=torch.bfloat16,enabled=precision=='bf16-mixed'):
        mixed = eager.net(x).float().cpu().numpy()
    hook.remove()
    evidence['boundary_probe'] = 'eager same-checkpoint inference; no hooks on compiled training net'
    del eager
    return fp32,mixed,evidence


def _telemetry(stop, samples, phase):
    while not stop.is_set():
        try:
            command = ['nvidia-smi','--query-gpu=memory.used,utilization.gpu','--format=csv,noheader,nounits']
            r = subprocess.run(command,capture_output=True,text=True,timeout=2)
            if r.returncode == 0:
                memory,util = map(float,r.stdout.splitlines()[0].split(','))
                samples.append(dict(monotonic=time.monotonic(),memory_MiB=memory,utilization=util,phase=phase[0]))
        except (OSError,ValueError,subprocess.TimeoutExpired): pass
        stop.wait(1)


def worker(request):
    output = Path(request['output']); output.mkdir(parents=True,exist_ok=True)
    result = dict(**MARKER,state='running',label=request['label'],smoke=request['smoke'],worker_pid=os.getpid())
    write_json(output/'trial.json',result)
    loaders = None; stop = threading.Event(); samples=[]
    phase = ['setup-training']
    monitor = threading.Thread(target=_telemetry,args=(stop,samples,phase),daemon=True)
    if not request['smoke']: monitor.start()
    began = time.perf_counter()
    try:
        s = _settings(request); s.apply_runtime_environment(); runtime.configure(s)
        result['environment'] = dict(torch=str(torch.__version__),lightning=L.__version__,
            cuda=torch.version.cuda,hip=torch.version.hip,cudnn=torch.backends.cudnn.version(),
            device='CPU fixture' if request['smoke'] else torch.cuda.get_device_name(),
            tf32_matmul=torch.backends.cuda.matmul.allow_tf32,tf32_cudnn=torch.backends.cudnn.allow_tf32,
            cudnn_benchmark=torch.backends.cudnn.benchmark)
        warm, measured, updates = (1,2,4) if request['smoke'] else (25,100,200)
        otf, loaders, fixed, ids = _data(s,request['smoke'],updates)
        profiler = SimpleProfiler(dirpath=str(output),filename='profile')
        overhead = request.get('checkpoint')
        options = dict(max_steps=updates+1 if overhead else updates, num_sanity_val_steps=2 if overhead else 0,
                       limit_val_batches=1.0 if overhead else 0, profiler=profiler)
        if overhead: options['val_check_interval'] = 1
        bundle = runtime.create_training(s,otf,loaders,trainer_options=options)
        trainer,module = bundle.trainer,bundle.lightning_model
        write_json(bundle.checkpoint_dir/'benchmark.json',dict(**MARKER,smoke=request['smoke']))
        if overhead:
            start = OverheadStart(fixed,len(loaders.val.dataset)%loaders.val.batch_size)
            trainer.callbacks.insert(0,start); trainer.callbacks.append(OverheadEnd(start))
            # Existing Lightning checkpoint restore ONLY for this bounded overhead trial;
            # no G4 runner/resume logic is added. One untimed update then real validation.
            trainer.fit(module,loaders.train,loaders.val,ckpt_path=request['checkpoint'])
            if start.seconds is None or start.validation_passes!=1 or trainer.global_step!=updates+1:
                raise RuntimeError('exactly one full validation/checkpoint was not reached')
            result.update(state='done',validation_checkpoint_seconds=None if request['smoke'] else start.seconds,
                          validation_warmup_seconds=None if request['smoke'] else start.sanity_seconds,
                          full_validation_pairs=len(otf.val) if otf else len(loaders.val.dataset),
                          validation_probability_dtype=start.score_dtype,full_validation_passes=start.validation_passes,
                          validation_warmed_image_shapes=[2*loaders.val.batch_size]+([2*start.partial_pairs] if start.partial_pairs else []),
                          restored_training_updates=updates,untimed_overhead_priming_updates=1,
                          validation_metric_dtypes={k:str(v.dtype) for k,v in trainer.callback_metrics.items()},
                          timing_note='Includes full validation, best+last checkpoint, normal epoch-end callbacks; two sanity batches warm eval compilation separately.',
                          checkpoint=str(bundle.checkpoint_callback.last_model_path))
        else:
            audit = AuditedLoader(loaders.train,updates)
            measure = Measurement(warm,measured,updates,output,request['smoke'])
            trainer.callbacks = [c for c in trainer.callbacks if c not in (bundle.checkpoint_callback,bundle.curves_callback)]
            trainer.callbacks.append(measure)
            initial = state_hash(module)
            evidence = {}
            original = module.training_logits_targets
            def capture_training_logits(batch):
                # This Python method is OUTSIDE net.compile: do not add forward
                # hooks/side effects that change the compiled training graph.
                logits,targets = original(batch)
                evidence['training_logits_dtype'] = str(logits.dtype)
                return logits,targets
            module.training_logits_targets = capture_training_logits
            trainer.fit(module,audit)
            module.training_logits_targets = original
            if (len(measure.losses)!=updates or len(measure.norms)!=updates or len(measure.lr)!=updates
                    or audit.count!=updates or measure.window_seconds is None):
                raise RuntimeError('incomplete benchmark window')
            stats = bn_record(module)
            if len(stats)!=13 or not bn_finite(stats): raise FloatingPointError('nonfinite or missing BN statistics')
            if (measure.layout['loss_dtype']!='torch.float32'
                    or evidence.get('training_logits_dtype')!=('torch.bfloat16' if s.precision=='bf16-mixed' else 'torch.float32')
                    or not all(p.dtype==torch.float32 for p in module.parameters())
                    or not all(v=='torch.float32' for row in stats.values() for v in row['dtypes'].values())
                    or measure.layout['input_channels_last']!=s.channels_last
                    or measure.layout['weight_channels_last']!=s.channels_last):
                raise RuntimeError('AMP/layout dtype contract failed')
            peaks = {} if request['smoke'] else dict(peak_allocated_GiB=torch.cuda.max_memory_allocated()/2**30,
                                                       peak_reserved_GiB=torch.cuda.max_memory_reserved()/2**30)
            checkpoint = output/'benchmark-last.ckpt'; trainer.save_checkpoint(checkpoint)
            phase[0] = 'fixed-tile-audit'
            fp32,mixed,inference_evidence = _inference(checkpoint,fixed,s.precision,s.device)
            evidence.update(inference_evidence)
            if (evidence['srm_clamp_dtype']!='torch.float32'
                    or evidence['adapter_output_dtype']!=('torch.bfloat16' if s.precision=='bf16-mixed' else 'torch.float32')):
                raise RuntimeError('eager AMP boundary contract failed')
            if not np.isfinite(fp32).all() or not np.isfinite(mixed).all():
                raise FloatingPointError('nonfinite fixed-tile logits')
            np.savez(output/'fixed-logits.npz',fp32=fp32,mixed=mixed)
            wait = next((v for k,v in profiler.recorded_durations.items() if 'train_dataloader_next' in k),[])[warm:warm+measured]
            step = measure.window_seconds/measured
            result.update(state='done',hardware_timing_valid=not request['smoke'],precision=s.precision,channels_last=s.channels_last,
                compile=s.compile,requested_compile='default',warmup_steps=warm,measured_steps=measured,updates=updates,
                steps_per_epoch=len(loaders.train),initial_sha256=initial,batch_sha256=audit.digest.hexdigest(),audited_batches=audit.count,
                fixed_pair_ids=ids,losses=torch.stack(measure.losses).float().cpu().tolist(),
                gradient_norms=torch.stack(measure.norms).float().cpu().tolist(),lr=measure.lr,bn=stats,bn_finite=bn_finite(stats),
                dtype_layout={**measure.layout,**evidence},checkpoint=str(checkpoint),
                checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                parameters_fp32=all(p.dtype==torch.float32 for p in module.parameters()),
                step_seconds=None if request['smoke'] else step,
                steps_per_second=None if request['smoke'] else 1/step,
                images_per_second=None if request['smoke'] else s.batch_size/step,
                gpu_event_seconds_per_step=(sum(a.elapsed_time(b) for a,b in measure.events)/1000/measured if measure.events else None),
                compile_first_step_seconds=None if request['smoke'] else measure.first_step_seconds,
                compile_excess_estimate_seconds=None if request['smoke'] else max(0,measure.first_step_seconds-step),
                compile_note='First training step includes cache-dependent lazy compilation+first execution; excess subtracts warmed step, not an exact compiler-only timer.',
                peak_note='Torch peak reset after warmup, read before fixed-tile audit; NVIDIA also reports whole-trial and pre-audit peaks.',
                loader_wait_mean_seconds=None if request['smoke'] or not wait else float(np.mean(wait)),
                loader_wait_p95_seconds=None if request['smoke'] or not wait else float(np.quantile(wait,.95)),
                hash_cpu_seconds=None if request['smoke'] else audit.hash_seconds,
                validation_recipe=otf.record.get('validation_recipe') if otf else None)
            result.update(peaks)
    except Exception as error:
        text = str(error)
        result.update(state='failed',error_type=type(error).__name__,error=text,traceback=traceback.format_exc(),
                      abort=isinstance(error,(torch.OutOfMemoryError,FloatingPointError)) or 'out of memory' in text.lower())
    finally:
        stop.set()
        if monitor.is_alive(): monitor.join(timeout=3)
        if loaders is not None:
            try: _shutdown(loaders)
            except Exception as error: result['cleanup_error']=str(error)
        if not request['smoke']:
            try:
                from torch._dynamo.utils import compile_times
                result['compiler_timing_report'] = compile_times(repr='str',aggregate=True)
            except (ImportError,TypeError,AttributeError) as error:
                result['compiler_timing_report'] = 'unavailable: '+str(error)
        result.update(wall_seconds=time.perf_counter()-began if not request['smoke'] else None,
                      nvidia_peak_MiB=max((r['memory_MiB'] for r in samples),default=None),
                      nvidia_training_peak_MiB=max((r['memory_MiB'] for r in samples if r['phase']=='setup-training'),default=None),
                      nvidia_sample_interval_seconds=1)
        write_json(output/'nvidia.json',samples)
        write_json(output/'trial.json',result)
    return result


def _stop_process(process):
    if process is None: return
    try: os.killpg(process.pid,signal.SIGTERM)
    except ProcessLookupError: pass
    try: process.wait(timeout=3)
    except subprocess.TimeoutExpired:
        try: os.killpg(process.pid,signal.SIGKILL)
        except ProcessLookupError: pass
        process.wait(timeout=2)
    # A crashed parent can leave loader children in its process group.
    try: os.killpg(process.pid,signal.SIGKILL)
    except ProcessLookupError: pass


def run(settings, output, *, smoke=False, budget_seconds=BUDGET_SECONDS, started_at=None):
    """Notebook coordinator; all GPU work is in one reaped subprocess at a time."""
    if not 0 < budget_seconds <= BUDGET_SECONDS: raise ValueError('budget must be within40min')
    if smoke and settings.profile=='g4': raise ValueError('CPU smoke is forbidden on G4')
    if not smoke and (settings.profile!='g4' or settings.device!='cuda'): raise ValueError('timing is G4-only; PC may run CPU smoke, never hardware timing')
    expected = dict(model='gbras',front_end='srm',norm='batch',clip_t=3.,batch_size=128,
        validation_recipe='60da4391043e8ce48ff2',epochs=30,lr=.0003,warmup_epochs=1,grad_clip=1.,
        lr_drops=((5,.3),(20,.1)),data='otf',source='decode',paired_batches=False,
        tiles_per_epoch=180000,tile_size=512,train_tile_size=256,train_grid='overlap_last',
        checkpoint_validation='train_matched',post_fit_eval='none')
    if (any(getattr(settings,k)!=v for k,v in expected.items())
            or dict(settings.source_mix)!={'raise':.4,'lsdir':.2,'alaska':.4}
            or Path(settings.train_yaml).name!='train-k1to4.yaml'):
        raise ValueError('benchmark requires unchanged real GBRAS128/60da recipe')
    output = Path(output); output.mkdir(parents=True,exist_ok=True)
    if (output/'benchmark-summary.json').exists(): raise ValueError('use a fresh benchmark output')
    deadline = (time.monotonic() if started_at is None else started_at)+budget_seconds
    summary = dict(**MARKER,smoke=smoke,state='running',budget_seconds=budget_seconds,
                   per_configuration_limit_seconds=480,overhead_limit_seconds=600,
                   gates=GATES,trials=[],skipped=[],winner=None,projection=None)
    process = None
    def execute(label,precision,channels_last,checkpoint=None):
        nonlocal process
        folder = output/label
        request = dict(settings=plain(settings.values),profile=settings.profile,output=str(folder.resolve()),
                       label=label,precision=precision,channels_last=channels_last,smoke=smoke,checkpoint=checkpoint)
        write_json(folder/'request.json',request)
        remaining = deadline-time.monotonic()-5
        if remaining<=0: return dict(**MARKER,label=label,state='budget_exhausted')
        print('BENCHMARK START',label,'(not a result)',flush=True)
        with (folder/'console.log').open('w') as log:
            process = subprocess.Popen([sys.executable,'-B','-m','cvdl.gbras_bench','--worker',str((folder/'request.json').resolve())],
                stdout=log,stderr=subprocess.STDOUT,start_new_session=True,
                env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1'})
            try: process.wait(timeout=min(remaining,600 if checkpoint else 480))
            except subprocess.TimeoutExpired:
                _stop_process(process); process=None
                progress=folder/'progress.json'
                return dict(**MARKER,label=label,state='budget_exhausted',
                            partial=json.loads(progress.read_text()) if progress.exists() else None)
            finally:
                _stop_process(process); process=None
        path=folder/'trial.json'
        row = json.loads(path.read_text()) if path.exists() else {}
        if row.get('state') not in {'done','failed'}:
            row = dict(**MARKER,label=label,state='failed',error='worker exited without a final result',abort=True)
        print('BENCHMARK END',label,row['state'],flush=True)
        return row
    try:
        for index,(label,precision,layout) in enumerate(CONFIGS):
            if index==3 and deadline-time.monotonic() < (10 if smoke else 600)+max((r.get('wall_seconds') or 0 for r in summary['trials']),default=0):
                summary['skipped'].append(dict(label=label,reason='reserve remaining budget for full validation')); break
            row = execute(label,precision,layout); summary['trials'].append(row)
            write_json(output/'benchmark-summary.json',summary)
            if row['state']!='done':
                summary['state']=row['state']; break
        completed = [r for r in summary['trials'] if r['state']=='done']
        numerical = dict(**MARKER,gates=GATES,comparisons={})
        if completed:
            ref=completed[0]; ref_logits=np.load(output/ref['label']/'fixed-logits.npz')['fp32']
            for row in completed:
                logits=np.load(output/row['label']/'fixed-logits.npz')
                if row['fixed_pair_ids'] != ref['fixed_pair_ids']: raise RuntimeError('fixed audit tiles changed')
                numerical['comparisons'][row['label']]=compare_numerics(ref,row,ref_logits,logits['fp32'],logits['mixed'],window=2 if smoke else GATES['loss_window'])
            passing=[r for r in completed if numerical['comparisons'][r['label']]['passed']]
            numerical['passed'] = any(r['precision']=='bf16-mixed' for r in passing)
            numerical['passing_configurations'] = [r['label'] for r in passing]
            write_json(output/'numerics.json',numerical)
            if summary['state']=='running' and passing:
                winner=passing[0] if smoke else min(passing,key=lambda r:r['step_seconds'])
                summary['winner']=winner['label']
                overhead=execute('winner-overhead',winner['precision'],winner['channels_last'],winner['checkpoint'])
                summary['overhead']=overhead
                if overhead['state']=='done':
                    summary['state']='smoke-complete' if smoke else 'complete'
                    if not smoke:
                        summary['projection']=projection(winner['step_seconds'],winner['steps_per_epoch'],overhead['validation_checkpoint_seconds'],
                            compile_seconds=winner['compile_excess_estimate_seconds']+overhead['validation_warmup_seconds'])
                else: summary['state']=overhead['state']
            elif summary['state']=='running': summary['state']='no-passing-configuration'
        else: write_json(output/'numerics.json',dict(**MARKER,gates=GATES,passed=False,reason='no completed reference'))
    except BaseException as error:
        summary.update(state='failed',error_type=type(error).__name__,error=str(error))
        if not (output/'numerics.json').exists():
            write_json(output/'numerics.json',dict(**MARKER,passed=False,reason='interrupted before numerical comparison',gates=GATES))
        raise
    finally:
        _stop_process(process)
        if summary['state']=='running': summary['state']='interrupted'
        for temp in output.rglob('*.writing'): temp.unlink(missing_ok=True)
        write_json(output/'benchmark-summary.json',summary)
    return summary


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker',type=Path,required=True)
    args=parser.parse_args()
    result=worker(json.loads(args.worker.read_text()))
    return 0 if result['state']=='done' else 1


if __name__=='__main__': raise SystemExit(main())
