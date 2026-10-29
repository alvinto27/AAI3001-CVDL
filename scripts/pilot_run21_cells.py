#!/usr/bin/env python3
"""One bounded local GPU correctness probe; not training/evaluation permission."""
import argparse
import copy
import gc
import json
import os
from pathlib import Path
import signal
import sys
import threading
import time
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
TOLERANCE=1e-4  # max absolute error / max absolute eager reference, each tensor
MAX_SECONDS=600
REPLAY_STEPS=32
FRESH_STEPS=8
SOURCE=ROOT/'runs/srm-e50-paired-gapcov32-20261026T115421Z-1917/checkpoints/srm-e50-paired-gapcov32-20261026T115421Z-1917/best-epoch43-val_pe0.0329-val_auc0.9946.ckpt'
SOURCE_SHA='4ba2a4b59a535b53ab9c0235c54231476b45e2193982bf28f7b7f2171d9898cf'


def settings(output,*,device='cuda'):
    from cvdl.settings import load,machine_profile
    from scripts.g4.preflight import top_values
    values=top_values(json.loads((ROOT/'notebooks/experiments/srm-cell-aux-paired-gapcov32.ipynb').read_text()))
    values.update(name='run21-cell-pc-pilot',pilot=True,batch_size=16,post_fit_eval='none',init_checkpoint=str(SOURCE))
    return load(environ={},_profile=machine_profile('pc',device=device,rocm=device=='cuda'),
        overrides=['workers=4','eval_workers=0','persistent_workers=false','prime_next_epoch=false','compile=off',
                   'main_threads=1','output_root='+str(output)],**values)


def aligned_batch(batch):
    import numpy as np
    import torch
    from lsb_cells import changed_cell_targets
    clean,stego,tags=batch
    if clean.shape!=stego.shape or clean.shape[1:]!=(3,256,256) or tags.shape!=(len(clean),2,130):
        raise ValueError('paired cell batch geometry mismatch')
    if not torch.equal(tags[:,1,0],tags[:,0,0]+1) or not (tags[:,:,1]==1).all():raise ValueError('paired IDs/original flags mismatch')
    if tags[:,0,2:66].any() or not (tags[:,0,66:]==1).all():raise ValueError('clean target mismatch')
    changed=(clean!=stego).any(1).cpu().numpy()
    for index,mask in enumerate(changed):
        pixels=np.flatnonzero(mask);counts,valid=changed_cell_targets(pixels*3)
        if not np.array_equal(counts,tags[index,1,2:66].cpu().numpy().reshape(8,8)) or not np.array_equal(valid,tags[index,1,66:].cpu().numpy().reshape(8,8)):
            raise ValueError('actual paired pixel changes and cell tags disagree')
    return dict(pairs=len(clean),images=2*len(clean),changed_pixels=int(changed.sum()),aligned=True)


def relative_error(actual,reference):
    a=actual.detach().double();b=reference.detach().to(device=a.device,dtype=a.dtype)
    error=float((a-b).abs().max());scale=float(b.abs().max())
    ratio=error/scale if scale else (0. if error==0 else float('inf'))
    if ratio>TOLERANCE or not __import__('math').isfinite(ratio):raise ValueError(f'eager/compiled mismatch {ratio} > {TOLERANCE}')
    return ratio


def frozen_equal(reference,current):
    """Lightning teardown can move the module to CPU after a GPU fit."""
    import torch
    return torch.equal(reference.detach().cpu(),current.detach().cpu())


def gradients(module):
    import torch
    observed={}
    for name,p in module.net.named_parameters():
        if p.requires_grad:
            if p.grad is None or not torch.isfinite(p.grad).all():raise ValueError('missing/nonfinite gradient: '+name)
            observed[name]=float(p.grad.abs().max())
    for name in ('conv1.weight','conv7.weight','cov_reduce.weight','classifier.weight','cell_head.weight'):
        if observed[name]<=0:raise ValueError('zero required gradient: '+name)
    if module.net.kv.conv.weight.grad is not None:raise ValueError('SRM received gradient')
    return observed


def continuation_check(module,batch,folder,rebuild,*,device='cuda'):
    """Real Lightning save at step1, continuous step2, then restore+step2."""
    import lightning as L
    import torch
    class SaveFirst(L.Callback):
        def on_train_batch_end(self,trainer,pl_module,outputs,batch,batch_idx):
            if trainer.global_step==1:trainer.save_checkpoint(folder/'last.ckpt')
    class Cached(torch.utils.data.Dataset):
        def __len__(self):return 8
        def __getitem__(self,index):return batch
    loader=torch.utils.data.DataLoader(Cached(),batch_size=None,num_workers=0)
    options=dict(accelerator='gpu' if device=='cuda' else 'cpu',devices=1,precision='32',max_epochs=15,max_steps=2,
        num_sanity_val_steps=0,limit_val_batches=0,logger=False,enable_checkpointing=False,
        enable_progress_bar=False,enable_model_summary=False,gradient_clip_val=1.,gradient_clip_algorithm='norm')
    trainer=L.Trainer(callbacks=[SaveFirst()],**options)
    trainer.fit(module,train_dataloaders=loader)
    expected={k:v.detach().cpu().clone() for k,v in module.state_dict().items()}
    expected_opt=copy.deepcopy(trainer.optimizers[0].state_dict())
    expected_sched=copy.deepcopy(trainer.lr_scheduler_configs[0].scheduler.state_dict())
    saved=torch.load(folder/'last.ckpt',map_location='cpu',weights_only=False)
    if (saved['global_step'],saved['epoch'])!=(1,0):raise ValueError('unexpected saved step/epoch')
    if any('_orig_mod' in key for key in saved['state_dict']):raise ValueError('compiled state prefix leaked')
    restored=rebuild();resumed=L.Trainer(**options)
    resumed.fit(restored,train_dataloaders=loader,ckpt_path=str(folder/'last.ckpt'))
    for key,value in expected.items():torch.testing.assert_close(value,restored.state_dict()[key].cpu(),rtol=0,atol=0)
    def exact(a,b):
        if isinstance(a,torch.Tensor):torch.testing.assert_close(a.cpu(),b.cpu(),rtol=0,atol=0)
        elif isinstance(a,dict):
            if a.keys()!=b.keys():raise ValueError('resume optimizer/scheduler keys differ')
            for key in a:exact(a[key],b[key])
        elif isinstance(a,(list,tuple)):
            if len(a)!=len(b):raise ValueError('resume state length differs')
            for x,y in zip(a,b):exact(x,y)
        elif a!=b:raise ValueError('resume state value differs')
    exact(expected_opt,resumed.optimizers[0].state_dict());exact(expected_sched,resumed.lr_scheduler_configs[0].scheduler.state_dict())
    if resumed.global_step!=2 or trainer.global_step!=2 or resumed.current_epoch!=trainer.current_epoch:
        raise ValueError('resume trainer step/epoch mismatch')
    if saved['hyper_parameters'].get('source_provenance')!=module.hparams.get('source_provenance'):
        raise ValueError('checkpoint source provenance lost')
    return dict(saved_global_step=1,saved_epoch=0,continuous_global_step=trainer.global_step,
        resumed_global_step=resumed.global_step,resumed_epoch=resumed.current_epoch,
        weights_optimizer_scheduler_bit_exact=True,no_orig_mod_prefix=True,
        source_provenance_retained=True,pilot_checkpoint=saved['hyper_parameters'].get('pilot') is True)


def run(output,scratch,report):
    import lightning as L
    import torch
    from cvdl import runtime
    from cvdl.cell_artifacts import sha
    from cvdl.curriculum import weights_only
    from cvdl.cell_diagnostics import cell_maps
    from cvdl.pilot import refuse_run
    from scripts.pc_safety import available_gb,check_workers
    if not torch.cuda.is_available() or available_gb()<12:raise ValueError('GPU/RAM safety floor')
    if os.environ.get('MIOPEN_GEMM_ENFORCE_BACKEND')=='5':raise ValueError('forbidden ROCm backend5')
    check_workers(4);s=settings(scratch/'runs');report['settings']=dict(s.values)
    runtime.configure(s)
    if not runtime.conv1_factory(s) or not s.legacy_globals()['ROCM_CONV1_FIX_ACTIVE']:raise ValueError('required ROCm workaround inactive')
    if sha(SOURCE)!=SOURCE_SHA:raise ValueError('native initializer SHA mismatch')
    otf=None;loaders=None;bundle=None;iterator=None;module=None;configured=None;original_log=None
    try:
        otf,tables=runtime.create_data(s);loaders,_,_=runtime.create_loaders(s,otf,tables)
        bundle=runtime.create_training(s,otf,loaders,trainer_options={'enable_model_summary':False})
        module=bundle.lightning_model;module.hparams['pilot']=True
        source=torch.load(SOURCE,map_location='cpu',weights_only=False)['state_dict']
        for name,value in source.items():torch.testing.assert_close(module.state_dict()[name],value,rtol=0,atol=0)
        if set(module.state_dict())-set(source)!={'net.cell_head.weight','net.cell_head.bias'}:raise ValueError('unexpected new tensors')
        if sum(p.numel() for p in module.net.cell_head.parameters())!=257:raise ValueError('wrong new head size')
        initial={k:v.detach().cpu().clone() for k,v in module.state_dict().items()}
        module.cuda().train()
        configured=module.configure_optimizers();optimizer=configured['optimizer'];scheduler=configured['lr_scheduler']['scheduler']
        if optimizer.state or scheduler.last_epoch!=0:raise ValueError('warm start restored optimizer/scheduler')
        report['initialization']=dict(base_exact=True,head_parameters=257,fresh_optimizer=True,fresh_scheduler=True,
            source_sha256=SOURCE_SHA,source_epoch=43,rocm_conv1_workaround=True)
        iterator=iter(loaders.train);cpu_batch=next(iterator);report['batch']=aligned_batch(cpu_batch)
        if report['batch']['images']!=16:raise ValueError('pilot needs batch16 images')
        batch=module.transfer_batch_to_device(cpu_batch,torch.device('cuda'),0)
        frozen=module.net.kv.conv.weight.detach().clone();logged={};original_log=module.log
        module.log=lambda name,value,**kwargs:logged.__setitem__(name,value.detach())
        torch.cuda.reset_peak_memory_stats();records=[];gradient_min={}
        def update(current,index):
            optimizer.zero_grad(set_to_none=True);logged.clear()
            result=module.training_step(current,index);loss=result['loss']
            if not torch.isfinite(loss) or result['count']!=16:raise ValueError('nonfinite loss or wrong image count')
            loss.backward();observed=gradients(module)
            for key,value in observed.items():gradient_min[key]=min(gradient_min.get(key,value),value)
            torch.nn.utils.clip_grad_norm_(module.parameters(),1.)
            optimizer.step();scheduler.step()
            if not torch.equal(frozen,module.net.kv.conv.weight):raise ValueError('frozen SRM changed')
            return dict(ce=float(logged['train_tile_ce']),bce=float(logged['train_cell_loss']),total=float(loss.detach()))
        replay_start=time.monotonic()
        for i in range(REPLAY_STEPS):records.append(update(batch,i))
        torch.cuda.synchronize()
        report['replay']=dict(updates=REPLAY_STEPS,seconds=time.monotonic()-replay_start,
            first8={k:sum(r[k] for r in records[:8])/8 for k in records[0]},
            last8={k:sum(r[k] for r in records[-8:])/8 for k in records[0]},losses=records,gradient_min_maxabs=gradient_min)
        report['replay']['total_falling']=report['replay']['last8']['total']<report['replay']['first8']['total']
        report['replay']['interpretation']='short replay only; no accuracy claim' if report['replay']['total_falling'] else 'Pending / investigate non-falling short replay'
        print('PILOT replay '+json.dumps(report['replay']),flush=True)
        # Compare an actual return_aux forward AND backward, at the real batch16 geometry.
        optimizer.zero_grad(set_to_none=True);eager=module.net(torch.cat(batch[:2]).float(),return_aux=True)
        module.training_step(batch,0)['loss'].backward()
        eager_grads={n:p.grad.detach().clone() for n,p in module.net.named_parameters() if p.grad is not None}
        optimizer.zero_grad(set_to_none=True);module.net.compile(mode='default',dynamic=False,fullgraph=False)
        started=time.monotonic();compiled=module.net(torch.cat(batch[:2]).float(),return_aux=True)
        module.training_step(batch,0)['loss'].backward();torch.cuda.synchronize()
        errors={'tile':relative_error(compiled[0],eager[0]),'aux':relative_error(compiled[1],eager[1])}
        for name,p in module.net.named_parameters():
            if name in eager_grads:errors[name]=relative_error(p.grad,eager_grads[name])
        report['compiled']=dict(first_forward_backward_seconds=time.monotonic()-started,tolerance=TOLERANCE,relative_errors=errors)
        reload_path=bundle.checkpoint_dir/'compile-reload.ckpt'
        torch.save(dict(hyper_parameters=dict(module.hparams),state_dict=module.state_dict(),**{'pytorch-lightning_version':L.__version__}),reload_path)
        restored=type(module).load_from_checkpoint(reload_path,map_location='cpu',weights_only=False,strict=True,conv1_factory=runtime.conv1_factory(s)).cuda().eval()
        with torch.no_grad():tile,aux=restored.net(torch.cat(batch[:2]).float(),return_aux=True)
        report['compiled'].update(reload_tile_error=relative_error(tile,eager[0]),reload_aux_error=relative_error(aux,eager[1]),
            no_orig_mod_prefix=not any('_orig_mod' in k for k in module.state_dict()))
        torch.cuda.synchronize();started=time.monotonic()
        for i in range(FRESH_STEPS):
            current_cpu=next(iterator);aligned_batch(current_cpu)
            current=module.transfer_batch_to_device(current_cpu,torch.device('cuda'),0);update(current,REPLAY_STEPS+i)
        torch.cuda.synchronize();elapsed=time.monotonic()-started
        report['fresh_window']=dict(updates=FRESH_STEPS,seconds=elapsed,steps_per_second=FRESH_STEPS/elapsed,scope='PC only; no G4 forecast')
        # Inference intentionally uses reconstructed eager model, as deployment does.
        with torch.inference_mode():
            primary,head,prob=cell_maps(restored.net,torch.cat(batch[:2]).float()[:2],include_primary=False)
            signed,dual,dual_prob=cell_maps(restored.net,torch.cat(batch[:2]).float()[:2])
        for value in (head,prob,signed,dual,dual_prob):
            if not torch.isfinite(value).all():raise ValueError('nonfinite photo map')
        if primary is not None:raise ValueError('head-only returned attribution')
        relative_error(head,dual);relative_error(prob,dual_prob)
        report['maps']=dict(head_only_finite=True,dual_finite=True,head_shape=list(head.shape),attribution_shape=list(signed.shape))
        report['gpu_peak']=dict(allocated_bytes=torch.cuda.max_memory_allocated(),reserved_bytes=torch.cuda.max_memory_reserved())
        # Return to the exact initialized pilot state; Lightning now owns fresh AdamW.
        module.net._compiled_call_impl=None;torch._dynamo.reset();module.load_state_dict(initial,strict=True)
        module.log=original_log;original_log=None
        del restored,eager,eager_grads,compiled,current,current_cpu,configured,optimizer,scheduler
        configured=None;gc.collect();torch.cuda.empty_cache()
        # Stop real workers before the cached-batch continuation check.
        loaders.train.shutdown();iterator=None
        def rebuild():
            result=runtime.create_training(s,otf,loaders,trainer_options={'enable_model_summary':False})
            close_loggers(result.trainer);result.lightning_model.hparams['pilot']=True
            return result.lightning_model
        report['resume']=continuation_check(module,cpu_batch,bundle.checkpoint_dir,rebuild)
        try:refuse_run(bundle.checkpoint_dir)
        except ValueError:report['resume']['evaluator_refused_pilot']=True
        else:raise ValueError('evaluator accepted disposable pilot checkpoint')
        if sha(SOURCE)!=SOURCE_SHA:raise ValueError('original initializer changed')
        report['srm_frozen']=frozen_equal(frozen,module.net.kv.conv.weight)
        if not report['srm_frozen']:raise ValueError('SRM changed across continuation check')
        report['status']='correctness checks complete; no accuracy claim' if report['replay']['total_falling'] else 'Pending / non-falling replay'
    finally:
        if module is not None and original_log is not None:module.log=original_log
        if loaders is not None:
            loaders.train.shutdown()
            val_iterator=getattr(loaders.val,'_iterator',None)
            if val_iterator is not None:val_iterator._shutdown_workers()
        if bundle is not None:close_loggers(bundle.trainer)
        try:
            from torch._inductor.async_compile import shutdown_compile_workers
            shutdown_compile_workers();torch._dynamo.reset()
        finally:
            gc.collect();torch.cuda.empty_cache()


def close_loggers(trainer):
    for logger in trainer.loggers:
        experiment=getattr(logger,'experiment',None)
        if hasattr(experiment,'close'):experiment.close()
        logger.finalize('success')


def main():
    from tempfile import TemporaryDirectory
    from scripts.pc_safety import own_process_group,available_gb
    own_process_group()
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    if args.output.exists():raise ValueError('use a new evidence folder')
    args.output.mkdir(parents=True);started=time.monotonic();stop=threading.Event()
    report=dict(started_utc=__import__('datetime').datetime.now(__import__('datetime').timezone.utc).isoformat(),
        max_seconds=MAX_SECONDS,minimum_available_ram_gib=available_gb(),status='running')
    def interrupt(signum,frame):raise RuntimeError('pilot safety stop: RAM floor or 600-second hard limit')
    signal.signal(signal.SIGUSR1,interrupt)
    def watch():
        while not stop.wait(.5):
            have=available_gb();report['minimum_available_ram_gib']=min(report['minimum_available_ram_gib'],have)
            if have<12 or time.monotonic()-started>MAX_SECONDS:os.kill(os.getpid(),signal.SIGUSR1);return
    watcher=threading.Thread(target=watch,daemon=True);watcher.start()
    try:
        import subprocess
        from scripts.test_support import suite_lock
        report['preflight_gpu_processes']=subprocess.check_output(['amd-smi','process'],text=True)
        report['preflight_gpu_memory']=subprocess.check_output(['amd-smi','metric','-u','-m'],text=True)
        if 'No running processes detected' not in report['preflight_gpu_processes']:raise ValueError('GPU already has active processes')
        if available_gb()<12:raise ValueError('preflight available RAM below12GiB')
        with suite_lock(ROOT,'bounded run21-cell GPU pilot',timeout=0),TemporaryDirectory(prefix='.disposable-',dir=args.output) as directory:
            scratch=Path(directory)
            os.environ.update(TORCHINDUCTOR_CACHE_DIR=str(scratch/'inductor'),TRITON_CACHE_DIR=str(scratch/'triton'),
                MIOPEN_CUSTOM_CACHE_DIR=str(scratch/'miopen'),TORCHINDUCTOR_COMPILE_THREADS='1')
            run(args.output,scratch,report)
    except BaseException as error:
        report.update(status='blocked',error=repr(error));raise
    finally:
        stop.set();watcher.join();report['elapsed_seconds']=time.monotonic()-started
        report['disposable_paths_remaining']=[p.name for p in args.output.glob('.disposable-*')]
        (args.output/'evidence.json').write_text(json.dumps(report,indent=2,default=str,allow_nan=False)+'\n')
        print('PILOT RESULT '+json.dumps(report,default=str),flush=True)


if __name__=='__main__':main()
