#!/usr/bin/env python3
"""Bounded batch16 GAP+covariance correctness/timing and branch-balance probe."""
import argparse
import gc
import json
from pathlib import Path
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
TOLERANCE=1e-4  # Frozen max-absolute-error / max-reference-magnitude, per tensor.


def relative_error(actual,reference):
    import torch
    error=(actual.detach().cpu().double()-reference.detach().cpu().double()).abs().max().item()
    scale=reference.detach().abs().max().item()
    ratio=error/scale if scale else (0. if error==0 else float('inf'))
    if not ratio<=TOLERANCE:raise ValueError(f'FP64 reference failed: {ratio} > {TOLERANCE}')
    return ratio


def fp64_head(features,reduction,weight,bias):
    """Independent functional reference; not the production FP32 wrapper."""
    import torch
    from torch.nn import functional as F
    gap=features.mean((-2,-1));h=F.conv2d(features,reduction).flatten(2)
    u=h-h.mean(-1,keepdim=True);c=u@u.transpose(1,2)/h.shape[-1]
    eye=torch.eye(32,dtype=features.dtype,device=features.device)[None]
    c=(c+c.transpose(1,2))/2
    c=c+(1e-4*c.diagonal(dim1=-2,dim2=-1).sum(-1)/32+1e-12)[:,None,None]*eye
    trace=c.diagonal(dim1=-2,dim2=-1).sum(-1)[:,None,None]
    y=c/trace;z=eye.expand_as(c)
    for _ in range(5):
        update=(3*eye-z@y)/2
        old_y,old_z=y,z;y=old_y@update;z=update@old_z
    root=(y+y.transpose(1,2))*trace.sqrt()/2
    i,j=torch.triu_indices(32,32,device=features.device)
    cov=root[:,i,j]
    return F.linear(torch.cat((gap,cov),1),weight,bias),cov,c,root


def reference_check():
    import torch
    from cvdl.models import XuNetWideGAPCov32
    results=[];torch.manual_seed(20261026)
    for kind,n,side in (('random',2,8),('random',1,16),('rank1',3,8),('zero',1,8),('random',16,8)):
        net=XuNetWideGAPCov32().cuda()
        x=torch.randn(n,256,side,side)
        if kind=='rank1':x=torch.randn(n,256,1,1)*torch.randn(n,1,side,side)
        if kind=='zero':x.zero_()
        f=x.double().requires_grad_()
        w=net.cov_reduce.weight.detach().cpu().double().requires_grad_()
        cw=net.classifier.weight.detach().cpu().double().requires_grad_()
        cb=net.classifier.bias.detach().cpu().double().requires_grad_()
        ref,rcov,c,root=fp64_head(f,w,cw,cb);dy=torch.randn_like(ref).float()
        gradients=torch.autograd.grad(ref,(f,w,cw,cb),dy.double())
        actual_f=x.cuda().requires_grad_();gap,cov=net.pooled_parts(actual_f)
        actual=net.classifier(torch.cat((gap,cov),1))
        grads=torch.autograd.grad(actual,(actual_f,net.cov_reduce.weight,net.classifier.weight,net.classifier.bias),dy.cuda())
        item=dict(kind=kind,shape=list(x.shape),forward=relative_error(actual,ref),cov_descriptor=relative_error(cov,rcov),
            dx=relative_error(grads[0],gradients[0]),reduce_dw=relative_error(grads[1],gradients[1]),
            fc_gap_dw=relative_error(grads[2][:,:256],gradients[2][:,:256]),
            fc_cov_dw=relative_error(grads[2][:,256:],gradients[2][:,256:]),db=relative_error(grads[3],gradients[3]))
        with torch.no_grad():
            vals,vec=torch.linalg.eigh(c)
            exact=(vec*vals.clamp_min(0).sqrt()[:,None,:])@vec.transpose(1,2)
            item['approximation_relative_to_true_sqrt']=float((root-exact).norm()/exact.norm())
            item['root_squared_residual']=float((root@root-c).norm()/c.norm())
        results.append(item)
        del net,actual_f,actual,grads;gc.collect();torch.cuda.empty_cache()
    return results


def one_model(name,output):
    import torch
    from cvdl import runtime
    from cvdl.settings import load,machine_profile,plain
    from scripts.g4.preflight import top_values
    from scripts.pc_safety import available_gb
    values=top_values(json.loads((ROOT/'notebooks/experiments/srm-e50-paired-gapcov32.ipynb').read_text()))
    values.update(name='gapcov32-pc-'+name,model=name,pilot=True,batch_size=16,
        checkpoint_validation='train_matched',validation_recipe='60da4391043e8ce48ff2',post_fit_eval='none')
    s=load(environ={},_profile=machine_profile('pc',device='cuda',rocm=True),
        overrides=['workers=4','eval_workers=4','prime_next_epoch=false','compile=off',
            'output_root='+str(output/'checkpoints')],**values)
    runtime.configure(s);otf,tables=runtime.create_data(s);loaders,_,_=runtime.create_loaders(s,otf,tables)
    bundle=runtime.create_training(s,otf,loaders)
    module=bundle.lightning_model.cuda().train();module.log=lambda *a,**k:None
    # Training_step and optimizer are the shared Lightning implementation.
    configured=module.configure_optimizers();optimizer=configured['optimizer'];scheduler=configured['lr_scheduler']['scheduler']
    iterator=iter(loaders.train);batch=next(iterator)
    assert batch[0].shape==batch[1].shape==(8,3,256,256)
    batch=module.transfer_batch_to_device(batch,torch.device('cuda'),0)
    from cvdl.cov_pool import branch_statistics
    paired_images=torch.cat(batch[:2]).float()
    branches_start=branch_statistics(module.net,paired_images) if name=='xunet_wide_gapcov32' else None
    frozen=module.net.kv.conv.weight.detach().clone();losses=[];gradient_min={};flags=[];ram_min=available_gb()
    torch.cuda.reset_peak_memory_stats()
    logits_hook=module.net.register_forward_hook(lambda m,a,o:flags.append(torch.isfinite(o).all()))
    def step(batch,index):
        nonlocal ram_min
        ram_min=min(ram_min,available_gb())
        if ram_min<12:raise ValueError('RAM safety floor')
        optimizer.zero_grad(set_to_none=True)
        result=module.training_step(batch,index);loss=result['loss']
        if result['count']!=16:raise ValueError('paired count mismatch')
        flags.append(torch.isfinite(loss.detach()))
        loss.backward()
        for parameter in module.net.parameters():
            if parameter.requires_grad:
                if parameter.grad is None:raise ValueError('missing trainable gradient')
                flags.append(torch.isfinite(parameter.grad).all())
        for layer in (('conv1','conv2','conv7','cov_reduce') if name=='xunet_wide_gapcov32' else ('conv1','conv2','conv7')):
            grad=getattr(module.net,layer).weight.grad
            if grad is None:raise ValueError('missing gradient '+layer)
            flags.append(torch.isfinite(grad).all() & (grad.abs().max()>0))
            magnitude=grad.detach().abs().max()
            gradient_min[layer]=torch.minimum(gradient_min[layer],magnitude) if layer in gradient_min else magnitude
        if module.net.kv.conv.weight.grad is not None:raise ValueError('frozen SRM gradient')
        torch.nn.utils.clip_grad_norm_(module.parameters(),1.)
        optimizer.step();scheduler.step()
        return loss.detach()
    for i in range(32):losses.append(step(batch,i))
    values_ce=torch.stack(losses).cpu().tolist()
    replay_updates=32
    # Replay is warmup. The loader window is 128 new complete batches.
    torch.cuda.synchronize();started=time.monotonic()
    for i in range(128):
        current=module.transfer_batch_to_device(next(iterator),torch.device('cuda'),0)
        step(current,replay_updates+i)
    torch.cuda.synchronize();elapsed=time.monotonic()-started
    if not torch.stack(flags).all().item():raise ValueError('nonfinite loss/gradient or zero gradient')
    gradient_min={k:float(v) for k,v in gradient_min.items()}
    changed=not torch.equal(frozen,module.net.kv.conv.weight)
    if changed:raise ValueError('frozen SRM changed')
    result=dict(model=name,batch_images=16,batch_pairs=8,replay_updates=replay_updates,first8_ce=sum(values_ce[:8])/8,last8_ce=sum(values_ce[-8:])/8,
        loss_decrease=values_ce[-8:]!=[] and sum(values_ce[-8:])<sum(values_ce[:8]),gradient_min_maxabs=gradient_min,
        srm_frozen=not changed,srm_max_abs_movement=float((module.net.kv.conv.weight-frozen).abs().max().detach()),finite=True,warmup_updates=replay_updates,measured_loader_updates=128,seconds=elapsed,steps_per_second=128/elapsed,
        peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),
        minimum_available_ram_gib=ram_min,rocm_conv1_fix_active=s.legacy_globals()['ROCM_CONV1_FIX_ACTIVE'],settings=plain(s.values))
    result['branches_step0']=branches_start
    result['branches_end']=branch_statistics(module.net,paired_images) if name=='xunet_wide_gapcov32' else None
    logits_hook.remove()
    loaders.train.shutdown();loaders.train.loader._iterator=None
    val_iterator=getattr(loaders.val,'_iterator',None)
    if val_iterator is not None:val_iterator._shutdown_workers()
    for logger in bundle.trainer.loggers:
        experiment=getattr(logger,'experiment',None)
        if hasattr(experiment,'close'):experiment.close()
        logger.finalize('success')
    del iterator,module,optimizer,bundle,loaders,otf,batch,configured,scheduler,current;gc.collect();torch.cuda.empty_cache()
    return result


def compiled_check(output):
    import torch
    from cvdl.lit import SteganalysisModule
    from cvdl.standardtest import load_network
    torch.manual_seed(3)
    module=SteganalysisModule(model='xunet_wide_gapcov32',front_end='srm',norm='none',width=4).cuda()
    x=torch.randn(2,3,64,64,device='cuda');target=torch.tensor([0,1],device='cuda')
    y=module.net(x);torch.nn.functional.cross_entropy(y,target).backward()
    gradients={n:p.grad.clone() for n,p in module.net.named_parameters() if p.grad is not None}
    module.zero_grad(set_to_none=True)
    module.net.compile(mode='default',dynamic=False,fullgraph=False)
    start=time.monotonic();cy=module.net(x);torch.nn.functional.cross_entropy(cy,target).backward();torch.cuda.synchronize()
    errors={'logits':relative_error(cy,y)}
    for n,p in module.net.named_parameters():
        if n in gradients:errors[n]=relative_error(p.grad,gradients[n])
    for part,sl in (('gap',slice(None,256)),('cov',slice(256,None))):
        errors['classifier_'+part+'_weight']=relative_error(module.net.classifier.weight.grad[:,sl],gradients['classifier.weight'][:,sl])
    checkpoint=output/'compile-reload.ckpt'
    torch.save(dict(hyper_parameters=dict(module.hparams),state_dict=module.state_dict()),checkpoint)
    copy=load_network(checkpoint,'cuda');reload_y=copy(x)
    relative_error(reload_y,y)
    if copy.kv.conv.weight.requires_grad:raise ValueError('reload unfroze SRM')
    with torch.inference_mode():
        cells,tile=copy.class_score_cells(x)
    relative_error(tile,reload_y)
    if not torch.isfinite(cells).all():raise ValueError('nonfinite attribution')
    checkpoint.unlink()
    return dict(seconds=time.monotonic()-start,relative_errors=errors,reload=True)


def main():
    import torch
    import os
    from scripts.pc_safety import available_gb,check_workers
    if os.environ.get('MIOPEN_GEMM_ENFORCE_BACKEND')=='5':raise ValueError('known-bad ROCm backend5 is forbidden')
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    if args.output.exists():raise ValueError('use a new pilot output')
    check_workers(4)
    if available_gb()<12 or not torch.cuda.is_available():raise ValueError('GPU/RAM safety floor')
    # The operator must inspect process list/VRAM before this bounded job.
    torch.set_num_threads(1);args.output.mkdir(parents=True)
    report=dict(tolerance=TOLERANCE,reference=reference_check(),models=[])
    (args.output/'pilot.json').write_text(json.dumps(report,indent=2)+'\n')
    for name in ('xunet_wide','xunet_wide_gapcov32'):
        report['models'].append(one_model(name,args.output))
        (args.output/'pilot.json').write_text(json.dumps(report,indent=2)+'\n')
    report['compiled']=compiled_check(args.output)
    (args.output/'pilot.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    from scripts.pc_safety import guarded,own_process_group
    own_process_group();guarded(main)
