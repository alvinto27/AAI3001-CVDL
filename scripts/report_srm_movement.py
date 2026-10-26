#!/usr/bin/env python3
"""CPU-only best/last SRM movement context; never changes the accuracy rule."""
from pathlib import Path
import argparse
import json
import math
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def movement(initial,current,decay_factor):
    import torch
    initial,current=initial.detach().cpu().double(),current.detach().cpu().double()
    if initial.shape!=(90,1,5,5) or current.shape!=initial.shape or not torch.isfinite(current).all():
        raise ValueError('invalid SRM weights')
    if not math.isfinite(decay_factor) or not 0<=decay_factor<=1:raise ValueError('invalid decay factor')
    filters=[]
    for i,(a,b) in enumerate(zip(initial.flatten(1),current.flatten(1))):
        norm=float(a.norm());new_norm=float(b.norm());delta=b-a
        filters.append(dict(colour=('red','green','blue')[i//30],filter=i%30,
            l2=float(delta.norm()),relative_l2=float(delta.norm())/norm,
            max_abs=float(delta.abs().max()),cosine=float(torch.dot(a,b))/(norm*new_norm) if new_norm else None,
            initial_norm=norm,norm=new_norm,initial_dc_sum=float(a.sum()),dc_sum=float(b.sum()),
            initially_zero=int((a==0).sum()),initially_zero_moved=int(((a==0)&(b!=0)).sum()),
            beyond_decay_l2=float((b-a*decay_factor).norm())))
    colours=current.reshape(3,30,25)
    return dict(decay_only_factor=decay_factor,filters=filters,
        l2=float((current-initial).norm()),relative_l2=float((current-initial).norm()/initial.norm()),
        beyond_decay_l2=float((current-initial*decay_factor).norm()),
        rgb_bank_divergence_l2={name:float((colours[i]-colours[j]).norm())
            for name,i,j in [('red_green',0,1),('red_blue',0,2),('green_blue',1,2)]})


def decay_only(h,steps):
    from cvdl.lit import v4_lr_factor
    return math.prod(1-h['lr']*h['weight_decay']*v4_lr_factor(t,h['steps_per_epoch'],h['max_epochs'],
        h['warmup_epochs'],h['lr_drops']) for t in range(steps))


def report(evaluation,output):
    import torch
    from cvdl.models import SRMFilter
    from scripts.training_decision import (sha,validate_evaluation,learn_srm_identity,WISER_REFERENCE,
        evaluation_inputs,LEARN_APPROVAL)
    evaluation,output=Path(evaluation).resolve(),Path(output).resolve()
    if output.exists() or evaluation in output.parents:raise ValueError('use a new output folder outside the evaluation')
    _,meta,_=validate_evaluation(evaluation)
    reference=ROOT/WISER_REFERENCE
    epoch,best=learn_srm_identity([reference,evaluation],
        [json.loads((reference/'evaluation_settings.json').read_text()),meta])
    if best.parent in output.parents:raise ValueError('output must not modify original training folder')
    paths=[*evaluation_inputs(evaluation),best,best.parent/'last.ckpt',best.parent/'run_settings.json',
        best.parent/'data_run_settings.json',best.parent/'model_run_settings.json',best.parent/'training_start.json',Path(__file__),ROOT/'cvdl/models.py']
    before={str(p):sha(p) for p in paths}
    # Construct only the fixed bank. Do not instantiate a random body or use GPU.
    initial=SRMFilter().conv.weight.detach()
    results={}
    for name,path in [('best',best),('last',best.parent/'last.ckpt')]:
        saved=torch.load(path,map_location='cpu',weights_only=False)
        h=saved['hyper_parameters'];steps=saved['global_step']
        results[name]=dict(checkpoint=str(path),checkpoint_sha256=sha(path),epoch=saved['epoch'],global_step=steps,
            **movement(initial,saved['state_dict']['net.kv.conv.weight'],decay_only(h,steps)))
    if before!={str(p):sha(p) for p in paths}:raise ValueError('report input drift')
    result=dict(context_only=True,main_rule_affected=False,selected_epoch_one_based=epoch,
        approval=LEARN_APPROVAL,inputs_sha256=before,checkpoints=results,
        interpretation='Movement includes AdamW decay. beyond_decay_l2 and initially-zero movement distinguish pure shrinkage. No movement threshold changes the frozen accuracy verdict.')
    output.mkdir(parents=True)
    (output/'report.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evaluation',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();report(args.evaluation,args.output)


if __name__=='__main__':main()
