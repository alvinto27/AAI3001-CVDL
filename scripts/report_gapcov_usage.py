#!/usr/bin/env python3
"""Best+last GAP/covariance block RMS on the FULL standard256 VALIDATION list.

No TEST data, thresholds, feature rescaling, ablation or verdict is produced.
"""
from pathlib import Path
import argparse
import gc
import json
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def run(folder,output,*,device='cpu',batch_images=32,workers=4):
    import torch
    from torch.nn import functional as F
    from torch.utils.data import DataLoader
    from cvdl.cov_pool import MODEL,SPEC
    from cvdl.cov_usage import BlockMoments
    from cvdl.deployment import load_recipe
    from cvdl.lit import select_best_checkpoint
    from cvdl.standardtest import load_network
    from cvdl.validation import MatchedValidationPairs
    from scripts.evaluate_run import checkpoint_folder
    from scripts.training_decision import sha,COV_APPROVAL,COV_ATTRIBUTION
    from scripts.pc_safety import available_gb,check_workers
    from lsb_torch import Corpus,RunSettings,PhotoPairTasks,initialize_worker,prefetch_for_images
    folder=checkpoint_folder(folder).resolve();output=Path(output).resolve()
    if batch_images<2 or batch_images>32 or batch_images%2:raise ValueError('batch_images must be even2..32')
    if not 0<=workers<=4:raise ValueError('usage report workers must be0..4')
    check_workers(workers)
    if output.is_relative_to(folder) or output.is_relative_to(ROOT/'data'):raise ValueError('output must not change checkpoint/data folders')
    if output.exists() and any(output.iterdir()):raise ValueError('use a new empty output')
    if available_gb()<12 or (device=='cuda' and not torch.cuda.is_available()):raise ValueError('RAM/GPU safety floor')
    torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.benchmark=torch.version.cuda is not None
    checkpoints=[Path(select_best_checkpoint('',folder)),folder/'last.ckpt']
    if sha(ROOT/'cvdl/models.py')!=COV_APPROVAL['model_source_sha256'] or sha(ROOT/'cvdl/cov_pool.py')!=COV_APPROVAL['math_source_sha256']:
        raise ValueError('usage head source drift')
    rid='60da4391043e8ce48ff2';frame,recipe,summary,_=load_recipe(ROOT,rid)
    if set(frame.split)!={'val'} or recipe['tile_size']!=256 or frame.cover_id.duplicated().any():raise ValueError('expected unique256px validation tiles')
    inputs=[*checkpoints,Path(__file__),ROOT/'cvdl/cov_usage.py',ROOT/'cvdl/models.py',ROOT/'cvdl/cov_pool.py',
        *sorted((ROOT/'data/val'/rid).glob('*')),*[ROOT/'data'/s/'photo_manifest.csv' for s in ('raise','div2k','alaska','lsdir')]]
    before={str(p):sha(p) for p in inputs};results={};start=time.monotonic()
    corpus=Corpus(ROOT,RunSettings(sources=('raise','div2k','alaska','lsdir'),source='decode'))
    dataset=MatchedValidationPairs(corpus,frame);tasks=PhotoPairTasks(dataset)
    kwargs=dict(batch_size=None,num_workers=workers,pin_memory=device=='cuda',worker_init_fn=initialize_worker)
    if workers:kwargs.update(multiprocessing_context='spawn',prefetch_factor=prefetch_for_images(tasks.max_images,128))
    for name,path in zip(('best','last'),checkpoints):
        saved=torch.load(path,map_location='cpu',weights_only=False);h=saved['hyper_parameters']
        if h.get('model')!=MODEL or h.get('cov_pool_identity')!=SPEC:raise ValueError('usage checkpoint head identity mismatch')
        meta=dict(checkpoint=str(path),sha256=sha(path),epoch=saved['epoch'],global_step=saved['global_step']);del saved
        net=load_network(path,device);moments=BlockMoments();seen=set()
        with torch.inference_mode():
            for indices,clean,stego in DataLoader(tasks,**kwargs):
                if available_gb()<12 or time.monotonic()-start>1800:raise ValueError('usage safety limit')
                ids=indices.tolist()
                if seen.intersection(ids):raise ValueError('duplicate validation pair')
                seen.update(ids)
                for lo in range(0,len(indices),batch_images//2):
                    part=slice(lo,lo+batch_images//2)
                    x=torch.cat((clean[part],stego[part])).to(device,dtype=torch.float32)
                    gap,cov=net.pooled_parts(net.features(x))
                    moments.add(F.linear(gap,net.classifier.weight[:,:256]),F.linear(cov,net.classifier.weight[:,256:]))
        if seen!=set(range(len(frame))) or moments.count!=2*len(frame):raise ValueError('incomplete validation usage report')
        results[name]=dict(**meta,**moments.report(),classifier_bias=net.classifier.bias.detach().cpu().tolist())
        del net,x,gap,cov;gc.collect()
        if device=='cuda':torch.cuda.empty_cache()
    after={str(p):sha(p) for p in inputs}
    if before!=after:raise ValueError('usage inputs changed')
    result=dict(context_only=True,main_rule_affected=False,model=MODEL,head_identity=SPEC,checkpoints=results,
        validation=dict(recipe_id=rid,recipe_sha256=summary['recipe_sha256'],pairs_sha256=summary['pairs_sha256'],
            pairs=len(frame),images=2*len(frame),geometry=dict(tile_size=256,grid='overlap_last'),split='val',quick=False),
        device=device,batch_images=batch_images,workers=workers,seconds=time.monotonic()-start,
        software=dict(torch=str(torch.__version__)),input_sha256=before,input_sha256_after=after,inputs_unchanged=True,
        attribution_definition=COV_ATTRIBUTION,
        interpretation='Block RMS shows contribution size/variation, not causal utility or an ablation. No minimum ratio or verdict threshold.')
    output.mkdir(parents=True,exist_ok=True);(output/'report.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    lines=['# GAP+covariance usage — validation context only','',
        'Full frozen standard256 VALIDATION tiles, both clean and stego. No TEST, fitted scale or threshold.',
        'Bias excluded; batch-centred is image-weighted and depends on recorded photo-batch composition.',
        '| Checkpoint | Block | Raw logit RMS | Batch-centred RMS | Dataset-centred RMS |', '|---|---|---:|---:|---:|']
    for name,r in results.items():
        for block,m in r['blocks'].items():lines.append(f'| {name} | {block} | {m["raw_rms"]:.9g} | {m["batch_centered_rms"]:.9g} | {m["dataset_centered_rms"]:.9g} |')
    lines+=['','Nonzero covariance output is not proof of improved detection. No ratio changes the cov_pool verdict.','',
        '## Cell attribution meaning',COV_ATTRIBUTION,'','Counts, input hashes, batch sizes and classifier bias: report.json.']
    (output/'REPORT.md').write_text('\n'.join(lines)+'\n');return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--device',choices=('cpu','cuda'),default='cpu');p.add_argument('--batch-images',type=int,default=32)
    p.add_argument('--workers',type=int,default=4);a=p.parse_args()
    run(a.run,a.output,device=a.device,batch_images=a.batch_images,workers=a.workers)


if __name__=='__main__':
    from scripts.pc_safety import guarded,own_process_group
    own_process_group();guarded(main)
