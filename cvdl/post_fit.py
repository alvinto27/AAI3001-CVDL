"""The shared notebook's post-fit sequence, reusable on the PC after download."""
from pathlib import Path
import json
from . import data, evaluation, diagnostics
from .lit import SteganalysisModule
from .metrics import metrics
from lsb_torch import FixedPairs


def run(settings, otf, tables, loaders, checkpoint, folder):
    s=settings;folder=Path(folder)
    model=SteganalysisModule.load_from_checkpoint(checkpoint,map_location=s.device,weights_only=False)
    model.to(s.device)
    predicted={name:evaluation.predict_pairs(model,loader,device=s.device,data_mode=s.data)
               for name,loader in (('train',loaders.train_eval),('val',loaders.val),('test',loaders.test))}
    scores={name:metrics(frame) for name,frame in predicted.items()}
    for name,score in scores.items():print(name,score,flush=True)
    print(evaluation.per_setting_table(predicted['test'],data_mode=s.data).to_string(index=False))
    path=evaluation.save_predictions(predicted['test'],folder/'test_predictions.csv')
    extras=evaluation.extra_panels(checkpoint,root=s.root,output=folder,panels=s.extra_tests,
        workers=s.eval_workers,device=s.device,quick=s.quick,data_mode=s.data,
        manifest_root=s.manifest_root,source_panels=getattr(otf,'extra_pairs',()),overwrite=True)
    for name,(info,table) in extras.items():
        print(f"EXTRA {name}: {info['pairs']} pairs\n{table.to_string(index=False)}",flush=True)
    evaluation.breakdown(path,labels_path=s.labels_path if s.data=='png' else None,show=False)
    checked={}
    for split in ('train','val'):
        if s.data=='otf':
            base=otf.fixed_train if split=='train' else FixedPairs(otf.corpus,'val',s.eval_profile)
        else:
            base=data.PairDataset(tables.pair_splits.loc[tables.pair_splits['split']==split],s.data_root)
        selected=diagnostics.diagnostic_pairs(base,quick=s.quick,data_mode=s.data,
            workers=s.diag_workers,data_root=s.data_root)
        if otf is not None:
            otf.record.setdefault('diagnostic_pair_ids',{})[split]=selected.frame['cover_id'].tolist()
        checked[split]=diagnostics.batch_dependence(model.net,selected,device=s.device,norm=s.norm,
            split_name=split,quick=s.quick,loader_workers=6 if s.data=='png' else 0)
        print(f'CHECK {split}: {checked[split]}',flush=True)
    if otf is not None:
        otf.save_record(folder/'evaluation_data_run_settings.json')
    result=dict(scores=scores,extra_panels={name:info for name,(info,_) in extras.items()},diagnostics=checked)
    (folder/'evaluation_results.json').write_text(json.dumps(result,indent=2,default=str)+'\n')
    return model,predicted,result
