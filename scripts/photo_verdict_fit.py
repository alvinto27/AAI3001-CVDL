#!/usr/bin/env python3
"""CPU-only validation fit/freeze followed by a separate, single TEST report."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
from cvdl.photo_verdict import CELL_FEATURES, FEATURES, fit_logistic, predict, calibrate_count, evaluate_scope
from cvdl.cell_metrics import fa_cutoff
from scripts.photo_verdict_cache import OUTPUT, dump


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_partition(output,selection,partition):
    # In particular, fit() calls this only for fit/calibration, never TEST.
    rows=[];hashes={}
    for selected in selection['photos']:
        if selected['partition']!=partition:continue
        path=output/'features'/partition/(selected['record']['photo_id']+'.json')
        data=json.loads(path.read_text())
        if data['selection_sha256']!=digest(output/'photo-list.json') or len(data['rows'])!=13:
            raise ValueError('invalid feature-cache identity or row count')
        if [r['variant'] for r in data['rows']]!=list(range(13)):
            raise ValueError('missing/reordered variants')
        if any(r['photo_id']!=selected['record']['photo_id'] or r['partition']!=partition for r in data['rows']):
            raise ValueError('cache row identity mismatch')
        rows.extend(data['rows']);hashes[str(path.relative_to(output))]=digest(path)
    return rows,hashes


def matrix(rows,names):
    return np.asarray([[r['features'][f] for f in names] for r in rows],dtype=np.float64)


def fit(output):
    path=output/'frozen-rules.json'
    if path.exists():raise ValueError('rules are already frozen; refusing any refit')
    selection=json.loads((output/'photo-list.json').read_text())
    training,fit_hashes=load_partition(output,selection,'fit')
    calibration,cal_hashes=load_partition(output,selection,'calibration')
    if len(training)!=224*13 or len(calibration)!=224*13:
        raise ValueError('requires complete approved validation cache')
    negatives=[r for r in calibration if r['label']==0]
    n,fa=calibrate_count([r['counts']['tiles_above'] for r in negatives])
    rules=dict(a=dict(score='number of tile probabilities >= tile_cutoff',N=n,cutoff=n,
        calibration_clean_FA=fa))
    labels=[r['label'] for r in training]
    weights=[.5 if not r['label'] else 1/24 for r in training]
    for key,names in (('b',CELL_FEATURES),('c',FEATURES)):
        model=fit_logistic(matrix(training,names),labels,weights)
        cutoff,fa=fa_cutoff(predict(model,matrix(negatives,names)))
        rules[key]=dict(features=list(names),model=model,cutoff=cutoff,calibration_clean_FA=fa)
    result=dict(version=1,selection_sha256=digest(output/'photo-list.json'),
        cache_identity=json.loads((output/'cache-identity.json').read_text()),
        validation_feature_sha256={**fit_hashes,**cal_hashes},
        code_sha256={p:digest(ROOT/p) for p in ('cvdl/photo_verdict.py','scripts/photo_verdict_cache.py','scripts/photo_verdict_fit.py')},
        fit_photos=224,calibration_photos=224,
        tile_cutoff=selection['tile_cutoff'],head_cutoff=selection['head_cutoff'],rules=rules,
        clean_weight=.5,each_stego_weight=1/24,standardization='fit rows only; population SD; constant scale=1',
        cutoff_rule='float64 score >= cutoff; nextafter(first excluded negative, +inf); clean calibration FA <= .05',
        test_policy='one final TEST scoring; no refit, cut-off change or feature search',
        bootstrap=dict(replicates=2000,seed=20261021,unit='photo',strata='five TEST panels',
            scopes='same full-population draws, then filter scope; keep clean partner and all eligible variants',
            interval='95% percentile; rules/cutoffs fixed; excludes fitting/calibration uncertainty'),
        pass_mark=dict(pooled_c_FA_max=.07,hard_c_PE_max=.25,added_value='paired c-reference P_E 95% upper bound < 0'))
    dump(path,result)
    print('FROZEN '+digest(path),flush=True)


def scoped(rows,predicate):
    positives=[r for r in rows if r['label'] and predicate(r)]
    ids={r['photo_id'] for r in positives}
    return [r for r in rows if not r['label'] and r['photo_id'] in ids]+positives


def score(output):
    rule_path=output/'frozen-rules.json'
    # Freeze guard executes before any TEST-cache file is opened.
    frozen=json.loads(rule_path.read_text())
    frozen_sha=digest(rule_path)
    if (output/'test-results.json').exists() or (output/'REPORT.md').exists():
        raise ValueError('TEST already scored; refusing to rerun')
    if frozen['selection_sha256']!=digest(output/'photo-list.json'):
        raise ValueError('frozen selection identity mismatch')
    for path,sha in frozen['validation_feature_sha256'].items():
        if digest(output/path)!=sha:raise ValueError('validation features changed after freeze')
    selection=json.loads((output/'photo-list.json').read_text())
    rows,test_hashes=load_partition(output,selection,'test')
    if len(rows)!=576*13:raise ValueError('requires complete TEST cache')
    for r in rows:r['a']=r['counts']['tiles_above']
    for key in ('b','c'):
        rule=frozen['rules'][key]
        for r,value in zip(rows,predict(rule['model'],matrix(rows,rule['features']))):r[key]=float(value)
    cutoffs={k:v['cutoff'] for k,v in frozen['rules'].items()}
    population={r['photo_id']:r['panel'] for r in rows}
    scopes={'pooled':rows}
    for panel in ('raise','div2k','alaska-seen','alaska-unseen','lsdir'):
        scopes[panel]=[r for r in rows if r['panel']==panel]
    for placement in ('contiguous','random','stride'):
        scopes['placement/'+placement]=scoped(rows,lambda r:r['placement']==placement)
    for k in range(1,5):scopes[f'k/{k}']=scoped(rows,lambda r:r['k']==k)
    scopes['hard/k1-effective-bpp-.05-.15']=scoped(rows,lambda r:r['k']==1 and .05<=r['embedding']['effective_bpp']<=.15)
    results={}
    for name,pop in scopes.items():
        print('CPU bootstrap '+name,flush=True)
        results[name]=evaluate_scope(pop,cutoffs,population=population) if pop else None
    sizes={}
    for panel in ('raise','div2k','alaska-seen','alaska-unseen','lsdir'):
        clean=[r for r in rows if r['panel']==panel and not r['label']]
        stego=[r for r in rows if r['panel']==panel and r['label']]
        sizes[panel]=dict(photos=len(clean),tiles=sum(r['counts']['tiles'] for r in clean),
            cells=sum(r['counts']['cells'] for r in clean),
            height_range=[min(r['height'] for r in clean),max(r['height'] for r in clean)],
            width_range=[min(r['width'] for r in clean),max(r['width'] for r in clean)],
            clipped_versions=sum(r['embedding']['clipped'] for r in stego),stego_versions=len(stego))
    hard=results['hard/k1-effective-bpp-.05-.15']
    practical=bool(results['pooled']['rules']['c']['FA']<=.07 and hard and hard['rules']['c']['P_E']<=.25)
    added={reference:results['pooled']['differences']['c-'+reference]['P_E']['high']<0 for reference in ('a','b')}
    result=dict(frozen_rules_sha256=frozen_sha,test_feature_sha256=test_hashes,scopes=results,size_breakdown=sizes,
        practical_mark_pass=practical,pooled_added_value=added)
    dump(output/'test-results.json',result)
    # Scores are an auditable final-result artifact, not a new cache for fitting.
    dump(output/'test-scores.json',[{k:r[k] for k in ('photo_id','panel','variant','label','k','placement','a','b','c')} for r in rows])
    report(output,frozen,result)
    print('REPORT COMPLETE '+str(output/'REPORT.md'),flush=True)


def report(output,frozen,result):
    name=frozen['cache_identity'].get('run_name','run 13')
    lines=[f'# Frozen {name}: photo verdict', '',
        '**Practical mark: '+('PASS' if result['practical_mark_pass'] else 'FAIL')+'**', '',
        'Pass requires (c) pooled TEST photo FA ≤7% AND hard-slice P_E ≤25%.',
        'Pooled added value over (a): '+str(result['pooled_added_value']['a'])+'; over (b): '+str(result['pooled_added_value']['b'])+'.',
        'Added value is claimed only where the paired P_E difference 95% upper bound is below zero.', '',
        '## Frozen rule and sample', '',
        f"Photo list SHA256: `{frozen['selection_sha256']}`.",
        f"Frozen rules SHA256: `{result['frozen_rules_sha256']}`.",
        f"Checkpoint SHA256: `{frozen['cache_identity']['checkpoint_sha256']}`.",
        'Validation: source-manifest val, including DIV2K and all LSDIR val; 224 fit + 224 calibration photos.',
        'TEST: 576 photos, each one clean + 12 stego versions. Six viewed demos excluded before sampling.',
        f"Tile cutoff {frozen['tile_cutoff']}; cell-head cutoff {frozen['head_cutoff']}.",
        f"(a) N = {frozen['rules']['a']['N']}; (b) logit cutoff = {frozen['rules']['b']['cutoff']:.17g}; (c) logit cutoff = {frozen['rules']['c']['cutoff']:.17g}.",
        'Logistic rules use fit-only scaling, float64 CPU, fixed L2=.01 excluding intercept; no search.',
        'Clean weight 1/2, each stego weight 1/24. Calibration uses only clean calibration photos and moves ties together.', '',
        '## TEST error and ranking', '',
        'FA, miss and P_E are percentages. AUC is on [0,1]. Each scope counts eligible clean photos once; stego misses and positive AUC weights are averaged within photo first.', '',
        '| Scope | Photos | Stego versions | Rule | FA % | Miss % | P_E % | AUC |',
        '| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: |']
    for name,scope in result['scopes'].items():
        if scope is None:
            lines.append(f'| {name} | 0 | 0 | Ineligible | — | — | — | — |');continue
        for key,m in scope['rules'].items():
            lines.append(f"| {name} | {scope['photos']} | {scope['stego_versions']} | ({key}) | {100*m['FA']:.3f} | {100*m['miss']:.3f} | {100*m['P_E']:.3f} | {m['AUC']:.5f} |")
    lines+=['','## Paired photo-bootstrap differences', '',
        '2,000 paired replicates, seed 20261021, resampled within the five full TEST panels. Each draw keeps a photo’s clean partner and variants, then applies the scope filter. Rules/cutoffs remain fixed. Intervals exclude fitting/calibration uncertainty.', '',
        'P_E differences are percentage points (negative favors c); AUC differences are on [0,1] (positive favors c).', '',
        '| Scope | Difference | P_E difference [95% interval] | AUC difference [95% interval] |',
        '| --- | --- | ---: | ---: |']
    for name,scope in result['scopes'].items():
        if not scope:continue
        for key,difference in scope['differences'].items():
            pe=difference['P_E'];auc=difference['AUC']
            lines.append(f"| {name} | {key} | {100*pe['point']:.3f} [{100*pe['low']:.3f}, {100*pe['high']:.3f}] | {auc['point']:.5f} [{auc['low']:.5f}, {auc['high']:.5f}] |")
    lines+=['','## Source and size breakdown', '',
        '| Panel | Photos | Tiles per clean set | Cells per clean set | Height range | Width range | Clipped / stego versions |',
        '| --- | ---: | ---: | ---: | --- | --- | ---: |']
    for panel,s in result['size_breakdown'].items():
        lines.append(f"| {panel} | {s['photos']} | {s['tiles']} | {s['cells']} | {s['height_range']} | {s['width_range']} | {s['clipped_versions']} / {s['stego_versions']} |")
    lines+=['','## Limits (approved plan section 6)', '',
        '- Validation uses source-manifest val as approved, not the CNN’s 60da list. That list covers 150 RAISE, 1,000 ALASKA and 1,347 LSDIR photos, with no DIV2K.',
        '- Val/TEST photo IDs are disjoint. All 28 ALASKA-seen camera models overlap validation; none of the 8 unseen models does. Seen-camera results are not unseen-camera evidence. A pooled 5% calibration target is not a per-camera guarantee.',
        '- Whole-photo bpp sets total payload, not payload per tile. Local difficulty differs from tile training, especially stride clipping and contiguous bands. Effective bpp and clipping are recorded in the feature cache.',
        '- The CNN and the existing cutoffs had already been selected/evaluated with earlier validation/TEST evidence. Excluding six viewed demos reduces direct reuse, but this is not a wholly untouched benchmark.',
        '- No caps, features, seeds, cutoffs or marks were changed after observing these results. Bootstrap intervals do not include fitting/calibration uncertainty.', '',
        '## Files', '',
        '- `photo-list.json`: pre-inference IDs, partitions and seed keys.',
        '- `cache-identity.json`: checkpoint/list hashes and fixed inference settings.',
        '- `features/{fit,calibration,test}/*.json`: aggregate features and embedding/size metadata; no truth/source/placement/bpp enters the rules.',
        '- `frozen-rules.json`: coefficients, scaling, cutoffs, N and validation-cache hashes; written before opening TEST cache.',
        '- `test-results.json`: full precision metrics, paired intervals, and TEST-cache hashes.',
        '- `test-scores.json`: one final scoring, for audit only.', '']
    (output/'REPORT.md').write_text('\n'.join(lines))


def main():
    import torch
    torch.set_num_threads(1)
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=('fit','score'))
    parser.add_argument('--output',type=Path,default=OUTPUT)
    args=parser.parse_args()
    (fit if args.stage=='fit' else score)(args.output)


if __name__=='__main__':main()
