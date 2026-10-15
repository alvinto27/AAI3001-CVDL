#!/usr/bin/env python3
"""Build new-source photo and split records from frozen selection CSVs.

Write only NEW files at --output-root/data/{alaska,lsdir}/. Masters stay untouched.
For a dry run, pass a git-ignored output root and a small selection CSV.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import sys

from build_manifests import PHOTO_FIELDS, SPLIT_FIELDS

SELECTION_FIELDS = {
    'alaska': {'alaska_id','camera_group','camera_model','split','block','width','height',
               'tiles512','bytes','sha256','in_v1','url'},
    'lsdir': {'lsdir_id','shard','path','split','width','height','tiles512','bytes','sha256'},
}
SPLITS = {'alaska': {'train','val','test_seen','test_unseen'},
          'lsdir': {'train','val','test'}}


def _positive(value, name):
    if not value.isdecimal() or int(value) < 1:
        raise ValueError(f'{name} must be a positive integer: {value!r}')
    return int(value)


def _relative(value):
    path=PurePosixPath(value)
    if not value or path.is_absolute() or '..' in path.parts or '\\' in value:
        raise ValueError(f'unsafe relative path {value!r}')
    return path.as_posix()


def _manifest(source, selection):
    if not SELECTION_FIELDS[source] <= set(selection[0]):
        raise ValueError(f'{source} selection misses columns: {sorted(SELECTION_FIELDS[source]-set(selection[0]))}')
    result=[];seen=set();paths=set();digests=set();provenance=[]
    key='alaska_id' if source=='alaska' else 'path'
    for row in sorted(selection,key=lambda item:item[key]):
        if row['split'] not in SPLITS[source]:
            raise ValueError(f'invalid {source} split: {row["split"]}')
        h,w=_positive(row['height'],'height'),_positive(row['width'],'width')
        if min(h,w)<512:
            raise ValueError('selected photo has no 512 tile')
        rows,cols=h//512,w//512
        if _positive(row['tiles512'],'tiles512') != rows*cols:
            raise ValueError('selection tiles512 disagrees with trim512 dimensions')
        size=_positive(row['bytes'],'bytes');digest=row['sha256'].lower()
        if not re.fullmatch(r'[0-9a-f]{64}',digest):
            raise ValueError('invalid SHA-256')
        if digest in digests:
            raise ValueError('duplicate SHA-256 across selected photos or splits')
        digests.add(digest)
        if source=='alaska':
            ident=row['alaska_id']
            if not re.fullmatch(r'[0-9]{5}',ident):raise ValueError(f'invalid Alaska ID: {ident}')
            photo_id=f'alaska_{ident}'
            master=f'Datasets/alaska/tiff_various_color/{ident}.tif'
            camera=row['camera_model']
            if not camera or row['in_v1'] not in {'0','1'} or not row['url'].startswith('https://'):
                raise ValueError('invalid ALASKA camera, in_v1, or URL')
            kind='tiff_rgb8'
        else:
            ident=_relative(row['path'])
            if not ident.endswith('.png') or not re.fullmatch(r'shard-\d+',row['shard']):
                raise ValueError('invalid LSDIR member/shard')
            photo_id='lsdir_'+ident.removesuffix('.png').replace('/','_')
            master=f'Datasets/lsdir/hr/{ident}'
            if not re.fullmatch(r'[A-Za-z0-9_.-]+',row['lsdir_id']):
                raise ValueError('invalid LSDIR selection ID')
            camera='';kind='png8'
        if photo_id in seen or master in paths:
            raise ValueError(f'duplicate source photo: {photo_id}')
        seen.add(photo_id);paths.add(master)
        item=dict(source=source,source_type=kind,photo_id=photo_id,
                  photo_index=(100000 if source=='alaska' else 300000)+len(result),
                  master_path=master,camera_make='',camera_model=camera,
                  source_height=h,source_width=w,noise='not_applicable',
                  grid_policy='trim512',overlap512_rows=rows,overlap512_cols=cols,
                  overlap512_y=';'.join(str(i*512) for i in range(rows)),
                  overlap512_x=';'.join(str(i*512) for i in range(cols)),
                  overlap256_rows='',overlap256_cols='',overlap256_y='',overlap256_x='',
                  legacy512_policy='',legacy512_rows='',legacy512_cols='',
                  legacy512_height='',legacy512_width='',split=row['split'])
        result.append(item)
        provenance.append(dict(photo_id=photo_id,master_path=master,split=row['split'],
                               bytes=size,sha256=digest,**{key:row[key] for key in row if key not in {'split','bytes','sha256'}}))
    return result,provenance


def build(source, selection_path, output_root, repo_root, verify_files=True):
    with Path(selection_path).open(newline='',encoding='utf-8') as stream:
        reader=csv.DictReader(stream)
        if not reader.fieldnames or not SELECTION_FIELDS[source] <= set(reader.fieldnames):
            raise ValueError('selection header is missing required fields')
        rows=list(reader)
    if not rows or any(None in row.values() for row in rows):raise ValueError('empty or malformed selection')
    photos,provenance=_manifest(source,rows)
    if verify_files:
        for detail in provenance:
            p=Path(repo_root)/detail['master_path']
            if not p.is_file() or p.stat().st_size!=detail['bytes'] or hashlib.sha256(p.read_bytes()).hexdigest()!=detail['sha256']:
                raise ValueError(f'missing/mismatched master: {p}')
    root=Path(output_root)/'data'/source
    outputs={root/'photo_manifest.csv':(PHOTO_FIELDS,photos),
             root/'split.csv':(SPLIT_FIELDS,[{key:p[key] for key in SPLIT_FIELDS} for p in photos]),
             root/'provenance.csv':(tuple(provenance[0]),provenance)}
    if any(path.exists() for path in outputs):raise ValueError('output exists; use a new root')
    created=[]
    try:
        root.mkdir(parents=True,exist_ok=True)
        for path,(fields,records) in outputs.items():
            with path.open('x',newline='',encoding='utf-8') as stream:
                created.append(path);writer=csv.DictWriter(stream,fieldnames=fields,lineterminator='\n')
                writer.writeheader();writer.writerows(records)
        report=dict(source=source,selection_sha256=hashlib.sha256(Path(selection_path).read_bytes()).hexdigest(),
                    photos=len(photos),tiles512=sum(int(r['overlap512_rows'])*int(r['overlap512_cols']) for r in photos),
                    splits=dict(Counter(p['split'] for p in photos)),verified_masters=verify_files,
                    decoder_policy='stored RGB8, no ICC conversion, no resize',grid_policy='trim512')
        with (root/'build_record.json').open('x',encoding='utf-8') as stream:
            created.append(root/'build_record.json');json.dump(report,stream,indent=2);stream.write('\n')
    except Exception:
        for path in created:path.unlink(missing_ok=True)
        raise
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',choices=('alaska','lsdir'),required=True)
    parser.add_argument('--selection',type=Path,required=True)
    parser.add_argument('--output-root',type=Path,required=True)
    parser.add_argument('--repo-root',type=Path,default=Path(__file__).resolve().parents[1])
    parser.add_argument('--no-verify-files',action='store_true',help='dry run only; do not use for final manifest')
    args=parser.parse_args()
    print(json.dumps(build(args.source,args.selection,args.output_root,args.repo_root,
                           not args.no_verify_files),indent=2))
