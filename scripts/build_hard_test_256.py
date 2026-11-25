"""Build/check hard256 lists from frozen noise scalars. No decoding, model or GPU."""
import argparse
from pathlib import Path
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from cvdl import hardtest256 as H


def publish(files,folder,*,check=False):
    folder=Path(folder)
    if check:
        for name,text in files.items():
            if not (folder/name).exists() or (folder/name).read_bytes()!=text.encode():
                raise ValueError(f'REBUILD MISMATCH: {name}')
        print('REBUILD IDENTICAL: hard256 four lists, recipe, summary',flush=True)
    else:
        if any((folder/name).exists() for name in files):raise ValueError('refusing to overwrite frozen hard256 lists')
        folder.mkdir(parents=True,exist_ok=True)
        for name,text in files.items():(folder/name).write_text(text)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--check',action='store_true')
    a=p.parse_args();started=time.perf_counter()
    files,summary=H.build(ROOT)
    publish(files,ROOT/H.DATA_DIR,check=a.check)
    print(f"{H.VERSION}: {summary['recipe_id']} {summary['pairs']} pairs; {time.perf_counter()-started:.3f}s",flush=True)
    for name,sha in summary['files_sha256'].items():print(name,sha,flush=True)


if __name__=='__main__':main()
