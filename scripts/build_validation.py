"""Build or verify a frozen train-matched validation list without executing a notebook."""
import argparse
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def notebook_settings(path, root=ROOT):
    from scripts.g4.preflight import top_values
    from cvdl.settings import load, machine_profile
    values = top_values(json.loads(Path(path).read_text()))
    # Resolve full training settings, not the PC QUICK reduction. No hardware probe.
    values.pop('validation_recipe', None)
    if values.get('checkpoint_validation') == 'train_matched':
        values['checkpoint_validation'] = 'combined'
    return load(environ={}, _profile=machine_profile('pc', root=root, device='cpu', precision_limits=('32',)),
                overrides={'quick': False}, **values)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--notebook', type=Path, required=True)
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    from cvdl.validation import build
    start = time.monotonic()
    settings = notebook_settings(args.notebook)
    rid, files, summary = build(ROOT, settings.to_run_settings(), settings.train_grid, settings.train_tile_size)
    folder = ROOT/'data/val'/rid
    if args.check or folder.exists():
        for name, text in files.items():
            if not (folder/name).is_file() or (folder/name).read_bytes() != text.encode():
                raise ValueError(f'{folder/name}: rebuild differs or file missing; never overwrite a frozen recipe')
        state = 'REBUILD IDENTICAL'
    else:
        folder.mkdir(parents=True)
        for name, text in files.items():
            (folder/name).write_bytes(text.encode())
        state = 'BUILT'
    print(json.dumps(dict(state=state, seconds=round(time.monotonic()-start, 3), **summary), indent=2))


if __name__ == '__main__':
    main()
