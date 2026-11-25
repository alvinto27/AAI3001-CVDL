"""Model-independent CPU KV noise on every val/test 256 overlap-last tile."""
from __future__ import annotations
import csv
import hashlib
import json
import multiprocessing as mp
from pathlib import Path
import time

import numpy as np

VERSION = 'kv-noise-256-v1'
DATA_DIR = 'data/hard-test/256-v1'
SOURCES = ('raise', 'div2k', 'alaska', 'lsdir')
COLUMNS = ('cover_id', 'source', 'source_photo', 'camera_model', 'split', 'photo_index',
           'tile_size', 'grid', 'tile_row', 'tile_col', 'y', 'x', 'kv_std')


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def inputs(root):
    out = {}
    for source in SOURCES:
        base = Path(root)/'data'/source
        split = 'split_seed0.csv' if source in ('raise', 'div2k') else 'split.csv'
        out[source] = dict(manifest_sha256=digest((base/'photo_manifest.csv').read_bytes()),
                           split_sha256=digest((base/split).read_bytes()))
    return out


def definition():
    from .models import KV_ARRAY
    return dict(version=VERSION, tile_size=256, grid='train-grid-v1/overlap_last',
        filter='fixed RGB depthwise KV 5x5', kernel=KV_ARRAY.tolist(), dtype='float32',
        clip=[-3, 3], border=2, std='torch population std over RGB interior; correction=0',
        batch_tiles=16, torch_threads_per_worker=1)


def records(root):
    photos = []
    for source in SOURCES:
        with (Path(root)/'data'/source/'photo_manifest.csv').open(newline='') as f:
            photos.extend(r for r in csv.DictReader(f) if r['split'] in ('val', 'test', 'test_seen', 'test_unseen'))
    if len({r['photo_id'] for r in photos}) != len(photos):
        raise ValueError('noise scan requires unique photo identities')
    return sorted(photos, key=lambda r: (r['source'], int(r['photo_index'])))


def initialize_worker():
    import torch
    from lsb_source import configure_source_worker
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    configure_source_worker()


def measure_photo(task):
    """Decode once; at most 16 float32 crops in flight. No model or stego generation."""
    import torch
    import torch.nn.functional as F
    from .models import KV_ARRAY
    from lsb_source import load_photo
    from lsb_train_grid import axes
    root, record = task
    started = time.perf_counter()
    photo = load_photo(record, Path(root))
    decoded = time.perf_counter()
    ys, xs = axes(record, 256, 'overlap_last')
    coordinates = [(r, c, y, x) for r, y in enumerate(ys) for c, x in enumerate(xs)]
    kernel = torch.from_numpy(KV_ARRAY).view(1, 1, 5, 5).repeat(3, 1, 1, 1)
    rows = []
    with torch.inference_mode():
        for lo in range(0, len(coordinates), 16):
            group = coordinates[lo:lo+16]
            tiles = np.stack([photo[y:y+256, x:x+256] for _, _, y, x in group])
            images = torch.from_numpy(tiles).permute(0, 3, 1, 2).contiguous().float()
            residual = F.conv2d(images, kernel, padding=2, groups=3)
            values = residual[:, :, 2:-2, 2:-2].clamp(-3, 3).flatten(1).std(dim=1, unbiased=False)
            for (r, c, y, x), value in zip(group, values.tolist()):
                rows.append(dict(cover_id=f"{record['photo_id']}:overlap_last:256:{r}:{c}:v1",
                    source=record['source'], source_photo=record['photo_id'], camera_model=record['camera_model'] or 'not_applicable',
                    split=record['split'], photo_index=int(record['photo_index']), tile_size=256, grid='overlap_last',
                    tile_row=r, tile_col=c, y=y, x=x, kv_std=value))
    return rows, dict(source=record['source'], photo=record['photo_id'], tiles=len(rows),
                      decode_seconds=decoded-started, seconds=time.perf_counter()-started)


def scan(root, *, workers=8, pilot_per_source=0):
    """Collect scalar rows, not decoded photos; completion order never changes CSV bytes."""
    from scripts.pc_safety import check_workers
    check_workers(workers)
    photos = records(root)
    totals = {s: sum(r['source'] == s for r in photos) for s in SOURCES}
    if pilot_per_source:
        selected = []
        for source in SOURCES:
            group = [r for r in photos if r['source'] == source]
            indices = np.linspace(0, len(group)-1, min(pilot_per_source, len(group)), dtype=int)
            selected.extend(group[i] for i in indices)
        photos = selected
    before = time.perf_counter()
    rows, timings = [], []
    with mp.get_context('spawn').Pool(workers, initializer=initialize_worker) as pool:
        for values, timing in pool.imap_unordered(measure_photo, [(str(root), r) for r in photos], chunksize=1):
            rows.extend(values); timings.append(timing)
            if len(timings) % 100 == 0:
                print(f'noise photos {len(timings)}/{len(photos)}; {time.perf_counter()-before:.1f}s', flush=True)
    elapsed = time.perf_counter()-before
    by_source = {}
    for source in SOURCES:
        group = [t for t in timings if t['source'] == source]
        by_source[source] = dict(photos=len(group), tiles=sum(t['tiles'] for t in group),
            decode_seconds=sum(t['decode_seconds'] for t in group), worker_seconds=sum(t['seconds'] for t in group),
            full_photos=totals[source])
    work_projection = sum(g['worker_seconds']/g['photos']*g['full_photos'] for g in by_source.values())/workers
    report = dict(photos=len(photos), tiles=len(rows), workers=workers, wall_seconds=elapsed,
                  per_source=by_source, projected_full_seconds=work_projection+elapsed if pilot_per_source else elapsed)
    return rows, report


def write_scan(root, rows, report, out):
    import pandas as pd
    out = Path(out)
    names = ('noise-val.csv', 'noise-test.csv', 'noise-scan.json')
    if any((out/n).exists() for n in names):
        raise ValueError('refusing to overwrite frozen noise')
    frame = pd.DataFrame(rows, columns=COLUMNS).sort_values(['source', 'photo_index', 'tile_row', 'tile_col']).reset_index(drop=True)
    if frame.cover_id.duplicated().any() or not np.isfinite(frame.kv_std).all():
        raise ValueError('invalid noise identities or values')
    files = {}
    for split in ('val', 'test'):
        subset = frame[frame['split'] == 'val'] if split == 'val' else frame[frame['split'] != 'val']
        files[f'noise-{split}.csv'] = subset.to_csv(index=False, float_format='%.17g', lineterminator='\n')
    info = dict(definition=definition(), inputs=inputs(root), files_sha256={n:digest(t.encode()) for n,t in files.items()},
                counts={n:len(t.splitlines())-1 for n,t in files.items()}, scan=report)
    out.mkdir(parents=True, exist_ok=True)
    for name, text in files.items(): (out/name).write_text(text)
    (out/'noise-scan.json').write_text(canonical(info)+'\n')
    return info


def load_noise(root, folder=None):
    import pandas as pd
    folder = Path(folder) if folder is not None else Path(root)/DATA_DIR
    info = json.loads((folder/'noise-scan.json').read_text())
    if info['inputs'] != inputs(root) or info['definition'] != definition():
        raise ValueError('frozen noise input/definition mismatch')
    frames = {}
    for name in ('noise-val.csv', 'noise-test.csv'):
        raw = (folder/name).read_bytes()
        if digest(raw) != info['files_sha256'][name]: raise ValueError(f'noise SHA256 mismatch: {name}')
        import io
        frames[name] = pd.read_csv(io.BytesIO(raw), float_precision='round_trip')
    return frames, info


def sample_check(root, folder=None):
    """Recompute one complete photo per source, matching the fixed batch partition exactly."""
    import pandas as pd
    frames, _ = load_noise(root, folder)
    combined = pd.concat(frames.values(), ignore_index=True)
    initialize_worker()
    by_id = {r['photo_id']:r for r in records(root)}
    for source in SOURCES:
        expected = combined[combined.source == source]
        photo = expected.source_photo.iloc[0]
        expected = expected[expected.source_photo == photo]
        measured, _ = measure_photo((str(root), by_id[photo]))
        actual = pd.DataFrame(measured).set_index('cover_id').kv_std
        reference = expected.set_index('cover_id').kv_std
        if not actual.equals(reference): raise ValueError(f'noise recompute mismatch: {photo}')
        print(f'NOISE RECOMPUTE EXACT: {source} {photo} {len(actual)} tiles', flush=True)
