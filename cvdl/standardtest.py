"""Frozen, model-independent 256 standard tests. Build offline; verify and replay at runtime."""
from __future__ import annotations
import csv
from dataclasses import replace
import hashlib
import io
import itertools
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd

from .validation import canonical, digest, csv_text, MatchedValidationPairs, histograms

VERSION = 'standard-test-256-v1'
SUITE = 'standard-256-v1'
DATA_DIR = f'data/test/{SUITE}'
CONFIG = 'configs/test-256-v1.yaml'
PANELS = {
    'standard': {'raise': ('test', 9696), 'div2k': ('test', 1248)},
    'alaska-seen': {'alaska': ('test_seen', 2688)},
    'alaska-unseen': {'alaska': ('test_unseen', 3168)},
    'lsdir': {'lsdir': ('test', 5280)},
}
COMBINATIONS = tuple(itertools.product(('lowest', 'plane'), ('replace', 'match'),
                                     ('contiguous', 'stride', 'random'), ('random_bytes', 'letters')))
SEMANTIC_KEYS = ('seed', 'bit_mode', 'k', 'method', 'placement', 'stride', 'payload', 'start', 'size',
                 'k_weights', 'bpp_mixture')


def seed_key(*parts):
    text = '/'.join(map(str, (VERSION, 1, *parts)))
    return [int(x) for x in np.frombuffer(hashlib.sha256(text.encode()).digest(), dtype='<u4')]


def rng(*parts):
    return np.random.default_rng(np.random.SeedSequence(seed_key(*parts)))


def filename(panel, higher=False):
    return f'pairs-{"k5to8-" if higher else ""}{panel}.csv'


def describe(root):
    """Read only frozen manifests/splits/config, never image pixels or training settings."""
    from lsb_batch import read_config
    from lsb_train_grid import axes
    root = Path(root)
    config = read_config(root/CONFIG)
    semantic = {k: v for k, v in config.as_dict().items() if k in SEMANTIC_KEYS}
    inputs, pools = {}, {}
    for source in ('raise', 'div2k', 'alaska', 'lsdir'):
        base = root/'data'/source
        manifest = base/'photo_manifest.csv'
        split = base/('split_seed0.csv' if source in ('raise', 'div2k') else 'split.csv')
        inputs[source] = dict(manifest_sha256=digest(manifest.read_bytes()), split_sha256=digest(split.read_bytes()))
        with manifest.open(newline='') as f:
            records = list(csv.DictReader(f))
        if len({r['photo_id'] for r in records}) != len(records):
            raise ValueError(f'{source}: duplicate photo identity')
        for panel, sources in PANELS.items():
            if source not in sources:
                continue
            purpose, count = sources[source]
            selected = sorted((r for r in records if r['split'] == purpose), key=lambda r: int(r['photo_index']))
            tiles = []
            for record in selected:
                ys, xs = axes(record, 256, 'overlap_last')
                tiles.append((record, [(row, col, y, x) for row, y in enumerate(ys) for col, x in enumerate(xs)]))
            if not tiles or count > sum(len(t) for _, t in tiles) or count < len(tiles):
                raise ValueError(f'{panel}/{source}: quota cannot represent every photo without repeated tiles')
            pools[(panel, source)] = tiles
    recipe = dict(version=VERSION, suite=SUITE, schema=1, geometry='train-grid-v1/overlap_last',
                  tile_size=256, seed=1, seed_domain=VERSION, config=semantic, inputs=inputs,
                  panels=PANELS, selection='photo-balanced unused-tile rounds; keyed final partial round',
                  balance='exact k x 24 combinations within each source/panel', higher_k='same tiles/slots/seeds; k+4',
                  variants_per_tile=1, augment=False)
    # Normalize tuples to JSON arrays for direct comparison with saved JSON.
    return json.loads(canonical(recipe)), pools, config


def select_tiles(photos, count, panel, source):
    """One unused tile per photo per round; final partial round is independently keyed."""
    order = rng('photo-order', panel, source).permutation(len(photos))
    shuffled = []
    for index in order:
        record, tiles = photos[index]
        tile_order = rng('tile-order', panel, source, record['photo_id']).permutation(len(tiles))
        shuffled.append((record, [tiles[i] for i in tile_order]))
    result, round_index = [], 0
    while len(result) < count:
        eligible = [(record, tiles[round_index]) for record, tiles in shuffled if len(tiles) > round_index]
        if not eligible:
            raise ValueError('tile pool exhausted; repeats are forbidden')
        order = rng('round', panel, source, round_index).permutation(len(eligible))
        result.extend(eligible[i] for i in order[:count-len(result)])
        round_index += 1
    return result


def make_rows(tiles, panel, source, config, recipe_sha, higher=False):
    from lsb_settings import draw_otf_settings
    n = len(tiles)
    if n % 96:
        raise ValueError('source quota must be a multiple of 96')
    slots = [(k, combo) for k in range(1, 5) for combo in COMBINATIONS for _ in range(n//96)]
    rng('slots', panel, source).shuffle(slots)
    image = np.zeros((256, 256, 3), dtype=np.uint8)
    rows = []
    for slot, ((record, (row, col, y, x)), (low_k, combo)) in enumerate(zip(tiles, slots)):
        bit_mode, method, placement, payload = combo
        k = low_k + (4 if higher else 0)
        settings_key = seed_key('settings', panel, source, slot)
        embedding_key = seed_key('embedding', panel, source, slot)
        fixed = replace(config, bit_mode=bit_mode, k=k, method=method, placement=placement, payload=payload,
                        stride=config.stride if placement == 'stride' else 1)
        kwargs, meta = draw_otf_settings(fixed, image, np.random.default_rng(np.random.SeedSequence(settings_key)))
        kwargs.pop('rng')
        cover = f"{record['photo_id']}:overlap_last:256:{row}:{col}:v1"
        rows.append(dict(pair_id=f'{recipe_sha[:20]}/{panel}/{source}/{slot:05d}/k{k}', cover_id=cover,
            panel=panel, source=source, source_photo=record['photo_id'], camera_model=record['camera_model'] or 'not_applicable',
            split=record['split'], photo_index=int(record['photo_index']), tile_size=256, grid='overlap_last',
            tile_row=row, tile_col=col, y=y, x=x, variant=1, slot=slot, k=k, bit_mode=bit_mode,
            method=method, placement=placement, stride=kwargs['stride'], payload_kind=payload,
            requested_bpp=meta['requested_bpp'], payload_bytes=meta['payload_bytes'], effective_bpp=meta['effective_bpp'],
            clipped=meta['clipped'], bpp_mixture_component=-1, draw_kwargs=canonical(kwargs), settings_seed_key=canonical(settings_key),
            embedding_seed_key=canonical(embedding_key)))
    return rows


def panel_stats(frame, photos):
    quotas = frame.groupby('source_photo').size()
    return dict(pairs=len(frame), photos=int(len(quotas)), distinct_tiles=int(frame.cover_id.nunique()),
                per_photo_quota_min=int(quotas.min()), per_photo_quota_max=int(quotas.max()),
                per_photo_quota_histogram={str(k): int(v) for k, v in quotas.value_counts().sort_index().items()},
                camera_pairs={str(k): int(v) for k, v in frame.camera_model.value_counts().sort_index().items()},
                camera_photos={str(k): int(v) for k, v in frame.groupby('camera_model').source_photo.nunique().items()},
                per_source={s: dict(photos=len(pool), tiles=sum(len(t) for _, t in pool),
                    selected_photos=int(frame.loc[frame.source == s, 'source_photo'].nunique()),
                    quota_histogram={str(k): int(v) for k, v in frame.loc[frame.source == s].groupby('source_photo').size().value_counts().sort_index().items()})
                    for s, pool in photos.items()}, histograms=histograms(frame))


def build(root):
    recipe, pools, config = describe(root)
    recipe_sha = digest(canonical(recipe).encode())
    files, stats, shas = {}, {}, {}
    for panel, sources in PANELS.items():
        tiles = {s: select_tiles(pools[(panel, s)], count, panel, s) for s, (_, count) in sources.items()}
        for higher in (False, True):
            rows = [r for source in sources for r in make_rows(tiles[source], panel, source, config, recipe_sha, higher)]
            frame = pd.DataFrame(rows)
            name = filename(panel, higher)
            text = csv_text(frame)
            files[name], shas[name] = text, digest(text.encode())
            stats[name] = panel_stats(frame, {s: pools[(panel, s)] for s in sources})
    summary = dict(version=VERSION, suite=SUITE, recipe_sha256=recipe_sha, recipe_id=recipe_sha[:20],
                   files_sha256=shas, panels=stats, raw_config_sha256=digest((Path(root)/CONFIG).read_bytes()))
    files['recipe.json'] = canonical(recipe)+'\n'
    files['summary.json'] = canonical(summary)+'\n'
    return files, summary


def verify_frame(frame, panel, higher, pools):
    from lsb_train_grid import axes
    if len(frame) != sum(n for _, n in PANELS[panel].values()) or frame.pair_id.duplicated().any() or frame.cover_id.duplicated().any():
        raise ValueError('frozen standard counts or unique identities mismatch')
    if set(frame.source) != set(PANELS[panel]) or set(frame.panel) != {panel} or set(frame.tile_size) != {256} or set(frame.grid) != {'overlap_last'}:
        raise ValueError('frozen standard panel/geometry mismatch')
    for source, (purpose, count) in PANELS[panel].items():
        group = frame[frame.source == source]
        photos = {r['photo_id']: r for r, _ in pools[(panel, source)]}
        if len(group) != count or set(group.source_photo) != set(photos) or set(group['split']) != {purpose}:
            raise ValueError('frozen standard source/split/photo quota mismatch')
        counts = group.groupby(['k', 'bit_mode', 'method', 'placement', 'payload_kind']).size()
        if len(counts) != 96 or set(counts) != {count//96} or set(group.k) != set(range(5 if higher else 1, 9 if higher else 5)):
            raise ValueError('frozen standard k/type balance mismatch')
        for row in group.itertuples():
            record = photos[row.source_photo]
            ys, xs = axes(record, 256, 'overlap_last')
            expected = f'{row.source_photo}:overlap_last:256:{row.tile_row}:{row.tile_col}:v1'
            if (row.cover_id != expected or not 0 <= row.tile_row < len(ys) or not 0 <= row.tile_col < len(xs)
                    or (row.y, row.x) != (ys[row.tile_row], xs[row.tile_col])):
                raise ValueError('frozen standard coordinate mismatch')


def load_frozen(root, *, higher_k=False):
    """Verify inputs, recipe, list bytes and geometry. No runtime writes or random draws."""
    recipe, pools, _ = describe(root)
    folder = Path(root)/DATA_DIR
    saved = json.loads((folder/'recipe.json').read_text())
    summary = json.loads((folder/'summary.json').read_text())
    expected_sha = digest(canonical(recipe).encode())
    expected_files = {filename(p, h) for p in PANELS for h in (False, True)}
    if (saved != recipe or summary.get('recipe_sha256') != expected_sha
            or summary.get('recipe_id') != expected_sha[:20] or summary.get('version') != VERSION
            or summary.get('suite') != SUITE or set(summary.get('files_sha256', {})) != expected_files):
        raise ValueError('frozen standard recipe/input mismatch')
    frames = {}
    # Validate the complete frozen suite, even when higher-k scoring is not requested.
    for higher in (False, True):
        for panel in PANELS:
            name = filename(panel, higher)
            raw = (folder/name).read_bytes()
            if digest(raw) != summary['files_sha256'][name]:
                raise ValueError(f'frozen standard list SHA256 mismatch: {name}')
            frame = pd.read_csv(io.BytesIO(raw), float_precision='round_trip')
            verify_frame(frame, panel, higher, pools)
            frames[name] = frame
            if higher:
                low = frames[filename(panel)]
                keys = ['cover_id', 'slot', 'requested_bpp', 'bit_mode', 'method', 'placement', 'payload_kind', 'stride',
                        'settings_seed_key', 'embedding_seed_key']
                if not frame[keys].equals(low[keys]) or not np.array_equal(frame.k, low.k+4):
                    raise ValueError('frozen higher-k pairing mismatch')
    if not higher_k:
        frames = {filename(panel): frames[filename(panel)] for panel in PANELS}
    return frames, recipe, summary


class StandardTestPairs(MatchedValidationPairs):
    """Explicit 256 overlap-last geometry and recorded draws; no legacy grid changes."""


def diagnostic_rows(frame):
    """Explicit names only: no confusion between test-optimized and deployment metrics."""
    from .metrics import metrics, threshold, rates
    rows = []
    groups = [('overall', 'all', frame)]
    groups += [('source', str(s), f) for s, f in frame.groupby('source', sort=True)]
    groups += [('k', str(k), f) for k, f in frame.groupby('k', sort=True)]
    groups += [('source/k', f'{s}/{k}', f) for (s, k), f in frame.groupby(['source', 'k'], sort=True)]
    for category, value, group in groups:
        m = metrics(group)
        cutoff = threshold(group, 'pe')[0]
        fixed = rates(group, .5)
        rows.append(dict(category=category, value=value, pairs=m['pairs'], images=m['images'],
            photos=int(group.source_photo.nunique()), AUC=m['AUC'], test_min_P_E=m['P_E'], test_min_cutoff=cutoff,
            FA_at_0_5=fixed['fa'], miss_at_0_5=fixed['miss'], P_E_at_0_5=fixed['pe_at_cutoff']))
    return pd.DataFrame(rows)


def main_placeholder(diagnostics):
    """Task #4 replaces these pending columns from saved validation predictions only."""
    result = diagnostics[['panel', 'category', 'value', 'pairs', 'images', 'photos', 'AUC']].copy()
    result['threshold_origin'] = 'pending_validation_min_pe'
    for key in ('cutoff', 'FA', 'miss', 'P_E_at_validation_cutoff'):
        result[key] = None
    return result


def prepare_output(path, *, overwrite=False):
    path = Path(path)
    if path.exists() and any(path.iterdir()) and not overwrite:
        raise ValueError(f'refusing to overwrite existing output (use --overwrite): {path}')
    path.mkdir(parents=True, exist_ok=True)
    return path


def run(settings, checkpoint, run_folder, *, higher_k=False, output=None, overwrite=False, provenance=None):
    """New-suite branch: test only; never execute legacy post_fit or modify training records."""
    import torch
    from lsb_torch import Corpus, RunSettings
    from .models import model_from_hparams
    from .hardtest import score
    from .settings import plain
    if not 0 <= settings.eval_workers <= 12:
        raise ValueError('standard test requires 0..12 workers')
    start = time.perf_counter()
    frames, recipe, frozen = load_frozen(settings.root, higher_k=higher_k)
    checkpoint = Path(checkpoint)
    checkpoint_sha = digest(checkpoint.read_bytes())
    selected = {}
    for name, frame in frames.items():
        if settings.quick:
            frame = frame.groupby(['source', 'k', 'bit_mode', 'method', 'placement', 'payload_kind'], sort=False).head(1).copy()
        selected[name] = frame.reset_index(drop=True)
    identity = dict(suite=SUITE, recipe_sha256=frozen['recipe_sha256'], checkpoint_sha256=checkpoint_sha,
        selected_pairs_sha256={n: digest(csv_text(f).encode()) for n, f in selected.items()},
        higher_k=higher_k, quick=bool(settings.quick), device=str(settings.device), batch_size=settings.batch_size,
        workers=settings.eval_workers, rocm_conv1_fix=bool(settings.rocm_conv1_fix))
    eid = digest(canonical(identity).encode())[:20]
    out = Path(output) if output is not None else Path(run_folder)/'evaluation'/SUITE/eid
    # Even explicit --output must never target a training/checkpoint record folder.
    run_folder = Path(run_folder).resolve()
    resolved = out.resolve()
    allowed = run_folder/'evaluation'/SUITE
    if resolved == allowed or not resolved.is_relative_to(allowed):
        raise ValueError(f'new suite output must be under {allowed}')
    out = prepare_output(out, overwrite=overwrite)
    checkpoint_data = torch.load(checkpoint, map_location='cpu', weights_only=False)
    hparams = checkpoint_data.get('hyper_parameters', {})
    network = model_from_hparams(hparams).eval()
    network.load_state_dict({k.removeprefix('net.'): v for k, v in checkpoint_data['state_dict'].items()}, strict=True)
    network.to(settings.device)
    # Test photo identities always come from the suite's verified root manifests,
    # never a model's borrowed/alternate training manifest_root.
    corpus = Corpus(settings.root, RunSettings(sources=('raise', 'div2k', 'alaska', 'lsdir'), source=settings.source))
    from .panels import code_commit
    import PIL
    meta = dict(**identity, evaluation_id=eid, full_files_sha256=frozen['files_sha256'],
        values=plain(settings.values), provenance=provenance or {}, checkpoint=str(checkpoint),
        train_geometry=dict(tile_size=settings.train_tile_size or settings.tile_size, grid=settings.train_grid),
        test_geometry=dict(tile_size=256, grid='overlap_last'),
        hard_test=dict(status='pending', reason='task #3: frozen hard test at 256 is not built'),
        validation_threshold=dict(status='pending', reason='task #4', main_file='main_summary.csv'),
        code_commit=code_commit(settings.root),
        runtime=dict(torch=str(torch.__version__), numpy=np.__version__, pillow=PIL.__version__))
    (out/'evaluation_settings.json').write_text(json.dumps(meta, indent=2)+'\n')
    timings, summaries = [], {False: [], True: []}
    for name, frame in selected.items():
        higher = name.startswith('pairs-k5to8-')
        panel = str(frame.panel.iloc[0])
        target = out/('higher-k' if higher else 'panels')/panel
        target.mkdir(parents=True, exist_ok=True)
        before = time.perf_counter()
        scored = score(network, StandardTestPairs(corpus, frame), device=settings.device,
                       batch_size=max(1, settings.batch_size//2), workers=settings.eval_workers)
        scored.to_csv(target/'predictions.csv', index=False, float_format='%.17g')
        diagnostics = diagnostic_rows(scored)
        diagnostics.insert(0, 'panel', panel)
        diagnostics.to_csv(target/'diagnostic_summary.csv', index=False)
        main_placeholder(diagnostics).to_csv(target/'main_summary.csv', index=False)
        summaries[higher].append(diagnostics)
        timings.append(dict(panel=panel, higher_k=higher, pairs=len(frame), seconds=time.perf_counter()-before))
    for higher, tables in summaries.items():
        if tables:
            target = out/'higher-k' if higher else out
            combined = pd.concat(tables, ignore_index=True)
            combined.to_csv(target/'diagnostic_summary.csv', index=False)
            main_placeholder(combined).to_csv(target/'main_summary.csv', index=False)
    result = dict(evaluation_id=eid, output=str(out), suite=SUITE, quick=bool(settings.quick), timings=timings,
                  total_seconds=time.perf_counter()-start, hard_test=meta['hard_test'], validation_threshold=meta['validation_threshold'])
    (out/'summary.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2), flush=True)
    return result
