"""Opt-in frozen train-matched validation. No image I/O during build/load.

Only the explicit builder writes files. Runtime verifies published inputs and lists.
Legacy FixedPairs, evaluation_config and seed streams are not changed.
"""
from __future__ import annotations
import csv
from fractions import Fraction
import hashlib
import io
import json
from pathlib import Path
import re

import numpy as np

VERSION = 'train-matched-val-v1'
ROUNDING = 'total=round_half_up(train_pairs/10); source=largest_remainder(normalized_mix), ties=source_name'
SEMANTIC_KEYS = ('bit_mode', 'k', 'method', 'placement', 'stride', 'payload', 'start', 'size', 'k_weights', 'bpp_mixture')


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def recipe_id(recipe):
    return digest(canonical(recipe).encode())[:20]


def quotas(training_pairs, mix):
    """Half-up total, Hamilton source allocation; exact where proportions permit."""
    if type(training_pairs) is not int or training_pairs < 1 or not mix:
        raise ValueError('train-matched validation needs positive tiles_per_epoch and source_mix')
    weights = {s: Fraction(str(w)) for s, w in sorted(mix.items())}
    if any(w <= 0 for w in weights.values()):
        raise ValueError('source_mix weights must be positive')
    total = (training_pairs + 5) // 10
    exact = {s: total * w / sum(weights.values()) for s, w in weights.items()}
    counts = {s: int(w) for s, w in exact.items()}
    for s in sorted(weights, key=lambda s: (-(exact[s] - counts[s]), s))[:total - sum(counts.values())]:
        counts[s] += 1
    if not total or any(n == 0 for n in counts.values()):
        raise ValueError('validation size must allow at least one pair per training source')
    return total, counts


def describe(root, settings, train_grid, train_tile=None):
    """Resolve semantic identity and pools from frozen records, never photos."""
    from lsb_batch import read_config
    from lsb_train_grid import axes
    root = Path(root)
    if train_grid != 'overlap_last':
        raise ValueError('train_matched requires train_grid=overlap_last')
    tile = train_tile or settings.tile_size
    if tile not in (256, 512):
        raise ValueError('train_matched tile size must be 256 or 512')
    if settings.variants_per_tile != 1:
        raise ValueError('unique-tile validation requires variants_per_tile=1')
    if set(settings.sources) != set(settings.source_mix or {}):
        raise ValueError('train_matched sources must match source_mix')
    total, counts = quotas(settings.tiles_per_epoch, settings.source_mix)
    config_path = Path(settings.settings_yaml or root/'configs/raise512-run1.yaml')
    config = read_config(config_path)
    all_config = config.as_dict()
    inputs, pools, pool_sizes = {}, {}, {}
    manifests = Path(settings.manifest_root) if settings.manifest_root else root
    for source in sorted(counts):
        base = (manifests if source in {'alaska', 'lsdir'} else root)/'data'/source
        manifest = base/'photo_manifest.csv'
        split = base/('split_seed0.csv' if source in {'raise', 'div2k'} else 'split.csv')
        inputs[source] = dict(manifest_sha256=digest(manifest.read_bytes()), split_sha256=digest(split.read_bytes()))
        with manifest.open(newline='') as f:
            photos = sorted((r for r in csv.DictReader(f) if r['split'] == 'val'), key=lambda r: int(r['photo_index']))
        if len({r['photo_id'] for r in photos}) != len(photos):
            raise ValueError(f'{source}: duplicate validation photo identity')
        candidates = []
        for record in photos:
            ys, xs = axes(record, tile, 'overlap_last')
            candidates.extend((record, row, col, y, x) for row, y in enumerate(ys) for col, x in enumerate(xs))
        pools[source] = candidates
        pool_sizes[source] = dict(photos=len(photos), tiles=len(candidates))
        if counts[source] > len(candidates):
            raise ValueError(f'{source}: validation quota {counts[source]} exceeds unique val tile pool {len(candidates)}; no repeats or caps allowed')
    recipe = dict(version=VERSION, geometry='train-grid-v1/overlap_last', tile_size=tile,
                  train_pairs_per_epoch=settings.tiles_per_epoch, fraction='1/10', rounding=ROUNDING,
                  pairs=total, source_mix={s: str(Fraction(str(settings.source_mix[s])) / sum(Fraction(str(v)) for v in settings.source_mix.values())) for s in counts},
                  source_quotas=counts, config={k: all_config[k] for k in SEMANTIC_KEYS if k in all_config},
                  inputs=inputs, selection_seed=1, seed_domain=VERSION, purpose='val', epoch=0,
                  evaluation_seed=1, variants_per_tile=1, augment=False)
    return recipe, pools, pool_sizes, config, digest(config_path.read_bytes())


def seed_key(record, tile, row, col, stream):
    from lsb_settings import otf_seed_sequence
    return otf_seed_sequence(run_seed=0, purpose='val', epoch=0, photo_index=int(record['photo_index']),
        grid_policy='overlap_last', tile_size=tile, tile_row=row, tile_col=col, variant=1, stream=stream)


def csv_text(frame):
    return frame.to_csv(index=False, float_format='%.17g', lineterminator='\n')


def histograms(frame):
    result = {}
    for column in ('source', 'k', 'bit_mode', 'method', 'placement', 'payload_kind', 'bpp_mixture_component'):
        result[column] = {str(k): int(v) for k, v in frame[column].value_counts().sort_index().items()}
    edges = (0.05, 0.10, 0.15, 0.35, 0.50)
    bpp = frame.requested_bpp.to_numpy()
    result['requested_bpp'] = {f'[{a},{b}{"]" if b == .5 else ")"}': int(((bpp >= a) & ((bpp <= b) if b == .5 else (bpp < b))).sum()) for a, b in zip(edges, edges[1:])}
    result['clipped'] = int(frame.clipped.sum())
    return result


def build(root, settings, train_grid, train_tile=None):
    """Build all canonical file bytes in memory; caller alone decides to write."""
    import pandas as pd
    from lsb_settings import draw_otf_settings
    recipe, pools, sizes, config, raw_sha = describe(root, settings, train_grid, train_tile)
    rid = recipe_id(recipe)
    tile = recipe['tile_size']
    placeholder = np.zeros((tile, tile, 3), dtype=np.uint8)
    rows = []
    for source, count in recipe['source_quotas'].items():
        entropy = np.frombuffer(hashlib.sha256(f'{VERSION}/select/1/{source}'.encode()).digest(), dtype='<u4')
        selected = sorted(np.random.default_rng(np.random.SeedSequence(entropy)).choice(len(pools[source]), count, replace=False))
        for index in selected:
            record, row, col, y, x = pools[source][index]
            settings_seed = seed_key(record, tile, row, col, 'settings')
            embedding_seed = seed_key(record, tile, row, col, 'embedding')
            kwargs, meta = draw_otf_settings(config, placeholder, np.random.default_rng(settings_seed))
            kwargs.pop('rng')
            cover = f"{record['photo_id']}:overlap_last:{tile}:{row}:{col}:v1"
            rows.append(dict(pair_id=f'{rid}/{cover}', cover_id=cover, source_photo=record['photo_id'],
                source=source, camera_model=record['camera_model'] or 'not_applicable', split='val',
                photo_index=int(record['photo_index']), tile_size=tile, grid='overlap_last',
                tile_row=row, tile_col=col, y=y, x=x, variant=1, k=kwargs['k'], bit_mode=kwargs['bit_mode'],
                method=kwargs['method'], placement=kwargs['placement'], stride=kwargs['stride'],
                payload_kind=kwargs['payload'], requested_bpp=meta['requested_bpp'],
                bpp_mixture_component=meta.get('bpp_mixture_component', -1), payload_bytes=meta['payload_bytes'],
                effective_bpp=meta['effective_bpp'], clipped=meta['clipped'],
                draw_kwargs=canonical(kwargs), settings_seed_key=canonical(settings_seed.entropy),
                embedding_seed_key=canonical(embedding_seed.entropy)))
    frame = pd.DataFrame(rows)
    if frame.cover_id.duplicated().any() or frame.pair_id.duplicated().any():
        raise ValueError('validation builder produced repeated tile identities')
    text = csv_text(frame)
    summary = dict(version=VERSION, recipe_id=rid, recipe_sha256=digest(canonical(recipe).encode()),
                   pairs_sha256=digest(text.encode()), pairs=len(frame), source_counts=recipe['source_quotas'],
                   pools=sizes, raw_yaml_sha256=raw_sha, numpy_version=np.__version__,
                   histograms=histograms(frame))
    files = {'recipe.json': canonical(recipe)+'\n', 'pairs.csv': text, 'summary.json': canonical(summary)+'\n'}
    return rid, files, summary


def load_frozen(root, settings, train_grid, train_tile, rid):
    """Fail closed for missing, altered or mismatched inputs. No writes."""
    import pandas as pd
    if not isinstance(rid, str) or not re.fullmatch(r'[0-9a-f]{20}', rid):
        raise ValueError('validation_recipe must be a 20-digit lowercase hex recipe id')
    recipe, pools, _, _, _ = describe(root, settings, train_grid, train_tile)
    if recipe_id(recipe) != rid:
        raise ValueError('validation recipe does not match the training data recipe')
    folder = Path(root)/'data/val'/rid
    try:
        saved = json.loads((folder/'recipe.json').read_text())
        summary = json.loads((folder/'summary.json').read_text())
        raw = (folder/'pairs.csv').read_bytes()
    except (OSError, ValueError) as error:
        raise ValueError(f'missing or invalid frozen validation recipe {rid}: {error}') from error
    if canonical(saved) != canonical(recipe) or summary['recipe_sha256'] != digest(canonical(recipe).encode()) or summary['recipe_id'] != rid:
        raise ValueError('frozen validation recipe identity mismatch')
    if digest(raw) != summary['pairs_sha256']:
        raise ValueError('frozen validation pairs SHA256 mismatch')
    frame = pd.read_csv(io.BytesIO(raw), float_precision='round_trip')
    counts = {s: int(n) for s, n in frame.source.value_counts().items()}
    if len(frame) != recipe['pairs'] or counts != recipe['source_quotas'] or counts != summary['source_counts'] or len(frame) != summary['pairs']:
        raise ValueError('frozen validation counts mismatch')
    if frame.cover_id.duplicated().any() or frame.pair_id.duplicated().any() or set(frame['split']) != {'val'}:
        raise ValueError('frozen validation must contain unique val tiles only')
    allowed = {f"{r['photo_id']}:overlap_last:{recipe['tile_size']}:{row}:{col}:v1" for pool in pools.values() for r, row, col, _, _ in pool}
    if not set(frame.cover_id) <= allowed:
        raise ValueError('frozen validation tile outside val pool')
    return frame, recipe, summary, raw.decode()


# Inherit only the photo-owned loader interface; do not invoke legacy FixedPairs.
from lsb_torch import FixedPairs


def evaluation_tensor(image):
    """Evaluator pixels: contiguous CHW RGB8, no resize, scaling or augmentation."""
    import torch
    return torch.from_numpy(np.ascontiguousarray(image)).permute(2,0,1).contiguous()


class MatchedValidationPairs(FixedPairs):
    """Replay recorded kwargs and embedding keys, with explicit overlap geometry."""
    def __init__(self, corpus, frame):
        self.corpus = corpus
        self.frame = frame.reset_index(drop=True)
        self._last_id = self._last_photo = None
        # PhotoPairTasks uses the historical entries interface for task ownership.
        self.entries = [(corpus.by_id[r.source_photo], int(r.tile_row), int(r.tile_col), 1, None)
                        for r in self.frame.itertuples()]

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, index):
        import torch
        from lsb_engine import embed_fast
        from lsb_train_grid import crop
        row = self.frame.iloc[index]
        record = self.corpus.by_id[row.source_photo]
        if self._last_id != row.source_photo:
            self._last_photo = self.corpus.photo(record)
            self._last_id = row.source_photo
        pixels, (y, x), pad = crop(self._last_photo, record, int(row.tile_size), int(row.tile_row), int(row.tile_col), 'overlap_last')
        if (y, x) != (int(row.y), int(row.x)) or any(pad):
            raise ValueError('frozen validation geometry mismatch')
        clean = pixels.copy()
        kwargs = json.loads(row.draw_kwargs)
        kwargs['rng'] = np.random.default_rng(np.random.SeedSequence(json.loads(row.embedding_seed_key)))
        stego, _, _ = embed_fast(pixels, **kwargs)
        return evaluation_tensor(clean), evaluation_tensor(stego), index
