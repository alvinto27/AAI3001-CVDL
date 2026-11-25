"""Frozen TEST-only colour-channel panel. No training or threshold fitting."""
from __future__ import annotations

import hashlib
import itertools
import json
from pathlib import Path
from statistics import NormalDist
import time

import numpy as np
import pandas as pd

VERSION = 'rgb-channel-analysis-v1'
SEED = 20261024
ARMS = ('r', 'g', 'b', 'rgb', 'native_rgb')
RATES = (.05, .15, .5)
PLACEMENTS = ('contiguous', 'random')
STRATA = (('raise', 'test'), ('div2k', 'test'), ('alaska', 'test_seen'),
          ('alaska', 'test_unseen'), ('lsdir', 'test'))
RUN18 = 'srm-cell-aux-curriculum50-paired-20261023T133537Z-1366'
MODEL_SPECS = {
    'r1': ('srm-e50-paired-20261023T022339Z-4518', '6ef349895655136bef90',
           'b00cd4b6d1f4c3648580691c9cf0e2a4bd83c55740a3cf2a7a561a8adf12f180',
           .514680027961731, 'bf21475124bc8e0abd8a'),
    'run18': (RUN18, 'b3d9a49137b5dbd7bd5a',
              'eee9c63a0bd9a91b63758dfc4ddfcca37e47a2a621fbb663557c37bec4b00b80',
              .5094263553619385, '2a3b68381c0b0baf7381'),
    'srm-e50': ('srm-e50-20261017T111310Z-0214', '35afd1c7938bae78968c',
                'bf64848a49bdc6e20b7fcdba47cb8eee0ed247b909a928cc45e4a4102db9f076',
                .450578898191452, 'e5dc5c8d82bf280630f9'),
}
CELL_SOURCE = 'runs/verification/cell-curriculum50-paired-2026-10-23/cells/summary.json'
CELL_SHA = '50ebc199f465d624a5dd766a464c9ceee9ffa3cae726b4eb859f8b4c5f98e6fb'
CELL_CUTOFF = .1966136991977692
CODE_FILES = ('LSB-Dataset-Generator/lsb_engine.py', 'LSB-Dataset-Generator/lsb_cells.py',
              'LSB-Dataset-Generator/lsb_source.py', 'cvdl/metrics.py', 'cvdl/standardtest.py',
              'cvdl/channel_analysis.py', 'cvdl/models.py', 'cvdl/cell_diagnostics.py',
              'cvdl/pixels.py', 'scripts/build_rgb_channel_panel.py',
              'scripts/evaluate_rgb_channel_panel.py')


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def file_hash(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def seed(*parts):
    return int.from_bytes(hashlib.sha256(canonical([VERSION, SEED, *parts]).encode()).digest()[:16], 'little')


def rng(*parts):
    return np.random.default_rng(seed(*parts))


def reading_limits():
    return dict(primary_model='run18', primary_k=1, primary_placement='random', primary_rates=[.05, .15],
        aggregation='equal mean within 5 source/split x 2 noise strata x 2 low-rate AUCs',
        contrasts=['r-rgb', 'g-rgb', 'b-rgb'], auc_minimum=.02, adjusted_interval=.9833333333333333,
        interval_rule='point absolute difference >= .02 AND adjusted interval excludes zero',
        equivalence='all three adjusted intervals contained in [-.02,+.02]',
        cutoff_PE_minimum=.01, cutoff_rule='secondary: absolute delta >= .01 AND paired95 interval excludes zero',
        bootstrap_replicates=5000, bootstrap_seed=SEED, bootstrap_unit='photo',
        bootstrap_strata='source/split/noise; shared photo multiplicities for every arm/rate/model',
        limits=['single-channel density is 3x per colour and changes colour relations',
                'native RGB also changes spatial support; no inference of benefit from new colour features',
                'existing TEST photos, not an untouched benchmark; no #20 training decision'])


def estimate_precision(root, correlation=.5, within_photo_correlation=.5):
    """Step 0: saved RGB scores only; paired AUC influence estimate, not power proof.

    Assume equal marginal AUC variances in new R/RGB and correlation .5 between
    their estimates. Perfect dependence across .05/.15 gives no rate-averaging
    bonus. One deterministic existing pair/photo/noise stratum; all k1 random
    low-bpp methods/payloads are pooled as an explicitly imperfect pilot proxy.
    """
    root = Path(root)
    path = root/'runs'/RUN18/'evaluation/standard-256-v1/b3d9a49137b5dbd7bd5a/hard-test/hard_predictions.csv'
    frame = pd.read_csv(path, float_precision='round_trip')
    frame = frame[(frame.k == 1) & frame.placement.eq('random') & frame.effective_bpp.between(.05, .1501)].copy()
    edges = json.loads((root/'data/hard-test/256-v1/recipe.json').read_text())['noise_edges']
    frame['noise'] = np.where(frame.tile_kv_std > frame.source.map(edges), 'noisy', 'other')
    summaries = []
    for (source, panel, noise), group in frame.groupby(['source', 'panel', 'noise'], sort=True):
        pairs = group.pivot(index=['source_photo', 'pair_id'], columns='label', values='prob_stego')
        pairs = pairs.reset_index().sort_values('pair_id').drop_duplicates('source_photo')
        if len(pairs) < 10 or pairs[[0, 1]].isna().any().any():
            raise ValueError('insufficient paired saved scores for precision estimate')
        clean, stego = pairs[0].to_numpy(), pairs[1].to_numpy()
        comparison = (stego[:, None] > clean[None, :]) + .5*(stego[:, None] == clean[None, :])
        auc = float(comparison.mean())
        influence = comparison.mean(1)+comparison.mean(0)-2*auc
        summaries.append(dict(source=source, panel=panel, noise=noise, photos=len(pairs), AUC=auc,
                              influence_variance=float(influence.var(ddof=1))))
    if len(summaries) != 10:
        raise ValueError('precision estimate requires all ten strata')
    macro_se = float(np.sqrt(sum(s['influence_variance'] for s in summaries)/(100*48)))
    z = NormalDist().inv_cdf(1-.05/6)
    half = z*macro_se*np.sqrt(2*(1-correlation))
    two = half*np.sqrt((1+within_photo_correlation)/2)
    tiles = 1 if half <= .02 else 2
    if tiles == 2 and two > .02:
        raise ValueError(f'BLOCKER: expected half-width {two:.6f} still > .02 at two tiles/photo')
    return dict(source=str(path.relative_to(root)), source_sha256=file_hash(path), strata=summaries,
        method='paired positive+negative AUC influence variance/n, macro variance=sum(stratum variances)/100',
        paired_R_RGB_correlation=correlation, equal_marginal_variance=True,
        rate_correlation=1., two_tile_within_photo_correlation=within_photo_correlation,
        photos_per_stratum=48, macro_RGB_standard_error=macro_se, adjusted_normal_z=z,
        expected_half_width_one_tile=float(half), expected_half_width_two_tiles=float(two), tiles_per_photo=tiles,
        sensitivity_one_tile={str(r):float(z*macro_se*np.sqrt(2*(1-r))) for r in (0., .5, .8)},
        caveat='approximation from existing heterogeneous low-bpp RGB scores; unobserved R/RGB covariance is assumed, not measured')


def select_base(frame, edges, photos_per_stratum=48, tiles_per_photo=1):
    """No scores: select noisy first, then disjoint other-noise photos."""
    selected, seen = [], set()
    for source, split in STRATA:
        pool = frame[frame.source.eq(source) & frame['split'].eq(split)].copy()
        for noise in ('noisy', 'other'):
            noisy = pool.kv_std > edges[source]
            eligible = pool[noisy if noise == 'noisy' else ~noisy]
            groups = {photo:g for photo,g in eligible.groupby('source_photo')
                      if photo not in seen and len(g) >= tiles_per_photo}
            ordered = sorted(groups, key=lambda p:seed('photo', source, split, noise, p))
            if len(ordered) < photos_per_stratum:
                raise ValueError(f'insufficient distinct photos: {source}/{split}/{noise}')
            for photo in ordered[:photos_per_stratum]:
                seen.add(photo)
                tiles = sorted(groups[photo].to_dict('records'),
                               key=lambda r:seed('tile', r['cover_id']))[:tiles_per_photo]
                for tile in tiles:
                    tile.update(base_index=len(selected), noise=noise,
                                stratum=f'{source}/{split}/{noise}')
                    selected.append(tile)
    return selected


def payload_for(base, rate):
    count = max(1, round(rate*256*256/8))
    return rng('message', base['source_photo'], base['cover_id'], rate).integers(0, 256, count, dtype=np.uint8).tobytes()


def make_variant(clean, base, k, rate, placement, arm):
    from lsb_engine import embed_fast, replace_at_units, _select_units
    if arm not in ARMS or k not in (1, 2) or rate not in RATES or placement not in PLACEMENTS:
        raise ValueError('unsupported channel-panel setting')
    payload = payload_for(base, rate)
    length = len(payload)*8//k
    if length > 65536:
        raise ValueError('single-channel payload exceeds capacity; clipping forbidden')
    if arm == 'native_rgb':
        message_seed = seed('message', base['source_photo'], base['cover_id'], rate)
        stego, meta, changed = embed_fast(clean, 'lowest', k, 'replace', placement,
            np.random.default_rng(message_seed), payload_bytes=len(payload), payload='random_bytes',
            return_changed_units=True)
        replay = np.random.default_rng(message_seed)
        assert replay.integers(0, 256, len(payload), dtype=np.uint8).tobytes() == payload
        units, _, _ = _select_units(clean.size, length, placement, None, 1, replay)
    else:
        spatial = rng('pixels', base['cover_id'], k, rate, placement)
        if placement == 'random':
            pixels = np.sort(spatial.choice(65536, size=length, replace=False))
        else:
            start = int(spatial.integers(0, 65536-length+1))
            pixels = np.arange(start, start+length)
        phase = int(rng('rgb-phase', base['cover_id'], k, rate, placement).integers(0, 3))
        channels = (np.arange(length)+phase) % 3 if arm == 'rgb' else ARMS.index(arm)
        units = 3*pixels+channels
        stego, meta, changed = replace_at_units(clean, payload, k, units)
    return stego, meta, changed, np.asarray(units, dtype=np.int64), payload


def models(root):
    root = Path(root)
    result = {}
    for name, (run, eid, sha, cutoff, threshold_id) in MODEL_SPECS.items():
        folder = root/'runs'/run
        calibration_path = folder/'evaluation/standard-256-v1'/eid/'validation/calibration.json'
        calibration = json.loads(calibration_path.read_text())
        if (calibration['checkpoint_sha256'] != sha or calibration['cutoff'] != cutoff or
                calibration['threshold_id'] != threshold_id or not calibration['deployment'] or calibration['quick'] or
                calibration['geometry'] != dict(grid='overlap_last', tile_size=256)):
            raise ValueError('model/calibration identity mismatch: '+name)
        candidates = sorted((folder/'checkpoints').rglob('*.ckpt'))
        matches = [p for p in candidates if file_hash(p) == sha]
        if not matches:
            raise ValueError('pinned checkpoint missing: '+name)
        result[name] = dict(checkpoint=str(matches[0].relative_to(root)), checkpoint_sha256=sha,
            calibration=str(calibration_path.relative_to(root)), calibration_sha256=file_hash(calibration_path),
            evaluation_id=eid, cutoff=cutoff, threshold_id=threshold_id)
    if file_hash(root/CELL_SOURCE) != CELL_SHA:
        raise ValueError('cell cutoff source mismatch')
    if json.loads((root/CELL_SOURCE).read_text())['maps']['aux']['cutoff'] != CELL_CUTOFF:
        raise ValueError('cell cutoff mismatch')
    return result


def build(root, output, reading_rule):
    from lsb_source import load_photo
    root, output = Path(root), Path(output)
    if output.exists():
        raise ValueError('refusing to overwrite a frozen panel')
    precision = estimate_precision(root)  # Step 0, before anything is frozen.
    pinned = models(root)
    folder = root/'data/hard-test/256-v1'
    hard_recipe = json.loads((folder/'recipe.json').read_text())
    inputs = {str(p.relative_to(root)):file_hash(p) for p in
              [folder/'noise-test.csv', folder/'recipe.json', folder/'summary.json']}
    if inputs['data/hard-test/256-v1/noise-test.csv'] != hard_recipe['noise_files_sha256']['noise-test.csv']:
        raise ValueError('frozen noise input hash mismatch')
    records = {}
    for source in ('raise', 'div2k', 'alaska', 'lsdir'):
        path = root/f'data/{source}/photo_manifest.csv'
        inputs[str(path.relative_to(root))] = file_hash(path)
        if inputs[str(path.relative_to(root))] != hard_recipe['inputs'][source]['manifest_sha256']:
            raise ValueError('source manifest hash mismatch')
        for record in pd.read_csv(path, dtype=str, keep_default_na=False).to_dict('records'):
            if record['photo_id'] in records:
                raise ValueError('duplicate photo ID')
            records[record['photo_id']] = record
    frame = pd.read_csv(folder/'noise-test.csv', float_precision='round_trip')
    base = select_base(frame, hard_recipe['noise_edges'], tiles_per_photo=precision['tiles_per_photo'])
    # A recipe is published only after every clean tile and variant is verified.
    output.mkdir(parents=True)
    try:
        cache = np.lib.format.open_memmap(output/'clean.npy', mode='w+', dtype=np.uint8,
                                         shape=(len(base), 256, 256, 3))
        variants = []
        for index, item in enumerate(base):
            record = records[item['source_photo']]
            if record['split'] != item['split'] or record['split'] in ('train', 'val'):
                raise ValueError('TEST photo/split mismatch')
            photo = load_photo(record, root)
            y, x = int(item['y']), int(item['x'])
            tile = np.ascontiguousarray(photo[y:y+256, x:x+256])
            if tile.shape != (256, 256, 3) or tile.dtype != np.uint8:
                raise ValueError('clean geometry/type mismatch')
            cache[index] = tile
            item['clean_sha256'] = digest(tile.tobytes())
            item['master_path'] = record['master_path']
            item['source_sha256'] = file_hash(root/record['master_path'])
            for k, rate, placement, arm in itertools.product((1, 2), RATES, PLACEMENTS, ARMS):
                stego, meta, changed, units, payload = make_variant(tile, item, k, rate, placement, arm)
                variants.append(dict(variant_id=len(variants), base_index=index, k=k, rate=rate,
                    placement=placement, arm=arm, payload_bytes=len(payload), message_sha256=digest(payload),
                    effective_bpp=meta['bpp_actual'], embedded_bits=meta['embedded_bits'], units_used=len(units),
                    units_sha256=digest(units.astype('<i8').tobytes()),
                    pixels_sha256=digest(np.unique(units//3).astype('<i8').tobytes()),
                    changed_sha256=digest(changed.astype('<i8').tobytes()), changed_units=len(changed),
                    changed_pixels=len(np.unique(changed//3)),
                    selected_by_channel=canonical(np.bincount(units % 3, minlength=3).tolist()),
                    changed_by_channel=canonical(np.bincount(changed % 3, minlength=3).tolist()),
                    stego_sha256=digest(stego.tobytes())))
            print(f'CPU freeze: {index+1}/{len(base)} clean tiles', flush=True)
        cache.flush()
        del cache
        pd.DataFrame(base).to_csv(output/'base-tiles.csv', index=False, lineterminator='\n')
        pd.DataFrame(variants).to_csv(output/'variants.csv', index=False, lineterminator='\n')
        rule = Path(reading_rule).read_bytes()
        (output/'reading-rule.md').write_bytes(rule)
        dump(output/'step0-precision.json', precision)
        recipe = dict(version=VERSION, seed=SEED, photos_per_stratum=48,
            tiles_per_photo=precision['tiles_per_photo'], photos=480, tiles=len(base),
            embedding_settings=len(base)*12, stego_images=len(variants), images_per_model=len(base)+len(variants),
            arms=list(ARMS), ks=[1, 2], rates=list(RATES), placements=list(PLACEMENTS),
            method='lowest/replace/random_bytes', geometry='overlap_last/256; stored RGB8; no augmentation',
            pixel_pairing='identical selected pixels/fields across r/g/b/rgb; native_rgb is a separate layout bridge',
            noise_edges=hard_recipe['noise_edges'], noise_rule='strict > validation edge; complementary other; clean only',
            selection='48 noisy photos then48 disjoint other photos per source/split; stable SHA ordering',
            inputs=inputs, precision=precision, models=pinned,
            cell=dict(source=CELL_SOURCE, sha256=CELL_SHA, cutoff=CELL_CUTOFF), reading=reading_limits(),
            code_sha256={p:file_hash(root/p) for p in CODE_FILES},
            files_sha256={p:file_hash(output/p) for p in
                          ('base-tiles.csv', 'variants.csv', 'clean.npy', 'reading-rule.md', 'step0-precision.json')})
        recipe_sha = digest(canonical(recipe).encode())
        dump(output/'recipe.json', recipe)
        (output/'recipe.sha256').write_text(recipe_sha+'\n')
        validate_panel(root, output)
        return recipe_sha, recipe
    except BaseException:
        # Failed builds are not frozen evidence. Keep successful output only.
        import shutil
        shutil.rmtree(output)
        raise


def validate_panel(root, output):
    root, output = Path(root), Path(output)
    recipe = json.loads((output/'recipe.json').read_text())
    if recipe['version'] != VERSION or digest(canonical(recipe).encode()) != (output/'recipe.sha256').read_text().strip():
        raise ValueError('recipe identity mismatch')
    for path, sha in recipe['files_sha256'].items():
        if file_hash(output/path) != sha:
            raise ValueError('frozen panel file changed: '+path)
    for path, sha in {**recipe['inputs'], **recipe['code_sha256']}.items():
        if file_hash(root/path) != sha:
            raise ValueError('frozen input/code changed: '+path)
    if models(root) != recipe['models']:
        raise ValueError('frozen model identity changed')
    base = pd.read_csv(output/'base-tiles.csv', float_precision='round_trip')
    variants = pd.read_csv(output/'variants.csv', float_precision='round_trip')
    cache = np.load(output/'clean.npy', mmap_mode='r', allow_pickle=False)
    if (len(base) != recipe['tiles'] or len(variants) != recipe['stego_images'] or
            base.source_photo.nunique() != 480 or cache.shape != (len(base), 256, 256, 3) or cache.dtype != np.uint8 or
            base.cover_id.duplicated().any() or list(base.base_index) != list(range(len(base))) or
            list(variants.variant_id) != list(range(len(variants)))):
        raise ValueError('frozen count/geometry mismatch')
    for item in base.to_dict('records'):
        if digest(cache[item['base_index']].tobytes()) != item['clean_sha256']:
            raise ValueError('clean tile hash mismatch')
    return recipe, base, variants


class ChannelDataset:
    """Evaluation-only CPU regeneration; no augmentation or training settings."""
    def __init__(self, output, base, variants):
        self.output = str(output)
        self.base = base.to_dict('records') if hasattr(base, 'to_dict') else base
        self.variants = variants.to_dict('records') if hasattr(variants, 'to_dict') else variants
        self.cache = None

    def __len__(self):
        return len(self.base)+len(self.variants)

    def __getstate__(self):
        return {**self.__dict__, 'cache':None}

    def __getitem__(self, index):
        from .pixels import evaluation_tensor
        from lsb_cells import changed_cell_targets
        if self.cache is None:
            self.cache = np.load(Path(self.output)/'clean.npy', mmap_mode='r', allow_pickle=False)
        if index < len(self.base):
            tile = self.cache[index].copy()
            changed = np.array([], dtype=np.int64)
        else:
            variant = self.variants[index-len(self.base)]
            base = self.base[int(variant['base_index'])]
            tile, _, changed, units, payload = make_variant(self.cache[int(variant['base_index'])], base,
                int(variant['k']), float(variant['rate']), variant['placement'], variant['arm'])
            if (digest(tile.tobytes()) != variant['stego_sha256'] or digest(payload) != variant['message_sha256'] or
                    digest(changed.astype('<i8').tobytes()) != variant['changed_sha256'] or
                    digest(units.astype('<i8').tobytes()) != variant['units_sha256']):
                raise ValueError('variant byte/identity mismatch')
        counts, valid = changed_cell_targets(changed)
        return evaluation_tensor(tile), index, counts.astype(np.int32), valid


def cell_statistics(values, counts, valid, cutoff=CELL_CUTOFF):
    from .metrics import auc_numpy
    positive = counts > 0
    negative = valid & ~positive
    eligible = positive.any() and negative.any()
    auc = float(auc_numpy(np.r_[np.ones(positive.sum()), np.zeros(negative.sum())],
                           np.r_[values[positive], values[negative]])) if eligible else None
    return dict(cell_AUC=auc, cell_eligible=bool(eligible), positive_cells=int(positive.sum()),
        negative_cells=int(negative.sum()),
        positive_hits=int(np.count_nonzero((values >= cutoff) & positive)),
        negative_hits=int(np.count_nonzero((values >= cutoff) & negative)))


def score(root, output, model, *, device='cuda', batch_size=32, workers=4):
    import torch
    from .cell_diagnostics import cell_maps
    from .standardtest import load_network
    from scripts.pc_safety import available_gb, check_workers
    root, output = Path(root), Path(output)
    if model not in MODEL_SPECS or device not in ('cpu', 'cuda'):
        raise ValueError('unsupported model/device')
    if batch_size != 32 or workers != 4:
        raise ValueError('frozen evaluation uses32 images and4 workers; no fallback')
    check_workers(workers)
    if available_gb() < 12:
        raise ValueError('PC memory safety floor requires12GiB available')
    if device == 'cuda' and not torch.cuda.is_available():
        raise ValueError('GPU unavailable; no fallback')
    recipe, base, variants = validate_panel(root, output)
    target = output/'models'/model
    if target.exists():
        raise ValueError('refusing to overwrite model scores')
    pinned = recipe['models'][model]
    network = load_network(root/pinned['checkpoint'], device)
    torch.set_num_threads(1)
    dataset = ChannelDataset(output, base, variants)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, num_workers=workers,
        shuffle=False, persistent_workers=False, multiprocessing_context='spawn')
    predictions, cell_values = [], []
    started = time.monotonic()
    with torch.inference_mode():
        for images, indices, counts, valid in loader:
            if available_gb() < 12 or time.monotonic()-started > 1800:
                raise ValueError('PC safety memory floor or30min model limit reached')
            images = images.to(device=device, dtype=torch.float32)
            _, head, probabilities = cell_maps(network, images)
            probabilities = probabilities.cpu().numpy()
            if not np.isfinite(probabilities).all():
                raise ValueError('nonfinite tile scores')
            heads = head.cpu().numpy() if model == 'run18' else None
            if heads is not None and not np.isfinite(heads).all():
                raise ValueError('nonfinite cell scores')
            for offset, index in enumerate(indices.tolist()):
                row = dict(image_index=index, prob_stego=float(probabilities[offset]))
                if heads is not None:
                    row.update(cell_statistics(heads[offset], counts[offset].numpy(), valid[offset].numpy()))
                    cell_values.append(heads[offset])
                predictions.append(row)
            print(f'{model}: {len(predictions)}/{len(dataset)} images', flush=True)
    if [p['image_index'] for p in predictions] != list(range(len(dataset))):
        raise ValueError('missing/reordered predictions')
    validate_panel(root, output)  # Recheck all immutable inputs after scoring.
    target.mkdir(parents=True)
    pd.DataFrame(predictions).to_csv(target/'predictions.csv', index=False, lineterminator='\n')
    if cell_values:
        np.save(target/'cell-logits.npy', np.asarray(cell_values, dtype=np.float32), allow_pickle=False)
    dump(target/'execution.json', dict(model=model, recipe_sha256=digest(canonical(recipe).encode()),
        source=pinned, device=device, batch_size=batch_size, workers=workers, precision='fp32',
        rocm_conv1_fix=False, compile=False, augmentation=False, images=len(predictions),
        seconds=time.monotonic()-started, torch_version=torch.__version__,
        predictions_sha256=file_hash(target/'predictions.csv')))
    return target


def weighted_auc(clean, stego, weights):
    """Exact weighted U-statistic, ties half credit; same photo weights on partners."""
    comparison = (stego[:, None] > clean[None, :]) + .5*(stego[:, None] == clean[None, :])
    numerator = ((weights @ comparison)*weights).sum(axis=1)
    return numerator/(weights.sum(axis=1)**2)


def interval(values, coverage=.95):
    values = np.asarray(values, dtype=float)
    finite = values[np.isfinite(values)]
    if not len(finite):
        return dict(low=None, high=None, valid_replicates=0)
    low, high = np.quantile(finite, [(1-coverage)/2, 1-(1-coverage)/2])
    return dict(low=float(low), high=float(high), valid_replicates=len(finite))


def practical_effect(point, bounds, minimum=.02):
    return bool(abs(point) >= minimum and (bounds['low'] > 0 or bounds['high'] < 0))


def bootstrap_weights(base, replicates=5000):
    """One matrix, shared across every model/arm/rate; photos are the draw unit."""
    weights = np.zeros((replicates+1, len(base)), dtype=float)
    weights[0] = 1.
    random = np.random.default_rng(SEED)
    for _, group in base.groupby('stratum', sort=True):
        photos = sorted(group.source_photo.unique())
        draws = random.multinomial(len(photos), np.full(len(photos), 1/len(photos)), size=replicates)
        by_photo = {photo:draws[:, i] for i, photo in enumerate(photos)}
        for index, row in group.iterrows():
            weights[1:, index] = by_photo[row.source_photo]
    return weights


def _summary(values):
    return dict(point=float(values[0]), interval95=interval(values[1:]))


def summarize_model(base, variants, predictions, cutoff, weights):
    clean = predictions.prob_stego.iloc[:len(base)].to_numpy()
    if len(predictions) != len(base)+len(variants) or not np.isfinite(predictions.prob_stego).all():
        raise ValueError('incomplete/nonfinite predictions')
    strata = [g.index.to_numpy() for _, g in base.groupby('stratum', sort=True)]
    groups = {name:g.index.to_numpy() for name, g in base.groupby('stratum', sort=True)}
    settings = {}
    rows = []
    for (k, rate, placement), condition in variants.groupby(['k', 'rate', 'placement'], sort=True):
        scores = {}
        for arm in ARMS:
            chosen = condition[condition.arm.eq(arm)].sort_values('base_index')
            if chosen.base_index.tolist() != list(range(len(base))):
                raise ValueError('variant group does not have all clean partners')
            stego = predictions.prob_stego.iloc[len(base)+chosen.variant_id.to_numpy()].to_numpy()
            per_stratum = [weighted_auc(clean[ix], stego[ix], weights[:, ix]) for ix in strata]
            macro = np.mean(per_stratum, axis=0)
            fa = (weights*(clean >= cutoff)).sum(1)/weights.sum(1)
            miss = (weights*(stego < cutoff)).sum(1)/weights.sum(1)
            by_stratum = {}
            for i, (name, ix) in enumerate(groups.items()):
                sfa = (weights[:, ix]*(clean[ix] >= cutoff)).sum(1)/weights[:, ix].sum(1)
                smiss = (weights[:, ix]*(stego[ix] < cutoff)).sum(1)/weights[:, ix].sum(1)
                by_stratum[name] = dict(AUC=per_stratum[i], FA=sfa, miss=smiss, P_E=(sfa+smiss)/2)
            scores[arm] = dict(AUC=macro, P_E=(fa+miss)/2, miss=miss, FA=fa, strata=by_stratum)
            ordinary = weighted_auc(clean, stego, weights[:1])[0]
            rows.append(dict(k=int(k), rate=float(rate), placement=placement, arm=arm,
                macro_AUC=_summary(macro), pooled_AUC=float(ordinary),
                FA=_summary(fa), miss=_summary(miss), P_E=_summary((fa+miss)/2),
                strata={name:{metric:_summary(values) for metric, values in measures.items()}
                        for name, measures in by_stratum.items()}))
        settings[(int(k), float(rate), placement)] = scores
    contrasts = []
    for setting, scores in settings.items():
        for arm, reference in [('r', 'rgb'), ('g', 'rgb'), ('b', 'rgb'), ('native_rgb', 'rgb'),
                               ('r', 'g'), ('r', 'b'), ('g', 'b')]:
            contrasts.append(dict(k=setting[0], rate=setting[1], placement=setting[2],
                contrast=arm+'-'+reference,
                differences={metric:_summary(scores[arm][metric]-scores[reference][metric])
                             for metric in ('AUC', 'P_E', 'miss', 'FA')},
                strata={name:{metric:_summary(scores[arm]['strata'][name][metric]-scores[reference]['strata'][name][metric])
                              for metric in ('AUC', 'P_E', 'miss', 'FA')} for name in groups}))
    primary = {}
    for arm in ARMS:
        primary[arm] = {metric:np.mean([settings[(1, rate, 'random')][arm][metric] for rate in (.05, .15)], axis=0)
                        for metric in ('AUC', 'P_E', 'miss', 'FA')}
    decisions = []
    for arm in ('r', 'g', 'b'):
        delta = primary[arm]['AUC']-primary['rgb']['AUC']
        bounds = interval(delta[1:], 1-.05/3)
        pe = primary[arm]['P_E']-primary['rgb']['P_E']
        pe_bounds = interval(pe[1:])
        decisions.append(dict(contrast=arm+'-rgb', difference=_summary(delta), adjusted_interval=bounds,
            meaningful=practical_effect(float(delta[0]), bounds),
            equivalent=bounds['low'] >= -.02 and bounds['high'] <= .02,
            secondary_PE=_summary(pe), secondary_PE_notable=practical_effect(float(pe[0]), pe_bounds, .01)))
    cell_summary = []
    if 'cell_AUC' in predictions:
        clean_cells = predictions.iloc[:len(base)]
        for (k, rate, placement, arm), selected in variants.groupby(['k', 'rate', 'placement', 'arm'], sort=True):
            if placement != 'contiguous':
                continue
            selected = selected.sort_values('base_index')
            values = predictions.iloc[len(base)+selected.variant_id.to_numpy()]
            aucs = values.cell_AUC.to_numpy()
            eligible = np.isfinite(aucs)
            denominator = (weights*eligible).sum(1)
            with np.errstate(invalid='ignore', divide='ignore'):
                sampled_auc = (weights*np.nan_to_num(aucs)).sum(1)/denominator
            cell_summary.append(dict(k=int(k), rate=float(rate), arm=arm,
                eligible_tiles=int(eligible.sum()), ineligible_tiles=int((~eligible).sum()),
                mean_eligible_tile_AUC=_summary(sampled_auc) if eligible.any() else None,
                positive_cells=int(values.positive_cells.sum()), negative_cells=int(values.negative_cells.sum()),
                recall=_summary((weights*values.positive_hits.to_numpy()).sum(1)/
                    (weights*values.positive_cells.to_numpy()).sum(1)) if values.positive_cells.sum() else None,
                valid_negative_FA=_summary((weights*values.negative_hits.to_numpy()).sum(1)/
                    (weights*values.negative_cells.to_numpy()).sum(1)) if values.negative_cells.sum() else None))
        clean_cell_fa = _summary((weights*clean_cells.negative_hits.to_numpy()).sum(1)/
            (weights*clean_cells.negative_cells.to_numpy()).sum(1))
    else:
        clean_cell_fa = None
    return dict(cutoff=cutoff, settings=rows, contrasts=contrasts,
        primary={arm:{m:_summary(v) for m, v in measures.items()} for arm, measures in primary.items()},
        primary_contrasts=decisions, all_primary_equivalent=all(d['equivalent'] for d in decisions),
        cell_contiguous=cell_summary, clean_cell_FA=clean_cell_fa)


def report(root, output):
    output = Path(output)
    recipe, base, variants = validate_panel(root, output)
    if (output/'report.json').exists() or (output/'REPORT.md').exists():
        raise ValueError('refusing to overwrite report')
    weights = bootstrap_weights(base, recipe['reading']['bootstrap_replicates'])
    results = {}
    for name in MODEL_SPECS:
        folder = output/'models'/name
        context = json.loads((folder/'execution.json').read_text())
        if (context['recipe_sha256'] != digest(canonical(recipe).encode()) or
                context['source'] != recipe['models'][name] or
                context['predictions_sha256'] != file_hash(folder/'predictions.csv')):
            raise ValueError('prediction identity mismatch')
        predictions = pd.read_csv(folder/'predictions.csv', float_precision='round_trip')
        if list(predictions.image_index) != list(range(len(base)+len(variants))):
            raise ValueError('prediction row identity mismatch')
        print('CPU bootstrap/report: '+name, flush=True)
        results[name] = summarize_model(base, variants, predictions, recipe['models'][name]['cutoff'], weights)
    dump(output/'report.json', dict(recipe_sha256=digest(canonical(recipe).encode()),
        reading=recipe['reading'], bootstrap_draws_sha256=digest(weights.astype('<f8').tobytes()), models=results))
    lines = ['# RGB channel analysis — fixed TEST diagnostic', '',
             'Recipe SHA256: '+digest(canonical(recipe).encode()), '',
             'Run18 primary: k1 random .05/.15, equal source/noise AUC. No threshold fit.', '',
             '| Run18 contrast | Delta AUC | Adjusted interval | Meaningful >=.02 |',
             '|---|---:|---|---|']
    for row in results['run18']['primary_contrasts']:
        bounds = row['adjusted_interval']
        lines.append(f"| {row['contrast']} | {row['difference']['point']:.6f} | [{bounds['low']:.6f}, {bounds['high']:.6f}] | {row['meaningful']} |")
    lines += ['', 'Tight practical equivalence across all three: '+str(results['run18']['all_primary_equivalent']), '',
              'All settings/strata/cutoff errors and paired intervals are in report.json. Pooled AUC is a secondary point estimate.',
              'Cell localization is secondary; undefined maps stay undefined, with eligibility counts.',
              'Single channel is denser AND changes colour relations. Native RGB also changes spatial support.',
              'No result proves that explicit colour features help. No training/model/threshold decision is authorized.',
              'Existing TEST photos were reused; this is not an untouched benchmark.']
    (output/'REPORT.md').write_text('\n'.join(lines)+'\n')
    return results
