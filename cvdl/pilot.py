"""PC-only pilots: cap draws after real recipe verification; never edit recipes."""
from dataclasses import replace
import hashlib
import json
from pathlib import Path

import numpy as np

MAX_TILES = 4096
VALIDATION_PAIRS = 256
SLICE_SEED = 20261019


def apply(otf, settings):
    """Called only after NotebookData has verified the complete real recipe.

    The declared run settings remain the identity checked by load_frozen. Only
    this corpus's effective draw budget changes. Validation uses original rows,
    including their frozen embedding keys; no recipe is built or regenerated.
    """
    from .validation import MatchedValidationPairs, csv_text
    import pandas as pd
    full = otf.val.frame.sort_values('pair_id').reset_index(drop=True)
    if len(full) < VALIDATION_PAIRS:
        raise ValueError('pilot needs at least 256 pairs in the real validation list')
    rng = np.random.default_rng(SLICE_SEED)
    selected = np.sort(rng.choice(len(full), VALIDATION_PAIRS, replace=False))
    frame = full.iloc[selected].reset_index(drop=True)
    declared = otf.corpus.settings.tiles_per_epoch
    effective = min(declared, MAX_TILES)
    otf.corpus.settings = replace(otf.corpus.settings, tiles_per_epoch=effective)
    otf.val = MatchedValidationPairs(otf.corpus, frame)
    otf.validation_pair_csv = csv_text(frame)
    selected_sha = hashlib.sha256(otf.validation_pair_csv.encode()).hexdigest()
    marker = dict(mode='pc-pilot-v1', declared_tiles_per_epoch=declared,
        effective_tiles_per_epoch=effective, validation_recipe=settings.validation_recipe,
        full_validation_pairs=len(full), full_validation_sha256=otf.record['validation_recipe']['pairs_sha256'],
        slice_pairs=len(frame), slice_seed=SLICE_SEED, slice_order='sort pair_id; seeded choice without replacement; sorted selected positions',
        slice_sha256=selected_sha, pair_ids=frame.pair_id.tolist(),
        tile_sizes=sorted(int(n) for n in frame.tile_size.unique()), deployment_eligible=False)
    otf.record.update(otf.corpus.run_record)
    otf.record['pilot'] = marker
    otf.record['validation_recipe'].update(pairs=len(frame), selected_pairs_sha256=selected_sha,
        source_counts=frame.source.value_counts().to_dict(), pilot=True)
    otf.pairs = pd.concat([otf.fixed_train.frame, otf.val.frame, otf.test.frame], ignore_index=True)


def refuse_run(folder):
    """Reject pilot artifacts before checkpoint inference or report loading."""
    folder = Path(folder)
    paths = [folder/'run_settings.json', folder/'data_run_settings.json']
    paths += list(folder.glob('checkpoints/*/run_settings.json'))
    paths += list(folder.glob('checkpoints/*/data_run_settings.json'))
    for path in paths:
        if path.is_file():
            record = json.loads(path.read_text())
            if record.get('pilot') or record.get('values', {}).get('pilot'):
                raise ValueError('pilot runs are smoke tests, not eligible for evaluation or reports')
