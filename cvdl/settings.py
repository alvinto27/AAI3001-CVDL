"""Typed settings, automatic machine defaults, legacy env bridge and provenance.

Order: schema defaults/profile -> notebook -> explicit override. Legacy env
fills unset fields; conflicting notebook/env values need an explicit override.
Machine-owned fields cannot be supplied by a notebook. No data is loaded here.
"""
from dataclasses import dataclass
from contextlib import contextmanager
from contextvars import ContextVar
from difflib import get_close_matches
import json
import math
import os
import re
from pathlib import Path
import subprocess
from types import MappingProxyType

ROOT = Path(__file__).resolve().parents[1]
_RUNNER_CONTEXT = ContextVar('cvdl_runner_settings', default=None)


@contextmanager
def runner_context(environ, overrides):
    """Keep explicit runner overrides/provenance when a top cell calls load()."""
    token = _RUNNER_CONTEXT.set((dict(environ), tuple(overrides)))
    try:
        yield
    finally:
        _RUNNER_CONTEXT.reset(token)


def boolean(x):
    if type(x) is bool:
        return x
    if isinstance(x, str) and x.lower() in {'true', '1', 'yes', 'on', 'false', '0', 'no', 'off'}:
        return x.lower() in {'true', '1', 'yes', 'on'}
    raise ValueError('expected bool (true/false/on/off)')


def integer(x):
    if isinstance(x, bool) or isinstance(x, float) and not x.is_integer():
        raise ValueError('expected integer, not bool/fraction')
    return int(x)


def number(x):
    if isinstance(x, bool):
        raise ValueError('expected finite number, not bool')
    x = float(x)
    if not math.isfinite(x):
        raise ValueError('expected finite number')
    return x


def optional_float(x):
    return None if x is None or str(x).lower() in {'none', 'noclip'} else number(x)


def optional_string(x):
    return None if x is None or x == '' else str(x)


def sequence(x):
    return tuple(part.strip() for part in x.split(',')) if isinstance(x, str) else tuple(x)


def mixture(x):
    if x is None or x == '':
        return None
    if isinstance(x, str):
        pairs = [item.strip().split(':') for item in x.split(',')]
        if any(len(p) != 2 for p in pairs) or len({p[0] for p in pairs}) != len(pairs):
            raise ValueError('expected unique source:weight pairs')
        x = dict(pairs)
    return {str(k): number(v) for k, v in x.items()}


def drops(x):
    if x is None:
        return None
    pairs = [p.split(':') for p in x.split(',')] if isinstance(x, str) else x
    if not pairs or any(len(p) != 2 for p in pairs):
        raise ValueError('expected epoch:factor pairs')
    return tuple((integer(a), number(b)) for a, b in pairs)


def speed_pairs(value):
    if value is None or value == '':
        return None
    entries = [item.split(':') for item in value.split(',')] if isinstance(value,str) else value
    pairs = []
    for fields in entries:
        if len(fields) != 2:
            raise ValueError('expected batch:precision pairs')
        pair = integer(fields[0]), str(fields[1])
        if pair[0] not in SCHEMA['speed_batches'].default or pair[1] not in SCHEMA['speed_precisions'].default or pair in pairs:
            raise ValueError('invalid or repeated batch:precision pair')
        pairs.append(pair)
    if not pairs:
        raise ValueError('expected at least one batch:precision pair')
    return tuple(pairs)


def compile_mode(x):
    return 'off' if str(x).lower() in {'', '0', 'false', 'no', 'off'} else str(x).lower()


@dataclass(frozen=True)
class Definition:
    kind: str
    parse: object
    default: object
    allowed: str
    meaning: str
    check: object = lambda v: True
    env: str | None = None
    legacy_global: str | None = None
    machine: bool = False
    read_only: bool = False


# The only definition table. Callable defaults depend on earlier resolved values.
SCHEMA = MappingProxyType({
    'root': Definition('path', str, str(ROOT), 'repository path', 'Checkout root.', machine=True, read_only=True),
    'device': Definition('str', str, 'cpu', 'cpu/cuda', 'PyTorch device.', lambda v: v in {'cpu','cuda'}, machine=True, read_only=True),
    'rocm': Definition('bool', boolean, False, 'bool', 'ROCm runtime available.', machine=True, read_only=True),
    'precision_limits': Definition('tuple', sequence, ('32',), 'supported precisions', 'Hard machine precision limits.', machine=True, read_only=True),
    'quick': Definition('bool', boolean, True, 'bool', 'Small trial mode.', env='QUICK', legacy_global='QUICK', machine=True),
    'workers': Definition('int', integer, 6, '>=0', 'Training workers.', lambda v:v>=0, 'V3_WORKERS', 'OTF_WORKERS', True),
    'eval_workers': Definition('int', integer, 4, '>=0', 'Evaluation workers.', lambda v:v>=0, 'V3_EVAL_WORKERS', 'EVAL_WORKERS', True),
    'diag_workers': Definition('int', integer, lambda c:c['eval_workers'], '>=0', 'Diagnostic photo workers.', lambda v:v>=0, 'V3_DIAG_WORKERS', machine=True),
    'headless': Definition('bool', boolean, False, 'bool', 'Compact headless progress.', env='CVDL_HEADLESS_LOG', machine=True),
    'main_threads': Definition('int', integer, 1, '>0', 'Main PyTorch threads.', lambda v:v>0, 'V3_MAIN_THREADS', 'MAIN_THREADS', True),
    'precision': Definition('str', str, '32', '32/bf16-mixed within profile limits', 'Training precision.', lambda v:v in {'32','bf16-mixed'}, 'V3_PRECISION', 'PRECISION', True),
    'persistent_workers': Definition('bool', boolean, True, 'bool', 'Persistent validation workers.', env='V3_PERSISTENT_WORKERS', machine=True),
    'compile': Definition('str', compile_mode, 'off', 'off/default', 'Inner network compilation.', lambda v:v in {'off','default'}, 'V3_COMPILE', 'COMPILE_MODE', True),
    'compile_threads': Definition('int', integer, 4, '>0', 'Inductor compile worker limit.', lambda v:v>0, 'TORCHINDUCTOR_COMPILE_THREADS', machine=True),
    'inductor_cache': Definition('path?', optional_string, None, 'absolute path outside checkout', 'Inductor cache path.', env='TORCHINDUCTOR_CACHE_DIR', machine=True),
    'triton_cache': Definition('path?', optional_string, None, 'absolute path outside checkout', 'Triton cache path.', env='TRITON_CACHE_DIR', machine=True),
    'rocm_conv1_fix': Definition('bool', boolean, True, 'bool', 'ROCm conv1 gradient workaround.', env='V3_ROCM_CONV1_FIX', legacy_global='ROCM_CONV1_FIX', machine=True),
    'output_root': Definition('path', str, lambda c:str(Path(c['root'])/'runs'), 'path', 'Run output root.', env='V3_OUTPUT_ROOT', legacy_global='CHECKPOINT_ROOT', machine=True),
    'cache_root': Definition('path?', optional_string, None, 'path or None', 'Optional decoded cache.', env='V3_CACHE_ROOT', machine=True),
    'manifest_root': Definition('path?', optional_string, None, 'path or None', 'Optional new-source record root.', env='V3_NEW_DATA_ROOT', machine=True),
    'data_root': Definition('path', str, lambda c:str(Path(c['root'])/'Datasets/generated/raise512-run1-records'), 'path', 'Legacy PNG records.', legacy_global='DATA_ROOT', machine=True),
    'labels_path': Definition('path', str, lambda c:str(Path(c['data_root'])/'labels.csv'), 'path', 'Legacy label CSV.', legacy_global='LABELS_PATH', machine=True),
    'split_path': Definition('path', str, lambda c:str(Path(c['data_root'])/'splits/split_seed0.csv'), 'path', 'Legacy split CSV.', legacy_global='SPLIT_PATH', machine=True),
    'quick_train_pairs': Definition('int', integer, 32, '>0', 'QUICK PNG train pairs.', lambda v:v>0, machine=True),
    'quick_eval_pairs': Definition('int', integer, 16, '>0', 'QUICK evaluation/extra-panel pairs.', lambda v:v>0, machine=True),
    'seed': Definition('int', integer, 0, '>=0', 'Framework seed.', lambda v:v>=0, legacy_global='SEED'),
    'paired_batches': Definition('bool', boolean, False, 'bool', 'Paired training batches.', legacy_global='PAIRED_BATCHES'),
    'data': Definition('str', str, 'otf', 'otf/png', 'Data mode.', lambda v:v in {'otf','png'}, 'V3_DATA', 'V3_DATA'),
    'clip_t': Definition('float?', optional_float, 3., 'None or >0', 'Residual clipping.', lambda v:v is None or v>0, 'V3_CLIP_T', 'CLIP_T'),
    'norm': Definition('str', str, 'none', 'none/batch/group', 'Network normalization.', lambda v:v in {'none','batch','group'}, 'V3_NORM', 'NORM'),
    'epochs': Definition('int', integer, lambda c:2 if c['quick'] else 20, '>0', 'Training epochs.', lambda v:v>0, 'V3_EPOCHS', 'EPOCHS'),
    'limit_train_batches': Definition('int', integer, 0, '>=0; 0=full epoch', 'Training batch cap.', lambda v:v>=0, 'V3_LIMIT_TRAIN_BATCHES', 'LIMIT_TRAIN_BATCHES'),
    'lr': Definition('float', number, .0003, '>0', 'AdamW learning rate.', lambda v:v>0, 'V3_LR', 'LR'),
    'warmup_epochs': Definition('int', integer, 1, '0..epochs', 'Warmup duration.', lambda v:v>=0, 'V3_WARMUP_EPOCHS', 'WARMUP_EPOCHS'),
    'grad_clip': Definition('float', number, 1., '>=0; 0=off', 'Gradient norm clipping.', lambda v:v>=0, 'V3_GRAD_CLIP', 'GRAD_CLIP'),
    'lr_drops': Definition('tuple?', drops, lambda c:((c['epochs']//2+1,.1),), 'None=automatic; increasing epoch:factor; factor (0,1]', 'LR drop plan.', env='V3_LR_DROPS', legacy_global='LR_DROPS'),
    'train_yaml': Definition('path?', optional_string, None, 'path or None; OTF only', 'Train embedding settings.', env='V3_TRAIN_YAML', legacy_global='TRAIN_YAML'),
    'train_grid': Definition('str', str, 'recorded', 'recorded/overlap_last/centre_trim/centre_pad', 'Training-only tiling rule.', lambda v:v in {'recorded','overlap_last','centre_trim','centre_pad'}),
    'post_fit_eval': Definition('str', str, 'full', 'full/none', 'Run post-fit evaluation.', lambda v:v in {'full','none'}),
    'name': Definition('str?', optional_string, None, 'notebook stem or None', 'Short experiment notebook name.'),
    'extra_tests': Definition('str', str, 'test-bpp1', 'none/test-bpp1/source-panels or both panels', 'Extra evaluation panels.', lambda v:set(v.split(','))<={'none','test-bpp1','source-panels'} and ('none' not in v or v=='none'), 'V3_EXTRA_TESTS', 'EXTRA_TESTS'),
    'model': Definition('str', str, 'xunet', 'xunet/xunet_wide', 'Network architecture.', lambda v:v in {'xunet','xunet_wide'}, 'V3_MODEL', 'MODEL'),
    'width': Definition('int', integer, 4, '1..8', 'Wide network multiplier.', lambda v:1<=v<=8, 'V3_WIDTH', 'WIDTH'),
    'batch_size': Definition('int', integer, lambda c:8 if c['quick'] else 32, 'positive multiple of 4', 'Training images per batch.', lambda v:v>0 and v%4==0, 'V3_BATCH_SIZE', 'BATCH_SIZE'),
    'source': Definition('str', str, 'decode', 'decode/cache', 'Photo input path.', lambda v:v in {'decode','cache'}, 'V3_SOURCE', 'OTF_SOURCE'),
    'source_mix': Definition('dict?', mixture, None, 'positive weights summing to 1', 'Fixed tile source mixture.', env='V3_SOURCE_MIX'),
    'sources': Definition('tuple', sequence, lambda c:tuple(c['source_mix']) if c['source_mix'] else ('raise','div2k'), 'unique known sources', 'Selected train sources.', env='V3_SOURCES'),
    'tiles_per_epoch': Definition('int?', lambda x:None if x is None else integer(x), lambda c:(225 if c['quick'] else 45000) if c['source_mix'] else None, 'positive for mixture; otherwise None', 'Mixed epoch tile count.', env='V3_TILES_PER_EPOCH'),
    'tile_size': Definition('int', integer, 512, '256/512', 'Square tile size.', lambda v:v in {256,512}, 'V3_TILE_SIZE'),
    'run_seed': Definition('int', integer, 0, '>=0', 'Keyed data seed.', lambda v:v>=0, 'V3_RUN_SEED'),
    'train_photo_fraction': Definition('float', number, lambda c:.005 if c['quick'] else 1., '(0,1]', 'Train photo fraction.', lambda v:0<v<=1, 'V3_TRAIN_PHOTO_FRACTION'),
    'checkpoint_validation': Definition('str', str, lambda c:'capped' if c['source_mix'] else 'standard', 'standard/combined/capped', 'Checkpoint validation selection.', lambda v:v in {'standard','combined','capped'}, 'V3_CHECKPOINT_VAL'),
    'val_cap_raise': Definition('int', integer, 5000, '>0', 'Capped RAISE validation pairs.', lambda v:v>0, 'V3_VAL_CAP_RAISE'),
    'val_cap_lsdir': Definition('int', integer, 1500, '>0', 'Capped LSDIR validation pairs.', lambda v:v>0, 'V3_VAL_CAP_LSDIR'),
    'eval_profile': Definition('str', str, 'overlap_full_mix', 'overlap_full_mix/legacy_v6', 'Fixed evaluation grid.', lambda v:v in {'overlap_full_mix','legacy_v6'}, 'V3_EVAL_PROFILE', 'EVAL_PROFILE'),
    'prefetch': Definition('int?', lambda x:None if x is None else integer(x), None, 'None or >0', 'Prefetch override.', lambda v:v is None or v>0, 'V3_PREFETCH', 'OTF_PREFETCH', True),
    'prefetch_images': Definition('int', integer, 128, '>0', 'Prefetch image budget.', lambda v:v>0, 'V3_PREFETCH_IMAGES', 'OTF_PREFETCH_IMAGES', True),
    'prime_next_epoch': Definition('bool', boolean, False, 'bool', 'Prime next train epoch.', env='V3_PRIME_NEXT_EPOCH', legacy_global='PRIME_NEXT_EPOCH', machine=True),
    'run_name': Definition('str?', optional_string, None, 'derived prefix + optional suffix', 'Derived run name / legacy suffix.', env='V3_RUN_NAME', legacy_global='RUN_NAME'),
    'notebook': Definition('path?', optional_string, None, 'notebook path', 'Runner notebook selector.', env='V3_NOTEBOOK', machine=True),
    'job_id': Definition('str?', optional_string, None, 'job identity or None', 'G4 job identity.', env='CVDL_JOB_ID', machine=True),
    'speed_matrix': Definition('bool', boolean, False, 'bool', 'Legacy speed tool mode.', env='V3_SPEED_MATRIX', machine=True),
    'speed_batches': Definition('tuple', lambda x:tuple(map(integer,sequence(x))), (32,64,128), 'distinct 32/64/128', 'Speed-tool batch sweep.', lambda v:bool(v) and len(v)==len(set(v)) and set(v)<={32,64,128}, 'V3_SPEED_BATCHES', machine=True),
    'speed_precisions': Definition('tuple', sequence, ('32','bf16-mixed'), '32/bf16-mixed', 'Speed-tool precision sweep.', lambda v:bool(v) and len(v)==len(set(v)) and set(v)<={'32','bf16-mixed'}, 'V3_SPEED_PRECISIONS', machine=True),
    'speed_pairs': Definition('tuple?', speed_pairs, None, 'distinct 32/64/128:32/bf16-mixed pairs', 'Optional speed-tool pairs.', env='V3_SPEED_PAIRS', machine=True),
    'speed_steps': Definition('int', integer, 600, '>0', 'Speed-tool measured steps.', lambda v:v>0, 'V3_SPEED_STEPS', machine=True),
    'speed_warmup': Definition('int', integer, 50, '>0', 'Speed-tool warmup steps.', lambda v:v>0, 'V3_SPEED_WARMUP', machine=True),
})
ALIASES = MappingProxyType({spec.env:name for name,spec in SCHEMA.items() if spec.env})


@dataclass(frozen=True)
class MachineProfile:
    name: str
    values: object


def machine_profile(name, *, root=ROOT, device='cuda', rocm=False, precision_limits=('32','bf16-mixed')):
    """Declared machine values; preflight may use G4 values without a local GPU."""
    if name not in {'pc','g4'}:
        raise ValueError('machine profile must be pc/g4')
    g4 = name == 'g4'
    values = dict(root=str(root), output_root=str(Path(root)/('runs-out' if g4 else 'runs')),
        device=device, rocm=rocm, quick=False if g4 else SCHEMA['quick'].default,
        workers=32 if g4 else SCHEMA['workers'].default,
        eval_workers=16 if g4 else SCHEMA['eval_workers'].default,
        precision_limits=precision_limits)
    if g4:
        values.update(compile='default', prime_next_epoch=True)
    return MachineProfile(name, MappingProxyType(values))


def automatic_profile():
    """Detect G4 from Google DMI + CUDA RTX PRO 6000; reject contradictions."""
    import torch
    dmi = []
    for field in ('product_name','sys_vendor'):
        try:
            dmi.append(Path('/sys/class/dmi/id',field).read_text().strip())
        except OSError:
            dmi.append('')
    gce = dmi[0] == 'Google Compute Engine' or dmi[1] == 'Google'
    gpu = torch.cuda.is_available()
    rocm = torch.version.hip is not None
    gpu_name = torch.cuda.get_device_name(0) if gpu and not rocm else ''
    if gce and (not gpu or rocm or 'RTX PRO 6000' not in gpu_name):
        raise ValueError(f'G4 profile mismatch: Google DMI requires CUDA RTX PRO 6000, found {gpu_name or "no CUDA GPU"}')
    if not gce and 'RTX PRO 6000' in gpu_name:
        raise ValueError('G4 profile mismatch: RTX PRO 6000 found without Google DMI; refuse PC fallback')
    name = 'g4' if gce else 'pc'
    expected = os.environ.get('CVDL_EXPECTED_PROFILE')
    if expected is not None and expected != name:
        raise ValueError(f'machine profile mismatch: job requires {expected!r}, detected {name!r}; DMI={dmi!r}')
    precisions = ('32','bf16-mixed') if gpu and torch.cuda.is_bf16_supported() else ('32',)
    return machine_profile(name, device='cuda' if gpu else 'cpu', rocm=rocm, precision_limits=precisions)


def canonical(name):
    if name in SCHEMA:
        return name
    if name in ALIASES:
        return ALIASES[name]
    hint = get_close_matches(name, list(SCHEMA)+list(ALIASES), n=1)
    raise ValueError(f"unknown setting {name!r}" + (f"; did you mean {hint[0]!r}?" if hint else ''))


def parse(name, value):
    spec = SCHEMA[canonical(name)]
    try:
        result = spec.parse(value)
        if not spec.check(result):
            raise ValueError('outside allowed range')
        return result
    except (ValueError, TypeError, AttributeError, OverflowError) as error:
        hints = get_close_matches(str(value), re.findall(r'[a-z][a-z0-9_-]*', spec.allowed), n=1) if spec.kind == 'str' else []
        hint = f'; did you mean {hints[0]!r}?' if hints else ''
        raise ValueError(f"setting {name}={value!r}: expected {spec.kind}; allowed {spec.allowed}: {error}{hint}") from error


def override_values(items):
    if isinstance(items, dict):
        return list(items.items())
    pairs = []
    for item in items:
        if '=' not in item:
            raise ValueError(f'override {item!r}: expected key=value')
        pairs.append(item.split('=',1))
    return pairs


def freeze(value):
    if isinstance(value, dict):
        return MappingProxyType({k:freeze(v) for k,v in value.items()})
    if isinstance(value, (tuple,list)):
        return tuple(freeze(v) for v in value)
    return value


def plain(value):
    if hasattr(value,'items'):
        return {k:plain(v) for k,v in value.items()}
    if isinstance(value,(tuple,list)):
        return [plain(v) for v in value]
    return value


@dataclass(frozen=True)
class Settings:
    values: object
    provenance: object
    profile: str
    overrides: tuple

    def __getattr__(self, name):
        try:
            return self.values[name]
        except KeyError as error:
            raise AttributeError(name) from error

    @property
    def run_directory_name(self):
        return (self.job_id or self.run_name) if self.name else self.run_name

    def to_run_settings(self):
        from lsb_torch import RunSettings
        return RunSettings(source=self.source, cache_root=self.cache_root,
            settings_yaml=self.train_yaml, tile_size=self.tile_size, run_seed=self.run_seed,
            train_photo_fraction=self.train_photo_fraction, sources=tuple(self.values['sources']),
            source_mix=None if self.source_mix is None else dict(self.source_mix), tiles_per_epoch=self.tiles_per_epoch,
            manifest_root=self.manifest_root, paired_batches=self.paired_batches, tensor_dtype='uint8')

    def speed_configuration(self):
        if self.data != 'otf' or self.source != 'decode':
            raise ValueError('speed_matrix requires data=otf and source=decode')
        pairs = [(batch, precision) for batch in self.speed_batches for precision in self.speed_precisions]
        if self.speed_pairs:
            pairs = list(self.speed_pairs)
        if any(precision not in self.precision_limits for _,precision in pairs):
            raise ValueError('speed_pairs exceeds machine precision limits')
        return pairs, self.speed_steps, self.speed_warmup, self.workers

    def legacy_globals(self):
        from importlib.metadata import version
        out = {spec.legacy_global:plain(self.values[name]) for name,spec in SCHEMA.items() if spec.legacy_global}
        out.update(EVAL_PERSISTENT_WORKERS=self.persistent_workers and self.eval_workers>0,
            QUICK_PAIRS={'train':self.quick_train_pairs,'val':self.quick_eval_pairs,'test':self.quick_eval_pairs},
            DEVICE=self.device, ROCM_CONV1_FIX_ACTIVE=self.rocm_conv1_fix and self.rocm and self.device=='cuda',
            PIL_VERSION=version('Pillow'))
        return out

    def apply_runtime_environment(self):
        """Set the supported compiler controls before lazy compiler imports."""
        if self.compile == 'default':
            os.environ['TORCHINDUCTOR_COMPILE_THREADS'] = str(self.compile_threads)
        for name, variable in (('inductor_cache','TORCHINDUCTOR_CACHE_DIR'), ('triton_cache','TRITON_CACHE_DIR')):
            if self.values[name] is not None:
                os.environ[variable] = self.values[name]

    def write(self, folder):
        folder = Path(folder); folder.mkdir(parents=True, exist_ok=True)
        commit = subprocess.check_output(['git','rev-parse','HEAD'],cwd=self.root,text=True).strip()
        dirty = bool(subprocess.check_output(['git','status','--porcelain'],cwd=self.root,text=True).strip())
        record = dict(values=plain(self.values), sources=plain(self.provenance), profile=self.profile,
                      overrides=plain(self.overrides), git=dict(commit=commit,dirty=dirty))
        (folder/'run_settings.json').write_text(json.dumps(record,indent=2)+'\n')
        return record


def validate_notebook_name(path, values):
    """New experiment cells state a name; paths, not optional prose, enforce it."""
    path=Path(path)
    if path.parent.name=='experiments' or values.get('name') is not None:
        if path.parent.name!='experiments' or values.get('name')!=path.stem:
            raise ValueError('experiment name must equal its notebooks/experiments file stem')


def _run_name(c, explicit):
    if c['name'] is not None:
        name=c['name']
        if not name or not name.isascii() or not name[0].isalnum() or not all(ch.isalnum() or ch in '_-' for ch in name):
            raise ValueError('name must be a simple ASCII notebook stem')
        if c['run_name'] is not None and c['run_name']!=name:
            raise ValueError('name and run_name must agree')
        return name
    if c['run_name'] is not None and (c['run_name'] in {'.','..'} or not c['run_name'].isascii()
            or not all(ch.isalnum() or ch in '._-' for ch in c['run_name'])):
        raise ValueError('run_name must be a simple ASCII path component')
    if c['speed_matrix']:
        return c['run_name'] or 'g4-speed-batch-bf16'
    clip = 'noclip' if c['clip_t'] is None else f"clip{c['clip_t']:g}"
    norm = 'nonorm' if c['norm']=='none' else c['norm']
    schedule = f"lr{c['lr']:g}_warm{c['warmup_epochs']}"
    schedule += ('_drops'+'-'.join(f'{epoch}x{factor:g}' for epoch,factor in c['lr_drops'])
                 if 'lr_drops' in explicit else f"_drop{c['epochs']//2}")
    prefix = f"xunet_v4_{clip}_{norm}_{schedule}"
    if c['train_yaml']:
        from lsb_batch import read_config
        k = tuple(read_config(c['train_yaml']).k)
        tag = f'k{k[0]}to{k[-1]}' if k==tuple(range(k[0],k[-1]+1)) else 'k'+'-'.join(map(str,k))
        prefix += '_'+tag
    prefix += f"_e{c['epochs']}"
    if c['model']=='xunet_wide': prefix += f"_wide-w{c['width']}"
    if c['batch_size'] != (8 if c['quick'] else 32): prefix += f"_b{c['batch_size']}"
    if c['precision']!='32': prefix += f"_prec-{c['precision']}"
    name = c['run_name'] or prefix+('_quick' if c['quick'] else '')+('_otf' if c['data']=='otf' else '')
    if not name.startswith(prefix) or (len(name)>len(prefix) and name[len(prefix)] not in '_-'):
        raise ValueError(f'run_name must start with derived prefix {prefix!r}')
    if not name.isascii() or not all(ch.isalnum() or ch in '._-' for ch in name):
        raise ValueError('run_name must be a simple ASCII path component')
    return name


def load(*, overrides=(), environ=None, _profile=None, **notebook_values):
    """Resolve settings without side effects. _profile is a test seam, not a notebook setting."""
    override_pairs = override_values(overrides)
    context = _RUNNER_CONTEXT.get() if environ is None else None
    if context is not None:
        environ, inherited_overrides = context
        override_pairs = [*override_values(inherited_overrides), *override_pairs]
    env = os.environ if environ is None else environ
    profile = automatic_profile() if _profile is None else _profile
    supplied, origins, history = {}, {}, []
    def put(name, raw, origin):
        name = canonical(name)
        if SCHEMA[name].read_only and origin != 'profile':
            raise ValueError(f'setting {name!r} is a read-only machine capability/path')
        supplied[name] = parse(name, raw); origins[name] = origin
        if origin=='override': history.append(dict(name=name,value=plain(supplied[name])))
    for name,value in profile.values.items():
        put(name,value,'profile')
    for name,value in notebook_values.items():
        resolved = canonical(name)
        if SCHEMA[resolved].machine:
            raise ValueError(f'setting {name!r} is machine-owned; use the automatic profile or a one-off override, not the notebook')
        put(name,value,'notebook')
    explicit_overrides = [(canonical(name), value) for name,value in override_pairs]
    override_names = {name for name,_ in explicit_overrides}
    for name,value in env.items():
        if name in ALIASES or name.startswith('V3_'):
            resolved = canonical(name)
            if origins.get(resolved) == 'notebook':
                if supplied[resolved] == parse(resolved, value):
                    continue  # The tracked notebook remains the source of an equal value.
                if resolved not in override_names:
                    raise ValueError(f'setting {resolved!r}: notebook value {supplied[resolved]!r} '
                                     f'conflicts with env {name}={value!r}; use --set for a one-off override')
            put(name,value,'env')
    for name,value in explicit_overrides:
        put(name,value,'override')
    c, origins_all = {}, {}
    for name in SCHEMA:
        spec = SCHEMA[name]
        raw = supplied[name] if name in supplied else spec.default(c) if callable(spec.default) else spec.default
        if name == 'lr_drops' and raw is None:
            raw = spec.default(c)
        c[name] = parse(name, raw); origins_all[name] = origins.get(name,'default')
    if c['warmup_epochs']>c['epochs']: raise ValueError('warmup_epochs must be <= epochs')
    if c['precision'] not in c['precision_limits']: raise ValueError(f"precision {c['precision']!r} exceeds profile limits {c['precision_limits']}")
    if supplied.get('lr_drops') is not None:
        previous = 1
        for epoch,factor in c['lr_drops']:
            if not previous<epoch<=c['epochs'] or not 0<factor<=1: raise ValueError('lr_drops needs increasing epochs in 2..epochs and factors in (0,1]')
            previous=epoch
    if c['train_yaml']:
        if c['data']!='otf': raise ValueError('train_yaml requires data=otf')
        c['train_yaml']=str((Path(c['root'])/c['train_yaml']).resolve())
        from lsb_batch import read_config
        try:read_config(c['train_yaml'])
        except (OSError,ValueError) as error:
            raise ValueError(f"invalid train_yaml {c['train_yaml']!r}: {error}") from error
    c['persistent_workers'] = c['persistent_workers'] and c['eval_workers'] > 0
    if c['compile'] == 'default':
        if 'MIOPEN_GEMM_ENFORCE_BACKEND' in env:
            raise ValueError('unset MIOPEN_GEMM_ENFORCE_BACKEND and restart before compiling')
        for name in ('inductor_cache','triton_cache'):
            if c[name] is not None:
                path = Path(c[name]).expanduser()
                root = Path(c['root']).resolve()
                if not path.is_absolute() or path.resolve()==root or root in path.resolve().parents:
                    raise ValueError(f'{name} must be an absolute path outside the repository')
    if c['checkpoint_validation']=='capped' and not {'raise','alaska','lsdir'} <= set(c['sources']):
        raise ValueError('checkpoint_validation=capped needs raise, alaska, and lsdir')
    if set(c['sources']) & {'alaska','lsdir'} and c['eval_profile'] != 'overlap_full_mix':
        raise ValueError('new sources require eval_profile=overlap_full_mix')
    if c['source']!='cache': c['cache_root']=None
    if c['source_mix'] and tuple(c['source_mix'])!=c['sources']: raise ValueError('sources conflicts with source_mix order')
    c['run_name']=_run_name(c, {name for name,value in supplied.items() if name!='lr_drops' or value is not None})
    if c['name'] and c['job_id'] is None and _profile is None:
        from datetime import datetime, timezone
        c['job_id']=f"{c['name']}-interactive-{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}"
        origins_all['job_id']='interactive'
    if c['name'] and c['job_id'] and not c['job_id'].startswith(c['name']+'-'):
        raise ValueError('job_id must start with experiment name')
    if c['train_grid']!='recorded' and c['data']!='otf':
        raise ValueError('train_grid requires OTF data')
    result = Settings(freeze(c),freeze(origins_all),profile.name,freeze(history))
    result.to_run_settings()  # shared source/mix/cache constraints; no photo reads
    if result.speed_matrix:
        result.speed_configuration()
    return result


def legacy_override(item, *, allow_native=False):
    """Translate legacy overrides, or validate a native thin-notebook setting."""
    if '=' not in item:
        raise ValueError('--set expects key=value')
    key, raw = (part.strip() for part in item.split('=',1))
    name = canonical(key)
    spec = SCHEMA[name]
    if not spec.env:
        if allow_native:
            return None, parse(name, raw)
        raise ValueError(f'{name} has no legacy notebook bridge; use the thin notebook')
    value = parse(name, raw)
    if isinstance(value, bool):
        value = ('on' if value else 'off') if name in {'persistent_workers','speed_matrix'} else str(value).lower()
    elif hasattr(value, 'items'):
        value = ','.join(f'{k}:{v}' for k,v in value.items())
    elif isinstance(value, tuple):
        if name=='lr_drops': value = ','.join(f'{a}:{b}' for a,b in value)
        elif name=='speed_pairs': value = ','.join(f'{a}:{b}' for a,b in value)
        else: value = ','.join(map(str,value))
    return spec.env, '' if value is None else str(value)


def extra_panel_limit(*, quick=None, environ=None):
    """Legacy extra-panel bridge; future callers pass their resolved Settings.quick.

    Standalone eval remains full unless QUICK was explicitly requested.
    """
    env = os.environ if environ is None else environ
    active = boolean(env.get('QUICK','false')) if quick is None else boolean(quick)
    return SCHEMA['quick_eval_pairs'].default if active else None
