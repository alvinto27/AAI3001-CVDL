"""Golden "before" values for the notebook-to-package refactor, and a checker.

  capture : run selected checks and write values to tests/golden/refactor-golden.json
  compare : run selected checks into memory and compare with the golden file

Run inside the PC GPU container, from anywhere:
  podman exec -w /work aai3001 python scripts/refactor_golden.py capture
  podman exec -w /work aai3001 python scripts/refactor_golden.py compare --only settings,models

Parts (--only takes a comma list; default excludes GPU):
  settings : effective settings, RUN_NAME and data records for the G4 env files and the QUICK env files,
             plus the standard test settings hash. Nothing is trained.
  data     : first train batches (epochs 0 and 1), val/test frames and first eval batches; workers 0 and 4.
  models   : 4 real checkpoints x 64 real tiles: CPU logits and 16x16 cell map hashes, strict loading.
  cpu      : deterministic CPU run of notebook 05 (20 steps + eval): per-step losses, weight hash,
             metrics, output file list, checkpoint key lists. EXACT compare.
  gpu      : GPU QUICK through scripts/run_notebook.py (6 train / 4 eval workers), explicit --only gpu.
             Smoke only: completion, counts, files and keys; numeric deltas are informational.

The notebook is never edited. A driver executes its code cells in order in one namespace; Trainer.fit is
wrapped to record values (and, in data/settings mode, to stop early). The notebook must therefore keep
exposing these globals: RUN_NAME, QUICK, BATCH_SIZE, EPOCHS, otf_data, settings, train_loader, val_loader.
"""
from __future__ import annotations

import argparse
import contextlib
import csv
import glob
import hashlib
import io
import json
import math
import os
import subprocess
import sys
import tempfile
import time
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))  # package import also works with --only models
GOLDEN = ROOT / "tests/golden/refactor-golden.json"
NOTEBOOK = ROOT / "notebooks/05_xunet_v4_nonorm.ipynb"
STANDARD_TEST_SHA = "fdd989f7bdab0ec6f9fd6f4c57bdde981d16b54aa8c37e5da8fd3dfcc8b06837"
SETTING_FIELDS = ["cover_id", "k", "bit_mode", "method", "placement", "payload_kind", "stride",
                  "requested_bpp", "effective_bpp", "clipped"]
ENV_FILES = {
    "newdata-1": ROOT / "scripts/g4/jobs/archive/newdata-1.env",
    "wide-1": ROOT / "scripts/g4/jobs/archive/wide-1.env",
    "speed-1": ROOT / "scripts/g4/jobs/speed-1.env",
    "quick-gpu": ROOT / "tests/golden/quick-gpu.env",
    "cpu-det": ROOT / "tests/golden/cpu-det.env",
}
SCRATCH = ROOT / "runs/.scratch"
PARTS = ("settings", "data", "models", "cpu", "gpu")
# Columns/files that contain wall-clock time or other run-to-run noise.
NOISE = ("time", "seconds", "speed", "wait", "elapsed", "it/s", "mem", "rss")
# Real checkpoints used for the model check (glob, relative to ROOT).
CHECKPOINTS = {
    "wide-w4_r2": "runs/xunet*wide-w4_r2-*/checkpoints/*/best*.ckpt",
    "wide-w4_mixr2": "runs/xunet*wide-w4_mixr2-*/checkpoints/*/best*.ckpt",
    "g4-v4-r2": "runs/g4-v4-r2/checkpoints/*/best*.ckpt",
    "g4-prime-1": "runs/g4-prime-1/checkpoints/*/best*.ckpt",
}


def sha(data) -> str:
    return hashlib.sha256(data if isinstance(data, bytes) else str(data).encode()).hexdigest()


def file_sha(path) -> str:
    return sha(Path(path).read_bytes())


def is_noise(name: str) -> bool:
    lowered = name.lower()
    return lowered.endswith("_s") or any(token in lowered for token in NOISE)


def read_env_file(path: Path) -> dict:
    values = {}
    for raw in Path(path).read_text().splitlines():
        line = raw.strip()
        if line and not line.startswith("#"):
            key, value = line.split("=", 1)
            values[key.strip()] = value.strip()
    return values


# --------------------------------------------------------------------------- driver (runs in a subprocess)
class StopCapture(BaseException):
    pass


def jsonable(value):
    import numpy as np
    if isinstance(value, (bool, int, str)) or value is None:
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else repr(value)
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return jsonable(value.item())
    return str(value)


def shrink(value, limit=100):
    """Replace long lists/dicts (photo selections, pair-id lists) by their length and hash."""
    if isinstance(value, dict):
        if len(value) > limit:
            return {"__len__": len(value), "__sha256__": sha(json.dumps(value, sort_keys=True, default=str))}
        return {k: shrink(v, limit) for k, v in value.items()}
    if isinstance(value, list):
        if len(value) > limit:
            return {"__len__": len(value), "__sha256__": sha(json.dumps(value, sort_keys=True, default=str))}
        return [shrink(v, limit) for v in value]
    return value


def tensors_sha(items) -> str:
    import torch
    h = hashlib.sha256()
    for item in items:
        if torch.is_tensor(item):
            h.update(str(item.dtype).encode() + str(tuple(item.shape)).encode())
            h.update(item.detach().cpu().contiguous().numpy().tobytes())
        else:
            h.update(repr(item).encode())
    return h.hexdigest()


def state_sha(module) -> str:
    return tensors_sha([v for _, v in module.state_dict().items()])


def frame_sha(frame, columns=None) -> str:
    return sha((frame if columns is None else frame[columns]).to_csv(index=False))


def drive(mode: str, env_file: Path | None, sets: list[str], out: Path) -> None:
    """Execute notebook 05 code cells in order, record values, write JSON to `out`."""
    os.chdir(ROOT)
    sys.path.insert(0, str(ROOT))
    if env_file:
        os.environ.update(read_env_file(env_file))
    for item in sets:
        if "=" not in item:  # "KEY" alone unsets the variable
            os.environ.pop(item, None)
            continue
        key, value = item.split("=", 1)
        os.environ[key] = value
    result: dict = {"mode": mode}
    notebook = json.loads(NOTEBOOK.read_text())
    namespace = {"__name__": "__main__"}
    recorder_state: dict = {"losses": [], "val": [], "weights_sha": None, "steps": []}

    import lightning as L
    original_fit = L.Trainer.fit

    class Recorder(L.Callback):
        def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs
            recorder_state["losses"].append(float(loss.detach().cpu()))
            recorder_state["steps"].append(int(trainer.global_step))

        def on_validation_end(self, trainer, pl_module):
            if not trainer.sanity_checking:
                recorder_state["val"].append({k: float(v) for k, v in sorted(trainer.callback_metrics.items())
                                              if not is_noise(k)})

        def on_fit_end(self, trainer, pl_module):
            recorder_state["weights_sha"] = state_sha(pl_module)

    def hash_batches(loader, count):
        iterator = iter(loader)
        hashes = []
        for _ in range(count):
            try:
                hashes.append(tensors_sha(list(next(iterator))))
            except StopIteration:
                break
        return hashes

    def patched_fit(self, model, *args, **kwargs):
        if mode == "run":
            self.callbacks.append(Recorder())
            original_fit(self, model, *args, **kwargs)
            # CPU reference is precisely 20 training steps + one validation pass.
            # Do not execute downstream test/panel/diagnostic notebook cells.
            raise StopCapture
        if mode == "data":
            data = namespace["otf_data"]
            train_loader, val_loader = namespace["train_loader"], namespace["val_loader"]
            batches = int(os.environ.get("GOLDEN_BATCHES", "6"))
            train = {}
            for epoch in (0, 1):
                data.train.set_epoch(epoch)
                train[f"epoch{epoch}"] = hash_batches(train_loader, batches)
            result["train_batch_sha"] = train
            result["val_first_batches_sha"] = hash_batches(val_loader, 2)
            result["test_first_batches_sha"] = hash_batches(namespace["test_loader"], 2)
            result["val_len"] = len(data.val)
            result["test_len"] = len(data.test)
            result["val_frame_sha"] = frame_sha(data.val.frame)
            result["test_frame_sha"] = frame_sha(data.test.frame)
            result["fixed_train_frame_sha"] = frame_sha(data.fixed_train.frame)
            result["train_len"] = len(data.train)
            for loader in (train_loader,):
                with contextlib.suppress(Exception):
                    loader.shutdown()
            raise StopCapture
        raise AssertionError(mode)

    L.Trainer.fit = patched_fit
    stopped = False
    try:
        for index, cell in enumerate(notebook["cells"]):
            if cell["cell_type"] != "code":
                continue
            exec(compile("".join(cell["source"]), f"<cell {index}>", "exec"), namespace)
            if mode == "settings" and {"RUN_NAME", "BATCH_SIZE", "EPOCHS", "otf_data", "settings"} <= set(namespace):
                stopped = True
                break
    except StopCapture:
        stopped = True
    finally:
        L.Trainer.fit = original_fit
    if mode == "settings":
        data = namespace["otf_data"]
        # The thin notebook exposes only runner globals. Compare the same complete
        # historical effective-settings view through the typed adapter, not the archive.
        from cvdl.settings import Settings
        if isinstance(namespace['settings'], Settings):
            result['globals'] = namespace['settings'].legacy_globals()
        else:
            result["globals"] = {name: jsonable(value) for name, value in sorted(namespace.items())
                                 if name.isupper() and not name.startswith("_") and not callable(value)
                                 and name not in {"ROOT", "F", "L"}}
        result["run_name"] = namespace["RUN_NAME"]
        result["run_settings"] = jsonable(settings_dict(namespace["settings"]))
        result["data_record"] = shrink(jsonable({k: v for k, v in data.record.items() if not is_noise(k)}))
        result["val_frame_sha"] = frame_sha(data.val.frame)
        result["test_frame_sha"] = frame_sha(data.test.frame)
        result["fixed_train_frame_sha"] = frame_sha(data.fixed_train.frame)
        result["test_settings_sha"] = frame_sha(data.test.frame, SETTING_FIELDS)
        result["val_settings_sha"] = frame_sha(data.val.frame, SETTING_FIELDS)
        corpus = data.corpus
        for epoch in (0, 1):
            if corpus.settings.source_mix is not None:
                schedule, repeats = corpus.train_tile_schedule(epoch)
            else:
                schedule, repeats = corpus.train_schedule(epoch), None
            result[f"train_schedule_epoch{epoch}"] = dict(groups=len(schedule), repeats=jsonable(repeats),
                                                          sha=sha(repr(list(schedule))))
        result["train_len"] = len(data.train)
        result["validation_pair_csv_sha"] = sha(getattr(data, "validation_pair_csv", "")) if hasattr(
            data, "validation_pair_csv") else None
    else:
        result["stopped_early"] = stopped
    if mode == "run":
        result.update(losses=recorder_state["losses"], steps=recorder_state["steps"],
                      val_metrics=recorder_state["val"], weights_sha=recorder_state["weights_sha"],
                      run_name=namespace.get("RUN_NAME"),
                      checkpoint_dir=str(namespace["checkpoint_dir"]))
    out.write_text(json.dumps(result, indent=1, sort_keys=True))


def settings_dict(settings):
    from dataclasses import asdict
    if hasattr(settings, 'to_run_settings'):
        settings = settings.to_run_settings()
    return asdict(settings)


# --------------------------------------------------------------------------- helpers in the parent process
def run_driver(mode, env_name, sets=(), extra_env=None, timeout=3600):
    SCRATCH.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile("w", suffix=".json", dir=SCRATCH, delete=False)
    handle.close()
    out = Path(handle.name)
    env = dict(os.environ)
    env.update(extra_env or {})
    command = [sys.executable, str(Path(__file__).resolve()), "_drive", "--mode", mode, "--out", str(out)]
    if env_name:
        command += ["--env-file", str(ENV_FILES[env_name])]
    for item in sets:
        command += ["--set", item]
    start = time.perf_counter()
    done = subprocess.run(command, cwd=ROOT, env=env, capture_output=True, text=True, timeout=timeout)
    if done.returncode != 0:
        out.unlink(missing_ok=True)
        raise RuntimeError(f"driver {mode}/{env_name} failed ({done.returncode}):\n{done.stdout[-3000:]}\n{done.stderr[-3000:]}")
    result = json.loads(out.read_text())
    out.unlink(missing_ok=True)
    result["_seconds"] = round(time.perf_counter() - start, 1)
    return result


def standard_test_hash():
    """Same formula as scripts/eval_checkpoint.py for the standard RAISE/DIV2K test."""
    sys.path[:0] = [str(ROOT / "LSB-Dataset-Generator")]
    from lsb_torch import Corpus, FixedPairs, RunSettings
    settings = RunSettings(evaluation_yaml=str(ROOT / "configs/raise512-run1.yaml"), sources=("raise", "div2k"))
    dataset = FixedPairs(Corpus(ROOT, settings), "test", sources=("raise", "div2k"))
    return dict(pairs=len(dataset), sha256=frame_sha(dataset.frame, SETTING_FIELDS))


def part_settings():
    out = {"standard_test": standard_test_hash()}
    out["standard_test"]["matches_documented"] = out["standard_test"]["sha256"] == STANDARD_TEST_SHA
    assert out["standard_test"]["matches_documented"], "standard test identity changed"
    # The speed matrix job never runs notebook cells (run_notebook.py starts scripts/speed_matrix.py),
    # so only its parsed env settings are recorded.
    sys.path.insert(0, str(ROOT))
    from scripts.run_notebook import settings_file
    speed_env = settings_file(ENV_FILES["speed-1"], [])
    from scripts import speed_matrix
    previous_env = dict(os.environ)
    try:
        os.environ.update(speed_env)
        speed_effective = speed_matrix.settings()
    finally:
        os.environ.clear()
        os.environ.update(previous_env)
    out["speed-1"] = dict(env_settings=speed_env, run_name=speed_env["V3_RUN_NAME"],
                          effective_pairs_steps_warmup_workers=jsonable(speed_effective),
                          note="speed matrix: no notebook cells run; settings() parsed without training")
    for name in ("newdata-1", "wide-1", "quick-gpu", "cpu-det"):
        result = run_driver("settings", name, extra_env=cpu_env() if name == "cpu-det" else None)
        result.pop("_seconds")
        result["env_settings"] = read_env_file(ENV_FILES[name])
        if name == "newdata-1":
            assert result["validation_pair_csv_sha"] == "6bb3c8efadad06af50ba014a95c98b4fc57008119bdd55b794ec2a1729e4d5ca"
        out[name] = result
    return out


def part_data():
    out = {}
    for workers in (0, 4):
        sets = [f"V3_WORKERS={workers}", f"V3_EVAL_WORKERS={workers}", "V3_BATCH_SIZE=8", "V3_PRIME_NEXT_EPOCH=off",
                "V3_PERSISTENT_WORKERS=off", "V3_TRAIN_PHOTO_FRACTION=0.01", "V3_EXTRA_TESTS=none",
                "V3_RUN_NAME", "V3_COMPILE=off"]
        with tempfile.TemporaryDirectory(prefix="golden-data-", dir=SCRATCH) as folder:
            result = run_driver("data", "newdata-1", sets=[*sets, f"V3_OUTPUT_ROOT={folder}"])
        result.pop("_seconds")
        result["note"] = "newdata-1 env, batch 8, 1% train photos, decode path; workers " + str(workers)
        out[f"workers{workers}"] = result
    return out


def cpu_env():
    return {"CUDA_VISIBLE_DEVICES": "", "HIP_VISIBLE_DEVICES": "", "ROCR_VISIBLE_DEVICES": "",
            "OMP_NUM_THREADS": "8", "MKL_NUM_THREADS": "8"}


def ckpt_structure(path: Path) -> dict:
    import torch
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    optimizers = ckpt.get("optimizer_states", [])
    return dict(
        top_level_keys=sorted(ckpt),
        state_dict_keys=[[k, list(v.shape), str(v.dtype)] for k, v in ckpt["state_dict"].items()],
        hyper_parameters={k: jsonable(v) for k, v in sorted(ckpt.get("hyper_parameters", {}).items())},
        callback_state_keys=sorted(ckpt.get("callbacks", {})),
        loops_keys=sorted(ckpt.get("loops", {})),
        optimizer_state_keys=[sorted(o) for o in optimizers],
        optimizer_param_group_keys=[sorted(g) for o in optimizers for g in o["param_groups"]],
        lr_scheduler_count=len(ckpt.get("lr_schedulers", [])),
        epoch=ckpt.get("epoch"), global_step=ckpt.get("global_step"),
        pytorch_lightning_version_present="pytorch-lightning_version" in ckpt,
    )


def read_csv_rows(path: Path):
    import pandas as pd
    frame = pd.read_csv(path)
    keep = [c for c in frame.columns if not is_noise(c)]
    rows = frame[keep].astype(object).where(frame[keep].notna(), None).values.tolist()
    return dict(columns=list(frame.columns), kept_columns=keep, rows=jsonable(rows))


def collect_run_folder(folder: Path) -> dict:
    """Summarise one finished run folder (names, sizes of tiny files, metrics, checkpoint structure)."""
    files = sorted(str(p.relative_to(folder)) for p in folder.rglob("*") if p.is_file())
    # Event files and checkpoints carry timestamps in names or bytes; list them by kind.
    listed = [f for f in files if "events.out.tfevents" not in f]
    info = dict(files=listed, tensorboard_event_files=sum("events.out.tfevents" in f for f in files))
    metrics = folder / "logs/csv/run/metrics.csv"
    if metrics.exists():
        info["metrics_csv"] = read_csv_rows(metrics)
    for name in ("lr_trace.csv", "validation_pairs.csv", "test_predictions.csv", "test_predictions_per_setting.csv"):
        path = folder / name
        if path.exists():
            info[name] = dict(sha256=file_sha(path))
            if name == "lr_trace.csv":
                info[name]["rows"] = read_csv_rows(path)["rows"]
            elif name == "test_predictions.csv":
                import pandas as pd
                prediction_frame = pd.read_csv(path)
                info[name]["prob_stego"] = prediction_frame["prob_stego"].tolist()
                info[name]["identities_sha256"] = frame_sha(prediction_frame.drop(columns=["prob_stego"]))
    timing = folder / "epoch_timing.csv"
    if timing.exists():
        info["epoch_timing_columns"] = read_csv_rows(timing)["columns"]
    for name in ("model_run_settings.json", "data_run_settings.json"):
        path = folder / name
        if path.exists():
            info[name] = shrink({k: v for k, v in json.loads(path.read_text()).items() if not is_noise(k)})
    last = folder / "last.ckpt"
    if last.exists():
        info["last_ckpt"] = ckpt_structure(last)
    bests = sorted(folder.glob("best-*.ckpt"))
    info["best_ckpt_names"] = [p.name for p in bests]
    if bests:
        info["best_ckpt_structure_equal_last"] = ckpt_structure(bests[0]) == info.get("last_ckpt")
    return info


def part_cpu():
    out_root = SCRATCH / f"golden-cpu-{os.getpid()}"
    out_root.mkdir(parents=True, exist_ok=True)
    try:
        sets = [f"V3_OUTPUT_ROOT={out_root}"]
        result = run_driver("run", "cpu-det", sets=sets, extra_env=cpu_env())
        seconds = result.pop("_seconds")
        folder = Path(result.pop("checkpoint_dir"))
        result["folder"] = collect_run_folder(folder)
        assert len(result["losses"]) == 20 and result["steps"][-1] == 20, "CPU reference must have 20 steps"
        assert len(result["val_metrics"]) == 1, "CPU reference must have one end-of-epoch validation"
        result["note"] = "deterministic CPU run, fixed 8 threads, 20 train steps + one validation pass"
        return result
    finally:
        shutil.rmtree(out_root, ignore_errors=True)


def part_gpu():
    import torch
    if not torch.cuda.is_available():
        return {"skipped": "no GPU"}
    out = SCRATCH / f"golden-gpu-{os.getpid()}"
    command = [sys.executable, "scripts/run_notebook.py", "--notebook", str(NOTEBOOK.relative_to(ROOT)),
               "--env-file", str(ENV_FILES["quick-gpu"].relative_to(ROOT)), "--output", str(out.relative_to(ROOT)),
               "--host-monitor", "off", "--set", "lr=0.0003"]
    start = time.perf_counter()
    try:
        done = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, timeout=3 * 3600)
        seconds = round(time.perf_counter() - start, 1)
        if done.returncode != 0:
            raise RuntimeError(f"run_notebook failed ({done.returncode}):\n{done.stdout[-3000:]}\n{done.stderr[-3000:]}")
        status = json.loads((out / "status.json").read_text()) if (out / "status.json").exists() else {}
        result = dict(note="GPU QUICK via run_notebook.py (6/4 workers)",
                      run_notebook_state=status.get("state"),
                      cells_run=len(status.get("cells", [])),
                      cell_states=[c.get("state") or c.get("status") for c in status.get("cells", [])])
        run_name = status.get("runtime_settings", {}).get("RUN_NAME") or status.get("effective_settings", {}).get("V3_RUN_NAME")
        top = sorted(str(p.relative_to(out)) for p in out.iterdir())
        result["output_top_level"] = [t for t in top if "events.out" not in t]
        folders = [p for p in out.rglob("last.ckpt")]
        if not folders:
            raise RuntimeError("run_notebook output has no last.ckpt; top level: " + ", ".join(top))
        result["run_folder_relative"] = str(folders[0].parent.relative_to(out))
        result["runtime_settings"] = shrink(jsonable({k: v for k, v in status.get("runtime_settings", {}).items()
                                               if not is_noise(k)}))
        folder = folders[0].parent
        saved_settings = json.loads((folder / 'run_settings.json').read_text())
        assert saved_settings['values']['lr'] == 0.0003
        assert saved_settings['sources']['lr'] == 'override'
        assert {'name':'lr', 'value':0.0003} in saved_settings['overrides']
        print('[override proof] run_notebook.py --set lr=0.0003: run_settings.json lr=0.0003 source=override', flush=True)
        # A score-byte hash can vary on GPU, but a malformed/missing setting table cannot.
        from cvdl.evaluation import load_breakdown, breakdown_tables
        from cvdl.metrics import global_summary
        frame, otf = load_breakdown(folder / 'test_predictions.csv')
        table, _, _ = breakdown_tables(frame, otf_metadata=otf, global_threshold=global_summary(frame)['threshold'])
        assert (folder / 'test_predictions_per_setting.csv').read_text() == table.to_csv(index=False)
        result["folder"] = collect_run_folder(folder)
        result["extra_panels"] = {}
        for summary in sorted(out.rglob("summary.csv")):
            entry = {"summary": read_csv_rows(summary)}
            record = summary.parent / "run.json"
            if record.exists():
                r = json.loads(record.read_text())
                entry["pairs"] = r["pairs"]
                entry["settings_sha256"] = r["settings_sha256"]
                assert r['quick_pair_cap'] == 16, 'GPU QUICK extra panels must record the phase-4a cap'
            result["extra_panels"][str(summary.parent.relative_to(out))] = entry
        return result
    finally:
        shutil.rmtree(out, ignore_errors=True)


# ---- models part (in-process)
def load_tiles():
    """64 real 512 px tiles: 8 clean + 8 stego from each of RAISE, DIV2K, ALASKA, LSDIR. Fully deterministic."""
    import numpy as np
    from PIL import Image
    sys.path[:0] = [str(ROOT / "LSB-Dataset-Generator")]
    from lsb_raw import decode_raw_pixels
    orig = ROOT / "Datasets/originals"
    photos = {
        "raise": sorted(glob.glob(str(orig / "raise/NEF/*.NEF")))[0],
        "div2k": str(orig / "div2k/DIV2K_train_HR/0001.png"),
        "alaska": sorted(glob.glob(str(orig / "alaska/tiff_various_color/*.tif")))[3],
        "lsdir": sorted(glob.glob(str(orig / "lsdir/hr/*/*.png")))[5],
    }
    rng = np.random.default_rng(7)
    tiles, table = [], []
    for name, path in photos.items():
        photo = (decode_raw_pixels(Path(path)) if path.endswith(".NEF")
                 else np.array(Image.open(path).convert("RGB")))
        for _ in range(8):
            y = int(rng.integers(0, photo.shape[0] - 512 + 1))
            x = int(rng.integers(0, photo.shape[1] - 512 + 1))
            clean = photo[y:y + 512, x:x + 512].copy()
            k = int(rng.integers(1, 5))
            mask = rng.random(clean.shape) < 0.5
            bits = rng.integers(0, 2 ** k, clean.shape).astype(np.uint8)
            stego = np.where(mask, (clean & np.uint8(256 - 2 ** k)) | bits, clean).astype(np.uint8)
            tiles += [clean, stego]
            table += [dict(source=name, photo=str(Path(path).relative_to(ROOT)), y=y, x=x, kind="clean"),
                      dict(source=name, photo=str(Path(path).relative_to(ROOT)), y=y, x=x, kind="stego", k=k)]
    return np.stack(tiles), table


def build_model(hparams):
    try:  # after the refactor the package owns the classes
        from cvdl.models import model_from_hparams
        origin = "cvdl.models"
    except ImportError:
        sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "LSB-Dataset-Generator")]
        from map_demo import model_from_hparams
        origin = "scripts/map_demo.py"
    return model_from_hparams(hparams), origin


def part_models():
    import numpy as np
    import torch
    images, table = load_tiles()
    x = torch.from_numpy(images).permute(0, 3, 1, 2).float()
    out = {"tile_count": len(table), "tiles": table, "tiles_sha": sha(json.dumps(table, sort_keys=True)),
           "pixel_sha": sha(images.tobytes()), "checkpoints": {}}
    devices = ["cpu"]  # GPU equivalence is checked in-process by test_model_copy_parity
    for name, pattern in CHECKPOINTS.items():
        paths = sorted(glob.glob(str(ROOT / pattern)))
        if len(paths) != 1:
            raise RuntimeError(f"{name}: expected one checkpoint for {pattern}, found {len(paths)}")
        ckpt = torch.load(paths[0], map_location="cpu", weights_only=False)
        hp = ckpt["hyper_parameters"]
        state = {k.removeprefix("net."): v for k, v in ckpt["state_dict"].items()}
        model, origin = build_model(hp)
        # Phase 0: also execute the trained notebook class, not just map_demo's copy.
        # Later phases keep the package-vs-before hashes as the independent oracle.
        notebook_model = None
        if origin == "scripts/map_demo.py":
            sys.path.insert(0, str(ROOT / "tests"))
            from test_model_copy_parity import notebook_model_namespace, notebook_outputs
            notebook_path = ("notebooks/04_baseline_kv_xunet_v3.ipynb" if name == "g4-prime-1"
                             else "notebooks/archive/05_baseline_kv_xunet_v4_pre-cvdl.ipynb")
            ns = notebook_model_namespace(notebook_path)
            kw = dict(clip_t=hp.get("clip_t", 3.0), norm=hp.get("norm", "group"))
            notebook_model = (ns["XuNetWide"](width=int(hp.get("width", 4)), **kw)
                              if hp.get("model") == "xunet_wide" else ns["XuNet"](**kw))
            notebook_model.load_state_dict(state, strict=True)
            notebook_model.eval()
        result = model.load_state_dict(state, strict=True)
        model.eval()
        entry = dict(path=str(Path(paths[0]).relative_to(ROOT)), file_sha256=file_sha(paths[0]),
                     model_class=type(model).__name__, class_source=origin,
                     hparams={k: jsonable(hp.get(k)) for k in ("model", "norm", "clip_t", "width")},
                     missing_keys=list(result.missing_keys), unexpected_keys=list(result.unexpected_keys),
                     state_keys_sha=sha(json.dumps(list(state))), training_flag_after_eval=model.training)
        reference = None
        for device in devices:
            model.to(device)
            if notebook_model is not None:
                notebook_model.to(device)
            logits, cells = [], []
            with torch.inference_mode():
                for i in range(0, len(x), 16):
                    batch = x[i:i + 16].to(device)
                    current_logits = model(batch)
                    c = model.cell_logits(batch)
                    current_cells = c[:, 1] - c[:, 0]
                    if notebook_model is not None:
                        nb_logits, nb_cells = notebook_outputs(notebook_model, batch)
                        if not (torch.equal(nb_logits, current_logits) and torch.equal(nb_cells, current_cells)):
                            raise AssertionError(f"notebook/model copy mismatch: {name}/{device}")
                    logits.append(current_logits.cpu())
                    cells.append(current_cells.cpu())
            logits, cells = torch.cat(logits), torch.cat(cells)
            summary = dict(logits_sha=sha(logits.numpy().tobytes()), cell_map_sha=sha(cells.numpy().tobytes()),
                           logit_abs_max=float(logits.abs().max()), cell_shape=list(cells.shape),
                           argmax_sha=sha(logits.argmax(1).numpy().tobytes()))
            if device == "cpu":
                reference = (logits, cells)
            else:
                summary["vs_cpu_logits_max_abs"] = float((logits - reference[0]).abs().max())
                summary["vs_cpu_cell_map_max_abs"] = float((cells - reference[1]).abs().max())
            entry[device] = summary
        out["checkpoints"][name] = entry
    return out


CAPTURE = {"settings": part_settings, "data": part_data, "models": part_models, "cpu": part_cpu, "gpu": part_gpu}


def versions():
    import torch
    import lightning
    import numpy
    import pandas
    return dict(torch=str(torch.__version__), lightning=lightning.__version__, numpy=numpy.__version__,
                pandas=pandas.__version__, python=sys.version.split()[0])


def capture_all(parts):
    result = {}
    SCRATCH.mkdir(parents=True, exist_ok=True)
    for part in parts:
        start = time.perf_counter()
        print(f"[capture] {part} ...", flush=True)
        result[part] = CAPTURE[part]()
        print(f"[capture] {part} done in {time.perf_counter() - start:.0f}s", flush=True)
    return result


# --------------------------------------------------------------------------- compare
def close(a, b, rel):
    if isinstance(a, float) or isinstance(b, float):
        try:
            return math.isclose(float(a), float(b), rel_tol=rel, abs_tol=rel * 1e-3)
        except (TypeError, ValueError):
            return a == b
    return a == b


def diff(old, new, path="", rel=0.0, found=None, floats_tolerant=False, limit=40):
    """Collect differences. Floats use `rel` only when floats_tolerant."""
    found = [] if found is None else found
    if len(found) >= limit:
        return found
    if isinstance(old, dict) and isinstance(new, dict):
        for key in sorted(set(old) | set(new)):
            if key not in old or key not in new:
                found.append(f"{path}/{key}: {'missing in golden' if key not in old else 'missing now'}")
            else:
                diff(old[key], new[key], f"{path}/{key}", rel, found, floats_tolerant, limit)
    elif isinstance(old, list) and isinstance(new, list):
        if len(old) != len(new):
            found.append(f"{path}: length {len(old)} != {len(new)}")
        else:
            for i, (a, b) in enumerate(zip(old, new)):
                diff(a, b, f"{path}[{i}]", rel, found, floats_tolerant, limit)
    elif floats_tolerant and isinstance(old, float) and isinstance(new, float):
        if not close(old, new, rel):
            found.append(f"{path}: {old!r} vs {new!r} (rel {abs(old - new) / max(abs(old), 1e-12):.2e})")
    elif old != new:
        found.append(f"{path}: {str(old)[:80]!r} != {str(new)[:80]!r}")
    return found


# Named, narrowly scoped transition allow-list. Every use is printed. The golden
# JSON stays unchanged; anything not handled here still reaches the exact diff.
EXPECTED_DIFFERENCES = {
    'run-settings-file': 'one new run_settings.json in the run folder',
    'thin-cell-layout': '16 old code cells become 10 completed thin-notebook cells',
    'quick-extra-panel-cap': 'bpp-1 extra panel: 10956 full pairs become the fixed 16-pair subset',
    'gpu-setting-table-scores': 'GPU setting-table CSV score bytes may vary; saved table is independently verified',
}


def allowed_difference(name, detail):
    assert name in EXPECTED_DIFFERENCES
    print(f'[allowed {name}] {EXPECTED_DIFFERENCES[name]}: {detail}', flush=True)


def allow_run_settings_file(old, new):
    a, b = old.get('folder', {}).get('files', []), new.get('folder', {}).get('files', [])
    if 'run_settings.json' not in a and b.count('run_settings.json') == 1:
        allowed_difference('run-settings-file', 'added run_settings.json')
        b.remove('run_settings.json')  # Every other filename is still compared exactly.


def allow_quick_panel_cap(old, new):
    """Validate the named intended cap against independently rebuilt pair metadata."""
    if not any(name.endswith('/extra-test-bpp1') and before.get('pairs') != new.get('extra_panels', {}).get(name, {}).get('pairs')
               for name, before in old.get('extra_panels', {}).items()):
        return set()
    allowed = set()
    from lsb_torch import Corpus, FixedPairs, RunSettings
    from lsb_notebook import fixed_subset
    import pandas as pd
    full = FixedPairs(Corpus(ROOT, RunSettings(evaluation_yaml=str(ROOT/'configs/test-bpp1.yaml'),
                     sources=('raise','div2k'))), 'test', sources=('raise','div2k'))
    selected = fixed_subset(full, 16, seed=1)
    pairs = selected.frame
    setting_sha = frame_sha(pairs, SETTING_FIELDS)
    # Metrics are GPU information, but category order, values and all counts are exact.
    frame = pd.concat([pairs.assign(label=0,prob_stego=0.), pairs.assign(label=1,prob_stego=0.)])
    frame['bpp_bin'] = pd.cut(frame['requested_bpp'], bins=[0.,.05,.15,.3,.5,.75,1.],
        labels=['<=0.05','0.05-0.15','0.15-0.30','0.30-0.50','0.50-0.75','0.75-1.0'], include_lowest=True, right=True)
    from cvdl.metrics import summary_rows
    expected = summary_rows(frame)
    columns = ['category','value','images','pairs','P_E','AUC','accuracy','false_alarm_rate','miss_rate']
    for name, before in old.get('extra_panels', {}).items():
        after = new.get('extra_panels', {}).get(name)
        if not name.endswith('/extra-test-bpp1') or after is None or before.get('pairs') == after.get('pairs'):
            continue
        table = after.get('summary', {})
        valid = (set(after) == {'summary','pairs','settings_sha256'} and before['pairs'] == len(full)
                 and after['pairs'] == 16 and after['settings_sha256'] == setting_sha
                 and table.get('columns') == columns and table.get('kept_columns') == columns
                 and len(table.get('rows', [])) == len(expected))
        if valid:
            for actual, wanted in zip(table['rows'], expected):
                if actual[:4] != [wanted[key] for key in columns[:4]]:
                    valid = False; break
                if any(not isinstance(v,(int,float)) or not math.isfinite(v) or not 0 <= v <= 1 for v in actual[4:]):
                    valid = False; break
        if valid:
            allowed_difference('quick-extra-panel-cap', f'{name}: {before["pairs"]} -> 16; settings SHA {setting_sha}; all group counts/schema checked')
            old['extra_panels'][name] = after
            allowed.add(name)
    return allowed


def repo_relative_paths(value, root):
    """Compare absolute repo paths by suffix; keep all other values exact.

    A tagged tuple (not a string placeholder) cannot match a non-path literal.
    Only whole absolute strings below root qualify; prefix lookalikes do not.
    """
    prefix=str(root).rstrip('/')
    if not prefix.startswith('/'):
        raise ValueError('repository root must be an absolute non-root path')
    if isinstance(value, dict):
        return {key:repo_relative_paths(item,root) for key,item in value.items()}
    if isinstance(value, list):
        return [repo_relative_paths(item,root) for item in value]
    if isinstance(value, str) and (value==prefix or value.startswith(prefix+'/')):
        return ('<ROOT>',value[len(prefix):])
    return value


def compare_part(part, old, new):
    from copy import deepcopy
    # The saved oracle was captured at /work. Relocation is not a data difference.
    old, new = repo_relative_paths(deepcopy(old), '/work'), repo_relative_paths(deepcopy(new), ROOT)
    if part == "models":
        # CPU hashes are the cross-run oracle. GPU kernels are not bit-exact
        # across processes; test_model_copy_parity checks GPU copies in-process.
        for item in (old, new):
            for entry in item.get("checkpoints", {}).values():
                entry.pop("class_source", None)
                entry.pop("cuda", None)
        return diff(old, new, "/models")
    if part in {'cpu','gpu'}:
        allow_run_settings_file(old, new)
    if part != "gpu":
        return diff(old, new, f"/{part}")

    if (old.get('cells_run') == 16 and old.get('cell_states') == ['done']*16
            and new.get('cells_run') == 10 and new.get('cell_states') == ['done']*10):
        allowed_difference('thin-cell-layout', '16 -> 10; every code cell completed')
        old['cells_run'], old['cell_states'] = new['cells_run'], new['cell_states']
    capped_panels = allow_quick_panel_cap(old, new)
    a = old.get('folder', {}).get('test_predictions_per_setting.csv', {})
    b = new.get('folder', {}).get('test_predictions_per_setting.csv', {})
    if set(a) == set(b) == {'sha256'} and a != b:
        allowed_difference('gpu-setting-table-scores', f'{a["sha256"]} -> {b["sha256"]}')
        a['sha256'] = b['sha256']

    # GPU QUICK is a smoke check: exact completion, file list, counts and keys;
    # numerical results are information only, not a cross-run equality proof.
    for section in ("metrics_csv",):
        a = old.get("folder", {}).get(section, {})
        b = new.get("folder", {}).get(section, {})
        if a.get("kept_columns") is not None and a.get("kept_columns") == b.get("kept_columns") and len(a.get("rows", [])) == len(b.get("rows", [])):
            for i, (ar, br) in enumerate(zip(a["rows"], b["rows"])):
                for j, column in enumerate(a["kept_columns"]):
                    if column not in {"epoch", "step"}:
                        delta = None if ar[j] is None or br[j] is None else float(br[j]) - float(ar[j])
                        print(f"[GPU info] {column}[{i}] delta={delta}")
                        br[j] = ar[j]
    a = old.get("folder", {}).get("test_predictions.csv", {})
    b = new.get("folder", {}).get("test_predictions.csv", {})
    # The phase-0 file predates explicit probability storage. Score hashes are
    # informational in either format; later captures also print numeric deltas.
    a.pop("sha256", None)
    b.pop("sha256", None)
    if "prob_stego" in a and "prob_stego" in b and len(a["prob_stego"]) == len(b["prob_stego"]):
        maximum = max((abs(float(av) - float(bv)) for av, bv in zip(a["prob_stego"], b["prob_stego"])), default=0.)
        print(f"[GPU info] test probability max abs delta={maximum:.8g}")
        b["prob_stego"] = a["prob_stego"]
    for name, panel in old.get("extra_panels", {}).items():
        if name in capped_panels:
            continue  # Different subsets: do not print artificial zero numeric deltas.
        other = new.get("extra_panels", {}).get(name, {})
        a, b = panel["summary"], other.get("summary", {})
        if a.get("kept_columns") != b.get("kept_columns") or len(a.get("rows", [])) != len(b.get("rows", [])):
            continue
        for ar, br in zip(a["rows"], b["rows"]):
            for j, metric in enumerate(a["kept_columns"]):
                if metric in {"P_E", "AUC", "accuracy", "false_alarm_rate", "miss_rate"}:
                    delta = None if ar[j] is None or br[j] is None else float(br[j]) - float(ar[j])
                    print(f"[GPU panel info] {ar[0]}={ar[1]} {metric} delta={delta}")
                    br[j] = ar[j]
    return diff(old, new, "/gpu")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("capture", "compare"):
        p = sub.add_parser(name)
        p.add_argument("--only", default="settings,data,models,cpu")
        p.add_argument("--golden", type=Path, default=GOLDEN)
    sub.choices["compare"].add_argument("--candidate", type=Path,
                                        help="compare a separately captured JSON instead of running again")
    sub.choices["capture"].add_argument("--out", type=Path, default=None)
    d = sub.add_parser("_drive")
    d.add_argument("--mode", required=True)
    d.add_argument("--env-file", type=Path)
    d.add_argument("--set", action="append", default=[])
    d.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "_drive":
        drive(args.mode, args.env_file, args.set, args.out)
        return 0
    parts = [p.strip() for p in args.only.split(",") if p.strip()]
    unknown = set(parts) - set(PARTS)
    if unknown:
        parser.error(f"unknown parts: {sorted(unknown)}")
    if args.command == "capture":
        target = args.out or args.golden
        existing = json.loads(target.read_text()) if target.exists() else {}
        captured = capture_all(parts)
        existing.update(captured)
        existing["_meta"] = dict(notebook=str(NOTEBOOK.relative_to(ROOT)), versions=versions(),
                                 parts_present=sorted(k for k in existing if not k.startswith("_")),
                                 standard_test_sha_documented=STANDARD_TEST_SHA,
                                 source_revision=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                                 notebook_sha256=file_sha(NOTEBOOK),
                                 env_file_sha256={k: file_sha(v) for k, v in ENV_FILES.items()})
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(existing, indent=1, sort_keys=True) + "\n")
        print(f"wrote {target} ({target.stat().st_size / 1024:.0f} KiB)")
        return 0
    golden = json.loads(args.golden.read_text())
    new = json.loads(args.candidate.read_text()) if args.candidate else capture_all(parts)
    failures = 0
    for part in parts:
        if part not in golden:
            print(f"[compare] {part}: NOT IN GOLDEN FILE")
            failures += 1
            continue
        found = compare_part(part, golden[part], new[part])
        if found:
            failures += 1
            print(f"[compare] {part}: DIFFERENT ({len(found)}{'+' if len(found) >= 40 else ''} differences)")
            for line in found:
                print("    " + line)
        else:
            tolerance = " (smoke: completion/files/counts/keys; numerical deltas informational)" if part == "gpu" else " (exact)"
            print(f"[compare] {part}: EQUAL{tolerance}")
    print("RESULT:", "ALL EQUAL" if not failures else f"{failures} part(s) differ")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
