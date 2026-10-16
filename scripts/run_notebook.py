"""Run a notebook headlessly and save a restart-safe executed copy after each cell.

Executes through IPython. The tracked notebook is read only.
Run from a fresh git-ignored output folder; notebook 05 uses an automatic machine profile.
"""
from __future__ import annotations

import argparse
import base64
from contextlib import nullcontext, redirect_stderr, redirect_stdout
from copy import deepcopy
from datetime import datetime, timezone
import json
import multiprocessing
import os
from pathlib import Path
import re
import resource
import signal
import subprocess
import sys
import threading
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
KEY = re.compile(r"^(?:QUICK|V3_[A-Z0-9_]+|CVDL_JOB_ID)$")


def now():
    return datetime.now(timezone.utc).isoformat()


def settings_file(path, overrides, *, allow_native=False):
    settings = {}
    for number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            raise ValueError(f"{path}:{number}: expected KEY=VALUE")
        key, value = (part.strip() for part in line.split("=", 1))
        if not KEY.fullmatch(key) or key in settings:
            raise ValueError(f"{path}:{number}: invalid or repeated key {key!r}")
        settings[key] = value
    sys.path.insert(0, str(ROOT))
    from cvdl.settings import legacy_override
    for item in overrides:
        key, value = legacy_override(item, allow_native=allow_native)
        if key is not None:
            settings[key] = value
    if "QUICK" in settings and settings["QUICK"].lower() not in {"true", "false", "1", "0", "yes", "no"}:
        raise ValueError("QUICK must be true or false")
    if "V3_OUTPUT_ROOT" in settings:
        raise ValueError("V3_OUTPUT_ROOT is set by --output, not the env file")
    if settings.get("V3_RUN_NAME") == "":
        settings.pop("V3_RUN_NAME")  # --set run_name= requests the derived name.
    name = settings.get("V3_RUN_NAME")
    if name and (name in {".", ".."} or Path(name).name != name or "\\" in name):
        raise ValueError("V3_RUN_NAME must be one folder name")
    return settings


def normalize_stream_outputs(notebook):
    """A library may still write to a saved stream after its cell returns."""
    for cell in notebook['cells']:
        for item in cell.get('outputs', ()):
            if item.get('output_type') == 'stream' and isinstance(item.get('text'), list):
                item['text'] = ''.join(item['text'])


def atomic_json(path, data):
    temporary = path.with_name(path.name + ".part")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(data, stream, indent=2, ensure_ascii=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def mime_data(data):
    result = {}
    for kind, value in data.items():
        if kind in {"image/png", "image/jpeg", "application/pdf"} and isinstance(value, bytes):
            result[kind] = base64.b64encode(value).decode("ascii")
        elif isinstance(value, bytes):
            result[kind] = value.decode("utf-8", errors="replace")
        else:
            result[kind] = value
    return result


class CellStream:
    def __init__(self, name, log, terminal, outputs, lock, compact=False):
        self.name, self.log, self.terminal = name, log, terminal
        self.outputs, self.lock, self.compact = outputs, lock, compact

    def write(self, text):
        if not text:
            return 0
        with self.lock:
            self.log.write(text)
            self.log.flush()
            if self.compact:
                # The job watcher gets only short step/rate reports, not the
                # complete notebook stream or thousands of progress redraws.
                if text.startswith(("TRAIN progress ", "TRAIN epoch-end ")):
                    self.terminal.write(text.rstrip("\n") + "\n")
                    self.terminal.flush()
                return len(text)  # Notebook 05 keeps full cell text only in run.log.
            self.terminal.write(text)
            self.terminal.flush()
            if self.outputs and self.outputs[-1].get("output_type") == "stream" and self.outputs[-1]["name"] == self.name:
                chunks = self.outputs[-1]["text"]
                if isinstance(chunks, list):
                    chunks.append(text)
                else:
                    self.outputs[-1]["text"] = [chunks, text]
            else:
                self.outputs.append({"output_type": "stream", "name": self.name, "text": [text]})
        return len(text)

    def flush(self):
        with self.lock:
            self.log.flush()
            if not self.compact:
                self.terminal.flush()

    def isatty(self):
        return False


class StopRequested(BaseException):
    def __init__(self, signum):
        self.signum = signum
        super().__init__(f"received {signal.Signals(signum).name}")


def cleanup_workers(namespace, log):
    for name in ("train_loader", "val_loader", "test_loader", "train_eval_loader"):
        loader = namespace.get(name)
        if loader is None:
            continue
        try:
            if callable(getattr(loader, "shutdown", None)):
                loader.shutdown()
            elif getattr(loader, "persistent_workers", False):
                iterator = getattr(loader, "_iterator", None)
                if iterator is not None:
                    iterator._shutdown_workers()
                    loader._iterator = None
        except BaseException:
            log.write(f"{name} shutdown failed:\n" + traceback.format_exc())
    children = multiprocessing.active_children()
    for child in children:
        if child.is_alive():
            child.terminate()
    for child in children:
        child.join(timeout=5)
        if child.is_alive():
            child.kill()
            child.join(timeout=5)
    remaining = [child.pid for child in multiprocessing.active_children() if child.is_alive()]
    threads = [thread.name for thread in threading.enumerate() if thread.name == "mixed-loader-join"]
    return {"remaining_workers": remaining, "remaining_join_threads": threads}


def raise_open_file_limit():
    """Raise the soft open-file limit; many loader workers need more than 1024."""
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    wanted = 65536 if hard == resource.RLIM_INFINITY else min(hard, 65536)
    if soft != resource.RLIM_INFINITY and soft < wanted:
        resource.setrlimit(resource.RLIMIT_NOFILE, (wanted, hard))
    return resource.getrlimit(resource.RLIMIT_NOFILE)


def speed_matrix_run(output, settings, notebook_path):
    """Allow one bounded benchmark job through the existing G4 notebook runner."""
    if notebook_path != ROOT / 'notebooks/05_xunet_v4_nonorm.ipynb':
        raise ValueError('V3_SPEED_MATRIX=on requires notebook 05')
    status_path = output / 'status.json'
    start = time.perf_counter()
    status = {'state': 'running', 'started_at': now(), 'ended_at': None, 'heartbeat_at': now(),
              'exit_code': None, 'effective_settings': settings, 'runtime_settings': {'speed_matrix': True},
              'cells': [], 'executed_notebook': None, 'log': str(output / 'run.log'),
              'run_output_dir': str(output), 'run_files': [], 'traceback': None}
    atomic_json(status_path, status)
    (output / 'executed.ipynb').write_text(json.dumps({'cells': [{'cell_type': 'markdown',
        'metadata': {}, 'source': ['Speed matrix job; no notebook cells were executed.']}],
        'metadata': {'headless_run': {'source': str(notebook_path), 'settings': settings}},
        'nbformat': 4, 'nbformat_minor': 5}, indent=2) + '\n')
    stop = threading.Event()
    def heartbeat():
        while not stop.wait(10):
            status['heartbeat_at'] = now()
            atomic_json(status_path, status)
    pulse = threading.Thread(target=heartbeat, daemon=True)
    pulse.start()
    with (output / 'run.log').open('w', encoding='utf-8') as log:
        log.write('SPEED MATRIX (notebook 05 bypassed): ' + json.dumps(settings, sort_keys=True) + '\n')
        log.flush()
        process = subprocess.Popen([sys.executable, '-u', 'scripts/speed_matrix.py', '--output', str(output)],
                                   cwd=ROOT, env=os.environ.copy(), stdout=log, stderr=subprocess.STDOUT,
                                   start_new_session=True)
        handlers = {sig: signal.getsignal(sig) for sig in (signal.SIGINT, signal.SIGTERM)}
        def forward(sig, _frame):
            try:
                os.killpg(process.pid, sig)
            except ProcessLookupError:
                pass
        try:
            for sig in handlers:
                signal.signal(sig, forward)
            code = process.wait()
        finally:
            for sig, handler in handlers.items():
                signal.signal(sig, handler)
            if process.poll() is None:
                process.terminate()
                process.wait(timeout=10)
    stop.set(); pulse.join(timeout=2)
    status.update(state='done' if code == 0 else 'failed', exit_code=code,
                  ended_at=now(), elapsed_s=time.perf_counter()-start,
                  run_files=[str(p) for p in output.rglob('*') if p.is_file()])
    atomic_json(status_path, status)
    return code


def run(args):
    notebook_path = args.notebook.resolve()
    if not notebook_path.is_file():
        raise ValueError(f"notebook not found: {notebook_path}")
    document = json.loads(notebook_path.read_text())
    package_notebook = document.get('metadata', {}).get('cvdl_package', False)
    if not package_notebook:
        from scripts.g4.preflight import top_values
        try:
            top_values(document)
            package_notebook = True
        except (ValueError, SyntaxError):
            pass
    compact = package_notebook or notebook_path == ROOT / "notebooks/05_xunet_v4_nonorm.ipynb"
    settings = settings_file(args.env_file.resolve(), args.set, allow_native=compact)
    provided_job_id = bool(settings.get('CVDL_JOB_ID'))
    declared = {}
    if compact:
        from scripts.g4.preflight import top_values
        from cvdl.settings import validate_notebook_name
        declared = top_values(document)
        validate_notebook_name(notebook_path,declared)
    if declared.get('name') and not settings.get('CVDL_JOB_ID'):
        import secrets
        prior_path=args.output.resolve()/'status.json'
        prior=json.loads(prior_path.read_text()) if args.resume and prior_path.exists() else {}
        settings['CVDL_JOB_ID']=prior.get('effective_settings',{}).get('CVDL_JOB_ID') or (
            f"{declared['name']}-{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}-{secrets.randbelow(10000):04d}")
        settings['V3_NOTEBOOK']=notebook_path.relative_to(ROOT).as_posix()
    if provided_job_id:
        from scripts.g4.preflight import validate_notebook
        relative = notebook_path.relative_to(ROOT).as_posix()
        if settings.get('V3_NOTEBOOK') != relative:
            raise ValueError('job notebook argument differs from the pinned job-env path')
        validate_notebook(relative, root=ROOT)
    output = args.output.resolve()
    if output == ROOT or ROOT not in output.parents:
        raise ValueError("--output must be inside this repository")
    ignored = subprocess.run(["git", "check-ignore", "-q", str(output)], cwd=ROOT, check=False)
    if ignored.returncode != 0:
        raise ValueError(f"--output must be git-ignored: {output}")
    resolved_settings = None
    resolution_env = None
    runner_overrides = [*args.set, f"output_root={output / 'checkpoints'}"]
    if compact:
        sys.path.insert(0, str(ROOT))
        from cvdl.settings import ALIASES, load, runner_context
        resolution_env = {key: value for key, value in os.environ.items()
                          if (key in ALIASES and key != "QUICK" and not key.startswith("V3_"))
                          or key == "MIOPEN_GEMM_ENFORCE_BACKEND"}
        resolution_env = {**resolution_env, **settings, 'CVDL_HEADLESS_LOG': '1'}
        resolved_settings = load(environ=resolution_env, overrides=runner_overrides,
                                 **(declared if declared.get('name') else {}))
        if declared.get('name'):
            validate_notebook_name(notebook_path,dict(resolved_settings.values))
        # Keep legacy env/status and resume records stable. The scoped load()
        # context retains the original profile provenance instead of this bridge.
        for key, value in (("QUICK", str(resolved_settings.quick).lower()),
                           ("V3_WORKERS", str(resolved_settings.workers)),
                           ("V3_EVAL_WORKERS", str(resolved_settings.eval_workers))):
            if key not in settings and (key == "QUICK" or resolved_settings.profile == "g4"):
                settings[key] = value
    previous = output / "status.json"
    if output.exists() and any(output.iterdir()):
        if not args.resume or not compact or not previous.is_file():
            raise ValueError(f"use a new, empty output folder: {output} (or --resume for interrupted notebook 05)")
        prior = json.loads(previous.read_text(encoding="utf-8"))
        if prior.get("state") == "done" or prior.get("executed_notebook") != str(output / "executed.ipynb"):
            raise ValueError("only an interrupted notebook 05 run can resume")
        if (prior.get("effective_settings") != {**settings, "V3_OUTPUT_ROOT": str(output / "checkpoints")}
                or prior.get('setting_overrides', []) != args.set):
            raise ValueError("resume settings differ from the saved run")
        resume_name=resolved_settings.run_directory_name if declared.get('name') else settings.get('V3_RUN_NAME')
        if not resume_name or not (output / "checkpoints" / resume_name / "last.ckpt").is_file():
            raise ValueError("resume needs the existing last.ckpt")
    output.mkdir(parents=True, exist_ok=True)
    if compact:
        os.environ["CVDL_HEADLESS_LOG"] = "1"
    sys.path.insert(0, str(ROOT))
    from scripts.host_monitor import monitor, on_gce
    monitor_enabled = args.host_monitor == 'on' or (args.host_monitor == 'auto' and on_gce())
    if monitor_enabled:
        args.monitor_stop = threading.Event()
        args.monitor_thread = threading.Thread(target=monitor,
            args=(output / 'host.csv', args.monitor_stop),
            kwargs={'runner_pid': os.getpid()}, name='host-monitor', daemon=True)
        args.monitor_thread.start()
    settings["V3_OUTPUT_ROOT"] = str(output / "checkpoints")
    for key in list(os.environ):
        if key == "QUICK" or key == "PRED_PATH" or key.startswith("V3_"):
            os.environ.pop(key)
    os.environ.update(settings)
    OPEN_FILE_LIMIT = raise_open_file_limit()  # inherited by loader workers
    os.chdir(ROOT)
    speed_mode = settings.get('V3_SPEED_MATRIX', 'off')
    if speed_mode not in {'on', 'off'}:
        raise ValueError('V3_SPEED_MATRIX must be on or off')
    if speed_mode == 'on':
        return speed_matrix_run(output, settings, notebook_path)

    executed = output / "executed.ipynb"
    status_path = output / "status.json"
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            cell["outputs"] = []
            cell["execution_count"] = None
    notebook.setdefault("metadata", {})["headless_run"] = {"source": str(notebook_path), "settings": settings}
    status = {"state": "running", "current_cell": None, "last_completed_cell": None,
              "started_at": now(), "ended_at": None, "heartbeat_at": now(), "exit_code": None,
              "effective_settings": settings.copy(), "setting_overrides": list(args.set),
              "runtime_settings": {}, "cells": [],
              "executed_notebook": str(executed), "log": str(output / "run.log"),
              "run_output_dir": None, "run_files": [], "traceback": None}
    status_lock = threading.RLock()

    def save_status():
        with status_lock:
            status["heartbeat_at"] = now()
            atomic_json(status_path, status)

    def collect_files(namespace):
        name = namespace.get("RUN_NAME")
        if not isinstance(name, str):
            return
        active=namespace.get('settings')
        run_dir = output / "checkpoints" / (active.run_directory_name if getattr(active,'name',None) else name)
        files = [str(path) for path in sorted(run_dir.rglob("*")) if path.is_file()] if run_dir.exists() else []
        with status_lock:
            status["run_output_dir"] = str(run_dir)
            status["run_files"] = files

    recorded_settings = set()

    def runtime_settings(namespace):
        values = {name: namespace[name] for name in ("RUN_NAME", "QUICK", "V3_DATA", "EPOCHS", "BATCH_SIZE", "COMPILE_MODE") if name in namespace}
        data = namespace.get("otf_data")
        if data is not None:
            values["data_run_settings"] = deepcopy(data.record)
        if resolved_settings is not None and namespace.get("RUN_NAME") not in recorded_settings and namespace.get("RUN_NAME"):
            from cvdl.settings import Settings
            active_settings = namespace.get('settings')
            if not isinstance(active_settings, Settings):
                active_settings = resolved_settings
            active_settings.write(output / "checkpoints" / active_settings.run_directory_name)
            recorded_settings.add(namespace["RUN_NAME"])
        with status_lock:
            status["runtime_settings"] = values

    atomic_json(executed, notebook)
    save_status()
    stop_heartbeat = threading.Event()

    def heartbeat():
        while not stop_heartbeat.wait(10):
            save_status()

    pulse = threading.Thread(target=heartbeat, name="notebook-status-heartbeat", daemon=True)
    pulse.start()
    old_handlers = {signum: signal.getsignal(signum) for signum in (signal.SIGINT, signal.SIGTERM)}
    interrupted = {"signum": None}

    def signal_handler(signum, _frame):
        interrupted["signum"] = signum
        raise StopRequested(signum)

    for signum in old_handlers:
        signal.signal(signum, signal_handler)
    start = time.perf_counter()
    exit_code = 1
    namespace = {}
    try:
        from IPython.terminal.interactiveshell import TerminalInteractiveShell
        with (output / "run.log").open("a", encoding="utf-8", buffering=1) as log:
            def emit(message):
                log.write(message + "\n")
                log.flush()
                print(message, flush=True)

            emit("Headless notebook: " + str(notebook_path))
            emit(f"Open-file limit (soft, hard): {OPEN_FILE_LIMIT}")
            emit("Effective settings: " + json.dumps(settings, sort_keys=True))
            shell = TerminalInteractiveShell.instance(user_ns=namespace, simple_prompt=True)
            shell.enable_matplotlib("inline")
            # The terminal shell registers PNG figures but disables that MIME
            # formatter by default. Enable it for the saved notebook copy.
            shell.display_formatter.formatters["image/png"].enabled = True
            count = 0
            for index, cell in enumerate(notebook["cells"]):
                if cell["cell_type"] != "code":
                    continue
                count += 1
                with status_lock:
                    status["current_cell"] = index
                save_status()
                emit(f"CELL START {index}")
                outputs = cell["outputs"]
                cell_lock = threading.RLock()
                stdout = CellStream("stdout", log, sys.stdout, outputs, cell_lock, compact=compact)
                stderr = CellStream("stderr", log, sys.stderr, outputs, cell_lock, compact=compact)
                original_publish = shell.display_pub.publish

                def publish(data, metadata=None, **_kwargs):
                    with cell_lock:
                        outputs.append({"output_type": "display_data", "data": mime_data(data),
                                        "metadata": metadata or {}})
                        log.write("[display: " + ", ".join(data) + "]\n")
                        log.flush()

                shell.display_pub.publish = publish
                cell_start = time.perf_counter()
                failure = None
                try:
                    context = runner_context(resolution_env, runner_overrides) if compact else nullcontext()
                    with context, redirect_stdout(stdout), redirect_stderr(stderr):
                        result = shell.run_cell("".join(cell["source"]), store_history=True)
                    failure = result.error_before_exec or result.error_in_exec
                    if interrupted["signum"] is not None:
                        failure = StopRequested(interrupted["signum"])
                except BaseException as error:
                    failure = error
                finally:
                    shell.display_pub.publish = original_publish
                if compact:
                    outputs.append({"output_type": "stream", "name": "stdout",
                                    "text": f"Cell {index}: full stdout/stderr in run.log.\n"})
                # Join once per completed cell. Repeated string += here copied the
                # entire fit log for every progress update (quadratic runtime).
                for item in outputs:
                    if item.get("output_type") == "stream" and isinstance(item.get("text"), list):
                        item["text"] = "".join(item["text"])
                cell["execution_count"] = count
                seconds = time.perf_counter() - cell_start
                record = {"cell": index, "seconds": seconds, "state": "failed" if failure else "done"}
                if failure:
                    trace = "".join(traceback.format_exception(failure))
                    if not trace.strip():
                        trace = repr(failure)
                    record["traceback"] = trace
                    outputs.append({"output_type": "error", "ename": type(failure).__name__,
                                    "evalue": str(failure), "traceback": trace.splitlines()})
                    with status_lock:
                        status["traceback"] = trace
                    exit_code = (128 + interrupted["signum"]) if interrupted["signum"] else 1
                    emit(f"CELL FAIL {index}: {type(failure).__name__}: {failure}")
                    log.write(trace + "\n")
                else:
                    with status_lock:
                        status["last_completed_cell"] = index
                    emit(f"CELL DONE {index}: {seconds:.3f}s")
                with status_lock:
                    status["cells"].append(record)
                runtime_settings(namespace)
                collect_files(namespace)
                normalize_stream_outputs(notebook)
                atomic_json(executed, notebook)
                save_status()
                if failure:
                    break
            else:
                exit_code = 0
                emit(f"ALL {count} CODE CELLS PASS in {time.perf_counter() - start:.3f}s")
    except BaseException as error:
        exit_code = (128 + interrupted["signum"]) if interrupted["signum"] else 1
        with status_lock:
            status["traceback"] = "".join(traceback.format_exception(error))
        with (output / "run.log").open("a", encoding="utf-8") as log:
            log.write(status["traceback"] + "\n")
        normalize_stream_outputs(notebook)
        atomic_json(executed, notebook)
    finally:
        for signum, handler in old_handlers.items():
            signal.signal(signum, handler)
        stop_heartbeat.set()
        pulse.join(timeout=2)
        with (output / "run.log").open("a", encoding="utf-8") as log:
            status["cleanup"] = cleanup_workers(namespace, log)
        collect_files(namespace)
        status["state"] = "done" if exit_code == 0 else "failed"
        status["exit_code"] = exit_code
        status["ended_at"] = now()
        status["current_cell"] = None
        status["elapsed_s"] = time.perf_counter() - start
        normalize_stream_outputs(notebook)
        atomic_json(executed, notebook)
        save_status()
    return exit_code


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--notebook", type=Path, default=ROOT / "notebooks/04_baseline_kv_xunet_v3.ipynb")
    parser.add_argument("--resume", action="store_true", help="resume an interrupted notebook 05 output with identical settings")
    parser.add_argument("--env-file", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                        help="one-off setting override (lr=0.0003 or legacy V3_LR=0.0003); repeatable")
    parser.add_argument('--host-monitor', choices=('auto', 'on', 'off'), default='auto',
                        help='host.csv: on for GCE by default, optional on the PC')
    args = parser.parse_args()
    try:
        return run(args)
    except (OSError, ValueError) as error:
        print(f"run_notebook: {error}", file=sys.stderr)
        return 2
    finally:
        if hasattr(args, 'monitor_stop'):
            args.monitor_stop.set()
            args.monitor_thread.join(timeout=5)


if __name__ == "__main__":
    sys.exit(main())
