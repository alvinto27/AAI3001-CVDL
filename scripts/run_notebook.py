"""Run a notebook headlessly and save a restart-safe executed copy after each cell.

Uses IPython and the standard library only. The tracked notebook is read only.
Run from a fresh git-ignored output folder; QUICK must be explicit.
"""
from __future__ import annotations

import argparse
import base64
from contextlib import redirect_stderr, redirect_stdout
from copy import deepcopy
from datetime import datetime, timezone
import json
import multiprocessing
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import threading
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
KEY = re.compile(r"^(?:QUICK|V3_[A-Z0-9_]+)$")


def now():
    return datetime.now(timezone.utc).isoformat()


def settings_file(path, overrides):
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
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"--set expects KEY=VALUE: {item!r}")
        key, value = (part.strip() for part in item.split("=", 1))
        if not KEY.fullmatch(key):
            raise ValueError(f"invalid --set key {key!r}")
        settings[key] = value
    if settings.get("QUICK", "").lower() not in {"true", "false", "1", "0", "yes", "no"}:
        raise ValueError("set QUICK explicitly to true or false")
    if "V3_OUTPUT_ROOT" in settings:
        raise ValueError("V3_OUTPUT_ROOT is set by --output, not the env file")
    name = settings.get("V3_RUN_NAME")
    if name and (name in {".", ".."} or Path(name).name != name or "\\" in name):
        raise ValueError("V3_RUN_NAME must be one folder name")
    return settings


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
    def __init__(self, name, log, terminal, outputs, lock):
        self.name, self.log, self.terminal = name, log, terminal
        self.outputs, self.lock = outputs, lock

    def write(self, text):
        if not text:
            return 0
        with self.lock:
            self.log.write(text)
            self.log.flush()
            self.terminal.write(text)
            self.terminal.flush()
            if self.outputs and self.outputs[-1].get("output_type") == "stream" and self.outputs[-1]["name"] == self.name:
                self.outputs[-1]["text"] += text
            else:
                self.outputs.append({"output_type": "stream", "name": self.name, "text": text})
        return len(text)

    def flush(self):
        with self.lock:
            self.log.flush()
            self.terminal.flush()

    def isatty(self):
        return False


class StopRequested(BaseException):
    def __init__(self, signum):
        self.signum = signum
        super().__init__(f"received {signal.Signals(signum).name}")


def cleanup_workers(namespace, log):
    loader = namespace.get("train_loader")
    if loader is not None and callable(getattr(loader, "shutdown", None)):
        try:
            loader.shutdown()
        except BaseException:
            log.write("Training loader shutdown failed:\n" + traceback.format_exc())
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


def run(args):
    notebook_path = args.notebook.resolve()
    if not notebook_path.is_file():
        raise ValueError(f"notebook not found: {notebook_path}")
    settings = settings_file(args.env_file.resolve(), args.set)
    output = args.output.resolve()
    if output == ROOT or ROOT not in output.parents:
        raise ValueError("--output must be inside this repository")
    ignored = subprocess.run(["git", "check-ignore", "-q", str(output)], cwd=ROOT, check=False)
    if ignored.returncode != 0:
        raise ValueError(f"--output must be git-ignored: {output}")
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"use a new, empty output folder: {output}")
    output.mkdir(parents=True, exist_ok=True)
    settings["V3_OUTPUT_ROOT"] = str(output / "checkpoints")
    for key in list(os.environ):
        if key == "QUICK" or key == "PRED_PATH" or key.startswith("V3_"):
            os.environ.pop(key)
    os.environ.update(settings)
    os.chdir(ROOT)
    sys.path.insert(0, str(ROOT))

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
              "effective_settings": settings.copy(), "runtime_settings": {}, "cells": [],
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
        run_dir = output / "checkpoints" / name
        files = [str(path) for path in sorted(run_dir.rglob("*")) if path.is_file()] if run_dir.exists() else []
        with status_lock:
            status["run_output_dir"] = str(run_dir)
            status["run_files"] = files

    def runtime_settings(namespace):
        values = {name: namespace[name] for name in ("RUN_NAME", "QUICK", "V3_DATA", "EPOCHS", "BATCH_SIZE", "COMPILE_MODE") if name in namespace}
        data = namespace.get("otf_data")
        if data is not None:
            values["data_run_settings"] = deepcopy(data.record)
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
                stdout = CellStream("stdout", log, sys.stdout, outputs, cell_lock)
                stderr = CellStream("stderr", log, sys.stderr, outputs, cell_lock)
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
                    with redirect_stdout(stdout), redirect_stderr(stderr):
                        result = shell.run_cell("".join(cell["source"]), store_history=True)
                    failure = result.error_before_exec or result.error_in_exec
                    if interrupted["signum"] is not None:
                        failure = StopRequested(interrupted["signum"])
                except BaseException as error:
                    failure = error
                finally:
                    shell.display_pub.publish = original_publish
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
        atomic_json(executed, notebook)
        save_status()
    return exit_code


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--notebook", type=Path, default=ROOT / "notebooks/04_baseline_kv_xunet_v3.ipynb")
    parser.add_argument("--env-file", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                        help="override one QUICK or V3_* value from the env file")
    args = parser.parse_args()
    try:
        return run(args)
    except (OSError, ValueError) as error:
        print(f"run_notebook: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
