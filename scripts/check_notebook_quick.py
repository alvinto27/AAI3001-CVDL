"""Execute every code cell of the unchanged scratch notebook, QUICK only.

No notebook client dependency. Capture each cell's stdout and traceback into a
separate scratch record. Notebook checkpoints/logs must use a scratch output root.
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import subprocess
import time
import traceback


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--notebook",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    args=p.parse_args()
    if os.environ.get("QUICK","").lower() not in {"true","1","yes"}:
        raise ValueError("QUICK must be explicit")
    output=Path(os.environ["V3_OUTPUT_ROOT"]).resolve()
    repo=Path("/srv/aai3001/AAI3001-CVDL").resolve()
    if output==repo or repo in output.parents:
        raise ValueError("smoke output must not be inside VM checkout")
    processes=subprocess.check_output(["nvidia-smi","--query-compute-apps=pid,process_name,used_memory",
                                      "--format=csv,noheader,nounits"],text=True).strip()
    gpu=subprocess.check_output(["nvidia-smi","--query-gpu=name,utilization.gpu",
                                "--format=csv,noheader,nounits"],text=True).strip()
    if processes or "L4" not in gpu or int(gpu.rsplit(",",1)[1]):
        raise ValueError(f"GPU must be idle L4: {gpu}; {processes}")
    import sys
    from IPython.terminal.interactiveshell import TerminalInteractiveShell
    sys.path.insert(0,str(Path.cwd()))
    # Terminal subclass implements enable_gui; run_cell is InteractiveShell's.
    shell=TerminalInteractiveShell.instance(user_ns=globals(),simple_prompt=True)
    shell.enable_matplotlib("inline")
    notebook=json.loads(args.notebook.read_text())
    start=time.perf_counter()
    records=[]
    try:
        for index,cell in enumerate(notebook["cells"]):
            if cell["cell_type"]!="code":
                continue
            print(f"CELL START {index}",flush=True)
            cell_start=time.perf_counter()
            # One IPython namespace, like a VS Code kernel, including display/magics.
            result=shell.run_cell("".join(cell["source"]),store_history=True)
            error=result.error_before_exec or result.error_in_exec
            if error is not None:
                raise RuntimeError(f"Notebook cell {index} failed") from error
            records.append(dict(cell=index,status="PASS",seconds=time.perf_counter()-cell_start))
            print(f"CELL DONE {index}: {records[-1]['seconds']:.3f}s",flush=True)
    except Exception:
        records.append(dict(cell=index,status="FAIL",traceback=traceback.format_exc()))
        raise
    finally:
        args.output.write_text(json.dumps(dict(cells=records,seconds=time.perf_counter()-start,
                                              idle_before=gpu),indent=2)+"\n")
    print(f"ALL CELLS PASS; seconds={time.perf_counter()-start:.6f}",flush=True)


if __name__=="__main__":
    main()
