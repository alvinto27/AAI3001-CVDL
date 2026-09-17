"""Use optional project-local dependencies; ordinary Python installations work too."""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if (ROOT / ".vendor").is_dir():
    sys.path.insert(0, str(ROOT / ".vendor"))
if (ROOT / ".tcl" / "tcl8.6" / "init.tcl").is_file():
    os.environ.setdefault("TCL_LIBRARY", str(ROOT / ".tcl" / "tcl8.6"))
    os.environ.setdefault("TK_LIBRARY", str(ROOT / ".tcl" / "tk8.6"))
