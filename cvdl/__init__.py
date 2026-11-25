"""Shared CVDL experiment code; importing this also exposes the flat LSB library.

The generator directory has a hyphen and is not a Python package. Keep this
single compatibility bridge until that independently documented library moves.
"""
from pathlib import Path
import sys

_GENERATOR = str(Path(__file__).resolve().parents[1] / "LSB-Dataset-Generator")
if _GENERATOR not in sys.path:
    sys.path.insert(0, _GENERATOR)
