"""Revision identity shared by current evaluation outputs."""
import subprocess


def code_commit(root):
    try:
        return subprocess.check_output(['git','rev-parse','--short','HEAD'],cwd=root,text=True,stderr=subprocess.DEVNULL).strip()
    except (OSError,subprocess.CalledProcessError):
        return 'unknown'
