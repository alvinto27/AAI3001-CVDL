"""Real pinned source fixtures for historical metadata/byte-validation tests.

These fixtures do NOT execute historical inference. Tests that need inference
must load/run the actual pinned checkout, not borrow its hashes for live code.
"""
from contextlib import contextmanager
import functools
from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]


@contextmanager
def pinned_source_tree(revision):
    with TemporaryDirectory() as tmp:
        root=Path(tmp)
        for name in ('configs','data','runs','notebooks','scripts','LSB-Dataset-Generator'):
            (root/name).symlink_to(ROOT/name,target_is_directory=True)
        (root/'cvdl').mkdir()
        names=subprocess.check_output(['git','ls-tree','--name-only',revision+':cvdl'],cwd=ROOT,text=True).splitlines()
        for name in names:
            if name.endswith('.py'):
                (root/'cvdl'/name).write_bytes(subprocess.check_output(['git','show',revision+':cvdl/'+name],cwd=ROOT))
        gitdir=subprocess.check_output(['git','rev-parse','--absolute-git-dir'],cwd=ROOT,text=True).strip()
        (root/'.git').write_text('gitdir: '+gitdir+'\n')
        yield root


def historical_sources(revision):
    def decorate(test):
        @functools.wraps(test)
        def wrapped(*args,**kwargs):
            with pinned_source_tree(revision) as root:
                with patch(test.__module__+'.ROOT',root),patch('scripts.training_decision.ROOT',root):
                    return test(*args,**kwargs)
        return wrapped
    return decorate
