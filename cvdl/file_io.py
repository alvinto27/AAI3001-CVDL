"""Same-folder exclusive temporary files with ordinary umask/default-ACL permissions."""
import os
from pathlib import Path
import secrets


def default_mode_temp(path):
    """Create like open(..., 'w'), without changing the process-wide umask."""
    path=Path(path)
    while True:
        name=path.parent/f'.{path.name}.{secrets.token_hex(16)}.tmp'
        try:
            fd=os.open(name,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o666)
        except FileExistsError:
            continue
        return fd,str(name)
