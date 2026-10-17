"""Guards for PC loader experiments: a worker cap, a memory check, and a process group that dies with its parent.

The PC is shared (16 cores, 31 GB). A real training worker can hold 0.2-0.8 GB, so 32 real workers need ~20 GB.
"""
from __future__ import annotations
import os
import signal
import sys

MAX_PC_WORKERS=12
GB_PER_WORKER=1.0   # measured ~0.65 GB per real training worker (32 workers ~20 GB); rounded up
GB_RESERVE=4.0


def available_gb():
    with open('/proc/meminfo') as stream:
        for line in stream:
            if line.startswith('MemAvailable:'):
                return int(line.split()[1])/1024/1024
    raise RuntimeError('MemAvailable not found')


def check_workers(workers, *, limit=MAX_PC_WORKERS, available=None):
    """Refuse a run that would exceed the worker cap or the memory budget. Returns the available GB."""
    if workers>limit:
        raise SystemExit(f'refusing {workers} real workers: the PC limit is {limit}. Use synthetic workers to test ordering.')
    have=available_gb() if available is None else available
    need=GB_PER_WORKER*workers+GB_RESERVE
    if have<need:
        raise SystemExit(f'refusing {workers} workers: need ~{need:.0f} GB available, only {have:.1f} GB')
    return have


def kill_group():
    """Error path only: SIGKILL every process in this group (this process included)."""
    try:
        signal.signal(signal.SIGTERM,signal.SIG_IGN)
        os.killpg(0,signal.SIGKILL)
    except OSError:
        pass


def own_process_group():
    """Become a process-group leader; SIGTERM/SIGINT kill the whole group so workers cannot be orphaned.

    A normal finish does not use this: shut the loader down, flush and return from main() (exit status 0).
    """
    try:
        os.setpgid(0,0)
    except OSError:
        pass
    def stop(signum, frame):
        try:
            kill_group()
        finally:
            os._exit(128+signum)
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)


def guarded(main):
    """Run main(); on an exception print it, kill the whole group and exit non-zero. A normal return exits 0."""
    import traceback
    try:
        main()
    except SystemExit:
        raise
    except BaseException:
        traceback.print_exc();sys.stdout.flush();sys.stderr.flush()
        kill_group()
        os._exit(1)
