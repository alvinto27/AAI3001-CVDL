"""Small, read-only host sampler for a notebook job. Uses Linux /proc only."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import os
from pathlib import Path
import re
import threading
import time

FIELDS = ('time_utc', 'cpu_pct', 'iowait_pct', 'load1', 'load5', 'load15',
          'ram_used_mib', 'ram_available_mib', 'page_cache_mib', 'swap_used_mib',
          'disk_read_mib_s', 'disk_write_mib_s', 'python_processes',
          'python_rss_mib', 'runner_open_fds', 'cgroup_memory_mib')
MIB = 1024 * 1024
DISK = re.compile(r'^(?:nvme\d+n\d+|sd[a-z]+|vd[a-z]+|xvd[a-z]+)$')


def counters():
    line = next(x for x in Path('/proc/stat').read_text().splitlines() if x.startswith('cpu '))
    v = [int(x) for x in line.split()[1:]]
    # user, nice, system, idle, iowait, irq, softirq, steal
    cpu = (sum(v[:8]), v[3], v[4])
    disk = [0, 0]
    for line in Path('/proc/diskstats').read_text().splitlines():
        f = line.split()
        if DISK.fullmatch(f[2]):
            disk[0] += int(f[5]) * 512  # sectors read
            disk[1] += int(f[9]) * 512  # sectors written
    return cpu, disk, time.monotonic()


def sample(previous=None, runner_pid=None):
    cpu, disk, clock = counters()
    mem = {}
    for line in Path('/proc/meminfo').read_text().splitlines():
        k, rest = line.split(':', 1)
        mem[k] = int(rest.split()[0]) / 1024  # proc reports KiB
    python_count, rss_pages = 0, 0
    for process in Path('/proc').iterdir():
        if not process.name.isdigit():
            continue
        try:
            comm = (process / 'comm').read_text().strip()
            if not comm.startswith('python'):
                continue
            rss_pages += int((process / 'statm').read_text().split()[1])
            python_count += 1
        except (OSError, ValueError, IndexError):
            continue  # Processes can end between reads.
    load = os.getloadavg()
    try:
        fds = len(list((Path('/proc') / str(runner_pid or os.getpid()) / 'fd').iterdir()))
    except OSError:
        fds = ''
    try:
        cgroup_mib = round(int(Path('/sys/fs/cgroup/memory.current').read_text()) / MIB, 2)
    except (OSError, ValueError):
        cgroup_mib = ''
    row = dict(time_utc=datetime.now(timezone.utc).isoformat(timespec='seconds'),
               load1=round(load[0], 2), load5=round(load[1], 2), load15=round(load[2], 2),
               ram_used_mib=round(mem['MemTotal'] - mem['MemAvailable'], 2),
               ram_available_mib=round(mem['MemAvailable'], 2),
               page_cache_mib=round(mem['Cached'], 2),
               swap_used_mib=round(mem['SwapTotal'] - mem['SwapFree'], 2),
               python_processes=python_count,
               python_rss_mib=round(rss_pages * os.sysconf('SC_PAGE_SIZE') / MIB, 2),
               runner_open_fds=fds, cgroup_memory_mib=cgroup_mib,
               cpu_pct='', iowait_pct='', disk_read_mib_s='', disk_write_mib_s='')
    if previous is not None:
        before_cpu, before_disk, before_clock = previous
        ticks = cpu[0] - before_cpu[0]
        dt = clock - before_clock
        if ticks > 0 and dt > 0:
            row['cpu_pct'] = round(100 * (ticks - (cpu[1] - before_cpu[1]) - (cpu[2] - before_cpu[2])) / ticks, 2)
            row['iowait_pct'] = round(100 * (cpu[2] - before_cpu[2]) / ticks, 2)
            row['disk_read_mib_s'] = round((disk[0] - before_disk[0]) / MIB / dt, 3)
            row['disk_write_mib_s'] = round((disk[1] - before_disk[1]) / MIB / dt, 3)
    return row, (cpu, disk, clock)


def monitor(path, stop, interval=60, samples=None, runner_pid=None):
    path = Path(path)
    previous = None
    with path.open('w', newline='', encoding='utf-8') as output:
        writer = csv.DictWriter(output, FIELDS)
        writer.writeheader()
        count = 0
        while True:
            try:
                row, previous = sample(previous, runner_pid)
                writer.writerow(row)
                output.flush()
            except (OSError, ValueError, IndexError, KeyError) as error:
                # A monitor must not fail the model or the job.
                print(f'host monitor sample failed: {error}', flush=True)
            count += 1
            if count == samples or stop.wait(interval):
                break


def on_gce():
    try:
        product = Path('/sys/class/dmi/id/product_name').read_text().strip()
        vendor = Path('/sys/class/dmi/id/sys_vendor').read_text().strip()
        return product == 'Google Compute Engine' or vendor == 'Google'
    except OSError:
        return False


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--interval', type=int, default=60)
    parser.add_argument('--samples', type=int, default=3)
    args = parser.parse_args()
    if args.interval < 1 or args.samples < 1:
        parser.error('interval and samples must be positive')
    monitor(args.output, threading.Event(), args.interval, args.samples)
