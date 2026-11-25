"""PC loader guards: worker cap, memory check, and exit status / child cleanup of the safety helpers."""
import os
import signal
import subprocess
import sys
import textwrap
import time
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.pc_safety import check_workers, MAX_PC_WORKERS


def run_script(body, timeout=30):
    code = ("import sys,os,subprocess,time\nsys.path.insert(0,%r)\nTESTS=%r\nfrom scripts import pc_safety\n"
            % (str(ROOT), str(ROOT / 'tests'))) + textwrap.dedent(body)
    return subprocess.Popen([sys.executable, '-c', code], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)


def alive(pid):
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    try:   # a zombie is not running
        return open(f'/proc/{pid}/stat').read().rsplit(')', 1)[1].split()[0] != 'Z'
    except OSError:
        return False


class SafetyTests(unittest.TestCase):
    def test_worker_cap_and_memory_budget(self):
        self.assertEqual(check_workers(12, available=20.0), 20.0)
        with self.assertRaisesRegex(SystemExit, 'PC limit is 12'):
            check_workers(MAX_PC_WORKERS + 1, available=100.0)
        with self.assertRaisesRegex(SystemExit, 'only 10.0 GB'):
            check_workers(12, available=10.0)
        self.assertEqual(check_workers(1, available=5.0), 5.0)

    def test_normal_run_through_a_real_dataloader_exits_zero_cleanly(self):
        """A normal finish: DataLoader workers are shut down properly, no os._exit, no child kill, exit 0."""
        before = set(os.listdir("/dev/shm"))
        process = run_script('''
            import torch
            from torch.utils.data import DataLoader
            sys.path.insert(0, TESTS)
            from pc_safety_dataset import Counting
            def main():
                pc_safety.own_process_group()
                loader = DataLoader(Counting(), batch_size=4, num_workers=3, multiprocessing_context="spawn", persistent_workers=True)
                total = sum(len(batch[1]) for batch in loader)
                workers = [w.pid for w in loader._iterator._workers]
                print("PIDS", os.getpid(), *workers, flush=True)
                loader._iterator._shutdown_workers()
                loader._iterator = None
                print("TOTAL", total, flush=True)
            pc_safety.guarded(main)
        ''')
        out, err = process.communicate(timeout=60)
        self.assertEqual(process.returncode, 0, err)
        self.assertIn("TOTAL 120", out)   # each of the 3 workers yields the 40 items (no sharding in this tiny dataset)
        pids = list(map(int, out.split("PIDS")[1].splitlines()[0].split()))
        time.sleep(0.5)
        for pid in pids:
            self.assertFalse(alive(pid), f"process {pid} still alive after a normal exit")
        self.assertNotIn("leaked", err)       # multiprocessing resource_tracker: no leaked semaphores/shared memory
        # No shared-memory or semaphore entry left behind by this run (other users of /dev/shm are ignored).
        self.assertEqual(sorted(set(os.listdir("/dev/shm")) - before - {n for n in os.listdir("/dev/shm") if not n.startswith(("torch_", "sem.mp-", "psm_"))}), [])

    def test_exception_kills_every_child_and_exits_nonzero(self):
        process = run_script('''
            def main():
                pc_safety.own_process_group()
                kids = [subprocess.Popen(["sleep", "60"]) for _ in range(3)]
                print(" ".join(str(k.pid) for k in kids), flush=True)
                raise RuntimeError("boom")
            pc_safety.guarded(main)
        ''')
        out, err = process.communicate(timeout=30)
        self.assertNotEqual(process.returncode, 0)
        self.assertIn("boom", err)
        time.sleep(0.5)
        for pid in map(int, out.split()):
            self.assertFalse(alive(pid), f"child {pid} survived an exception")

    def test_no_os_exit_or_child_kill_on_the_normal_path(self):
        import inspect
        for name in ('main',):
            for path in (ROOT/'scripts/loader_hashes.py', ROOT/'scripts/profile_loader.py'):
                text = path.read_text()
                self.assertNotIn('os._exit', text, path.name)
                self.assertNotIn('kill_children_at_exit', text, path.name)
        self.assertFalse(hasattr(__import__('scripts.pc_safety', fromlist=['x']), 'kill_children_at_exit'))

    def test_sigterm_kills_the_whole_group(self):
        process = run_script('''
            pc_safety.own_process_group()
            kids = [subprocess.Popen(["sleep", "60"]) for _ in range(3)]
            print(" ".join(str(k.pid) for k in kids), flush=True)
            time.sleep(60)
        ''')
        pids = list(map(int, process.stdout.readline().split()))
        self.assertEqual(len(pids), 3)
        self.assertTrue(all(alive(p) for p in pids))
        process.send_signal(signal.SIGTERM)
        process.wait(timeout=30)
        process.stdout.close(); process.stderr.close()
        time.sleep(0.5)
        for pid in pids:
            self.assertFalse(alive(pid), f'child {pid} survived SIGTERM')
        self.assertIn(process.returncode, (-9, 128 + signal.SIGTERM))

    def test_group_kill_at_normal_exit_would_fail_the_exit_status_test(self):
        # Documents the earlier bug: a group-wide SIGKILL at normal exit also kills the caller (status -9 / 137).
        process = run_script('''
            pc_safety.own_process_group()
            os.killpg(0, 9)
        ''')
        process.communicate(timeout=30)
        self.assertEqual(process.returncode, -9)


if __name__ == '__main__':
    unittest.main()
