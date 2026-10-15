# G4 batch job: setup, run, send results, stop

The Spot VM `rtx6000pro` is in `us-central1-c`. It has one RTX PRO 6000.
Its external IP changes at boot. The machine costs about $1.63 per hour when
running. **Do not start it just to test these scripts.** Its GCE maximum run
duration is 120 minutes from boot, with action **Stop**. Maintenance mode does
not bypass that limit.

The boot service is enabled only after setup. It reads **instance** metadata
(not project metadata). Its default mode is **off**: it resends pending
results and shuts down. Set `job-mode=maintenance` explicitly to keep it up.
In run mode it sends pending results, runs one new job, tries to send its files
to the Pi for no more than five minutes in total, and calls `shutdown -h now`.
The service uses systemd `Type=exec`, so it does not delay the boot target.
The service has `TimeoutStopSec=120s`. A Spot preemption may give only about
30 seconds: an upload can fail, and the next boot resends the partial folder.
Records left in `starting` or `running` with `delivered=no` are also resent.
If a stop gives no exit code, the recovered record says `unknown`; its end time
is the next boot's detection time, not the exact stop time.
It records each job at `runs-out/.g4-state/jobs/<job-id>.record` on the VM disk.
The record includes start and end time, exit code, state and delivery status.
The VM disk survives a normal stop. An interrupted upload remains pending and
is tried before the next job. A recorded job ID is **never run again**, even if
its result is pending or failed. Use a new ID to rerun training.

## Boot scripts: bootstrap and job logic

Two scripts run at boot. `scripts/g4/job.sh` is a small **bootstrap**.
`setup.sh base` installs it as `~/.local/lib/cvdl-g4/job.sh`, and the
service starts that copy. All job logic is in `scripts/g4/job_run.sh`.

In run mode the bootstrap does these steps:

1. It checks that the checkout is clean, fetches `origin/yx`, and checks out the job commit (`git checkout --detach`).
2. It runs `scripts/g4/job_run.sh` **from the checkout**, with `exec`. A repo change therefore needs no new install.
3. If the checked-out `job_run.sh` is missing or fails `bash -n`, it runs the last-good copy `~/.local/lib/cvdl-g4/job_run.last-good`. It saves a new last-good copy each time the checked-out file passes `bash -n`.
4. If no usable copy exists, it logs the error and shuts down (except in maintenance mode).

If the bootstrap git step fails, it does not stop. `job_run.sh` repeats the
step, records the failure, sends the error files and shuts down. The record,
already-recorded skip, pending delivery, timeouts and shutdown trap are all in
`job_run.sh`. The environment flag `CVDL_G4_STAGE` stops the bootstrap from
running twice. `job_run.sh` stops if its own checkout step moves the commit
after it started (the file could change while Bash reads it).

`boot.log` shows which copy ran. The bootstrap line has `bootstrap sha256=`.
The next line is `job_run copy: checkout|last-good path=... sha256=...`.
The `Run` line includes `notebook=`.

**One-time step.** The installed `job.sh` must be the bootstrap. The old
installed copy holds the job logic and does not run the checkout copy. In
the next **maintenance** boot, update the checkout and install both files:

```sh
cd ~/AAI3001-CVDL && git fetch -q origin yx && git checkout -q --detach origin/yx
install -m 755 scripts/g4/job.sh ~/.local/lib/cvdl-g4/job.sh
install -m 755 scripts/g4/job_run.sh ~/.local/lib/cvdl-g4/job_run.last-good
bash -n ~/.local/lib/cvdl-g4/job.sh && grep -c CVDL_G4_STAGE ~/.local/lib/cvdl-g4/job.sh
```

The count must be 1 or more. After that, do not install `job.sh` again unless
`job.sh` itself changes (rare). The service file does not change. A change
to `job.sh` needs a new maintenance install. A change to `job_run.sh` needs only
a pushed commit.

## First boot (person, maintenance mode)

Get the current external IP from the Google Cloud console. Install the PC
SSH public key for user `yangxuan` first. Start maintenance with
`submit.sh maintenance` after gcloud is installed, or set `job-mode=maintenance`
on the instance in the console before the first boot. Copy the setup script
and the write-only inbox private key from the PC:

```sh
scp -i ~/.ssh/cvdl-g4 scripts/g4/setup.sh yangxuan@EXTERNAL_IP:~/setup.sh
scp -i ~/.ssh/cvdl-g4 ~/.ssh/cvdl-g4-inbox yangxuan@EXTERNAL_IP:~/.ssh/cvdl-g4-inbox
ssh -i ~/.ssh/cvdl-g4 yangxuan@EXTERNAL_IP 'chmod 600 ~/.ssh/cvdl-g4-inbox; bash ~/setup.sh base'
```

The inbox key permits only write-only `rsync` into `/srv/cvdl-results` on
`rpi400-server`; it gives no shell or read access. Do not put the key in Git
or a job folder. `base` installs apt tools (including `build-essential` and `python3-dev`, which `torch.compile` needs) and uv, clones the public `yx`
branch over HTTPS, runs `uv sync --locked` (CUDA torch on this VM), installs
Tailscale, runs `tailscale up --hostname=rtx6000pro` **without** `--ssh`,
installs the inbox key, and enables the boot service. Approve the printed
Tailscale login URL once. It prints `nvidia-smi`, CUDA availability, the GPU
name and free disk. It is safe to rerun `base` after an interrupted setup; it
stops if the checkout has user changes. It never formats a disk or deletes
masters. The service starts on the next boot, not during setup.

## Masters, cache and short speed check (separate steps)

The PC has the masters. Copy only those three folders **from the PC**, not
from the Seoul VM. `setup.sh upload` runs `scripts/g4/upload_masters.sh`: it
splits all master files into disjoint lists (largest first, dealt round-robin)
and runs **10 parallel `rsync` streams** (`--streams N` changes this, 1 to 32)
with `--partial --append-verify`, a cheaper SSH cipher and no compression.
It is resumable: rerun the same command after any stop, and a half-copied file
continues. If any stream fails, the script prints its error and exits non-zero.
At the end it compares the file count and total bytes of each folder with the
PC and exits non-zero on a difference. Measured on the first real G4 boot: one
stream gave only about 3.5 MB/s (about 100 minutes for the data); 10 streams
gave about 20–21 MB/s, and 22.15 GB (RAISE 1,011 files, DIV2K train 800,
DIV2K valid 100) took about 25 minutes. The PC CPU and load were not the limit.

```sh
bash scripts/g4/setup.sh upload --host EXTERNAL_IP
ssh -i ~/.ssh/cvdl-g4 yangxuan@EXTERNAL_IP 'bash ~/AAI3001-CVDL/scripts/g4/setup.sh cache'
ssh -i ~/.ssh/cvdl-g4 yangxuan@EXTERNAL_IP 'bash ~/AAI3001-CVDL/scripts/g4/setup.sh speed'
```

`cache` builds `~/otf-cache/v1` on the G4; it does not copy a PC cache. It
checks all 1,900 arrays against the G4 masters. It also checks that the
1,900 normalized JSON records match the PC **and** Seoul VM reference digest
`bde842a00a9060d2a3b5e6c9e214a5d6f976a52f3e494f503a1290a93c8ae482`.
Only `master_mtime_ns` is ignored. The cache build is resumable. It needs
about 53 GB plus room for the 21 GB masters, code, dependencies and results.
G4 defaults to `nproc - 8` cache builders (40 on 48 vCPUs); `--workers N`
overrides it. `build_cache.py` still defaults to a maximum of eight workers;
the G4 step uses an explicit high-worker option. **Budget about 1 GiB of RAM
per builder worker** (up to about 40 GiB at 40 workers), not a measured peak.
This is below the G4's 180 GB RAM, even with the input/cache page cache;
stop or lower `--workers` if memory pressure appears. Measured on the first
real boot with 40 workers: build wall time **363.047 s** for 1,900 photos
(52,761,268,224 array bytes, 52,763,054,963 file bytes), an average write rate
of about **145 MB/s**. That is disk-limited: it matches the 140 MB/s Hyperdisk
baseline, not the CPU. The strict verify pass took 5.17 s with 0 mismatches
and the record digest matched the reference above. About 11 GiB RAM was in use
at the moment of one check, far below the 1 GiB per worker budget.

`speed` first runs `sync` and drops the Linux page cache with sudo. It reads
48 training cache photos, drops the page cache again, then reads 48 training
master photos. Neither first-pass sample is a full cold epoch. The report
gives MB/s, photos, logical bytes and kernel-attributed storage bytes. If
physical reads are below 80% of sampled bytes, `likely_cold` is false:
do not treat that rate as cold. It measures **loader only, without GPU** at
4/8/16 cache workers (15 seconds each) and 16/24/32/40 decode workers
(40 seconds each), with process and worker CPU use. It also measures
notebook-04 loader **plus** model training at 4/8/16 cache workers, eager and
compiled, and both modes on a fixed GPU batch: 20 warmup and **150 timed
steps** each. Allow about 10–12 minutes. The report compares the best cache
and decode loader rates to the fastest GPU-only rate. For each first-epoch
estimate it takes the slower of disk read time, CPU loader time, and GPU
training time. This is a lower bound because those tasks can overlap and
there is other work. Cache disk time uses train cache bytes (about 38 GB);
decode disk time uses train master bytes. The report is under
`runs-out/g4-speed-*.json`.

## Submit a run

Install and authenticate gcloud on the PC or Pi separately. Inspect commands
without starting the VM:

```sh
bash scripts/g4/submit.sh run --env-file path/to/job.env --dry-run
bash scripts/g4/submit.sh maintenance --dry-run
bash scripts/g4/submit.sh off --dry-run
bash scripts/g4/submit.sh status --dry-run
```

Remove `--dry-run` only when ready. The env file must include `QUICK` and a
simple `V3_RUN_NAME`; for a full run use `QUICK=False`, `V3_DATA=otf`,
`V3_SOURCE=cache`, `V3_CACHE_ROOT=/home/yangxuan/otf-cache/v1`, and a new
`V3_RUN_NAME`. See the [runner instructions](../README.md#run-notebook-04-without-jupyter).
`submit.sh` makes an ID from the run name and UTC time, sends the env text as
`job-env` instance metadata, sets `job-mode=run`, `job-id`, `job-commit`,
`job-timeout-min`, and `max-run-minutes`, then starts the stopped VM. It refuses
an already-running VM. The default commit is `origin/yx`. Set `--commit SHA`
to pin another reachable commit. The checkout must be clean. Git fetch and
checkout failures prevent training. After checkout, the job runs
`uv sync --locked` to match the pinned environment; failure uploads the error
files and shuts down.

The optional `V3_NOTEBOOK` env key selects only `notebooks/04_baseline_kv_xunet_v3.ipynb` (default if absent) or `notebooks/05_xunet_v4_nonorm.ipynb`; both submit and boot reject other values. It is never executed as shell code. For a v4 R1 full-data job, use a fresh env file like this (submit only after the notebook commit is available on the VM; do not reuse an older checkpoint):

```text
QUICK=False
V3_NOTEBOOK=notebooks/05_xunet_v4_nonorm.ipynb
V3_DATA=otf
V3_SOURCE=decode
V3_RUN_NAME=xunet_v4_clip3_nonorm_lr0.0003_warm1_drop10_e20_r1
V3_EPOCHS=20
V3_LR=3e-4
V3_WARMUP_EPOCHS=1
V3_GRAD_CLIP=1.0
V3_NORM=none
V3_WORKERS=32
V3_EVAL_WORKERS=16
V3_COMPILE=default
V3_PRIME_NEXT_EPOCH=on
```

Run `bash scripts/g4/submit.sh run --env-file path/to/v4-r1.env --timeout-min 60 --dry-run` first; remove `--dry-run` only after review and approval. For the 20-epoch budget, measured full-data training has **107,900 train images / 32 = 3,372 steps per epoch** (rounded up), and `g4-prime-1` fit took 915 s over 10 epochs. Its full runner took 18:58 (1,138 s), including 87 s for diagnostic cell 29. Doubling fit and retaining the 223 s non-fit overhead estimates **2,053 s (34:13)** for 20 epochs. A 60-minute runner timeout leaves about 26 minutes above that estimate. The remaining VM time (at most 60 minutes, less boot and setup time) covers delivery and shutdown; the script's minimum reserve is 15 minutes. This is an estimate, not a guarantee: compilation, IO, or diagnostics can run longer. Do not use the incorrect 1,686-step estimate.

### R2 full-data job (notebook 05)

Use a **new** job ID. Make an env file with these exact lines. The R2 notebook
commit must be on `origin/yx` before you submit the job. The VM still has the
old installed one-file `job.sh`; it supports `V3_NOTEBOOK=05`, so R2 does not
need the bootstrap install. Install the new bootstrap on a later maintenance
boot (see the one-time step above).

```text
QUICK=False
V3_NOTEBOOK=notebooks/05_xunet_v4_nonorm.ipynb
V3_DATA=otf
V3_SOURCE=decode
V3_RUN_NAME=xunet_v4_clip3_nonorm_lr0.0003_warm1_drops5x0.3-20x0.1_k1to4_e30_r2
V3_WORKERS=32
V3_EVAL_WORKERS=16
V3_COMPILE=default
V3_PRIME_NEXT_EPOCH=on
V3_EPOCHS=30
V3_LR=3e-4
V3_WARMUP_EPOCHS=1
V3_GRAD_CLIP=1.0
V3_NORM=none
V3_LR_DROPS=5:0.3,20:0.1
V3_TRAIN_YAML=configs/train-k1to4.yaml
V3_EXTRA_TESTS=test-bpp1
```

Use `bash scripts/g4/submit.sh run --env-file path/to/v4-r2.env
--timeout-min 75 --dry-run` to inspect the command. Do not start the G4
without the user's approval. R1 measured 1,627 s to fit 20 epochs; 30 epochs
estimate 2,441 s. Its test cell took 97 s and diagnostics took 86 s. Allow
100–120 s for the new extra test, plus about 38 s for other work: the total
estimate is **2,762–2,782 s (46–47 min)**. A 75-minute timeout leaves about
28 minutes above this estimate. It is at most `120 - 15 = 105` minutes, so the
job script accepts it. The VM maximum also needs room for boot, upload and
shutdown. These are estimates, not guarantees.

The default runner timeout is **100 minutes**. The job refuses a timeout above
`max-run-minutes - 15` (default maximum: 120). The reserve leaves time for
boot, upload and shutdown. `timeout` sends SIGTERM to the runner and then
SIGKILL after 30 seconds; interrupted results are sent if possible. The job
also logs GPU use, memory, power and temperature every 60 seconds in
`runs-out/<job-id>/nvidia.csv`. Check `status.json` in the delivered folder
before using its results. The Pi receives `<job-id>/` under
`/srv/cvdl-results`; the inbox account cannot read back. Each `rsync` attempt
has a bounded SSH connection timeout. A Pi outage leaves the VM record
`delivered=no`; the next run tries it first. Before each attempt, it waits
up to 30 seconds (inside the five-minute retry budget) for Tailscale and
`rpi400-server` name resolution. A Spot preemption can interrupt a transfer;
the next boot resends the files available on disk.

## First real boot checklist (in order)

1. In the Google Cloud console set **instance metadata** `job-mode=maintenance`,
   then start the VM. Get its new external IP. The service does not exist yet
   on the first setup boot, but set this value for the next boot too.
2. Copy the inbox key from the PC with the `scp` command above. Copy and run
   `setup.sh base` over SSH. Approve the Tailscale URL printed by `tailscale up`.
   Check NVIDIA, CUDA and free disk in the summary.
3. From the PC run `setup.sh upload --host EXTERNAL_IP`. On the VM run
   `setup.sh cache`; verify the 1,900 records. Then run `setup.sh speed` and
   inspect the cold rates and loader/GPU comparison.
   **Next boot after the first setup:** the service treats a start without
   `job-mode` metadata as **off**: it sends pending results and shuts down at
   once. To run `setup.sh speed` (the first setup boot skipped it), set
   `job-mode=maintenance` in the console **before** starting the VM, then
   connect over SSH with the new external IP.
4. Stop the maintenance VM (`sudo shutdown -h now`). In the Google Cloud
   console replace instance metadata with these keys, then start it again:

   | Key | QUICK smoke-test value |
   | --- | --- |
   | `job-mode` | `run` |
   | `job-id` | `g4-quick-YYYYMMDDTHHMMSSZ` (new, one folder name) |
   | `job-commit` | `origin/yx` (after this change is pushed) |
   | `job-timeout-min` | `20` |
   | `max-run-minutes` | `120` |
   | `job-env` | Literal multi-line text below, not a filename |

   ```text
   QUICK=True
   V3_DATA=otf
   V3_SOURCE=cache
   V3_CACHE_ROOT=/home/yangxuan/otf-cache/v1
   V3_RUN_NAME=g4-quick
   V3_WORKERS=8
   V3_EVAL_WORKERS=4
   V3_COMPILE=off
   ```

5. Confirm boot service `cvdl-g4-job` ran, the Pi received
   `/srv/cvdl-results/<job-id>/status.json` and `executed.ipynb`, its status is
   `done` with exit code 0, and the VM shut down. The write-only key cannot
   inspect the Pi: check with the Pi admin account. A plain console start with
   missing `job-mode` now **sends pending folders and shuts down**, not idle
   maintenance. For maintenance later, set the mode explicitly each time.

## PC-only checks

`job.sh --test --metadata-file FILE` (the bootstrap, which then runs
`job_run.sh` from `G4_REPO_ROOT`; `G4_LIB_DIR` sets the last-good folder) reads a local text file with lines such
as `job-mode=run`, `job-id=test-id`, `job-env=/path/to/env-file` and
`job-timeout-min=0.05`. It runs Python **inside** `aai3001` and prints the
shutdown step; it never calls GCE metadata or shuts down the PC. Set
`G4_REPO_ROOT` and `G4_STATE_ROOT` (inside the ignored `runs-out/` folder)
for isolated tests. `--inbox-host WRONG_HOST` and `--retry-seconds 2` are test
options. Do not use test options on the VM. Test transfers can be inspected
and removed using the allowed `rpi400` account; do not modify other Pi paths.
