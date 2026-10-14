# G4 batch job: setup, run, send results, stop

The Spot VM `rtx6000pro` is in `europe-north1-a`. It has one RTX PRO 6000.
Its external IP changes at boot. The machine costs about $1.63 per hour when
running. **Do not start it just to test these scripts.** Its GCE maximum run
duration is 120 minutes from boot, with action **Stop**. Maintenance mode does
not bypass that limit.

The boot service is enabled only after setup. It reads **instance** metadata
(not project metadata). In maintenance mode it does not run work or shut down.
In run mode it sends pending results, runs one new job, tries to send its files
to the Pi for no more than five minutes in total, and calls `shutdown -h now`.
It records each job at `runs-out/.g4-state/jobs/<job-id>.record` on the VM disk.
The record includes start and end time, exit code, state and delivery status.
The VM disk survives a normal stop. An interrupted upload remains pending and
is tried before the next job. A recorded job ID is **never run again**, even if
its result is pending or failed. Use a new ID to rerun training.

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
or a job folder. `base` installs apt tools and uv, clones the public `yx`
branch over HTTPS, runs `uv sync --locked` (CUDA torch on this VM), installs
Tailscale, runs `tailscale up --hostname=rtx6000pro` **without** `--ssh`,
installs the inbox key, and enables the boot service. Approve the printed
Tailscale login URL once. It prints `nvidia-smi`, CUDA availability, the GPU
name and free disk. It is safe to rerun `base` after an interrupted setup; it
stops if the checkout has user changes. It never formats a disk or deletes
masters. The service starts on the next boot, not during setup.

## Masters, cache and short speed check (separate steps)

The PC has the masters. Copy only those three folders **from the PC**, not
from the Seoul VM. `rsync --append-verify` resumes interrupted files:

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

`speed` first runs `sync` and drops the Linux page cache with sudo. It reads
a bounded first-pass sample of training cache photos; this is **not** a full
cold epoch. Its report gives MB/s, photos, logical bytes and kernel-attributed
storage bytes. If kernel-attributed reads are below 80% of sampled bytes,
`likely_cold` is false and the estimate must not be treated as cold. It estimates
the first full training epoch's cache read time
from the sample rate and the train photo bytes. It then measures notebook-04
loader **and** model training at 4, 8 and 16 workers, eager and compiled:
12 warmup and 32 timed steps each. It compares the estimated cold read time
to the estimated steady training epoch time at the fastest worker count. Aim
for five minutes total; compilation can make it longer. The report is under
`runs-out/g4-speed-*.json`. Do not call a cached or first-pass sample a measured
full cold epoch.

## Submit a run

Install and authenticate gcloud on the PC or Pi separately. Inspect commands
without starting the VM:

```sh
bash scripts/g4/submit.sh run --env-file path/to/job.env --dry-run
bash scripts/g4/submit.sh maintenance --dry-run
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
to pin another reachable commit. The checkout must be clean; Git fetch and
checkout failures prevent training.

The default runner timeout is **100 minutes**. The job refuses a timeout above
`max-run-minutes - 15` (default maximum: 120). The reserve leaves time for
boot, upload and shutdown. `timeout` sends SIGTERM to the runner and then
SIGKILL after 30 seconds; interrupted results are sent if possible. The job
also logs GPU use, memory, power and temperature every 60 seconds in
`runs-out/<job-id>/nvidia.csv`. Check `status.json` in the delivered folder
before using its results. The Pi receives `<job-id>/` under
`/srv/cvdl-results`; the inbox account cannot read back. Each `rsync` attempt
has a bounded SSH connection timeout. A Pi outage leaves the VM record
`delivered=no`; the next run tries it first. A Spot preemption can interrupt
a transfer; the next boot resends it.

## PC-only checks

`job.sh --test --metadata-file FILE` reads a local text file with lines such
as `job-mode=run`, `job-id=test-id`, `job-env=/path/to/env-file` and
`job-timeout-min=0.05`. It runs Python **inside** `aai3001` and prints the
shutdown step; it never calls GCE metadata or shuts down the PC. Set
`G4_REPO_ROOT` and `G4_STATE_ROOT` (inside the ignored `runs-out/` folder)
for isolated tests. `--inbox-host WRONG_HOST` and `--retry-seconds 2` are test
options. Do not use test options on the VM. Test transfers can be inspected
and removed using the allowed `rpi400` account; do not modify other Pi paths.
