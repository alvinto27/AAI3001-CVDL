# G4 batch job: setup, run, send results, stop

> **The G4 disk uses the new data layout (`Datasets/originals/…`) since 2026-10-16 (checkout `9798b56`). Commits before `76f6257` no longer match it. The one-time change is recorded in the runbook section below.**

The Spot VM `rtx6000pro` is in `us-central1-c`. It has one RTX PRO 6000.
Its external IP changes at boot. The machine costs about $1.63 per hour when
running. **Do not start it just to test these scripts.** Its GCE maximum run
duration is ~~120 minutes~~ 18000 s (300 minutes, since 2026-10-22) from boot,
with action **Stop**. Change it only while the VM is TERMINATED. Maintenance mode does
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
The `Run` line includes `notebook=`. The installed bootstrap was verified on
2026-10-16: SHA-256
`89cc83e9c73cc43476f6daf5e77825ddc298a32a22a52af8ded0909f15b1f2af`.
This is the unchanged repository `job.sh`, not the old one-file runner. It does
not inspect `V3_NOTEBOOK`. Its source selects
`target="$ROOT/scripts/g4/job_run.sh"` and ends with
`exec /usr/bin/env bash "$target" "$@"`. Thus the job's checkout owns notebook
validation and runs its own `scripts/run_notebook.py`. **No reinstall or
maintenance boot is needed for this submission change.** No selector shim is
needed. `boot.log` must identify the checkout copy, not a last-good fallback.

A G4 notebook job writes `host.csv` into its job folder at 60-second intervals.
The checked-out runner starts the sampler; `job_run.sh` passes
`--host-monitor on` explicitly.
The sampler needs `/proc` and Linux cgroup files and delivers `host.csv` with
other job files. The first CPU and disk rates are blank until the next sample.
Notebook 05 also writes `checkpoints/<run-name>/epoch_timing.csv`: one row per
zero-based epoch with first-batch wait, train and validation spans, transition
spans, and callback durations. See [the time-field guide](../archive/plan-epoch-overhead.md#new-read-only-timers-for-the-next-run).
The monitor and timers do not change data streams or model selection.

The stable bootstrap is already installed. Do not reinstall it for changes
to `job_run.sh`, `run_notebook.py`, notebooks or profiles: push the job commit
instead. `setup.sh base` installs it only when setting up a new VM. A future
change to `job.sh` itself needs a maintenance install; that is not part of this
workflow.

**Automatic-update gate (`499296a`, 2026-10-26):** in run and resume modes, after the owned attempt start is recorded and before package work or training, `job_run.sh` requires the five units from `setup.sh freeze-updates` (`apt-daily.timer`, `apt-daily-upgrade.timer`, `apt-daily.service`, `apt-daily-upgrade.service`, `unattended-upgrades.service`) to be masked and inactive. A failed query, a blank or unknown state, or an unmasked or active unit refuses the attempt (exit 2): the console and the attempt-end reason name each unit, then normal delivery and shutdown follow. The gate is read-only (no repair, no stop of a package transaction). Maintenance and off modes are unchanged. Cause: run 20 was stopped by `apt-daily-upgrade` on 2026-10-26 ([journal](../journal/2026-10-26/README.md)); `freeze-updates` was applied to the VM that day (`17b5aac`).

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

The PC has the masters. Copy them **from the PC**, not from the Seoul VM, into
`Datasets/originals/{raise,div2k,alaska,lsdir}` (RAISE and DIV2K whole; ALASKA and
LSDIR only the files named in `data/<source>/photo_manifest.csv`, so the 3,896 unused
ALASKA files stay on the PC). `setup.sh upload` runs `scripts/g4/upload_masters.sh`: it
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

## One-time data-layout change (maintenance boot)

**Why:** the G4 disk still has `ImageSets/` and `Datasets/{alaska,lsdir}` (the old
layout). Records from commit `76f6257` on name `Datasets/originals/…`. Run this
once, before the next job. It needs no network copy: photos are renamed in place.

**Before the boot (PC):**

1. Push the branch so the VM can fetch the commit (`git push origin yx`) and note its SHA.
2. Optional, only for `--clean-runs`: `bash scripts/g4/delivered_manifest.sh /tmp/pi.manifest`
   (read-only on the Pi: lists the delivered files and the SHA-256 of each `.ckpt`).

**Maintenance boot (user starts the VM with `job-mode=maintenance`):**

```sh
scp -i ~/.ssh/cvdl-g4 scripts/g4/reorg_data_layout.sh yangxuan@EXTERNAL_IP:   # the VM checkout is old and lacks it
ssh -i ~/.ssh/cvdl-g4 yangxuan@EXTERNAL_IP
bash ~/reorg_data_layout.sh --commit SHA            # dry run: inventory, plan, checks; changes nothing
bash ~/reorg_data_layout.sh --commit SHA --apply    # act (add --keep-cache to keep a cache)
git -C ~/AAI3001-CVDL status --porcelain            # must print nothing
sudo shutdown -h now
```

What `--apply` does, in order: refuses (changes nothing) if the checkout is dirty, a
target exists, a path is a symlink or on another filesystem, or the target commit
does not have the new layout or does not ignore `runs-out/`; renames the five photo
folders (same filesystem, instant); keeps the small `raise512-run1` files under
`Datasets/generated/raise512-run1-records/` and deletes its `clean/ stego/ masks/`
(only if present); deletes an `otf-cache` found at `~/otf-cache` or
`Datasets/otf-cache` (only if it holds a `cache.json`); checks out the target commit
(detached); then checks every `master_path`: existence for all, and size plus
SHA-256 for ALASKA and LSDIR against `provenance.csv`. RAISE and DIV2K records hold
no SHA, so only existence is checked (the PC proof compared their SHA before and
after the move). The script exits non-zero and prints `Master check FAILED` on any
problem; do not start a job then. Unknown items in `ImageSets/` and `Datasets/` are
listed and never deleted. It never touches `runs-out/.g4-state` or
`~/.local/lib/cvdl-g4`. Rerunning `--apply` after a stop is safe.

**Time (estimate; not measured on the VM):** renames and checkout, seconds. Deleting a
53 GB cache, under a minute. The SHA check reads about 46 GB (18,200 ALASKA + 11,000
LSDIR files; 1,900 RAISE/DIV2K files are only looked up): about 5 to 10 minutes at
100 to 200 MB/s. `--no-sha` checks sizes only (seconds). Use the 120-minute boot limit as the bound.

**Clean old job folders (optional, separate):** the dry run already lists, per job
folder under `runs-out/`, the size and the per-epoch checkpoints and TensorBoard logs.
It deletes nothing by default. `--clean-runs --delivered-manifest FILE` (with
`--apply`) removes only per-epoch `*.ckpt` (not `last*`/`best*`) and `logs/tb/` files
of jobs whose VM record says `delivered=yes` **and** whose file size (and SHA-256 for
`.ckpt`) match the Pi manifest. Status, logs, results and last/best checkpoints stay.

## Submit a run

The PC has a user-local Google Cloud CLI in `~/.local/opt/google-cloud-sdk`
with `~/.local/bin/gcloud` linked to it. The installed version is 589.0.0.
The official unversioned Linux x86_64 archive matched Google's published
SHA-256 `817026aec0adea0e2ee8ed3a1ba3eabfe9e5177581fbe16595a8fe495a601300`
([Google install guide](https://docs.cloud.google.com/sdk/docs/install-sdk)).
No login was done during installation. The user must do these steps in a
browser session. The project `personal-compute-511601` is recorded in this
repository's project journal; `submit.sh` sets zone `us-central1-c` on each
command.

```sh
gcloud auth login
gcloud config set project personal-compute-511601
gcloud config set compute/zone us-central1-c
```

After the user says **go**, review a dry run before any paid boot:

```sh
git fetch origin yx
# The notebook and its settings must already be committed and pushed.
bash scripts/g4/submit.sh run --notebook notebooks/newdata-1.ipynb --commit SHA --timeout-min 105 --dry-run
# Only after approval: repeat the same command without --dry-run.
```

The command is `run --notebook PATH [--commit SHA] [--timeout-min N]
[--dry-run]`. The default commit is `HEAD`; the submitted value is always the
resolved SHA, never a moving branch name. Even in dry-run, it refuses a dirty
checkout or a commit not reachable from the local `origin/yx` tracking ref.
Fetch that ref after a push. A notebook must be a regular tracked `.ipynb`
under `notebooks/`, outside every `archive/` directory, **at that commit**.
A missing file, symlink, traversal path or archived notebook fails on the PC.
Notebook jobs require a phase-5-or-newer commit that understands the minimal
job-ID metadata; pre-flight rejects older commits before any boot. The speed
tool retains its old metadata interface.

### PC pre-flight, before metadata or boot

`submit.sh` runs Python only through `podman exec -w /work aai3001`. It reads
the selected commit's notebook, settings package and configs in a temporary
snapshot, not the working-tree notebook or current shell's experiment env.
It reads the first code cell's literal `settings = load(...)` keyword values
and resolves them with the declared G4 profile. No photos, model, data loader
or GPU computation is started. It prints final settings, sources and RUN_NAME.
`<checkout>` in printed paths means the VM's selected checkout root. A typo,
bad value or machine-owned notebook value stops here, including in dry-run.

Use the same first-cell format as notebook 05: import `load` from
`cvdl.settings`, then one `settings = load(...)` call. Values must be literals
(numbers, strings, tuples, lists or dictionaries), not computed expressions,
`**kwargs`, env reads or profile/override arguments. All experiment choices
belong here, in git. Frozen notebook 04 lacks this settings format and remains
unchanged; it is not a new-format submission recipe.

### G4 detection and machine-owned values

Runtime detection requires both Google DMI (`product_name = Google Compute
Engine` or `sys_vendor = Google`) and an available **CUDA, not ROCm, RTX PRO
6000** GPU. Google DMI without that GPU fails; that GPU without Google DMI also
fails rather than falling back to PC settings. The checked-out job runner sets
`CVDL_EXPECTED_PROFILE=g4`, so any PC result on a G4 job fails loudly too.
The PC pre-flight uses declared G4 capabilities only; it does not weaken these
runtime hardware checks.

| G4 setting | Value |
| --- | --- |
| Train/eval workers | 32 / 16 |
| QUICK | false |
| Precision | 32; BF16 mixed allowed only when runtime capability permits it |
| Compile / next-epoch prime | default / true (machine performance flags) |
| Persistent validation workers | true when eval workers are nonzero |
| Paths | checkout-relative data/configs; `runs-out/` output root; `--output` chooses the job's checkpoint root |

PC keeps 6/4 workers, QUICK true, eager compilation and no next-epoch prime.
Notebook top cells cannot set machine fields. Worker/path/capability fields
are not copied from experiment env files.

### Metadata and checkout handover

A notebook submission generates a temporary **two-line** job env:

```text
V3_NOTEBOOK=notebooks/newdata-1.ipynb
CVDL_JOB_ID=newdata-1-YYYYMMDDTHHMMSSZ-NNNN
```

It contains no experiment values. The file is sent through the existing
`job-env` instance-metadata channel and removed locally, including after a dry
run. The separate instance keys stay `job-mode`, `job-id`, `job-commit`,
`job-timeout-min`, and `max-run-minutes`. The matching ID is checked on the VM.
The installed `job.sh` fetches/checks out the commit and execs its checked-out
`job_run.sh`. That script validates the two-line env and notebook against the
pinned checkout before training, then calls that checkout's runner. A malformed
or missing notebook fails and follows the existing delivery/shutdown trap.
`V3_NOTEBOOK` and `CVDL_JOB_ID` are machine metadata, not experiment settings;
the settings parser accepts them and records their source.

`submit.sh` changes no GCE state in dry-run. Otherwise it checks the VM is
`TERMINATED` or `STOPPED`, sets metadata, and starts it. **Removing --dry-run
starts a billable VM.** Maintenance/off still change only `job-mode`; status
only describes the instance. Other metadata is left alone. The user needs
Compute Instance Admin (v1), plus Service Account User when required for the
attached account; see [Google's metadata permissions](https://docs.cloud.google.com/compute/docs/metadata/setting-custom-metadata).
The generated env is far below the 256 KiB single-value / 512 KiB total
metadata limits. Credentials are the user's `gcloud auth login`, not the VM's
service account.

### Reproduce the past recipes

| Recipe | Tracked experiment notebook | Historical env fixture |
| --- | --- | --- |
| Mixed-source wide R2 | [newdata-1](../../notebooks/newdata-1.ipynb) | [archived env](../../scripts/g4/jobs/archive/newdata-1.env) |
| Standard-source wide R2 | [wide-1](../../notebooks/wide-1.ipynb) | [archived env](../../scripts/g4/jobs/archive/wide-1.env) |

Each is a copy of notebook 05 with only its top cell filled in. Tests require
exact experiment values, RunSettings, RUN_NAME and every comparable
~~golden~~ global (since 2026-10-21 resolved directly from the archived env, without the golden). The only two named machine differences are asserted explicitly:
`CHECKPOINT_ROOT` is `runs-out/`, not the old PC capture's `/work/runs`, and
`ROCM_CONV1_FIX_ACTIVE` is false on CUDA G4, not true as in that ROCm capture.
A third deliberate difference since `7921d58` (2026-10-21, user): notebook 05,
`newdata-1` and `wide-1` set `post_fit_eval="none"`, so a run skips the old 512
end-of-training test (the archived envs still resolve `full`). Evaluate on the PC
with the 256 suite. These recipes have no train-matched 256 validation, so a new
checkpoint needs an approved calibration mapping before the deployment threshold.
The archived env files are read-only parity fixtures and cannot submit a
notebook job. The historical golden JSON stays unchanged (archived 2026-10-21 to `tests/archive/golden/`).

Timing estimates are not promises. Historical full-data counts were 107,900
images / batch 32 = 3,372 steps per epoch. Prime-1 measured 915 s fit over 10
epochs and 1,138 s for the full runner (including 87 s diagnostics); doubling
fit plus 223 s overhead estimated 2,053 s (34:13) for 20 epochs. R1 later
measured 1,627 s fit over 20 epochs. At that rate 30 epochs estimate 2,441 s;
97 s standard test, 86 s diagnostics, 100–120 s extra test and 38 s other work
estimate 2,762–2,782 s (46–47 min). A 75-minute timeout leaves about 28 minutes
above that estimate. Do not use the old incorrect 1,686-step estimate.

### Separate speed matrix (env retained)

The speed matrix is a tool, not a notebook experiment. It still uses
[speed-1.env](../../scripts/g4/jobs/speed-1.env), unchanged:

```sh
bash scripts/g4/submit.sh run --env-file scripts/g4/jobs/speed-1.env --timeout-min 50 --dry-run
```

`--env-file` is accepted **only** for `V3_SPEED_MATRIX=on` with the notebook-05
selector. Its settings are also validated on the PC against the selected
commit and declared G4 capabilities before any metadata/boot command. The
checked-out `run_notebook.py` dispatches `scripts/speed_matrix.py`
instead of notebook cells. It runs six independent batch/precision train speed
configurations, with 50 warm-up and 600 measured steps, and saves
`speed-matrix.json`, one `*-nvidia.csv` per configuration, `run.log` and
`status.json`. There is no validation, diagnostics, extra panel or convergence
comparison. BF16 uses autocast without loss scaling. Priming is set but not
exercised before the first epoch boundary. LR is manual, never batch-scaled.
Reports include steps/images per second, loader wait, model/transfer time,
peak memory and measure-phase NVIDIA samples. Estimate 20–35 minutes
($0.54–$0.95 at $1.63/hour); a full 50-minute timeout is about $1.36. These are
estimates, not results. Normal notebook submissions cannot use legacy envs.

The default runner timeout is 100 minutes, never above `max-run-minutes - 15`
(default maximum 120). SIGTERM is followed by SIGKILL after 30 s; interrupted
files are delivered when possible. GPU telemetry is sampled every 60 s.
Spot preemption or Pi downtime can leave `delivered=no`; the next boot retries
pending transfers, with the existing bounded five-minute total retry budget.
A recorded ID never runs twice. Use a new ID for a new attempt.

## Approved-boot checklist

1. For a new machine only, complete the maintenance setup/upload/cache/speed
   steps above. Existing G4 bootstrap and disk-layout setup are already done.
2. Commit/push the intended notebook and code; fetch `origin/yx` on the PC.
   On that commit, full tests (`scripts/test_cpu.py`) must pass ([test levels](test-levels.md)).
   ~~and the golden compare~~ (golden retired 2026-10-21).
   **Tensor-size check (`2054c99`, 2026-10-26):** full preflight runs the production model with fake (meta-backed) CPU tensors: training forward + backward and checkpoint-validation forward, capped or not. It reports the largest op outputs per stage and **refuses** any tensor over 2,147,483,647 elements (exactly that passes); an op without fake support also refuses, by name. No GPU, real allocation or compilation. It checks logical tensor sizes only, not VRAM, backend workspaces, kernel compatibility or speed. R1's uncapped notebook now refuses (SRM conv `[128,90,512,512]` = 3,019,898,880); cap 64 passes. `--resume-identity-only` stays an identity check.
   **Launch gate (`42cdfe8`, 2026-10-26):** run and resume (also `--dry-run` and the watchdog's pending resume) make read-only Pi and GCE checks before any metadata write or start. Pi (`scripts/g4/pi_launch_check.sh`, the same unprivileged script over the admin SSH from the PC, or locally for the watchdog): `/srv` mounted from `/dev/sda1` with read-write filesystem (superblock) options; the upload user (`cvdl-inbox`) owns `/srv/cvdl-results` with owner write, and can search `/srv` and the folder by mode bits. This is a metadata check, not a write probe or an ACL audit. GCE: `maxRunDuration` 18000 s and termination action STOP. Each check prints PASS/FAIL on stderr; a failure, timeout or malformed answer refuses. No keys, IAM, upload restrictions or service hardening changed. Live read-only check by the Orchestrator 2026-10-26 (as `rpi400` and as `cvdl-watchdog`): all five Pi checks pass.
3. Run the exact notebook + SHA command with `--dry-run`. Review effective
   settings, RUN_NAME, job ID, timeout and the two-line metadata env.
4. Only after explicit user approval, repeat without `--dry-run`.
5. Check delivered `status.json`, `executed.ipynb`, `run_settings.json`,
   `boot.log` checkout-script evidence and final VM stop. The write-only inbox
   key cannot inspect results; use the Pi admin account.

## Result folders and headless output

Keep all PC and G4 results together in `runs/<run>/` on this PC checkout.
**Exception:** `scripts/g4/job.sh`, `scripts/g4/job_run.sh`, and the VM-side
`speed` subcommand of `scripts/g4/setup.sh` keep using `runs-out/` as their
output root **on the VM** (for example `runs-out/.g4-state`,
`runs-out/<job-id>/`); that is the VM's own checkout, unrelated to this
repository's `runs/`, and is left unchanged. When a finished job's result
folder is copied from the Pi (`/srv/cvdl-results/<job-id>/`) onto this PC, it
lands under `runs/<job-id>/`, not `runs-out/<job-id>/`. A notebook 05
headless job writes its working checkpoints in
`<output-root>/<run>/checkpoints/<model-run>/` (`<output-root>` is
`runs-out/` on the VM, `runs/` once copied to this PC): one overwritten
`last.ckpt` to resume, plus one best validation-P_E checkpoint for
evaluation. It does not save every epoch. `run.log` is the only full text copy of notebook output. `executed.ipynb` keeps figures and pointers to that log. The G4 service console retains short job and step/rate reports, not another copy of the notebook output. On a PC, use the explicit notebook 05 runner `--resume` option only for an interrupted run with matching settings and `last.ckpt`. On G4, a Spot preemption requires a new job ID and a fresh run; the service refuses an existing job ID and never resumes its folder. Notebook 04 behavior is unchanged.

## PC pilot of a train-matched notebook

Use `pilot=true` only on the PC. It caps actual training draws at 4,096 per
epoch and selects a fixed 256-pair slice (seed 20261019) from the run's full,
verified train-matched validation list. The full recipe guard runs first with
the declared training budget; no frozen file is changed or rebuilt. The slice
keeps the real validation geometry (256 px for recipe `60da4391043e8ce48ff2`).
QUICK does not resample this slice.

This command runs 256 training steps in two epochs at batch 64. The short
schedule, batch and machine options are explicit overrides, not pilot defaults:

```sh
podman exec -d -w /work aai3001 bash -c 'python -B scripts/run_notebook.py --notebook notebooks/experiments/srm-delayed.ipynb --env-file /dev/null --output runs/.scratch/pilot-srm-delayed --set pilot=true --set epochs=2 --set batch_size=64 --set lr_drops=2:1 --set rocm_conv1_fix=off --set workers=4 --set eval_workers=4 > logs/pilot-srm-delayed.log 2>&1; echo $? > logs/pilot-srm-delayed.rc'
```

Use a new empty output folder. Check the `.rc` file and log after the run.
`run_settings.json` records `pilot=true`; `data_run_settings.json` records the
declared and effective training budgets, full-list hash, slice hash, seed and
pair IDs. The output folder also holds the exact slice in `validation_pairs.csv`.
G4 preflight and the G4 runner refuse pilot mode. The evaluator and report
builder also refuse pilot runs. Do not use pilot scores as experiment results.
Delete the pilot output folder and its log/rc files after you record the smoke
check. Without `pilot=true`, all existing settings and guards stay unchanged.

## PC-only checks

`job.sh --test --metadata-file FILE` (the bootstrap, which then runs
`job_run.sh` from `G4_REPO_ROOT`; `G4_LIB_DIR` sets the last-good folder) reads a local text file with lines such
as `job-mode=run`, `job-id=test-id`, `job-env=/path/to/env-file` and
`job-timeout-min=0.05`. It runs Python **inside** `aai3001` and prints the
shutdown step; it never calls GCE metadata or shuts down the PC. Set
`G4_REPO_ROOT` and `G4_STATE_ROOT` (inside the ignored `runs/` folder, for
example `runs/.g4-test-state`, so a local test run does not recreate a
`runs-out/` folder in this PC checkout) for isolated tests. `--inbox-host WRONG_HOST` and `--retry-seconds 2` are test
options. Do not use test options on the VM. Test transfers can be inspected
and removed using the allowed `rpi400` account; do not modify other Pi paths.
