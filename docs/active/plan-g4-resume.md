# G4 resume after a spot preemption (ideas #19)

**Status: approved by the user 2026-10-23 as the SLIM version below; built `58aa081` (FULL 458 pass); install on the VM pending (a maintenance boot: `install -m755` of `scripts/g4/job.sh` to `~/.local/lib/cvdl-g4/job.sh` and `job_run.sh` to `job_run.last-good`; exact commands in `runs/.scratch/g4-resume-impl-2026-10-23/README.md`). Worker_2's full plan follows for reference (the slim list wins where they differ).**

| Part | Slim version (approved) |
| --- | --- |
| Command | `submit.sh resume <job>`: the same job ID and output `runs-out/<job>/`, the same commit and notebook from the job record; an attempt counter so a reboot cannot run the same resume twice |
| Checks | same commit, notebook and settings (`run_notebook.py --resume` checks the settings); `last.ckpt` present and readable, SHA recorded; job state interrupted (not done, failed or timed out); refuse on any mismatch, never start fresh silently |
| Checkpoint safety | every checkpoint save writes a temporary file in the same folder, flushes it (fsync), renames it over the target (`os.replace`), then flushes the folder |
| Records | the same Pi folder, overwritten at the end; `run.log` keeps every attempt; `attempts.jsonl` (start, end, epoch and step in/out, reason) for time and cost |
| Tests | mocked VM/job states for the runner logic; one CPU run interrupted at an epoch boundary and resumed, compared with an uninterrupted run where exact |
| Not included | receipts and single-use tokens, staged Pi generations and manifests, the watchdog |

Resume continues after the last completed epoch (the current epoch is lost). The 2026-10-23 curriculum50 run (`77f0ecd`, old runner) cannot use it.

---

**Read-only proposal: no code, VM access, launch or GPU work.** Standalone resume, not GBRAS; watchdog excluded. Inspected submit.sh → installed job.sh → job_run.sh → run_notebook.py, shared runtime/callbacks/curriculum and existing tests; cross-checked Lightning2.6.6 docs and locked fsspec2026.9.0 source.

## 1. Keep one training identity; number boot attempts

Reuse **runs-out/<job>/** and CVDL_JOB_ID. A new training ID changes run-directory/settings identity and checkpoint callback paths; translating those is larger/riskier. Store numbered attempts inside the stable job.

Propose **submit.sh resume <job>**: recover original published commit/notebook/env from an immutable submission receipt; require VM stopped, pinned preflight and a unique resume-request token. Metadata carries the resume action/token; the two-key job-env stays unchanged. Update installed job.sh's existing-record gate to select the SAME original commit for authorized resume. job_run.sh accepts eligible interrupted jobs, consumes each token once under a lock, records an attempt and passes **run_notebook.py --resume --output runs-out/<job>**. Repeated metadata must never execute twice. Fresh run still needs a new ID; separate immutable identity from mutable state.

## 2. Validate before overwriting records or restoring

Bind commit, notebook bytes/path, env, ordered overrides, resolved settings (including workers/packing/precision/compiler controls for v1), output/run identity, data/config/validation hashes and initialization. Existing run_notebook validates settings/path/overrides and checkpoint existence, but not commit/notebook bytes independently: add those guards.

Require interrupted state, no active writer, readable **last.ckpt**, SHA256 recorded, complete model/optimizer/scheduler/loop/callback state, consistent epoch/global-step/LR, original epoch budget and valid best-checkpoint references. Refuse success/code failure/timeout, missing/corrupt state or any mismatch. Never silently start fresh, load weights-only, reset Adam/warmup, extend50 epochs or substitute best.

Existing Trainer.fit(ckpt_path='last') restores training state. Curriculum must match training_start.json and checkpoint schedule identity/table/quotas. OTF_Epoch reselects the restored epoch; SeedSequence-based schedules/embedding/augmentation/pairing/order can match exactly with unchanged controls. **Resume after the last committed validated epoch:** discard the partial current epoch/validation and prefetched batches. No mid-epoch cursor or general process-RNG restoration is promised; nondeterministic CUDA kernels prevent a bit-identical GPU trajectory guarantee.

## 3. Checkpoint safety needs hardening

TimedModelCheckpoint delegates to Lightning TorchCheckpointIO → _atomic_save → fsspec transaction. It is **not unconditionally atomic/durable locally**: locked fsspec uses a default-temp-directory file and shutil.move; across filesystems that can copy over the destination. Explicit fsync is absent. VM mount layout was not inspected.

Smallest robust fix: shared local CheckpointIO writes a unique **same-directory** temporary file, flushes/fsyncs, os.replace's the destination, then fsyncs its directory. Clean orphan temps safely. Protect best saves too; defer deleting the prior best until last commits, or refuse inconsistent recovery. Record accepted resume SHA/state per attempt. Test interruption during save and between best/last updates; never silently fall back to fresh training.

## 4. One Pi result, several boots; preserve accounting

Separate training state from delivery: current deliver_pending can mark an abruptly stopped job pending/delivered with unknown exit, not proof of completion. Snapshot old status.json/executed.ipynb before replacement; run.log already appends. Scope host/NVIDIA/console telemetry by attempt rather than truncate it; preserve CSV/TensorBoard and identify replayed timing/count rows.

Deliver to the same Pi job directory through staged generations plus an atomic attempt manifest/checksums/active-file inventory: partial uploads or obsolete best files must not look final. Done requires successful final attempt and matching checkpoint hashes. Upload retry never triggers training.

Keep earliest start and cumulative boot/attempt intervals, checkpoint SHA/epoch/step in/out, preemption evidence, compile/validation/save overhead and replayed work. Estimate spend through confirmed VM stop, including wasted work/boot/delivery; label estimates and bound unknown intervals. Latest elapsed_s/checkpoint steps alone cannot represent total cost.

## 5. Operator use; existing77f0ecd jobs

Manual resume requires verified Spot interruption and explicit remaining time/spend allowance; exit137/stale heartbeat alone is insufficient. Retain timeout/shutdown margins and actual GCE duration/STOP checks: max-run-minutes metadata is not scheduling. Later watchdog invokes this same command; no automatic retry here.

**Existing77f0ecd cannot use the new protocol as-is:** local notebook/Trainer resume exists, but its pinned runner rejects recorded IDs and lacks new manifest/write guarantees. Switching job-commit changes training identity. Legacy rescue requires separately approved, audited migration/independent runner pin—not hot-patching an active run. Never merge original epoch37 output with the fresh relaunch or transfer only its weights into a new identity.

## 6. Ordered implementation and effort:12–20 engineering hours

1. Immutable identity, receipt/action/token, bootstrap/runner state transition (3–5h).
2. Durable checkpoint writer and full-state/best-consistency guards (3–5h).
3. Attempt-aware status, Pi delivery and cumulative accounting (2–4h).
4. No-cloud tests/review (4–6h): extend test_g4_jobs with mocked metadata/gcloud/SSH/rsync, duplicates/concurrency, mismatches, terminal states, truncated checkpoints and interrupted writes/uploads. Extend the existing tiny shared-runtime curriculum50 CPU resume fixture: compare uninterrupted vs epoch-boundary interrupted/resumed data/tags, optimizer/scheduler/step, stages/counts and model results exactly where deterministic; test7→8/32→33 boundaries, non-improving last saves and partial-epoch discard. Required suite plus FULL before a later G4 launch; install/deployment needs separate approval.

API reference: [Lightning2.6.6 checkpoint contents/resume](https://github.com/Lightning-AI/pytorch-lightning/blob/2.6.6/docs/source-pytorch/common/checkpointing_basic.rst), Context7 `/lightning-ai/pytorch-lightning/2.6.6`; atomic-write details checked directly against installed, lock-matching source.
