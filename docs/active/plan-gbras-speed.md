# GBRAS speed-ups and unattended long runs

**Status: approved by the user 2026-10-22.** Sequence as in section 5: (1) BF16 boundary (SRM + clamp in FP32) and channels-last; (2) one G4 timing and numerics boot (<= 45 min); (3) resume in the G4 runner; (4) watchdog; (5) the long GBRAS run. Caps: 3 preemption relaunches, 15 stockout attempts, **$20 total** (timing boot included). BF16 training is recorded as a difference from SRM-30 (FP32). **The watchdog runs on the Pi** (user: the PC is not on overnight); the Pi needs the gcloud CLI, limited credentials, VM SSH access and a system service (setup details for the user's OK when step 4 starts). Written by Worker_2; copied unchanged below apart from this header.

Read-only investigation: no code, training, GPU probe or VM start. Run14:128images/batch,2.48steps/s,37–42GB, approximately570min/30epochs ($15.49 at$1.63/h). PC timings are not transferable.

## 1. Speed options: one shared training path

`cvdl/settings.py` supports `precision=32|bf16-mixed`, `compile=off|default`; G4 uses default compile. `runtime.create_training` passes precision to Lightning and compiles net **in place**, preserving checkpoint keys. Reuse these controls and its precision plugin, not another training implementation.

| Option | Unmeasured incremental gain | Math/risk/support |
|---|---|---|
| BF16 mixed | Planning1.3–2×; may be smaller | Same equations, less precision. Lightning autocast; FP32 weights/Adam states, no gradient scaler. Residual rounding can damage steganalysis. |
| Channels-last weights **and inputs**, logical shape unchanged | 1–1.4×, possibly regression | Same equations; kernel/reduction rounding changes. Repeated format conversions negate gains. No setting exists: add a shared runtime/transfer control. |
| Default vs off; later reduce-overhead/max-autotune | Off is a control; alternatives perhaps0–20% beyond default | Same equations, different fusion/reduction. Extra compilation/CUDA-graph memory. Other modes currently rejected: extend central settings/runtime/records first. |
| cuDNN benchmark | No extra gain to claim | **Already enabled on CUDA** in `runtime.configure`; retain/record. |

**BF16 prerequisite:** `GBRASNet.features` currently has no autocast exclusion. Float input does not protect SRM convolution. Add a shared FP32 exclusion around **SRM90 and clamp**, then autocast adapter/body. Keep BN parameters, fixed gamma, running statistics and statistics accumulation FP32; verify CUDA backend behavior/dtypes, without forcing every BN activation to FP32. Cross-entropy is FP32 under CUDA autocast: assert this, and use FP32 validation probabilities/metrics. Never cast the whole model to BF16. Retain/record existing TF32 policy.

Keep128, LR, data, pairing and body unchanged. Existing pinned/nonblocking transfers remain. Measure loader waits before adding workers. No checkpoint-recomputation, pooling changes or BN removal as “cheap” fixes.

## 2. Numerical guard

Compare about200 matched FP32/BF16 updates: identical initialization, tile/label/augmentation sequence,128-image batches, optimizer/LR schedule and steps. Record loss windows, gradient norms/nonfinite counts and all13 BN statistics. On fixed balanced validation tiles across sources/k1–4, record logits/margins/probabilities initially and at matched checkpoints. Evaluate **both checkpoints FP32** to isolate learning drift; separately compare BF16-vs-FP32 inference on identical weights.

Proposed gates to approve before boot: no nonfinite values, no sustained adverse loss drift (window gap ≤max(.02,5% of FP32 mean)), same-weight inference probability p99 absolute gap≤.01. Record logit absolute/RMS/p99 differences, P_E/AUC and source results. Investigate failures; do not relax gates afterward. This screens risk, not proves full-run accuracy equivalence.

SRM-30 was FP32: declare BF16 training as an experimental difference. Save AMP boundaries, layout/compiler/TF32 flags, framework/CUDA/cuDNN versions, seeds, frozen validation hashes and control steps. Use common FP32 final evaluation; short controls do not eliminate the training-precision difference.

## 3. One bounded measurement boot

Approve a tracked timing-only notebook/tool using **shared runtime/Lightning**, sequential fresh trial processes, no GPU concurrency. Existing `scripts/speed_matrix.py` cannot be used unchanged: hardcoded XuNet/manual optimizer and per-step synchronization. G4 refuses `pilot=true`; use an explicit benchmark tool, not the PC pilot mode; outputs are not architecture results.

At128: FP32/default/NCHW → BF16/default/NCHW → BF16/default/channels-last. Only if time remains, try off or one centrally supported alternative compile mode. Allow compile/autotune plus20–30 warm steps, then100 measured steps/config. Reuse200 matched updates for section2. Synchronize only window boundaries; use asynchronous GPU events, no per-step `.item()`/sync. Save throughput, allocated/reserved/NVIDIA peak memory, compile time, loader waits and dtype/layout evidence. Time real validation/checkpoint overhead for the winner.

Target30–45min ($.82–$1.22): timeout45, **actual GCE backstop60min** ($1.63), reserving15min for delivery/shutdown. Drop extra trials if compilation consumes time. Abort on OOM/nonfinite values/bad numerics; no silent recipe changes. Project full training **plus full60da validation/checkpoints/boot/delivery**.

## 4. Resume: template yes, G4 automation no

GBRAS notebook calls `fit(...,ckpt_path="last")`; runtime overwrites `last.ckpt` after each validation epoch and keeps best. Lightning restores model, optimizer, scheduler, epoch/step. `run_notebook.py --resume` accepts interrupted package notebooks despite old “notebook05” help text, requiring the same output/settings/overrides/run identity and existing checkpoint.

But `job_run.sh` never passes resume, rejects recorded IDs and creates empty new-ID outputs. **Current relaunch starts from scratch**, although persistent disk retains checkpoints. Resume needs an approved extension to this same runner: separate boot-attempt ID from stable training identity/output, validate commit/recipe/precision/checkpoint, invoke existing resume, preserve best paths and test OTF epoch/seed continuity. Add atomic/fallback checkpoint protection against interruption during overwrite. Epoch-boundary resume loses the current epoch; exact mid-epoch sample replay is not promised. Verify disk retention/Spot termination action; never delete/recreate the VM. Without resume,5–10h Spot attempts can lose **all** progress repeatedly.

## 5. Sequence and prediction

Implement/test AMP boundary/layout → one numerical/timing boot → select measured winner → implement/test resume/watchdog if still long. Speculative combined **2–3×**, not independent guaranteed multipliers, means285–190min/30epochs. Optimistic3×: **190min/$5.16 training**, roughly205min/$5.57 with15min overhead; still over180min. A165min training allowance requires≥3.45×.

User must approve longer actual scheduling limit, timeout and spend. `submit.sh --max-run-minutes` sets metadata/validation, **not GCE scheduling**; verify both. This plan authorizes no launch.

## 6. Watchdog: reuse runner, supervise safely

`runs/.scratch/retry-launch.sh` covers capacity at launch only (4min,15attempts), filtering useful errors. Reuse cadence, inspect full results. Use canonical `scripts/g4/submit.sh` for submission/status. `off` changes metadata/starts off-mode boot; immediate stop requires `gcloud compute instances stop`.

Every60s inspect VM state plus bounded SSH reads: matching `status.json` (10s heartbeat), `run.log` (TRAIN progress about45s), job record and `checkpoints/<run>/epoch_timing.csv`. Heartbeat alone is not learning progress. Track step/epoch/log advancement and last epoch duration; bounded compile grace, then stall threshold max(2×last epoch,30min). Pi `/srv/cvdl-results/<job-id>/` delivery can be partial/stale; pending uploads resend next boot, not live monitoring.

| Case | Action |
|---|---|
| Matching done/exit0 | Stop/confirm off, check delivery; no restart, even if later preempted. |
| Code failure/traceback, timeout124 or duration cap | Stop/report; **never retry**.137 alone proves nothing. |
| Unfinished + authoritative GCE preemption event for this boot, no code failure | Bounded retry; resume only after section4 implementation. Currently fresh restart, losing interrupted compute. |
| Capacity/STOCKOUT operation failure | Retry4min, at most15capacity attempts globally; not IAM/quota/network errors. |
| STOPPING | Bounded wait; never start concurrently. |
| TERMINATED without proven cause, contradictory/unknown/stale state | Fail closed: stop/report, no speculative retry. |

Proposed approval caps: **3preemption relaunches,15stockout attempts,$20 total**, including benchmark/boots/wasted work. Persist counters/cost across watchdog restarts; single-instance lock, one capped log and atomic state/status line: UTC/job/VM/epoch/progress-age/spend/retries. Conservatively charge lifecycle time until confirmed off; reserve shutdown margin. Fresh retries can consume four attempts subject to the dollar cap: require explicit wasted-work approval without resume.

**Recommend always-on Pi after preparation.** Read-only SSH found Tailscale, but no gcloud on PATH or `~/.config/gcloud` for the checked account: credentials/IAM unverified. Needs CLI, least-privilege noninteractive VM-control/audit permissions, VM SSH access and boot-enabled system service. Submit currently hardcodes PC's `podman exec aai3001` preflight: make the **same preflight.py** portable to Pi CPU Python, not a duplicate launcher/bypass. PC has `Linger=no`; ordinary background/nohup/user services are unsafe at logout. An approved system-level PC service is a fallback, not assumed installed.

Final/cap/unknown: request stop and verify **TERMINATED**; bounded stop retries, operator-visible HIGH-PRIORITY status/log alert if confirmation fails. Keep runner shutdown trap and independently verified GCE backstop if watchdog dies; never claim guaranteed shutdown during API failure. Mock all decisions, completion/preemption races, partial delivery/status, stale heartbeat, duplicate instances, persisted caps, stockout exhaustion, stop failure and missing/corrupt checkpoints—**no real VM tests**.

Evidence: shared settings/runtime/lit/GBRAS, notebook, run_notebook/speed_matrix, g4 submit/job/job_run/preflight, retry script. API cross-check: Lightning precision/checkpoint docs (`/lightning-ai/pytorch-lightning`), PyTorch2.14 AMP docs (`/pytorch/pytorch/v2.14.0`).
