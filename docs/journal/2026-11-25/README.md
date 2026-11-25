# 2026-11-25 — submission prep; V3c baseline re-scored; model-history ladder; ONNX engine

## Did

- Started submission prep. The README is the report; the grading rubric has five rows: dataset, model development, deployment (local + cloud, optimized model), user interface, scalability.
- User approved: re-score only the V3c baseline `g4-prime-1` (the course-style KV XuNet) on the frozen standard-256 / hard-256 suites, so that the README can compare the shipped model against it. No other model is re-run; all others already have evaluations on the same suites.
- Code (no change to the scoring method):
  - `scripts/evaluate_run.py`: second pinned legacy migration `g4-prime-1-v3c-v1` (evidence hashes and notebook-derived model values; [record](../../active/legacy-wide-r2-migration.md#g4-prime-1-v3c-baseline-migration-2026-11-25)).
  - `scripts/build_calibration.py`: `g4-prime-1` added; it shares the wide-R2 list `8328cc6b0673cade0312` because its saved training photos and data settings are identical (the builder checks this).
  - `configs/evaluation-calibration-v1.json`: entry 16 appended; the first 15 entries are byte-identical. `MAPPING_SHA` `9d349b11...9c11` → `6ab71f35338d384575e04101fc8cc666a6381b646e2c977da8ceec228f20cf7c` (`cvdl/deployment.py`, `configs/comparison-constants-v1.json`).
  - Tests: `tests/test_deployment.py` (16 entries, appended entry = wide-R2 entry with the baseline identity); `tests/test_legacy_wide_migration.py` (4 new baseline tests, including the real-record test). Focused 27 OK. FAST level: **663 tests OK (35 skipped), 327 s, exit 0**.
- Evaluation: `podman exec -w /work aai3001 python -B -u scripts/evaluate_run.py runs/g4-prime-1` → **exit 0, about 3 min**, eval **`29fb97d0c311a0eca3b1`**. Validation cut-off .5790 (validation FA 1.52%, miss 6.73%, AUC .9888; 18,000 pairs, 270 photos).
- One-off ladder (saved outputs only, no inference): `runs/verification/hallmark-ladder-2026-11-25/ladder.py`, outputs `ladder.md`, `ladder.json`. Paired source-photo bootstrap, 2,000 draws, seed 20261019, method of `scripts/architecture_decision.py`. Each model keeps its own frozen validation cut-off.
- Model optimization for deployment (user approved: ONNX first, then a measured INT8 test). Work in `runs/verification/onnx-2026-11-25/` (git-ignored): `export_probe.py`, `variants.py` (INT8 builds, timing, frozen evaluation with ONNX Runtime in place of PyTorch), `compare.py` (paired comparison with the recorded PyTorch evaluation `a1ab2faad8f27bca309e`). Container packages added: onnx 1.23.2, onnxruntime 1.32.0, onnxscript 0.7.2.
- `pipeline` branch, commits `da70753` (code) and `b96ca7c` (docs): ONNX Runtime engine (`engine='onnx'`, `CVDL_ENGINE`), torch-free bundle contract module, `python -m cvdl.pipeline.export_onnx`, `model.onnx` + `onnx.json` for the run 13/18/22 bundles, `deploy/Dockerfile` now ONNX (old image is `deploy/Dockerfile.torch`), `tests/test_pipeline_onnx.py` (6 tests OK). FULL level in the pipeline worktree: 425 tests, 3 failures, all also present on the unchanged `044c1ad` because the worktree has no `runs/` training records.
- `pipeline` merged into `app` (`73fa326`, pushed). New branch `report` = `app` + `yx` (merge `3c96319`; `yx` docs kept on conflicts). FULL level after the merge commit: all failures explained (uncommitted merge, no photo data in the worktree).
- Cloud deployment (user: use the team L4 VM instead of a Hugging Face Space, because HF Docker Spaces now need a paid plan). VM `gcp-aai3001` (`aai3001-510802`, `asia-northeast3-a`, `g2-standard-8`, Spot). Recovered admin access through the GCP console SSH (user); temporary passwordless `sudo` for `aai-team`, removed at the end.
  - Cleanup (user approved): VM-only checkpoints copied to `runs/vm-backup-2026-11-25/` first (117 files, SHA-256 match). Deleted `otf-cache` (50 GB), `ImageSets` (21 GB; the PC has all 1,000 RAISE NEF and 900 DIV2K), `.venv` and the `uv` cache (11.3 GB), `.vscode-server`. Disk 107 GB → 24 GB used.
  - Installed Podman 4.9.3. Image `cvdl-pipeline:report-fb1f6ad` (ONNX) built on the PC and loaded on the VM; Quadlet unit `deploy/gcp/cvdl-pipeline.container` (loopback port 8000, starts at boot). App built with `VITE_API_URL=/api` into `/srv/demo/app`.
  - Tailscale Funnel on 443: `/` = app, `/api` = API. User enabled Funnel and added `tag:aai3001` to the funnel `nodeAttrs`.
  - Checks on the public route: health ready (engine onnx); DIV2K 0829 clean 0.009 `not_detected`, stego 0.839 `message_detected` (same as local), about 4.3 s per request; browser verdict after 4.4 s, no console errors; reboot test: service and Funnel came back by themselves. [Runbook](../../active/deploy-gcp-funnel.md).
- README rewritten as the project report, in paper form (abstract, related work, dataset, method, protocol, results, system, scalability, limitations, 18 references checked online). The old README moved to `docs/development.md` (links fixed). New: `scripts/build_report_figures.py` (model-history chart from `docs/results/model-history-2026-11-25.json`; heatmap example from DIV2K test photo 0829 with a seeded k=1 contiguous 0.25 bpp message, ONNX engine: clean 0.009 not detected, stego 0.839 detected), and 3 UI screenshots of `app/` against the real ONNX container. FAST level: **697 tests OK (35 skipped)**.

## Found

| Model | Change | Std P_E % | Std AUC | ALASKA seen P_E % | ALASKA unseen P_E % | LSDIR P_E % | Hard k=1 low AUC | Cut-off origin |
|---|---|---:|---:|---:|---:|---:|---:|---|
| V3c baseline | KV XuNet + GroupNorm, RAISE/DIV2K, 512 tiles, 10 ep | 4.05 | 0.9900 | 30.30 | 26.88 | 14.79 | 0.627 | post_hoc_256 `8328` |
| Wide-R2 | wider XuNet (x4), no norm, k1-4, 30 ep | 3.78 | 0.9909 | 32.65 | 31.08 | 14.02 | 0.616 | post_hoc_256 `8328` |
| newdata-1 | + ALASKA and LSDIR training photos | 5.13 | 0.9878 | 24.29 | 21.21 | 12.64 | 0.667 | post_hoc_256 `60da` |
| overlap-512 | same recipe re-run (run-to-run spread) | 5.61 | 0.9869 | 23.94 | 21.43 | 12.62 | 0.670 | post_hoc_256 `60da` |
| overlap-256 | 256 training tiles | 3.43 | 0.9939 | 21.21 | 18.75 | 11.27 | 0.714 | post_hoc_256 `60da` |
| SRM-30 | 30 fixed SRM filters replace KV | 3.07 | 0.9952 | 20.03 | 17.31 | 9.63 | 0.739 | post_hoc_256 `60da` |
| SRM-e50 | 50 epochs | 2.54 | 0.9964 | 19.42 | 16.30 | 9.39 | 0.745 | post_hoc_256 `60da` |
| R1 paired | paired clean/stego batches | 2.55 | 0.9964 | 18.99 | 15.58 | 8.84 | 0.757 | post_hoc_256 `60da` |
| Run 21 | + covariance pooling head | 2.15 | 0.9972 | 17.50 | 15.37 | 7.76 | 0.787 | post_hoc_256 `60da` |
| Run 22 (shipped) | + trained cell head (CE + 0.2 cell BCE), 15 ep from run 21 | 2.54 | 0.9971 | 17.62 | 14.87 | 7.59 | 0.798 | train_matched `60da` |

| Step | Δ standard P_E, points [95%] | Δ hard k=1 low AUC [95%] |
|---|---|---|
| V3c baseline → Wide-R2 | -0.26 [-0.54, -0.00] | -0.011 [-0.018, -0.003] |
| Wide-R2 → newdata-1 | +1.35 [+1.06, +1.63] | +0.051 [+0.043, +0.059] |
| newdata-1 → overlap-512 | +0.48 [+0.21, +0.73] | +0.003 [-0.002, +0.008] |
| overlap-512 → overlap-256 | -2.17 [-2.47, -1.87] | +0.045 [+0.038, +0.052] |
| overlap-256 → SRM-30 | -0.37 [-0.63, -0.11] | +0.025 [+0.018, +0.032] |
| SRM-30 → SRM-e50 | -0.52 [-0.72, -0.32] | +0.006 [+0.002, +0.010] |
| SRM-e50 → R1 paired | +0.00 [-0.16, +0.17] | +0.012 [+0.008, +0.017] |
| R1 paired → Run 21 | -0.40 [-0.65, -0.15] | +0.029 [+0.023, +0.034] |
| Run 21 → Run 22 (shipped) | +0.39 [+0.22, +0.58] | +0.012 [+0.009, +0.015] |
| **V3c baseline → Run 22 (shipped)** | **-1.50 [-1.90, -1.12]** | **+0.172 [+0.163, +0.180]** |

- Shipped model vs the course-style baseline: standard error 4.05% → 2.54%; hard k=1 low AUC .627 → .798; ALASKA unseen-camera error 26.9% → 14.9%; LSDIR error 14.8% → 7.6%.
- All older values reproduce the recorded ledger (wide-R2 3.78, SRM-e50 2.54, R1 2.55 / .757, run 21 2.15 / .787, run 22 2.54 / .798).
- The newdata-1 → overlap-512 re-run of the same recipe moved standard P_E by +0.48 points: a rough size of run-to-run spread. Steps smaller than this need care. Every run is one training seed; the intervals cover test photos, not training luck.
- Limits: the V3c and wide-R2 cut-offs come from list `8328` (RAISE/DIV2K, their training sources); later models use `60da`. V3c trained on k 1-8 at 512; the test uses k 1-4 at 256.

### Model optimization (run 22)

| Check | Result |
|---|---|
| ONNX FP32 export (opset 18, dynamic batch) | 2.67 MB; same graph and weights as the evaluated file (only debug name metadata differs) |
| Frozen evaluation, ONNX FP32 vs PyTorch | validation cut-off .5245214 vs .5245211; **0 decision changes** on 21,888 standard tiles (largest probability difference 1.6e-4); standard P_E 2.545% both; ALASKA seen/unseen, LSDIR P_E equal; hard k=1 low AUC .7983 both |
| HTTP, both containers, 3 photos (clean, LSB-changed, 12 MP) | same verdicts, logits equal to 4 decimals, same flagged cells |
| INT8 static (Conv+Gemm, SRM filter FP32, 512 validation tiles) | **rejected**: standard P_E 5.78% (+3.24 points [+2.76, +3.68]); hard AUC .742 (−.057 [−.062, −.051]); 1.4× faster on 2 threads, slower on 8 |

Model-only time per tile (batch 8): PyTorch 52.1 ms (2 threads); ONNX FP32 31.3 ms (2 threads), 9.2 ms (8 threads); INT8 (SRM kept FP32) 22.3 / 12.1 ms.

Full `/analyze` call, container images built from the same tree, `--cpus` = `CVDL_THREADS`, `include_images=false`, median of 3:

| CPUs | Photo | PyTorch image | ONNX image | Speed-up |
|---|---|---:|---:|---:|
| 2 | 2.8 MP (DIV2K 0801, 48 tiles) | 2.71 s | 1.54 s | 1.76× |
| 2 | 12 MP (4000×3000, 192 tiles) | 10.80 s | 6.21 s | 1.74× |
| 8 | 2.8 MP | 1.19 s | 0.53 s | 2.27× |
| 8 | 12 MP | 4.74 s | 2.11 s | 2.25× |

Image size 1.14 GB → 329 MB. Memory after the runs: PyTorch 198–246 MB, ONNX 578 MB (ONNX Runtime keeps a memory pool).

## Decided

- User: re-score only the V3c baseline.
- ~~User (Option A): host on one Hugging Face Docker Space that serves the backend and the React build; no Vercel.~~ Replaced: HF Docker Spaces now need a paid plan. User: host on the team L4 VM behind Tailscale Funnel.
- Ship ONNX FP32 as the deployed engine. Reject INT8: it fails the frozen tests. Do not resize photos for speed: resizing destroys the LSB signal.

## Next

- Pending (user): push `pipeline` (`da70753`, `b96ca7c`).
- Pending (user): may `pipeline` merge into `app` (Babydage's branch)? Who owns the Hugging Face account?
- Pending (user): switch the VM from Spot to standard provisioning for the demo period (VM stop, change, start; the service restarts by itself).
- Pending: repo cleanup; merge to `main`.
