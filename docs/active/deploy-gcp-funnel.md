# Cloud deployment: GCP VM + L4 GPU + Tailscale Funnel

Status: **live** since 2026-11-25 at <https://gcp-aai3001.taild62d21.ts.net> when the VM is running. The service runs on the L4 GPU; the ONNX CPU service is the fallback. Record: [2026-11-25 journal](../journal/2026-11-25/README.md).

## Layout

| Part | Value |
| --- | --- |
| VM | `gcp-aai3001`, project `aai3001-510802`, zone `asia-northeast3-a`, `g2-standard-8` (8 vCPU = 4 cores at 2.2 GHz, 31 GB RAM, 1 NVIDIA L4 24 GB), Ubuntu 24.04, driver 595.91.07 (CUDA 13.2) |
| GPU access | NVIDIA Container Toolkit 1.20.2 with a Podman-4.9-compatible CDI spec from [`deploy/gcp/make-cdi-spec.sh`](../../deploy/gcp/make-cdi-spec.sh) (rewrites cdiVersion 0.7.0 to 0.6.0 and masks `nvidia-cdi-refresh`) |
| GPU service | Quadlet unit [`deploy/gcp/cvdl-pipeline.container`](../../deploy/gcp/cvdl-pipeline.container): image `localhost/cvdl-pipeline:gpu-5970a3b` (5.88 GB, `deploy/Dockerfile.torch` with torch 2.14.0 cu130), `--engine torch --device cuda`. Starts at boot |
| CPU fallback | Quadlet unit [`deploy/gcp/cvdl-pipeline-cpu.container`](../../deploy/gcp/cvdl-pipeline-cpu.container): image `localhost/cvdl-pipeline:report-fb1f6ad` (329 MB, `deploy/Dockerfile`, ONNX Runtime). Not started at boot |
| Port | Both units publish `127.0.0.1:8000`. Each unit has `Conflicts=` and `After=` on the other, so starting one stops the other first |
| Web app | `app/` built with `VITE_API_URL=/api`, copied to `/srv/demo/app` (root-owned, world-readable) |
| Public entry | Tailscale Funnel on port 443 (`--bg`, kept across reboots): `/` serves `/srv/demo/app`; `/api` proxies to `http://127.0.0.1:8000` with the `/api` prefix removed |
| Tailnet policy | `nodeAttrs` funnel entry includes `tag:aai3001` (the VM is a tagged device). A grant lets `autogroup:member` reach `tag:aai3001` on port 443, so tailnet devices can open the site too |

Same origin for the app and the API, so the browser needs no CORS. The API still allows `https://gcp-aai3001.taild62d21.ts.net` explicitly.

## Checks (2026-11-25)

### CPU service (first deployment)

- Public route: `/api/health` ready (engine `onnx`); `/` HTTP 200. DIV2K 0829 clean 0.009 `not_detected`, stego 0.839 `message_detected`, the same as the local run; about 4.3 s per request. Browser: verdict after 4.4 s, no console errors. Reboot: service and Funnel came back by themselves.

### GPU check (2026-11-25)

25 test-split photos (DIV2K 10, LSDIR 6, ALASKA 6, RAISE 3), each clean and with one k=1 message (seed 20261125), sent over loopback to the GPU service and the ONNX CPU service. Scripts and result: `runs/verification/gpu-l4-2026-11-25/` (git-ignored; the raw JSON responses were lost in the reboot test, the printed result is kept).

| Measure | Result |
| --- | --- |
| Changed photo verdicts | 0 of 50 |
| Changed flagged tiles | 0 of 2,796 |
| Changed flagged cells | 7 of 168,486 |
| Largest difference: tile probability / cell logit / photo logit | 1.6e-3 / 7.8e-3 / 2.0e-2 |
| Median server time ≤1.2 MP, 2–3 MP, 16 MP: CPU → GPU | 0.53 → 0.08 s; 2.10 → 0.27 s; 10.16 → 1.67 s |

Public route after the switch: 3.7 MB DIV2K photo 2.89 s total, 23.8 MB RAISE photo 4.50 s total (the upload through the Funnel relay is most of the time). Browser: header "run22-rule-c · CUDA", verdict after 1.8 s, no console errors. Reboot: GPU service and Funnel came back by themselves; CDI spec stayed at 0.6.0. Fallback switch in both directions: no errors.

## Operate

Commands need `sudo` on the VM. The `aai-team` account has no `sudo`. Use the GCP console **SSH** button (it gives a user with `sudo`) or the `aai-admin` account.

| Task | Command (on the VM) |
| --- | --- |
| State and logs | `systemctl status cvdl-pipeline`; `journalctl -u cvdl-pipeline -n 50`; `nvidia-smi` |
| Engine in use | `curl -s 127.0.0.1:8000/health` (`"device":"cuda"` or `"engine":"onnx"`) |
| Switch to the CPU fallback | `sudo systemctl start cvdl-pipeline-cpu` |
| Switch back to the GPU | `sudo systemctl start cvdl-pipeline` |
| Funnel state / stop public access | `sudo tailscale funnel status`; `sudo tailscale funnel --https=443 off` |
| After an NVIDIA driver update | `sudo sh deploy/gcp/make-cdi-spec.sh`, then `sudo systemctl restart cvdl-pipeline` |

### Replace the model or the app

1. GPU image, on the VM: `git archive <commit> | tar -x -C /srv/demo/build`, then `sudo podman build -f deploy/Dockerfile.torch --build-arg TORCH_INDEX_URL=https://download.pytorch.org/whl/cu130 -t localhost/cvdl-pipeline:gpu-<commit> .` CPU image, on the PC: `podman build -f deploy/Dockerfile -t cvdl-pipeline:<tag> .` and `podman save cvdl-pipeline:<tag> | ssh gcp-aai3001 'sudo podman load'`.
2. Change `Image=` in the unit file, then `sudo systemctl daemon-reload && sudo systemctl restart cvdl-pipeline`.
3. For the app: `cd app && VITE_API_URL=/api npm run build`, then copy `app/dist/` to `/srv/demo/app/` (keep root ownership).

## Limits

- The site is up only while the VM runs. A Spot VM can stop at any time; switch to standard provisioning for the demo period (user action in the GCP console).
- One analysis at a time (the service queues the rest).
- The VM bills for the L4 every hour it runs.
