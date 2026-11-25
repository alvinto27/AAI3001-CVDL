# Cloud deployment: GCP VM + Tailscale Funnel

Status: **live** since 2026-11-25 at <https://gcp-aai3001.taild62d21.ts.net> when the VM is running. Record: [2026-11-25 journal](../journal/2026-11-25/README.md).

## Layout

| Part | Value |
| --- | --- |
| VM | `gcp-aai3001`, project `aai3001-510802`, zone `asia-northeast3-a`, `g2-standard-8` (8 vCPU, 31 GB RAM, 1 L4, not used) |
| Container | `localhost/cvdl-pipeline:report-fb1f6ad`, built from `deploy/Dockerfile` (ONNX Runtime, run 22 bundle), rootful Podman 4.9.3 |
| Service | Quadlet unit [`deploy/gcp/cvdl-pipeline.container`](../../deploy/gcp/cvdl-pipeline.container), installed as `/etc/containers/systemd/cvdl-pipeline.container`. Port `127.0.0.1:8000` only. Starts at boot, restarts on failure |
| Web app | `app/` built with `VITE_API_URL=/api`, copied to `/srv/demo/app` (root-owned, world-readable) |
| Public entry | Tailscale Funnel on port 443 (`--bg`, kept across reboots): `/` serves `/srv/demo/app`; `/api` proxies to `http://127.0.0.1:8000` with the `/api` prefix removed |
| Tailnet policy | `nodeAttrs` entry with `attr: ["funnel"]` includes `tag:aai3001` (the VM is a tagged device) |

Same origin for the app and the API, so the browser needs no CORS. The API still allows `https://gcp-aai3001.taild62d21.ts.net` explicitly.

## Checks done (2026-11-25)

- Public route (Funnel relay IP, not the tailnet IP): `/api/health` ready, engine `onnx`; `/` HTTP 200.
- `POST /api/analyze` with DIV2K 0829 clean and with a k=1 contiguous 0.25 bpp message: `not_detected` 0.009 and `message_detected` 0.839, the same as the local run; about 4.3 s per request including the 6 MB upload.
- Browser (Playwright, public route): verdict shown after 4.4 s, no console errors.
- VM reboot: service active and Funnel on again without manual steps.

## Operate

Commands need `sudo` on the VM. The `aai-team` account has no `sudo`. Use the GCP console **SSH** button (it gives a user with `sudo`) or the `aai-admin` account.

| Task | Command (on the VM) |
| --- | --- |
| Service state and logs | `systemctl status cvdl-pipeline`; `journalctl -u cvdl-pipeline -n 50` |
| Funnel state | `sudo tailscale funnel status` |
| Stop public access | `sudo tailscale funnel --https=443 off` |
| Stop the API | `sudo systemctl stop cvdl-pipeline` |

### Replace the model or the app

1. On the PC: `podman build -f deploy/Dockerfile -t cvdl-pipeline:<tag> .` and `podman save cvdl-pipeline:<tag> | ssh gcp-aai3001 'sudo podman load'`.
2. Change `Image=` in the unit file, then `sudo systemctl daemon-reload && sudo systemctl restart cvdl-pipeline`.
3. For the app: `cd app && VITE_API_URL=/api npm run build`, then copy `app/dist/` to `/srv/demo/app/` (keep root ownership).

## Limits

- The site is up only while the VM runs. A Spot VM can stop at any time; switch to standard provisioning for the demo period (user action in the GCP console).
- One analysis at a time (the service queues the rest). 8 vCPU: about 0.5 s model time for 2.8 MP and 2.1 s for 12 MP.
- The L4 GPU is not used by the ONNX image. The VM bills for it while it runs.
