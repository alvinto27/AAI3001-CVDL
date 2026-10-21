# Photo analysis API

The API tiles an original photo, runs **run 13 with photo rule (c)**, and returns a verdict and heatmap. It does not extract messages. No front end is included.

## Quick start: container

Run from the repository root. Allow about **4 GiB RAM**. The CPU image contains the model and downloads nothing at startup.

| Step | Docker | Podman |
| --- | --- | --- |
| Build | `docker build -f deploy/Dockerfile -t cvdl-pipeline:cpu .` | `podman build -f deploy/Dockerfile -t cvdl-pipeline:cpu .` |
| Serve | `docker run --rm --name cvdl-pipeline --memory=4g -p 127.0.0.1:8000:8000 cvdl-pipeline:cpu` | `podman run --rm --name cvdl-pipeline --memory=4g -p 127.0.0.1:8000:8000 cvdl-pipeline:cpu` |
| Stop | `docker stop cvdl-pipeline` | `podman stop cvdl-pipeline` |

Inside: `0.0.0.0:8000`, non-root user 10001, one worker/analysis at a time. These commands publish only on host loopback. `.dockerignore` excludes datasets, runs and unrelated training files.

### Without a container

Use Python 3.12 with `venv` support (`python3.12-venv` on Debian/Ubuntu). Create a **separate** environment:

```sh
python3.12 -m venv "$HOME/.venvs/cvdl-pipeline"
. "$HOME/.venvs/cvdl-pipeline/bin/activate"
python -m pip install torch==2.14.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r deploy/requirements-http.txt
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
python -m cvdl.pipeline.serve --host 127.0.0.1 --port 8000
```

No Lightning/pandas/YAML/torchvision/RAW decoders are needed. Leave training dependencies, `pyproject.toml` and `uv.lock` unchanged.

### GPU option

**Passing GPU devices does not turn the CPU image into a GPU image.** A compatible wheel/runtime and host driver are required. Build with `--build-arg TORCH_INDEX_URL=...` for the matching torch 2.14.0 index. GPU builds are untested here.

For a prepared NVIDIA image tagged `cvdl-pipeline:gpu`:

| Docker (NVIDIA Container Toolkit) | Podman (NVIDIA CDI configured on the host) |
| --- | --- |
| `docker run --rm --gpus all -p 127.0.0.1:8000:8000 cvdl-pipeline:gpu --device cuda` | `podman run --rm --device nvidia.com/gpu=all -p 127.0.0.1:8000:8000 cvdl-pipeline:gpu --device cuda` |

AMD needs ROCm, matching drivers and `/dev/kfd`/`/dev/dri` permissions, not NVIDIA flags. `cuda` fails if unavailable; `auto` can fall back to CPU.

## HTTP API

```sh
curl http://127.0.0.1:8000/health
curl -F 'photo=@photo.png' -F 'include_images=true' \
  http://127.0.0.1:8000/analyze > result.json
```

| Route / form field | Meaning |
| --- | --- |
| `GET /health` | `ready`, bundle identity and device; HTTP 503 when not ready. |
| `POST /analyze` | Multipart upload. `photo` must be one file, not a server path. |
| `include_images` | `true` returns base64 PNG heatmap and overlay; default `false`. |
| `allow_rgba` | Explicitly discard alpha without changing RGB; default `false`. |

A successful response includes:

| Field | Meaning |
| --- | --- |
| `bundle` | Name, version, format version and weights hash. Also present in health/error responses; `null` if startup failed before a bundle loaded. |
| `photo_verdict` | `message_detected` or `not_detected`. |
| `photo_rule` | Raw `logit`, its 0–1 `probability` score, and **logit** `cutoff`. Use the supplied verdict; this is not a calibrated real-world chance. |
| `schema_version`, `device` | Response contract version and execution device. |
| `width`, `height`, `input` | Stored dimensions, format, orientation and warnings. |
| `tiles` | Row-major tile `box`, stego `probability`, and `flagged` value. |
| `heatmap` | Raw `values`, boolean `flagged` mask, map kind/score, cutoff, cell size, shape and x/y edges. |
| `features`, `counts` | The eight photo-rule inputs and summary counts. |
| `images` | Optional `image/png` type, display scale, `heatmap_png_base64` and `overlay_png_base64`. Decode to display/save PNGs; no download store. |

Example fields from a validation-photo response (other fields omitted):

```json
{
  "photo_verdict": "not_detected",
  "photo_rule": {
    "type": "logistic_v1",
    "logit": -2.735547244773273,
    "probability": 0.060908095934515474,
    "cutoff": 0.16645031654738587
  },
  "width": 2040,
  "height": 1152
}
```

| Error | Action |
| --- | --- |
| 413 | Request exceeds 64 MiB, or decoded photo exceeds 32 MP. |
| 415 | Unsupported format/pixel representation, or not a multipart upload. |
| 422 | Corrupt/tiny photo, missing file, or invalid form options. |
| 429 | Another upload/analysis is active. Retry later; there is no analysis queue. |
| 503 | Model not ready or analysis unavailable. Check `/health` and server configuration. |

Errors have an `error` object and `bundle`, without server paths or tracebacks. Upload files and generated responses are not retained after the response.

## Inputs and coordinates

Prefer the original RGB8 PNG, not a screenshot or resized/recompressed copy. Accepted input is **8-bit RGB**, with both dimensions at least 256. There is no scaling, colour correction or EXIF rotation.

- PNG: RGB8 without colour-key transparency, or RGBA8 with explicit opt-in. Alpha-channel payloads are not examined.
- JPEG: decoded RGB is accepted with a warning. Compression can destroy earlier RGB-LSB changes. JPEG/DCT steganography is not examined; CMYK JPEG is rejected.
- TIFF: one uncompressed RGB8 frame with orientation 1 only.
- Reject grey, palette, 16-bit, premultiplied alpha, RAW, animations and other formats. No implicit conversion or padding.
- Limits: **64 MiB for the whole streamed multipart request**, including form overhead; **32 million decoded pixels**.

Tile boxes are `[x0,y0,x1,y1)`: top-left included, bottom/right excluded. The last tile moves to the edge and can overlap. Heatmaps use **32×32 px cells**, shape `[ceil(height/32), ceil(width/32)]`, with smaller edge cells. Pixel overlaps are averaged first. Head values are unbounded scores (logits); CAM is message-class minus clean-class score, not pixel probability.

Use **stored-pixel orientation** for photo and overlay: browsers may otherwise rotate only the original from EXIF. Heatmaps are evidence, not exact payload locations.

## Swap or build a bundle

Selection: `--bundle DIR` > `CVDL_BUNDLE` > shipped default. Restart to swap supported bundles without app code changes. Do not hand-edit weights/rules/cutoffs: identity checks reject mismatches.

| Mounted bundle | Docker | Podman with SELinux |
| --- | --- | --- |
| Serve | `docker run --rm -p 127.0.0.1:8000:8000 -v "$PWD/my-bundle:/bundle:ro" -e CVDL_BUNDLE=/bundle cvdl-pipeline:cpu` | `podman run --rm -p 127.0.0.1:8000:8000 -v "$PWD/my-bundle:/bundle:ro,Z" -e CVDL_BUNDLE=/bundle cvdl-pipeline:cpu` |

User 10001 must be able to read/traverse the bundle. SELinux `:Z` gives a private label: mount a dedicated directory, not your home/repository. `:z` is shared. Appending `--bundle /bundle` after the image overrides the environment; host/port defaults remain.

Build **offline in the project environment**. First obtain the model's own full validation threshold from `evaluate_run.py`, map cell cutoff and evaluation ID. Never borrow another model's thresholds/rule. Set `RUN`, `EVAL`, `CELLS`, `FIT`, `MAP`, then:

```sh
python scripts/photo_verdict_cache.py --run "$RUN" --evaluation-id "$EVAL" \
  --cell-summary "$CELLS" --map-kind "$MAP" --output "$FIT" --validation-only
python scripts/photo_verdict_fit.py fit --output "$FIT"
python scripts/build_deploy_bundle.py --run "$RUN" --evaluation-id "$EVAL" \
  --cells-summary "$CELLS" --map-kind "$MAP" \
  --photo-rule "$FIT/frozen-rules.json" --output deploy/my-bundle
```

Headless models use `MAP=cam` with their **own CAM cutoff and fit**; otherwise `MAP=trained_head`. Cache defaults to GPU; `--device cpu` is available. Original full cache: ~25 min, hardware-dependent. Unsupported architectures/grids need implementation.

## Host settings and limits

Use `--allow-origin http://localhost:3000` for the app's exact origin; repeat for more origins, or set comma-separated `CVDL_ALLOWED_ORIGINS`. Wildcards are refused. CORS is not authentication. For public hosting, add HTTPS, authentication, upload timeouts and rate limits at a reverse proxy; expose non-loopback ports only deliberately. Local Python defaults to `127.0.0.1`; `--host` changes it.

On the measured TEST pool, the accepted rule falsely flagged **8.5% of clean photos**, and about **17% on new ALASKA cameras**. These are measured panel averages, not guarantees for another camera/app. The earlier practical false-alarm mark failed, and this was not a wholly untouched benchmark. **“Not detected” is not proof that no message exists.** This model examines spatial RGB LSB patterns only, not general image authenticity or every form of steganography.
