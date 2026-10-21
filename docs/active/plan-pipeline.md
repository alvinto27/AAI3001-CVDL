# Web-app pipeline plan (run 13 + photo rule c)

**Status: approved by the user 2026-10-21.** Decision 1 (strict input policy, JPEG warning, tracked bundle, 64 MiB / 32 MP limits): yes. Decision 2: **API only, no demo page.** Branch `pipeline`. Written by Worker_2; copied unchanged below apart from this header. Build order: (1) core library + bundle + builder + parity tests; (2) HTTP API; (3) container + local launch + deployer README.

**Read-only proposal, 2026-10-21.** Branch `pipeline`, base `8e49ad8`. No code, inference, GPU job or benchmark run. Deliver three thin layers; the app developer builds the front end.

## 1. Core Python library: photo → plain result

Proposed `cvdl/pipeline/` interface:

- `load_pipeline(bundle_dir=None, *, device="cpu", batch_size=8)` loads once. Swap by folder, without code changes: argument/CLI `--bundle` > `CVDL_BUNDLE` environment variable > shipped default. HTTP uses the same startup setting, not per-request model uploads.
- `pipeline.analyze(photo, *, allow_rgba=False, include_images=False)` accepts a path, encoded bytes, PIL image or NumPy array; never modifies caller data.
- CLI: `python -m cvdl.pipeline photo.png --output result.json [--overlay overlay.png] [--device cuda]`.

JSON result: bundle/schema ID, dimensions/warnings, verdict (`message_detected`/`not_detected`), float64 rule logit, sigmoid(logit), cutoff; row-major tile probabilities/flags and `[x0,y0,x1,y1)` boxes; selected raw map (`trained_head`/`cam`) and boolean mask. For run13, maps are `[ceil(H/32),ceil(W/32)]` lists plus cell edges, covering partial edges—not H×W JSON arrays. Optional PNGs expand to H×W, as base64 or CLI file paths. Display images never feed decisions; sigmoid is **not a calibrated real-world probability**.

**No second copies:** move `photo_heatmap.py` infer/stitch/reduce/expand helpers into a shared lightweight module; make the old script import them. Move `evaluation_tensor` into a light module and re-export it from the evaluator (its current imports pull in training loaders/pandas). Reuse `lsb_train_grid.axes/crop`, `cvdl.cell_diagnostics.cell_maps`, and `cvdl.photo_verdict.features/predict`. Parameterize shared inference by bundle-selected map(s), preserving the old script's defaults. Preserve pixel-first overlap averaging, float64 reduction, diagonal connectivity and feature order. Only the bundle's fitted photo rule decides the verdict.

## 2. HTTP layer and container

**HTTP:** FastAPI, one model loaded at startup; inference runs off the event loop, with one active analysis and bounded admission (429 when busy). One server worker avoids duplicate model/memory allocations.

| Route | Contract |
| --- | --- |
| `POST /analyze` | Multipart upload `photo`; options `include_images` and `allow_rgba`. Returns the core JSON unchanged, including optional `heatmap_png_base64` / `overlay_png_base64` with `image/png` type. The front end can display/download these without a stored-results service. |
| `GET /health` | Ready status, bundle name/version/device; 503 if unavailable. Every analysis response also identifies its answering bundle. |

Enforce **64 MiB per request**, counting streamed bytes before multipart parsing; do not trust Content-Length/MIME/filename. Check signatures, dimensions and modes. JSON errors: 413 oversized, 415 unsupported, 422 corrupt/tiny, 429 busy, 503 unavailable; no tracebacks/server paths. Close temporary uploads; retain no photos/images after response. Local binding and restricted CORS by default; public hosting needs HTTPS/authentication/timeouts/rate limits.

**Container:** one `deploy/Dockerfile`, CPU by default, with code and frozen bundle inside; non-root process, one Uvicorn worker, no downloads/training dependencies at startup. After building/publishing the image, run `docker run --rm -p 127.0.0.1:8000:8000 cvdl-pipeline:cpu`. Plain local launch after dependency setup: `python -m cvdl.pipeline.serve --host 127.0.0.1 --port 8000`. Both expose the same routes. The container binds internally to 0.0.0.0.

Separate pinned `deploy/requirements-core.txt` (torch, numpy, Pillow) and `deploy/requirements-http.txt` (+ fastapi, uvicorn, python-multipart), installed in an isolated environment. **Do not edit `pyproject.toml`, `uv.lock` or team CUDA dependencies.** No Lightning/torchvision/pandas/YAML/rawpy runtime imports. Same Dockerfile supports tested GPU build choices: NVIDIA CUDA wheels/runtime + host driver/Container Toolkit (`--gpus all`); AMD ROCm wheels/runtime + compatible driver and `/dev/kfd`, `/dev/dri` permissions. Separate builds, not one universal GPU image.

## 3. Versioned, swappable bundle and builder

Track default `deploy/run13-rule-c-v1/{weights.pt,deploy.json}`: weights estimated **2–3 MiB** (full checkpoint 7.3 MiB; confirm export), JSON a few KiB. **Every model-specific choice lives here:** format version, bundle name/version; architecture fields; RGB8 preprocessing; tile size/grid; map kind/cell size/score semantics; tile/cell cutoffs; photo-rule type, ordered features, scaling, coefficients/intercept and cutoff. Include weights SHA256 and checkpoint/evaluation/threshold/photo-fit identities. No run folders/training YAML at runtime.

Default: `xunet_wide`, width4, SRM, norm none, clip3, changed-cells head; tile256/overlap-last/cell32; trained-head logits; exact tile **.5116998553**, cell **.3110628426**, rule-(c) logit **.16645031654738587**. Copy rule values without rounding from frozen rules `a4cff9e6…f2b3eb4`, checkpoint `4dc9ab8e…dd78b9`.

Builder: `python scripts/build_deploy_bundle.py --run RUN --evaluation-id ID --photo-rule FIT --output BUNDLE`. Export network tensors only, strip `net.` prefixes, verify strict reload. A new model first needs **its own** validation tile threshold from `evaluate_run.py`, validation cell cutoff for its selected map, and photo cache + fit/calibration tied to those exact weights/cutoffs/map. Generalize existing cache/fit entry points, not duplicate them. Previous full run-13 cache took ~25 min; new-model time varies. Fit on validation only; never borrow another model's thresholds/rule or fit on TEST.

**Headless models (e.g. SRM-e50):** select `cam`, the existing tile-head class-1 minus class-0 CAM margin, with its own cell cutoff and newly fitted rule on CAM-derived cell features. Never apply a trained-head rule to CAM.

Strict load: supported schema/architecture/grid/map/rule; matching checkpoint/evaluation/fit identities, preprocessing/cutoffs; tensor hash, feature order, finite values/positive scales. Allowlisted `model_from_hparams`, `torch.load(weights_only=True, map_location="cpu")`, strict keys/shapes/map capability, `eval()`. No substitutes/refits. Supported bundles swap without code; arbitrary architectures need implementation. Hashes establish integrity, not authenticity.

## 4. Preserve pixels, not appearance

Training input is **H×W×3 uint8 RGB**. Convert tiles to CHW float32, still **0–255**: no divide-by-255, resize, enhancement, ICC/gamma conversion or augmentation.

| Input | Proposed v1 behavior |
| --- | --- |
| RGB8 PNG / PIL RGB / array | Preserve samples. Arrays must be HWC uint8 RGB; never guess BGR. |
| RGBA8 | Reject by default; explicit opt-in drops alpha without compositing/changing RGB, and warns alpha payloads are unexamined. Reject premultiplied alpha. |
| Greyscale / palette / 16-bit | Reject, not replicate/expand/downcast. Check encoded bit depth before decoding: multichannel RGB16 can silently become RGB8. Already-decoded inputs cannot prove original format; caller owns that guarantee. |
| EXIF orientation | **Never rotate, mirror or EXIF-transpose.** Report orientation; boxes/maps use stored pixels, not the app's display orientation. |
| JPEG | Accept decoded RGB with warning: spatial RGB-LSB detection only, not JPEG/DCT steganography; recompression can destroy earlier LSB payloads. Reject CMYK conversion. |
| Other formats | Existing strict single-frame, uncompressed RGB8 TIFF with orientation 1 only; reject other TIFF orientations before decoding to prevent automatic rotation. RAW, animation/multiframe, HEIC/WebP/GIF are outside v1. |

Reject either side <256: no padding/upsampling on unvalidated input. Proposed decoded cap **32 MP**; check headers/decompression limits before allocation, never shrink to fit. Decode once, batch 8 tiles, zero loader workers, one photo at a time. Shared float64 pixel stitching uses hundreds of MiB at the cap; verify <2 GiB extra RAM. Explicit higher limits require memory budgeting.

CPU default; explicit `cuda` supports CUDA/ROCm and errors if unavailable; optional `auto` selects GPU. Inference mode, float32 CNN, float64 aggregation/rule, no mixed precision. Earlier RX 9070 XT cache achieved ~579 tiles/s including embedding/decode/stitch, **not app latency**. Unmeasured warm planning estimates for 1 MP / 12 MP: GPU ~0.2–1 s / ~1–5 s; CPU ~1–10 s / ~10–60 s. Benchmark decode/infer/stitch and cold load separately; no speed promise.

## 5. Verification and handoff

- Parity against existing helpers and saved features on a few **validation** photos, clean + fixed stego; include off-32-pixel tile origins/partial cells. Compare boxes, tile scores, raw maps, eight features, rule score and verdict. Predeclare tight numerical tolerances; require matching flags/verdict, investigate threshold edges. Never refit.
- Input-format/orientation/pixel-preservation tests, including encoded RGB16; malformed, tiny, oversized and multiframe inputs. Bundle corruption/mismatched-provenance tests, two-folder swap, headless CAM routing, CPU smoke, CLI/images and unavailable-device tests.
- HTTP upload/error/stream-limit/busy/health/image tests; container and plain-local smoke checks without run folders. Run required-level tests before commits; retain GPU/RAM guards.
- Ship `cvdl/pipeline/README.md`: installation/one-command launches, API/schema, formats/coordinates, limits and model identity. Heatmaps are evidence, not exact payload locations; negative verdict is not proof of no message. Record accepted rule's pooled FA **8.507%**, hard P_E **22.971%**: the practical mark failed and the benchmark was not wholly untouched.

## 6. Few user decisions

1. Accept the strict input policy, JPEG warning, tracked small bundle, 64 MiB request / 32 MP decoded limits?
2. API-only handoff, or also a minimal upload/results demo page? Front-end implementation otherwise stays with the app developer.

Docs checked: Pillow modes/orientation (`/python-pillow/pillow`); FastAPI UploadFile/multipart/threadpool (`/websites/fastapi_tiangolo`), via Context7. No dependency installation was performed.
