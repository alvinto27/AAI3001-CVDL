# Landing preview videos

Two 10-second loops for the "Inspection preview" panels on the landing page, recorded from the real app with real `run18-rule-c` inference. Both clips replay one shot table, so they stay in step when played side by side.

| Clip | Shows |
| --- | --- |
| `out/inspection-original.mp4` | The inspector in **Original** mode: hover, zoom into the embedded region, zoom back out. |
| `out/inspection-heatmap.mp4` | The same moves in **Detection Overlay** mode with region R1 selected. |

`encode.mjs` writes them to `out/` (git-ignored); the shipped copies and their first-frame posters live in `app/public/preview/`. Both are H.264 High, 1280 × 720, 30 fps, 300 frames, BT.709, no audio, a keyframe every second. The last frame matches the first, so they loop without a jump.

## Sample and evidence

`samples/kodim23-clean.png` is Kodak test image 23 (lossless PNG, 768 × 512, from r0k.us/graphics/kodak, sha256 `e3111a2f…5d2d5baf`). `make_samples.py` writes `samples/kodim23-stego.png` by embedding a random payload in the box `[416, 96, 640, 320)` with the repository's own `lsb_engine.embed_fast` (k = 3, replace, random placement, 3 bpp, seed 7; see `samples/kodim23-embed.json`).

The `*.response.json` files are the real `/analyze` responses (images omitted):

| File | Verdict | Score | Flagged tiles |
| --- | --- | --- | --- |
| clean | `not_detected` | 0.090 | 0 of 6 |
| stego | `message_detected` | 0.979 | 3 of 6 |

In the stego heatmap, 48 of the 49 cells inside the embedded box are flagged (mean cell value 4.27 inside, 0.10 outside). The clean photo still has 97 flagged cells scattered over the frame, so the heatmap clip shows that background texture as well.

Lighter payloads (k = 1 at 0.5 or 3 bpp, k = 2 at 1 bpp) were not detected. Of the 24 clean Kodak photos, 15 were reported as `message_detected`: these film scans are outside run 18's training sources, and kodim23 was chosen because it reads clean.

## Rebuild

Prerequisites: the API on `127.0.0.1:8000` (see `cvdl/pipeline/README.md`), `npm run dev` on `127.0.0.1:3000`, EffectCraft, and an FFmpeg build with libx264.

1. `.venv/Scripts/python app/preview-video/make_samples.py path/to/kodim23.png`
2. `cd app && node preview-video/record.mjs`. This captures `build/<clip>/f####.png`, `build/cursor.json` and `build/cursor.png`. It fails if the stage resizes, if the cover zoom misses, or if the two clips disagree on the pointer path. It reports the loop drift.
3. Open `inspection-preview.ecproj` in EffectCraft. Each `Preview <clip>` comp stacks the frame sequence and the `cursor.svg` layer, with one position keyframe per frame from `cursor.json`. Render both comps as PNG sequences to `build/composited/<clip>/c####.png`.
4. `FFMPEG=path/to/ffmpeg node preview-video/encode.mjs`, then copy `out/*.mp4` to `app/public/preview/` and refresh the `.jpg` posters from frame 0.
5. `node preview-video/verify-landing.mjs` plays both MP4s inside the live landing panels (Edge, desktop/tablet/mobile) and writes screenshots to `build/landing/`.

`build/` is regenerated and git-ignored (about 0.9 GB of PNG frames).
