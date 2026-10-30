// Records the landing preview clips from the running app and the real inference API.
// Each clip is captured frame by frame at 30 fps: the pointer is moved and the wheel is
// turned for that frame, the page renders, and the inspector stage is screenshotted.
// Both clips replay the same shot table, so they stay in step when played side by side.
//
//   node preview-video/record.mjs [--app http://127.0.0.1:3000] [--image preview-video/samples/kodim23-stego.png]
import { chromium } from '@playwright/test';
import { mkdir, readFile, rm, writeFile } from 'node:fs/promises';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { parseArgs } from 'node:util';

const HERE = dirname(fileURLToPath(import.meta.url));
const { values: args } = parseArgs({
  options: {
    app: { type: 'string', default: 'http://127.0.0.1:3000' },
    image: { type: 'string', default: join(HERE, 'samples/kodim23-stego.png') },
    out: { type: 'string', default: join(HERE, 'build') },
  },
});

const FPS = 30;
const FRAMES = 10 * FPS;
// At this width the stage is 2/3 of the viewport wide and 132 px shorter than it: 960 × 540,
// wide enough that the readout bar under it stays on one line and never resizes the stage.
const VIEWPORT = { width: 1440, height: 672 };
const STAGE = { w: 960, h: 540 };
const SCALE = 4 / 3; // stage CSS px → 1280 × 720 video px
// usePanZoom: factor = exp(-deltaY * 0.0018). Chromium divides injected wheel deltas by the device
// scale factor, so the gain per injected unit is smaller by the same ratio.
const WHEEL_GAIN = 0.0018 / SCALE;
const IMAGE_W = 768;
const COVER = STAGE.w / IMAGE_W;

const CLIPS = [
  { name: 'original', mode: 'Original' },
  { name: 'heatmap', mode: 'Detection Overlay' },
];

// Pointer path in image px. Z is the centre of the embedded region (samples/kodim23-embed.json).
const REST = [250, 360];
const Z = [528, 208];
const MOVES = [
  [0.6, 2.2, REST, [436, 128]],
  [2.2, 3.4, [436, 128], [604, 262]],
  [3.4, 3.8, [604, 262], Z],
  [5.0, 5.7, Z, [470, 160]],
  [5.7, 6.3, [470, 160], [590, 250]],
  [6.3, 6.7, [590, 250], Z],
  [7.9, 9.6, Z, REST],
];
// [start, end, zoom factor]; each zoom is anchored on the pointer resting at Z.
const ZOOMS = [
  [3.8, 5.0, 2.3],
  [6.7, 7.9, 1 / 2.3],
];

const ease = (u) => (u < 0.5 ? 4 * u ** 3 : 1 - (-2 * u + 2) ** 3 / 2);
const progress = (t, t0, t1) => ease(Math.min(1, Math.max(0, (t - t0) / (t1 - t0))));

function pointerAt(t) {
  let p = REST;
  for (const [t0, t1, a, b] of MOVES) {
    if (t < t0) break;
    const u = progress(t, t0, t1);
    p = [a[0] + (b[0] - a[0]) * u, a[1] + (b[1] - a[1]) * u];
  }
  return p;
}

const logZoomAt = (t) => ZOOMS.reduce((sum, [t0, t1, f]) => sum + Math.log(f) * progress(t, t0, t1), 0);

/** One entry per frame: pointer in image px and the wheel delta to apply on that frame. */
const timeline = Array.from({ length: FRAMES }, (_, i) => {
  const t = i / FPS;
  const prev = i === 0 ? 0 : logZoomAt((i - 1) / FPS);
  return { pointer: pointerAt(t), wheel: -(logZoomAt(t) - prev) / WHEEL_GAIN };
});

const nextPaint = (page) => page.evaluate(() => new Promise((r) => requestAnimationFrame(() => requestAnimationFrame(r))));

async function record(browser, clip) {
  const page = await browser.newPage({ viewport: VIEWPORT, deviceScaleFactor: SCALE });
  await page.goto(args.app);
  await page.locator('input[type=file]').setInputFiles(resolve(args.image));
  await page.getByRole('button', { name: 'Analyse Image' }).click();
  await page.getByRole('button', { name: 'Analyse Another Image' }).waitFor({ timeout: 120_000 });
  const verdict = (await page.locator('h2').filter({ hasText: /DETECTED|HIDDEN|INCONCLUSIVE/ }).first().innerText()).trim();

  // Select the top region (outlined in the overlay), then return to the fitted view.
  await page.getByRole('button', { name: /^R1 / }).click();
  await page.getByRole('radio', { name: clip.mode }).click();
  await page.getByTitle('Fit to window').click();

  const stage = page.getByRole('img', { name: /^Uploaded image/ });
  const box = await stage.boundingBox();
  const boxKey = JSON.stringify(box);
  if (Math.round(box.width) !== STAGE.w || Math.round(box.height) !== STAGE.h) throw new Error(`stage is ${box.width}×${box.height}, expected ${STAGE.w}×${STAGE.h}`);
  const layer = stage.locator('> div').first();
  const readView = async () => {
    const m = (await layer.evaluate((el) => el.style.transform)).match(/translate\(([-\d.e]+)px, ([-\d.e]+)px\) scale\(([-\d.e]+)\)/);
    return { x: +m[1], y: +m[2], s: +m[3] };
  };
  const moveStage = ([x, y]) => page.mouse.move(box.x + x, box.y + y);

  // Zoom from the fitted view until the image covers the stage width, anchored at the centre.
  await moveStage([STAGE.w / 2, STAGE.h / 2]);
  for (let i = 0; i < 8; i++) {
    const { s } = await readView();
    if (Math.abs(s - COVER) < 1e-6) break;
    await page.mouse.wheel(0, -Math.log(COVER / s) / WHEEL_GAIN);
    await nextPaint(page);
  }
  const start = await readView();
  if (Math.abs(start.s - COVER) > 1e-4) throw new Error(`cover scale is ${start.s}, expected ${COVER}`);

  const dir = join(args.out, clip.name);
  await rm(dir, { recursive: true, force: true });
  await mkdir(dir, { recursive: true });
  let view = start;
  const views = [];
  const pointers = [];
  for (const [i, f] of timeline.entries()) {
    // Map through the view before this frame's wheel: the zoom is anchored under the pointer.
    const at = [view.x + f.pointer[0] * view.s, view.y + f.pointer[1] * view.s];
    pointers.push(at);
    await moveStage(at);
    if (Math.abs(f.wheel) > 1e-9) await page.mouse.wheel(0, f.wheel);
    await nextPaint(page);
    view = await readView();
    views.push(view);
    await page.screenshot({ path: join(dir, `f${String(i).padStart(4, '0')}.png`), clip: box, animations: 'disabled' });
  }
  if (JSON.stringify(await stage.boundingBox()) !== boxKey) throw new Error('the stage moved or resized during recording');
  await page.close();
  const end = views.at(-1);
  const drift = Math.max(Math.abs(end.x - start.x), Math.abs(end.y - start.y)) + Math.abs(end.s - start.s) * IMAGE_W;
  return { verdict, start, peakScale: Math.max(...views.map((v) => v.s)), driftPx: +drift.toFixed(4), pointers };
}

const browser = await chromium.launch();
const results = {};
for (const clip of CLIPS) results[clip.name] = await record(browser, clip);

// The cursor is composited later (EffectCraft); rasterise it at video scale with its tip at (2, 2).
const cursorPage = await browser.newPage({ viewport: { width: 26, height: 38 } });
await cursorPage.setContent(`<style>html,body{margin:0;background:transparent}</style>${await readFile(join(HERE, 'cursor.svg'), 'utf8')}`);
await cursorPage.screenshot({ path: join(args.out, 'cursor.png'), omitBackground: true });
await browser.close();

const [a, b] = [results.original.pointers, results.heatmap.pointers];
const apart = Math.max(...a.map((p, i) => Math.hypot(p[0] - b[i][0], p[1] - b[i][1])));
if (apart > 0.05) throw new Error(`clips disagree on the pointer path by ${apart} px`);

await writeFile(join(args.out, 'cursor.json'), JSON.stringify({
  fps: FPS,
  size: [STAGE.w * SCALE, STAGE.h * SCALE],
  hotspot: 'arrow tip',
  points: a.map((p) => p.map((v) => +(v * SCALE).toFixed(2))),
}) + '\n');
for (const r of Object.values(results)) delete r.pointers;
console.log(JSON.stringify(results));
