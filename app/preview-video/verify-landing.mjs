// Plays the MP4s inside the live landing preview panels without changing the app source: the two
// illustration SVGs are swapped for <video> elements in the page, then playback is checked and
// screenshotted. Uses Edge because Playwright's bundled Chromium has no H.264 decoder.
//   node preview-video/verify-landing.mjs [--app http://127.0.0.1:3000] [--shots DIR]
import { chromium } from '@playwright/test';
import { mkdir } from 'node:fs/promises';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { parseArgs } from 'node:util';

const HERE = dirname(fileURLToPath(import.meta.url));
const { values: args } = parseArgs({
  options: {
    app: { type: 'string', default: 'http://127.0.0.1:3000' },
    shots: { type: 'string', default: join(HERE, 'build/landing') },
  },
});
await mkdir(args.shots, { recursive: true });

const browser = await chromium.launch({ channel: 'msedge' });
const report = {};
for (const [name, viewport] of [['desktop', { width: 1920, height: 1080 }], ['tablet', { width: 820, height: 1180 }], ['mobile', { width: 390, height: 844 }]]) {
  const page = await browser.newPage({ viewport });
  await page.goto(args.app);
  const panels = await page.evaluate(async () => {
    const srcs = ['/preview-video/out/inspection-original.mp4', '/preview-video/out/inspection-heatmap.mp4'];
    const svgs = [...document.querySelectorAll('figure svg')];
    const videos = svgs.map((svg, i) => {
      const v = document.createElement('video');
      Object.assign(v, { src: srcs[i], muted: true, loop: true, playsInline: true, preload: 'auto' });
      v.className = 'absolute inset-0 size-full object-cover';
      svg.replaceWith(v);
      return v;
    });
    await Promise.all(videos.map((v) => new Promise((ok, fail) => {
      v.addEventListener('canplaythrough', ok, { once: true });
      v.addEventListener('error', () => fail(new Error(`${v.src}: media error ${v.error?.code}`)), { once: true });
    })));
    await Promise.all(videos.map((v) => v.play()));
    return videos.map((v) => {
      const r = v.getBoundingClientRect();
      return { src: v.src.split('/').pop(), duration: v.duration, size: [v.videoWidth, v.videoHeight], box: [Math.round(r.width), Math.round(r.height)] };
    });
  });
  const checks = [];
  for (const t of [1.0, 5.5, 9.9]) {
    const times = await page.evaluate(async (t) => {
      const vs = [...document.querySelectorAll('figure video')];
      vs.forEach((v) => v.pause());
      await Promise.all(vs.map((v) => new Promise((ok) => { v.addEventListener('seeked', ok, { once: true }); v.currentTime = t; })));
      return vs.map((v) => v.currentTime);
    }, t);
    await page.locator('figure').screenshot({ path: join(args.shots, `${name}-${t}.png`) });
    checks.push({ t, times });
  }
  const overflow = await page.evaluate(() => document.documentElement.scrollWidth > innerWidth);
  report[name] = { panels, checks, horizontalOverflow: overflow };
  if (name === 'desktop') await page.screenshot({ path: join(args.shots, 'desktop-full.png') });
  await page.close();
}
await browser.close();
console.log(JSON.stringify(report, null, 1));
