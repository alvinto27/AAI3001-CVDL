import type { Analysis, HeatGrid } from '../types/analysis';
import { localScore } from './regions';

// The heatmap carries the interface's only saturated colour.
const RAMP: [number, number, number][] = [
  [92, 56, 168],
  [208, 62, 118],
  [240, 142, 44],
  [255, 236, 150],
];

export function heatColour(score: number): [number, number, number, number] {
  const t = Math.min(0.999, score) * (RAMP.length - 1);
  const i = Math.floor(t);
  const f = t - i;
  const a = RAMP[i]!;
  const b = RAMP[i + 1]!;
  // Cells below the cutoff fade out so colour marks only what the model scored as abnormal.
  const alpha = Math.min(0.95, Math.max(0, (score - 0.3) / 0.35));
  return [a[0] + (b[0] - a[0]) * f, a[1] + (b[1] - a[1]) * f, a[2] + (b[2] - a[2]) * f, alpha];
}

/** Paints heat cells onto ctx at `scale` canvas pixels per image pixel. */
export function paintHeat(ctx: CanvasRenderingContext2D, heat: HeatGrid, scale: number) {
  ctx.imageSmoothingEnabled = false;
  for (let r = 0; r < heat.rows; r++) {
    for (let c = 0; c < heat.cols; c++) {
      const [R, G, B, A] = heatColour(localScore(heat.values[r]![c]!, heat.cutoff));
      if (A <= 0) continue;
      const x0 = Math.round(heat.xEdges[c]! * scale);
      const y0 = Math.round(heat.yEdges[r]! * scale);
      ctx.fillStyle = `rgba(${R | 0},${G | 0},${B | 0},${A.toFixed(3)})`;
      ctx.fillRect(x0, y0, Math.round(heat.xEdges[c + 1]! * scale) - x0, Math.round(heat.yEdges[r + 1]! * scale) - y0);
    }
  }
}

/** Longest side of the display heat canvas; 32 px cells stay at least 9 px wide on 32 MP inputs. */
export const DISPLAY_MAX_SIDE = 2048;

export async function renderHeatmapPng(img: HTMLImageElement, a: Analysis, strength: number, maxSide = 4096): Promise<Blob> {
  const { width, height } = a.image;
  const scale = Math.min(1, maxSide / Math.max(width, height));
  const canvas = document.createElement('canvas');
  canvas.width = Math.round(width * scale);
  canvas.height = Math.round(height * scale);
  const ctx = canvas.getContext('2d')!;
  ctx.filter = `brightness(${1 - 0.45 * strength}) saturate(${1 - 0.5 * strength})`;
  ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
  ctx.filter = 'none';
  ctx.globalAlpha = strength;
  paintHeat(ctx, a.heat, scale);
  ctx.globalAlpha = 1;
  ctx.lineWidth = Math.max(2, canvas.width / 600);
  ctx.strokeStyle = '#f0f2f4';
  ctx.font = `${Math.max(12, canvas.width / 90)}px sans-serif`;
  ctx.fillStyle = '#f0f2f4';
  for (const r of a.regions) {
    const { x0, y0, x1, y1 } = r.box;
    ctx.strokeRect(x0 * scale, y0 * scale, (x1 - x0) * scale, (y1 - y0) * scale);
    ctx.fillText(r.id, x0 * scale + 4, y0 * scale + Math.max(14, canvas.width / 80));
  }
  return new Promise((resolve, reject) => canvas.toBlob((b) => (b ? resolve(b) : reject(new Error('Canvas export failed'))), 'image/png'));
}
