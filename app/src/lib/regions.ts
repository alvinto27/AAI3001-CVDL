import type { Box, HeatGrid, Region } from '../types/analysis';

export const sigmoid = (x: number) => 1 / (1 + Math.exp(-x));

/** 0..1 display score for one heat cell; 0.5 sits exactly on the cell cutoff. */
export const localScore = (value: number, cutoff: number) => sigmoid(value - cutoff);

const NAMES = [
  ['upper-left', 'upper-centre', 'upper-right'],
  ['left', 'centre', 'right'],
  ['lower-left', 'lower-centre', 'lower-right'],
] as const;

export function describeLocation(cx: number, cy: number, width: number, height: number): string {
  const col = Math.min(2, Math.floor((cx / width) * 3));
  const row = Math.min(2, Math.floor((cy / height) * 3));
  return NAMES[row]![col]!;
}

/** Index of the cell whose [edge[i], edge[i+1]) range holds v, or -1. */
export function cellIndex(edges: number[], v: number): number {
  if (v < edges[0]! || v >= edges[edges.length - 1]!) return -1;
  let lo = 0;
  let hi = edges.length - 2;
  while (lo < hi) {
    const mid = (lo + hi + 1) >> 1;
    if (edges[mid]! <= v) lo = mid;
    else hi = mid - 1;
  }
  return lo;
}

/** 8-connected components of flagged cells, largest first. */
export function findRegions(heat: HeatGrid, width: number, height: number): Region[] {
  const { rows, cols, flagged, values, cutoff, xEdges, yEdges } = heat;
  const seen = Array.from({ length: rows }, () => new Uint8Array(cols));
  const found: Omit<Region, 'id'>[] = [];

  for (let r0 = 0; r0 < rows; r0++) {
    for (let c0 = 0; c0 < cols; c0++) {
      if (!flagged[r0]![c0] || seen[r0]![c0]) continue;
      const stack: [number, number][] = [[r0, c0]];
      seen[r0]![c0] = 1;
      let rMin = r0, rMax = r0, cMin = c0, cMax = c0, cells = 0, sum = 0, peak = 0, sx = 0, sy = 0;
      while (stack.length) {
        const [r, c] = stack.pop()!;
        const s = localScore(values[r]![c]!, cutoff);
        cells++; sum += s; peak = Math.max(peak, s);
        sx += (xEdges[c]! + xEdges[c + 1]!) / 2;
        sy += (yEdges[r]! + yEdges[r + 1]!) / 2;
        rMin = Math.min(rMin, r); rMax = Math.max(rMax, r);
        cMin = Math.min(cMin, c); cMax = Math.max(cMax, c);
        for (let dr = -1; dr <= 1; dr++) {
          for (let dc = -1; dc <= 1; dc++) {
            const nr = r + dr, nc = c + dc;
            if (nr < 0 || nc < 0 || nr >= rows || nc >= cols) continue;
            if (!flagged[nr]![nc] || seen[nr]![nc]) continue;
            seen[nr]![nc] = 1;
            stack.push([nr, nc]);
          }
        }
      }
      const box: Box = { x0: xEdges[cMin]!, y0: yEdges[rMin]!, x1: xEdges[cMax + 1]!, y1: yEdges[rMax + 1]! };
      found.push({
        box, cells, peak, mean: sum / cells,
        location: describeLocation(sx / cells, sy / cells, width, height),
        category: 'unclassified',
      });
    }
  }
  return found.sort((a, b) => b.cells - a.cells).map((r, i) => ({ ...r, id: `R${i + 1}` }));
}

/** One phrase for where the flagged area sits, weighted by region size. */
export function overallLocation(regions: Region[], width: number, height: number): string | null {
  const total = regions.reduce((n, r) => n + r.cells, 0);
  if (!total) return null;
  let cx = 0, cy = 0;
  for (const r of regions) {
    cx += ((r.box.x0 + r.box.x1) / 2) * r.cells;
    cy += ((r.box.y0 + r.box.y1) / 2) * r.cells;
  }
  return describeLocation(cx / total, cy / total, width, height);
}
