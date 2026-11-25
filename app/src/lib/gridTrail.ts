/** Pointer trail on the 32 px analysis grid: which cells a path crosses, how bright each is, and its colour. */
export const CELL = 32;
export const FADE_MS = 900;
export const VISIT_ALPHA = 0.25;
export const NEIGHBOUR_ALPHA = 0.08;

const PALETTE = ['#f08e2c', '#d03e76', '#96613c'] as const;
/** Cumulative share of cells per colour: mostly orange, some rose, a little brown. */
const SHARES = [45, 75, 100];

export interface TrailCell {
  c: number;
  r: number;
  /** performance.now() of the last visit. */
  t: number;
  peak: number;
  colour: string;
}

/** Stable per cell, so a revisit never changes colour. */
export function colourOf(c: number, r: number): string {
  const h = (Math.imul(c, 73856093) ^ Math.imul(r, 19349663)) >>> 0;
  return PALETTE[SHARES.findIndex((s) => h % 100 < s)]!;
}

export function alphaOf(cell: Pick<TrailCell, 't' | 'peak'>, now: number): number {
  const age = Math.max(0, now - cell.t);
  return cell.peak * Math.max(0, 1 - age / FADE_MS);
}

/** Every cell the segment crosses, in order, so a fast pointer never skips one (grid traversal, not sampling). */
export function cellsAlong(x0: number, y0: number, x1: number, y1: number): [number, number][] {
  let c = Math.floor(x0 / CELL);
  let r = Math.floor(y0 / CELL);
  const endC = Math.floor(x1 / CELL);
  const endR = Math.floor(y1 / CELL);
  const out: [number, number][] = [[c, r]];
  const dx = x1 - x0;
  const dy = y1 - y0;
  const stepC = Math.sign(dx);
  const stepR = Math.sign(dy);
  const deltaX = dx === 0 ? Infinity : CELL / Math.abs(dx);
  const deltaY = dy === 0 ? Infinity : CELL / Math.abs(dy);
  let nextX = dx === 0 ? Infinity : ((dx > 0 ? c + 1 : c) * CELL - x0) / dx;
  let nextY = dy === 0 ? Infinity : ((dy > 0 ? r + 1 : r) * CELL - y0) / dy;
  for (let n = Math.abs(endC - c) + Math.abs(endR - r); n > 0; n--) {
    if (r === endR || (c !== endC && nextX < nextY)) {
      c += stepC;
      nextX += deltaX;
    } else {
      r += stepR;
      nextY += deltaY;
    }
    out.push([c, r]);
  }
  return out;
}
