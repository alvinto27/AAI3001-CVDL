import { useEffect, type RefObject } from 'react';
import { alphaOf, CELL, cellsAlong, colourOf, NEIGHBOUR_ALPHA, VISIT_ALPHA, type TrailCell } from '../lib/gridTrail';

const ROW_KEY = 1 << 16;
const NEIGHBOURS = [[-1, -1], [0, -1], [1, -1], [-1, 0], [1, 0], [-1, 1], [0, 1], [1, 1]] as const;

/**
 * Lights the grid cells under the pointer on `canvas`, which must fill its parent. Each cell keeps its own
 * timestamp and fades out on its own; the frame loop runs only while a cell is still visible.
 */
export function useGridTrail(canvas: RefObject<HTMLCanvasElement | null>) {
  useEffect(() => {
    const el = canvas.current;
    const host = el?.parentElement;
    const ctx = el?.getContext('2d');
    if (!el || !host || !ctx) return;

    const reduced = matchMedia('(prefers-reduced-motion: reduce)');
    const cells = new Map<number, TrailCell>();
    let frame = 0;
    let last: [number, number] | null = null;
    let width = 0;
    let height = 0;

    const draw = (now: number) => {
      ctx.clearRect(0, 0, width, height);
      for (const [key, cell] of cells) {
        const alpha = alphaOf(cell, now);
        if (alpha <= 0) {
          cells.delete(key);
          continue;
        }
        ctx.globalAlpha = alpha;
        ctx.fillStyle = cell.colour;
        ctx.fillRect(cell.c * CELL, cell.r * CELL, CELL, CELL);
      }
    };

    const tick = (now: number) => {
      draw(now);
      frame = cells.size > 0 ? requestAnimationFrame(tick) : 0;
    };

    const resize = () => {
      const box = el.getBoundingClientRect();
      const dpr = window.devicePixelRatio || 1;
      width = box.width;
      height = box.height;
      el.width = Math.round(width * dpr);
      el.height = Math.round(height * dpr);
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      draw(performance.now());
    };

    const light = (c: number, r: number, peak: number, now: number) => {
      if (c < 0 || r < 0 || c * CELL >= width || r * CELL >= height) return;
      const key = r * ROW_KEY + c;
      const current = cells.get(key);
      if (current && alphaOf(current, now) > peak) return;
      cells.set(key, { c, r, t: now, peak, colour: colourOf(c, r) });
    };

    const onMove = (e: PointerEvent) => {
      if (reduced.matches) return;
      const box = el.getBoundingClientRect();
      const x = e.clientX - box.left;
      const y = e.clientY - box.top;
      const now = performance.now();
      const path = cellsAlong(...(last ?? [x, y]), x, y);
      last = [x, y];
      for (const [c, r] of path) light(c, r, VISIT_ALPHA, now);
      for (const [c, r] of path) for (const [dc, dr] of NEIGHBOURS) light(c + dc, r + dr, NEIGHBOUR_ALPHA, now);
      if (!frame) frame = requestAnimationFrame(tick);
    };

    const onLeave = () => {
      last = null;
    };

    const onReducedChange = () => {
      if (!reduced.matches) return;
      cells.clear();
      draw(performance.now());
    };

    resize();
    const observer = new ResizeObserver(resize);
    observer.observe(el);
    window.addEventListener('resize', resize);
    host.addEventListener('pointermove', onMove);
    host.addEventListener('pointerleave', onLeave);
    reduced.addEventListener('change', onReducedChange);

    return () => {
      cancelAnimationFrame(frame);
      observer.disconnect();
      window.removeEventListener('resize', resize);
      host.removeEventListener('pointermove', onMove);
      host.removeEventListener('pointerleave', onLeave);
      reduced.removeEventListener('change', onReducedChange);
      cells.clear();
    };
  }, [canvas]);
}
