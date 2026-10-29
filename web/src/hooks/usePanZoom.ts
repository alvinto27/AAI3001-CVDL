import { useCallback, useEffect, useLayoutEffect, useRef, useState, type PointerEvent, type RefObject } from 'react';
import type { Box } from '../types/analysis';

export interface View {
  /** Screen pixels per image pixel. */
  s: number;
  x: number;
  y: number;
}

const PAD = 16;
const KEEP_VISIBLE = 80;
const MAX_SCALE = 32;

export function usePanZoom(viewport: RefObject<HTMLDivElement | null>, imgW: number, imgH: number) {
  const [size, setSize] = useState({ w: 0, h: 0 });
  const [view, setViewState] = useState<View>({ s: 1, x: 0, y: 0 });
  const fitted = useRef(true);
  const drag = useRef<{ px: number; py: number; vx: number; vy: number; moved: boolean } | null>(null);
  const moved = useRef(false);

  const fitView = useCallback((): View => {
    const s = Math.max(0.01, Math.min((size.w - PAD * 2) / imgW, (size.h - PAD * 2) / imgH));
    return { s, x: (size.w - imgW * s) / 2, y: (size.h - imgH * s) / 2 };
  }, [size, imgW, imgH]);

  const clamp = useCallback((v: View): View => {
    const fit = fitView().s;
    const s = Math.min(MAX_SCALE, Math.max(fit * 0.5, v.s));
    const w = imgW * s;
    const h = imgH * s;
    return {
      s,
      x: Math.min(size.w - KEEP_VISIBLE, Math.max(KEEP_VISIBLE - w, v.x)),
      y: Math.min(size.h - KEEP_VISIBLE, Math.max(KEEP_VISIBLE - h, v.y)),
    };
  }, [fitView, imgW, imgH, size]);

  useLayoutEffect(() => {
    const el = viewport.current;
    if (!el) return;
    const measure = () => setSize({ w: el.clientWidth, h: el.clientHeight });
    measure();
    const ro = new ResizeObserver(measure);
    ro.observe(el);
    return () => ro.disconnect();
  }, [viewport]);

  useEffect(() => {
    if (size.w && fitted.current) setViewState(fitView());
  }, [size, fitView]);

  const setView = useCallback((v: View) => {
    fitted.current = false;
    setViewState(clamp(v));
  }, [clamp]);

  const zoomAt = useCallback((factor: number, cx: number, cy: number) => {
    const nextS = Math.min(MAX_SCALE, Math.max(fitView().s * 0.5, view.s * factor));
    const k = nextS / view.s;
    setView({ s: nextS, x: cx - (cx - view.x) * k, y: cy - (cy - view.y) * k });
  }, [view, setView, fitView]);

  const zoomBy = useCallback((factor: number) => zoomAt(factor, size.w / 2, size.h / 2), [zoomAt, size]);

  const reset = useCallback(() => {
    fitted.current = true;
    setViewState(fitView());
  }, [fitView]);

  const actualSize = useCallback(() => {
    const s = 1;
    setView({ s, x: (size.w - imgW) / 2, y: (size.h - imgH) / 2 });
  }, [setView, size, imgW, imgH]);

  const flyTo = useCallback((box: Box) => {
    const bw = box.x1 - box.x0;
    const bh = box.y1 - box.y0;
    const fit = fitView().s;
    // Keep the region to under half the viewport so the surrounding image stays readable as context.
    const s = Math.min(8, Math.max(fit, Math.min((size.w * 0.45) / bw, (size.h * 0.45) / bh)));
    setView({ s, x: size.w / 2 - ((box.x0 + box.x1) / 2) * s, y: size.h / 2 - ((box.y0 + box.y1) / 2) * s });
  }, [fitView, setView, size]);

  /** Viewport pointer position → image pixel coordinates. */
  const toImage = useCallback((clientX: number, clientY: number) => {
    const rect = viewport.current!.getBoundingClientRect();
    return { x: (clientX - rect.left - view.x) / view.s, y: (clientY - rect.top - view.y) / view.s };
  }, [viewport, view]);

  // React's onWheel is passive; zooming needs preventDefault.
  const zoomAtRef = useRef(zoomAt);
  zoomAtRef.current = zoomAt;
  useEffect(() => {
    const el = viewport.current;
    if (!el) return;
    const onWheel = (e: WheelEvent) => {
      e.preventDefault();
      const rect = el.getBoundingClientRect();
      zoomAtRef.current(Math.exp(-e.deltaY * (e.ctrlKey ? 0.01 : 0.0018)), e.clientX - rect.left, e.clientY - rect.top);
    };
    el.addEventListener('wheel', onWheel, { passive: false });
    return () => el.removeEventListener('wheel', onWheel);
  }, [viewport]);

  const handlers = {
    onPointerDown: (e: PointerEvent) => {
      if (e.button !== 0) return;
      (e.currentTarget as HTMLElement).setPointerCapture(e.pointerId);
      drag.current = { px: e.clientX, py: e.clientY, vx: view.x, vy: view.y, moved: false };
      moved.current = false;
    },
    onPointerMove: (e: PointerEvent) => {
      const d = drag.current;
      if (!d) return;
      const dx = e.clientX - d.px;
      const dy = e.clientY - d.py;
      if (!d.moved && Math.hypot(dx, dy) < 4) return;
      d.moved = true;
      moved.current = true;
      setView({ s: view.s, x: d.vx + dx, y: d.vy + dy });
    },
    onPointerUp: () => {
      drag.current = null;
    },
  };

  return { view, size, handlers, zoomBy, reset, actualSize, flyTo, toImage, wasDrag: moved };
}
