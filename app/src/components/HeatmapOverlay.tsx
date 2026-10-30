import { useEffect, useRef } from 'react';
import type { HeatGrid } from '../types/analysis';
import { DISPLAY_MAX_SIDE, paintHeat } from '../lib/heat';

interface Props {
  heat: HeatGrid;
  width: number;
  height: number;
  opacity: number;
}

/** Continuous heatmap in image coordinates. Drawn once; opacity is a CSS property. */
export function HeatmapOverlay({ heat, width, height, opacity }: Props) {
  const ref = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = ref.current!;
    const scale = Math.min(1, DISPLAY_MAX_SIDE / Math.max(width, height));
    canvas.width = Math.round(width * scale);
    canvas.height = Math.round(height * scale);
    const ctx = canvas.getContext('2d')!;
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    paintHeat(ctx, heat, scale);
  }, [heat, width, height]);

  return (
    <canvas
      ref={ref}
      className="pointer-events-none absolute inset-0 size-full"
      style={{ opacity, imageRendering: 'pixelated' }}
      aria-hidden
    />
  );
}
