import { useMemo } from 'react';
import type { Analysis } from '../types/analysis';

export type ViewMode = 'original' | 'overlay' | 'regions';
export interface CellRef {
  row: number;
  col: number;
}

interface Props {
  analysis: Analysis;
  mode: ViewMode;
  /** Scrim darkness in Suspicious Regions view, 0..1. */
  strength: number;
  showGrid: boolean;
  showTiles: boolean;
  selectedId: string | null;
  hover: CellRef | null;
  pinned: CellRef | null;
  /** Screen pixels per image pixel; keeps strokes and labels a constant on-screen size. */
  scale: number;
}

/** Merges horizontally adjacent flagged cells into one rectangle per run. */
function flaggedPath(a: Analysis): string {
  const { flagged, xEdges, yEdges } = a.heat;
  const parts: string[] = [];
  flagged.forEach((row, r) => {
    let c = 0;
    while (c < row.length) {
      if (!row[c]) { c++; continue; }
      const start = c;
      while (c < row.length && row[c]) c++;
      const x0 = xEdges[start]!, x1 = xEdges[c]!, y0 = yEdges[r]!, y1 = yEdges[r + 1]!;
      parts.push(`M${x0} ${y0}H${x1}V${y1}H${x0}Z`);
    }
  });
  return parts.join('');
}

function gridPath(a: Analysis): string {
  const { xEdges, yEdges } = a.heat;
  const w = xEdges[xEdges.length - 1]!;
  const h = yEdges[yEdges.length - 1]!;
  return xEdges.map((x) => `M${x} 0V${h}`).join('') + yEdges.map((y) => `M0 ${y}H${w}`).join('');
}

export function SuspiciousRegionLayer({ analysis: a, mode, strength, showGrid, showTiles, selectedId, hover, pinned, scale }: Props) {
  const { width, height } = a.image;
  const flagged = useMemo(() => flaggedPath(a), [a]);
  const grid = useMemo(() => gridPath(a), [a]);
  const px = 1 / scale;

  const cellRect = (c: CellRef, stroke: string, w: number) => {
    const x0 = a.heat.xEdges[c.col]!, x1 = a.heat.xEdges[c.col + 1]!;
    const y0 = a.heat.yEdges[c.row]!, y1 = a.heat.yEdges[c.row + 1]!;
    return <rect x={x0} y={y0} width={x1 - x0} height={y1 - y0} fill="none" stroke={stroke} strokeWidth={w * px} />;
  };

  const showBoxes = mode === 'regions';

  return (
    <svg viewBox={`0 0 ${width} ${height}`} preserveAspectRatio="none" className="pointer-events-none absolute inset-0 size-full" aria-hidden>
      {mode === 'regions' && (
        <>
          <defs>
            <mask id="reveal-flagged" maskUnits="userSpaceOnUse" x="0" y="0" width={width} height={height}>
              <rect width={width} height={height} fill="#fff" />
              <path d={flagged} fill="#000" />
            </mask>
          </defs>
          <rect width={width} height={height} fill="#0a0c0e" opacity={0.3 + 0.6 * strength} mask="url(#reveal-flagged)" />
          <path d={flagged} fill="none" stroke="#f08e2c" strokeWidth={1 * px} opacity={0.8} />
        </>
      )}

      {showGrid && <path d={grid} fill="none" stroke="#fff" strokeOpacity={0.22} strokeWidth={px} />}

      {showTiles &&
        a.tiles.map((t, i) => (
          <rect
            key={i}
            x={t.box.x0} y={t.box.y0} width={t.box.x1 - t.box.x0} height={t.box.y1 - t.box.y0}
            fill="none"
            stroke={t.flagged ? '#f08e2c' : '#5cbab2'}
            strokeOpacity={t.flagged ? 0.85 : 0.35}
            strokeDasharray={`${4 * px} ${3 * px}`}
            strokeWidth={1.25 * px}
          />
        ))}

      {a.regions.map((r) => {
        const selected = r.id === selectedId;
        if (!showBoxes && !selected) return null;
        const bw = r.box.x1 - r.box.x0;
        const bh = r.box.y1 - r.box.y0;
        const labelW = (r.id.length * 7 + 10) * px;
        const labelH = 16 * px;
        return (
          <g key={r.id}>
            <rect
              x={r.box.x0} y={r.box.y0} width={bw} height={bh}
              fill="none"
              stroke={selected ? '#ffffff' : '#eceef1'}
              strokeWidth={(selected ? 2.5 : 1.5) * px}
            />
            <rect x={r.box.x0} y={Math.max(0, r.box.y0 - labelH)} width={labelW} height={labelH} fill={selected ? '#ffffff' : '#eceef1'} />
            <text
              x={r.box.x0 + 5 * px}
              y={Math.max(0, r.box.y0 - labelH) + 12 * px}
              fontSize={11 * px}
              fontWeight={600}
              fill="#121417"
              fontFamily="IBM Plex Sans, sans-serif"
            >
              {r.id}
            </text>
          </g>
        );
      })}

      {pinned && cellRect(pinned, '#5cbab2', 2.5)}
      {hover && !(pinned && pinned.row === hover.row && pinned.col === hover.col) && cellRect(hover, '#ffffff', 1.5)}
    </svg>
  );
}
