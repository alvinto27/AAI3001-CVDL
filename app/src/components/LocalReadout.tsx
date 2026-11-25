import type { Analysis, Region } from '../types/analysis';
import { localScore } from '../lib/regions';
import type { CellRef } from './SuspiciousRegionLayer';

export interface Probe {
  cell: CellRef;
  x: number;
  y: number;
  pinned: boolean;
}

interface Props {
  analysis: Analysis | null;
  probe: Probe | null;
  region: Region | null;
}

/** Fixed-height strip under the canvas. Reserves its space so hovering never shifts layout. */
export function LocalReadout({ analysis, probe, region }: Props) {
  if (!analysis) {
    return <Strip><span className="text-ink-3">Local scores appear here after analysis.</span></Strip>;
  }
  if (!probe) {
    return <Strip><span className="text-ink-3">Hover or tap to read the local anomaly score. Click to pin a cell. Scroll to zoom, drag to pan.</span></Strip>;
  }

  const { cell } = probe;
  const raw = analysis.heat.values[cell.row]![cell.col]!;
  const score = localScore(raw, analysis.heat.cutoff);
  const flagged = analysis.heat.flagged[cell.row]![cell.col]!;
  const covering = analysis.tiles.filter((t) => probe.x >= t.box.x0 && probe.x < t.box.x1 && probe.y >= t.box.y0 && probe.y < t.box.y1);
  const topTile = covering.reduce<number | null>((m, t) => (m === null || t.probability > m ? t.probability : m), null);

  return (
    <Strip>
      {probe.pinned && <span className="font-medium text-accent">Pinned</span>}
      <Item label="Position" value={`${Math.floor(probe.x)}, ${Math.floor(probe.y)} px`} />
      <Item label="Cell" value={`r${cell.row + 1} c${cell.col + 1}`} />
      <Item label="Local score" value={score.toFixed(2)} strong accent={flagged ? 'text-signal' : undefined} />
      <Item label="Margin" value={`${raw - analysis.heat.cutoff >= 0 ? '+' : ''}${(raw - analysis.heat.cutoff).toFixed(2)}`} />
      <span className={flagged ? 'font-medium text-signal' : 'text-ink-3'}>{flagged ? 'Flagged' : 'Below cutoff'}</span>
      {topTile !== null && <Item label={`Tile score (${covering.length} overlapping)`} value={topTile.toFixed(2)} />}
      {region && <Item label="Region" value={region.id} />}
    </Strip>
  );
}

function Strip({ children }: { children: React.ReactNode }) {
  return (
    <div className="tnum flex min-h-9 flex-wrap items-center gap-x-5 gap-y-0.5 border-t border-line bg-surface px-3 py-1.5 text-sm" aria-live="off">
      {children}
    </div>
  );
}

function Item({ label, value, strong, accent }: { label: string; value: string; strong?: boolean; accent?: string }) {
  return (
    <span className="whitespace-nowrap">
      <span className="text-ink-3">{label} </span>
      <span className={`${strong ? 'font-semibold' : ''} ${accent ?? 'text-ink'}`}>{value}</span>
    </span>
  );
}
