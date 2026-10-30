import { memo, useEffect, useMemo, useRef, useState, type KeyboardEvent, type PointerEvent } from 'react';
import type { Analysis } from '../types/analysis';
import { usePanZoom } from '../hooks/usePanZoom';
import { cellIndex } from '../lib/regions';
import { HeatmapOverlay } from './HeatmapOverlay';
import { LocalReadout, type Probe } from './LocalReadout';
import { SuspiciousRegionLayer, type CellRef, type ViewMode } from './SuspiciousRegionLayer';
import { ViewModeToggle } from './ViewModeToggle';

export interface InspectorControls {
  mode: ViewMode;
  strength: number;
  showGrid: boolean;
  showTiles: boolean;
}

interface Props {
  url: string;
  width: number;
  height: number;
  /** Null while staged or analysing. */
  analysis: Analysis | null;
  busy: boolean;
  controls: InspectorControls;
  onControls: (patch: Partial<InspectorControls>) => void;
  selectedId: string | null;
  onSelectRegion: (id: string | null) => void;
  /** Bump `nonce` to fly the view to a region. */
  focus: { id: string; nonce: number } | null;
}

const MemoHeat = memo(HeatmapOverlay);

export function ImageInspector({ url, width, height, analysis, busy, controls, onControls, selectedId, onSelectRegion, focus }: Props) {
  const viewport = useRef<HTMLDivElement>(null);
  const { view, handlers, zoomBy, reset, actualSize, flyTo, toImage, wasDrag } = usePanZoom(viewport, width, height);
  const [hover, setHover] = useState<Probe | null>(null);
  const [pinned, setPinned] = useState<Probe | null>(null);

  const locate = (clientX: number, clientY: number): Omit<Probe, 'pinned'> | null => {
    if (!analysis) return null;
    const { x, y } = toImage(clientX, clientY);
    const col = cellIndex(analysis.heat.xEdges, x);
    const row = cellIndex(analysis.heat.yEdges, y);
    return col < 0 || row < 0 ? null : { cell: { row, col }, x, y };
  };

  const onMove = (e: PointerEvent) => {
    handlers.onPointerMove(e);
    if (e.buttons) return;
    const p = locate(e.clientX, e.clientY);
    setHover((prev) => {
      if (!p) return null;
      return prev && prev.cell.row === p.cell.row && prev.cell.col === p.cell.col && Math.abs(prev.x - p.x) < 4 && Math.abs(prev.y - p.y) < 4
        ? prev
        : { ...p, pinned: false };
    });
  };

  const regionAt = (p: Omit<Probe, 'pinned'> | null) =>
    !p || !analysis?.heat.flagged[p.cell.row]![p.cell.col]
      ? null
      : analysis.regions.find((r) => p.x >= r.box.x0 && p.x < r.box.x1 && p.y >= r.box.y0 && p.y < r.box.y1) ?? null;

  const onClick = (e: React.MouseEvent) => {
    if (wasDrag.current || !analysis) return;
    const p = locate(e.clientX, e.clientY);
    if (!p) {
      setPinned(null);
      onSelectRegion(null);
      return;
    }
    const same = pinned && pinned.cell.row === p.cell.row && pinned.cell.col === p.cell.col;
    setPinned(same ? null : { ...p, pinned: true });
    onSelectRegion(same ? null : regionAt(p)?.id ?? null);
  };

  const onKey = (e: KeyboardEvent) => {
    if (e.key === '+' || e.key === '=') zoomBy(1.25);
    else if (e.key === '-') zoomBy(0.8);
    else if (e.key === '0') reset();
    else if (e.key === 'Escape') {
      setPinned(null);
      onSelectRegion(null);
    }
  };

  const lastNonce = useRef(0);
  useEffect(() => {
    if (!focus || !analysis || focus.nonce === lastNonce.current) return;
    lastNonce.current = focus.nonce;
    const r = analysis.regions.find((x) => x.id === focus.id);
    if (r) flyTo(r.box);
  }, [focus, analysis, flyTo]);

  // Drop pinned/hover probes that belong to a previous analysis.
  useEffect(() => {
    setPinned(null);
    setHover(null);
  }, [analysis]);

  const shown = hover ?? pinned;
  const region = useMemo(() => regionAt(shown), [shown, analysis]); // eslint-disable-line react-hooks/exhaustive-deps
  const hoverCell: CellRef | null = hover?.cell ?? null;
  const hasAnalysis = analysis !== null;
  const pct = Math.round(controls.strength * 100);

  return (
    <section className="flex h-full min-h-0 min-w-0 flex-col bg-ground" aria-label="Image inspection">
      <div className="flex min-h-[52px] flex-wrap items-center justify-between gap-x-4 gap-y-2 border-b border-line bg-surface px-3 py-2">
        {hasAnalysis ? (
          <ViewModeToggle mode={controls.mode} onChange={(mode) => onControls({ mode })} />
        ) : (
          <span className="text-sm font-medium text-ink-2">Original</span>
        )}

        <div className="flex flex-wrap items-center gap-x-4 gap-y-2 text-sm text-ink-2">
          <label className={`flex items-center gap-2 ${controls.mode === 'original' || !hasAnalysis ? 'opacity-40' : ''}`}>
            <span>Heatmap strength</span>
            <input
              type="range" min={0} max={100} value={pct}
              disabled={controls.mode === 'original' || !hasAnalysis}
              onChange={(e) => onControls({ strength: +e.target.value / 100 })}
              className="w-24"
            />
            <span className="tnum w-9 text-right text-ink">{pct}%</span>
          </label>
          <label className={`flex items-center gap-1.5 ${hasAnalysis ? '' : 'opacity-40'}`}>
            <input type="checkbox" checked={controls.showGrid} disabled={!hasAnalysis} onChange={(e) => onControls({ showGrid: e.target.checked })} />
            Cell grid
          </label>
          <label className={`flex items-center gap-1.5 ${hasAnalysis ? '' : 'opacity-40'}`}>
            <input type="checkbox" checked={controls.showTiles} disabled={!hasAnalysis} onChange={(e) => onControls({ showTiles: e.target.checked })} />
            Tiles
          </label>
          <span className="flex items-center border border-line-strong" role="group" aria-label="Zoom">
            <button type="button" className="h-7 w-7 bg-raised text-ink hover:bg-[#272c33]" onClick={() => zoomBy(0.8)} aria-label="Zoom out">−</button>
            <button type="button" className="tnum h-7 min-w-14 border-x border-line-strong bg-raised px-2 text-ink hover:bg-[#272c33]" onClick={reset} title="Fit to window">
              {Math.round(view.s * 100)}%
            </button>
            <button type="button" className="h-7 w-7 bg-raised text-ink hover:bg-[#272c33]" onClick={() => zoomBy(1.25)} aria-label="Zoom in">+</button>
            <button type="button" className="h-7 border-l border-line-strong bg-raised px-2 text-ink hover:bg-[#272c33]" onClick={actualSize} title="Actual pixels (1:1)">1:1</button>
          </span>
        </div>
      </div>

      <div
        ref={viewport}
        tabIndex={0}
        role="img"
        aria-label="Uploaded image. Scroll to zoom, drag to pan, plus and minus keys to zoom."
        className="relative min-h-0 flex-1 touch-none select-none overflow-hidden bg-[#0d0f11] outline-offset-[-2px]"
        style={{ cursor: hasAnalysis ? 'crosshair' : 'grab' }}
        onPointerDown={handlers.onPointerDown}
        onPointerMove={onMove}
        onPointerUp={handlers.onPointerUp}
        onPointerLeave={() => setHover(null)}
        onClick={onClick}
        onKeyDown={onKey}
      >
        <div
          className="absolute left-0 top-0 origin-top-left will-change-transform"
          style={{ width, height, transform: `translate(${view.x}px, ${view.y}px) scale(${view.s})` }}
        >
          <img
            src={url}
            alt=""
            draggable={false}
            className="absolute inset-0 size-full"
            style={{
              imageRendering: view.s >= 3 ? 'pixelated' : 'auto',
              filter: analysis && controls.mode === 'overlay' ? `brightness(${1 - 0.45 * controls.strength}) saturate(${1 - 0.5 * controls.strength})` : undefined,
            }}
          />
          {analysis && controls.mode === 'overlay' && <MemoHeat heat={analysis.heat} width={width} height={height} opacity={controls.strength} />}
          {analysis && (
            <SuspiciousRegionLayer
              analysis={analysis}
              mode={controls.mode}
              strength={controls.strength}
              showGrid={controls.showGrid}
              showTiles={controls.showTiles}
              selectedId={controls.mode === 'original' ? null : selectedId}
              hover={hoverCell}
              pinned={pinned?.cell ?? null}
              scale={view.s}
            />
          )}
          {busy && (
            <div className="pointer-events-none absolute inset-0 overflow-hidden" aria-hidden>
              <div className="anim-scan size-full" style={{ borderBottom: `${2 / view.s}px solid #5cbab2`, background: 'rgba(92,186,178,0.06)' }} />
            </div>
          )}
        </div>
        <div className="tnum pointer-events-none absolute bottom-2 left-2 bg-ground/80 px-1.5 py-0.5 text-xs text-ink-2">
          {width} × {height} px
        </div>
      </div>

      <LocalReadout analysis={analysis} probe={shown} region={region} />
    </section>
  );
}
