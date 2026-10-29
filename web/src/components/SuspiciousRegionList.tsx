import type { Analysis } from '../types/analysis';
import { Section } from './Section';

interface Props {
  analysis: Analysis;
  selectedId: string | null;
  onSelect: (id: string) => void;
}

const MAX_LISTED = 8;

export function SuspiciousRegionList({ analysis: a, selectedId, onSelect }: Props) {
  const listed = a.regions.slice(0, MAX_LISTED);
  const hidden = a.regions.length - listed.length;

  return (
    <Section title="Suspicious regions" aside={`${a.regions.length} found`}>
      {a.regions.length === 0 ? (
        <p className="text-sm text-ink-3">No cells scored above the cutoff, so no region is highlighted.</p>
      ) : (
        <>
          <ul className="-mx-1">
            {listed.map((r) => {
              const active = r.id === selectedId;
              return (
                <li key={r.id}>
                  <button
                    type="button"
                    onClick={() => onSelect(r.id)}
                    aria-pressed={active}
                    className={`grid w-full grid-cols-[2rem_1fr_auto] items-baseline gap-2 border-l-2 px-2 py-1.5 text-left text-sm transition-colors ${
                      active ? 'border-ink bg-raised' : 'border-transparent hover:bg-raised'
                    }`}
                  >
                    <span className="font-semibold text-ink">{r.id}</span>
                    <span className="min-w-0">
                      <span className="block text-ink first-letter:uppercase">{r.location}</span>
                      <span className="tnum block text-xs text-ink-3">
                        {Math.round(r.box.x1 - r.box.x0)} × {Math.round(r.box.y1 - r.box.y0)} px at {Math.round(r.box.x0)}, {Math.round(r.box.y0)}
                      </span>
                    </span>
                    <span className="tnum text-right">
                      <span className="block text-ink">{r.peak.toFixed(2)}</span>
                      <span className="block text-xs text-ink-3">{r.cells} {r.cells === 1 ? 'cell' : 'cells'}</span>
                    </span>
                  </button>
                </li>
              );
            })}
          </ul>
          {hidden > 0 && <p className="mt-1 text-xs text-ink-3">{hidden} smaller {hidden === 1 ? 'region is' : 'regions are'} in the exported report.</p>}
          <p className="mt-2 text-xs text-ink-3">Score is the peak local anomaly score in the region. 0.50 is the cell cutoff.</p>
        </>
      )}
    </Section>
  );
}
