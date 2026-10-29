import type { ViewMode } from './SuspiciousRegionLayer';

const MODES: { id: ViewMode; label: string }[] = [
  { id: 'original', label: 'Original' },
  { id: 'overlay', label: 'Detection Overlay' },
  { id: 'regions', label: 'Suspicious Regions' },
];

interface Props {
  mode: ViewMode;
  onChange: (m: ViewMode) => void;
}

export function ViewModeToggle({ mode, onChange }: Props) {
  return (
    <div role="radiogroup" aria-label="View mode" className="inline-flex border border-line-strong">
      {MODES.map((m, i) => {
        const active = m.id === mode;
        return (
          <button
            key={m.id}
            type="button"
            role="radio"
            aria-checked={active}
            onClick={() => onChange(m.id)}
            className={`h-8 px-3 text-sm font-medium transition-colors ${i > 0 ? 'border-l border-line-strong' : ''} ${
              active ? 'bg-ink text-ground' : 'bg-raised text-ink-2 hover:text-ink'
            }`}
          >
            {m.label}
          </button>
        );
      })}
    </div>
  );
}
