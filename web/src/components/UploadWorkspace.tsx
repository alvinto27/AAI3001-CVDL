import { useRef } from 'react';
import type { HealthState } from '../lib/api';
import type { Problem } from '../hooks/useSession';
import { ErrorState } from './ErrorState';

interface Props {
  status: HealthState;
  problem: Problem | null;
  dragging: boolean;
  onFile: (file: File) => void;
  onDismissProblem: () => void;
}

const REQUIREMENTS: [string, string][] = [
  ['Formats', 'PNG, JPEG, uncompressed TIFF'],
  ['Pixels', '8-bit RGB, 256 px minimum side'],
  ['Limits', '64 MB, 32 megapixels'],
  ['Best input', 'Original, never resized or recompressed'],
];

export function UploadWorkspace({ status, problem, dragging, onFile, onDismissProblem }: Props) {
  const input = useRef<HTMLInputElement>(null);

  return (
    <div className="mx-auto flex h-full w-full max-w-[1100px] flex-col justify-center gap-5 px-4 py-6 sm:px-6">
      <div className="max-w-[720px]">
        <h2 className="text-[1.75rem] font-semibold leading-tight text-ink sm:text-[2rem]">
          Inspect an image for hidden data and suspicious modifications.
        </h2>
        <p className="mt-2 text-ink-2">
          The model scores local pixel behaviour and shows where it looks abnormal. Results are evidence for review, not proof.
        </p>
      </div>

      {problem && <ErrorState {...problem} onDismiss={onDismissProblem} />}
      {status.kind === 'offline' && !problem && (
        <ErrorState title="Analysis service is not reachable" message="Start the inference API, then retry. You can still stage an image." />
      )}

      <div className="grid gap-5 lg:grid-cols-[minmax(0,1fr)_280px]">
        <button
          type="button"
          onClick={() => input.current?.click()}
          className={`group relative flex min-h-[300px] flex-col items-start justify-end gap-1 overflow-hidden border border-dashed p-5 text-left transition-colors sm:min-h-[340px] ${
            dragging ? 'border-accent bg-raised' : 'border-line-strong bg-surface hover:border-ink-3'
          }`}
          aria-label="Choose an image to analyse, or drop one here"
        >
          <GridBackdrop active={dragging} />
          <span className="relative text-lg font-semibold text-ink">{dragging ? 'Release to stage this image' : 'Drop an image here'}</span>
          <span className="relative text-ink-2">or choose a file from this computer</span>
          <span className="btn btn-primary relative mt-3 pointer-events-none">Choose Image</span>
        </button>

        <aside className="self-end" aria-label="File requirements">
          <h3 className="section-title mb-1">File requirements</h3>
          <dl className="text-sm">
            {REQUIREMENTS.map(([k, v]) => (
              <div key={k} className="dl-row">
                <dt className="text-ink-3">{k}</dt>
                <dd className="text-right text-ink-2">{v}</dd>
              </div>
            ))}
          </dl>
          <p className="mt-3 text-xs text-ink-3">The service does not retain uploaded files or results.</p>
        </aside>
      </div>

      <input
        ref={input}
        type="file"
        accept=".png,.jpg,.jpeg,.tif,.tiff,image/png,image/jpeg,image/tiff"
        className="sr-only"
        tabIndex={-1}
        onChange={(e) => {
          const f = e.target.files?.[0];
          if (f) onFile(f);
          e.target.value = '';
        }}
      />
    </div>
  );
}

/** The 32 px analysis grid, drawn faintly behind the drop target: the tool's unit of measure. */
function GridBackdrop({ active }: { active: boolean }) {
  const stroke = active ? '#5cbab2' : '#2a2f36';
  return (
    <svg className="pointer-events-none absolute inset-0 size-full" aria-hidden>
      <defs>
        <pattern id="cells" width="32" height="32" patternUnits="userSpaceOnUse">
          <path d="M32 0H0V32" fill="none" stroke={stroke} strokeWidth="1" />
        </pattern>
      </defs>
      <rect width="100%" height="100%" fill="url(#cells)" />
      <rect x="96" y="64" width="32" height="32" fill="#f08e2c" opacity={active ? 0.9 : 0.55} />
      <rect x="128" y="64" width="32" height="32" fill="#d03e76" opacity={active ? 0.7 : 0.4} />
      <rect x="128" y="96" width="32" height="32" fill="#f08e2c" opacity={active ? 0.6 : 0.3} />
    </svg>
  );
}
