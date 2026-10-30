import { useRef } from 'react';
import type { HealthState } from '../lib/api';
import { heatColour } from '../lib/heat';
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

const CAPABILITIES: [string, string][] = [
  ['Pixel-level scoring', 'Scores spatial RGB least-significant-bit behaviour across the image.'],
  ['Anomaly heatmap', 'Maps abnormal tiles over the original on a 32 px analysis grid.'],
  ['Flagged regions', 'Groups adjacent hot tiles into numbered, zoomable regions.'],
  ['Category hints', 'Labels a region as steganographic or manipulation-like when the model reports one.'],
  ['Exportable evidence', 'Saves the heatmap overlay as a PNG for review.'],
];

export function UploadWorkspace({ status, problem, dragging, onFile, onDismissProblem }: Props) {
  const input = useRef<HTMLInputElement>(null);

  return (
    <div className="h-full overflow-y-auto">
      <div className="mx-auto flex w-full max-w-[1200px] flex-col gap-4 px-4 py-5 sm:px-6 lg:py-6">
        <div className="flex flex-col gap-1 lg:flex-row lg:items-baseline lg:justify-between lg:gap-8">
          <h2 className="text-xl font-semibold leading-tight text-ink sm:text-2xl">
            Inspect an image for hidden data and suspicious modifications.
          </h2>
          <p className="text-sm text-ink-2 lg:text-right">Results are evidence for review, not proof.</p>
        </div>

        {problem && <ErrorState {...problem} onDismiss={onDismissProblem} />}
        {status.kind === 'offline' && !problem && (
          <ErrorState title="Analysis service is not reachable" message="Start the inference API, then retry. You can still stage an image." />
        )}

        <div className="grid gap-4 lg:grid-cols-[minmax(0,7fr)_minmax(0,5fr)]">
          <div className="flex min-w-0 flex-col gap-3">
            <button
              type="button"
              onClick={() => input.current?.click()}
              className={`group relative flex min-h-[260px] flex-1 flex-col items-start justify-end gap-1 overflow-hidden border border-dashed p-5 text-left transition-colors lg:min-h-[300px] ${
                dragging ? 'border-accent bg-raised' : 'border-line-strong bg-surface hover:border-accent'
              }`}
              aria-label="Choose an image to analyse, or drop one here"
            >
              <GridBackdrop active={dragging} />
              <span className="relative text-xl font-semibold text-ink">{dragging ? 'Release to stage this image' : 'Drop an image here'}</span>
              <span className="relative text-ink-2">or choose a file from this computer</span>
              <span className="btn btn-primary pointer-events-none relative mt-3 h-9 px-4">Choose Image</span>
            </button>

            <div aria-label="File requirements">
              <dl className="grid gap-x-6 gap-y-2 text-sm sm:grid-cols-2 xl:grid-cols-4">
                {REQUIREMENTS.map(([k, v]) => (
                  <div key={k} className="min-w-0">
                    <dt className="text-xs text-ink-3">{k}</dt>
                    <dd className="text-ink-2">{v}</dd>
                  </div>
                ))}
              </dl>
              <p className="mt-2 text-xs text-ink-3">The service does not retain uploaded files or results.</p>
            </div>
          </div>

          <div className="flex min-w-0 flex-col gap-4">
            <section aria-labelledby="capabilities-title" className="bg-surface px-4 py-3.5">
              <h3 id="capabilities-title" className="section-title mb-1">Analysis capabilities</h3>
              <dl className="text-sm">
                {CAPABILITIES.map(([k, v]) => (
                  <div key={k} className="dl-row flex-col gap-0 sm:flex-row sm:gap-4">
                    <dt className="w-[8.5rem] shrink-0 font-medium text-ink">{k}</dt>
                    <dd className="min-w-0 text-left text-ink-2">{v}</dd>
                  </div>
                ))}
              </dl>
              <p className="mt-2 text-xs text-ink-3">Spatial LSB analysis only. JPEG/DCT steganography is not covered.</p>
            </section>
            <Comparison />
          </div>
        </div>
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

const CELL = 10;
const COLS = 24;
const ROWS = 14;

/** Synthetic heat cluster for the illustration only; not produced by the model. */
const DEMO_CELLS = Array.from({ length: ROWS * COLS }, (_, i) => {
  const c = i % COLS;
  const r = Math.floor(i / COLS);
  const d = Math.hypot((c - 15.5) / 4.6, (r - 6.5) / 3.6);
  const texture = ((c * 7 + r * 13) % 5) * 0.04;
  return { c, r, score: Math.max(0, 1 - d) * 1.1 - texture };
}).filter((cell) => cell.score > 0.3);

function Scene() {
  return (
    <>
      <rect width={COLS * CELL} height={ROWS * CELL} fill="#39414a" />
      <rect y="92" width={COLS * CELL} height="48" fill="#262c32" />
      <rect x="20" y="40" width="64" height="52" fill="#4a535d" />
      <rect x="30" y="52" width="14" height="18" fill="#2b3138" />
      <rect x="56" y="52" width="14" height="18" fill="#2b3138" />
      <rect x="120" y="24" width="8" height="68" fill="#58626d" />
      <rect x="150" y="56" width="70" height="36" fill="#4f5963" />
      <rect x="166" y="66" width="38" height="12" fill="#6a7580" />
    </>
  );
}

/** Original vs. anomaly heatmap, drawn in code and labelled illustrative so it is never read as a result. */
function Comparison() {
  const w = COLS * CELL;
  const h = ROWS * CELL;
  return (
    <figure className="bg-surface px-4 py-3.5" aria-label="Illustrative comparison of an original image and an anomaly heatmap">
      <div className="mb-2 flex items-baseline justify-between gap-3">
        <h3 className="section-title">Original vs. heatmap</h3>
        <span className="text-xs text-caution">Illustrative, not a model result</span>
      </div>
      <div className="grid grid-cols-2 gap-2">
        <div>
          <svg viewBox={`0 0 ${w} ${h}`} className="block w-full" role="img" aria-label="Synthetic scene, unmodified">
            <Scene />
          </svg>
          <figcaption className="mt-1 text-xs text-ink-3">Original</figcaption>
        </div>
        <div>
          <svg viewBox={`0 0 ${w} ${h}`} className="block w-full" role="img" aria-label="The same synthetic scene with a cluster of heat cells over one area">
            <Scene />
            <rect width={w} height={h} fill="#121417" opacity="0.45" />
            {DEMO_CELLS.map(({ c, r, score }) => {
              const [R, G, B, A] = heatColour(Math.min(1, score));
              return <rect key={`${c}-${r}`} x={c * CELL} y={r * CELL} width={CELL} height={CELL} fill={`rgb(${R | 0} ${G | 0} ${B | 0})`} opacity={A} />;
            })}
            <rect x={11 * CELL} y={3 * CELL} width={9 * CELL} height={8 * CELL} fill="none" stroke="#f0f2f4" strokeWidth="1.5" />
          </svg>
          <figcaption className="mt-1 text-xs text-ink-3">Anomaly heatmap with a flagged region</figcaption>
        </div>
      </div>
    </figure>
  );
}

/** The 32 px analysis grid, drawn faintly behind the drop target: the tool's unit of measure. */
function GridBackdrop({ active }: { active: boolean }) {
  const stroke = active ? '#5cbab2' : '#2a2f36';
  return (
    <svg className="pointer-events-none absolute inset-0 size-full" aria-hidden>
      <defs>
        <pattern id="upload-cells" width="32" height="32" patternUnits="userSpaceOnUse">
          <path d="M32 0H0V32" fill="none" stroke={stroke} strokeWidth="1" />
        </pattern>
      </defs>
      <rect width="100%" height="100%" fill="url(#upload-cells)" />
      <rect x="96" y="64" width="32" height="32" fill="#f08e2c" opacity={active ? 0.9 : 0.55} />
      <rect x="128" y="64" width="32" height="32" fill="#d03e76" opacity={active ? 0.7 : 0.4} />
      <rect x="128" y="96" width="32" height="32" fill="#f08e2c" opacity={active ? 0.6 : 0.3} />
    </svg>
  );
}
