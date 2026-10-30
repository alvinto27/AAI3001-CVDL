import { useEffect, useRef } from 'react';
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

const CAPABILITIES: [string, string][] = [
  ['LSB inspection', 'Scores least-significant-bit behaviour in spatial RGB pixels to surface hidden data. JPEG/DCT steganography is not covered.'],
  ['Spatial anomaly localization', 'Maps abnormal tiles on a 32 px grid and groups neighbouring ones into numbered regions.'],
  ['Forensic evidence review', 'Zoom the heatmap over the original, read each flagged region, and export the overlay as a PNG.'],
];

export function UploadWorkspace({ status, problem, dragging, onFile, onDismissProblem }: Props) {
  const input = useRef<HTMLInputElement>(null);

  return (
    <div className="h-full overflow-y-auto">
      <div className="mx-auto flex min-h-full w-[92%] max-w-[1600px] flex-col gap-4 py-4 lg:gap-5 lg:py-5">
        <div className="flex flex-col gap-1 lg:flex-row lg:items-baseline lg:justify-between lg:gap-8">
          <h2 className="text-xl font-semibold leading-tight text-ink sm:text-2xl">
            Inspect an image for hidden data and suspicious modifications.
          </h2>
          <p className="text-[0.9375rem] text-ink-2">Results are evidence for review, not proof.</p>
        </div>

        {problem && <ErrorState {...problem} onDismiss={onDismissProblem} />}
        {status.kind === 'offline' && !problem && (
          <ErrorState title="Analysis service is not reachable" message="Start the inference API, then retry. You can still stage an image." />
        )}

        <div className="grid flex-1 gap-5 lg:grid-cols-[minmax(0,58fr)_minmax(0,42fr)]">
          <div className="flex min-w-0 flex-col gap-3">
            <button
              type="button"
              onClick={() => input.current?.click()}
              className={`group relative flex min-h-[280px] flex-1 flex-col items-center justify-center gap-1.5 overflow-hidden border border-dashed p-6 text-center transition-colors lg:p-8 ${
                dragging ? 'border-accent bg-raised' : 'border-line-strong bg-surface hover:border-accent'
              }`}
            >
              <GridBackdrop active={dragging} />
              <span className="relative text-2xl font-semibold text-ink lg:text-3xl">{dragging ? 'Release to stage this image' : 'Drop an image here'}</span>
              <span className="relative text-base text-ink-2">or choose a file from this computer</span>
              <span className="btn btn-primary pointer-events-none relative mt-4 h-11 px-6 text-sm lg:h-10">Choose Image</span>
            </button>

            <div aria-label="File requirements" className="bg-surface px-4 py-3">
              <dl className="grid gap-x-6 gap-y-2 text-[0.9375rem] sm:grid-cols-2 xl:grid-cols-4">
                {REQUIREMENTS.map(([k, v]) => (
                  <div key={k} className="min-w-0">
                    <dt className="text-sm font-medium text-ink-2">{k}</dt>
                    <dd className="text-ink">{v}</dd>
                  </div>
                ))}
              </dl>
              <p className="mt-2 text-sm text-ink-2">The service does not retain uploaded files or results.</p>
            </div>
          </div>

          <Comparison />
        </div>

        <section aria-label="Analysis capabilities" className="grid gap-3 md:grid-cols-3">
          {CAPABILITIES.map(([k, v]) => (
            <div key={k} className="bg-surface px-4 py-3">
              <h3 className="text-[0.9375rem] font-semibold text-ink">{k}</h3>
              <p className="mt-1 text-[0.9375rem] leading-snug text-ink-2">{v}</p>
            </div>
          ))}
        </section>
      </div>

      <input
        ref={input}
        type="file"
        accept=".png,.jpg,.jpeg,.tif,.tiff,image/png,image/jpeg,image/tiff"
        aria-label="Image file" className="sr-only"
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

/** Recorded from this app with real run 18 inference on a sample photo; see app/preview-video/README.md. */
const CLIPS = [
  {
    name: 'original',
    caption: 'Original',
    label: 'Recording of the inspector zooming into a photo of two parrots, shown unmodified',
  },
  {
    name: 'heatmap',
    caption: 'Anomaly heatmap, top flagged region outlined',
    label: 'The same recording with the model anomaly heatmap over the photo and the top flagged region outlined',
  },
];

/** Plays both clips from the same instant so the panels stay in step; reduced motion keeps the poster frames. */
function useSyncedLoop() {
  const videos = useRef<(HTMLVideoElement | null)[]>([]);
  useEffect(() => {
    const vs = videos.current.filter((v): v is HTMLVideoElement => v !== null);
    if (matchMedia('(prefers-reduced-motion: reduce)').matches) return;
    let cancelled = false;
    const ready = (v: HTMLVideoElement) =>
      v.readyState >= HTMLMediaElement.HAVE_ENOUGH_DATA ? Promise.resolve() : new Promise<void>((ok) => v.addEventListener('canplaythrough', () => ok(), { once: true }));
    void Promise.all(vs.map(ready)).then(() => {
      if (cancelled) return;
      for (const v of vs) {
        v.currentTime = 0;
        void v.play().catch(() => {});
      }
    });
    return () => {
      cancelled = true;
      for (const v of vs) v.pause();
    };
  }, []);
  return videos;
}

function Comparison() {
  const videos = useSyncedLoop();
  const panel = 'relative aspect-video overflow-hidden bg-ground lg:aspect-auto lg:min-h-[150px] lg:flex-1';
  return (
    <figure className="flex min-w-0 flex-col gap-2 bg-surface p-4" aria-label="Example comparison of an original image and its anomaly heatmap">
      <div className="flex flex-col gap-0.5 sm:flex-row sm:items-baseline sm:justify-between sm:gap-3">
        <h3 className="text-[0.9375rem] font-semibold text-ink">Inspection preview</h3>
        <span className="text-sm text-caution">Example: real model output on a sample photo</span>
      </div>
      <div className="flex flex-col gap-3 lg:flex-1">
        {CLIPS.map((clip, i) => (
          <div key={clip.name} className="flex flex-col gap-1 lg:flex-1">
            <div className={panel}>
              <video
                ref={(el) => { videos.current[i] = el; }}
                src={`/preview/inspection-${clip.name}.mp4`}
                poster={`/preview/inspection-${clip.name}.jpg`}
                muted
                loop
                playsInline
                preload="auto"
                aria-label={clip.label}
                className="absolute inset-0 size-full object-cover"
              />
            </div>
            <figcaption className="text-sm text-ink-2">{clip.caption}</figcaption>
          </div>
        ))}
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
