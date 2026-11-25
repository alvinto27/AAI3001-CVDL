import type { Analysis, VerdictKind } from '../types/analysis';
import { Section } from './Section';

export const VERDICT_TONE: Record<VerdictKind, { text: string; bg: string; fill: string }> = {
  suspicious_modification: { text: 'text-signal', bg: 'bg-signal', fill: '#f08e2c' },
  possible_hidden_data: { text: 'text-signal', bg: 'bg-signal', fill: '#f08e2c' },
  inconclusive: { text: 'text-caution', bg: 'bg-caution', fill: '#d9c06a' },
  no_anomaly: { text: 'text-accent', bg: 'bg-accent', fill: '#5cbab2' },
};

interface Props {
  analysis: Analysis;
  /** "R2 of 3" after the user has stepped through regions. */
  regionCursor: string | null;
  onInspect: () => void;
  onDownload: () => void;
  onExport: () => void;
  onAnother: () => void;
  downloading: boolean;
}

export function VerdictPanel({ analysis: a, regionCursor, onInspect, onDownload, onExport, onAnother, downloading }: Props) {
  const tone = VERDICT_TONE[a.verdict.kind];
  return (
    <Section title="Analysis verdict">
      <div className="anim-rise">
        <div className="flex items-start gap-2.5">
          <span className={`mt-[7px] inline-block size-3 shrink-0 ${tone.bg}`} aria-hidden />
          <h2 className={`text-[1.375rem] font-semibold leading-[1.15] ${tone.text}`}>{a.verdict.headline}</h2>
        </div>
        <p className="mt-3 text-[0.9375rem] leading-relaxed text-ink">{a.verdict.summary}</p>
        <p className="mt-2 text-sm text-ink-2">{a.verdict.caveat}</p>
      </div>

      <div className="mt-4 grid grid-cols-2 gap-2">
        <button type="button" className="btn btn-primary col-span-2" onClick={onInspect} disabled={a.regions.length === 0}>
          Inspect Suspicious Regions
          {regionCursor && <span className="tnum font-normal opacity-80">{regionCursor}</span>}
        </button>
        <button type="button" className="btn" onClick={onDownload} disabled={downloading}>
          Download Heatmap
        </button>
        <button type="button" className="btn" onClick={onExport}>
          Export Analysis
        </button>
        <button type="button" className="btn col-span-2" onClick={onAnother}>
          Analyse Another Image
        </button>
      </div>
    </Section>
  );
}
