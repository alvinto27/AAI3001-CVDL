import { useEffect, useState } from 'react';
import type { HealthState } from '../lib/api';
import { formatBytes } from '../lib/file';
import type { Loaded, Problem } from '../hooks/useSession';
import { ErrorState } from './ErrorState';
import { Row, Section } from './Section';

interface Props {
  loaded: Loaded;
  analysing: boolean;
  status: HealthState;
  problem: Problem | null;
  onAnalyse: () => void;
  onCancel: () => void;
}

function useElapsed(running: boolean) {
  const [s, setS] = useState(0);
  useEffect(() => {
    if (!running) return;
    const t0 = performance.now();
    setS(0);
    const id = setInterval(() => setS((performance.now() - t0) / 1000), 100);
    return () => clearInterval(id);
  }, [running]);
  return s;
}

const MODEL_TEXT: Record<HealthState['kind'], string> = {
  ready: 'Ready',
  loading: 'Still loading',
  offline: 'Unreachable',
  checking: 'Checking',
};

/** Right column before results exist. Same width and rhythm as the evidence column, so nothing jumps when results arrive. */
export function IntakePanel({ loaded, analysing, status, problem, onAnalyse, onCancel }: Props) {
  const elapsed = useElapsed(analysing);
  const { file, meta } = loaded;
  const ready = status.kind === 'ready';

  return (
    <div className="flex flex-col">
      <Section title={analysing ? 'Analysis in progress' : 'Image staged'}>
        {problem && (
          <div className="mb-3">
            <ErrorState {...problem} />
          </div>
        )}
        <dl>
          <Row label="File">
            <span title={file.name} className="inline-block max-w-[14rem] truncate align-bottom">{file.name}</span>
          </Row>
          <Row label="Dimensions">{meta.width} × {meta.height} px</Row>
          <Row label="Format">{meta.format} · {formatBytes(file.size)}</Row>
          <Row label="Model" warn={!ready}>{MODEL_TEXT[status.kind]}</Row>
          <Row label="Status">
            {analysing ? <span className="text-accent">Scanning tiles · {elapsed.toFixed(1)} s</span> : 'Waiting for analysis'}
          </Row>
        </dl>

        <div className="relative mt-3 h-0.5 overflow-hidden bg-line" aria-hidden>
          {analysing && <div className="anim-sweep absolute inset-y-0 left-0 w-2/5 bg-accent" />}
        </div>

        <div className="mt-4 flex gap-2">
          {analysing ? (
            <button type="button" className="btn flex-1" onClick={onCancel}>
              Cancel
            </button>
          ) : (
            <>
              <button type="button" className="btn btn-primary flex-1" onClick={onAnalyse} disabled={!ready}>
                Analyse Image
              </button>
              <button type="button" className="btn" onClick={onCancel}>
                Choose Another
              </button>
            </>
          )}
        </div>
        {!ready && !analysing && <p className="mt-2 text-sm text-ink-3">Analysis needs the model service. Check the status indicator above.</p>}
        {meta.warnings.map((w) => (
          <p key={w} className="mt-2 text-sm text-caution">{w}</p>
        ))}
      </Section>

      <Section title="Analysis verdict">
        <div className="space-y-2" aria-hidden>
          <div className={`h-6 w-4/5 bg-raised ${analysing ? 'animate-pulse' : ''}`} />
          <div className={`h-3 w-full bg-raised ${analysing ? 'animate-pulse' : ''}`} />
          <div className={`h-3 w-3/5 bg-raised ${analysing ? 'animate-pulse' : ''}`} />
        </div>
        <p className="mt-3 text-sm text-ink-3">The verdict, flagged regions and evidence appear here.</p>
      </Section>
    </div>
  );
}
