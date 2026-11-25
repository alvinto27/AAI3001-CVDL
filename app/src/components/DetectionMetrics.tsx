import type { Analysis } from '../types/analysis';
import { Row, Section } from './Section';
import { VERDICT_TONE } from './VerdictPanel';

export function DetectionMetrics({ analysis: a }: { analysis: Analysis }) {
  const fill = VERDICT_TONE[a.verdict.kind].fill;
  const pct = (n: number) => `${Math.min(100, Math.max(0, n * 100))}%`;
  const cellShare = (a.flagged.cells / a.flagged.cellTotal) * 100;

  return (
    <Section title="Confidence and thresholds">
      <div className="flex items-end justify-between gap-3">
        <div>
          <div className="tnum text-[2.25rem] font-semibold leading-none text-ink">{a.score.toFixed(2)}</div>
          <div className="mt-1 text-xs text-ink-3">Confidence score. Uncalibrated, not a probability of tampering.</div>
        </div>
      </div>

      <div
        className="relative mt-3 h-2 bg-raised"
        role="meter"
        aria-label="Confidence score against decision threshold"
        aria-valuemin={0}
        aria-valuemax={1}
        aria-valuenow={a.score}
        aria-valuetext={`${a.score.toFixed(2)}, threshold ${a.threshold.toFixed(2)}`}
      >
        <div className="h-full" style={{ width: pct(a.score), background: fill }} />
        <div className="absolute -top-1 h-4 w-0.5 bg-ink" style={{ left: pct(a.threshold) }} aria-hidden />
      </div>
      <div className="tnum relative mt-1 h-4 text-xs text-ink-3">
        <span className="absolute -translate-x-1/2 whitespace-nowrap" style={{ left: `clamp(10%, ${pct(a.threshold)}, 90%)` }}>
          Threshold {a.threshold.toFixed(2)}
        </span>
      </div>

      <dl className="mt-2">
        <Row label="Decision threshold">{a.threshold.toFixed(2)}</Row>
        <Row label="Margin over threshold">
          {a.margin >= 0 ? '+' : ''}
          {a.margin.toFixed(2)} logit
        </Row>
        <Row label="Flagged tiles">
          {a.flagged.tiles} of {a.flagged.tileTotal}
        </Row>
        <Row label="Flagged 32 px cells">
          {a.flagged.cells} of {a.flagged.cellTotal} ({cellShare < 10 ? cellShare.toFixed(1) : Math.round(cellShare)}%)
        </Row>
      </dl>
    </Section>
  );
}
