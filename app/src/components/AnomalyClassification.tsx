import type { Analysis, CategoryId } from '../types/analysis';
import { CATEGORIES, FAMILY_LABEL } from '../lib/categories';
import { Section } from './Section';

const ALL = Object.keys(CATEGORIES) as CategoryId[];

export function AnomalyClassification({ analysis: a }: { analysis: Analysis }) {
  const reported = a.categories.filter((c) => c.reported);
  const none = a.verdict.kind === 'no_anomaly';

  return (
    <Section title="Anomaly type">
      {none ? (
        <p className="text-sm text-ink-3">Not applicable. No anomaly was flagged.</p>
      ) : reported.length > 0 ? (
        <ul>
          {reported.map((c) => (
            <li key={c.category} className="dl-row text-sm">
              <span>
                <span className="text-ink">{CATEGORIES[c.category].label}</span>
                <span className="ml-2 text-xs text-ink-3">{FAMILY_LABEL[CATEGORIES[c.category].family]}</span>
              </span>
              {c.confidence !== undefined && <span className="tnum text-ink">{c.confidence.toFixed(2)}</span>}
            </li>
          ))}
        </ul>
      ) : (
        <>
          <p className="text-sm text-ink">Unclassified anomaly</p>
          <p className="mt-1 text-sm text-ink-2">
            This model version detects abnormal pixel behaviour but does not separate steganographic from manipulation anomalies, and cannot identify an editing tool.
          </p>
        </>
      )}
      {a.explanation && <p className="mt-2 text-sm text-ink-2">{a.explanation}</p>}

      <details className="group mt-2">
        <summary className="text-xs text-ink-3 hover:text-ink">
          <span className="underline underline-offset-2">Categories the engine can report</span>
        </summary>
        <ul className="mt-1.5 grid grid-cols-1 gap-x-4 text-xs text-ink-3 sm:grid-cols-2">
          {ALL.map((id) => (
            <li key={id} className={reported.some((c) => c.category === id) ? 'text-ink' : ''}>
              {CATEGORIES[id].label}
            </li>
          ))}
        </ul>
        <p className="mt-1.5 text-xs text-ink-3">Shown as Not assessed until a model version reports them.</p>
      </details>
    </Section>
  );
}
