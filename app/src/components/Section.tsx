import type { ReactNode } from 'react';

interface Props {
  title: string;
  aside?: ReactNode;
  children: ReactNode;
}

/** One block of the evidence column: a heading, thin top separator, no card chrome. */
export function Section({ title, aside, children }: Props) {
  return (
    <section className="border-t border-line px-4 py-3.5 first:border-t-0">
      <div className="mb-2 flex items-baseline justify-between gap-3">
        <h3 className="section-title">{title}</h3>
        {aside && <div className="tnum text-xs text-ink-3">{aside}</div>}
      </div>
      {children}
    </section>
  );
}

export function Row({ label, children, warn }: { label: string; children: ReactNode; warn?: boolean }) {
  return (
    <div className="dl-row text-sm">
      <dt className="shrink-0 text-ink-3">{label}</dt>
      <dd className={`tnum min-w-0 break-words text-right ${warn ? 'text-caution' : 'text-ink'}`}>{children}</dd>
    </div>
  );
}
