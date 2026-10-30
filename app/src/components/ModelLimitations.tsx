import { Section } from './Section';

const LIMITS = [
  'Examines spatial RGB least-significant-bit patterns only. It is not a general authenticity test and does not cover JPEG/DCT steganography.',
  'Heat regions mark where the model scores pixel behaviour as abnormal. They are not exact payload or edit boundaries.',
  'Measured on the project test pool, about 1 in 12 clean photos was flagged, and more on cameras the model had not seen. Treat a flag as a lead.',
  'A clean result is not proof of authenticity. Very short messages and recompressed or resized images can go undetected.',
  'The model cannot identify which tool or technique produced an anomaly.',
];

export function ModelLimitations() {
  return (
    <Section title="Model limitations">
      <ul className="list-disc space-y-1.5 pl-4 text-sm text-ink-2 marker:text-ink-3">
        {LIMITS.map((l) => (
          <li key={l}>{l}</li>
        ))}
      </ul>
    </Section>
  );
}
