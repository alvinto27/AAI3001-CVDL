import type { Analysis } from '../types/analysis';
import { CATEGORIES } from './categories';

export function download(blob: Blob, name: string) {
  const link = document.createElement('a');
  link.href = URL.createObjectURL(blob);
  link.download = name;
  link.click();
  setTimeout(() => URL.revokeObjectURL(link.href), 1000);
}

export const stem = (name: string) => name.replace(/\.[^.]+$/, '');

/** Self-describing report; carries the interpretation limits with the evidence. */
export function buildReport(a: Analysis, fileName: string, seconds: number) {
  return {
    report: 'Image Integrity Inspector analysis',
    generated_at: new Date().toISOString(),
    file: fileName,
    verdict: { label: a.verdict.headline, summary: a.verdict.summary, caveat: a.verdict.caveat },
    score: { value: a.score, threshold: a.threshold, note: 'Sigmoid of the rule logit. Not a calibrated real-world probability.' },
    flagged: a.flagged,
    categories: a.categories.map((c) => ({ ...c, label: CATEGORIES[c.category].label })),
    regions: a.regions.map((r) => ({ id: r.id, box_px: r.box, location: r.location, cells: r.cells, peak_local_score: r.peak, mean_local_score: r.mean })),
    tiles: a.tiles,
    image: a.image,
    model: a.model,
    device: a.device,
    duration_seconds: seconds,
    warnings: a.warnings,
    next_step: a.nextStep,
    limitations: 'Forensic anomaly detector for spatial RGB LSB patterns. A positive result is not proof of manipulation. A negative result is not proof of authenticity.',
  };
}
