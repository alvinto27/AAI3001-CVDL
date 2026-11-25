import type { AnalyzeResponse } from '../types/api';
import type { Analysis, CategoryAssessment, FileMeta, Region, VerdictKind } from '../types/analysis';
import { asCategoryId } from './categories';
import { findRegions, overallLocation, sigmoid } from './regions';

// Interpretation heuristics. The engine is a forensic anomaly detector; these
// constants only decide how the evidence is worded, never whether it is shown.

/** Rule logit within this distance of its cutoff is reported as inconclusive. */
export const INCONCLUSIVE_MARGIN = 0.5;
/** Flagged share of the heatmap above which the pattern counts as image-wide. */
export const DIFFUSE_SHARE = 0.5;

const CAVEAT = 'Further forensic review is recommended. A positive result is not definitive proof of manipulation.';
const CLEAR_CAVEAT = 'A negative result is not proof that the image is unmodified or free of hidden data.';

interface Verdict {
  kind: VerdictKind;
  headline: string;
  summary: string;
  caveat: string;
}

const HEADLINES: Record<VerdictKind, string> = {
  suspicious_modification: 'SUSPICIOUS MODIFICATION DETECTED',
  possible_hidden_data: 'POSSIBLE HIDDEN DATA',
  no_anomaly: 'NO SIGNIFICANT ANOMALY DETECTED',
  inconclusive: 'INCONCLUSIVE',
};

export function decideVerdict(
  detected: boolean,
  margin: number,
  flaggedShare: number,
  location: string | null,
  compressed: boolean,
): Verdict {
  let kind: VerdictKind;
  if (detected) {
    if (margin < INCONCLUSIVE_MARGIN) kind = 'inconclusive';
    else if (flaggedShare === 0 || flaggedShare >= DIFFUSE_SHARE) kind = 'possible_hidden_data';
    else kind = 'suspicious_modification';
  } else {
    kind = compressed ? 'inconclusive' : 'no_anomaly';
  }

  const where = location ? `concentrated in the ${location} region of the image` : 'without a clear spatial concentration';
  const spread = flaggedShare >= DIFFUSE_SHARE ? 'across most of the image' : 'without a clear spatial concentration';
  const summaries: Record<VerdictKind, string> = {
    suspicious_modification: `The model identified abnormal low-level pixel patterns ${where}. These patterns may be associated with image manipulation or hidden-data embedding.`,
    possible_hidden_data: `The model identified abnormal low-level pixel patterns spread ${spread}. This distribution is consistent with embedded data, though other processing can produce it.`,
    no_anomaly: 'The least-significant-bit structure resembles that of unmodified photographs. The model found no localised abnormal pixel behaviour.',
    inconclusive: detected
      ? 'The score sits close to the decision threshold. The evidence neither supports nor rules out modification or hidden data.'
      : 'No anomaly was found, but the input was lossy-compressed. Compression can erase the low-level traces this model examines.',
  };
  return {
    kind,
    headline: HEADLINES[kind],
    summary: summaries[kind],
    caveat: kind === 'no_anomaly' ? CLEAR_CAVEAT : CAVEAT,
  };
}

function assessCategories(raw: AnalyzeResponse): CategoryAssessment[] {
  if (raw.anomaly_categories?.length) {
    return raw.anomaly_categories.map((c) => ({ category: asCategoryId(c.category), confidence: c.confidence, reported: true }));
  }
  if (raw.anomaly_category) return [{ category: asCategoryId(raw.anomaly_category), reported: true }];
  return [{ category: 'unclassified', reported: false }];
}

export function interpret(raw: AnalyzeResponse, meta?: FileMeta): Analysis {
  const { heatmap: h, width, height } = raw;
  const heat = {
    rows: h.shape[0], cols: h.shape[1], xEdges: h.x_edges, yEdges: h.y_edges,
    cutoff: h.cutoff, values: h.values, flagged: h.flagged, kind: h.kind,
  };

  const computed = findRegions(heat, width, height);
  // Engine-supplied regions win over the ones derived from the heatmap.
  const regions: Region[] = raw.regions?.length
    ? raw.regions.map((r, i) => ({
        id: `R${i + 1}`,
        box: { x0: r.box[0], y0: r.box[1], x1: r.box[2], y1: r.box[3] },
        cells: 0, peak: r.score ?? 0, mean: r.score ?? 0,
        location: r.label ?? 'reported region',
        category: asCategoryId(r.category),
      }))
    : computed;

  const cells = h.flagged.reduce((n, row) => n + row.filter(Boolean).length, 0);
  const cellTotal = heat.rows * heat.cols;
  const margin = raw.photo_rule.logit - raw.photo_rule.cutoff;
  const warnings = [...raw.input.warnings];
  if (meta?.hasAlpha) warnings.unshift('Alpha channel was discarded and not examined.');

  const detected = raw.photo_verdict === 'message_detected';
  const verdict = decideVerdict(detected, margin, cells / cellTotal, overallLocation(computed, width, height), raw.input.format === 'JPEG');
  const nextStep = verdict.kind === 'no_anomaly'
    ? 'No further action is required unless other evidence raises concern.'
    : 'Inspect the highlighted regions at full zoom, compare against a known original if one exists, and corroborate with independent forensic methods.';

  return {
    verdict,
    score: raw.photo_rule.probability,
    threshold: sigmoid(raw.photo_rule.cutoff),
    margin,
    flagged: { tiles: raw.tiles.filter((t) => t.flagged).length, tileTotal: raw.tiles.length, cells, cellTotal },
    regions,
    categories: assessCategories(raw),
    heat,
    tiles: raw.tiles.map((t) => ({ box: { x0: t.box[0], y0: t.box[1], x1: t.box[2], y1: t.box[3] }, probability: t.probability, flagged: t.flagged })),
    model: { name: raw.bundle.name, version: String(raw.bundle.version) },
    image: { width, height, format: raw.input.format, orientation: raw.input.orientation },
    device: raw.device,
    warnings,
    explanation: raw.explanation,
    uncertainty: raw.uncertainty,
    nextStep: raw.recommended_next_step ?? nextStep,
  };
}
