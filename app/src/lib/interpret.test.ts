import { describe, expect, it } from 'vitest';
import type { HeatGrid } from '../types/analysis';
import { decideVerdict, DIFFUSE_SHARE, INCONCLUSIVE_MARGIN } from './interpret';
import { cellIndex, findRegions } from './regions';

const grid = (rows: string[]): HeatGrid => {
  const flagged = rows.map((r) => [...r].map((ch) => ch === '#'));
  const cols = flagged[0]!.length;
  return {
    rows: rows.length, cols, kind: 'test', cutoff: 0,
    xEdges: Array.from({ length: cols + 1 }, (_, i) => i * 32),
    yEdges: Array.from({ length: rows.length + 1 }, (_, i) => i * 32),
    flagged,
    values: flagged.map((r) => r.map((f) => (f ? 2 : -2))),
  };
};

describe('findRegions', () => {
  it('splits separated clusters, joins diagonal neighbours, sorts by size', () => {
    const regions = findRegions(grid(['#.....', '.#....', '......', '....##', '....##']), 192, 160);
    expect(regions.map((r) => [r.id, r.cells])).toEqual([['R1', 4], ['R2', 2]]);
    expect(regions[0]!.box).toEqual({ x0: 128, y0: 96, x1: 192, y1: 160 });
    expect(regions[0]!.location).toBe('lower-right');
  });

  it('returns nothing when no cell is flagged', () => {
    expect(findRegions(grid(['...', '...']), 96, 64)).toEqual([]);
  });
});

describe('cellIndex', () => {
  const edges = [0, 32, 64, 80];
  it.each([[0, 0], [31.9, 0], [32, 1], [79, 2], [80, -1], [-1, -1]])('maps %d to %d', (v, i) => {
    expect(cellIndex(edges, v)).toBe(i);
  });
});

describe('decideVerdict', () => {
  const far = INCONCLUSIVE_MARGIN + 1;
  it('localised detection is a suspicious modification', () => {
    expect(decideVerdict(true, far, 0.1, 'lower-right', false).kind).toBe('suspicious_modification');
  });
  it('image-wide detection is possible hidden data', () => {
    expect(decideVerdict(true, far, DIFFUSE_SHARE, null, false).kind).toBe('possible_hidden_data');
  });
  it('a detection near the cutoff is inconclusive', () => {
    expect(decideVerdict(true, 0.1, 0.1, 'centre', false).kind).toBe('inconclusive');
  });
  it('a clean JPEG is inconclusive, a clean lossless image is not', () => {
    expect(decideVerdict(false, -3, 0, null, true).kind).toBe('inconclusive');
    expect(decideVerdict(false, -3, 0, null, false).kind).toBe('no_anomaly');
  });
  it('never claims proof', () => {
    for (const d of [true, false]) {
      const v = decideVerdict(d, far, 0.1, 'centre', false);
      expect(v.summary + v.caveat).not.toMatch(/was (edited|modified|tampered)|Magic Eraser/i);
    }
  });
});

describe('contract with the real engine', () => {
  it('interprets a captured /analyze response', async () => {
    const raw = (await import('./__fixtures__/real-response.json')).default as unknown as import('../types/api').AnalyzeResponse;
    const { interpret } = await import('./interpret');
    const a = interpret(raw);
    expect(a.verdict.kind).toBe('possible_hidden_data');
    expect(a.flagged.cellTotal).toBe(raw.heatmap.shape[0] * raw.heatmap.shape[1]);
    expect(a.regions.length).toBeGreaterThan(0);
    expect(a.threshold).toBeCloseTo(1 / (1 + Math.exp(0.05636)), 4);
    expect(a.model.name).toBe(raw.bundle.name);
  });
});
