import { describe, expect, it } from 'vitest';
import { alphaOf, CELL, cellsAlong, colourOf, FADE_MS, NEIGHBOUR_ALPHA, VISIT_ALPHA } from './gridTrail';

const touching = (a: [number, number], b: [number, number]) => Math.abs(a[0] - b[0]) + Math.abs(a[1] - b[1]) === 1;

describe('cellsAlong', () => {
  it('returns the one cell when the pointer has not left it', () => {
    expect(cellsAlong(40, 40, 60, 50)).toEqual([[1, 1]]);
  });

  it('visits every cell of a fast horizontal sweep', () => {
    expect(cellsAlong(5, 10, 5 + 6 * CELL, 10)).toEqual([0, 1, 2, 3, 4, 5, 6].map((c) => [c, 0]));
  });

  it('visits every cell of a fast diagonal jump with no gap between neighbours', () => {
    const path = cellsAlong(5, 5, 300, 200);
    expect(path[0]).toEqual([0, 0]);
    expect(path.at(-1)).toEqual([Math.floor(300 / CELL), Math.floor(200 / CELL)]);
    for (let i = 1; i < path.length; i++) expect(touching(path[i - 1]!, path[i]!)).toBe(true);
  });

  it('works leftwards and upwards, and ends exactly on a cell boundary', () => {
    const path = cellsAlong(300, 200, 64, 32);
    expect(path[0]).toEqual([9, 6]);
    expect(path.at(-1)).toEqual([2, 1]);
    for (let i = 1; i < path.length; i++) expect(touching(path[i - 1]!, path[i]!)).toBe(true);
  });
});

describe('alphaOf', () => {
  const cell = { t: 1000, peak: VISIT_ALPHA };

  it('starts at the peak, halves at half the lifetime and is gone after 900 ms', () => {
    expect(alphaOf(cell, 1000)).toBeCloseTo(VISIT_ALPHA);
    expect(alphaOf(cell, 1000 + FADE_MS / 2)).toBeCloseTo(VISIT_ALPHA / 2);
    expect(alphaOf(cell, 1000 + FADE_MS)).toBe(0);
    expect(alphaOf(cell, 5000)).toBe(0);
  });

  it('never exceeds the peak when the clock reads slightly before the visit', () => {
    expect(alphaOf(cell, 990)).toBeCloseTo(VISIT_ALPHA);
  });

  it('keeps neighbours dimmer than visited cells', () => {
    expect(NEIGHBOUR_ALPHA).toBeLessThan(VISIT_ALPHA);
    expect(alphaOf({ t: 0, peak: NEIGHBOUR_ALPHA }, 0)).toBeCloseTo(0.08);
  });
});

describe('colourOf', () => {
  it('is stable per cell and drawn from the three trail colours', () => {
    const seen = new Set<string>();
    for (let c = 0; c < 40; c++) {
      for (let r = 0; r < 40; r++) {
        expect(colourOf(c, r)).toBe(colourOf(c, r));
        seen.add(colourOf(c, r));
      }
    }
    expect([...seen].sort()).toEqual(['#96613c', '#d03e76', '#f08e2c']);
  });

  it('favours orange over rose over brown', () => {
    const count: Record<string, number> = {};
    for (let c = 0; c < 60; c++) for (let r = 0; r < 60; r++) count[colourOf(c, r)] = (count[colourOf(c, r)] ?? 0) + 1;
    expect(count['#f08e2c']!).toBeGreaterThan(count['#d03e76']!);
    expect(count['#d03e76']!).toBeGreaterThan(count['#96613c']!);
  });
});
