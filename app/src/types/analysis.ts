// Domain model the UI renders. Built from the wire response by lib/interpret.ts.

export type VerdictKind = 'suspicious_modification' | 'possible_hidden_data' | 'no_anomaly' | 'inconclusive';

export type CategoryId =
  | 'lsb_steganography'
  | 'object_removal'
  | 'generative_fill'
  | 'copy_move'
  | 'local_reconstruction'
  | 'recompression'
  | 'unknown_manipulation'
  | 'unclassified';

export interface Box {
  x0: number;
  y0: number;
  x1: number;
  y1: number;
}

export interface Region {
  id: string;
  box: Box;
  cells: number;
  /** Peak local score, 0..1, 0.5 = cell cutoff. */
  peak: number;
  mean: number;
  location: string;
  category: CategoryId;
}

export interface HeatGrid {
  rows: number;
  cols: number;
  xEdges: number[];
  yEdges: number[];
  cutoff: number;
  values: number[][];
  flagged: boolean[][];
  kind: string;
}

export interface Tile {
  box: Box;
  probability: number;
  flagged: boolean;
}

export interface CategoryAssessment {
  category: CategoryId;
  /** Present only when the engine reports it. */
  confidence?: number;
  reported: boolean;
}

export interface Analysis {
  verdict: { kind: VerdictKind; headline: string; summary: string; caveat: string };
  score: number;
  threshold: number;
  /** Distance of the raw rule logit from its cutoff. */
  margin: number;
  flagged: { tiles: number; tileTotal: number; cells: number; cellTotal: number };
  regions: Region[];
  categories: CategoryAssessment[];
  heat: HeatGrid;
  tiles: Tile[];
  model: { name: string; version: string };
  image: { width: number; height: number; format: string; orientation: number };
  device: string;
  warnings: string[];
  explanation?: string;
  uncertainty?: number;
  nextStep: string;
}

export interface FileMeta {
  width: number;
  height: number;
  /** PNG colour type 6 */
  hasAlpha: boolean;
  format: string;
  warnings: string[];
}
