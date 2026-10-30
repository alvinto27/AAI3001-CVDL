// Wire contract of the Python inference service (cvdl/pipeline). Fields the service
// may add later (category, explanation, regions...) are optional so the adapter in
// lib/interpret.ts can pick them up without touching components.

export interface BundleIdentity {
  name: string;
  version: string | number;
  format_version?: string | number;
  weights_sha256?: string;
}

export interface HealthResponse {
  ready: boolean;
  bundle: BundleIdentity | null;
  device: string | null;
}

export interface RawTile {
  box: [number, number, number, number];
  probability: number;
  flagged: boolean;
}

export interface RawHeatmap {
  kind: string;
  score: string;
  cell_size: number;
  shape: [number, number];
  x_edges: number[];
  y_edges: number[];
  cutoff: number;
  values: number[][];
  flagged: boolean[][];
}

export interface RawRegion {
  box: [number, number, number, number];
  score?: number;
  category?: string;
  label?: string;
}

export interface AnalyzeResponse {
  schema_version: number;
  bundle: BundleIdentity;
  device: string;
  width: number;
  height: number;
  input: { format: string; orientation: number; warnings: string[] };
  photo_verdict: 'message_detected' | 'not_detected';
  photo_rule: { type: string; logit: number; probability: number; cutoff: number; probability_note?: string };
  features: Record<string, number>;
  counts: Record<string, number>;
  tile_cutoff: number;
  tiles: RawTile[];
  heatmap: RawHeatmap;
  images?: { content_type: string; display_scale: string; heatmap_png_base64: string; overlay_png_base64: string };

  // Reserved for future engine versions.
  anomaly_category?: string;
  anomaly_categories?: { category: string; confidence?: number }[];
  regions?: RawRegion[];
  explanation?: string;
  uncertainty?: number;
  recommended_next_step?: string;
}

export interface ApiErrorBody {
  error: { status: number; message: string };
  bundle: BundleIdentity | null;
}
