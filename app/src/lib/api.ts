import type { AnalyzeResponse, ApiErrorBody, HealthResponse } from '../types/api';

export const API_URL: string = (import.meta.env.VITE_API_URL as string | undefined) ?? 'http://127.0.0.1:8000';

export type HealthState =
  | { kind: 'checking' }
  | { kind: 'ready'; health: HealthResponse }
  | { kind: 'loading'; health: HealthResponse }
  | { kind: 'offline' };

export async function fetchHealth(signal?: AbortSignal): Promise<HealthState> {
  try {
    const res = await fetch(`${API_URL}/health`, { cache: 'no-store', signal });
    const health = (await res.json()) as HealthResponse;
    return res.ok && health.ready ? { kind: 'ready', health } : { kind: 'loading', health };
  } catch {
    if (signal?.aborted) return { kind: 'checking' };
    return { kind: 'offline' };
  }
}

export class AnalysisError extends Error {
  constructor(readonly status: number | 'network', readonly title: string, message: string) {
    super(message);
  }
}

const STATUS_COPY: Record<number, [string, string]> = {
  413: ['Image too large', 'The service accepts up to 64 MB and 32 megapixels.'],
  415: ['Unsupported format', 'Upload an 8-bit RGB PNG, JPEG, or uncompressed RGB TIFF.'],
  422: ['Image could not be read', 'The file may be damaged, or one of its sides is under 256 px.'],
  429: ['Service busy', 'Another analysis is running. Try again in a few seconds.'],
  503: ['Model unavailable', 'The model has not finished loading or analysis failed. Check service health, then retry.'],
};

export async function analyzeImage(file: File, opts: { allowRgba: boolean; signal?: AbortSignal }): Promise<AnalyzeResponse> {
  const form = new FormData();
  form.append('photo', file);
  form.append('include_images', 'false');
  form.append('allow_rgba', String(opts.allowRgba));

  let res: Response;
  try {
    res = await fetch(`${API_URL}/analyze`, { method: 'POST', body: form, signal: opts.signal });
  } catch (e) {
    if (opts.signal?.aborted) throw e;
    // A busy service can reject before reading the upload; browsers report that as a network failure.
    throw new AnalysisError('network', 'Upload did not complete', 'The service may be busy or unreachable. Check its status and try again.');
  }
  if (!res.ok) {
    const [title, message] = STATUS_COPY[res.status] ?? ['Analysis failed', 'The service returned an unexpected error.'];
    let detail = '';
    try {
      detail = ((await res.json()) as ApiErrorBody).error?.message ?? '';
    } catch { /* body is optional */ }
    throw new AnalysisError(res.status, title, detail && !STATUS_COPY[res.status] ? detail : message);
  }
  return (await res.json()) as AnalyzeResponse;
}
