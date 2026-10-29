import { useCallback, useEffect, useReducer, useRef } from 'react';
import { AnalysisError, analyzeImage } from '../lib/api';
import { FileProblem, preflight } from '../lib/file';
import { interpret } from '../lib/interpret';
import type { Analysis, FileMeta } from '../types/analysis';

export interface Problem {
  title: string;
  message: string;
}

export interface Loaded {
  file: File;
  meta: FileMeta;
  url: string;
}

export type Session =
  | { phase: 'upload'; problem: Problem | null }
  | ({ phase: 'staged'; problem: Problem | null } & Loaded)
  | ({ phase: 'analysing' } & Loaded)
  | ({ phase: 'results'; analysis: Analysis; seconds: number } & Loaded);

type Action =
  | { type: 'problem'; problem: Problem }
  | { type: 'stage'; loaded: Loaded }
  | { type: 'start' }
  | { type: 'done'; analysis: Analysis; seconds: number }
  | { type: 'fail'; problem: Problem }
  | { type: 'reset' };

function reducer(s: Session, a: Action): Session {
  switch (a.type) {
    case 'problem':
      return s.phase === 'staged' ? { ...s, problem: a.problem } : { phase: 'upload', problem: a.problem };
    case 'stage':
      return { phase: 'staged', problem: null, ...a.loaded };
    case 'start':
      return s.phase === 'staged' ? { phase: 'analysing', file: s.file, meta: s.meta, url: s.url } : s;
    case 'done':
      return s.phase === 'analysing' ? { ...s, phase: 'results', analysis: a.analysis, seconds: a.seconds } : s;
    case 'fail':
      return s.phase === 'analysing'
        ? { phase: 'staged', problem: a.problem, file: s.file, meta: s.meta, url: s.url }
        : s;
    case 'reset':
      return { phase: 'upload', problem: null };
  }
}

/** Upload → staged → analysing → results. Owns the object URL and the in-flight request. */
export function useSession() {
  const [session, dispatch] = useReducer(reducer, { phase: 'upload', problem: null });
  const sessionRef = useRef(session);
  sessionRef.current = session;
  const inflight = useRef<AbortController | null>(null);
  const urlRef = useRef<string | null>(null);

  const release = useCallback(() => {
    inflight.current?.abort();
    inflight.current = null;
    if (urlRef.current) URL.revokeObjectURL(urlRef.current);
    urlRef.current = null;
  }, []);

  useEffect(() => release, [release]);

  const stage = useCallback(async (file: File) => {
    let meta: FileMeta;
    try {
      meta = await preflight(file);
    } catch (e) {
      const message = e instanceof FileProblem ? e.message : 'The file could not be read.';
      dispatch({ type: 'problem', problem: { title: `${file.name} cannot be analysed`, message } });
      return;
    }
    release();
    const url = URL.createObjectURL(file);
    urlRef.current = url;
    dispatch({ type: 'stage', loaded: { file, meta, url } });
  }, [release]);

  const analyse = useCallback(async () => {
    const s = sessionRef.current;
    if (s.phase !== 'staged') return;
    const ctl = new AbortController();
    inflight.current = ctl;
    dispatch({ type: 'start' });

    const started = performance.now();
    try {
      const raw = await analyzeImage(s.file, { allowRgba: s.meta.hasAlpha, signal: ctl.signal });
      if (ctl.signal.aborted) return;
      dispatch({ type: 'done', analysis: interpret(raw, s.meta), seconds: (performance.now() - started) / 1000 });
    } catch (e) {
      if (ctl.signal.aborted) return;
      const problem = e instanceof AnalysisError
        ? { title: e.title, message: e.message }
        : { title: 'Analysis failed', message: 'The response could not be interpreted.' };
      dispatch({ type: 'fail', problem });
    }
  }, []);

  const reset = useCallback(() => {
    release();
    dispatch({ type: 'reset' });
  }, [release]);

  return { session, stage, analyse, reset };
}
