import { useCallback, useEffect, useState } from 'react';
import { fetchHealth, type HealthState } from '../lib/api';

const INTERVAL_MS = 15_000;

export function useApiStatus() {
  const [state, setState] = useState<HealthState>({ kind: 'checking' });
  const [tick, setTick] = useState(0);

  useEffect(() => {
    const ctl = new AbortController();
    fetchHealth(ctl.signal).then((s) => !ctl.signal.aborted && setState(s));
    const id = setInterval(() => setTick((t) => t + 1), INTERVAL_MS);
    return () => {
      ctl.abort();
      clearInterval(id);
    };
  }, [tick]);

  const recheck = useCallback(() => {
    setState({ kind: 'checking' });
    setTick((t) => t + 1);
  }, []);

  return { status: state, recheck };
}
