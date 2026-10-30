import type { HealthState } from '../lib/api';
import { API_URL } from '../lib/api';

interface Props {
  status: HealthState;
  onRetry: () => void;
}

export function ApiStatus({ status, onRetry }: Props) {
  const view = (() => {
    switch (status.kind) {
      case 'ready':
        return { dot: 'bg-accent', label: 'Model ready', detail: `${status.health.bundle?.name ?? ''} · ${(status.health.device ?? '').toUpperCase()}` };
      case 'loading':
        return { dot: 'bg-caution', label: 'Model loading', detail: '' };
      case 'offline':
        return { dot: 'bg-signal', label: 'Service offline', detail: API_URL.replace(/^https?:\/\//, '') };
      default:
        return { dot: 'bg-ink-3', label: 'Checking service', detail: '' };
    }
  })();

  return (
    <div className="flex items-center gap-3 text-sm text-ink-2" role="status" aria-live="polite">
      <span className="flex items-center gap-2">
        <span className={`inline-block size-2 ${view.dot}`} aria-hidden />
        <span className="text-ink">{view.label}</span>
      </span>
      {view.detail && <span className="tnum hidden text-ink-3 sm:inline">{view.detail}</span>}
      {(status.kind === 'offline' || status.kind === 'loading') && (
        <button type="button" onClick={onRetry} className="text-accent underline underline-offset-2 hover:text-ink">
          Retry
        </button>
      )}
    </div>
  );
}
