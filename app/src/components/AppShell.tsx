import type { ReactNode } from 'react';
import type { HealthState } from '../lib/api';
import { ApiStatus } from './ApiStatus';
import { Header } from './Header';

interface Props {
  status: HealthState;
  onRetry: () => void;
  children: ReactNode;
}

export function AppShell({ status, onRetry, children }: Props) {
  return (
    <div className="flex h-full min-h-0 flex-col">
      <Header>
        <ApiStatus status={status} onRetry={onRetry} />
      </Header>
      <main className="min-h-0 flex-1">{children}</main>
    </div>
  );
}
