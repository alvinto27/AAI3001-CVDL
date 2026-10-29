import type { ReactNode } from 'react';

function Mark() {
  // A 2×2 analysis grid with one flagged cell: the product in one glyph.
  return (
    <svg width="20" height="20" viewBox="0 0 20 20" aria-hidden className="shrink-0">
      <rect x="0.5" y="0.5" width="19" height="19" fill="none" stroke="#7f8791" />
      <path d="M10 0.5v19M0.5 10h19" stroke="#7f8791" />
      <rect x="10" y="10" width="9.5" height="9.5" fill="#f08e2c" />
    </svg>
  );
}

export function Header({ children }: { children?: ReactNode }) {
  return (
    <header className="flex h-11 shrink-0 items-center justify-between gap-4 border-b border-line bg-surface px-4">
      <div className="flex min-w-0 items-center gap-3">
        <Mark />
        <h1 className="truncate text-base font-semibold text-ink">Image Integrity Inspector</h1>
        <span className="hidden truncate text-sm text-ink-3 md:inline">Steganography &amp; Image Manipulation Analysis</span>
      </div>
      {children}
    </header>
  );
}
