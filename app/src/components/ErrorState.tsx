interface Props {
  title: string;
  message: string;
  onDismiss?: () => void;
}

export function ErrorState({ title, message, onDismiss }: Props) {
  return (
    <div role="alert" className="flex items-start justify-between gap-4 border-l-2 border-signal bg-raised px-3 py-2.5">
      <div>
        <p className="font-medium text-ink">{title}</p>
        <p className="mt-0.5 text-sm text-ink-2">{message}</p>
      </div>
      {onDismiss && (
        <button type="button" onClick={onDismiss} className="shrink-0 text-sm text-ink-3 hover:text-ink" aria-label="Dismiss message">
          Dismiss
        </button>
      )}
    </div>
  );
}
