import { useCallback, useState } from 'react';
import { AnalysisWorkspace } from './components/AnalysisWorkspace';
import { AppShell } from './components/AppShell';
import { UploadWorkspace } from './components/UploadWorkspace';
import { useApiStatus } from './hooks/useApiStatus';
import { useFileDrop } from './hooks/useFileDrop';
import { useSession } from './hooks/useSession';

export function App() {
  const { status, recheck } = useApiStatus();
  const { session, stage, analyse, reset } = useSession();
  const [dropNote, setDropNote] = useState<string | null>(null);

  const canStage = session.phase === 'upload' || session.phase === 'staged';
  const onMany = useCallback(() => setDropNote('Drop one image at a time.'), []);
  const onFile = useCallback((f: File) => {
    setDropNote(null);
    void stage(f);
  }, [stage]);
  const dragging = useFileDrop(canStage, onFile, onMany);

  return (
    <AppShell status={status} onRetry={recheck}>
      {session.phase === 'upload' ? (
        <UploadWorkspace
          status={status}
          problem={dropNote ? { title: 'One image at a time', message: dropNote } : session.problem}
          dragging={dragging}
          onFile={onFile}
          onDismissProblem={() => {
            setDropNote(null);
            reset();
          }}
        />
      ) : (
        <AnalysisWorkspace session={session} status={status} onAnalyse={() => void analyse()} onReset={reset} />
      )}
    </AppShell>
  );
}
