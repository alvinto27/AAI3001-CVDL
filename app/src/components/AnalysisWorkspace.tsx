import { useCallback, useEffect, useState } from 'react';
import type { HealthState } from '../lib/api';
import { buildReport, download, stem } from '../lib/export';
import { renderHeatmapPng } from '../lib/heat';
import type { Session } from '../hooks/useSession';
import { AnalysisMetadata } from './AnalysisMetadata';
import { AnomalyClassification } from './AnomalyClassification';
import { DetectionMetrics } from './DetectionMetrics';
import { ImageInspector, type InspectorControls } from './ImageInspector';
import { IntakePanel } from './IntakePanel';
import { ModelLimitations } from './ModelLimitations';
import { SuspiciousRegionList } from './SuspiciousRegionList';
import { VerdictPanel } from './VerdictPanel';

type Active = Exclude<Session, { phase: 'upload' }>;

interface Props {
  session: Active;
  status: HealthState;
  onAnalyse: () => void;
  onReset: () => void;
}

const DEFAULTS: InspectorControls = { mode: 'overlay', strength: 0.7, showGrid: false, showTiles: false };

/** Staged, analysing and results share one layout: image left, evidence right. */
export function AnalysisWorkspace({ session, status, onAnalyse, onReset }: Props) {
  const [controls, setControls] = useState<InspectorControls>(DEFAULTS);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [focus, setFocus] = useState<{ id: string; nonce: number } | null>(null);
  const [cursor, setCursor] = useState(-1);
  const [downloading, setDownloading] = useState(false);
  const analysis = session.phase === 'results' ? session.analysis : null;

  // A new analysis starts from defaults, with the first region in view when the user asks for it.
  useEffect(() => {
    setControls(DEFAULTS);
    setSelectedId(null);
    setFocus(null);
    setCursor(-1);
  }, [analysis]);

  const patch = useCallback((p: Partial<InspectorControls>) => setControls((c) => ({ ...c, ...p })), []);

  const flyTo = (id: string) => {
    setSelectedId(id);
    setFocus((f) => ({ id, nonce: (f?.nonce ?? 0) + 1 }));
  };

  const inspectNext = () => {
    if (!analysis?.regions.length) return;
    const next = (cursor + 1) % analysis.regions.length;
    setCursor(next);
    patch({ mode: 'regions' });
    flyTo(analysis.regions[next]!.id);
  };

  const selectFromList = (id: string) => {
    setCursor(analysis!.regions.findIndex((r) => r.id === id));
    if (controls.mode === 'original') patch({ mode: 'regions' });
    flyTo(id);
  };

  const onDownload = async () => {
    if (!analysis) return;
    setDownloading(true);
    try {
      const img = new Image();
      img.src = session.url;
      await img.decode();
      const strength = controls.mode === 'original' ? 0.7 : controls.strength;
      download(await renderHeatmapPng(img, analysis, strength), `${stem(session.file.name)}-heatmap.png`);
    } finally {
      setDownloading(false);
    }
  };

  const onExport = () => {
    if (session.phase !== 'results') return;
    const json = JSON.stringify(buildReport(session.analysis, session.file.name, session.seconds), null, 2);
    download(new Blob([json], { type: 'application/json' }), `${stem(session.file.name)}-analysis.json`);
  };

  return (
    <div className="h-full overflow-y-auto lg:grid lg:grid-cols-[minmax(0,2fr)_minmax(340px,1fr)] lg:overflow-hidden">
      <div className="h-[68vh] min-h-[360px] lg:h-full">
        <ImageInspector
          url={session.url}
          width={session.meta.width || analysis?.image.width || 1}
          height={session.meta.height || analysis?.image.height || 1}
          analysis={analysis}
          busy={session.phase === 'analysing'}
          controls={controls}
          onControls={patch}
          selectedId={selectedId}
          onSelectRegion={setSelectedId}
          focus={focus}
        />
      </div>

      <aside className="min-h-0 border-t border-line bg-surface lg:overflow-y-auto lg:border-l lg:border-t-0" aria-label="Forensic evidence" aria-busy={session.phase === 'analysing'}>
        {session.phase === 'results' ? (
          <>
            <VerdictPanel
              analysis={session.analysis}
              regionCursor={cursor >= 0 ? `${cursor + 1} of ${session.analysis.regions.length}` : null}
              onInspect={inspectNext}
              onDownload={onDownload}
              onExport={onExport}
              onAnother={onReset}
              downloading={downloading}
            />
            <SuspiciousRegionList analysis={session.analysis} selectedId={selectedId} onSelect={selectFromList} />
            <DetectionMetrics analysis={session.analysis} />
            <AnomalyClassification analysis={session.analysis} />
            <AnalysisMetadata analysis={session.analysis} fileName={session.file.name} fileSize={session.file.size} seconds={session.seconds} />
            <ModelLimitations />
          </>
        ) : (
          <IntakePanel
            loaded={session}
            analysing={session.phase === 'analysing'}
            status={status}
            problem={session.phase === 'staged' ? session.problem : null}
            onAnalyse={onAnalyse}
            onCancel={onReset}
          />
        )}
      </aside>
    </div>
  );
}
