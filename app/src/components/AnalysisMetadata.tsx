import type { Analysis } from '../types/analysis';
import { formatBytes } from '../lib/file';
import { Row, Section } from './Section';

interface Props {
  analysis: Analysis;
  fileName: string;
  fileSize: number;
  seconds: number;
}

export function AnalysisMetadata({ analysis: a, fileName, fileSize, seconds }: Props) {
  return (
    <Section title="Analysis metadata">
      <dl>
        <Row label="File">
          <span title={fileName} className="inline-block max-w-[14rem] truncate align-bottom">{fileName}</span>
        </Row>
        <Row label="Dimensions">{a.image.width} × {a.image.height} px</Row>
        <Row label="Format">{a.image.format} · {formatBytes(fileSize)}</Row>
        <Row label="Model">{a.model.name} v{a.model.version}</Row>
        <Row label="Device">{a.device.toUpperCase()}</Row>
        <Row label="Duration">{seconds.toFixed(1)} s</Row>
        {a.uncertainty !== undefined && <Row label="Uncertainty">{a.uncertainty.toFixed(2)}</Row>}
      </dl>

      {a.warnings.length > 0 && (
        <div className="mt-3" role="note">
          <h4 className="mb-1 text-sm font-medium text-caution">Input warnings</h4>
          <ul className="list-disc space-y-1 pl-4 text-sm text-ink-2 marker:text-caution">
            {a.warnings.map((w) => (
              <li key={w}>{w}</li>
            ))}
          </ul>
        </div>
      )}
    </Section>
  );
}
