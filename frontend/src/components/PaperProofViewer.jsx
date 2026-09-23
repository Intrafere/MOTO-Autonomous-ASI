import React, { useMemo, useState } from 'react';
import LatexRenderer from './LatexRenderer';
import { parsePaperProofs, resolvePaperMetrics } from '../utils/paperProofs';
import './PaperProofViewer.css';

const MetricValue = ({ label, value }) => (
  <span className="paper-proof-metric-value">
    <strong>{label}</strong>
    <span>{value.words.toLocaleString()} words</span>
  </span>
);

export function PaperProofMetrics({ content = '', metrics }) {
  const parsed = useMemo(() => parsePaperProofs(content), [content]);
  const resolved = useMemo(() => resolvePaperMetrics(metrics, parsed), [metrics, parsed]);
  return (
    <div className="paper-proof-metrics" aria-label="Paper content metrics">
      <MetricValue label="Paper" value={resolved.paper} />
      <MetricValue label={`Proofs (${resolved.proofCount})`} value={resolved.proofs} />
      <MetricValue label="Total" value={resolved.total} />
    </div>
  );
}

export default function PaperProofViewer({
  content = '',
  prefixContent = '',
  documentId,
  metrics,
  className = '',
  showMetrics = true,
}) {
  const [viewMode, setViewMode] = useState('rendered');
  const parsed = useMemo(() => parsePaperProofs(content), [content]);

  return (
    <div className={`paper-proof-viewer ${className}`}>
      <div className="paper-proof-viewer-toolbar">
        <div className="paper-proof-view-toggle" role="group" aria-label="Paper view">
          <button
            type="button"
            className={viewMode === 'rendered' ? 'active' : ''}
            aria-pressed={viewMode === 'rendered'}
            onClick={() => setViewMode('rendered')}
          >
            Rendered View
          </button>
          <button
            type="button"
            className={viewMode === 'raw' ? 'active' : ''}
            aria-pressed={viewMode === 'raw'}
            onClick={() => setViewMode('raw')}
          >
            Raw View
          </button>
        </div>
        {showMetrics && <PaperProofMetrics content={content} metrics={metrics} />}
      </div>

      {viewMode === 'raw' ? (
        <pre className="paper-proof-raw-content">{prefixContent}{content}</pre>
      ) : (
        <div className="paper-proof-rendered-content">
          {prefixContent && (
            <LatexRenderer documentId={documentId == null ? undefined : `${documentId}:prefix`} content={prefixContent} showToggle={false} defaultRaw={false} />
          )}
          {parsed.segments.map((segment, index) => (
            segment.type === 'paper' ? (
              segment.content && (
                <LatexRenderer
                  key={`paper-${index}`}
                  documentId={documentId == null ? undefined : `${documentId}:paper:${index}`}
                  content={segment.content}
                  showToggle={false}
                  defaultRaw={false}
                />
              )
            ) : (
              <details className="paper-proof-details" key={`${documentId ?? ''}:proof-${segment.proofId || segment.number}-${index}`}>
                <summary>Proof {segment.number}: {segment.theoremName || segment.proofId}</summary>
                <LatexRenderer
                  documentId={documentId == null ? undefined : `${documentId}:proof:${segment.proofId || segment.number}:${index}`}
                  content={segment.content}
                  className="paper-proof-entry"
                  showToggle={false}
                  defaultRaw={false}
                />
              </details>
            )
          ))}
        </div>
      )}
    </div>
  );
}
