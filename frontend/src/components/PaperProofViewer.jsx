import React, { useEffect, useId, useMemo, useRef, useState } from 'react';
import LatexRenderer from './LatexRenderer';
import TranslationLanguageSwitch from './translation/TranslationLanguageSwitch';
import TranslationPendingNotice from './translation/TranslationPendingNotice';
import useTranslatedDocuments from '../hooks/useTranslatedDocuments';
import { parsePaperProofs, resolvePaperMetrics } from '../utils/paperProofs';
import { BRAINSTORM_DISCLAIMER, PAPER_DISCLAIMER } from '../utils/disclaimerHelper';
import './PaperProofViewer.css';

const TRANSLATED_STATUSES = new Set(['complete', 'empty']);
const FRONTEND_DISCLAIMER_PREFIXES = [PAPER_DISCLAIMER, BRAINSTORM_DISCLAIMER]
  .map((disclaimer) => `${disclaimer}\n\n`);

// prependDisclaimer() adds a display-only disclaimer to the content it is given. It is
// interface text, not paper content, so it stays out of translation and the registered
// paper matches what the backend registers for the same document.
function splitFrontendDisclaimer(content) {
  const prefix = FRONTEND_DISCLAIMER_PREFIXES.find((candidate) => content.startsWith(candidate));
  return prefix
    ? { disclaimer: prefix, body: content.slice(prefix.length) }
    : { disclaimer: '', body: content };
}

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
  translationKey,
  translationKind = 'paper',
  translationTitle = '',
  translationEnabled = true,
  translateAllModes = null,
  // Called with (viewerId, { source, language, content }) while a translation is shown and
  // (viewerId, null) otherwise, so downloads beside the viewer save the language shown.
  onTranslationView,
}) {
  const [viewMode, setViewMode] = useState('rendered');
  const viewerId = useId();
  const onTranslationViewRef = useRef(onTranslationView);
  const sourceContent = typeof content === 'string' ? content : String(content ?? '');
  const docKey = translationKey ?? documentId;
  const translationDocKey = docKey == null ? '' : String(docKey);
  const translatable = translationEnabled !== false && translationDocKey !== '';
  const { disclaimer, body } = useMemo(() => splitFrontendDisclaimer(sourceContent), [sourceContent]);
  const translationDocuments = useMemo(() => (translatable ? [{
    docKey: translationDocKey,
    kind: translationKind,
    title: translationTitle,
    profile: 'document',
    parts: { content: body },
  }] : []), [translatable, translationDocKey, translationKind, translationTitle, body]);
  const translation = useTranslatedDocuments(translationDocuments, { enabled: translatable });

  const translationEntry = translatable && translation.active ? translation.byKey[translationDocKey] : null;
  const translatedBody = translationEntry
    && TRANSLATED_STATUSES.has(translationEntry.status)
    && typeof translationEntry.parts?.content === 'string'
    ? translationEntry.parts.content
    : null;
  const isTranslated = translatedBody !== null;
  const displayContent = isTranslated ? `${disclaimer}${translatedBody}` : sourceContent;
  const baseIdentity = documentId == null ? null : String(documentId);
  const viewIdentity = isTranslated
    ? `${baseIdentity ?? translationDocKey}:lang:${translation.language}`
    : baseIdentity;
  const segmentId = (suffix) => (viewIdentity == null ? undefined : `${viewIdentity}:${suffix}`);
  const parsed = useMemo(() => parsePaperProofs(displayContent), [displayContent]);

  useEffect(() => {
    onTranslationViewRef.current = onTranslationView;
  });
  // The reported source and content exclude the display-only disclaimer, which downloads add themselves.
  const viewedLanguage = isTranslated ? translation.language : null;
  useEffect(() => {
    const report = onTranslationViewRef.current;
    if (typeof report !== 'function') {
      return undefined;
    }
    report(viewerId, viewedLanguage ? { source: body, language: viewedLanguage, content: translatedBody } : null);
    return () => report(viewerId, null);
  }, [viewerId, viewedLanguage, body, translatedBody]);

  return (
    <div className={`paper-proof-viewer ${className}`}>
      <div className="paper-proof-viewer-toolbar">
        <div className="paper-proof-viewer-controls">
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
          {translatable && <TranslationLanguageSwitch translation={translation} />}
        </div>
        {showMetrics && (
          // Backend metrics describe the original text; a translation is counted as shown.
          <PaperProofMetrics content={displayContent} metrics={isTranslated ? undefined : metrics} />
        )}
      </div>
      {translatable && <TranslationPendingNotice translation={translation} translateAllModes={translateAllModes} />}

      {viewMode === 'raw' ? (
        <pre className="paper-proof-raw-content">{prefixContent}{displayContent}</pre>
      ) : (
        <div className="paper-proof-rendered-content">
          {prefixContent && (
            <LatexRenderer documentId={segmentId('prefix')} content={prefixContent} showToggle={false} defaultRaw={false} />
          )}
          {parsed.segments.map((segment, index) => (
            segment.type === 'paper' ? (
              segment.content && (
                <LatexRenderer
                  key={`paper-${index}`}
                  documentId={segmentId(`paper:${index}`)}
                  content={segment.content}
                  showToggle={false}
                  defaultRaw={false}
                />
              )
            ) : (
              // Keyed by the untranslated identity so an open proof stays open when its translation arrives.
              <details className="paper-proof-details" key={`${documentId ?? ''}:proof-${segment.proofId || segment.number}-${index}`}>
                <summary>Proof {segment.number}: {segment.theoremName || segment.proofId}</summary>
                <LatexRenderer
                  documentId={segmentId(`proof:${segment.proofId || segment.number}:${index}`)}
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
