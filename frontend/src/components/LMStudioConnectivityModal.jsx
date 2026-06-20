import React from 'react';
import './settings-common.css';

export default function LMStudioConnectivityModal({
  isOpen,
  onClose,
  status,
  capabilities,
  onRefresh,
}) {
  if (!isOpen) return null;

  const genericMode = Boolean(capabilities?.genericMode);
  const available = Boolean(status?.available && status?.has_models);
  const models = Array.isArray(status?.models) ? status.models : [];

  return (
    <div className="inline-modal-overlay" style={{ zIndex: 10000 }} onClick={(event) => event.target === event.currentTarget && onClose()}>
      <div className="inline-modal-content" style={{ width: '600px', maxWidth: '90vw', backgroundColor: '#1a1a2e', borderRadius: '12px' }}>
        <div className="settings-header-row" style={{ marginBottom: '1.5rem' }}>
          <h2 style={{ margin: 0, color: '#fff', fontSize: '1.4rem' }}>LM Studio Connectivity</h2>
          <button onClick={onClose} className="modal-close-btn" aria-label="Close LM Studio connectivity">×</button>
        </div>

        {genericMode ? (
          <div className="test-result-banner test-result-banner--error" style={{ marginBottom: '1rem' }}>
            LM Studio is disabled in hosted/generic mode. Use OpenRouter/OAuth for inference.
          </div>
        ) : (
          <>
            <div className={`test-result-banner ${available ? 'test-result-banner--success' : 'test-result-banner--error'}`} style={{ marginBottom: '1rem' }}>
              {available
                ? `LM Studio is active with ${status.model_count || models.length || 0} loaded model${(status.model_count || models.length || 0) === 1 ? '' : 's'}.`
                : (status?.error || 'LM Studio is not reachable or has no loaded chat model.')}
            </div>

            <div className="submitter-config-section" style={{ marginBottom: '1rem' }}>
              <h3 style={{ marginTop: 0, color: '#fff', fontSize: '1rem' }}>Readiness</h3>
              <p className="settings-hint">Default server: <code>http://127.0.0.1:1234</code></p>
              <ul className="settings-hint" style={{ lineHeight: 1.7 }}>
                <li>Chat model: {status?.has_usable_chat_model ? `ready (${status.usable_chat_model_id})` : 'not detected'}</li>
                <li>Embedding model: {status?.has_embedding_model ? 'ready' : 'not detected'}</li>
                <li>Loaded models: {status?.model_count || models.length || 0}</li>
              </ul>
              {models.length > 0 && (
                <div className="test-result-banner" style={{ maxHeight: '140px', overflow: 'auto' }}>
                  {models.map((model) => <div key={model}>{model}</div>)}
                </div>
              )}
            </div>

            <div className="submitter-config-section">
              <h3 style={{ marginTop: 0, color: '#fff', fontSize: '1rem' }}>Setup</h3>
              <p className="settings-hint">
                Open LM Studio, load at least one chat model, and load <code>nomic-ai/nomic-embed-text-v1.5</code> for local embeddings.
              </p>
              <button onClick={onRefresh} className="btn-ghost">Refresh LM Studio Status</button>
            </div>
          </>
        )}
      </div>
    </div>
  );
}

