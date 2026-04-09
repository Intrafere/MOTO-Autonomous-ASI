import React from 'react';
import './settings-common.css';

export default function StartupProviderSetupModal({
  isOpen,
  lmStudioAvailable,
  hasUsableLmStudioChatModel = false,
  lmStudioModelCount = 0,
  lmStudioError = '',
  statusMessage = '',
  isCheckingLmStudio = false,
  onChooseOpenRouter,
  onConfirmLmStudio,
}) {
  if (!isOpen) return null;

  return (
    <div
      className="inline-modal-overlay"
      style={{ zIndex: 10000 }}
      onClick={(e) => e.stopPropagation()}
    >
      <div
        className="inline-modal-content"
        style={{
          width: '760px',
          maxWidth: '92vw',
          backgroundColor: '#141426',
          borderRadius: '14px',
        }}
      >
        <div className="settings-header-row" style={{ marginBottom: '1rem' }}>
          <h2 style={{ margin: 0, color: '#fff', fontSize: '1.45rem' }}>
            Choose Your Startup Setup
          </h2>
        </div>

        <p style={{ color: '#ddd', lineHeight: '1.6', marginBottom: '0.9rem' }}>
          MOTO needs <strong>an OpenRouter API key or a running LM Studio server</strong> before you start.
          The best experience is to use both: OpenRouter for cloud models and LM Studio for free, faster local RAG and embeddings.
        </p>

        <div
          style={{
            marginBottom: '1rem',
            padding: '0.9rem 1rem',
            borderRadius: '8px',
            backgroundColor: 'rgba(30, 255, 28, 0.08)',
            border: '1px solid rgba(30, 255, 28, 0.25)',
            color: '#d8ffd8',
            lineHeight: '1.55',
          }}
        >
          <strong>Highly recommended:</strong> install LM Studio even if you plan to use OpenRouter. LM Studio
          gives MOTO free local embedding/RAG calls and noticeably faster retrieval than OpenRouter embeddings.
        </div>

        <div style={{ display: 'grid', gap: '1rem', gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))' }}>
          <div
            style={{
              padding: '1rem',
              borderRadius: '10px',
              backgroundColor: '#1c1c33',
              border: '1px solid #34345c',
            }}
          >
            <h3 style={{ marginTop: 0, color: '#a29bfe' }}>OpenRouter Setup</h3>
            <ol style={{ margin: '0 0 1rem 1.1rem', padding: 0, color: '#d7d7e8', lineHeight: '1.55' }}>
              <li>Create or sign in to your account at <a href="https://openrouter.ai/" target="_blank" rel="noopener noreferrer" style={{ color: '#8ab4ff' }}>openrouter.ai</a>.</li>
              <li>Generate an API key at <a href="https://openrouter.ai/keys" target="_blank" rel="noopener noreferrer" style={{ color: '#8ab4ff' }}>openrouter.ai/keys</a>.</li>
              <li>Paste that key into MOTO. The recommended default profile will be applied right away.</li>
            </ol>
            <button
              type="button"
              onClick={onChooseOpenRouter}
              style={{
                width: '100%',
                padding: '0.8rem 1rem',
                backgroundColor: '#6c5ce7',
                border: 'none',
                borderRadius: '8px',
                color: '#fff',
                fontSize: '0.95rem',
                fontWeight: '600',
                cursor: 'pointer',
              }}
            >
              Enter OpenRouter Key
            </button>
          </div>

          <div
            style={{
              padding: '1rem',
              borderRadius: '10px',
              backgroundColor: '#1c1c33',
              border: '1px solid #2f5c36',
            }}
          >
            <h3 style={{ marginTop: 0, color: '#7CFC90' }}>LM Studio Setup</h3>
            <ol style={{ margin: '0 0 1rem 1.1rem', padding: 0, color: '#d7d7e8', lineHeight: '1.55' }}>
              <li>Install LM Studio from <a href="https://lmstudio.ai/" target="_blank" rel="noopener noreferrer" style={{ color: '#8ab4ff' }}>lmstudio.ai</a>.</li>
              <li>Enable Developer or Power User mode if needed, then open the server tab.</li>
              <li>Load the embedding model <code>nomic-ai/nomic-embed-text-v1.5</code>.</li>
              <li>Optionally load one or more local chat models, then start the local server on <code>http://127.0.0.1:1234</code>.</li>
            </ol>
            <button
              type="button"
              onClick={onConfirmLmStudio}
              disabled={isCheckingLmStudio}
              style={{
                width: '100%',
                padding: '0.8rem 1rem',
                backgroundColor: lmStudioAvailable && hasUsableLmStudioChatModel ? '#1f7a33' : '#21492a',
                border: '1px solid #2f8f45',
                borderRadius: '8px',
                color: '#fff',
                fontSize: '0.95rem',
                fontWeight: '600',
                cursor: isCheckingLmStudio ? 'not-allowed' : 'pointer',
                opacity: isCheckingLmStudio ? 0.7 : 1,
              }}
            >
              {isCheckingLmStudio ? 'Checking LM Studio...' : "I'm Running LM Studio"}
            </button>
          </div>
        </div>

        <div
          style={{
            marginTop: '1rem',
            padding: '0.9rem 1rem',
            borderRadius: '8px',
            backgroundColor: lmStudioAvailable ? 'rgba(30, 255, 28, 0.08)' : 'rgba(255, 184, 77, 0.08)',
            border: lmStudioAvailable ? '1px solid rgba(30, 255, 28, 0.25)' : '1px solid rgba(255, 184, 77, 0.28)',
            color: lmStudioAvailable ? '#dbffdd' : '#ffe1ad',
            lineHeight: '1.5',
          }}
        >
          {lmStudioAvailable && hasUsableLmStudioChatModel
            ? `LM Studio is currently detected with ${lmStudioModelCount} loaded model${lmStudioModelCount === 1 ? '' : 's'}, including a usable chat model.`
            : lmStudioAvailable
              ? 'LM Studio is running, but you still need at least one loaded chat model in addition to embeddings.'
              : `LM Studio is not detected yet${lmStudioError ? `: ${lmStudioError}` : '.'}`}
        </div>

        {statusMessage && (
          <div
            className="test-result-banner test-result-banner--error"
            style={{ marginTop: '1rem' }}
          >
            {statusMessage}
          </div>
        )}

        <p style={{ color: '#aaa', lineHeight: '1.55', marginTop: '1rem', marginBottom: 0 }}>
          After setup, open <strong>Autonomous Model Selection &amp; Settings</strong> to pick your saved team
          profile or switch to any built-in default profile.
        </p>
      </div>
    </div>
  );
}
