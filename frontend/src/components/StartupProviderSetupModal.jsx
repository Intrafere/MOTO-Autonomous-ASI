import React from 'react';
import './settings-common.css';
import './StartupModals.css';
import useOverflowAssist from '../hooks/useOverflowAssist';
import { CLOUD_ACCESS_PROVIDERS } from '../utils/oauthProviders';

const SUBSCRIPTION_PROVIDER_NAMES = CLOUD_ACCESS_PROVIDERS.map((provider) => provider.label).join(', ');

export default function StartupProviderSetupModal({
  isOpen,
  capabilities,
  lmStudioAvailable,
  hasUsableLmStudioChatModel = false,
  hasLmStudioEmbeddingModel = false,
  lmStudioModelCount = 0,
  lmStudioError = '',
  statusMessage = '',
  isCheckingLmStudio = false,
  onChooseOpenRouter,
  onConfirmLmStudio,
}) {
  const {
    scrollRef,
    hasOverflow,
    isAtEnd,
  } = useOverflowAssist(isOpen);

  if (!isOpen) return null;

  const lmStudioEnabled = capabilities?.lmStudioEnabled !== false;
  // Startup only advertises add-ons the backend explicitly reported as available.
  const oauthAvailable = !capabilities?.genericMode
    && CLOUD_ACCESS_PROVIDERS.some((provider) => capabilities?.[provider.capabilityKey] === true);

  return (
    <div
      className="inline-modal-overlay startup-modal-overlay"
      style={{ zIndex: 10000 }}
      onClick={(e) => e.stopPropagation()}
    >
      <div
        className="inline-modal-content startup-modal-shell startup-provider-modal"
        role="dialog"
        aria-modal="true"
        aria-labelledby="startup-provider-title"
      >
        <div className="settings-header-row startup-modal-header">
          <h2 id="startup-provider-title" style={{ margin: 0, color: '#fff' }}>
            Choose Your Startup Setup
          </h2>
        </div>

        <div className="startup-modal-scroll-shell">
          <div
            ref={scrollRef}
            className="startup-modal-scroll-region"
          >
            <div className="startup-modal-scroll-content">
        <p style={{ color: '#ddd', lineHeight: '1.6', marginBottom: '0.9rem' }}>
          {lmStudioEnabled ? (
            <>
              MOTO needs <strong>OpenRouter or a running LM Studio server</strong> before you start.
              Cloud role providers are supplementary because RAG embeddings still need OpenRouter or LM Studio.
            </>
          ) : (
            <>
              This hosted deployment needs <strong>an OpenRouter API key</strong> before you start.
              LM Studio is intentionally disabled here, so all model selection and inference routes through OpenRouter.
            </>
          )}
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
          {lmStudioEnabled ? (
            <>
              <strong>Highly recommended:</strong> install LM Studio even if you plan to use cloud models. LM Studio
              gives MOTO free local embedding/RAG calls and noticeably faster retrieval than OpenRouter embeddings.
            </>
          ) : (
            <>
              <strong>Hosted mode:</strong> after you save your OpenRouter key, MOTO will apply the recommended
              OpenRouter profile immediately. You can fine-tune role models later in the settings screens.
            </>
          )}
        </div>

        <div className="startup-provider-grid">
          <div
            style={{
              padding: '1rem',
              borderRadius: '10px',
              backgroundColor: '#1c1c33',
              border: '1px solid #18cc17',
            }}
          >
            <h3 style={{ marginTop: 0, color: '#18cc17' }}>OpenRouter Setup</h3>
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
                backgroundColor: '#18cc17',
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

          {oauthAvailable && (
            <div
              style={{
                padding: '1rem',
                borderRadius: '10px',
                backgroundColor: '#1c1c33',
                border: '1px solid rgba(197, 109, 45, 0.55)',
                opacity: 0.82,
              }}
            >
              <h3 style={{ marginTop: 0, color: '#ff9f4a' }}>Cloud Provider Add-On</h3>
              <ol style={{ margin: '0 0 1rem 1.1rem', padding: 0, color: '#d7d7e8', lineHeight: '1.55' }}>
                <li>Use a supported subscription provider ({SUBSCRIPTION_PROVIDER_NAMES}) for chat/model roles.</li>
                <li>Set up OpenRouter or LM Studio first so RAG embeddings are available.</li>
                <li>After startup, add cloud provider access from OpenRouter & Cloud Subscriptions and select it in role settings.</li>
              </ol>
              <button
                type="button"
                disabled
                title="OAuth cannot be the only startup provider because RAG embeddings require OpenRouter, LM Studio, or hosted FastEmbed."
                style={{
                  width: '100%',
                  padding: '0.8rem 1rem',
                  backgroundColor: '#47311f',
                  border: '1px solid rgba(255, 159, 74, 0.35)',
                  borderRadius: '8px',
                  color: '#d7b28c',
                  fontSize: '0.95rem',
                  fontWeight: '600',
                  cursor: 'not-allowed',
                }}
              >
                Add After Startup
              </button>
            </div>
          )}

          {lmStudioEnabled && (
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
                backgroundColor: lmStudioAvailable && hasUsableLmStudioChatModel && hasLmStudioEmbeddingModel ? '#1f7a33' : '#21492a',
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
          )}
        </div>

        <div
          style={{
            marginTop: '1rem',
            padding: '0.9rem 1rem',
            borderRadius: '8px',
            backgroundColor: lmStudioEnabled
              ? (lmStudioAvailable ? 'rgba(30, 255, 28, 0.08)' : 'rgba(255, 184, 77, 0.08)')
              : 'rgba(24, 204, 23, 0.12)',
            border: lmStudioEnabled
              ? (lmStudioAvailable ? '1px solid rgba(30, 255, 28, 0.25)' : '1px solid rgba(255, 184, 77, 0.28)')
              : '1px solid rgba(24, 204, 23, 0.3)',
            color: lmStudioEnabled ? (lmStudioAvailable ? '#dbffdd' : '#ffe1ad') : '#dbffdd',
            lineHeight: '1.5',
          }}
        >
          {lmStudioEnabled
            ? (
              lmStudioAvailable && hasUsableLmStudioChatModel && hasLmStudioEmbeddingModel
                ? `LM Studio is currently detected with ${lmStudioModelCount} loaded model${lmStudioModelCount === 1 ? '' : 's'}, including a usable chat model and embedding model.`
                : lmStudioAvailable
                  ? 'LM Studio is running, but you still need both nomic-ai/nomic-embed-text-v1.5 and at least one loaded chat model.'
                  : `LM Studio is not detected yet${lmStudioError ? `: ${lmStudioError}` : '.'}`
            )
            : 'Hosted web mode is active. LM Studio is disabled in this deployment, so OpenRouter is the required provider path.'}
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
          {hasOverflow && !isAtEnd && (
            <div className="startup-modal-overflow-fade" aria-hidden="true" />
          )}
        </div>
        {hasOverflow && !isAtEnd && (
          <div className="startup-modal-more-indicator" aria-hidden="true">
            <span>More below</span>
            <span className="startup-modal-more-arrow">↓</span>
          </div>
        )}
      </div>
    </div>
  );
}
