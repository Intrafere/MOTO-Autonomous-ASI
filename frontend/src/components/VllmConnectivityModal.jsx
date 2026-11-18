import React, { useEffect, useRef, useState } from 'react';
import './settings-common.css';
import SecretInput from './SecretInput';
import { vllmAPI } from '../services/api';

const DEFAULT_SERVER_PLACEHOLDER = 'http://127.0.0.1:8001';
const AUTO_EMBEDDING = '';

export function parseVllmServerList(text) {
  return String(text || '')
    .split(/[\s,]+/)
    .map((entry) => entry.trim())
    .filter(Boolean);
}

function formatContext(contextLength) {
  if (!contextLength) return 'context window not reported';
  return `${Number(contextLength).toLocaleString()} token context`;
}

function ServerResults({ servers }) {
  if (!Array.isArray(servers) || servers.length === 0) return null;
  return (
    <div className="test-result-banner" style={{ maxHeight: '200px', overflow: 'auto' }}>
      {servers.map((server) => (
        <div key={server.url} style={{ marginBottom: '0.5rem' }}>
          <strong>{server.url}</strong>
          {server.models.length === 0 && <div>No models served.</div>}
          {server.models.map((model) => (
            <div key={`${server.url}:${model.id}`}>
              {model.kind === 'embedding' ? 'Embedding model' : 'Chat model'}: <code>{model.id}</code>
              {' '}({formatContext(model.context_length)})
            </div>
          ))}
        </div>
      ))}
    </div>
  );
}

// Setup window for the user's vLLM servers: addresses, optional API key, and the
// embedding model used for RAG. Saving validates the servers first and detects
// each served model and its context window.
export default function VllmConnectivityModal({
  isOpen,
  onClose,
  status,
  capabilities,
  onChanged,
  onRefresh,
}) {
  const [serversText, setServersText] = useState('');
  const [apiKey, setApiKey] = useState('');
  const [clearApiKey, setClearApiKey] = useState(false);
  const [embeddingModel, setEmbeddingModel] = useState(AUTO_EMBEDDING);
  const [saved, setSaved] = useState(null);
  const [detected, setDetected] = useState(null);
  const [busy, setBusy] = useState('');
  const [message, setMessage] = useState({ kind: '', text: '' });
  const [confirmRemove, setConfirmRemove] = useState(false);
  const requestRef = useRef(0);

  const genericMode = Boolean(capabilities?.genericMode);

  useEffect(() => {
    if (!isOpen || genericMode) return undefined;
    const requestId = requestRef.current + 1;
    requestRef.current = requestId;
    setApiKey('');
    setClearApiKey(false);
    setDetected(null);
    setConfirmRemove(false);
    setMessage({ kind: '', text: '' });
    vllmAPI.getStatus()
      .then((result) => {
        if (requestRef.current !== requestId) return;
        const nextSaved = result.status || {};
        setSaved(nextSaved);
        setServersText((nextSaved.servers || []).join('\n'));
        setEmbeddingModel(AUTO_EMBEDDING);
      })
      .catch((error) => {
        if (requestRef.current !== requestId) return;
        setMessage({ kind: 'error', text: error.message || 'vLLM status could not be loaded.' });
      });
    return () => {
      requestRef.current += 1;
    };
  }, [isOpen, genericMode]);

  if (!isOpen) return null;

  const servers = parseVllmServerList(serversText);
  const configured = Boolean(saved?.configured);
  const fromEnvironment = saved?.server_source === 'environment';
  const embeddingOptions = detected?.embedding_models || [];
  const chatModels = Array.isArray(status?.models) ? status.models : [];
  const active = Boolean(status?.available && (status?.has_usable_chat_model || status?.embedding_ready));

  const run = async (action, work) => {
    const requestId = requestRef.current + 1;
    requestRef.current = requestId;
    setBusy(action);
    setMessage({ kind: '', text: '' });
    try {
      await work(() => requestRef.current === requestId);
    } catch (error) {
      if (requestRef.current === requestId) {
        setMessage({ kind: 'error', text: error.message || 'The vLLM request failed.' });
      }
    } finally {
      if (requestRef.current === requestId) {
        setBusy('');
      }
    }
  };

  const requestOptions = () => ({
    servers,
    apiKey: apiKey.trim(),
    embeddingModel,
  });

  const describe = (result) => {
    const chatCount = (result.chat_models || []).length;
    const embedding = result.embedding_model
      ? `Embedding model for RAG: ${result.embedding_model}.`
      : 'No embedding model was found, so RAG embeddings still need LM Studio or OpenRouter.';
    return `${chatCount} chat model${chatCount === 1 ? '' : 's'} detected. ${embedding}`;
  };

  const handleTest = () => run('test', async (isCurrent) => {
    if (servers.length === 0) {
      throw new Error('Enter at least one vLLM server address.');
    }
    const result = await vllmAPI.test(requestOptions());
    if (!isCurrent()) return;
    setDetected(result);
    setMessage({ kind: 'success', text: `Servers reachable. ${describe(result)} Nothing was saved yet.` });
  });

  const handleSave = () => run('save', async (isCurrent) => {
    if (servers.length === 0) {
      throw new Error('Enter at least one vLLM server address.');
    }
    const result = await vllmAPI.saveConfig({ ...requestOptions(), clearApiKey });
    if (!isCurrent()) return;
    setDetected(result);
    setSaved(result.status || { configured: true });
    setServersText((result.status?.servers || servers).join('\n'));
    setApiKey('');
    setClearApiKey(false);
    setMessage({ kind: 'success', text: `vLLM setup saved. ${describe(result)}` });
    onChanged?.(true, result);
  });

  const handleRemove = () => run('remove', async (isCurrent) => {
    await vllmAPI.clear();
    if (!isCurrent()) return;
    setSaved({ configured: false, servers: [] });
    setServersText('');
    setDetected(null);
    setConfirmRemove(false);
    setEmbeddingModel(AUTO_EMBEDDING);
    setMessage({ kind: 'success', text: 'vLLM setup removed.' });
    onChanged?.(false, null);
  });

  return (
    <div
      className="inline-modal-overlay"
      style={{ zIndex: 10000 }}
      onClick={(event) => event.target === event.currentTarget && onClose()}
    >
      <div
        className="inline-modal-content"
        role="dialog"
        aria-modal="true"
        aria-labelledby="vllm-connectivity-title"
        style={{ width: '680px', maxWidth: '92vw', maxHeight: '90vh', overflowY: 'auto', backgroundColor: '#1a1a2e', borderRadius: '12px' }}
      >
        <div className="settings-header-row" style={{ marginBottom: '1.5rem' }}>
          <h2 id="vllm-connectivity-title" style={{ margin: 0, color: '#fff', fontSize: '1.4rem' }}>vLLM Connectivity</h2>
          <button onClick={onClose} className="modal-close-btn" aria-label="Close vLLM connectivity">×</button>
        </div>

        {genericMode ? (
          <div className="test-result-banner test-result-banner--error" style={{ marginBottom: '1rem' }}>
            vLLM is disabled in hosted/generic mode. Use OpenRouter for inference.
          </div>
        ) : (
          <>
            <div
              className={`test-result-banner ${active ? 'test-result-banner--success' : 'test-result-banner--error'}`}
              style={{ marginBottom: '1rem' }}
              role="status"
            >
              {!configured && 'vLLM is not set up yet. Add your server address below.'}
              {configured && active && (
                `vLLM is active with ${chatModels.length} chat model${chatModels.length === 1 ? '' : 's'}`
                + `${status?.embedding_ready ? ' and RAG embeddings' : ''}.`
              )}
              {configured && !active && (status?.error || 'The saved vLLM servers are not answering.')}
            </div>

            <div className="submitter-config-section" style={{ marginBottom: '1rem' }}>
              <h3 style={{ marginTop: 0, color: '#fff', fontSize: '1rem' }}>Servers</h3>
              <label htmlFor="vllm-servers" className="settings-hint" style={{ display: 'block', marginBottom: '0.4rem' }}>
                Server addresses, one per line. A vLLM server usually serves one model, so list every
                server you run (chat models and an embedding model).
              </label>
              <textarea
                id="vllm-servers"
                className="input-dark"
                rows={3}
                value={serversText}
                disabled={Boolean(busy) || fromEnvironment}
                placeholder={DEFAULT_SERVER_PLACEHOLDER}
                onChange={(event) => {
                  setServersText(event.target.value);
                  setDetected(null);
                }}
                spellCheck={false}
                style={{ width: '100%', fontFamily: 'monospace', resize: 'vertical' }}
              />
              {fromEnvironment && (
                <small className="settings-hint">
                  These addresses come from the MOTO_VLLM_BASE_URL environment variable and cannot be edited here.
                </small>
              )}
              <small className="settings-hint" style={{ display: 'block', marginTop: '0.4rem' }}>
                vLLM and MOTO both default to port 8000. Start vLLM on another port, for example
                {' '}<code>vllm serve &lt;model&gt; --port 8001</code>.
              </small>

              <label htmlFor="vllm-api-key" className="settings-hint" style={{ display: 'block', margin: '0.9rem 0 0.4rem' }}>
                API key (optional). Only needed when the server was started with <code>--api-key</code>.
                The key is sent to every listed server.
                {saved?.has_api_key && ' A key is already saved; leave this empty to keep it.'}
              </label>
              <SecretInput
                id="vllm-api-key"
                name="moto-vllm-api-key"
                className="input-dark"
                value={apiKey}
                disabled={Boolean(busy) || clearApiKey}
                placeholder={saved?.has_api_key ? 'Saved key is kept' : 'No key'}
                onChange={(event) => setApiKey(event.target.value)}
                style={{ width: '100%' }}
              />
              {saved?.has_api_key && (
                <label className="settings-hint" style={{ display: 'flex', alignItems: 'center', gap: '0.4rem', marginTop: '0.4rem' }}>
                  <input
                    type="checkbox"
                    checked={clearApiKey}
                    disabled={Boolean(busy)}
                    onChange={(event) => {
                      setClearApiKey(event.target.checked);
                      if (event.target.checked) setApiKey('');
                    }}
                  />
                  Remove the saved API key when saving
                </label>
              )}

              <label htmlFor="vllm-embedding-model" className="settings-hint" style={{ display: 'block', margin: '0.9rem 0 0.4rem' }}>
                Embedding model for RAG
              </label>
              <select
                id="vllm-embedding-model"
                className="input-dark"
                value={embeddingModel}
                disabled={Boolean(busy)}
                onChange={(event) => setEmbeddingModel(event.target.value)}
                style={{ width: '100%' }}
              >
                <option value={AUTO_EMBEDDING}>
                  Automatic{saved?.embedding_model ? ` (currently ${saved.embedding_model})` : ' (first embedding model found)'}
                </option>
                {embeddingOptions.map((model) => (
                  <option key={`${model.server}:${model.id}`} value={model.id}>{model.id}</option>
                ))}
              </select>
              <small className="settings-hint" style={{ display: 'block', marginTop: '0.4rem' }}>
                MOTO finds embedding models by asking each served model for one embedding. Without one,
                RAG embeddings use LM Studio or OpenRouter.
              </small>

              <div style={{ display: 'flex', gap: '0.6rem', flexWrap: 'wrap', marginTop: '1rem' }}>
                <button type="button" className="btn-ghost" onClick={handleTest} disabled={Boolean(busy)}>
                  {busy === 'test' ? 'Checking...' : 'Test Servers'}
                </button>
                <button type="button" className="btn-success-sm" onClick={handleSave} disabled={Boolean(busy)}>
                  {busy === 'save' ? 'Saving...' : 'Save vLLM Setup'}
                </button>
                {configured && !confirmRemove && (
                  <button type="button" className="btn-ghost" onClick={() => setConfirmRemove(true)} disabled={Boolean(busy)}>
                    Remove vLLM Setup
                  </button>
                )}
                {configured && confirmRemove && (
                  <>
                    <button type="button" className="btn-danger" onClick={handleRemove} disabled={Boolean(busy)}>
                      {busy === 'remove' ? 'Removing...' : 'Confirm Remove'}
                    </button>
                    <button type="button" className="btn-ghost" onClick={() => setConfirmRemove(false)} disabled={Boolean(busy)}>
                      Keep
                    </button>
                  </>
                )}
              </div>

              {message.text && (
                <div
                  className={`test-result-banner ${message.kind === 'error' ? 'test-result-banner--error' : 'test-result-banner--success'}`}
                  style={{ marginTop: '1rem' }}
                  role={message.kind === 'error' ? 'alert' : 'status'}
                >
                  {message.text}
                </div>
              )}
              {detected && (
                <div style={{ marginTop: '0.8rem' }}>
                  <ServerResults servers={detected.servers} />
                </div>
              )}
            </div>

            <div className="submitter-config-section">
              <h3 style={{ marginTop: 0, color: '#fff', fontSize: '1rem' }}>Readiness</h3>
              <ul className="settings-hint" style={{ lineHeight: 1.7 }}>
                <li>Chat models: {chatModels.length > 0 ? chatModels.join(', ') : 'not detected'}</li>
                <li>
                  Embedding model: {status?.embedding_ready
                    ? `ready (${status.embedding_model})`
                    : (status?.embedding_model ? `${status.embedding_model} is not answering` : 'not set up')}
                </li>
              </ul>
              <p className="settings-hint">
                After saving, choose <strong>vLLM</strong> as the provider for any role in Settings. Each
                model&apos;s context window is read from the server and filled in for you.
                See <a href="https://docs.vllm.ai/" target="_blank" rel="noopener noreferrer" style={{ color: '#8ab4ff' }}>docs.vllm.ai</a> for
                serving chat and embedding models.
              </p>
              <button type="button" onClick={onRefresh} className="btn-ghost" disabled={Boolean(busy)}>
                Refresh vLLM Status
              </button>
            </div>
          </>
        )}
      </div>
    </div>
  );
}
