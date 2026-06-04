import React, { useState, useEffect } from 'react';
import { cloudAccessAPI, openRouterAPI } from '../services/api';
import './settings-common.css';

/**
 * Modal for configuring cloud provider access.
 * 
 * Shows when:
 * 1. User clicks the Cloud Access & Keys header chip
 * 2. User clicks "Use OpenRouter" on any role but no API key is configured
 * 3. LM Studio is unavailable and user needs cloud access as primary provider
 */
export default function OpenRouterApiKeyModal({
  isOpen,
  onClose,
  onKeySet,
  onCloudAccessChanged,
  reason = 'setup',
  capabilities,
}) {
  const [apiKey, setApiKey] = useState('');
  const [testing, setTesting] = useState(false);
  const [saving, setSaving] = useState(false);
  const [testResult, setTestResult] = useState(null);
  const [error, setError] = useState('');
  const [hasStoredKey, setHasStoredKey] = useState(false);
  const [codexStatus, setCodexStatus] = useState({ configured: false });
  const [codexLoading, setCodexLoading] = useState(false);
  const [codexState, setCodexState] = useState('');
  const [codexRedirectUri, setCodexRedirectUri] = useState('');
  const [codexCallbackInput, setCodexCallbackInput] = useState('');
  const [codexMessage, setCodexMessage] = useState('');
  const [codexLoginBaselineConfigured, setCodexLoginBaselineConfigured] = useState(false);
  const [codexLoginBaselineUpdatedAt, setCodexLoginBaselineUpdatedAt] = useState(0);
  const [codexModelsStatus, setCodexModelsStatus] = useState({
    checking: false,
    count: null,
    error: '',
  });
  const codexModelCheckRequestRef = React.useRef(0);
  const genericMode = Boolean(capabilities?.genericMode);
  const lmStudioEnabled = capabilities?.lmStudioEnabled !== false;

  const verifyCodexModels = async () => {
    if (genericMode) {
      codexModelCheckRequestRef.current += 1;
      setCodexModelsStatus({ checking: false, count: null, error: '' });
      return false;
    }

    const requestId = codexModelCheckRequestRef.current + 1;
    codexModelCheckRequestRef.current = requestId;
    setCodexModelsStatus({ checking: true, count: null, error: '' });
    try {
      const result = await cloudAccessAPI.getOpenAICodexModels();
      if (codexModelCheckRequestRef.current !== requestId) {
        return null;
      }
      const models = Array.isArray(result.models) ? result.models : [];
      if (models.length === 0) {
        setCodexModelsStatus({
          checking: false,
          count: 0,
          error: 'Codex login is saved, but no Codex models were returned. Reconnect OAuth or check that this ChatGPT account has Codex model access.',
        });
        return false;
      }
      setCodexModelsStatus({ checking: false, count: models.length, error: '' });
      return true;
    } catch (err) {
      if (codexModelCheckRequestRef.current !== requestId) {
        return null;
      }
      setCodexModelsStatus({
        checking: false,
        count: null,
        error: `Codex login is saved, but models could not be loaded: ${err.message || 'unknown error'}. Reconnect OAuth or check this account's Codex access.`,
      });
      return false;
    }
  };

  // Reset state when modal opens
  useEffect(() => {
    if (isOpen) {
      setApiKey('');
      setTestResult(null);
      setError('');
      setCodexMessage('');
      setCodexLoginBaselineConfigured(false);
      setCodexLoginBaselineUpdatedAt(0);
      codexModelCheckRequestRef.current += 1;
      setCodexModelsStatus({ checking: false, count: null, error: '' });
      let isCancelled = false;

      const loadKeyStatus = async () => {
        try {
          const status = await openRouterAPI.getApiKeyStatus();
          if (!isCancelled) {
            setHasStoredKey(Boolean(status.has_key));
          }
        } catch {
          if (!isCancelled) {
            setHasStoredKey(false);
          }
        }
      };
      const loadCloudStatus = async () => {
        try {
          const status = await cloudAccessAPI.getOpenAICodexStatus();
          if (!isCancelled) {
            const nextStatus = status.status || { configured: false };
            setCodexStatus(nextStatus);
            if (nextStatus.configured) {
              verifyCodexModels();
            }
          }
        } catch {
          if (!isCancelled) {
            setCodexStatus({ configured: false });
            setCodexModelsStatus({ checking: false, count: null, error: '' });
          }
        }
      };

      loadKeyStatus();
      loadCloudStatus();

      return () => {
        isCancelled = true;
        codexModelCheckRequestRef.current += 1;
      };
    }
    codexModelCheckRequestRef.current += 1;
    setHasStoredKey(false);
    return undefined;
  }, [isOpen]);

  useEffect(() => {
    if (!isOpen || !codexState) return undefined;
    const interval = window.setInterval(async () => {
      try {
        const status = await cloudAccessAPI.getOpenAICodexStatus();
        const nextStatus = status.status || { configured: false };
        setCodexStatus(nextStatus);
        const statusUpdatedAt = Number(nextStatus.updated_at || 0);
        const loginCompleted = nextStatus.configured
          && (
            !codexLoginBaselineConfigured
            || (statusUpdatedAt > 0 && statusUpdatedAt > codexLoginBaselineUpdatedAt)
          );
        if (loginCompleted) {
          const modelsReady = await verifyCodexModels();
          if (modelsReady === null) {
            return;
          }
          setCodexState('');
          setCodexCallbackInput('');
          setCodexMessage(modelsReady
            ? 'OpenAI Codex login saved and model list loaded.'
            : 'OpenAI Codex login saved, but model loading needs attention.'
          );
          if (onCloudAccessChanged) {
            onCloudAccessChanged(true, 'openai_codex_oauth', { modelsReady });
          }
        }
      } catch {
        // Keep waiting; manual paste remains available.
      }
    }, 2000);
    return () => window.clearInterval(interval);
  }, [isOpen, codexState, codexLoginBaselineConfigured, codexLoginBaselineUpdatedAt, onCloudAccessChanged]);

  const handleTestConnection = async () => {
    if (!apiKey.trim()) {
      setError('Please enter an API key');
      return;
    }

    setTesting(true);
    setError('');
    setTestResult(null);

    try {
      const result = await openRouterAPI.testConnection(apiKey.trim());
      setTestResult(result);
      if (!result.connected) {
        setError(result.message || 'Connection failed');
      }
    } catch (err) {
      setError(err.message || 'Failed to test connection');
      setTestResult({ connected: false });
    } finally {
      setTesting(false);
    }
  };

  const handleSaveKey = async () => {
    if (!apiKey.trim()) {
      setError('Please enter an API key');
      return;
    }

    setSaving(true);
    setError('');

    try {
      // Save to backend
      await openRouterAPI.setApiKey(apiKey.trim());
      setHasStoredKey(true);

      // Notify parent
      if (onKeySet) {
        await onKeySet(apiKey.trim());
      }

      onClose();
    } catch (err) {
      setError(err.message || 'Failed to save API key');
    } finally {
      setSaving(false);
    }
  };

  const handleClearKey = async () => {
    try {
      await openRouterAPI.clearApiKey();
      setApiKey('');
      setTestResult(null);
      setError('');
      setHasStoredKey(false);
    } catch (err) {
      setError(err.message || 'Failed to clear API key');
    }
  };

  const handleStartCodexLogin = async () => {
    setCodexLoading(true);
    setCodexMessage('');
    setError('');
    setCodexLoginBaselineConfigured(Boolean(codexStatus?.configured));
    setCodexLoginBaselineUpdatedAt(Number(codexStatus?.updated_at || 0));
    try {
      const result = await cloudAccessAPI.startOpenAICodexLogin();
      setCodexState(result.state || '');
      setCodexRedirectUri(result.redirect_uri || '');
      if (result.authorization_url) {
        window.open(result.authorization_url, '_blank', 'noopener,noreferrer');
      }
      setCodexMessage(result.callback_available
        ? 'OpenAI login opened. MOTO will capture the callback automatically; paste the callback URL or code below if the browser cannot return to MOTO.'
        : 'OpenAI login opened. The local callback port is unavailable, so paste the full callback URL or authorization code below after sign-in.'
      );
    } catch (err) {
      setError(err.message || 'Failed to start OpenAI Codex login');
    } finally {
      setCodexLoading(false);
    }
  };

  const handleCompleteCodexLogin = async () => {
    if (!codexCallbackInput.trim()) {
      setError('Paste the OpenAI callback URL or authorization code first');
      return;
    }
    setCodexLoading(true);
    setCodexMessage('');
    setError('');
    try {
      const isUrl = /^https?:\/\//i.test(codexCallbackInput.trim());
      const result = await cloudAccessAPI.exchangeOpenAICodexCode({
        code: isUrl ? '' : codexCallbackInput.trim(),
        redirectUrl: isUrl ? codexCallbackInput.trim() : '',
        state: codexState,
        redirectUri: codexRedirectUri || null,
      });
      setCodexStatus(result.status || { configured: true });
      const modelsReady = await verifyCodexModels();
      if (modelsReady === null) {
        return;
      }
      setCodexCallbackInput('');
      setCodexState('');
      setCodexMessage(modelsReady
        ? 'OpenAI Codex login saved and model list loaded.'
        : 'OpenAI Codex login saved, but model loading needs attention.'
      );
      if (onCloudAccessChanged) {
        onCloudAccessChanged(true, 'openai_codex_oauth', { modelsReady });
      }
    } catch (err) {
      setError(err.message || 'Failed to complete OpenAI Codex login');
    } finally {
      setCodexLoading(false);
    }
  };

  const handleClearCodexLogin = async () => {
    setCodexLoading(true);
    setCodexMessage('');
    setError('');
    codexModelCheckRequestRef.current += 1;
    try {
      await cloudAccessAPI.clearOpenAICodexLogin();
      setCodexStatus({ configured: false });
      setCodexCallbackInput('');
      setCodexState('');
      setCodexLoginBaselineConfigured(false);
      setCodexLoginBaselineUpdatedAt(0);
      setCodexModelsStatus({ checking: false, count: null, error: '' });
      setCodexMessage('OpenAI Codex login cleared.');
      if (onCloudAccessChanged) {
        onCloudAccessChanged(false, 'openai_codex_oauth');
      }
    } catch (err) {
      setError(err.message || 'Failed to clear OpenAI Codex login');
    } finally {
      setCodexLoading(false);
    }
  };

  if (!isOpen) return null;

  const reasonMessages = {
    setup: 'Configure cloud model access for MOTO roles.',
    startup_setup: 'Save cloud access credentials to unlock cloud models. MOTO will apply the recommended default profile immediately, and you can switch profiles later in Settings.',
    startup_codex_oauth: 'Sign in with OpenAI Codex OAuth to use subscription-backed Codex models. MOTO will apply Codex-backed startup defaults immediately after login.',
    lm_studio_unavailable: lmStudioEnabled
      ? 'LM Studio is not available. Configure cloud access to continue.'
      : 'This deployment disables LM Studio. Configure cloud access to continue.',
    no_key: 'An OpenRouter API key is required to use OpenRouter models.',
  };
  const storedKeyCopy = genericMode
    ? 'An OpenRouter API key is already loaded in this running backend instance. Enter a new key below to replace it for this session.'
    : 'An OpenRouter API key is already stored securely on the backend for this machine. Enter a new key below to replace it.';
  const keyStorageFooter = genericMode
    ? 'This API key is held in backend memory for the active hosted/runtime instance and sent to the backend for OpenRouter API calls. API Boost can reuse this key automatically, or you can override it inside the boost modal.'
    : 'This API key is stored securely through the backend keyring integration and sent to the backend for OpenRouter API calls. API Boost can reuse this key automatically, or you can override it inside the boost modal.';

  return (
    <div 
      className="inline-modal-overlay"
      style={{
        zIndex: 10000,
      }}
      onClick={(e) => e.target === e.currentTarget && onClose()}
    >
      <div 
        className="inline-modal-content"
        style={{
          width: '640px',
          maxWidth: '90vw',
          backgroundColor: '#1a1a2e',
          borderRadius: '12px',
        }}
      >
        <div className="settings-header-row" style={{ marginBottom: '1.5rem' }}>
          <h2 style={{ margin: 0, color: '#fff', fontSize: '1.4rem' }}>
            Cloud Access & Keys
          </h2>
          <button
            onClick={onClose}
            style={{
              background: 'none',
              border: 'none',
              color: '#888',
              fontSize: '1.5rem',
              cursor: 'pointer',
              padding: '0.25rem',
            }}
          >
            ×
          </button>
        </div>

        <p style={{ color: '#aaa', marginBottom: '1.5rem', fontSize: '0.95rem' }}>
          {reasonMessages[reason] || reasonMessages.setup}
        </p>

        <div className="submitter-config-section" style={{ marginBottom: '1rem' }}>
          <h3 style={{ marginTop: 0, color: '#fff', fontSize: '1rem' }}>OpenRouter API Key</h3>
          <p className="settings-hint">
            Use OpenRouter models across MOTO roles. This is the existing cloud-provider path and can still be reused by API Boost.
          </p>
          <div style={{ marginBottom: '1rem' }}>
            <label style={{ display: 'block', color: '#ccc', marginBottom: '0.5rem', fontSize: '0.9rem' }}>
              API Key
            </label>
            <input
              type="password"
              value={apiKey}
              onChange={(e) => setApiKey(e.target.value)}
              placeholder="sk-or-v1-..."
              className="input-dark"
              style={{
                fontSize: '0.95rem',
              }}
            />
            <small className="hint-text hint-text--dim">
              Get your API key at{' '}
              <a
                href="https://openrouter.ai/keys"
                target="_blank"
                rel="noopener noreferrer"
                style={{ color: '#18cc17' }}
              >
                openrouter.ai/keys
              </a>
            </small>
          </div>

          {testResult && testResult.connected && (
            <div className="test-result-banner test-result-banner--success" style={{
              marginBottom: '1rem',
            }}>
              Connection successful! {testResult.model_count} models available.
            </div>
          )}

          {hasStoredKey && !apiKey.trim() && (
            <div className="test-result-banner test-result-banner--success" style={{
              marginBottom: '1rem',
            }}>
              {storedKeyCopy}
            </div>
          )}

          <div style={{ display: 'flex', gap: '0.75rem', marginTop: '1.5rem' }}>
            <button
              onClick={handleTestConnection}
              disabled={testing || !apiKey.trim()}
              style={{
                flex: 1,
                padding: '0.75rem 1rem',
                backgroundColor: '#333',
                border: '1px solid #444',
                borderRadius: '6px',
                color: '#fff',
                cursor: testing || !apiKey.trim() ? 'not-allowed' : 'pointer',
                opacity: testing || !apiKey.trim() ? 0.6 : 1,
                fontSize: '0.95rem',
              }}
            >
              {testing ? 'Testing...' : 'Test Connection'}
            </button>

            <button
              onClick={handleSaveKey}
              disabled={saving || !apiKey.trim()}
              style={{
                flex: 1,
                padding: '0.75rem 1rem',
                backgroundColor: '#18cc17',
                border: 'none',
                borderRadius: '6px',
                color: '#fff',
                cursor: saving || !apiKey.trim() ? 'not-allowed' : 'pointer',
                opacity: saving || !apiKey.trim() ? 0.6 : 1,
                fontSize: '0.95rem',
                fontWeight: '500',
              }}
            >
              {saving ? 'Saving...' : 'Save API Key'}
            </button>
          </div>

          {(apiKey || hasStoredKey) && (
            <button
              onClick={handleClearKey}
              className="btn-ghost"
              style={{
                width: '100%',
                marginTop: '1rem',
                fontSize: '0.85rem',
              }}
            >
              Clear Stored API Key
            </button>
          )}
        </div>

        <div className="submitter-config-section" style={{ marginBottom: '1rem' }}>
          <h3 style={{ marginTop: 0, color: '#fff', fontSize: '1rem' }}>OpenAI Codex Login (ChatGPT Subscription)</h3>
          <p className="settings-hint">
            Sign in with OpenAI Codex OAuth for subscription-backed Codex models. This is separate from regular OpenAI API-key billing.
          </p>
          {genericMode ? (
            <div className="test-result-banner test-result-banner--error" style={{ marginBottom: '1rem' }}>
              OpenAI Codex login is desktop-only until hosted callback/proxy support is designed.
            </div>
          ) : codexStatus?.configured ? (
            <div
              className={`test-result-banner ${
                codexModelsStatus.error
                  ? 'test-result-banner--error'
                  : (codexModelsStatus.checking ? '' : 'test-result-banner--success')
              }`}
              style={{ marginBottom: '1rem' }}
            >
              {codexModelsStatus.error
                ? `OpenAI Codex OAuth token saved${codexStatus.email ? ` for ${codexStatus.email}` : ''}, but model access is not verified.`
                : `OpenAI Codex login configured${codexStatus.email ? ` for ${codexStatus.email}` : ''}.`}
            </div>
          ) : (
            <div className="test-result-banner" style={{ marginBottom: '1rem' }}>
              OpenAI Codex login is not configured.
            </div>
          )}

          {codexModelsStatus.checking && (
            <div className="test-result-banner" style={{ marginBottom: '1rem' }}>
              Checking Codex model list...
            </div>
          )}

          {!codexModelsStatus.checking && codexModelsStatus.count > 0 && (
            <div className="test-result-banner test-result-banner--success" style={{ marginBottom: '1rem' }}>
              Codex model list loaded. {codexModelsStatus.count} model{codexModelsStatus.count === 1 ? '' : 's'} available.
            </div>
          )}

          {!codexModelsStatus.checking && codexModelsStatus.error && (
            <div className="test-result-banner test-result-banner--error" style={{ marginBottom: '1rem' }}>
              {codexModelsStatus.error}
            </div>
          )}

          {codexMessage && (
            <div
              className={`test-result-banner ${codexMessage.includes('needs attention') ? 'test-result-banner--error' : 'test-result-banner--success'}`}
              style={{ marginBottom: '1rem' }}
            >
              {codexMessage}
            </div>
          )}

          <div style={{ display: 'flex', gap: '0.75rem', marginTop: '1rem' }}>
            <button
              onClick={handleStartCodexLogin}
              disabled={codexLoading || genericMode}
              style={{
                flex: 1,
                padding: '0.75rem 1rem',
                backgroundColor: '#333',
                border: '1px solid #444',
                borderRadius: '6px',
                color: '#fff',
                cursor: codexLoading || genericMode ? 'not-allowed' : 'pointer',
                opacity: codexLoading || genericMode ? 0.6 : 1,
                fontSize: '0.95rem',
              }}
            >
              {codexLoading ? 'Working...' : 'Start OpenAI Login'}
            </button>
            {codexStatus?.configured && (
              <button
                onClick={handleClearCodexLogin}
                disabled={codexLoading}
                className="btn-ghost"
                style={{ flex: 1 }}
              >
                Clear Codex Login
              </button>
            )}
          </div>

          {codexState && (
            <div style={{ marginTop: '1rem' }}>
              <label style={{ display: 'block', color: '#ccc', marginBottom: '0.5rem', fontSize: '0.9rem' }}>
                Callback URL or Authorization Code
              </label>
              <input
                type="password"
                value={codexCallbackInput}
                onChange={(e) => setCodexCallbackInput(e.target.value)}
                placeholder="Paste callback URL or code from OpenAI login"
                className="input-dark"
                style={{ fontSize: '0.95rem' }}
              />
              <button
                onClick={handleCompleteCodexLogin}
                disabled={codexLoading || !codexCallbackInput.trim()}
                style={{
                  width: '100%',
                  marginTop: '0.75rem',
                  padding: '0.75rem 1rem',
                  backgroundColor: '#18cc17',
                  border: 'none',
                  borderRadius: '6px',
                  color: '#fff',
                  cursor: codexLoading || !codexCallbackInput.trim() ? 'not-allowed' : 'pointer',
                  opacity: codexLoading || !codexCallbackInput.trim() ? 0.6 : 1,
                  fontSize: '0.95rem',
                  fontWeight: '500',
                }}
              >
                Complete OpenAI Login
              </button>
            </div>
          )}
        </div>

        {/* Error Message */}
        {error && (
          <div className="test-result-banner test-result-banner--error" style={{
            marginBottom: '1rem',
          }}>
            {error}
          </div>
        )}

        {/* Info Note */}
        <p style={{ 
          color: '#666', 
          fontSize: '0.8rem', 
          marginTop: '1.5rem',
          padding: '0.75rem',
          backgroundColor: '#0d0d1a',
          borderRadius: '6px',
        }}>
          {keyStorageFooter}
        </p>
      </div>
    </div>
  );
}

