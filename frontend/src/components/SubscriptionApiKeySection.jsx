import React, { useCallback, useEffect, useRef, useState } from 'react';
import { cloudAccessAPI } from '../services/api';
import { getCloudAccessProvider } from '../utils/oauthProviders';
import SecretInput from './SecretInput';

function formatResetTime(resetAt) {
  if (!resetAt) return 'reset time not reported';
  const date = new Date(resetAt * 1000);
  return Number.isNaN(date.getTime()) ? 'reset time not reported' : `resets ${date.toLocaleString()}`;
}

function UsagePools({ usage }) {
  const pools = Array.isArray(usage?.pools) ? usage.pools : [];
  if (pools.length === 0) {
    return <small className="hint-text hint-text--dim">The provider did not report quota pools.</small>;
  }
  return (
    <ul className="subscription-usage-list" style={{ listStyle: 'none', padding: 0, margin: '0.5rem 0 0' }}>
      {pools.map((pool) => {
        const ratio = typeof pool.used_ratio === 'number' ? Math.max(0, Math.min(1, pool.used_ratio)) : null;
        return (
          <li key={pool.id} style={{ marginBottom: '0.5rem' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', color: '#ccc', fontSize: '0.85rem' }}>
              <span>{pool.label || pool.id}</span>
              <span>{ratio === null ? 'usage unknown' : `${Math.round(ratio * 100)}% used`}</span>
            </div>
            {ratio !== null && (
              <div
                role="progressbar"
                aria-label={`${pool.label || pool.id} quota used`}
                aria-valuemin={0}
                aria-valuemax={100}
                aria-valuenow={Math.round(ratio * 100)}
                style={{ height: '6px', background: '#333', borderRadius: '3px', overflow: 'hidden' }}
              >
                <div style={{ width: `${ratio * 100}%`, height: '100%', background: ratio >= 0.9 ? '#ff6b6b' : '#18cc17' }} />
              </div>
            )}
            <small className="hint-text hint-text--dim">{formatResetTime(pool.reset_at)}</small>
          </li>
        );
      })}
    </ul>
  );
}

// Save/remove UI for one subscription API-key provider in the OpenRouter/OAuth modal.
// Status and model verification are owned by the modal so every provider's
// "(configured)" state is known before its section is opened.
export default function SubscriptionApiKeySection({
  providerId,
  genericMode,
  status,
  modelsStatus,
  verifyModels,
  onStatusChange,
  onModelsStatusChange,
  onCloudAccessChanged,
  onError,
  successBannerStyle,
  getMessageBannerClass,
}) {
  const provider = getCloudAccessProvider(providerId);
  const apiKeyConfig = provider?.apiKey || {};
  const [apiKey, setApiKey] = useState('');
  const [loading, setLoading] = useState(false);
  const [message, setMessage] = useState('');
  const [usage, setUsage] = useState({ loading: false, data: null, error: '' });
  const usageRequestRef = useRef(0);
  const configured = Boolean(status?.configured);
  const success = Boolean(
    !genericMode
    && configured
    && !modelsStatus.checking
    && modelsStatus.count > 0
    && !modelsStatus.error
  );

  const loadUsage = useCallback(async () => {
    if (!apiKeyConfig.supportsUsage || genericMode) return;
    const requestId = usageRequestRef.current + 1;
    usageRequestRef.current = requestId;
    setUsage((previous) => ({ ...previous, loading: true, error: '' }));
    try {
      const result = await cloudAccessAPI.getProviderUsage(providerId);
      if (usageRequestRef.current !== requestId) return;
      setUsage({ loading: false, data: result.usage || null, error: '' });
    } catch (err) {
      if (usageRequestRef.current !== requestId) return;
      setUsage({ loading: false, data: null, error: err.message || 'Usage could not be loaded.' });
    }
  }, [apiKeyConfig.supportsUsage, genericMode, providerId]);

  useEffect(() => {
    if (configured) {
      loadUsage();
    } else {
      usageRequestRef.current += 1;
      setUsage({ loading: false, data: null, error: '' });
    }
    return () => {
      usageRequestRef.current += 1;
    };
  }, [configured, loadUsage]);

  if (!provider) return null;
  const { label } = provider;

  const handleSave = async () => {
    if (!apiKey.trim()) {
      onError(`Please enter a ${label} API key`);
      return;
    }
    setLoading(true);
    setMessage('');
    onError('');
    try {
      const result = await cloudAccessAPI.setProviderApiKey(providerId, apiKey.trim());
      onStatusChange(result.status || { configured: true });
      const models = Array.isArray(result.models) ? result.models : [];
      let modelsReady = models.length > 0;
      if (modelsReady) {
        onModelsStatusChange({ checking: false, count: models.length, error: '' });
      } else {
        modelsReady = Boolean(await verifyModels());
      }
      setApiKey('');
      setMessage(modelsReady
        ? `${label} API key saved and model list loaded.`
        : `${label} API key saved, but model loading needs attention.`);
      if (onCloudAccessChanged) {
        onCloudAccessChanged(true, providerId, { modelsReady });
      }
    } catch (err) {
      onError(err.message || `Failed to save ${label} API key`);
    } finally {
      setLoading(false);
    }
  };

  const handleClear = async () => {
    setLoading(true);
    setMessage('');
    onError('');
    try {
      await cloudAccessAPI.clearProviderApiKey(providerId);
      setApiKey('');
      onStatusChange({ configured: false });
      onModelsStatusChange({ checking: false, count: null, error: '' });
      setMessage(`${label} API key cleared.`);
      if (onCloudAccessChanged) {
        onCloudAccessChanged(false, providerId);
      }
    } catch (err) {
      onError(err.message || `Failed to clear ${label} API key`);
    } finally {
      setLoading(false);
    }
  };

  const saveDisabled = loading || genericMode || !apiKey.trim();

  return (
    <>
      <h4 style={{ marginTop: '0.25rem', color: '#ff9f4a', fontSize: '0.95rem' }}>
        {label} API
      </h4>
      {apiKeyConfig.description && <p className="settings-hint">{apiKeyConfig.description}</p>}
      {genericMode ? (
        <div className="test-result-banner test-result-banner--error" style={{ marginBottom: '1rem' }}>
          {label} direct API access is desktop-only in this build. Hosted mode should use OpenRouter.
        </div>
      ) : configured ? (
        <div
          className={`test-result-banner ${
            modelsStatus.error ? 'test-result-banner--error' : (success ? 'test-result-banner--success' : '')
          }`}
          style={{ marginBottom: '1rem' }}
        >
          {modelsStatus.error
            ? `${label} API key is saved, but model access is not verified.`
            : `${label} API key is configured${status.region_label ? ` (${status.region_label})` : ''}.`}
        </div>
      ) : (
        <div className="test-result-banner" style={{ marginBottom: '1rem' }}>
          {label} API key is not configured.
        </div>
      )}

      {modelsStatus.checking && (
        <div className="test-result-banner" style={{ marginBottom: '1rem' }}>
          Checking {label} model list...
        </div>
      )}

      {!modelsStatus.checking && modelsStatus.count > 0 && (
        <div className="test-result-banner test-result-banner--success" style={successBannerStyle}>
          SUCCESS! {label} connected. {modelsStatus.count} model{modelsStatus.count === 1 ? '' : 's'} available.
        </div>
      )}

      {!modelsStatus.checking && modelsStatus.error && (
        <div className="test-result-banner test-result-banner--error" style={{ marginBottom: '1rem' }}>
          {modelsStatus.error}
        </div>
      )}

      {message && (
        <div className={getMessageBannerClass(message)} style={{ marginBottom: '1rem' }}>
          {message}
        </div>
      )}

      {configured && apiKeyConfig.supportsUsage && !genericMode && (
        <div className="submitter-config-section" style={{ marginBottom: '1rem' }} aria-live="polite">
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
            <strong style={{ color: '#fff', fontSize: '0.9rem' }}>Subscription quota</strong>
            <button type="button" className="btn-ghost" onClick={loadUsage} disabled={usage.loading}>
              {usage.loading ? 'Refreshing...' : 'Refresh'}
            </button>
          </div>
          {usage.error ? (
            <small className="hint-text hint-text--dim">Quota could not be loaded: {usage.error}</small>
          ) : usage.data ? (
            <UsagePools usage={usage.data} />
          ) : (
            !usage.loading && <small className="hint-text hint-text--dim">Quota not loaded yet.</small>
          )}
        </div>
      )}

      <div style={{ marginTop: '1rem' }}>
        <label htmlFor={apiKeyConfig.inputId} style={{ display: 'block', color: '#ccc', marginBottom: '0.5rem', fontSize: '0.9rem' }}>
          {label} API Key
        </label>
        <SecretInput
          id={apiKeyConfig.inputId}
          name={apiKeyConfig.inputName}
          value={apiKey}
          onChange={(event) => setApiKey(event.target.value)}
          placeholder={`Paste ${label} API key`}
          className="input-dark"
          disabled={genericMode || loading}
          style={{ fontSize: '0.95rem' }}
        />
        {apiKeyConfig.consoleUrl && (
          <small className="hint-text hint-text--dim">
            Get a key from{' '}
            <a href={apiKeyConfig.consoleUrl} target="_blank" rel="noopener noreferrer" style={{ color: '#18cc17' }}>
              {apiKeyConfig.consoleLabel || apiKeyConfig.consoleUrl}
            </a>
            . MOTO stores it in the backend keyring.
          </small>
        )}
      </div>

      <div style={{ display: 'flex', gap: '0.75rem', marginTop: '1rem' }}>
        <button
          type="button"
          onClick={handleSave}
          disabled={saveDisabled}
          style={{
            flex: 1,
            padding: '0.75rem 1rem',
            backgroundColor: '#18cc17',
            border: 'none',
            borderRadius: '6px',
            color: '#fff',
            cursor: saveDisabled ? 'not-allowed' : 'pointer',
            opacity: saveDisabled ? 0.6 : 1,
            fontSize: '0.95rem',
            fontWeight: '500',
          }}
        >
          {loading ? 'Working...' : `Save ${provider.shortLabel} Key`}
        </button>
        {configured && (
          <button type="button" onClick={handleClear} disabled={loading} className="btn-ghost" style={{ flex: 1 }}>
            Clear {provider.shortLabel} Key
          </button>
        )}
      </div>
    </>
  );
}
