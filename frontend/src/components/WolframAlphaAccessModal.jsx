import React, { useEffect, useState } from 'react';
import { api, connectivityAPI } from '../services/api';
import './settings-common.css';

export default function WolframAlphaAccessModal({
  isOpen,
  onClose,
  connectivityStatus,
  onConnectivityChanged,
  capabilities,
  anyWorkflowRunning = false,
}) {
  const [apiKey, setApiKey] = useState('');
  const [status, setStatus] = useState(null);
  const [message, setMessage] = useState('');
  const [loading, setLoading] = useState(false);
  const genericMode = Boolean(capabilities?.genericMode);
  const enabled = Boolean(connectivityStatus?.skills?.wolfram_alpha?.enabled);

  const loadStatus = async () => {
    try {
      const nextStatus = await api.getWolframStatus();
      setStatus(nextStatus);
    } catch (error) {
      setMessage(error.message || 'Failed to load Wolfram Alpha status');
    }
  };

  useEffect(() => {
    if (!isOpen) return;
    setApiKey('');
    setMessage('');
    loadStatus();
  }, [isOpen]);

  const handleToggle = async (nextEnabled) => {
    setLoading(true);
    setMessage('');
    try {
      const next = await connectivityAPI.updateToggles({ wolfram_alpha_enabled: nextEnabled });
      onConnectivityChanged?.(next);
      await loadStatus();
      setMessage(nextEnabled ? 'Wolfram Alpha enabled for future construction runs.' : 'Wolfram Alpha disabled for future construction runs.');
    } catch (error) {
      setMessage(error.message || 'Failed to update Wolfram Alpha toggle');
    } finally {
      setLoading(false);
    }
  };

  const handleTestAndSave = async () => {
    if (!apiKey.trim()) {
      setMessage('Please enter a Wolfram Alpha App ID.');
      return;
    }
    setLoading(true);
    setMessage('Testing...');
    try {
      const result = await api.testWolframQuery({
        query: 'What is 2+2?',
        api_key: apiKey.trim(),
      });
      if (!result.success) {
        setMessage(`Failed: ${result.message || 'Wolfram Alpha test query failed'}`);
        return;
      }
      await api.setWolframApiKey(apiKey.trim());
      const next = await connectivityAPI.updateToggles({ wolfram_alpha_enabled: true });
      onConnectivityChanged?.(next);
      setApiKey('');
      await loadStatus();
      setMessage(`Success. Test result: ${result.result}`);
    } catch (error) {
      setMessage(error.message || 'Failed to save Wolfram Alpha App ID');
    } finally {
      setLoading(false);
    }
  };

  const handleClearKey = async () => {
    setLoading(true);
    setMessage('');
    try {
      await api.clearWolframApiKey();
      const next = await connectivityAPI.getStatus();
      onConnectivityChanged?.(next);
      setApiKey('');
      await loadStatus();
      setMessage('Wolfram Alpha App ID cleared.');
    } catch (error) {
      setMessage(error.message || 'Failed to clear Wolfram Alpha App ID');
    } finally {
      setLoading(false);
    }
  };

  if (!isOpen) return null;

  const wolframConnectivity = connectivityStatus?.skills?.wolfram_alpha;
  const hasStatusResponse = Boolean(status || wolframConnectivity);
  const hasStoredKey = Boolean(status?.has_key || wolframConnectivity?.has_key);
  const keyStatusMessage = hasStatusResponse
    ? (
        hasStoredKey
          ? (enabled ? 'A Wolfram Alpha App ID is configured and enabled.' : 'A Wolfram Alpha App ID is configured but disabled.')
          : 'No Wolfram Alpha App ID is configured.'
      )
    : 'Checking Wolfram Alpha App ID status...';

  return (
    <div className="inline-modal-overlay" style={{ zIndex: 10000 }} onClick={(event) => event.target === event.currentTarget && onClose()}>
      <div className="inline-modal-content" style={{ width: '620px', maxWidth: '90vw', backgroundColor: '#1a1a2e', borderRadius: '12px' }}>
        <div className="settings-header-row" style={{ marginBottom: '1.5rem' }}>
          <h2 style={{ margin: 0, color: '#fff', fontSize: '1.4rem' }}>Wolfram Alpha Access</h2>
          <button onClick={onClose} className="modal-close-btn" aria-label="Close Wolfram Alpha access">×</button>
        </div>

        <p className="settings-hint">
          Enable Wolfram Alpha tool calls for computational verification during compiler/autonomous construction mode. Use a Wolfram Alpha App ID with full results enabled. Wolfram Alpha is like giving your AI a very advanced calculator.
        </p>

        <label className="settings-checkbox-label settings-checkbox-label--stacked" style={{ marginBottom: '1rem' }}>
          <input
            type="checkbox"
            checked={enabled}
            onChange={(event) => handleToggle(event.target.checked)}
            disabled={loading || anyWorkflowRunning}
          />
          <span>Enable Wolfram Alpha</span>
        </label>
        {anyWorkflowRunning && (
          <div className="test-result-banner" style={{ marginBottom: '1rem' }}>
            Stop the active workflow before changing run-level Wolfram Alpha availability.
          </div>
        )}

        <div className={`test-result-banner ${hasStoredKey && enabled ? 'test-result-banner--success' : ''}`} style={{ marginBottom: '1rem' }}>
          {keyStatusMessage}
          {genericMode && (
            <>
              <br />
              <small>Hosted/generic mode keeps this key in backend memory for the active runtime instance.</small>
            </>
          )}
        </div>

        {message && (
          <div className={message.includes('Success') || message.includes('enabled') || message.includes('disabled') || message.includes('cleared') ? 'test-result-banner test-result-banner--success' : 'test-result-banner test-result-banner--error'} style={{ marginBottom: '1rem' }}>
            {message}
          </div>
        )}

        <div className="submitter-config-section">
          <label style={{ display: 'block', color: '#ccc', marginBottom: '0.5rem', fontSize: '0.9rem' }}>
            Wolfram Alpha App ID
          </label>
          <input
            type="password"
            value={apiKey}
            onChange={(event) => setApiKey(event.target.value)}
            placeholder={hasStoredKey ? 'Enter a new App ID to replace the stored one' : 'Paste Wolfram Alpha App ID'}
            className="input-dark"
          />
          <small className="hint-text hint-text--dim">
            Get an App ID from{' '}
            <a href="https://products.wolframalpha.com/api" target="_blank" rel="noopener noreferrer">
              developer.wolframalpha.com
            </a>
            .
          </small>
          <div style={{ display: 'flex', gap: '0.75rem', flexWrap: 'wrap', marginTop: '1rem' }}>
            <button onClick={handleTestAndSave} disabled={loading || !apiKey.trim()} className="btn-ghost">
              {loading ? 'Working...' : 'Test And Save'}
            </button>
            <button onClick={handleClearKey} disabled={loading || !hasStoredKey} className="btn-ghost">
              Clear App ID
            </button>
            <button onClick={loadStatus} disabled={loading} className="btn-ghost">
              Refresh Status
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}

