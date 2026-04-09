import React, { useState, useEffect } from 'react';
import { openRouterAPI } from '../services/api';
import './settings-common.css';

/**
 * Modal for configuring the global OpenRouter API key.
 * This key is used for per-role OpenRouter model selection and can also be reused by boost.
 * 
 * Shows when:
 * 1. User clicks "Use OpenRouter" on any role but no API key is configured
 * 2. LM Studio is unavailable and user needs OpenRouter as primary provider
 * 3. User explicitly wants to manage their API key
 */
export default function OpenRouterApiKeyModal({ isOpen, onClose, onKeySet, reason = 'setup' }) {
  const [apiKey, setApiKey] = useState('');
  const [testing, setTesting] = useState(false);
  const [saving, setSaving] = useState(false);
  const [testResult, setTestResult] = useState(null);
  const [error, setError] = useState('');
  const [hasStoredKey, setHasStoredKey] = useState(false);

  // Reset state when modal opens
  useEffect(() => {
    if (isOpen) {
      setApiKey('');
      setTestResult(null);
      setError('');
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

      loadKeyStatus();

      return () => {
        isCancelled = true;
      };
    }
    setHasStoredKey(false);
    return undefined;
  }, [isOpen]);

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

  if (!isOpen) return null;

  const reasonMessages = {
    setup: 'Configure your OpenRouter API key to use OpenRouter models for any role.',
    startup_setup: 'Save your OpenRouter API key to unlock cloud models. MOTO will apply the recommended default profile immediately, and you can switch to your team profile or another default profile later in Settings.',
    lm_studio_unavailable: 'LM Studio is not available. Configure OpenRouter to continue.',
    no_key: 'An OpenRouter API key is required to use OpenRouter models.',
  };

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
          width: '500px',
          maxWidth: '90vw',
          backgroundColor: '#1a1a2e',
          borderRadius: '12px',
        }}
      >
        <div className="settings-header-row" style={{ marginBottom: '1.5rem' }}>
          <h2 style={{ margin: 0, color: '#fff', fontSize: '1.4rem' }}>
            OpenRouter API Key
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

        {/* API Key Input */}
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
              style={{ color: '#6c5ce7' }}
            >
              openrouter.ai/keys
            </a>
          </small>
        </div>

        {/* Error Message */}
        {error && (
          <div className="test-result-banner test-result-banner--error" style={{
            marginBottom: '1rem',
          }}>
            {error}
          </div>
        )}

        {/* Test Result */}
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
            An OpenRouter API key is already stored securely on the backend for this machine.
            Enter a new key below to replace it.
          </div>
        )}

        {/* Action Buttons */}
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
              backgroundColor: '#6c5ce7',
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

        {/* Clear Key Button */}
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

        {/* Info Note */}
        <p style={{ 
          color: '#666', 
          fontSize: '0.8rem', 
          marginTop: '1.5rem',
          padding: '0.75rem',
          backgroundColor: '#0d0d1a',
          borderRadius: '6px',
        }}>
          This API key is stored securely through the backend keyring integration and sent to the backend for OpenRouter API calls.
          API Boost can reuse this key automatically, or you can override it inside the boost modal.
        </p>
      </div>
    </div>
  );
}

