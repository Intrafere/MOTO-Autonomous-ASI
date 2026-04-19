import React, { useState, useEffect, useRef } from 'react';
import { boostAPI, openRouterAPI } from '../services/api';
import './BoostControlModal.css';

const BOOST_SETTINGS_STORAGE_KEY = 'boost_modal_settings';

export default function BoostControlModal({ isOpen, onClose }) {
  const [apiKey, setApiKey] = useState('');
  const [boostModel, setBoostModel] = useState('');
  const [selectedProvider, setSelectedProvider] = useState('');
  const [contextWindow, setContextWindow] = useState(131072);
  const [maxOutputTokens, setMaxOutputTokens] = useState(25000);
  const [models, setModels] = useState([]);
  const [providers, setProviders] = useState([]);
  const [loading, setLoading] = useState(false);
  const [loadingProviders, setLoadingProviders] = useState(false);
  const [testing, setTesting] = useState(false);
  const [error, setError] = useState('');
  const [success, setSuccess] = useState('');
  const [boostStatus, setBoostStatus] = useState(null);
  const [freeOnly, setFreeOnly] = useState(false);
  const [hasGlobalKey, setHasGlobalKey] = useState(false);
  
  // Track mouse down position to prevent closing on text selection drag
  const mouseDownTargetRef = useRef(null);

  const hasAvailableKey = Boolean(apiKey.trim() || hasGlobalKey);
  
  // Load saved settings from localStorage on mount
  useEffect(() => {
    try {
      const saved = localStorage.getItem(BOOST_SETTINGS_STORAGE_KEY);
      if (saved) {
        const settings = JSON.parse(saved);
        if (settings.boostModel) setBoostModel(settings.boostModel);
        if (settings.selectedProvider) setSelectedProvider(settings.selectedProvider);
        if (settings.contextWindow) setContextWindow(settings.contextWindow);
        if (settings.maxOutputTokens) setMaxOutputTokens(settings.maxOutputTokens);
        if (settings.freeOnly !== undefined) setFreeOnly(settings.freeOnly);
      }
    } catch (e) {
      console.error('Failed to load boost settings from localStorage:', e);
    }
  }, []);
  
  // Save settings to localStorage whenever they change
  useEffect(() => {
    // Only save if we have meaningful values (not initial empty state)
    if (boostModel || selectedProvider || contextWindow !== 131072 || maxOutputTokens !== 25000 || freeOnly) {
      try {
        const settings = {
          boostModel,
          selectedProvider,
          contextWindow,
          maxOutputTokens,
          freeOnly
        };
        localStorage.setItem(BOOST_SETTINGS_STORAGE_KEY, JSON.stringify(settings));
      } catch (e) {
        console.error('Failed to save boost settings to localStorage:', e);
      }
    }
  }, [boostModel, selectedProvider, contextWindow, maxOutputTokens, freeOnly]);

  const fetchProviders = async (modelId, keyOverride = undefined) => {
    if (!modelId) {
      setProviders([]);
      return;
    }

    const effectiveKey = keyOverride === undefined ? apiKey.trim() : keyOverride;

    setLoadingProviders(true);
    try {
      const response = await boostAPI.getModelProviders(effectiveKey || null, modelId);
      if (response.providers) {
        setProviders(response.providers);
      } else {
        setProviders([]);
      }
    } catch (error) {
      console.error('Failed to fetch providers:', error);
      setProviders([]);
    } finally {
      setLoadingProviders(false);
    }
  };

  const fetchBoostStatus = async (keyOverride = undefined) => {
    const effectiveKey = keyOverride === undefined ? apiKey.trim() : keyOverride;

    try {
      const response = await boostAPI.getStatus();
      if (response.status) {
        setBoostStatus(response.status);
        if (response.status.enabled) {
          // Boost is enabled - use backend values (they're authoritative)
          setBoostModel(response.status.model_id);
          setSelectedProvider(response.status.provider || '');
          setContextWindow(response.status.context_window);
          setMaxOutputTokens(response.status.max_output_tokens);
          if (response.status.model_id) {
            await fetchProviders(response.status.model_id, effectiveKey);
          }
          
          // Also save to localStorage so settings persist even if backend restarts
          try {
            const settings = {
              boostModel: response.status.model_id,
              selectedProvider: response.status.provider || '',
              contextWindow: response.status.context_window,
              maxOutputTokens: response.status.max_output_tokens,
              freeOnly
            };
            localStorage.setItem(BOOST_SETTINGS_STORAGE_KEY, JSON.stringify(settings));
          } catch (e) {
            console.error('Failed to sync boost settings to localStorage:', e);
          }
        } else {
          setProviders([]);
          // Boost not enabled - localStorage values are already loaded in useEffect
        }
      }
    } catch (error) {
      console.error('Failed to fetch boost status:', error);
    }
  };

  // Handle model selection change
  const handleModelChange = (modelId) => {
    setBoostModel(modelId);
    setSelectedProvider(''); // Reset provider when model changes
    if (modelId) {
      fetchProviders(modelId);
    } else {
      setProviders([]);
    }
  };

  const fetchModels = async (
    freeFilter = freeOnly,
    { silent = false, keyOverride = undefined } = {}
  ) => {
    const effectiveKey = keyOverride === undefined ? apiKey.trim() : keyOverride;

    setLoading(true);
    if (!silent) {
      setError('');
      setSuccess('');
    }

    try {
      const response = await boostAPI.getOpenRouterModels(effectiveKey || null);
      if (response.models) {
        const filtered = freeFilter
          ? response.models.filter(model => model.pricing && model.pricing.prompt === '0' && model.pricing.completion === '0')
          : response.models;
        setModels(filtered);
        if (!silent) {
          setSuccess(`Models loaded successfully (${filtered.length} ${freeFilter ? 'free ' : ''}models)`);
        }
      }
    } catch (error) {
      if (!silent) {
        setError(error.message || 'Failed to fetch models');
      }
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    if (!isOpen) {
      return;
    }

    const initializeModal = async () => {
      setApiKey('');
      setError('');
      setSuccess('');

      let useGlobalKey = false;
      try {
        const keyStatus = await openRouterAPI.getApiKeyStatus();
        useGlobalKey = Boolean(keyStatus.has_key);
        setHasGlobalKey(useGlobalKey);
      } catch (error) {
        console.error('Failed to check OpenRouter key status for boost modal:', error);
        setHasGlobalKey(false);
      }

      const preferredKey = null;
      await fetchBoostStatus(preferredKey);

      if (useGlobalKey) {
        await fetchModels(freeOnly, { silent: true, keyOverride: preferredKey });
      } else {
        setModels([]);
      }
    };

    initializeModal();
  }, [isOpen]);

  // Refetch models when free-only toggle changes
  useEffect(() => {
    if (isOpen && hasAvailableKey && models.length > 0) {
      fetchModels(freeOnly, { silent: true });
    }
  }, [freeOnly]);

  const testConnection = async () => {
    if (!hasAvailableKey) {
      setError('Please enter an API key or use an active OpenRouter key');
      return;
    }

    const effectiveKey = apiKey.trim() || null;
    const usingGlobalKey = !apiKey.trim() && hasGlobalKey;

    setTesting(true);
    setError('');
    setSuccess('');

    try {
      const response = await boostAPI.getOpenRouterModels(effectiveKey);
      if (response.models && response.models.length > 0) {
        setSuccess(`✓ Connected successfully${usingGlobalKey ? ' using the active OpenRouter key' : ''}! Found ${response.models.length} models.`);
        setModels(response.models);
      } else {
        setError('Connected but no models found');
      }
    } catch (error) {
      setError(error.message || 'Connection failed');
    } finally {
      setTesting(false);
    }
  };

  const enableBoost = async () => {
    if (!boostModel) {
      setError('Please select a model');
      return;
    }

    const trimmedApiKey = apiKey.trim();

    setLoading(true);
    setError('');
    setSuccess('');

    try {
      const config = {
        enabled: true,
        openrouter_api_key: trimmedApiKey,
        boost_model_id: boostModel,
        boost_provider: selectedProvider || null,
        boost_context_window: contextWindow,
        boost_max_output_tokens: maxOutputTokens
      };

      // Check if boost is already enabled - if so, use update endpoint instead
      const isBoostCurrentlyEnabled = boostStatus && boostStatus.enabled;
      
      let response;
      if (isBoostCurrentlyEnabled) {
        // Boost already enabled - update model seamlessly (preserves boost state)
        response = await boostAPI.updateModel(config);
        
        if (response.success) {
          setSuccess(`✓ Boost model updated! State preserved: ${response.preserved_state.boost_next_count} next calls`);
          await fetchBoostStatus();
          
          // REMOVED: Auto-close modal - user wants it to stay open
          // Modal stays open so user can continue configuring boost settings
        }
      } else {
        // Boost not enabled - use enable endpoint
        response = await boostAPI.enable(config);
        
        if (response.success) {
          setSuccess('✓ Boost enabled successfully!');
          await fetchBoostStatus();
          
          // REMOVED: Auto-close modal - user wants it to stay open
          // Modal stays open so user can continue configuring boost settings
        }
      }
    } catch (error) {
      setError(error.message || 'Failed to enable boost');
    } finally {
      setLoading(false);
    }
  };

  const disableBoost = async () => {
    setLoading(true);
    setError('');
    setSuccess('');

    try {
      const response = await boostAPI.disable();
      
      if (response.success) {
        setSuccess('✓ Boost disabled');
        await fetchBoostStatus();
        
        // REMOVED: Auto-close modal - user wants it to stay open
        // Modal stays open so user can re-enable or configure boost settings
      }
    } catch (error) {
      setError(error.message || 'Failed to disable boost');
    } finally {
      setLoading(false);
    }
  };

  // Handle overlay click - only close if mousedown AND mouseup both happened on overlay
  // This prevents closing when user drags to select text in inputs
  const handleOverlayMouseDown = (e) => {
    mouseDownTargetRef.current = e.target;
  };
  
  const handleOverlayClick = (e) => {
    // Only close if both mousedown and click happened on the overlay itself
    if (e.target === e.currentTarget && mouseDownTargetRef.current === e.currentTarget) {
      onClose();
    }
    mouseDownTargetRef.current = null;
  };

  if (!isOpen) return null;

  return (
    <div 
      className="modal-overlay" 
      onMouseDown={handleOverlayMouseDown}
      onClick={handleOverlayClick}
    >
      <div className="boost-modal">
        <div className="modal-header">
          <h2>API Boost Configuration</h2>
          <button className="close-btn" onClick={onClose}>×</button>
        </div>

        <div className="modal-body">
          {boostStatus && boostStatus.enabled && (
            <div className="boost-status-banner enabled">
              <span>✓ Boost Active</span>
              <span className="boost-model">{boostStatus.model_id}</span>
              {boostStatus.provider ? (
                <span className="boost-provider">via {boostStatus.provider}</span>
              ) : (
                <span className="boost-provider auto-route">auto-routing</span>
              )}
              <span className="boost-tasks">{boostStatus.boosted_task_count} tasks boosted</span>
            </div>
          )}

          {boostStatus && !boostStatus.enabled && (
            <div className="boost-status-banner disabled">
              <span>Boost Disabled</span>
            </div>
          )}

          <div className="boost-form-group">
            <label>OpenRouter API Key</label>
            <input
              type="password"
              value={apiKey}
              onChange={(e) => setApiKey(e.target.value)}
              placeholder="sk-or-..."
              disabled={loading}
            />
            <small>Leave this blank to reuse the active OpenRouter key, or paste a different key just for boost.</small>
          </div>

          <div className="boost-button-group">
            <button 
              onClick={testConnection} 
              disabled={testing || !hasAvailableKey}
              className="secondary"
            >
              {testing ? 'Testing...' : 'Test Connection'}
            </button>
            <button 
              onClick={() => fetchModels(freeOnly)} 
              disabled={loading || !hasAvailableKey}
              className="secondary"
            >
              {loading ? 'Loading...' : 'Load Models'}
            </button>
            <label style={{ display: 'inline-flex', alignItems: 'center', marginLeft: '1rem', fontSize: '0.9rem' }}>
              <input
                type="checkbox"
                checked={freeOnly}
                onChange={(e) => setFreeOnly(e.target.checked)}
                style={{ marginRight: '0.5rem' }}
              />
              Show only free models
            </label>
          </div>

          <div className="boost-form-group">
            <label>Boost Model</label>
            <select
              value={boostModel}
              onChange={(e) => handleModelChange(e.target.value)}
              disabled={loading || models.length === 0}
            >
              <option value="">Select a model...</option>
              {models.map(model => (
                <option key={model.id} value={model.id}>
                  {model.id}
                </option>
              ))}
            </select>
            {models.length === 0 && (
              <small>Models load automatically when an OpenRouter key is active. Use "Load Models" to refresh.</small>
            )}
          </div>

          {boostModel && (
            <div className="boost-form-group">
              <label>Provider</label>
              <select
                value={selectedProvider}
                onChange={(e) => setSelectedProvider(e.target.value)}
                disabled={loading || loadingProviders}
              >
                <option value="">Default (OpenRouter chooses)</option>
                {providers.map(provider => (
                  <option key={provider} value={provider}>
                    {provider}
                  </option>
                ))}
              </select>
              {loadingProviders && (
                <small>Loading providers...</small>
              )}
              {!loadingProviders && providers.length === 0 && boostModel && (
                <small>No specific providers available - OpenRouter will auto-route</small>
              )}
              {!loadingProviders && providers.length > 0 && (
                <small>Select a specific provider or leave as default</small>
              )}
            </div>
          )}

          <div className="form-row">
            <div className="boost-form-group">
              <label>Context Window</label>
              <input
                type="number"
                value={contextWindow}
                onChange={(e) => setContextWindow(parseInt(e.target.value) || 131072)}
                min="4096"
                max="999999"
                step="1024"
                disabled={loading}
              />
            </div>

            <div className="boost-form-group">
              <label>Max Output Tokens</label>
              <input
                type="number"
                value={maxOutputTokens}
                onChange={(e) => setMaxOutputTokens(parseInt(e.target.value) || 25000)}
                min="1000"
                max="100000"
                step="1000"
                disabled={loading}
              />
            </div>
          </div>

          {error && (
            <div className="message error">
              {error}
            </div>
          )}

          {success && (
            <div className="message success">
              {success}
            </div>
          )}

          <div className="info-box">
            <h4>How API Boost Works</h4>
            <ul>
              <li>Click tasks in the MOTO Workflow panel to toggle boost</li>
              <li>Boosted tasks use your OpenRouter model instead of LM Studio</li>
              <li>If credits run out, system falls back to LM Studio automatically</li>
              <li>You can toggle which tasks use the boost at any time</li>
            </ul>
          </div>
        </div>

        <div className="modal-footer">
          {boostStatus && boostStatus.enabled ? (
            <>
              <button 
                onClick={enableBoost} 
                disabled={loading || !boostModel}
                className="primary"
                title="Update boost model (preserves boost_next_count and categories)"
              >
                {loading ? 'Updating...' : 'Update Model'}
              </button>
              <button 
                onClick={disableBoost} 
                disabled={loading}
                className="danger"
              >
                {loading ? 'Disabling...' : 'Disable Boost'}
              </button>
            </>
          ) : (
            <button 
              onClick={enableBoost} 
              disabled={loading || !boostModel}
              className="primary"
            >
              {loading ? 'Enabling...' : 'Enable Boost'}
            </button>
          )}
          <button onClick={onClose} className="secondary">
            Close
          </button>
        </div>
      </div>
    </div>
  );
}

