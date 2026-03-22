import React, { useState, useEffect } from 'react';
import { api, boostAPI } from '../services/api';
import './BoostControlModal.css';

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

  // Load saved API key from localStorage
  useEffect(() => {
    const savedKey = localStorage.getItem('openrouter_api_key');
    if (savedKey) {
      setApiKey(savedKey);
    }
  }, []);

  // Fetch boost status
  useEffect(() => {
    if (isOpen) {
      fetchBoostStatus();
    }
  }, [isOpen]);

  const fetchBoostStatus = async () => {
    try {
      const response = await boostAPI.getStatus();
      if (response.status) {
        setBoostStatus(response.status);
        if (response.status.enabled) {
          setBoostModel(response.status.model_id);
          setSelectedProvider(response.status.provider || '');
          setContextWindow(response.status.context_window);
          setMaxOutputTokens(response.status.max_output_tokens);
        }
      }
    } catch (error) {
      console.error('Failed to fetch boost status:', error);
    }
  };

  // Fetch providers when model is selected
  const fetchProviders = async (modelId) => {
    if (!apiKey || !modelId) {
      setProviders([]);
      return;
    }

    setLoadingProviders(true);
    try {
      const response = await boostAPI.getModelProviders(apiKey, modelId);
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

  const fetchModels = async (freeFilter = freeOnly) => {
    if (!apiKey) {
      setError('Please enter an API key first');
      return;
    }

    setLoading(true);
    setError('');

    try {
      const response = await boostAPI.getOpenRouterModels(apiKey);
      if (response.models) {
        // Filter for free models only if enabled
        const filtered = freeFilter 
          ? response.models.filter(model => model.pricing && model.pricing.prompt === '0' && model.pricing.completion === '0')
          : response.models;
        setModels(filtered);
        setSuccess(`Models loaded successfully (${filtered.length} ${freeFilter ? 'free ' : ''}models)`);
      }
    } catch (error) {
      setError(error.message || 'Failed to fetch models');
    } finally {
      setLoading(false);
    }
  };

  // Refetch models when free-only toggle changes
  useEffect(() => {
    if (apiKey && models.length > 0) {
      fetchModels(freeOnly);
    }
  }, [freeOnly]);

  const testConnection = async () => {
    if (!apiKey) {
      setError('Please enter an API key');
      return;
    }

    setTesting(true);
    setError('');
    setSuccess('');

    try {
      const response = await boostAPI.getOpenRouterModels(apiKey);
      if (response.models && response.models.length > 0) {
        setSuccess(`✓ Connected successfully! Found ${response.models.length} models.`);
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
    if (!apiKey || !boostModel) {
      setError('Please enter API key and select a model');
      return;
    }

    setLoading(true);
    setError('');
    setSuccess('');

    try {
      const config = {
        enabled: true,
        openrouter_api_key: apiKey,
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
          // Save API key to localStorage
          localStorage.setItem('openrouter_api_key', apiKey);
          
          setSuccess(`✓ Boost model updated! State preserved: ${response.preserved_state.boost_next_count} next calls`);
          await fetchBoostStatus();
          
          // REMOVED: Auto-close modal - user wants it to stay open
          // Modal stays open so user can continue configuring boost settings
        }
      } else {
        // Boost not enabled - use enable endpoint
        response = await boostAPI.enable(config);
        
        if (response.success) {
          // Save API key to localStorage
          localStorage.setItem('openrouter_api_key', apiKey);
          
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

  if (!isOpen) return null;

  return (
    <div className="modal-overlay" onClick={onClose}>
      <div className="boost-modal" onClick={(e) => e.stopPropagation()}>
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

          <div className="form-group">
            <label>OpenRouter API Key</label>
            <input
              type="password"
              value={apiKey}
              onChange={(e) => setApiKey(e.target.value)}
              placeholder="sk-or-..."
              disabled={loading}
            />
            <small>Your API key is stored locally and never sent to our servers</small>
          </div>

          <div className="button-group">
            <button 
              onClick={testConnection} 
              disabled={testing || !apiKey}
              className="secondary"
            >
              {testing ? 'Testing...' : 'Test Connection'}
            </button>
            <button 
              onClick={() => fetchModels(freeOnly)} 
              disabled={loading || !apiKey}
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

          <div className="form-group">
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
              <small>Click "Load Models" to fetch available models</small>
            )}
          </div>

          {boostModel && (
            <div className="form-group">
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
            <div className="form-group">
              <label>Context Window</label>
              <input
                type="number"
                value={contextWindow}
                onChange={(e) => setContextWindow(parseInt(e.target.value))}
                min="4096"
                max="999999"
                step="1024"
                disabled={loading}
              />
            </div>

            <div className="form-group">
              <label>Max Output Tokens</label>
              <input
                type="number"
                value={maxOutputTokens}
                onChange={(e) => setMaxOutputTokens(parseInt(e.target.value))}
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
              <li>You can continuously select which tasks use the boost</li>
            </ul>
          </div>
        </div>

        <div className="modal-footer">
          {boostStatus && boostStatus.enabled ? (
            <>
              <button 
                onClick={enableBoost} 
                disabled={loading || !apiKey || !boostModel}
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
              disabled={loading || !apiKey || !boostModel}
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

