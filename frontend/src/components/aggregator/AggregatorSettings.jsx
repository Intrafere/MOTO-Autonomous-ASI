import React, { useState, useEffect } from 'react';
import { api, openRouterAPI } from '../../services/api';

const DEFAULT_SUBMITTER_CONFIG = {
  submitterId: 1,
  provider: 'lm_studio',
  modelId: '',
  openrouterProvider: null,
  lmStudioFallbackId: null,
  contextWindow: 131072,
  maxOutputTokens: 25000
};

export default function AggregatorSettings({ config, setConfig }) {
  const [lmStudioModels, setLmStudioModels] = useState([]);
  const [openRouterModels, setOpenRouterModels] = useState([]);
  const [modelProviders, setModelProviders] = useState({}); // { modelId: [providers] }
  const [loading, setLoading] = useState(true);
  const [saveMessage, setSaveMessage] = useState('');
  const [numSubmitters, setNumSubmitters] = useState(
    config.submitterConfigs?.length || 3
  );
  const [submitterConfigs, setSubmitterConfigs] = useState(
    config.submitterConfigs || [
      { ...DEFAULT_SUBMITTER_CONFIG, submitterId: 1 },
      { ...DEFAULT_SUBMITTER_CONFIG, submitterId: 2 },
      { ...DEFAULT_SUBMITTER_CONFIG, submitterId: 3 }
    ]
  );
  const [validatorMaxOutput, setValidatorMaxOutput] = useState(config.validatorMaxOutput || 25000);
  
  // Validator OpenRouter state
  const [validatorProvider, setValidatorProvider] = useState(config.validatorProvider || 'lm_studio');
  const [validatorOpenrouterProvider, setValidatorOpenrouterProvider] = useState(config.validatorOpenrouterProvider || null);
  const [validatorLmStudioFallback, setValidatorLmStudioFallback] = useState(config.validatorLmStudioFallback || null);
  
  // OpenRouter API key status
  const [hasOpenRouterKey, setHasOpenRouterKey] = useState(false);
  const [loadingOpenRouter, setLoadingOpenRouter] = useState(false);
  const [freeOnly, setFreeOnly] = useState(false);
  const [freeModelLooping, setFreeModelLooping] = useState(true);
  const [freeModelAutoSelector, setFreeModelAutoSelector] = useState(true);
  const [isLoaded, setIsLoaded] = useState(false);

  // Load settings from localStorage on mount
  useEffect(() => {
    const loadSettings = async () => {
      const savedSettings = localStorage.getItem('aggregator_settings');
      if (savedSettings) {
        try {
          const settings = JSON.parse(savedSettings);
          // Restore all state variables
          if (settings.numSubmitters) setNumSubmitters(settings.numSubmitters);
          if (settings.submitterConfigs) setSubmitterConfigs(settings.submitterConfigs);
          if (settings.validatorProvider) setValidatorProvider(settings.validatorProvider);
          if (settings.validatorOpenrouterProvider) setValidatorOpenrouterProvider(settings.validatorOpenrouterProvider);
          if (settings.validatorLmStudioFallback) setValidatorLmStudioFallback(settings.validatorLmStudioFallback);
          if (settings.validatorMaxOutput) setValidatorMaxOutput(settings.validatorMaxOutput);
          if (settings.freeOnly !== undefined) setFreeOnly(settings.freeOnly);
          if (settings.freeModelLooping !== undefined) setFreeModelLooping(settings.freeModelLooping);
          if (settings.freeModelAutoSelector !== undefined) setFreeModelAutoSelector(settings.freeModelAutoSelector);
          if (settings.modelProviders) setModelProviders(settings.modelProviders);
        } catch (error) {
          console.error('Failed to load aggregator settings:', error);
        }
      }
      setIsLoaded(true);
    };
    loadSettings();
  }, []);

  // Fetch providers for any OpenRouter models after settings are loaded
  useEffect(() => {
    if (!isLoaded || !hasOpenRouterKey) return;
    
    // Fetch providers for submitter configs
    submitterConfigs.forEach(cfg => {
      if (cfg.provider === 'openrouter' && cfg.modelId) {
        fetchProvidersForModel(cfg.modelId);
      }
    });
    
    // Fetch providers for validator
    if (validatorProvider === 'openrouter' && config.validatorModel) {
      fetchProvidersForModel(config.validatorModel);
    }
  }, [isLoaded, hasOpenRouterKey, submitterConfigs, validatorProvider, config.validatorModel]);

  // Save settings to localStorage whenever values change
  useEffect(() => {
    if (!isLoaded) return;
    
    const settings = {
      numSubmitters,
      submitterConfigs,
      validatorProvider,
      validatorOpenrouterProvider,
      validatorLmStudioFallback,
      validatorMaxOutput,
      freeOnly,
      freeModelLooping,
      freeModelAutoSelector,
      modelProviders
    };
    localStorage.setItem('aggregator_settings', JSON.stringify(settings));
  }, [isLoaded, numSubmitters, submitterConfigs, validatorProvider, validatorOpenrouterProvider, validatorLmStudioFallback, validatorMaxOutput, freeOnly, freeModelLooping, freeModelAutoSelector, modelProviders]);

  useEffect(() => {
    fetchModels();
    checkOpenRouterKey();
  }, []);

  const checkOpenRouterKey = async () => {
    try {
      const status = await openRouterAPI.getApiKeyStatus();
      setHasOpenRouterKey(status.has_key);
      if (status.has_key) {
        fetchOpenRouterModels();
      }
    } catch (err) {
      console.error('Failed to check OpenRouter key status:', err);
    }
  };

  const fetchOpenRouterModels = async (freeFilter = freeOnly) => {
    setLoadingOpenRouter(true);
    try {
      const result = await openRouterAPI.getModels(null, freeFilter);
      setOpenRouterModels(result.models || []);
    } catch (err) {
      console.error('Failed to fetch OpenRouter models:', err);
    } finally {
      setLoadingOpenRouter(false);
    }
  };

  // Refetch models when free-only toggle changes
  useEffect(() => {
    if (hasOpenRouterKey && isLoaded) {
      fetchOpenRouterModels(freeOnly);
    }
  }, [freeOnly]);

  const fetchProvidersForModel = async (modelId) => {
    if (!modelId || modelProviders[modelId]) return;
    try {
      const result = await openRouterAPI.getProviders(modelId);
      setModelProviders(prev => ({ ...prev, [modelId]: result.providers || [] }));
    } catch (err) {
      console.error(`Failed to fetch providers for ${modelId}:`, err);
    }
  };

  // Handle number of submitters change - expand/contract configs
  const handleNumSubmittersChange = (newNum) => {
    const num = parseInt(newNum);
    setNumSubmitters(num);
    
    const newConfigs = [];
    for (let i = 1; i <= num; i++) {
      const existing = submitterConfigs.find(c => c.submitterId === i);
      if (existing) {
        newConfigs.push(existing);
      } else {
        // Use first submitter's settings as template for new submitters
        const template = submitterConfigs[0] || DEFAULT_SUBMITTER_CONFIG;
        newConfigs.push({
          ...DEFAULT_SUBMITTER_CONFIG,
          submitterId: i,
          provider: template.provider,
          modelId: template.modelId,
          openrouterProvider: template.openrouterProvider,
          lmStudioFallbackId: template.lmStudioFallbackId,
          contextWindow: template.contextWindow,
          maxOutputTokens: template.maxOutputTokens
        });
      }
    }
    setSubmitterConfigs(newConfigs);
    setConfig(prev => ({ ...prev, submitterConfigs: newConfigs }));
  };

  const fetchModels = async () => {
    try {
      const data = await api.getModels();
      setLmStudioModels(data);
      
      // Auto-select first model if none selected (only for LM Studio provider)
      if (data.length > 0) {
        const firstModelId = data[0].id;
        
        // Update submitter configs with first model if needed
        const updatedConfigs = submitterConfigs.map(s => ({
          ...s,
          modelId: (s.provider === 'lm_studio' && !s.modelId) ? firstModelId : s.modelId
        }));
        setSubmitterConfigs(updatedConfigs);
        
        setConfig(prev => ({
          ...prev,
          validatorModel: (validatorProvider === 'lm_studio' && !prev.validatorModel) ? firstModelId : prev.validatorModel,
          submitterConfigs: updatedConfigs
        }));
      }
    } catch (error) {
      console.error('Failed to fetch models:', error);
    } finally {
      setLoading(false);
    }
  };

  const updateSubmitterConfig = (submitterId, field, value) => {
    // Handle NaN for numeric fields - use defaults
    let safeValue = value;
    if (field === 'contextWindow' && isNaN(value)) {
      safeValue = 131072;
    } else if (field === 'maxOutputTokens' && isNaN(value)) {
      safeValue = 25000;
    }
    
    const newConfigs = submitterConfigs.map(c => {
      if (c.submitterId !== submitterId) return c;
      
      const updated = { ...c, [field]: safeValue };
      
      // If switching provider, reset model-specific fields
      if (field === 'provider') {
        updated.modelId = '';
        updated.openrouterProvider = null;
        updated.lmStudioFallbackId = null;
      }
      
      // If selecting OpenRouter model, fetch providers
      if (field === 'modelId' && c.provider === 'openrouter' && safeValue) {
        fetchProvidersForModel(safeValue);
      }
      
      return updated;
    });
    
    setSubmitterConfigs(newConfigs);
    setConfig(prev => ({ ...prev, submitterConfigs: newConfigs }));
  };

  const applyToAll = (fromSubmitterId) => {
    const source = submitterConfigs.find(c => c.submitterId === fromSubmitterId);
    if (!source) return;
    
    const newConfigs = submitterConfigs.map(c => ({
      ...c,
      provider: source.provider,
      modelId: source.modelId,
      openrouterProvider: source.openrouterProvider,
      lmStudioFallbackId: source.lmStudioFallbackId,
      contextWindow: source.contextWindow,
      maxOutputTokens: source.maxOutputTokens
    }));
    setSubmitterConfigs(newConfigs);
    setConfig(prev => ({ ...prev, submitterConfigs: newConfigs }));
    setSaveMessage('Applied to all submitters ✓');
  };

  // Update validator config
  const updateValidatorProvider = (provider) => {
    setValidatorProvider(provider);
    if (provider === 'lm_studio') {
      setValidatorOpenrouterProvider(null);
      setValidatorLmStudioFallback(null);
    }
    setConfig(prev => ({
      ...prev,
      validatorProvider: provider,
      validatorModel: '',
      validatorOpenrouterProvider: null,
      validatorLmStudioFallback: null
    }));
  };

  // Model selector component for either provider
  const ModelSelector = ({ 
    provider, 
    modelId, 
    openrouterProvider: orProvider, 
    lmStudioFallbackId, 
    onModelChange, 
    onProviderChange, 
    onOpenrouterProviderChange, 
    onFallbackChange,
    label = 'Model'
  }) => {
    const models = provider === 'openrouter' ? openRouterModels : lmStudioModels;
    const providers = modelId && provider === 'openrouter' ? (modelProviders[modelId] || []) : [];
    
    return (
      <>
        {/* Provider Toggle */}
        <div className="form-group" style={{ margin: 0 }}>
          <label style={{ fontSize: '0.85rem' }}>Provider</label>
          <div style={{ display: 'flex', gap: '0.5rem' }}>
            <button
              type="button"
              onClick={() => onProviderChange('lm_studio')}
              style={{
                flex: 1,
                padding: '0.5rem',
                backgroundColor: provider === 'lm_studio' ? '#4CAF50' : '#333',
                border: 'none',
                borderRadius: '4px',
                color: '#fff',
                cursor: 'pointer',
                fontSize: '0.8rem'
              }}
            >
              LM Studio
            </button>
            <button
              type="button"
              onClick={() => hasOpenRouterKey && onProviderChange('openrouter')}
              disabled={!hasOpenRouterKey}
              style={{
                flex: 1,
                padding: '0.5rem',
                backgroundColor: provider === 'openrouter' ? '#6c5ce7' : '#333',
                border: 'none',
                borderRadius: '4px',
                color: hasOpenRouterKey ? '#fff' : '#666',
                cursor: hasOpenRouterKey ? 'pointer' : 'not-allowed',
                fontSize: '0.8rem'
              }}
              title={!hasOpenRouterKey ? 'Set OpenRouter API key first' : 'Use OpenRouter'}
            >
              OpenRouter
            </button>
          </div>
        </div>

        {/* Model Selection */}
        <div className="form-group" style={{ margin: 0 }}>
          <label style={{ fontSize: '0.85rem' }}>{label}</label>
          <select
            value={modelId || ''}
            onChange={(e) => onModelChange(e.target.value)}
            style={{ fontSize: '0.85rem' }}
          >
            <option value="">Select model...</option>
            {models.map(model => {
              const isFree = provider === 'openrouter' && 
                            model.pricing?.prompt === "0" && 
                            model.pricing?.completion === "0";
              const displayName = model.name || model.id;
              const contextInfo = model.context_length ? ` (${Math.round(model.context_length/1000)}K)` : '';
              
              return (
                <option key={model.id} value={model.id}>
                  {displayName}{contextInfo}{isFree ? ' [FREE]' : ''}
                </option>
              );
            })}
          </select>
        </div>

        {/* OpenRouter Provider Selection (only for OpenRouter) */}
        {provider === 'openrouter' && modelId && (
          <div className="form-group" style={{ margin: 0 }}>
            <label style={{ fontSize: '0.85rem' }}>Host Provider (optional)</label>
            <select
              value={orProvider || ''}
              onChange={(e) => onOpenrouterProviderChange(e.target.value || null)}
              style={{ fontSize: '0.85rem' }}
            >
              <option value="">Auto (let OpenRouter choose)</option>
              {providers.map(p => (
                <option key={p} value={p}>{p}</option>
              ))}
            </select>
          </div>
        )}

        {/* LM Studio Fallback (only for OpenRouter) */}
        {provider === 'openrouter' && (
          <div className="form-group" style={{ margin: 0 }}>
            <label style={{ fontSize: '0.85rem', color: '#999' }}>
              LM Studio Fallback (optional)
            </label>
            <select
              value={lmStudioFallbackId || ''}
              onChange={(e) => onFallbackChange(e.target.value || null)}
              style={{ fontSize: '0.85rem', borderColor: '#444' }}
            >
              <option value="">No fallback</option>
              {lmStudioModels.map(model => (
                <option key={model.id} value={model.id}>{model.id}</option>
              ))}
            </select>
            <small style={{ color: '#666', display: 'block', marginTop: '0.25rem' }}>
              Used if OpenRouter credits run out
            </small>
          </div>
        )}
      </>
    );
  };

  return (
    <div>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <h1>Aggregator Settings</h1>
        {saveMessage && (
          <div style={{ color: '#4CAF50', fontSize: '0.9rem', fontWeight: '500' }}>
            {saveMessage}
          </div>
        )}
      </div>

      {/* OpenRouter Status Banner */}
      {!hasOpenRouterKey && (
        <div style={{
          backgroundColor: 'rgba(108, 92, 231, 0.1)',
          border: '1px solid #6c5ce7',
          borderRadius: '8px',
          padding: '1rem',
          marginBottom: '1.5rem'
        }}>
          <p style={{ color: '#a29bfe', margin: 0 }}>
            <strong>💡 OpenRouter Available:</strong> Set your OpenRouter API key in the header to enable cloud model selection for any role.
          </p>
        </div>
      )}

      {loading ? (
        <div>Loading models...</div>
      ) : lmStudioModels.length === 0 && !hasOpenRouterKey ? (
        <div style={{ color: '#f44336' }}>
          <p>No models found. Make sure LM Studio is running on http://127.0.0.1:1234 or configure OpenRouter.</p>
          <button onClick={fetchModels} className="secondary">Retry</button>
        </div>
      ) : (
        <>
          {/* Number of Submitters Slider */}
          <div className="form-group" style={{ marginBottom: '2rem', padding: '1rem', background: '#1a2332', borderRadius: '8px' }}>
            <label style={{ fontSize: '1.1rem', fontWeight: '600' }}>
              Number of Aggregator Submitters: {numSubmitters}
            </label>
            <input
              type="range"
              min="1"
              max="10"
              value={numSubmitters}
              onChange={(e) => handleNumSubmittersChange(e.target.value)}
              style={{ width: '100%', marginTop: '0.5rem' }}
            />
            <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.8rem', color: '#888' }}>
              <span>1</span>
              <span>5</span>
              <span>10</span>
            </div>
            <small style={{ color: '#999', display: 'block', marginTop: '0.5rem' }}>
              Multiple submitters run in parallel exploring different avenues. Each can use a different model.
            </small>
          </div>

          {/* Per-Submitter Configuration Cards */}
          <div style={{ marginBottom: '2rem' }}>
            <h3 style={{ marginBottom: '1rem', borderBottom: '1px solid #333', paddingBottom: '0.5rem' }}>
              Submitter Configurations
            </h3>
            
            {submitterConfigs.map((cfg, idx) => (
              <div 
                key={cfg.submitterId}
                style={{
                  background: cfg.submitterId === 1 ? '#1a2838' : '#1a1a24',
                  border: cfg.provider === 'openrouter' 
                    ? '2px solid #6c5ce7' 
                    : (cfg.submitterId === 1 ? '2px solid #4CAF50' : '1px solid #333'),
                  borderRadius: '8px',
                  padding: '1rem',
                  marginBottom: '1rem'
                }}
              >
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1rem' }}>
                  <h4 style={{ margin: 0, color: cfg.provider === 'openrouter' ? '#a29bfe' : (cfg.submitterId === 1 ? '#4CAF50' : '#fff') }}>
                    Submitter {cfg.submitterId} 
                    {cfg.submitterId === 1 && <span style={{ fontWeight: 'normal' }}> (Main Submitter)</span>}
                    {cfg.provider === 'openrouter' && <span style={{ fontWeight: 'normal', color: '#6c5ce7' }}> [OpenRouter]</span>}
                  </h4>
                  {cfg.submitterId === 1 && numSubmitters > 1 && (
                    <button 
                      onClick={() => applyToAll(1)}
                      style={{ 
                        fontSize: '0.8rem', 
                        padding: '0.3rem 0.6rem',
                        background: '#4CAF50',
                        border: 'none',
                        borderRadius: '4px',
                        cursor: 'pointer',
                        color: '#fff'
                      }}
                    >
                      Apply to All
                    </button>
                  )}
                </div>

                <div style={{ display: 'grid', gridTemplateColumns: cfg.provider === 'openrouter' ? '1fr 1fr' : '1fr 1fr 1fr', gap: '1rem' }}>
                  <ModelSelector
                    provider={cfg.provider}
                    modelId={cfg.modelId}
                    openrouterProvider={cfg.openrouterProvider}
                    lmStudioFallbackId={cfg.lmStudioFallbackId}
                    onProviderChange={(p) => updateSubmitterConfig(cfg.submitterId, 'provider', p)}
                    onModelChange={(m) => updateSubmitterConfig(cfg.submitterId, 'modelId', m)}
                    onOpenrouterProviderChange={(p) => updateSubmitterConfig(cfg.submitterId, 'openrouterProvider', p)}
                    onFallbackChange={(f) => updateSubmitterConfig(cfg.submitterId, 'lmStudioFallbackId', f)}
                  />

                  <div className="form-group" style={{ margin: 0 }}>
                    <label style={{ fontSize: '0.85rem' }}>Context Window</label>
                    <input
                      type="number"
                      value={cfg.contextWindow}
                      onChange={(e) => updateSubmitterConfig(cfg.submitterId, 'contextWindow', parseInt(e.target.value))}
                      min="4096"
                      max="999999"
                      step="1024"
                      style={{ fontSize: '0.85rem' }}
                    />
                  </div>

                  <div className="form-group" style={{ margin: 0 }}>
                    <label style={{ fontSize: '0.85rem' }}>Max Output Tokens</label>
                    <input
                      type="number"
                      value={cfg.maxOutputTokens}
                      onChange={(e) => updateSubmitterConfig(cfg.submitterId, 'maxOutputTokens', parseInt(e.target.value))}
                      min="1000"
                      max="100000"
                      step="1000"
                      style={{ fontSize: '0.85rem' }}
                    />
                  </div>
                </div>
              </div>
            ))}
          </div>

          {/* Validator Configuration (Single) */}
          <div style={{ 
            marginBottom: '2rem', 
            padding: '1rem', 
            background: validatorProvider === 'openrouter' ? '#1a1a2e' : '#241a1a', 
            border: validatorProvider === 'openrouter' ? '2px solid #6c5ce7' : '1px solid #663333', 
            borderRadius: '8px' 
          }}>
            <h3 style={{ marginBottom: '1rem', color: validatorProvider === 'openrouter' ? '#a29bfe' : '#ff6b6b' }}>
              Validator Configuration (Single)
              {validatorProvider === 'openrouter' && <span style={{ fontWeight: 'normal', marginLeft: '0.5rem' }}>[OpenRouter]</span>}
            </h3>
            <small style={{ color: '#999', display: 'block', marginBottom: '1rem' }}>
              Only one validator is allowed to maintain single Markov chain evolution of the database.
            </small>

            <div style={{ display: 'grid', gridTemplateColumns: validatorProvider === 'openrouter' ? '1fr 1fr' : '1fr', gap: '1rem' }}>
              <ModelSelector
                provider={validatorProvider}
                modelId={config.validatorModel}
                openrouterProvider={validatorOpenrouterProvider}
                lmStudioFallbackId={validatorLmStudioFallback}
                onProviderChange={updateValidatorProvider}
                onModelChange={(m) => {
                  setConfig({ ...config, validatorModel: m });
                  if (validatorProvider === 'openrouter' && m) {
                    fetchProvidersForModel(m);
                  }
                }}
                onOpenrouterProviderChange={(p) => {
                  setValidatorOpenrouterProvider(p);
                  setConfig({ ...config, validatorOpenrouterProvider: p });
                }}
                onFallbackChange={(f) => {
                  setValidatorLmStudioFallback(f);
                  setConfig({ ...config, validatorLmStudioFallback: f });
                }}
                label="Validator Model"
              />
            </div>

            <div className="form-group" style={{ marginTop: '1rem' }}>
              <label>Validator Context Window Size (tokens)</label>
              <input
                type="number"
                value={config.validatorContextSize}
                onChange={(e) => {
                  const parsed = parseInt(e.target.value);
                  setConfig({ ...config, validatorContextSize: isNaN(parsed) ? 131072 : parsed });
                }}
                min="4096"
                max="999999"
                step="1024"
              />
              <small style={{ color: '#999', display: 'block', marginTop: '0.5rem' }}>
                {validatorProvider === 'lm_studio' 
                  ? 'Must match the context length you set in LM Studio for this model.'
                  : 'Set based on the OpenRouter model\'s context window.'
                }
              </small>
            </div>

            <div className="form-group">
              <label>
                Validator Max Output Tokens{' '}
                <span 
                  title="Default: 25000"
                  style={{ cursor: 'help', marginLeft: '0.5rem', color: '#888' }}
                >
                  ℹ️
                </span>
              </label>
              <input
                type="number"
                value={validatorMaxOutput}
                onChange={(e) => {
                  const parsed = parseInt(e.target.value);
                  const value = isNaN(parsed) ? 25000 : parsed;
                  setValidatorMaxOutput(value);
                  setConfig({ ...config, validatorMaxOutput: value });
                }}
                min="1000"
                max="100000"
                step="1000"
              />
            </div>
          </div>

          <button onClick={fetchModels} className="secondary" style={{ marginRight: '0.5rem' }}>
            Refresh LM Studio Models
          </button>
          {hasOpenRouterKey && (
            <>
              <button onClick={() => fetchOpenRouterModels(freeOnly)} className="secondary" disabled={loadingOpenRouter} style={{ marginRight: '0.5rem' }}>
                {loadingOpenRouter ? 'Loading...' : 'Refresh OpenRouter Models'}
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
              <div style={{ display: 'flex', flexDirection: 'column', gap: '0.4rem', marginTop: '0.5rem' }}>
                <label style={{ display: 'inline-flex', alignItems: 'center', fontSize: '0.9rem' }}>
                  <input
                    type="checkbox"
                    checked={freeModelLooping}
                    onChange={(e) => {
                      setFreeModelLooping(e.target.checked);
                      openRouterAPI.setFreeModelSettings(e.target.checked, freeModelAutoSelector).catch(() => {});
                    }}
                    style={{ marginRight: '0.5rem' }}
                  />
                  Enable Free Model Looping
                  <span
                    title="When a free model is rate-limited, automatically try the next available free model sorted by highest context limit. Prevents workflow stalls from rate limits."
                    style={{ marginLeft: '0.4rem', cursor: 'help', color: '#888', fontSize: '0.85rem' }}
                  >(?)</span>
                </label>
                <label style={{ display: 'inline-flex', alignItems: 'center', fontSize: '0.9rem' }}>
                  <input
                    type="checkbox"
                    checked={freeModelAutoSelector}
                    onChange={(e) => {
                      setFreeModelAutoSelector(e.target.checked);
                      openRouterAPI.setFreeModelSettings(freeModelLooping, e.target.checked).catch(() => {});
                    }}
                    style={{ marginRight: '0.5rem' }}
                  />
                  Use OpenRouter Free Models Auto-Selector as Backup
                  <span
                    title="When all selected free models are rate-limited, use OpenRouter's Free Models Router (openrouter/free) as a last resort backup. Works independently of Free Model Looping."
                    style={{ marginLeft: '0.4rem', cursor: 'help', color: '#888', fontSize: '0.85rem' }}
                  >(?)</span>
                </label>
              </div>
            </>
          )}
        </>
      )}

      {/* Current Configuration Summary */}
      <div style={{ marginTop: '2rem', padding: '1rem', background: '#1a1a1a', borderRadius: '6px' }}>
        <h3>Current Configuration Summary</h3>
        <pre style={{ color: '#4CAF50', fontSize: '0.85rem', overflow: 'auto' }}>
          {JSON.stringify({
            numSubmitters: submitterConfigs.length,
            submitterConfigs: submitterConfigs.map(s => ({
              id: s.submitterId,
              provider: s.provider,
              model: s.modelId?.split('/').pop() || 'Not selected',
              host: s.provider === 'openrouter' ? (s.openrouterProvider || 'Auto') : 'N/A',
              fallback: s.provider === 'openrouter' ? (s.lmStudioFallbackId?.split('/').pop() || 'None') : 'N/A',
              context: s.contextWindow,
              maxOutput: s.maxOutputTokens
            })),
            validator: {
              provider: validatorProvider,
              model: config.validatorModel?.split('/').pop() || 'Not selected',
              host: validatorProvider === 'openrouter' ? (validatorOpenrouterProvider || 'Auto') : 'N/A',
              fallback: validatorProvider === 'openrouter' ? (validatorLmStudioFallback?.split('/').pop() || 'None') : 'N/A',
              context: config.validatorContextSize,
              maxOutput: validatorMaxOutput
            },
            uploadedFiles: config.uploadedFiles?.length || 0
          }, null, 2)}
        </pre>
      </div>
    </div>
  );
}
