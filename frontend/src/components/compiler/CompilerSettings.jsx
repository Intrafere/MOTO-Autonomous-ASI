import React, { useState, useEffect } from 'react';
import { openRouterAPI, api, aggregatorAPI } from '../../services/api';

const SETTINGS_KEY = 'compiler_settings';

function CompilerSettings() {
  // LM Studio and OpenRouter models
  const [lmStudioModels, setLmStudioModels] = useState([]);
  const [openRouterModels, setOpenRouterModels] = useState([]);
  const [modelProviders, setModelProviders] = useState({});
  const [hasOpenRouterKey, setHasOpenRouterKey] = useState(false);
  const [loadingModels, setLoadingModels] = useState(true);
  const [freeOnly, setFreeOnly] = useState(false);

  // Validator settings
  const [validatorProvider, setValidatorProvider] = useState('lm_studio');
  const [validatorModel, setValidatorModel] = useState('');
  const [validatorOpenrouterProvider, setValidatorOpenrouterProvider] = useState(null);
  const [validatorLmStudioFallback, setValidatorLmStudioFallback] = useState(null);
  const [validatorContextSize, setValidatorContextSize] = useState(131072);
  const [validatorMaxOutput, setValidatorMaxOutput] = useState(25000);

  // High-Context settings
  const [highContextProvider, setHighContextProvider] = useState('lm_studio');
  const [highContextModel, setHighContextModel] = useState('');
  const [highContextOpenrouterProvider, setHighContextOpenrouterProvider] = useState(null);
  const [highContextLmStudioFallback, setHighContextLmStudioFallback] = useState(null);
  const [highContextContextSize, setHighContextContextSize] = useState(131072);
  const [highContextMaxOutput, setHighContextMaxOutput] = useState(25000);

  // High-Param settings
  const [highParamProvider, setHighParamProvider] = useState('lm_studio');
  const [highParamModel, setHighParamModel] = useState('');
  const [highParamOpenrouterProvider, setHighParamOpenrouterProvider] = useState(null);
  const [highParamLmStudioFallback, setHighParamLmStudioFallback] = useState(null);
  const [highParamContextSize, setHighParamContextSize] = useState(131072);
  const [highParamMaxOutput, setHighParamMaxOutput] = useState(25000);

  // Critique Submitter settings
  const [critiqueSubmitterProvider, setCritiqueSubmitterProvider] = useState('lm_studio');
  const [critiqueSubmitterModel, setCritiqueSubmitterModel] = useState('');
  const [critiqueSubmitterOpenrouterProvider, setCritiqueSubmitterOpenrouterProvider] = useState(null);
  const [critiqueSubmitterLmStudioFallback, setCritiqueSubmitterLmStudioFallback] = useState(null);
  const [critiqueSubmitterContextSize, setCritiqueSubmitterContextSize] = useState(131072);
  const [critiqueSubmitterMaxOutput, setCritiqueSubmitterMaxOutput] = useState(25000);

  const [saveStatus, setSaveStatus] = useState('');
  const [isLoaded, setIsLoaded] = useState(false);

  // Wolfram Alpha settings
  const [wolframEnabled, setWolframEnabled] = useState(false);
  const [wolframApiKey, setWolframApiKey] = useState('');
  const [wolframTestResult, setWolframTestResult] = useState('');
  const [testingWolfram, setTestingWolfram] = useState(false);

  // Critique prompt editor state
  const [critiquePromptExpanded, setCritiquePromptExpanded] = useState(false);
  const [customCritiquePrompt, setCustomCritiquePrompt] = useState('');
  const [critiquePromptSaved, setCritiquePromptSaved] = useState(false);
  const [defaultCritiquePrompt, setDefaultCritiquePrompt] = useState('');

  // Load settings from localStorage on mount
  useEffect(() => {
    const loadSettings = async () => {
      // Check OpenRouter key status
      try {
        const status = await openRouterAPI.getApiKeyStatus();
        setHasOpenRouterKey(status.has_key);
        if (status.has_key) {
          fetchOpenRouterModels();
        }
      } catch (err) {
        console.error('Failed to check OpenRouter key:', err);
      }

      // Fetch LM Studio models
      try {
        const models = await api.getModels();
        setLmStudioModels(models);
      } catch (err) {
        console.error('Failed to fetch LM Studio models:', err);
      }

      // Load saved settings
      const savedSettings = localStorage.getItem(SETTINGS_KEY);
      if (savedSettings) {
        try {
          const settings = JSON.parse(savedSettings);
          // Validator
          if (settings.validatorProvider) setValidatorProvider(settings.validatorProvider);
          if (settings.validatorModel) setValidatorModel(settings.validatorModel);
          if (settings.validatorOpenrouterProvider) setValidatorOpenrouterProvider(settings.validatorOpenrouterProvider);
          if (settings.validatorLmStudioFallback) setValidatorLmStudioFallback(settings.validatorLmStudioFallback);
          if (settings.validatorContextSize) setValidatorContextSize(settings.validatorContextSize);
          if (settings.validatorMaxOutput) setValidatorMaxOutput(settings.validatorMaxOutput);
          // High-Context
          if (settings.highContextProvider) setHighContextProvider(settings.highContextProvider);
          if (settings.highContextModel) setHighContextModel(settings.highContextModel);
          if (settings.highContextOpenrouterProvider) setHighContextOpenrouterProvider(settings.highContextOpenrouterProvider);
          if (settings.highContextLmStudioFallback) setHighContextLmStudioFallback(settings.highContextLmStudioFallback);
          if (settings.highContextContextSize) setHighContextContextSize(settings.highContextContextSize);
          if (settings.highContextMaxOutput) setHighContextMaxOutput(settings.highContextMaxOutput);
          // High-Param
          if (settings.highParamProvider) setHighParamProvider(settings.highParamProvider);
          if (settings.highParamModel) setHighParamModel(settings.highParamModel);
          if (settings.highParamOpenrouterProvider) setHighParamOpenrouterProvider(settings.highParamOpenrouterProvider);
          if (settings.highParamLmStudioFallback) setHighParamLmStudioFallback(settings.highParamLmStudioFallback);
          if (settings.highParamContextSize) setHighParamContextSize(settings.highParamContextSize);
          if (settings.highParamMaxOutput) setHighParamMaxOutput(settings.highParamMaxOutput);
          // Critique Submitter
          if (settings.critiqueSubmitterProvider) setCritiqueSubmitterProvider(settings.critiqueSubmitterProvider);
          if (settings.critiqueSubmitterModel) setCritiqueSubmitterModel(settings.critiqueSubmitterModel);
          if (settings.critiqueSubmitterOpenrouterProvider) setCritiqueSubmitterOpenrouterProvider(settings.critiqueSubmitterOpenrouterProvider);
          if (settings.critiqueSubmitterLmStudioFallback) setCritiqueSubmitterLmStudioFallback(settings.critiqueSubmitterLmStudioFallback);
          if (settings.critiqueSubmitterContextSize) setCritiqueSubmitterContextSize(settings.critiqueSubmitterContextSize);
          if (settings.critiqueSubmitterMaxOutput) setCritiqueSubmitterMaxOutput(settings.critiqueSubmitterMaxOutput);
          // Wolfram Alpha
          if (settings.wolframEnabled !== undefined) setWolframEnabled(settings.wolframEnabled);
          // wolframApiKey not loaded from localStorage (sensitive data - must re-enter per session)
          // Free-only toggle
          if (settings.freeOnly !== undefined) setFreeOnly(settings.freeOnly);
          // Restore cached model providers
          if (settings.modelProviders) setModelProviders(settings.modelProviders);
        } catch (error) {
          console.error('Failed to load compiler settings:', error);
        }
      }
      
      // Load Wolfram Alpha status from backend
      const loadWolframStatus = async () => {
        try {
          const response = await api.getWolframStatus();
          if (response.enabled) {
            setWolframEnabled(true);
          }
        } catch (err) {
          console.error('Failed to load Wolfram Alpha status:', err);
        }
      };
      
      loadWolframStatus();
      
      setIsLoaded(true);
      setLoadingModels(false);
    };

    loadSettings();
  }, []);

  // Fetch providers for any OpenRouter models after settings are loaded
  useEffect(() => {
    if (!isLoaded || !hasOpenRouterKey) return;
    
    // Fetch providers for validator
    if (validatorProvider === 'openrouter' && validatorModel) {
      fetchProvidersForModel(validatorModel);
    }
    
    // Fetch providers for high-context
    if (highContextProvider === 'openrouter' && highContextModel) {
      fetchProvidersForModel(highContextModel);
    }
    
    // Fetch providers for high-param
    if (highParamProvider === 'openrouter' && highParamModel) {
      fetchProvidersForModel(highParamModel);
    }
    
    // Fetch providers for critique submitter
    if (critiqueSubmitterProvider === 'openrouter' && critiqueSubmitterModel) {
      fetchProvidersForModel(critiqueSubmitterModel);
    }
  }, [isLoaded, hasOpenRouterKey, validatorProvider, validatorModel, highContextProvider, highContextModel, highParamProvider, highParamModel, critiqueSubmitterProvider, critiqueSubmitterModel]);

  // Save settings to localStorage whenever values change
  useEffect(() => {
    if (!isLoaded) return;
    
    const settings = {
      validatorProvider, validatorModel, validatorOpenrouterProvider, validatorLmStudioFallback,
      validatorContextSize, validatorMaxOutput,
      highContextProvider, highContextModel, highContextOpenrouterProvider, highContextLmStudioFallback,
      highContextContextSize, highContextMaxOutput,
      highParamProvider, highParamModel, highParamOpenrouterProvider, highParamLmStudioFallback,
      highParamContextSize, highParamMaxOutput,
      critiqueSubmitterProvider, critiqueSubmitterModel, critiqueSubmitterOpenrouterProvider, critiqueSubmitterLmStudioFallback,
      critiqueSubmitterContextSize, critiqueSubmitterMaxOutput,
      wolframEnabled,
      freeOnly,
      modelProviders // Cache provider lists to avoid re-fetching
    };
    // Note: wolframApiKey intentionally excluded from localStorage (sensitive data)
    localStorage.setItem(SETTINGS_KEY, JSON.stringify(settings));
    setSaveStatus('Settings saved ✓');
    const timer = setTimeout(() => setSaveStatus(''), 2000);
    return () => clearTimeout(timer);
  }, [
    isLoaded, validatorProvider, validatorModel, validatorOpenrouterProvider, validatorLmStudioFallback,
    validatorContextSize, validatorMaxOutput,
    highContextProvider, highContextModel, highContextOpenrouterProvider, highContextLmStudioFallback,
    highContextContextSize, highContextMaxOutput,
    highParamProvider, highParamModel, highParamOpenrouterProvider, highParamLmStudioFallback,
    highParamContextSize, highParamMaxOutput,
    critiqueSubmitterProvider, critiqueSubmitterModel, critiqueSubmitterOpenrouterProvider, critiqueSubmitterLmStudioFallback,
    critiqueSubmitterContextSize, critiqueSubmitterMaxOutput,
    wolframEnabled,
    freeOnly, modelProviders
  ]);

  const fetchOpenRouterModels = async (freeFilter = freeOnly) => {
    try {
      const result = await openRouterAPI.getModels(null, freeFilter);
      setOpenRouterModels(result.models || []);
    } catch (err) {
      console.error('Failed to fetch OpenRouter models:', err);
    }
  };

  // Refetch models when free-only toggle changes
  useEffect(() => {
    if (hasOpenRouterKey && isLoaded) {
      fetchOpenRouterModels(freeOnly);
    }
  }, [freeOnly]);

  // Load critique prompt settings
  useEffect(() => {
    // Load custom prompt from localStorage
    const savedPrompt = localStorage.getItem('compiler_critique_custom_prompt');
    if (savedPrompt) {
      setCustomCritiquePrompt(savedPrompt);
    }
    
    // Fetch default prompt from backend
    const fetchDefaultPrompt = async () => {
      try {
        const { compilerAPI } = await import('../../services/api');
        const response = await compilerAPI.getDefaultCritiquePrompt();
        if (response.data?.prompt) {
          setDefaultCritiquePrompt(response.data.prompt);
          // If no custom prompt saved, use default
          if (!savedPrompt) {
            setCustomCritiquePrompt(response.data.prompt);
          }
        }
      } catch (err) {
        console.error('Failed to fetch default critique prompt:', err);
        // Fallback default prompt
        const fallback = `You are an expert academic reviewer providing an honest, thorough critique of a research paper.

Evaluate this paper and provide:
1. NOVELTY (1-10): How original and innovative is this work?
2. CORRECTNESS (1-10): How mathematically/logically sound is the content?
3. IMPACT ON RELATED FIELD (1-10): How significant could this contribution be?

For each category, provide the numeric rating (1-10) and detailed feedback explaining your assessment.

Be honest and constructive. Identify both strengths and weaknesses.`;
        setDefaultCritiquePrompt(fallback);
        if (!savedPrompt) {
          setCustomCritiquePrompt(fallback);
        }
      }
    };
    fetchDefaultPrompt();
  }, []);

  const fetchProvidersForModel = async (modelId) => {
    if (!modelId || modelProviders[modelId]) return;
    try {
      const result = await openRouterAPI.getProviders(modelId);
      setModelProviders(prev => ({ ...prev, [modelId]: result.providers || [] }));
    } catch (err) {
      console.error(`Failed to fetch providers for ${modelId}:`, err);
    }
  };

  // Critique prompt handlers
  const handleSaveCritiquePrompt = () => {
    localStorage.setItem('compiler_critique_custom_prompt', customCritiquePrompt);
    setCritiquePromptSaved(true);
    setTimeout(() => setCritiquePromptSaved(false), 2000);
  };

  const handleRestoreCritiquePrompt = () => {
    localStorage.removeItem('compiler_critique_custom_prompt');
    setCustomCritiquePrompt(defaultCritiquePrompt);
    setCritiquePromptSaved(false);
  };

  const isUsingCustomCritiquePrompt = customCritiquePrompt && customCritiquePrompt !== defaultCritiquePrompt;

  // Wolfram Alpha handlers
  const handleTestWolframConnection = async () => {
    if (!wolframApiKey.trim()) {
      setWolframTestResult('Please enter an API key');
      return;
    }
    
    setTestingWolfram(true);
    setWolframTestResult('Testing...');
    
    try {
      const response = await api.testWolframQuery({
        query: 'What is 2+2?',
        api_key: wolframApiKey
      });
      
      if (response.success) {
        setWolframTestResult(`✓ Success! Result: ${response.result}`);
        // Save the key to backend
        await api.setWolframApiKey(wolframApiKey);
        setWolframEnabled(true);
      } else {
        setWolframTestResult('✗ Failed: ' + response.message);
      }
    } catch (err) {
      setWolframTestResult('✗ Error: ' + err.message);
    } finally {
      setTestingWolfram(false);
      setTimeout(() => setWolframTestResult(''), 5000);
    }
  };
  
  const handleClearWolframKey = async () => {
    try {
      await api.clearWolframApiKey();
      setWolframApiKey('');
      setWolframEnabled(false);
      setWolframTestResult('Key cleared');
      setTimeout(() => setWolframTestResult(''), 3000);
    } catch (err) {
      console.error('Failed to clear Wolfram Alpha key:', err);
    }
  };

  // Handler for "Use Aggregator Models" button
  const handleUseAggregatorModels = async () => {
    try {
      const response = await aggregatorAPI.getSettings();
      const settings = response.data;
      
      if (settings.submitter_model) {
        // Set all models to use the aggregator's model configuration
        setValidatorProvider('lm_studio');
        setValidatorModel(settings.validator_model || settings.submitter_model);
        setValidatorOpenrouterProvider(null);
        setValidatorLmStudioFallback(null);
        
        setHighContextProvider('lm_studio');
        setHighContextModel(settings.submitter_model);
        setHighContextOpenrouterProvider(null);
        setHighContextLmStudioFallback(null);
        
        setHighParamProvider('lm_studio');
        setHighParamModel(settings.submitter_model);
        setHighParamOpenrouterProvider(null);
        setHighParamLmStudioFallback(null);
        
        setCritiqueSubmitterProvider('lm_studio');
        setCritiqueSubmitterModel(settings.submitter_model);
        setCritiqueSubmitterOpenrouterProvider(null);
        setCritiqueSubmitterLmStudioFallback(null);
        
        alert('Successfully loaded aggregator models for all roles!');
      } else {
        alert('Aggregator is not running yet. Please start the aggregator first.');
      }
    } catch (err) {
      console.error('Failed to load aggregator settings:', err);
      alert('Failed to load aggregator settings: ' + err.message);
    }
  };

  // Reusable Role Configuration Component
  const RoleConfig = ({ 
    title, 
    description, 
    provider, setProvider,
    model, setModel,
    openrouterProv, setOpenrouterProv,
    fallback, setFallback,
    contextSize, setContextSize,
    maxOutput, setMaxOutput,
    borderColor = '#333'
  }) => {
    const models = provider === 'openrouter' ? openRouterModels : lmStudioModels;
    const providers = model && provider === 'openrouter' ? (modelProviders[model] || []) : [];

    return (
      <div style={{
        marginBottom: '2rem',
        padding: '1.5rem',
        background: provider === 'openrouter' ? '#1a1a2e' : '#1a1a24',
        border: `2px solid ${provider === 'openrouter' ? '#6c5ce7' : borderColor}`,
        borderRadius: '8px'
      }}>
        <h3 style={{ 
          margin: '0 0 0.5rem 0', 
          color: provider === 'openrouter' ? '#a29bfe' : borderColor 
        }}>
          {title}
          {provider === 'openrouter' && <span style={{ fontWeight: 'normal', marginLeft: '0.5rem' }}>[OpenRouter]</span>}
        </h3>
        <small style={{ color: '#888', display: 'block', marginBottom: '1rem' }}>{description}</small>

        {/* Provider Toggle */}
        <div className="form-group">
          <label>Provider</label>
          <div style={{ display: 'flex', gap: '0.5rem' }}>
            <button
              type="button"
              onClick={() => {
                setProvider('lm_studio');
                setModel('');
                setOpenrouterProv(null);
                setFallback(null);
              }}
              style={{
                flex: 1,
                padding: '0.5rem',
                backgroundColor: provider === 'lm_studio' ? '#4CAF50' : '#333',
                border: 'none',
                borderRadius: '4px',
                color: '#fff',
                cursor: 'pointer'
              }}
            >
              LM Studio
            </button>
            <button
              type="button"
              onClick={() => {
                if (hasOpenRouterKey) {
                  setProvider('openrouter');
                  setModel('');
                  setOpenrouterProv(null);
                  setFallback(null);
                }
              }}
              disabled={!hasOpenRouterKey}
              style={{
                flex: 1,
                padding: '0.5rem',
                backgroundColor: provider === 'openrouter' ? '#6c5ce7' : '#333',
                border: 'none',
                borderRadius: '4px',
                color: hasOpenRouterKey ? '#fff' : '#666',
                cursor: hasOpenRouterKey ? 'pointer' : 'not-allowed'
              }}
              title={!hasOpenRouterKey ? 'Set OpenRouter API key first' : 'Use OpenRouter'}
            >
              OpenRouter
            </button>
          </div>
        </div>

        {/* Model Selection */}
        <div className="form-group">
          <label>Model</label>
          <select
            value={model || ''}
            onChange={(e) => {
              const m = e.target.value;
              setModel(m);
              if (provider === 'openrouter' && m) {
                fetchProvidersForModel(m);
              }
            }}
          >
            <option value="">Select model...</option>
            {models.map(m => {
              const isFree = provider === 'openrouter' && 
                            m.pricing?.prompt === "0" && 
                            m.pricing?.completion === "0";
              const displayName = m.name || m.id;
              const contextInfo = m.context_length ? ` (${Math.round(m.context_length/1000)}K)` : '';
              
              return (
                <option key={m.id} value={m.id}>
                  {displayName}{contextInfo}{isFree ? ' [FREE]' : ''}
                </option>
              );
            })}
          </select>
        </div>

        {/* OpenRouter Provider (if OpenRouter) */}
        {provider === 'openrouter' && model && (
          <div className="form-group">
            <label>Host Provider (optional)</label>
            <select
              value={openrouterProv || ''}
              onChange={(e) => setOpenrouterProv(e.target.value || null)}
            >
              <option value="">Auto (let OpenRouter choose)</option>
              {providers.map(p => (
                <option key={p} value={p}>{p}</option>
              ))}
            </select>
          </div>
        )}

        {/* LM Studio Fallback (if OpenRouter) */}
        {provider === 'openrouter' && (
          <div className="form-group">
            <label style={{ color: '#888' }}>LM Studio Fallback (optional)</label>
            <select
              value={fallback || ''}
              onChange={(e) => setFallback(e.target.value || null)}
            >
              <option value="">No fallback</option>
              {lmStudioModels.map(m => (
                <option key={m.id} value={m.id}>{m.id}</option>
              ))}
            </select>
            <small>Used if OpenRouter credits run out</small>
          </div>
        )}

        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1rem' }}>
          <div className="form-group" style={{ margin: 0 }}>
            <label>Context Window (tokens)</label>
            <input
              type="number"
              value={contextSize}
              onChange={(e) => {
                const parsed = parseInt(e.target.value);
                setContextSize(isNaN(parsed) ? 131072 : parsed);
              }}
              min={4096}
              max={999999}
              step={1024}
            />
          </div>

          <div className="form-group" style={{ margin: 0 }}>
            <label>Max Output Tokens</label>
            <input
              type="number"
              value={maxOutput}
              onChange={(e) => {
                const parsed = parseInt(e.target.value);
                setMaxOutput(isNaN(parsed) ? 25000 : parsed);
              }}
              min={1000}
              max={100000}
              step={1000}
            />
          </div>
        </div>
      </div>
    );
  };

  if (loadingModels) {
    return <div>Loading models...</div>;
  }

  return (
    <div className="compiler-settings">
      <h2>Compiler Settings</h2>

      {saveStatus && (
        <div className="save-status" style={{ color: '#4CAF50', marginBottom: '1rem' }}>
          {saveStatus}
        </div>
      )}

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

      <div className="settings-section">
        <h3 style={{ borderBottom: '1px solid #333', paddingBottom: '0.5rem' }}>Model Configuration</h3>
        
        <RoleConfig
          title="Validator"
          description="Validates all submissions for coherence, rigor, placement, and non-redundancy."
          borderColor="#ff6b6b"
          provider={validatorProvider} setProvider={setValidatorProvider}
          model={validatorModel} setModel={setValidatorModel}
          openrouterProv={validatorOpenrouterProvider} setOpenrouterProv={setValidatorOpenrouterProvider}
          fallback={validatorLmStudioFallback} setFallback={setValidatorLmStudioFallback}
          contextSize={validatorContextSize} setContextSize={setValidatorContextSize}
          maxOutput={validatorMaxOutput} setMaxOutput={setValidatorMaxOutput}
        />

        <RoleConfig
          title="High-Context Model"
          description="Handles construction, outline creation/updates, and review modes. Needs large context for comprehensive outlines."
          borderColor="#4CAF50"
          provider={highContextProvider} setProvider={setHighContextProvider}
          model={highContextModel} setModel={setHighContextModel}
          openrouterProv={highContextOpenrouterProvider} setOpenrouterProv={setHighContextOpenrouterProvider}
          fallback={highContextLmStudioFallback} setFallback={setHighContextLmStudioFallback}
          contextSize={highContextContextSize} setContextSize={setHighContextContextSize}
          maxOutput={highContextMaxOutput} setMaxOutput={setHighContextMaxOutput}
        />

        <RoleConfig
          title="High-Parameter Model"
          description="Rigor enhancement mode: adds citations, strengthens methodology, clarifies assumptions."
          borderColor="#f1c40f"
          provider={highParamProvider} setProvider={setHighParamProvider}
          model={highParamModel} setModel={setHighParamModel}
          openrouterProv={highParamOpenrouterProvider} setOpenrouterProv={setHighParamOpenrouterProvider}
          fallback={highParamLmStudioFallback} setFallback={setHighParamLmStudioFallback}
          contextSize={highParamContextSize} setContextSize={setHighParamContextSize}
          maxOutput={highParamMaxOutput} setMaxOutput={setHighParamMaxOutput}
        />

        <RoleConfig
          title="Critique Submitter"
          description="Generates peer review critiques and decides on rewrites after body completion."
          borderColor="#e74c3c"
          provider={critiqueSubmitterProvider} setProvider={setCritiqueSubmitterProvider}
          model={critiqueSubmitterModel} setModel={setCritiqueSubmitterModel}
          openrouterProv={critiqueSubmitterOpenrouterProvider} setOpenrouterProv={setCritiqueSubmitterOpenrouterProvider}
          fallback={critiqueSubmitterLmStudioFallback} setFallback={setCritiqueSubmitterLmStudioFallback}
          contextSize={critiqueSubmitterContextSize} setContextSize={setCritiqueSubmitterContextSize}
          maxOutput={critiqueSubmitterMaxOutput} setMaxOutput={setCritiqueSubmitterMaxOutput}
        />
      </div>

      {/* Model Refresh Controls */}
      <div style={{ marginBottom: '2rem', padding: '1rem', background: '#1a1a24', borderRadius: '8px' }}>
        <h3 style={{ marginBottom: '1rem' }}>Model Management</h3>
        <div style={{ display: 'flex', alignItems: 'center', gap: '1rem', flexWrap: 'wrap' }}>
          <button 
            onClick={handleUseAggregatorModels}
            className="secondary"
            style={{
              backgroundColor: '#2196F3',
              border: 'none',
              color: '#fff'
            }}
          >
            Use Aggregator Models
          </button>
          <button 
            onClick={async () => {
              const models = await api.getModels();
              setLmStudioModels(models);
            }} 
            className="secondary"
          >
            Refresh LM Studio Models
          </button>
          {hasOpenRouterKey && (
            <>
              <button onClick={() => fetchOpenRouterModels(freeOnly)} className="secondary">
                Refresh OpenRouter Models
              </button>
              <label style={{ display: 'inline-flex', alignItems: 'center', fontSize: '0.9rem' }}>
                <input
                  type="checkbox"
                  checked={freeOnly}
                  onChange={(e) => setFreeOnly(e.target.checked)}
                  style={{ marginRight: '0.5rem' }}
                />
                Show free models only
              </label>
            </>
          )}
        </div>
        <small style={{ color: '#888', display: 'block', marginTop: '0.75rem' }}>
          "Use Aggregator Models" copies your aggregator's model selection to all compiler roles.
        </small>
      </div>

      {/* Wolfram Alpha Integration */}
      <div className="settings-section">
        <h3>Wolfram Alpha Integration (Optional)</h3>
        <small style={{ color: '#888', display: 'block', marginBottom: '1rem' }}>
          Enable Wolfram Alpha API for computational verification in rigor mode. 
          Get your API key from <a href="https://products.wolframalpha.com/api" target="_blank" rel="noopener noreferrer">developer.wolframalpha.com</a>
        </small>
        
        <label style={{ display: 'flex', alignItems: 'center', marginBottom: '1rem' }}>
          <input
            type="checkbox"
            checked={wolframEnabled}
            onChange={async (e) => {
              const checked = e.target.checked;
              if (!checked) {
                // Unchecking - clear key from backend
                await handleClearWolframKey();
              } else {
                // Checking - just show UI (key will be saved on Test Connection)
                setWolframEnabled(true);
              }
            }}
            style={{ marginRight: '0.75rem' }}
          />
          <span style={{ fontWeight: '500' }}>Enable Wolfram Alpha Verification in Rigor Mode</span>
        </label>
        
        {wolframEnabled && (
          <div style={{ marginLeft: '1.75rem' }}>
            <div className="form-group">
              <label>Wolfram Alpha API Key:</label>
              <input
                type="password"
                value={wolframApiKey}
                onChange={(e) => setWolframApiKey(e.target.value)}
                placeholder="Enter your Wolfram Alpha App ID"
                style={{
                  padding: '0.5rem',
                  backgroundColor: '#2a2a2a',
                  border: '1px solid #444',
                  borderRadius: '4px',
                  color: '#fff',
                  width: '100%',
                  marginBottom: '0.5rem'
                }}
              />
            </div>
            
            <div style={{ display: 'flex', gap: '0.5rem', marginTop: '0.75rem' }}>
              <button 
                onClick={handleTestWolframConnection}
                disabled={testingWolfram}
                style={{
                  padding: '0.5rem 1rem',
                  backgroundColor: '#4CAF50',
                  border: 'none',
                  borderRadius: '4px',
                  color: '#fff',
                  cursor: testingWolfram ? 'wait' : 'pointer',
                  opacity: testingWolfram ? 0.6 : 1
                }}
              >
                {testingWolfram ? 'Testing...' : 'Test Connection'}
              </button>
              
              <button 
                onClick={handleClearWolframKey}
                style={{
                  padding: '0.5rem 1rem',
                  backgroundColor: 'transparent',
                  border: '1px solid #666',
                  borderRadius: '4px',
                  color: '#888',
                  cursor: 'pointer'
                }}
              >
                Clear Key
              </button>
            </div>
            
            {wolframTestResult && (
              <div style={{ 
                marginTop: '0.75rem', 
                padding: '0.5rem', 
                borderRadius: '4px',
                backgroundColor: wolframTestResult.includes('✓') ? '#1a3a1a' : '#3a1a1a',
                color: wolframTestResult.includes('✓') ? '#4CAF50' : '#ff6b6b',
                fontSize: '0.85rem'
              }}>
                {wolframTestResult}
              </div>
            )}
            
            <small style={{ color: '#888', display: 'block', marginTop: '1rem' }}>
              In rigor mode, the AI can request Wolfram Alpha verification of mathematical claims. 
              This enables computational checking of theorems, solving equations, and verifying properties.
            </small>
          </div>
        )}
      </div>

      <div className="settings-section">
        <h3>Workflow Configuration</h3>
        
        <div className="info-box">
          <h4>Sequential Markov Chain</h4>
          <p>The compiler runs one submitter at a time in sequential order:</p>
          <ol>
            <li><strong>Outline Creation:</strong> Generate initial outline (loops until accepted)</li>
            <li><strong>Paper Construction:</strong> Write paper sections following outline</li>
            <li><strong>Outline Updates:</strong> Periodically review and update outline</li>
            <li><strong>Paper Review:</strong> Clean up errors and redundancy</li>
            <li><strong>Rigor Enhancement:</strong> Add scientific rigor (loops until rejection)</li>
          </ol>
        </div>
      </div>

      {/* Validator Critique Prompt Editor */}
      <div style={{ marginBottom: '2rem', padding: '1rem', background: '#1a1a24', borderRadius: '8px' }}>
        <div 
          onClick={() => setCritiquePromptExpanded(!critiquePromptExpanded)}
          style={{
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'space-between',
            cursor: 'pointer',
            padding: '0.5rem 0'
          }}
        >
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
            <span style={{ fontSize: '1.1rem' }}>📝</span>
            <h3 style={{ margin: 0 }}>Edit Validator Critique Prompt</h3>
            {isUsingCustomCritiquePrompt && (
              <span style={{
                backgroundColor: '#9b59b6',
                color: '#fff',
                padding: '2px 8px',
                borderRadius: '12px',
                fontSize: '0.7rem',
                fontWeight: 'bold'
              }}>CUSTOM</span>
            )}
          </div>
          <span style={{ 
            transform: critiquePromptExpanded ? 'rotate(180deg)' : 'rotate(0deg)',
            transition: 'transform 0.2s',
            fontSize: '1.2rem'
          }}>▼</span>
        </div>

        {critiquePromptExpanded && (
          <div style={{ marginTop: '1rem' }}>
            <p style={{ color: '#888', fontSize: '0.85rem', marginBottom: '1rem' }}>
              Customize the prompt sent to your validator when requesting a paper critique. 
              The JSON output schema is automatically appended and cannot be modified.
            </p>

            <textarea
              value={customCritiquePrompt}
              onChange={(e) => setCustomCritiquePrompt(e.target.value)}
              style={{
                width: '100%',
                minHeight: '200px',
                padding: '0.75rem',
                backgroundColor: '#2a2a2a',
                border: '1px solid #444',
                borderRadius: '4px',
                color: '#fff',
                fontFamily: 'monospace',
                fontSize: '0.85rem',
                resize: 'vertical',
                lineHeight: '1.5'
              }}
              placeholder="Enter your custom critique prompt..."
            />

            <div style={{ 
              display: 'flex', 
              justifyContent: 'space-between', 
              alignItems: 'center',
              marginTop: '1rem' 
            }}>
              <button
                onClick={handleRestoreCritiquePrompt}
                style={{
                  padding: '0.5rem 1rem',
                  backgroundColor: 'transparent',
                  border: '1px solid #666',
                  borderRadius: '4px',
                  color: '#888',
                  cursor: 'pointer',
                  fontSize: '0.85rem'
                }}
              >
                Restore to Default
              </button>

              <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
                {critiquePromptSaved && (
                  <span style={{ color: '#4CAF50', fontSize: '0.85rem' }}>✓ Saved!</span>
                )}
                <button
                  onClick={handleSaveCritiquePrompt}
                  style={{
                    padding: '0.5rem 1.5rem',
                    backgroundColor: '#9b59b6',
                    border: 'none',
                    borderRadius: '4px',
                    color: '#fff',
                    cursor: 'pointer',
                    fontWeight: '500',
                    fontSize: '0.85rem'
                  }}
                >
                  Save Prompt
                </button>
              </div>
            </div>
          </div>
        )}
      </div>

      {/* Configuration Summary */}
      <div style={{ marginTop: '2rem', padding: '1rem', background: '#1a1a1a', borderRadius: '6px' }}>
        <h3>Current Configuration Summary</h3>
        <pre style={{ color: '#4CAF50', fontSize: '0.85rem', overflow: 'auto' }}>
          {JSON.stringify({
            validator: {
              provider: validatorProvider,
              model: validatorModel?.split('/').pop() || 'Not selected',
              host: validatorProvider === 'openrouter' ? (validatorOpenrouterProvider || 'Auto') : 'N/A',
              fallback: validatorProvider === 'openrouter' ? (validatorLmStudioFallback?.split('/').pop() || 'None') : 'N/A',
              context: validatorContextSize,
              maxOutput: validatorMaxOutput
            },
            highContext: {
              provider: highContextProvider,
              model: highContextModel?.split('/').pop() || 'Not selected',
              host: highContextProvider === 'openrouter' ? (highContextOpenrouterProvider || 'Auto') : 'N/A',
              fallback: highContextProvider === 'openrouter' ? (highContextLmStudioFallback?.split('/').pop() || 'None') : 'N/A',
              context: highContextContextSize,
              maxOutput: highContextMaxOutput
            },
            highParam: {
              provider: highParamProvider,
              model: highParamModel?.split('/').pop() || 'Not selected',
              host: highParamProvider === 'openrouter' ? (highParamOpenrouterProvider || 'Auto') : 'N/A',
              fallback: highParamProvider === 'openrouter' ? (highParamLmStudioFallback?.split('/').pop() || 'None') : 'N/A',
              context: highParamContextSize,
              maxOutput: highParamMaxOutput
            },
            critiqueSubmitter: {
              provider: critiqueSubmitterProvider,
              model: critiqueSubmitterModel?.split('/').pop() || 'Not selected',
              host: critiqueSubmitterProvider === 'openrouter' ? (critiqueSubmitterOpenrouterProvider || 'Auto') : 'N/A',
              fallback: critiqueSubmitterProvider === 'openrouter' ? (critiqueSubmitterLmStudioFallback?.split('/').pop() || 'None') : 'N/A',
              context: critiqueSubmitterContextSize,
              maxOutput: critiqueSubmitterMaxOutput
            }
          }, null, 2)}
        </pre>
      </div>
    </div>
  );
}

export default CompilerSettings;
