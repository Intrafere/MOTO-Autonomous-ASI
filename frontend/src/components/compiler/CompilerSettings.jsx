import React, { useState, useEffect } from 'react';
import { openRouterAPI, api, aggregatorAPI, compilerAPI } from '../../services/api';
import {
  computeOpenRouterAutoSettings,
  findOpenRouterModel,
  getProviderNames,
  hasEndpointMetadata,
} from '../../utils/openRouterSelection';
import HelpTooltip from '../HelpTooltip';
import '../settings-common.css';

const SETTINGS_KEY = 'compiler_settings';

function CompilerSettings({ capabilities }) {
  // LM Studio and OpenRouter models
  const [lmStudioModels, setLmStudioModels] = useState([]);
  const [openRouterModels, setOpenRouterModels] = useState([]);
  const [modelProviders, setModelProviders] = useState({});
  const [hasOpenRouterKey, setHasOpenRouterKey] = useState(false);
  const [loadingModels, setLoadingModels] = useState(true);
  const [freeOnly, setFreeOnly] = useState(false);
  const [freeModelLooping, setFreeModelLooping] = useState(true);
  const [freeModelAutoSelector, setFreeModelAutoSelector] = useState(true);

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
  const [hasStoredWolframKey, setHasStoredWolframKey] = useState(false);
  const [wolframTestResult, setWolframTestResult] = useState('');
  const [testingWolfram, setTestingWolfram] = useState(false);

  // Critique prompt editor state
  const [critiquePromptExpanded, setCritiquePromptExpanded] = useState(false);
  const [customCritiquePrompt, setCustomCritiquePrompt] = useState('');
  const [critiquePromptSaved, setCritiquePromptSaved] = useState(false);
  const [defaultCritiquePrompt, setDefaultCritiquePrompt] = useState('');
  const lmStudioEnabled = capabilities?.lmStudioEnabled !== false;
  const genericMode = Boolean(capabilities?.genericMode);

  const normalizeRoleState = (provider, model, openrouterProvider) => {
    const keepOpenRouterState = provider === 'openrouter';
    return {
      provider: 'openrouter',
      model: keepOpenRouterState ? (model || '') : '',
      openrouterProvider: keepOpenRouterState ? (openrouterProvider || null) : null,
      lmStudioFallback: null,
    };
  };

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
      if (lmStudioEnabled) {
        try {
          const models = await api.getModels();
          setLmStudioModels(models.models || models || []);
        } catch (err) {
          console.error('Failed to fetch LM Studio models:', err);
        }
      } else {
        setLmStudioModels([]);
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
          // Free-only toggle
          if (settings.freeOnly !== undefined) setFreeOnly(settings.freeOnly);
          if (settings.freeModelLooping !== undefined) setFreeModelLooping(settings.freeModelLooping);
          if (settings.freeModelAutoSelector !== undefined) setFreeModelAutoSelector(settings.freeModelAutoSelector);
          if (settings.modelProviders) setModelProviders(settings.modelProviders);
        } catch (error) {
          console.error('Failed to load compiler settings:', error);
        }
      }
      
      const loadWolframStatus = async () => {
        try {
          const response = await api.getWolframStatus();
          setHasStoredWolframKey(Boolean(response.has_key));
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
  }, [lmStudioEnabled]);

  useEffect(() => {
    if (lmStudioEnabled) {
      return;
    }

    setLmStudioModels([]);

    const nextValidator = normalizeRoleState(
      validatorProvider,
      validatorModel,
      validatorOpenrouterProvider
    );
    const nextHighContext = normalizeRoleState(
      highContextProvider,
      highContextModel,
      highContextOpenrouterProvider
    );
    const nextHighParam = normalizeRoleState(
      highParamProvider,
      highParamModel,
      highParamOpenrouterProvider
    );
    const nextCritique = normalizeRoleState(
      critiqueSubmitterProvider,
      critiqueSubmitterModel,
      critiqueSubmitterOpenrouterProvider
    );

    if (validatorProvider !== nextValidator.provider) setValidatorProvider(nextValidator.provider);
    if (validatorModel !== nextValidator.model) setValidatorModel(nextValidator.model);
    if (validatorOpenrouterProvider !== nextValidator.openrouterProvider) {
      setValidatorOpenrouterProvider(nextValidator.openrouterProvider);
    }
    if (validatorLmStudioFallback !== null) setValidatorLmStudioFallback(null);

    if (highContextProvider !== nextHighContext.provider) setHighContextProvider(nextHighContext.provider);
    if (highContextModel !== nextHighContext.model) setHighContextModel(nextHighContext.model);
    if (highContextOpenrouterProvider !== nextHighContext.openrouterProvider) {
      setHighContextOpenrouterProvider(nextHighContext.openrouterProvider);
    }
    if (highContextLmStudioFallback !== null) setHighContextLmStudioFallback(null);

    if (highParamProvider !== nextHighParam.provider) setHighParamProvider(nextHighParam.provider);
    if (highParamModel !== nextHighParam.model) setHighParamModel(nextHighParam.model);
    if (highParamOpenrouterProvider !== nextHighParam.openrouterProvider) {
      setHighParamOpenrouterProvider(nextHighParam.openrouterProvider);
    }
    if (highParamLmStudioFallback !== null) setHighParamLmStudioFallback(null);

    if (critiqueSubmitterProvider !== nextCritique.provider) {
      setCritiqueSubmitterProvider(nextCritique.provider);
    }
    if (critiqueSubmitterModel !== nextCritique.model) setCritiqueSubmitterModel(nextCritique.model);
    if (critiqueSubmitterOpenrouterProvider !== nextCritique.openrouterProvider) {
      setCritiqueSubmitterOpenrouterProvider(nextCritique.openrouterProvider);
    }
    if (critiqueSubmitterLmStudioFallback !== null) setCritiqueSubmitterLmStudioFallback(null);
  }, [
    lmStudioEnabled,
    validatorProvider,
    validatorModel,
    validatorOpenrouterProvider,
    validatorLmStudioFallback,
    highContextProvider,
    highContextModel,
    highContextOpenrouterProvider,
    highContextLmStudioFallback,
    highParamProvider,
    highParamModel,
    highParamOpenrouterProvider,
    highParamLmStudioFallback,
    critiqueSubmitterProvider,
    critiqueSubmitterModel,
    critiqueSubmitterOpenrouterProvider,
    critiqueSubmitterLmStudioFallback,
  ]);

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
      freeModelLooping,
      freeModelAutoSelector,
      modelProviders
    };
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
    freeOnly, freeModelLooping, freeModelAutoSelector, modelProviders
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
    if (!modelId) return null;

    const cachedProviderData = modelProviders[modelId];
    if (hasEndpointMetadata(cachedProviderData)) {
      return cachedProviderData;
    }

    try {
      const result = await openRouterAPI.getProviders(modelId);
      const providerData = {
        providers: result.providers || [],
        endpoints: result.endpoints || [],
      };
      setModelProviders(prev => ({ ...prev, [modelId]: providerData }));
      return providerData;
    } catch (err) {
      console.error(`Failed to fetch providers for ${modelId}:`, err);
      return cachedProviderData || null;
    }
  };

  const getAutoSettingsForModel = async (modelId, selectedProvider = null) => {
    const model = findOpenRouterModel(openRouterModels, modelId);
    if (!model) {
      console.debug('[CompilerAutoFill] model not in loaded list, skipping auto-fill', { modelId });
      return null;
    }

    const providerData = await fetchProvidersForModel(modelId);
    const autoSettings = computeOpenRouterAutoSettings(model, providerData, selectedProvider);
    if (autoSettings) {
      console.debug('[CompilerAutoFill] computed auto-settings', {
        modelId,
        selectedProvider,
        source: autoSettings.source,
        contextWindow: autoSettings.contextWindow,
        maxOutputTokens: autoSettings.maxOutputTokens,
        warnings: autoSettings.warnings,
      });
      if (autoSettings.warnings && autoSettings.warnings.length > 0) {
        console.warn('[CompilerAutoFill] auto-settings fallback used:', autoSettings.warnings);
      }
    }
    return autoSettings;
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
        await api.setWolframApiKey(wolframApiKey);
        setHasStoredWolframKey(true);
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
      setHasStoredWolframKey(false);
      setWolframTestResult('Key cleared');
      setTimeout(() => setWolframTestResult(''), 3000);
    } catch (err) {
      console.error('Failed to clear Wolfram Alpha key:', err);
    }
  };

  // Handler for "Use Aggregator Models" button
  const handleUseAggregatorModels = async () => {
    if (!lmStudioEnabled) {
      alert('Use Aggregator Models is unavailable when this deployment disables LM Studio.');
      return;
    }

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
    const effectiveProvider = lmStudioEnabled ? provider : 'openrouter';
    const models = effectiveProvider === 'openrouter' ? openRouterModels : lmStudioModels;
    const providers = model && effectiveProvider === 'openrouter'
      ? getProviderNames(modelProviders[model])
      : [];

    return (
      <div
        className={`role-config-card role-config-card--highlight${effectiveProvider === 'openrouter' ? ' role-config-card--openrouter' : ''}`}
        style={{ borderColor: effectiveProvider === 'openrouter' ? undefined : borderColor, padding: '1.5rem' }}
      >
        <h3 style={{ margin: '0 0 0.5rem 0', color: effectiveProvider === 'openrouter' ? '#18cc17' : borderColor }}>
          {title}
          {effectiveProvider === 'openrouter' && <span className="provider-badge-inline">[OpenRouter]</span>}
        </h3>
        <small className="role-description">{description}</small>

        {/* Provider Toggle */}
        <div className="form-group">
          <label>Provider</label>
          {lmStudioEnabled ? (
            <div className="provider-toggle-group">
              <button
                type="button"
                onClick={() => {
                  setProvider('lm_studio');
                  setModel('');
                  setOpenrouterProv(null);
                  setFallback(null);
                }}
                className={`provider-toggle-btn${provider === 'lm_studio' ? ' active-lm' : ''}`}
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
                className={`provider-toggle-btn${provider === 'openrouter' ? ' active-or' : ''}`}
                title={!hasOpenRouterKey ? 'Set OpenRouter API key first' : 'Use OpenRouter'}
              >
                OpenRouter
              </button>
            </div>
          ) : (
            <small className="hint-text hint-text--dim">
              OpenRouter is required in this deployment.
            </small>
          )}
        </div>

        {/* Model Selection */}
        <div className="form-group">
          <label>Model</label>
          <select
            value={model || ''}
            onChange={async (e) => {
              const m = e.target.value;
              setModel(m);
              setOpenrouterProv(null);
              if (effectiveProvider === 'openrouter' && m) {
                const autoSettings = await getAutoSettingsForModel(m, null);
                if (autoSettings) {
                  if (autoSettings.contextWindowKnown) {
                    setContextSize(autoSettings.contextWindow);
                  }
                  if (autoSettings.outputCapKnown) {
                    setMaxOutput(autoSettings.maxOutputTokens);
                  }
                }
              }
            }}
          >
            <option value="">Select model...</option>
            {models.map(m => {
              const isFree = effectiveProvider === 'openrouter' && 
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
        {effectiveProvider === 'openrouter' && model && (
          <div className="form-group">
            <label>Host Provider (optional)</label>
            <select
              value={openrouterProv || ''}
              onChange={async (e) => {
                const providerName = e.target.value || null;
                setOpenrouterProv(providerName);
                const autoSettings = await getAutoSettingsForModel(model, providerName);
                if (autoSettings) {
                  if (autoSettings.contextWindowKnown) {
                    setContextSize(autoSettings.contextWindow);
                  }
                  if (autoSettings.outputCapKnown) {
                    setMaxOutput(autoSettings.maxOutputTokens);
                  }
                }
              }}
            >
              <option value="">Auto (let OpenRouter choose)</option>
              {providers.map(p => (
                <option key={p} value={p}>{p}</option>
              ))}
            </select>
          </div>
        )}

        {/* LM Studio Fallback (if OpenRouter) */}
        {effectiveProvider === 'openrouter' && lmStudioEnabled && (
          <div className="form-group">
            <label className="label--muted">LM Studio Fallback (optional)</label>
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

        <div className="config-grid config-grid--2col">
          <div className="form-group form-group--compact">
            <label>Context Window (tokens)</label>
            <input
              type="number"
              value={contextSize}
              onChange={(e) => {
                const parsed = parseInt(e.target.value);
                setContextSize(isNaN(parsed) ? 131072 : parsed);
              }}
              min={4096}
              max={50000000}
              step={1024}
            />
          </div>

          <div className="form-group form-group--compact">
            <label>Max Output Tokens</label>
            <input
              type="number"
              value={maxOutput}
              onChange={(e) => {
                const parsed = parseInt(e.target.value);
                setMaxOutput(isNaN(parsed) ? 25000 : parsed);
              }}
              min={1000}
              max={50000000}
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
        <div className="save-message" style={{ marginBottom: '1rem' }}>
          {saveStatus}
        </div>
      )}

      {/* OpenRouter Status Banner */}
      {!hasOpenRouterKey && (
        <div className="openrouter-banner">
          <p className="openrouter-banner__text">
            <strong>💡 OpenRouter Available:</strong> Set your OpenRouter API key in the header to enable cloud model selection for any role.
          </p>
        </div>
      )}

      <div className="settings-section">
        <h3 className="section-heading--bordered">Model Configuration</h3>
        
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
          description="Rigor enhancement mode: adds citations, strengthens methodology, and clarifies assumptions."
          borderColor="#2a2a2a"
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
      <div className="settings-panel settings-panel--blue">
        <h3 style={{ marginBottom: '1rem' }}>Model Management</h3>
        <div className="model-refresh-controls">
          {lmStudioEnabled && (
            <>
              <button 
                onClick={handleUseAggregatorModels}
                className="secondary btn-primary-blue"
              >
                Use Aggregator Models
              </button>
              <button 
                onClick={async () => {
                  const models = await api.getModels();
                  setLmStudioModels(models.models || models || []);
                }} 
                className="secondary"
              >
                Refresh LM Studio Models
              </button>
            </>
          )}
          {hasOpenRouterKey && (
            <>
              <button onClick={() => fetchOpenRouterModels(freeOnly)} className="secondary">
                Refresh OpenRouter Models
              </button>
              <label className="settings-checkbox-label">
                <input
                  type="checkbox"
                  checked={freeOnly}
                  onChange={(e) => setFreeOnly(e.target.checked)}
                />
                Show only free models
              </label>
              <div className="checkbox-group-col">
                <label className="settings-checkbox-label">
                  <input
                    type="checkbox"
                    checked={freeModelLooping}
                    onChange={(e) => {
                      setFreeModelLooping(e.target.checked);
                      openRouterAPI.setFreeModelSettings(e.target.checked, freeModelAutoSelector).catch(() => {});
                    }}
                  />
                  Enable Free Model Looping
                  <HelpTooltip
                    label="Learn about free model looping"
                    anchorClassName="help-tooltip-anchor--inline"
                  >
                    When a free model is rate-limited, automatically try the next available free model sorted by highest context limit. Prevents workflow stalls from rate limits.
                  </HelpTooltip>
                </label>
                <label className="settings-checkbox-label">
                  <input
                    type="checkbox"
                    checked={freeModelAutoSelector}
                    onChange={(e) => {
                      setFreeModelAutoSelector(e.target.checked);
                      openRouterAPI.setFreeModelSettings(freeModelLooping, e.target.checked).catch(() => {});
                    }}
                  />
                  Use OpenRouter Free Models Auto-Selector as Backup
                  <HelpTooltip
                    label="Learn about the free models auto-selector backup"
                    anchorClassName="help-tooltip-anchor--inline"
                  >
                    When all selected free models are rate-limited, use OpenRouter&apos;s Free Models Router (`openrouter/free`) as a last resort backup. Works independently of Free Model Looping.
                  </HelpTooltip>
                </label>
              </div>
            </>
          )}
        </div>
        <small className="hint-text" style={{ marginTop: '0.75rem' }}>
          {lmStudioEnabled
            ? '"Use Aggregator Models" copies your aggregator\'s model selection to all compiler roles.'
            : 'LM Studio tools are hidden in hosted mode. Configure compiler roles directly with OpenRouter models below.'}
        </small>
      </div>

      {/* Wolfram Alpha Integration */}
      <div className="settings-section">
        <h3>Wolfram Alpha Integration (Optional)</h3>
        <small className="hint-text" style={{ marginBottom: '1rem' }}>
          Enable Wolfram Alpha API for computational verification in rigor mode. 
          Get your API key from <a href="https://products.wolframalpha.com/api" target="_blank" rel="noopener noreferrer">developer.wolframalpha.com</a>
        </small>
        
        <label className="settings-checkbox-label" style={{ marginBottom: '1rem' }}>
          <input
            type="checkbox"
            checked={wolframEnabled}
            onChange={async (e) => {
              const checked = e.target.checked;
              if (!checked) {
                await handleClearWolframKey();
              } else {
                setWolframEnabled(true);
              }
            }}
          />
          <span className="label-medium">Enable Wolfram Alpha Verification in Rigor Mode</span>
        </label>
        
        {wolframEnabled && (
          <div className="indented-section">
            <div className="form-group">
              <label>Wolfram Alpha API Key:</label>
              <input
                type="password"
                value={wolframApiKey}
                onChange={(e) => setWolframApiKey(e.target.value)}
                placeholder={
                  hasStoredWolframKey && !wolframApiKey
                    ? (
                      genericMode
                        ? 'Loaded in the current backend session. Enter a new App ID to replace it.'
                        : 'Stored securely on backend. Enter a new App ID to replace it.'
                    )
                    : 'Enter your Wolfram Alpha App ID'
                }
                className="input-dark"
                style={{ marginBottom: '0.5rem' }}
              />
              {hasStoredWolframKey && !wolframApiKey && (
                <small className="hint-text">
                  {genericMode
                    ? 'A Wolfram Alpha key is already loaded in the current backend session.'
                    : 'A Wolfram Alpha key is already stored securely on the backend for this machine.'}
                </small>
              )}
            </div>
            
            <div className="provider-toggle-group" style={{ marginTop: '0.75rem' }}>
              <button 
                onClick={handleTestWolframConnection}
                disabled={testingWolfram}
                className="btn-success-sm"
                style={{
                  cursor: testingWolfram ? 'wait' : 'pointer',
                  opacity: testingWolfram ? 0.6 : 1
                }}
              >
                {testingWolfram ? 'Testing...' : 'Test Connection'}
              </button>
              
              <button 
                onClick={handleClearWolframKey}
                className="btn-ghost"
              >
                Clear Key
              </button>
            </div>
            
            {wolframTestResult && (
              <div className={`test-result-banner ${wolframTestResult.includes('✓') ? 'test-result-banner--success' : 'test-result-banner--error'}`}>
                {wolframTestResult}
              </div>
            )}
            
            <small className="hint-text" style={{ marginTop: '1rem' }}>
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
      <div className="settings-panel settings-panel--blue">
        <div 
          onClick={() => setCritiquePromptExpanded(!critiquePromptExpanded)}
          className="collapsible-trigger"
          style={{ padding: '0.5rem 0', background: 'transparent', border: 'none' }}
        >
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
            <span style={{ fontSize: '1.1rem' }}>📝</span>
            <h3 style={{ margin: 0 }}>Edit Validator Critique Prompt</h3>
            {isUsingCustomCritiquePrompt && (
              <span className="tag-badge tag-badge--purple">CUSTOM</span>
            )}
          </div>
          <span className={`collapse-chevron${critiquePromptExpanded ? ' collapse-chevron--open' : ''}`}>▼</span>
        </div>

        {critiquePromptExpanded && (
          <div style={{ marginTop: '1rem' }}>
            <p className="text-muted-sm">
              Customize the prompt sent to your validator when requesting a paper critique. 
              The JSON output schema is automatically appended and cannot be modified.
            </p>

            <textarea
              value={customCritiquePrompt}
              onChange={(e) => setCustomCritiquePrompt(e.target.value)}
              className="textarea-dark-mono"
              placeholder="Enter your custom critique prompt..."
            />

            <div className="actions-row">
              <button
                onClick={handleRestoreCritiquePrompt}
                className="btn-ghost"
                style={{ fontSize: '0.85rem' }}
              >
                Restore to Default
              </button>

              <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
                {critiquePromptSaved && (
                  <span className="status-success-text">✓ Saved!</span>
                )}
                <button
                  onClick={handleSaveCritiquePrompt}
                  className="btn-accent-purple"
                >
                  Save Prompt
                </button>
              </div>
            </div>
          </div>
        )}
      </div>

      {/* Configuration Summary */}
      <div className="settings-panel" style={{ marginTop: '2rem' }}>
        <h3>Current Configuration Summary</h3>
        <pre className="config-summary-pre">
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
