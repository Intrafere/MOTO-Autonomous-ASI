/**
 * AutonomousResearchSettings - Settings panel for autonomous research mode.
 * Supports configurable multi-submitter brainstorm aggregation (1-10 submitters).
 * Compiler settings (high-context, high-param) remain separate.
 * Now supports per-role OpenRouter model selection with provider and fallback options.
 */
import React, { useState, useEffect } from 'react';
import { openRouterAPI, api } from '../../services/api';
import { loadModelCache, getModelApiId } from '../../utils/modelCache';
import './AutonomousResearch.css';

const DEFAULT_SUBMITTER_CONFIG = {
  submitterId: 1,
  provider: 'lm_studio',
  modelId: '',
  openrouterProvider: null,
  lmStudioFallbackId: null,
  contextWindow: 131072,
  maxOutputTokens: 25000
};

// Single recommended profile with hard-coded model IDs (NO pattern matching)
const RECOMMENDED_PROFILES = {
  'recommended_fastest_cheapest': {
    name: 'Recommended - Fastest, cheapest, lowest knowledge',
    numSubmitters: 3,
    submitters: [
      { 
        modelId: 'openai/gpt-oss-120b',
        provider: 'openrouter',
        openrouterProvider: 'Google',
        lmStudioFallbackId: null,
        contextWindow: 131000,
        maxOutputTokens: 25000
      },
      { 
        modelId: 'openai/gpt-oss-20b',
        provider: 'openrouter',
        openrouterProvider: 'Groq',
        lmStudioFallbackId: null,
        contextWindow: 131000,
        maxOutputTokens: 25000
      },
      { 
        modelId: 'openai/gpt-oss-120b',
        provider: 'openrouter',
        openrouterProvider: 'Google',
        lmStudioFallbackId: null,
        contextWindow: 131000,
        maxOutputTokens: 25000
      }
    ],
    validator: { 
      modelId: 'openai/gpt-oss-120b',
      provider: 'openrouter',
      openrouterProvider: 'Google',
      lmStudioFallbackId: null,
      contextWindow: 131000,
      maxOutputTokens: 25000
    },
    highContext: { 
      modelId: 'openai/gpt-oss-120b',
      provider: 'openrouter',
      openrouterProvider: 'Google',
      lmStudioFallbackId: null,
      contextWindow: 131000,
      maxOutputTokens: 25000
    },
    highParam: { 
      modelId: 'openai/gpt-oss-120b',
      provider: 'openrouter',
      openrouterProvider: 'Google',
      lmStudioFallbackId: null,
      contextWindow: 131000,
      maxOutputTokens: 25000
    },
    critique: { 
      modelId: 'openai/gpt-oss-120b',
      provider: 'openrouter',
      openrouterProvider: 'Google',
      lmStudioFallbackId: null,
      contextWindow: 131000,
      maxOutputTokens: 25000
    }
  }
};

// ModelSelector component - extracted outside to prevent recreation on every render
const ModelSelector = ({ provider, modelId, openrouterProv, fallback, onProviderChange, onModelChange, onOpenrouterProviderChange, onFallbackChange, lmStudioModels, openRouterModels, modelProviders, hasOpenRouterKey, isRunning }) => {
  const currentModels = provider === 'openrouter' ? openRouterModels : lmStudioModels;
  const providers = modelId && provider === 'openrouter' ? (modelProviders[modelId] || []) : [];

  return (
    <>
      {/* Provider Toggle */}
      <div className="settings-row">
        <label>Provider</label>
        <div style={{ display: 'flex', gap: '0.5rem' }}>
          <button
            type="button"
            onClick={() => onProviderChange('lm_studio')}
            disabled={isRunning}
            style={{
              flex: 1,
              padding: '0.5rem',
              backgroundColor: provider === 'lm_studio' ? '#4CAF50' : '#333',
              border: 'none',
              borderRadius: '4px',
              color: '#fff',
              cursor: isRunning ? 'not-allowed' : 'pointer',
              opacity: isRunning ? 0.6 : 1
            }}
          >
            LM Studio
          </button>
          <button
            type="button"
            onClick={() => hasOpenRouterKey && onProviderChange('openrouter')}
            disabled={isRunning || !hasOpenRouterKey}
            style={{
              flex: 1,
              padding: '0.5rem',
              backgroundColor: provider === 'openrouter' ? '#FF6700' : '#333',
              border: 'none',
              borderRadius: '4px',
              color: hasOpenRouterKey ? '#fff' : '#666',
              cursor: (isRunning || !hasOpenRouterKey) ? 'not-allowed' : 'pointer',
              opacity: (isRunning || !hasOpenRouterKey) ? 0.6 : 1
            }}
            title={!hasOpenRouterKey ? 'Set OpenRouter API key first' : 'Use OpenRouter'}
          >
            OpenRouter
          </button>
        </div>
      </div>

      {/* Model Selection */}
      <div className="settings-row">
        <label>Model</label>
        <select
          value={modelId || ''}
          onChange={(e) => onModelChange(e.target.value)}
          disabled={isRunning}
        >
          <option value="">Select model...</option>
          {currentModels.map(m => {
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
      {provider === 'openrouter' && modelId && (
        <div className="settings-row">
          <label>Host Provider (optional)</label>
          <select
            value={openrouterProv || ''}
            onChange={(e) => onOpenrouterProviderChange(e.target.value || null)}
            disabled={isRunning}
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
        <div className="settings-row">
          <label style={{ color: '#888' }}>LM Studio Fallback (optional)</label>
          <select
            value={fallback || ''}
            onChange={(e) => onFallbackChange(e.target.value || null)}
            disabled={isRunning}
          >
            <option value="">No fallback</option>
            {lmStudioModels.map(m => (
              <option key={m.id} value={m.id}>{m.id}</option>
            ))}
          </select>
          <small className="settings-hint">Used if OpenRouter credits run out</small>
        </div>
      )}
    </>
  );
};

// RoleConfig component - extracted outside to prevent recreation on every render
const RoleConfig = ({ title, hint, rolePrefix, borderColor = '#333', localConfig, handleProviderChange, handleModelChange, handleChange, handleNumericBlur, isRunning, lmStudioModels, openRouterModels, modelProviders, hasOpenRouterKey }) => {
  const provider = localConfig[`${rolePrefix}_provider`] || 'lm_studio';
  const modelId = localConfig[`${rolePrefix}_model`] || '';
  const openrouterProv = localConfig[`${rolePrefix}_openrouter_provider`];
  const fallback = localConfig[`${rolePrefix}_lm_studio_fallback`];
  const contextWindow = localConfig[`${rolePrefix}_context_window`] || 131072;
  const maxTokens = localConfig[`${rolePrefix}_max_tokens`] || 25000;

  return (
    <div className="submitter-config-section" style={{
      borderColor: provider === 'openrouter' ? '#FF6700' : borderColor
    }}>
      <h5 style={{ color: provider === 'openrouter' ? '#FF6700' : borderColor }}>
        {title}
        {provider === 'openrouter' && <span style={{ fontWeight: 'normal', marginLeft: '0.5rem' }}>[OpenRouter]</span>}
      </h5>
      {hint && <p className="settings-hint">{hint}</p>}

      <ModelSelector
        provider={provider}
        modelId={modelId}
        openrouterProv={openrouterProv}
        fallback={fallback}
        onProviderChange={(p) => handleProviderChange(rolePrefix, p)}
        onModelChange={(m) => handleModelChange(rolePrefix, m)}
        onOpenrouterProviderChange={(p) => handleChange(`${rolePrefix}_openrouter_provider`, p)}
        onFallbackChange={(f) => handleChange(`${rolePrefix}_lm_studio_fallback`, f)}
        lmStudioModels={lmStudioModels}
        openRouterModels={openRouterModels}
        modelProviders={modelProviders}
        hasOpenRouterKey={hasOpenRouterKey}
        isRunning={isRunning}
      />

      <div className="settings-row">
        <label>Context Window</label>
        <input
          type="number"
          value={contextWindow}
          onChange={(e) => handleChange(`${rolePrefix}_context_window`, e.target.value)}
          onBlur={(e) => handleNumericBlur(`${rolePrefix}_context_window`, e.target.value)}
          disabled={isRunning}
          min={4096}
          max={999999}
          step={1024}
        />
      </div>

      <div className="settings-row">
        <label>Max Output Tokens</label>
        <input
          type="number"
          value={maxTokens}
          onChange={(e) => handleChange(`${rolePrefix}_max_tokens`, e.target.value)}
          onBlur={(e) => handleNumericBlur(`${rolePrefix}_max_tokens`, e.target.value)}
          disabled={isRunning}
          min={1000}
          max={100000}
          step={1000}
        />
      </div>
    </div>
  );
};

const AutonomousResearchSettings = ({ config, onConfigChange, models, isRunning }) => {
  // Models and OpenRouter state
  const [lmStudioModels, setLmStudioModels] = useState(models || []);
  const [openRouterModels, setOpenRouterModels] = useState([]);
  const [modelProviders, setModelProviders] = useState({});
  const [hasOpenRouterKey, setHasOpenRouterKey] = useState(false);
  const [loadingOpenRouter, setLoadingOpenRouter] = useState(false);
  const [freeOnly, setFreeOnly] = useState(false);
  const [isLoadedFromStorage, setIsLoadedFromStorage] = useState(false);
  const [showKothTooltip, setShowKothTooltip] = useState(false);
  const [showTestedModelsTooltip, setShowTestedModelsTooltip] = useState(false);

  // Profile management state
  const [userProfiles, setUserProfiles] = useState({});
  const [selectedProfile, setSelectedProfile] = useState('');
  const [showSaveDialog, setShowSaveDialog] = useState(false);
  const [newProfileName, setNewProfileName] = useState('');

  // Wolfram Alpha settings (shared with compiler)
  const [wolframEnabled, setWolframEnabled] = useState(false);
  const [wolframApiKey, setWolframApiKey] = useState('');
  const [wolframTestResult, setWolframTestResult] = useState('');
  const [testingWolfram, setTestingWolfram] = useState(false);
  
  // Critique prompt editor state
  const [critiquePromptExpanded, setCritiquePromptExpanded] = useState(false);
  const [customCritiquePrompt, setCustomCritiquePrompt] = useState('');
  const [critiquePromptSaved, setCritiquePromptSaved] = useState(false);
  const [defaultCritiquePrompt, setDefaultCritiquePrompt] = useState('');

  // Parse submitter configs from config
  const parseSubmitterConfigs = (cfg) => {
    if (cfg?.submitter_configs && Array.isArray(cfg.submitter_configs)) {
      return cfg.submitter_configs.map(c => ({
        ...DEFAULT_SUBMITTER_CONFIG,
        ...c
      }));
    }
    return [
      { ...DEFAULT_SUBMITTER_CONFIG, submitterId: 1 },
      { ...DEFAULT_SUBMITTER_CONFIG, submitterId: 2 },
      { ...DEFAULT_SUBMITTER_CONFIG, submitterId: 3 }
    ];
  };

  const [numSubmitters, setNumSubmitters] = useState(
    config?.submitter_configs?.length || 3
  );
  const [submitterConfigs, setSubmitterConfigs] = useState(
    parseSubmitterConfigs(config)
  );
  
  const [localConfig, setLocalConfig] = useState({
    // Validator
    validator_provider: 'lm_studio',
    validator_model: '',
    validator_openrouter_provider: null,
    validator_lm_studio_fallback: null,
    validator_context_window: 131072,
    validator_max_tokens: 25000,
    // High-Context
    high_context_provider: 'lm_studio',
    high_context_model: '',
    high_context_openrouter_provider: null,
    high_context_lm_studio_fallback: null,
    high_context_context_window: 131072,
    high_context_max_tokens: 25000,
    // High-Param
    high_param_provider: 'lm_studio',
    high_param_model: '',
    high_param_openrouter_provider: null,
    high_param_lm_studio_fallback: null,
    high_param_context_window: 131072,
    high_param_max_tokens: 25000,
    // Critique Submitter
    critique_submitter_provider: 'lm_studio',
    critique_submitter_model: '',
    critique_submitter_openrouter_provider: null,
    critique_submitter_lm_studio_fallback: null,
    critique_submitter_context_window: 131072,
    critique_submitter_max_tokens: 25000,
    ...config
  });

  // Normalize profile models (fix common issues like blank submitters)
  const normalizeProfile = (profile) => {
    if (!profile) return profile;
    
    const normalized = { ...profile };
    
    // Normalize submitters: fix blank submitter 3
    if (normalized.submitters && Array.isArray(normalized.submitters)) {
      normalized.submitters = normalized.submitters.map((submitter, idx) => {
        let normalized_submitter = { ...submitter };
        
        // Fix blank submitter 3 - copy from submitter 1
        if (idx === 2 && (!normalized_submitter.modelId || normalized_submitter.modelId.trim() === '')) {
          if (normalized.submitters[0] && normalized.submitters[0].modelId) {
            normalized_submitter.modelId = normalized.submitters[0].modelId;
            normalized_submitter.provider = normalized.submitters[0].provider || 'openrouter';
            normalized_submitter.openrouterProvider = normalized.submitters[0].openrouterProvider || null;
            normalized_submitter.lmStudioFallbackId = normalized.submitters[0].lmStudioFallbackId || null;
            console.log(`[Profile Normalization] Fixed blank submitter 3: using "${normalized_submitter.modelId}"`);
          }
        }
        
        // Remove any legacy modelPattern field
        delete normalized_submitter.modelPattern;
        
        return normalized_submitter;
      });
    }
    
    // Normalize other roles (validator, highContext, highParam, critique) - remove legacy modelPattern
    const normalizeRole = (role) => {
      if (!role) return role;
      const norm = { ...role };
      delete norm.modelPattern; // Remove legacy field
      return norm;
    };
    
    normalized.validator = normalizeRole(normalized.validator);
    normalized.highContext = normalizeRole(normalized.highContext);
    normalized.highParam = normalizeRole(normalized.highParam);
    normalized.critique = normalizeRole(normalized.critique);
    
    return normalized;
  };

  // Check OpenRouter key status and fetch models on mount
  useEffect(() => {
    const init = async () => {
      // Load user profiles from localStorage
      const savedProfiles = localStorage.getItem('autonomous_research_profiles');
      if (savedProfiles) {
        try {
          let profiles = JSON.parse(savedProfiles);
          // Normalize all profiles on load
          const normalized = {};
          for (const [key, profile] of Object.entries(profiles)) {
            normalized[key] = normalizeProfile(profile);
          }
          setUserProfiles(normalized);
          // Save normalized profiles back to localStorage if any changes were made
          if (JSON.stringify(normalized) !== JSON.stringify(profiles)) {
            localStorage.setItem('autonomous_research_profiles', JSON.stringify(normalized));
            console.log('[Profile Normalization] Profiles updated and saved to localStorage');
          }
        } catch (err) {
          console.error('Failed to load user profiles:', err);
        }
      }

      // Load settings from localStorage
      const savedSettings = localStorage.getItem('autonomous_research_settings');
      if (savedSettings) {
        try {
          const settings = JSON.parse(savedSettings);
          if (settings.numSubmitters) setNumSubmitters(settings.numSubmitters);
          if (settings.submitterConfigs) setSubmitterConfigs(settings.submitterConfigs);
          if (settings.localConfig) {
            setLocalConfig(prev => ({ ...prev, ...settings.localConfig }));
          }
          if (settings.freeOnly !== undefined) setFreeOnly(settings.freeOnly);
          // Restore cached model providers
          if (settings.modelProviders) setModelProviders(settings.modelProviders);
        } catch (err) {
          console.error('Failed to load autonomous research settings:', err);
        }
      }
      
      try {
        const status = await openRouterAPI.getApiKeyStatus();
        setHasOpenRouterKey(status.has_key);
        if (status.has_key) {
          fetchOpenRouterModels();
        }
      } catch (err) {
        console.error('Failed to check OpenRouter key:', err);
      }
      
      // Load Wolfram Alpha status from backend
      try {
        const wolframStatus = await api.getWolframStatus();
        if (wolframStatus.enabled) {
          setWolframEnabled(true);
        }
      } catch (err) {
        console.error('Failed to load Wolfram Alpha status:', err);
      }

      // Try to fetch fresh LM Studio models
      try {
        const freshModels = await api.getModels();
        setLmStudioModels(freshModels);
      } catch (err) {
        console.error('Failed to fetch LM Studio models:', err);
      }
      
      setIsLoadedFromStorage(true);
    };
    init();
  }, []);

  // Fetch providers for any OpenRouter models after settings are loaded
  useEffect(() => {
    if (!isLoadedFromStorage || !hasOpenRouterKey) return;
    
    // Fetch providers for submitter configs
    submitterConfigs.forEach(cfg => {
      if (cfg.provider === 'openrouter' && cfg.modelId) {
        fetchProvidersForModel(cfg.modelId);
      }
    });
    
    // Fetch providers for validator
    if (localConfig.validator_provider === 'openrouter' && localConfig.validator_model) {
      fetchProvidersForModel(localConfig.validator_model);
    }
    
    // Fetch providers for high-context
    if (localConfig.high_context_provider === 'openrouter' && localConfig.high_context_model) {
      fetchProvidersForModel(localConfig.high_context_model);
    }
    
    // Fetch providers for high-param
    if (localConfig.high_param_provider === 'openrouter' && localConfig.high_param_model) {
      fetchProvidersForModel(localConfig.high_param_model);
    }
    
    // Fetch providers for critique submitter
    if (localConfig.critique_submitter_provider === 'openrouter' && localConfig.critique_submitter_model) {
      fetchProvidersForModel(localConfig.critique_submitter_model);
    }
  }, [isLoadedFromStorage, hasOpenRouterKey, submitterConfigs, localConfig]);

  // Save settings to localStorage whenever values change
  useEffect(() => {
    if (!isLoadedFromStorage) return;
    
    const settings = {
      numSubmitters,
      submitterConfigs: submitterConfigs.slice(0, numSubmitters),
      localConfig,
      freeOnly,
      modelProviders // Cache provider lists to avoid re-fetching
    };
    localStorage.setItem('autonomous_research_settings', JSON.stringify(settings));
  }, [isLoadedFromStorage, numSubmitters, submitterConfigs, localConfig, freeOnly, modelProviders]);

  // Update LM Studio models when prop changes
  useEffect(() => {
    if (models && models.length > 0) {
      setLmStudioModels(models);
    }
  }, [models]);

  // Initialize from config only once on mount
  const [initialized, setInitialized] = useState(false);
  useEffect(() => {
    if (!initialized && config) {
      setLocalConfig(prev => ({ ...prev, ...config }));
      if (config.submitter_configs) {
        setSubmitterConfigs(parseSubmitterConfigs(config));
        setNumSubmitters(config.submitter_configs.length);
      }
      setInitialized(true);
    }
  }, [config, initialized]);

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
    if (hasOpenRouterKey && isLoadedFromStorage) {
      fetchOpenRouterModels(freeOnly);
    }
  }, [freeOnly]);

  // Load critique prompt settings
  useEffect(() => {
    // Load custom prompt from localStorage
    const savedPrompt = localStorage.getItem('autonomous_critique_custom_prompt');
    if (savedPrompt) {
      setCustomCritiquePrompt(savedPrompt);
    }
    
    // Fetch default prompt from backend
    const fetchDefaultPrompt = async () => {
      try {
        const { autonomousAPI } = await import('../../services/api');
        const response = await autonomousAPI.getDefaultCritiquePrompt();
        if (response.prompt) {
          setDefaultCritiquePrompt(response.prompt);
          // If no custom prompt saved, use default
          if (!savedPrompt) {
            setCustomCritiquePrompt(response.prompt);
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

  const handleChange = (field, value) => {
    const numericFields = [
      'validator_context_window', 'high_context_context_window', 
      'high_param_context_window', 'critique_submitter_context_window',
      'validator_max_tokens', 'high_context_max_tokens', 
      'high_param_max_tokens', 'critique_submitter_max_tokens'
    ];

    let newValue = value;
    if (numericFields.includes(field)) {
      // For numeric fields, just store the raw value without parsing
      // This allows typing multi-digit numbers without losing focus
      newValue = value;
    }
    
    const newConfig = { ...localConfig, [field]: newValue };
    setLocalConfig(newConfig);
    
    // CRITICAL FIX: Don't propagate numeric field changes to parent on every keystroke
    // This prevents re-renders that cause input focus loss
    if (!numericFields.includes(field)) {
      onConfigChange({ ...newConfig, submitter_configs: submitterConfigs.slice(0, numSubmitters) });
    }
  };

  // Handler for when user finishes editing a numeric field (blur event)
  const handleNumericBlur = (field, value) => {
    const numericFields = [
      'validator_context_window', 'high_context_context_window', 
      'high_param_context_window', 'critique_submitter_context_window',
      'validator_max_tokens', 'high_context_max_tokens', 
      'high_param_max_tokens', 'critique_submitter_max_tokens'
    ];
    
    if (numericFields.includes(field)) {
      const parsed = parseInt(value, 10);
      const isContextField = field.includes('context_window');
      const finalValue = isNaN(parsed) ? (isContextField ? 131072 : 25000) : parsed;
      
      const newConfig = { ...localConfig, [field]: finalValue };
      setLocalConfig(newConfig);
      onConfigChange({ ...newConfig, submitter_configs: submitterConfigs.slice(0, numSubmitters) });
    }
  };

  // Handle provider change for a role (keeps existing model settings)
  const handleProviderChange = (rolePrefix, provider) => {
    const updates = {
      [`${rolePrefix}_provider`]: provider
      // Keep existing model, openrouter_provider, and lm_studio_fallback - don't reset them
    };
    const newConfig = { ...localConfig, ...updates };
    setLocalConfig(newConfig);
    onConfigChange({ ...newConfig, submitter_configs: submitterConfigs.slice(0, numSubmitters) });
  };

  // Handle model change with provider fetching for OpenRouter
  const handleModelChange = (rolePrefix, modelId) => {
    handleChange(`${rolePrefix}_model`, modelId);
    if (localConfig[`${rolePrefix}_provider`] === 'openrouter' && modelId) {
      fetchProvidersForModel(modelId);
    }
  };

  // Handle number of submitters change
  const handleNumSubmittersChange = (newCount) => {
    const count = Math.max(1, Math.min(10, parseInt(newCount, 10) || 1));
    setNumSubmitters(count);
    
    // Expand or contract submitter configs
    const newConfigs = [...submitterConfigs];
    while (newConfigs.length < count) {
      newConfigs.push({
        ...DEFAULT_SUBMITTER_CONFIG,
        submitterId: newConfigs.length + 1,
        provider: submitterConfigs[0]?.provider || 'lm_studio',
        modelId: submitterConfigs[0]?.modelId || ''
      });
    }
    const slicedConfigs = newConfigs.slice(0, count);
    setSubmitterConfigs(newConfigs);
    // Don't propagate immediately - will propagate on blur
  };

  // Handler for when user finishes editing number of submitters
  const handleNumSubmittersBlur = (value) => {
    const count = Math.max(1, Math.min(10, parseInt(value, 10) || 1));
    setNumSubmitters(count);
    
    const newConfigs = [...submitterConfigs];
    while (newConfigs.length < count) {
      newConfigs.push({
        ...DEFAULT_SUBMITTER_CONFIG,
        submitterId: newConfigs.length + 1,
        provider: submitterConfigs[0]?.provider || 'lm_studio',
        modelId: submitterConfigs[0]?.modelId || ''
      });
    }
    const slicedConfigs = newConfigs.slice(0, count);
    setSubmitterConfigs(newConfigs);
    onConfigChange({ ...localConfig, submitter_configs: slicedConfigs });
  };

  // Handle per-submitter config change
  const handleSubmitterConfigChange = (index, field, value) => {
    const newConfigs = [...submitterConfigs];
    const numericFields = ['contextWindow', 'maxOutputTokens'];
    let newValue = value;
    
    if (numericFields.includes(field)) {
      // Store raw value without parsing to allow multi-digit input
      newValue = value;
    }
    
    // Handle provider change - reset model fields
    if (field === 'provider') {
      newConfigs[index] = {
        ...newConfigs[index],
        provider: newValue,
        modelId: '',
        openrouterProvider: null,
        lmStudioFallbackId: null
      };
    } else {
      newConfigs[index] = {
        ...newConfigs[index],
        [field]: newValue
      };
    }
    
    // Fetch providers if selecting OpenRouter model
    if (field === 'modelId' && newConfigs[index].provider === 'openrouter' && newValue) {
      fetchProvidersForModel(newValue);
    }
    
    setSubmitterConfigs(newConfigs);
    
    // CRITICAL FIX: Don't propagate numeric field changes on every keystroke
    if (!numericFields.includes(field)) {
      onConfigChange({ ...localConfig, submitter_configs: newConfigs.slice(0, numSubmitters) });
    }
  };

  // Handler for when user finishes editing a submitter numeric field
  const handleSubmitterNumericBlur = (index, field, value) => {
    const numericFields = ['contextWindow', 'maxOutputTokens'];
    
    if (numericFields.includes(field)) {
      const parsed = parseInt(value, 10);
      const finalValue = isNaN(parsed) ? (field === 'contextWindow' ? 131072 : 25000) : parsed;
      
      const newConfigs = [...submitterConfigs];
      newConfigs[index] = {
        ...newConfigs[index],
        [field]: finalValue
      };
      
      setSubmitterConfigs(newConfigs);
      onConfigChange({ ...localConfig, submitter_configs: newConfigs.slice(0, numSubmitters) });
    }
  };

  // Copy submitter 1 settings to all others
  const copyMainToAll = () => {
    if (submitterConfigs.length > 0) {
      const main = submitterConfigs[0];
      const newConfigs = submitterConfigs.map((cfg, idx) => ({
        ...cfg,
        provider: main.provider,
        modelId: main.modelId,
        openrouterProvider: main.openrouterProvider,
        lmStudioFallbackId: main.lmStudioFallbackId,
        contextWindow: main.contextWindow,
        maxOutputTokens: main.maxOutputTokens
      }));
      setSubmitterConfigs(newConfigs);
      onConfigChange({ ...localConfig, submitter_configs: newConfigs.slice(0, numSubmitters) });
    }
  };

  // Critique prompt handlers
  const handleSaveCritiquePrompt = () => {
    localStorage.setItem('autonomous_critique_custom_prompt', customCritiquePrompt);
    setCritiquePromptSaved(true);
    setTimeout(() => setCritiquePromptSaved(false), 2000);
  };
  
  // Wolfram Alpha handlers (shared with compiler)
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

  const handleRestoreCritiquePrompt = () => {
    localStorage.removeItem('autonomous_critique_custom_prompt');
    setCustomCritiquePrompt(defaultCritiquePrompt);
    setCritiquePromptSaved(false);
  };

  const isUsingCustomCritiquePrompt = customCritiquePrompt && customCritiquePrompt !== defaultCritiquePrompt;

  // Apply a profile (recommended or user-saved)
  const applyProfile = async (profileKey) => {
    const isRecommended = profileKey.startsWith('recommended_');
    const profile = isRecommended
      ? RECOMMENDED_PROFILES[profileKey]
      : userProfiles[profileKey];
    
    if (!profile) {
      console.error(`Profile not found: ${profileKey}`);
      return;
    }

    console.log(`Applying profile: ${profile.name} (${isRecommended ? 'recommended' : 'user'})`);

    // Load model cache to convert display names to API IDs
    const modelCache = await loadModelCache();
    
    // Helper to convert display name to API ID
    const convertToApiId = (displayNameOrId) => {
      if (!displayNameOrId) return '';
      const apiId = getModelApiId(displayNameOrId);
      if (apiId !== displayNameOrId) {
        console.log(`  Converted "${displayNameOrId}" -> "${apiId}"`);
      }
      return apiId;
    };

    // Apply submitter configs using hard-coded modelId values (NO pattern matching)
    const newSubmitterConfigs = [];
    for (let i = 0; i < profile.numSubmitters; i++) {
      const submitterProfile = profile.submitters[i];
      
      let modelId, provider, openrouterProv, fallback;
      
      if (isRecommended) {
        // Recommended profiles: convert display name to API ID
        modelId = convertToApiId(submitterProfile.modelId || '');
        provider = submitterProfile.provider || 'openrouter';
        openrouterProv = submitterProfile.openrouterProvider || null;
        fallback = null; // No fallback for recommended
      } else {
        // User profile: use stored settings directly (already in API format)
        modelId = submitterProfile.modelId || '';
        provider = submitterProfile.provider || 'openrouter';
        openrouterProv = submitterProfile.openrouterProvider || null;
        fallback = submitterProfile.lmStudioFallbackId || null;
      }
      
      newSubmitterConfigs.push({
        submitterId: i + 1,
        provider,
        modelId,
        openrouterProvider: openrouterProv,
        lmStudioFallbackId: fallback,
        contextWindow: submitterProfile.contextWindow,
        maxOutputTokens: submitterProfile.maxOutputTokens
      });
    }

    // Helper to get model ID (convert display name to API ID for recommended profiles)
    const getModelId = (roleProfile) => {
      if (isRecommended) {
        return convertToApiId(roleProfile.modelId || '');
      }
      return roleProfile.modelId || '';
    };

    // Helper to get OpenRouter provider
    const getOpenrouterProvider = (roleProfile) => {
      if (isRecommended) {
        return roleProfile.openrouterProvider || null;
      }
      return roleProfile.openrouterProvider || null;
    };

    // Apply validator, high-context, high-param, and critique configs
    const validatorModelId = getModelId(profile.validator);
    const highContextModelId = getModelId(profile.highContext);
    const highParamModelId = getModelId(profile.highParam);
    const critiqueModelId = getModelId(profile.critique);

    // Update all state
    setNumSubmitters(profile.numSubmitters);
    setSubmitterConfigs(newSubmitterConfigs);
    
    const newConfig = {
      ...localConfig,
      validator_provider: isRecommended ? 'openrouter' : (profile.validator.provider || 'openrouter'),
      validator_model: validatorModelId,
      validator_openrouter_provider: getOpenrouterProvider(profile.validator),
      validator_lm_studio_fallback: isRecommended ? null : (profile.validator.lmStudioFallbackId || null),
      validator_context_window: profile.validator.contextWindow,
      validator_max_tokens: profile.validator.maxOutputTokens,
      high_context_provider: isRecommended ? 'openrouter' : (profile.highContext.provider || 'openrouter'),
      high_context_model: highContextModelId,
      high_context_openrouter_provider: getOpenrouterProvider(profile.highContext),
      high_context_lm_studio_fallback: isRecommended ? null : (profile.highContext.lmStudioFallbackId || null),
      high_context_context_window: profile.highContext.contextWindow,
      high_context_max_tokens: profile.highContext.maxOutputTokens,
      high_param_provider: isRecommended ? 'openrouter' : (profile.highParam.provider || 'openrouter'),
      high_param_model: highParamModelId,
      high_param_openrouter_provider: getOpenrouterProvider(profile.highParam),
      high_param_lm_studio_fallback: isRecommended ? null : (profile.highParam.lmStudioFallbackId || null),
      high_param_context_window: profile.highParam.contextWindow,
      high_param_max_tokens: profile.highParam.maxOutputTokens,
      critique_submitter_provider: isRecommended ? 'openrouter' : (profile.critique.provider || 'openrouter'),
      critique_submitter_model: critiqueModelId,
      critique_submitter_openrouter_provider: getOpenrouterProvider(profile.critique),
      critique_submitter_lm_studio_fallback: isRecommended ? null : (profile.critique.lmStudioFallbackId || null),
      critique_submitter_context_window: profile.critique.contextWindow,
      critique_submitter_max_tokens: profile.critique.maxOutputTokens
    };
    
    setLocalConfig(newConfig);
    onConfigChange({ ...newConfig, submitter_configs: newSubmitterConfigs });
    setSelectedProfile(profileKey);
  };

  // Save current settings as a new profile
  const saveCurrentAsProfile = () => {
    if (!newProfileName.trim()) {
      alert('Please enter a profile name');
      return;
    }

    const profileKey = `user_${Date.now()}`;
    const newProfile = {
      name: newProfileName.trim(),
      numSubmitters,
      submitters: submitterConfigs.slice(0, numSubmitters).map(cfg => ({
        modelId: cfg.modelId,
        provider: cfg.provider,
        openrouterProvider: cfg.openrouterProvider,
        lmStudioFallbackId: cfg.lmStudioFallbackId,
        contextWindow: cfg.contextWindow,
        maxOutputTokens: cfg.maxOutputTokens
      })),
      validator: {
        modelId: localConfig.validator_model,
        provider: localConfig.validator_provider,
        openrouterProvider: localConfig.validator_openrouter_provider,
        lmStudioFallbackId: localConfig.validator_lm_studio_fallback,
        contextWindow: localConfig.validator_context_window,
        maxOutputTokens: localConfig.validator_max_tokens
      },
      highContext: {
        modelId: localConfig.high_context_model,
        provider: localConfig.high_context_provider,
        openrouterProvider: localConfig.high_context_openrouter_provider,
        lmStudioFallbackId: localConfig.high_context_lm_studio_fallback,
        contextWindow: localConfig.high_context_context_window,
        maxOutputTokens: localConfig.high_context_max_tokens
      },
      highParam: {
        modelId: localConfig.high_param_model,
        provider: localConfig.high_param_provider,
        openrouterProvider: localConfig.high_param_openrouter_provider,
        lmStudioFallbackId: localConfig.high_param_lm_studio_fallback,
        contextWindow: localConfig.high_param_context_window,
        maxOutputTokens: localConfig.high_param_max_tokens
      },
      critique: {
        modelId: localConfig.critique_submitter_model,
        provider: localConfig.critique_submitter_provider,
        openrouterProvider: localConfig.critique_submitter_openrouter_provider,
        lmStudioFallbackId: localConfig.critique_submitter_lm_studio_fallback,
        contextWindow: localConfig.critique_submitter_context_window,
        maxOutputTokens: localConfig.critique_submitter_max_tokens
      }
    };

    const updatedProfiles = { ...userProfiles, [profileKey]: newProfile };
    setUserProfiles(updatedProfiles);
    localStorage.setItem('autonomous_research_profiles', JSON.stringify(updatedProfiles));
    setSelectedProfile(profileKey);
    setShowSaveDialog(false);
    setNewProfileName('');
  };

  // Delete a user profile
  const deleteProfile = (profileKey) => {
    if (!profileKey.startsWith('user_')) {
      alert('Cannot delete recommended profiles');
      return;
    }

    // Safety check for profile existence
    const profileToDelete = userProfiles[profileKey];
    if (!profileToDelete) {
      console.error(`Profile ${profileKey} not found`);
      return;
    }

    if (!confirm(`Delete profile "${profileToDelete.name}"?`)) {
      return;
    }

    const updatedProfiles = { ...userProfiles };
    delete updatedProfiles[profileKey];
    setUserProfiles(updatedProfiles);
    localStorage.setItem('autonomous_research_profiles', JSON.stringify(updatedProfiles));
    
    if (selectedProfile === profileKey) {
      setSelectedProfile('');
    }
  };

  return (
    <div className="autonomous-settings-layout">
      {/* Left Sidebar - Known Compatible Models */}
      <div className="settings-left-sidebar">
        <div className="known-models-sidebar">
          <h3 style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
            <span>📦 Tested Compatible Models</span>
            <div style={{ position: 'relative', display: 'inline-block' }}>
              <button
                style={{
                  backgroundColor: 'transparent',
                  border: '2px solid #FF6700',
                  color: '#FF6700',
                  padding: '0',
                  borderRadius: '4px',
                  cursor: 'pointer',
                  fontSize: '0.7rem',
                  fontWeight: 'bold',
                  width: '16px',
                  height: '16px',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  boxShadow: '0 0 8px rgba(255, 103, 0, 0.3)',
                  transition: 'all 0.2s ease'
                }}
                onMouseEnter={() => setShowTestedModelsTooltip(true)}
                onMouseLeave={() => setShowTestedModelsTooltip(false)}
              >
                ?
              </button>
              {showTestedModelsTooltip && (
                <div style={{
                  position: 'absolute',
                  backgroundColor: '#1a1a1a',
                  border: '2px solid #FF6700',
                  borderRadius: '6px',
                  padding: '12px 16px',
                  fontSize: '0.85rem',
                  color: '#FF6700',
                  fontWeight: '500',
                  maxWidth: '280px',
                  width: '260px',
                  zIndex: 1000,
                  boxShadow: '0 8px 24px rgba(255, 103, 0, 0.4)',
                  textShadow: '0 1px 2px rgba(0, 0, 0, 0.5)',
                  pointerEvents: 'none',
                  top: 'calc(100% + 8px)',
                  left: '50%',
                  transform: 'translateX(-50%)'
                }}>
                  These models and/or hosts are not affiliated with the MOTO program, or Intrafere LLC. This chart contains potential models and related roles to help guide users through developer-tested configurations. Any statements about pricing, cost, models, roles, rankings, effects, or otherwise are speculative and based on individual developer testing experience. Intrafere LLC (Intrafere Research Group), and the MOTO developement team make no guarantees or warranties about the accuracy or truth of this chart. MOTO is a harness that works with the majority of models, including many more models that are not listed here.
                </div>
              )}
            </div>
          </h3>
          <p style={{ fontSize: '.70rem', color: '#888', marginTop: '0.5rem', marginBottom: '1rem', lineHeight: '1.4', marginLeft: '20px' }}>
            Note: Computer science and/or non-general purpose models may have trouble performing as validators, critique submitters, or in the tier 2 compilation stage. These models generally perform fine for brainstorming. Note that some models in this category may work without issue.
          </p>
          <div className="models-list">
            {/* King of the Hill - Gold */}
            <div className="model-item" style={{ 
              backgroundColor: 'linear-gradient(135deg, rgba(212, 175, 55, 0.35) 0%, rgba(212, 175, 55, 0.15) 100%)',
              borderLeft: '5px solid #d4af37',
              borderRadius: '6px',
              boxShadow: '0 0 15px rgba(212, 175, 55, 0.3), inset 0 1px 0 rgba(255, 255, 255, 0.15)',
              paddingLeft: '12px'
            }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                <div className="model-item-name">GPT OSS 120B</div>
                <div style={{
                  background: 'linear-gradient(135deg, #e8c547 0%, #d4af37 50%, #c9a227 100%)',
                  color: '#000',
                  padding: '2px 8px',
                  borderRadius: '12px',
                  fontSize: '0.7rem',
                  fontWeight: 'bold',
                  boxShadow: '0 2px 8px rgba(212, 175, 55, 0.4), inset 0 1px 0 rgba(255, 255, 255, 0.3)',
                  textShadow: '0 1px 1px rgba(0, 0, 0, 0.1)'
                }}>👑 KING OF THE HILL</div>
                <div style={{ position: 'relative', display: 'inline-block', zIndex: 100 }}>
                  <button
                    style={{
                      backgroundColor: 'transparent',
                      border: '2px solid #d4af37',
                      color: '#d4af37',
                      padding: '2px 6px',
                      borderRadius: '4px',
                      cursor: 'pointer',
                      fontSize: '0.8rem',
                      fontWeight: 'bold',
                      width: '20px',
                      height: '20px',
                      display: 'flex',
                      alignItems: 'center',
                      justifyContent: 'center',
                      boxShadow: '0 0 8px rgba(212, 175, 55, 0.3)',
                      transition: 'all 0.2s ease'
                    }}
                    onMouseEnter={() => setShowKothTooltip(true)}
                    onMouseLeave={() => setShowKothTooltip(false)}
                  >
                    ?
                  </button>
                  {showKothTooltip && (
                    <div style={{
                      position: 'fixed',
                      backgroundColor: '#1a1a1a',
                      border: '2px solid #FF6700',
                      borderRadius: '6px',
                      padding: '12px 16px',
                      fontSize: '0.85rem',
                      color: '#FF6700',
                      fontWeight: '500',
                      maxWidth: '300px',
                      width: '280px',
                      zIndex: 2147483647,
                      boxShadow: '0 8px 24px rgba(255, 103, 0, 0.4)',
                      textShadow: '0 1px 2px rgba(0, 0, 0, 0.5)',
                      pointerEvents: 'none',
                      top: '50px',
                      right: '20px'
                    }}>
                      This model was chosen by the Intrafere developers as the CURRENT best overall performer in the MOTO harness, optimized for cost, speed, and knowledge.
                    </div>
                  )}
                </div>
              </div>
              <div className="model-item-badge">Balanced knowledge and speed</div>
              <div className="model-item-note">(outputs may corrupt over time depending on host)</div>
            </div>

            {/* Runner Up - Silver */}
            <div className="model-item" style={{ 
              backgroundColor: 'linear-gradient(135deg, rgba(192, 192, 192, 0.35) 0%, rgba(192, 192, 192, 0.15) 100%)',
              borderLeft: '5px solid #c0c0c0',
              borderRadius: '6px',
              boxShadow: '0 0 15px rgba(192, 192, 192, 0.3), inset 0 1px 0 rgba(255, 255, 255, 0.15)',
              paddingLeft: '12px'
            }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                <div className="model-item-name">Grok 4.1 Fast</div>
                <div style={{
                  background: 'linear-gradient(135deg, #e8e8e8 0%, #c0c0c0 50%, #a9a9a9 100%)',
                  color: '#000',
                  padding: '2px 8px',
                  borderRadius: '12px',
                  fontSize: '0.7rem',
                  fontWeight: 'bold',
                  boxShadow: '0 2px 8px rgba(192, 192, 192, 0.4), inset 0 1px 0 rgba(255, 255, 255, 0.3)',
                  textShadow: '0 1px 1px rgba(0, 0, 0, 0.1)'
                }}>🥈 SILVER</div>
              </div>
              <div className="model-item-badge">Fast validator</div>
            </div>

            {/* Bronze - DeepSeek V3.2 */}
            <div className="model-item" style={{ 
              backgroundColor: 'linear-gradient(135deg, rgba(205, 127, 50, 0.35) 0%, rgba(205, 127, 50, 0.15) 100%)',
              borderLeft: '5px solid #cd7f32',
              borderRadius: '6px',
              boxShadow: '0 0 15px rgba(205, 127, 50, 0.3), inset 0 1px 0 rgba(255, 255, 255, 0.15)',
              paddingLeft: '12px'
            }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                <div className="model-item-name">DeepSeek V3.2</div>
                <div style={{
                  background: 'linear-gradient(135deg, #d9a574 0%, #cd7f32 50%, #b86f28 100%)',
                  color: '#fff',
                  padding: '2px 8px',
                  borderRadius: '12px',
                  fontSize: '0.7rem',
                  fontWeight: 'bold',
                  boxShadow: '0 2px 8px rgba(205, 127, 50, 0.4), inset 0 1px 0 rgba(255, 255, 255, 0.2)',
                  textShadow: '0 1px 2px rgba(0, 0, 0, 0.3)'
                }}>🥉 BRONZE</div>
              </div>
              <div className="model-item-badge">Highly knowledgeable</div>
            </div>

            {/* Alphabetical list (rest of models) */}
            
            <div className="model-item">
              <div className="model-item-name">Claude Opus 4.5</div>
              <div className="model-item-badge">Highly knowledgeable</div>
            </div>
            
            <div className="model-item">
              <div className="model-item-name">DeepSeek V3.2 Speciale</div>
              <div className="model-item-badge">Highly knowledgeable</div>
            </div>
            
            <div className="model-item">
              <div className="model-item-name">Gemini 3.0 Pro</div>
              <div className="model-item-badge">Highly knowledgeable</div>
            </div>
            
            <div className="model-item">
              <div className="model-item-name">Gemini Flash 2.5</div>
              <div className="model-item-badge">Fast validator</div>
            </div>
            
            <div className="model-item">
              <div className="model-item-name">Gemini Flash 2.5 Light</div>
              <div className="model-item-badge">Fast validator</div>
            </div>
            
            <div className="model-item">
              <div className="model-item-name">Gemini Flash 3.0 Preview</div>
              <div className="model-item-badge">Fast validator</div>
            </div>
            
            <div className="model-item">
              <div className="model-item-name">GPT OSS 20B</div>
              <div className="model-item-badge">Balanced knowledge and speed</div>
              <div className="model-item-note">(outputs may corrupt over time depending on host)</div>
            </div>
            
            <div className="model-item">
              <div className="model-item-name">Kimi K2</div>
              <div className="model-item-badge">Highly knowledgeable</div>
            </div>
            
            <div className="model-item">
              <div className="model-item-name">GPT 5.2 Pro</div>
              <div className="model-item-badge">Highly knowledgeable</div>
            </div>

            <div className="model-item">
              <div className="model-item-name">GPT 5.2 Codex</div>
              <div className="model-item-badge">Computer science</div>
            </div>
            
            <div className="model-item">
              <div className="model-item-name">Perplexity: Sonar</div>
              <div className="model-item-badge">Internet search capability</div>
            </div>
            
            <div className="model-item">
              <div className="model-item-name">Qwen3 Coder 480B</div>
              <div className="model-item-badge">Computer science</div>
            </div>
          </div>
        </div>
      </div>

      {/* Main Content Area */}
      <div className="autonomous-settings">
      {/* Profile Selection Section */}
      <div className="settings-group" style={{ marginBottom: '1.5rem' }}>
        <h4>Profile Selection</h4>
        <p className="settings-info">
          Load a recommended profile or create your own custom profile. (These models and hosts are not affiliated with MOTO/Intrafere)
        </p>
        
        <div className="settings-row">
          <label>Select Profile</label>
          <select
            value={selectedProfile}
            onChange={(e) => {
              const value = e.target.value;
              if (value) {
                if (!hasOpenRouterKey) {
                  alert('OpenRouter API key required to use profiles. Please set your API key first.');
                  return;
                }
                if (openRouterModels.length === 0) {
                  alert('Please wait for OpenRouter models to load, or click "Refresh OpenRouter Models" button below.');
                  return;
                }
                applyProfile(value);
              }
            }}
            disabled={isRunning}
          >
            <option value="">-- Custom Settings --</option>
            <optgroup label="Recommended Profiles">
              {['recommended_fastest_cheapest']
                .filter(key => RECOMMENDED_PROFILES[key])
                .map(key => (
                  <option key={key} value={key}>
                    {RECOMMENDED_PROFILES[key].name}
                  </option>
                ))}
            </optgroup>
            {Object.keys(userProfiles).length > 0 && (
              <optgroup label="My Profiles">
                {Object.keys(userProfiles)
                  .sort((a, b) => userProfiles[a].name.localeCompare(userProfiles[b].name))
                  .map(key => (
                    <option key={key} value={key}>
                      {userProfiles[key].name}
                    </option>
                  ))}
              </optgroup>
            )}
          </select>
          
          <button
            className="secondary"
            onClick={() => setShowSaveDialog(true)}
            disabled={isRunning}
            style={{ marginLeft: '0.5rem' }}
            title="Save current settings as a profile"
          >
            Save as Profile
          </button>
          
          {selectedProfile && selectedProfile.startsWith('user_') && (
            <button
              className="secondary"
              onClick={() => deleteProfile(selectedProfile)}
              disabled={isRunning}
              style={{ marginLeft: '0.5rem', backgroundColor: '#e74c3c' }}
              title="Delete this profile"
            >
              Delete Profile
            </button>
          )}
        </div>
      </div>

      {/* Save Profile Dialog */}
      {showSaveDialog && (
        <div style={{
          position: 'fixed',
          top: 0,
          left: 0,
          right: 0,
          bottom: 0,
          backgroundColor: 'rgba(0, 0, 0, 0.7)',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          zIndex: 9999
        }}>
          <div style={{
            backgroundColor: '#1e1e1e',
            padding: '2rem',
            borderRadius: '8px',
            border: '1px solid #333',
            minWidth: '400px'
          }}>
            <h3 style={{ marginTop: 0 }}>Save Profile</h3>
            <p style={{ color: '#888' }}>
              Enter a name for this profile. Current settings will be saved.
            </p>
            <input
              type="text"
              value={newProfileName}
              onChange={(e) => setNewProfileName(e.target.value)}
              placeholder="Profile name..."
              style={{
                width: '100%',
                padding: '0.5rem',
                marginBottom: '1rem',
                backgroundColor: '#2a2a2a',
                border: '1px solid #444',
                borderRadius: '4px',
                color: '#fff'
              }}
              onKeyPress={(e) => {
                if (e.key === 'Enter') {
                  saveCurrentAsProfile();
                }
              }}
              autoFocus
            />
            <div style={{ display: 'flex', gap: '0.5rem', justifyContent: 'flex-end' }}>
              <button
                className="secondary"
                onClick={() => {
                  setShowSaveDialog(false);
                  setNewProfileName('');
                }}
              >
                Cancel
              </button>
              <button
                onClick={saveCurrentAsProfile}
                style={{
                  padding: '0.5rem 1rem',
                  backgroundColor: '#4CAF50',
                  border: 'none',
                  borderRadius: '4px',
                  color: '#fff',
                  cursor: 'pointer'
                }}
              >
                Save Profile
              </button>
            </div>
          </div>
        </div>
      )}

      {/* OpenRouter Status Banner */}
      {!hasOpenRouterKey && (
        <div style={{
          backgroundColor: 'rgba(255, 103, 0, 0.1)',
          border: '1px solid #FF6700',
          borderRadius: '8px',
          padding: '1rem',
          marginBottom: '1.5rem'
        }}>
          <p style={{ color: '#FF6700', margin: 0 }}>
            <strong>💡 OpenRouter Available:</strong> Set your OpenRouter API key in the header to enable cloud model selection for any role.
          </p>
        </div>
      )}

      {/* Brainstorm Submitters Section */}
      <div className="settings-group">
        <h4>Brainstorm Submitters (Tier 1 Aggregation)</h4>
        <p className="settings-info">
          Configure multiple parallel submitters for brainstorm exploration. Each submitter can use a different model or provider.
        </p>
        
        <div className="settings-row">
          <label title="Number of parallel brainstorm submitters (1-10)">
            Number of Submitters
          </label>
          <input
            type="number"
            value={numSubmitters}
            onChange={(e) => handleNumSubmittersChange(e.target.value)}
            onBlur={(e) => handleNumSubmittersBlur(e.target.value)}
            disabled={isRunning}
            min={1}
            max={10}
            step={1}
          />
          {numSubmitters > 1 && (
            <button 
              className="copy-btn"
              onClick={copyMainToAll}
              disabled={isRunning}
              title="Copy Main Submitter settings to all others"
            >
              Copy Main to All
            </button>
          )}
        </div>

        {/* Per-submitter configuration */}
        {submitterConfigs.slice(0, numSubmitters).map((cfg, idx) => (
          <div 
            key={idx} 
            className="submitter-config-section"
            style={{
              borderColor: cfg.provider === 'openrouter' ? '#FF6700' : (idx === 0 ? '#4CAF50' : '#333')
            }}
          >
            <h5 style={{ color: cfg.provider === 'openrouter' ? '#FF6700' : (idx === 0 ? '#4CAF50' : '#fff') }}>
              {idx === 0 ? 'Submitter 1 (Main Submitter)' : `Submitter ${idx + 1}`}
              {cfg.provider === 'openrouter' && <span style={{ fontWeight: 'normal', marginLeft: '0.5rem' }}>[OpenRouter]</span>}
            </h5>
            
            <ModelSelector
              provider={cfg.provider}
              modelId={cfg.modelId}
              openrouterProv={cfg.openrouterProvider}
              fallback={cfg.lmStudioFallbackId}
              onProviderChange={(p) => handleSubmitterConfigChange(idx, 'provider', p)}
              onModelChange={(m) => handleSubmitterConfigChange(idx, 'modelId', m)}
              onOpenrouterProviderChange={(p) => handleSubmitterConfigChange(idx, 'openrouterProvider', p)}
              onFallbackChange={(f) => handleSubmitterConfigChange(idx, 'lmStudioFallbackId', f)}
              lmStudioModels={lmStudioModels}
              openRouterModels={openRouterModels}
              modelProviders={modelProviders}
              hasOpenRouterKey={hasOpenRouterKey}
              isRunning={isRunning}
            />

            <div className="settings-row">
              <label title="Context window for this submitter">Context Window</label>
              <input
                type="number"
                value={cfg.contextWindow}
                onChange={(e) => handleSubmitterConfigChange(idx, 'contextWindow', e.target.value)}
                onBlur={(e) => handleSubmitterNumericBlur(idx, 'contextWindow', e.target.value)}
                disabled={isRunning}
                min={4096}
                max={999999}
                step={1024}
              />
            </div>

            <div className="settings-row">
              <label title="Max output tokens for this submitter">Max Output Tokens</label>
              <input
                type="number"
                value={cfg.maxOutputTokens}
                onChange={(e) => handleSubmitterConfigChange(idx, 'maxOutputTokens', e.target.value)}
                onBlur={(e) => handleSubmitterNumericBlur(idx, 'maxOutputTokens', e.target.value)}
                disabled={isRunning}
                min={1000}
                max={100000}
                step={1000}
              />
            </div>
          </div>
        ))}
      </div>

      {/* Validator (Single) */}
      <div className="settings-group">
        <h4>Validator (Single Instance)</h4>
        <p className="settings-info">
          Single validator maintains coherent Markov chain evolution for database alignment. This models speed will be your biggest bottleneck for the system, however their knowledge is also very important. Choose this model wisely, about half of all API calls will be to this model. A single validator as the markov chain bottleneck for the solution progression is important to mitigate the "alignment problem" with AI and user prompts. This is the model that will reject wrong answers, off-track answers, etc. at all stages of solution creation.
        </p>

        <RoleConfig
          title="Validator"
          rolePrefix="validator"
          borderColor="#ff6b6b"
          localConfig={localConfig}
          handleProviderChange={handleProviderChange}
          handleModelChange={handleModelChange}
          handleChange={handleChange}
          handleNumericBlur={handleNumericBlur}
          isRunning={isRunning}
          lmStudioModels={lmStudioModels}
          openRouterModels={openRouterModels}
          modelProviders={modelProviders}
          hasOpenRouterKey={hasOpenRouterKey}
        />
      </div>

      {/* Paper Compilation (Tier 2) - Compiler Settings */}
      <div className="settings-group">
        <h4>Paper Compilation (Tier 2 - Compiler)</h4>
        <p className="settings-info">
          Separate compiler submitters for paper construction and rigor enhancement.
        </p>

        <RoleConfig
          title="High-Context Submitter"
          hint="Handles outline, construction, and review modes."
          rolePrefix="high_context"
          borderColor="#4CAF50"
          localConfig={localConfig}
          handleProviderChange={handleProviderChange}
          handleModelChange={handleModelChange}
          handleChange={handleChange}
          handleNumericBlur={handleNumericBlur}
          isRunning={isRunning}
          lmStudioModels={lmStudioModels}
          openRouterModels={openRouterModels}
          modelProviders={modelProviders}
          hasOpenRouterKey={hasOpenRouterKey}
        />

        <RoleConfig
          title="High-Parameter Submitter"
          hint="Handles mathematical rigor enhancement."
          rolePrefix="high_param"
          borderColor="#f1c40f"
          localConfig={localConfig}
          handleProviderChange={handleProviderChange}
          handleModelChange={handleModelChange}
          handleChange={handleChange}
          handleNumericBlur={handleNumericBlur}
          isRunning={isRunning}
          lmStudioModels={lmStudioModels}
          openRouterModels={openRouterModels}
          modelProviders={modelProviders}
          hasOpenRouterKey={hasOpenRouterKey}
        />

        <RoleConfig
          title="Critique Submitter"
          hint="Handles post-body peer review feedback and rewrite decisions."
          rolePrefix="critique_submitter"
          borderColor="#e74c3c"
          localConfig={localConfig}
          handleProviderChange={handleProviderChange}
          handleModelChange={handleModelChange}
          handleChange={handleChange}
          handleNumericBlur={handleNumericBlur}
          isRunning={isRunning}
          lmStudioModels={lmStudioModels}
          openRouterModels={openRouterModels}
          modelProviders={modelProviders}
          hasOpenRouterKey={hasOpenRouterKey}
        />
      </div>

      {/* Wolfram Alpha Integration */}
      <div className="settings-group">
        <h3>Wolfram Alpha Integration (Optional)</h3>
        <small style={{ color: '#888', display: 'block', marginBottom: '1rem' }}>
          Enable Wolfram Alpha API for computational verification in rigor mode. Shared with manual compiler mode.
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
          <div style={{ marginLeft: '1.75rem', marginTop: '1rem' }}>
            <div className="form-group">
              <label>Wolfram Alpha API Key:</label>
              <input
                type="password"
                value={wolframApiKey}
                onChange={(e) => setWolframApiKey(e.target.value)}
                placeholder="Enter your Wolfram Alpha App ID"
                style={{
                  padding: '0.6rem',
                  backgroundColor: '#1e1e1e',
                  border: '1px solid #444',
                  borderRadius: '4px',
                  color: '#fff',
                  width: '100%',
                  marginBottom: '0.75rem'
                }}
              />
            </div>
            
            <div style={{ display: 'flex', gap: '0.75rem', marginTop: '1rem' }}>
              <button 
                onClick={handleTestWolframConnection}
                disabled={testingWolfram}
                style={{
                  padding: '0.6rem 1.25rem',
                  backgroundColor: '#4CAF50',
                  border: 'none',
                  borderRadius: '4px',
                  color: '#fff',
                  cursor: testingWolfram ? 'wait' : 'pointer',
                  opacity: testingWolfram ? 0.6 : 1,
                  fontWeight: '500'
                }}
              >
                {testingWolfram ? 'Testing...' : 'Test Connection'}
              </button>
              
              <button 
                onClick={handleClearWolframKey}
                style={{
                  padding: '0.6rem 1.25rem',
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
                marginTop: '1rem', 
                padding: '0.75rem', 
                borderRadius: '4px',
                backgroundColor: wolframTestResult.includes('✓') ? '#1a3a1a' : '#3a1a1a',
                color: wolframTestResult.includes('✓') ? '#4CAF50' : '#ff6b6b',
                fontSize: '0.9rem'
              }}>
                {wolframTestResult}
              </div>
            )}
            
            <small style={{ color: '#888', display: 'block', marginTop: '1rem', lineHeight: '1.5' }}>
              In rigor mode, the AI can request Wolfram Alpha verification of mathematical claims. 
              This enables computational checking of theorems, solving equations, and verifying properties.
              This setting is shared with the manual compiler mode.
            </small>
          </div>
        )}
      </div>

      {/* Validator Critique Prompt Editor */}
      <div className="settings-group">
        <div 
          className="collapsible-header"
          onClick={() => setCritiquePromptExpanded(!critiquePromptExpanded)}
          style={{
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'space-between',
            cursor: 'pointer',
            padding: '0.75rem',
            backgroundColor: '#1e1e1e',
            borderRadius: '6px',
            border: '1px solid #333',
            marginBottom: critiquePromptExpanded ? '1rem' : 0,
            width: '100%',
            maxWidth: '100%',
            boxSizing: 'border-box'
          }}
        >
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
            <h4 style={{ margin: 0 }}>[OPTIONAL] Edit Validator Critique Prompt (for the user feedback mode on individual papers - this is not for the critique submitter used for the internal research workflow)</h4>
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
          <div style={{
            padding: '1rem',
            backgroundColor: '#1a1a1a',
            borderRadius: '6px',
            border: '1px solid #333'
          }}>
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

      <div style={{ marginTop: '1rem' }}>
        <label style={{ display: 'inline-flex', alignItems: 'center', fontSize: '0.9rem' }}>
          <input
            type="checkbox"
            checked={(() => {
              const saved = localStorage.getItem('banner_shimmer_enabled');
              return saved !== null ? JSON.parse(saved) : true;
            })()}
            onChange={(e) => {
              localStorage.setItem('banner_shimmer_enabled', JSON.stringify(e.target.checked));
              window.location.reload(); // Reload to apply change
            }}
            style={{ marginRight: '0.5rem' }}
          />
          Enable banner shimmer (disable for video recording)
        </label>
      </div>

      {isRunning && (
        <div className="settings-notice">
          Settings cannot be changed while autonomous research is running.
        </div>
      )}

      {/* Refresh buttons */}
      <div style={{ marginTop: '1rem' }}>
        <button 
          className="secondary"
          onClick={async () => {
            try {
              const freshModels = await api.getModels();
              setLmStudioModels(freshModels);
            } catch (err) {
              console.error('Failed to refresh LM Studio models:', err);
            }
          }}
          disabled={isRunning}
          style={{ marginRight: '0.5rem' }}
        >
          Refresh LM Studio Models
        </button>
        {hasOpenRouterKey && (
        <>
          <button 
            className="secondary"
            onClick={() => fetchOpenRouterModels(freeOnly)}
            disabled={isRunning || loadingOpenRouter}
            style={{ marginRight: '0.5rem' }}
          >
            {loadingOpenRouter ? 'Loading...' : 'Refresh OpenRouter Models'}
          </button>
          <button
            className="secondary"
            onClick={() => window.open('https://openrouter.ai/models', '_blank', 'noopener,noreferrer')}
            style={{ marginRight: '0.5rem' }}
            title="Browse all available OpenRouter models"
          >
            🔗 OpenRouter Model List
          </button>
          <label style={{ display: 'inline-flex', alignItems: 'center', fontSize: '0.9rem' }}>
            <input
              type="checkbox"
              checked={freeOnly}
              onChange={(e) => setFreeOnly(e.target.checked)}
              disabled={isRunning}
              style={{ marginRight: '0.5rem' }}
            />
            Show free models only
          </label>
        </>
        )}
      </div>
    </div>
    </div>
  );
};

export default AutonomousResearchSettings;
