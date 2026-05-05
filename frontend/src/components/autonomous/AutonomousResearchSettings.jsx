/**
 * AutonomousResearchSettings - Settings panel for autonomous research mode.
 * Supports configurable multi-submitter brainstorm aggregation (1-10 submitters).
 * Compiler settings (high-context, high-param) remain separate.
 * Now supports per-role OpenRouter model selection with provider and fallback options.
 */
import React, { useState, useEffect } from 'react';
import { openRouterAPI, api, autonomousAPI } from '../../services/api';
import {
  computeOpenRouterAutoSettings,
  findOpenRouterModel,
  getProviderNames,
  hasEndpointMetadata,
} from '../../utils/openRouterSelection';
import {
  AUTONOMOUS_SETTINGS_STORAGE_KEY,
  AUTONOMOUS_PROFILES_STORAGE_KEY,
  RECOMMENDED_PROFILE_KEYS,
  RECOMMENDED_PROFILES,
  applyAutonomousProfileSelection,
  getStoredAutonomousSettings,
} from '../../utils/autonomousProfiles';
import HelpTooltip from '../HelpTooltip';
import './AutonomousResearch.css';
import '../settings-common.css';

const DEFAULT_SUBMITTER_CONFIG = {
  submitterId: 1,
  provider: 'lm_studio',
  modelId: '',
  openrouterProvider: null,
  lmStudioFallbackId: null,
  contextWindow: 131072,
  maxOutputTokens: 25000
};

const OsTag = () => (
  <span className="os-tag-tooltip-anchor">
    <span className="os-tag">OS</span>
    <span className="os-tag-tooltip">
      Open source — weights available on Hugging Face for local use with LM Studio.
    </span>
  </span>
);

// ModelSelector component - extracted outside to prevent recreation on every render
const ModelSelector = ({
  provider,
  modelId,
  openrouterProv,
  fallback,
  onProviderChange,
  onModelChange,
  onOpenrouterProviderChange,
  onFallbackChange,
  lmStudioModels,
  openRouterModels,
  modelProviders,
  hasOpenRouterKey,
  isRunning,
  lmStudioEnabled,
}) => {
  const effectiveProvider = lmStudioEnabled ? provider : 'openrouter';
  const currentModels = effectiveProvider === 'openrouter' ? openRouterModels : lmStudioModels;
  const providers = modelId && effectiveProvider === 'openrouter'
    ? getProviderNames(modelProviders[modelId])
    : [];

  return (
    <>
      {/* Provider Toggle */}
      <div className="settings-row">
        <label>Provider</label>
        {lmStudioEnabled ? (
          <div className="provider-toggle-group">
            <button
              type="button"
              className={`provider-toggle-btn${provider === 'lm_studio' ? ' active-lm' : ''}`}
              onClick={() => onProviderChange('lm_studio')}
              disabled={isRunning}
            >
              LM Studio
            </button>
            <button
              type="button"
              className={`provider-toggle-btn${provider === 'openrouter' ? ' active-or-orange' : ''}`}
              onClick={() => hasOpenRouterKey && onProviderChange('openrouter')}
              disabled={isRunning || !hasOpenRouterKey}
              style={!hasOpenRouterKey ? { color: '#666' } : undefined}
              title={!hasOpenRouterKey ? 'Set OpenRouter API key first' : 'Use OpenRouter'}
            >
              OpenRouter
            </button>
          </div>
        ) : (
          <small className="settings-hint">OpenRouter is required in this deployment.</small>
        )}
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
      {effectiveProvider === 'openrouter' && modelId && (
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
      {effectiveProvider === 'openrouter' && lmStudioEnabled && (
        <div className="settings-row">
          <label className="label--muted">LM Studio Fallback (optional)</label>
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
const RoleConfig = ({
  title,
  hint,
  rolePrefix,
  borderColor = '#333',
  localConfig,
  handleProviderChange,
  handleModelChange,
  handleOpenRouterProviderChange,
  handleChange,
  handleNumericBlur,
  isRunning,
  lmStudioModels,
  openRouterModels,
  modelProviders,
  hasOpenRouterKey,
  lmStudioEnabled,
}) => {
  const storedProvider = localConfig[`${rolePrefix}_provider`] || 'lm_studio';
  const provider = lmStudioEnabled ? storedProvider : 'openrouter';
  const modelId = localConfig[`${rolePrefix}_model`] || '';
  const openrouterProv = localConfig[`${rolePrefix}_openrouter_provider`];
  const fallback = localConfig[`${rolePrefix}_lm_studio_fallback`];
  const contextWindow = localConfig[`${rolePrefix}_context_window`] || 131072;
  const maxTokens = localConfig[`${rolePrefix}_max_tokens`] || 25000;

  return (
    <div className={`submitter-config-section${provider === 'openrouter' ? ' role-config-card--openrouter-orange' : ''}`} style={{
      borderColor: provider === 'openrouter' ? undefined : borderColor
    }}>
      <h5 className={provider === 'openrouter' ? 'card-title--orange' : ''} style={provider !== 'openrouter' ? { color: borderColor } : undefined}>
        {title}
        {provider === 'openrouter' && <span className="provider-badge-inline">[OpenRouter]</span>}
      </h5>
      {hint && <p className="settings-hint">{hint}</p>}

      <ModelSelector
        provider={provider}
        modelId={modelId}
        openrouterProv={openrouterProv}
        fallback={fallback}
        onProviderChange={(p) => handleProviderChange(rolePrefix, p)}
        onModelChange={(m) => handleModelChange(rolePrefix, m)}
        onOpenrouterProviderChange={(p) => handleOpenRouterProviderChange(rolePrefix, p)}
        onFallbackChange={(f) => handleChange(`${rolePrefix}_lm_studio_fallback`, f)}
        lmStudioModels={lmStudioModels}
        openRouterModels={openRouterModels}
        modelProviders={modelProviders}
        hasOpenRouterKey={hasOpenRouterKey}
        isRunning={isRunning}
        lmStudioEnabled={lmStudioEnabled}
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
          max={50000000}
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
          max={50000000}
          step={1000}
        />
      </div>
    </div>
  );
};

const AutonomousResearchSettings = ({ config, onConfigChange, models, capabilities, isRunning }) => {
  // Models and OpenRouter state
  const [lmStudioModels, setLmStudioModels] = useState(models || []);
  const [openRouterModels, setOpenRouterModels] = useState([]);
  const [modelProviders, setModelProviders] = useState({});
  const [hasOpenRouterKey, setHasOpenRouterKey] = useState(false);
  const [loadingOpenRouter, setLoadingOpenRouter] = useState(false);
  const [freeOnly, setFreeOnly] = useState(false);
  const [freeModelLooping, setFreeModelLooping] = useState(true);
  const [freeModelAutoSelector, setFreeModelAutoSelector] = useState(true);
  const [tier3Enabled, setTier3Enabled] = useState(false);
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
  const [hasStoredWolframKey, setHasStoredWolframKey] = useState(false);
  const [wolframTestResult, setWolframTestResult] = useState('');
  const [testingWolfram, setTestingWolfram] = useState(false);
  const [proofStatus, setProofStatus] = useState(null);
  const [proofSettingsEnabled, setProofSettingsEnabled] = useState(false);
  const [proofSettingsTimeout, setProofSettingsTimeout] = useState('120');
  const [proofSettingsLspEnabled, setProofSettingsLspEnabled] = useState(false);
  const [proofSettingsLspIdleTimeout, setProofSettingsLspIdleTimeout] = useState('600');
  const [proofSettingsSmtEnabled, setProofSettingsSmtEnabled] = useState(false);
  const [proofSettingsZ3Path, setProofSettingsZ3Path] = useState('');
  const [proofSettingsSmtTimeout, setProofSettingsSmtTimeout] = useState('30');
  const [savingProofSettings, setSavingProofSettings] = useState(false);
  const [proofSettingsMessage, setProofSettingsMessage] = useState('');
  
  // Critique prompt editor state
  const [advancedSettingsExpanded, setAdvancedSettingsExpanded] = useState(false);
  const [critiquePromptExpanded, setCritiquePromptExpanded] = useState(false);
  const [customCritiquePrompt, setCustomCritiquePrompt] = useState('');
  const [critiquePromptSaved, setCritiquePromptSaved] = useState(false);
  const [defaultCritiquePrompt, setDefaultCritiquePrompt] = useState('');
  const lmStudioEnabled = capabilities?.lmStudioEnabled !== false;
  const genericMode = Boolean(capabilities?.genericMode);
  const showLean4Settings = Boolean(lmStudioEnabled && proofStatus?.lean4_path && !genericMode);

  const handleCollapsibleKeyDown = (event, toggleFn) => {
    if (event.key === 'Enter' || event.key === ' ') {
      event.preventDefault();
      toggleFn();
    }
  };

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
      const savedProfiles = localStorage.getItem(AUTONOMOUS_PROFILES_STORAGE_KEY);
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
            localStorage.setItem(AUTONOMOUS_PROFILES_STORAGE_KEY, JSON.stringify(normalized));
            console.log('[Profile Normalization] Profiles updated and saved to localStorage');
          }
        } catch (err) {
          console.error('Failed to load user profiles:', err);
        }
      }

      const settings = getStoredAutonomousSettings();
      if (settings.numSubmitters) setNumSubmitters(settings.numSubmitters);
      if (settings.submitterConfigs) setSubmitterConfigs(settings.submitterConfigs);
      if (settings.localConfig) {
        setLocalConfig(prev => ({ ...prev, ...settings.localConfig }));
      }
      setSelectedProfile(settings.selectedProfile || '');
      if (settings.freeOnly !== undefined) setFreeOnly(settings.freeOnly);
      if (settings.freeModelLooping !== undefined) setFreeModelLooping(settings.freeModelLooping);
      if (settings.freeModelAutoSelector !== undefined) setFreeModelAutoSelector(settings.freeModelAutoSelector);
      if (settings.tier3Enabled !== undefined) setTier3Enabled(settings.tier3Enabled);
      if (settings.modelProviders) setModelProviders(settings.modelProviders);
      
      try {
        const status = await openRouterAPI.getApiKeyStatus();
        setHasOpenRouterKey(status.has_key);
        if (status.has_key) {
          fetchOpenRouterModels();
        }
      } catch (err) {
        console.error('Failed to check OpenRouter key:', err);
      }
      
      try {
        const wolframStatus = await api.getWolframStatus();
        setHasStoredWolframKey(Boolean(wolframStatus.has_key));
        if (wolframStatus.enabled) {
          setWolframEnabled(true);
        }
      } catch (err) {
        console.error('Failed to load Wolfram Alpha status:', err);
      }

      // Try to fetch fresh LM Studio models
      if (lmStudioEnabled) {
        try {
          const freshModels = await api.getModels();
          setLmStudioModels(freshModels.models || freshModels || []);
        } catch (err) {
          console.error('Failed to fetch LM Studio models:', err);
        }
      } else {
        setLmStudioModels([]);
      }
      
      setIsLoadedFromStorage(true);
    };
    init();
  }, [lmStudioEnabled]);

  useEffect(() => {
    if (genericMode) {
      setProofStatus(null);
      return;
    }

    const loadProofStatus = async () => {
      try {
        const status = await autonomousAPI.getProofStatus();
        setProofStatus(status);
        setProofSettingsEnabled(Boolean(status.lean4_enabled));
        setProofSettingsTimeout(String(status.lean4_proof_timeout ?? 120));
        setProofSettingsLspEnabled(Boolean(status.lean4_lsp_enabled));
        setProofSettingsLspIdleTimeout(String(status.lean4_lsp_idle_timeout ?? 600));
        setProofSettingsSmtEnabled(Boolean(status.smt_enabled));
        setProofSettingsZ3Path(status.z3_path || '');
        setProofSettingsSmtTimeout(String(status.smt_timeout ?? 30));
      } catch (err) {
        console.error('Failed to load Lean 4 proof status:', err);
      }
    };

    loadProofStatus();
  }, [genericMode]);

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
      freeModelLooping,
      freeModelAutoSelector,
      tier3Enabled,
      modelProviders,
      selectedProfile,
    };
    localStorage.setItem(AUTONOMOUS_SETTINGS_STORAGE_KEY, JSON.stringify(settings));
  }, [isLoadedFromStorage, numSubmitters, submitterConfigs, localConfig, freeOnly, freeModelLooping, freeModelAutoSelector, tier3Enabled, modelProviders, selectedProfile]);

  useEffect(() => {
    if (!isLoadedFromStorage || lmStudioEnabled) {
      return;
    }

    setLmStudioModels([]);

    const normalizedSubmitters = submitterConfigs.map((submitterConfig) => {
      const keepOpenRouterState = submitterConfig.provider === 'openrouter';
      return {
        ...submitterConfig,
        provider: 'openrouter',
        modelId: keepOpenRouterState ? (submitterConfig.modelId || '') : '',
        openrouterProvider: keepOpenRouterState ? (submitterConfig.openrouterProvider || null) : null,
        lmStudioFallbackId: null,
      };
    });

    const normalizedLocalConfig = { ...localConfig };
    ['validator', 'high_context', 'high_param', 'critique_submitter'].forEach((rolePrefix) => {
      const providerKey = `${rolePrefix}_provider`;
      const modelKey = `${rolePrefix}_model`;
      const openRouterProviderKey = `${rolePrefix}_openrouter_provider`;
      const fallbackKey = `${rolePrefix}_lm_studio_fallback`;
      const keepOpenRouterState = normalizedLocalConfig[providerKey] === 'openrouter';

      normalizedLocalConfig[providerKey] = 'openrouter';
      normalizedLocalConfig[modelKey] = keepOpenRouterState ? (normalizedLocalConfig[modelKey] || '') : '';
      normalizedLocalConfig[openRouterProviderKey] = keepOpenRouterState
        ? (normalizedLocalConfig[openRouterProviderKey] || null)
        : null;
      normalizedLocalConfig[fallbackKey] = null;
    });

    if (JSON.stringify(normalizedSubmitters) !== JSON.stringify(submitterConfigs)) {
      setSubmitterConfigs(normalizedSubmitters);
    }
    if (JSON.stringify(normalizedLocalConfig) !== JSON.stringify(localConfig)) {
      setLocalConfig(normalizedLocalConfig);
    }

    const currentConfig = {
      ...localConfig,
      submitter_configs: submitterConfigs.slice(0, numSubmitters),
      tier3_enabled: tier3Enabled,
    };
    const nextConfig = {
      ...normalizedLocalConfig,
      submitter_configs: normalizedSubmitters.slice(0, numSubmitters),
      tier3_enabled: tier3Enabled,
    };
    if (JSON.stringify(nextConfig) !== JSON.stringify(currentConfig)) {
      onConfigChange(nextConfig);
    }
  }, [
    isLoadedFromStorage,
    lmStudioEnabled,
    submitterConfigs,
    localConfig,
    numSubmitters,
    tier3Enabled,
    onConfigChange,
  ]);

  // Update LM Studio models when prop changes
  useEffect(() => {
    if (!lmStudioEnabled) {
      setLmStudioModels([]);
      return;
    }

    if (models && models.length > 0) {
      setLmStudioModels(models);
    }
  }, [models, lmStudioEnabled]);

  // Propagate tier3Enabled to parent config whenever it changes
  useEffect(() => {
    if (!isLoadedFromStorage) return;
    onConfigChange({ ...localConfig, submitter_configs: submitterConfigs.slice(0, numSubmitters), tier3_enabled: tier3Enabled });
  }, [tier3Enabled]); // eslint-disable-line react-hooks/exhaustive-deps

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
      console.debug('[AutonomousAutoFill] model not in loaded list, skipping auto-fill', { modelId });
      return null;
    }

    const providerData = await fetchProvidersForModel(modelId);
    const autoSettings = computeOpenRouterAutoSettings(model, providerData, selectedProvider);
    if (autoSettings) {
      console.debug('[AutonomousAutoFill] computed auto-settings', {
        modelId,
        selectedProvider,
        source: autoSettings.source,
        contextWindow: autoSettings.contextWindow,
        maxOutputTokens: autoSettings.maxOutputTokens,
        warnings: autoSettings.warnings,
      });
      if (autoSettings.warnings && autoSettings.warnings.length > 0) {
        console.warn('[AutonomousAutoFill] auto-settings fallback used:', autoSettings.warnings);
      }
    }
    return autoSettings;
  };

  const markProfileAsCustom = () => {
    if (selectedProfile) {
      setSelectedProfile('');
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
    markProfileAsCustom();
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
      markProfileAsCustom();
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
    markProfileAsCustom();
    setLocalConfig(newConfig);
    onConfigChange({ ...newConfig, submitter_configs: submitterConfigs.slice(0, numSubmitters) });
  };

  // Handle model change with provider fetching for OpenRouter
  const handleModelChange = async (rolePrefix, modelId) => {
    const newConfig = {
      ...localConfig,
      [`${rolePrefix}_model`]: modelId,
      [`${rolePrefix}_openrouter_provider`]: null,
    };
    markProfileAsCustom();
    setLocalConfig(newConfig);
    onConfigChange({ ...newConfig, submitter_configs: submitterConfigs.slice(0, numSubmitters) });

    if (localConfig[`${rolePrefix}_provider`] !== 'openrouter' || !modelId) {
      return;
    }

    const autoSettings = await getAutoSettingsForModel(modelId, null);
    if (!autoSettings) {
      return;
    }

    const autofilledConfig = {
      ...newConfig,
      ...(autoSettings.contextWindowKnown ? { [`${rolePrefix}_context_window`]: autoSettings.contextWindow } : {}),
      ...(autoSettings.outputCapKnown ? { [`${rolePrefix}_max_tokens`]: autoSettings.maxOutputTokens } : {}),
    };
    setLocalConfig(autofilledConfig);
    onConfigChange({ ...autofilledConfig, submitter_configs: submitterConfigs.slice(0, numSubmitters) });
  };

  const handleOpenRouterProviderChange = async (rolePrefix, providerName) => {
    const newConfig = {
      ...localConfig,
      [`${rolePrefix}_openrouter_provider`]: providerName,
    };
    markProfileAsCustom();
    setLocalConfig(newConfig);
    onConfigChange({ ...newConfig, submitter_configs: submitterConfigs.slice(0, numSubmitters) });

    const modelId = newConfig[`${rolePrefix}_model`];
    if (newConfig[`${rolePrefix}_provider`] !== 'openrouter' || !modelId) {
      return;
    }

    const autoSettings = await getAutoSettingsForModel(modelId, providerName);
    if (!autoSettings) {
      return;
    }

    const autofilledConfig = {
      ...newConfig,
      ...(autoSettings.contextWindowKnown ? { [`${rolePrefix}_context_window`]: autoSettings.contextWindow } : {}),
      ...(autoSettings.outputCapKnown ? { [`${rolePrefix}_max_tokens`]: autoSettings.maxOutputTokens } : {}),
    };
    setLocalConfig(autofilledConfig);
    onConfigChange({ ...autofilledConfig, submitter_configs: submitterConfigs.slice(0, numSubmitters) });
  };

  // Handle number of submitters change
  const handleNumSubmittersChange = (newCount) => {
    const count = Math.max(1, Math.min(10, parseInt(newCount, 10) || 1));
    markProfileAsCustom();
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
    markProfileAsCustom();
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

    markProfileAsCustom();
    setSubmitterConfigs(newConfigs);
    
    // CRITICAL FIX: Don't propagate numeric field changes on every keystroke
    if (!numericFields.includes(field)) {
      onConfigChange({ ...localConfig, submitter_configs: newConfigs.slice(0, numSubmitters) });
    }
  };

  const handleSubmitterModelChange = async (index, modelId) => {
    const newConfigs = [...submitterConfigs];
    newConfigs[index] = {
      ...newConfigs[index],
      modelId,
      openrouterProvider: null,
    };

    markProfileAsCustom();
    setSubmitterConfigs(newConfigs);
    onConfigChange({ ...localConfig, submitter_configs: newConfigs.slice(0, numSubmitters) });

    if (newConfigs[index].provider !== 'openrouter' || !modelId) {
      return;
    }

    const autoSettings = await getAutoSettingsForModel(modelId, null);
    if (!autoSettings) {
      return;
    }

    const autofilledConfigs = [...newConfigs];
    autofilledConfigs[index] = {
      ...autofilledConfigs[index],
      ...(autoSettings.contextWindowKnown ? { contextWindow: autoSettings.contextWindow } : {}),
      ...(autoSettings.outputCapKnown ? { maxOutputTokens: autoSettings.maxOutputTokens } : {}),
    };

    setSubmitterConfigs(autofilledConfigs);
    onConfigChange({ ...localConfig, submitter_configs: autofilledConfigs.slice(0, numSubmitters) });
  };

  const handleSubmitterOpenRouterProviderChange = async (index, providerName) => {
    const newConfigs = [...submitterConfigs];
    newConfigs[index] = {
      ...newConfigs[index],
      openrouterProvider: providerName,
    };

    markProfileAsCustom();
    setSubmitterConfigs(newConfigs);
    onConfigChange({ ...localConfig, submitter_configs: newConfigs.slice(0, numSubmitters) });

    const modelId = newConfigs[index].modelId;
    if (newConfigs[index].provider !== 'openrouter' || !modelId) {
      return;
    }

    const autoSettings = await getAutoSettingsForModel(modelId, providerName);
    if (!autoSettings) {
      return;
    }

    const autofilledConfigs = [...newConfigs];
    autofilledConfigs[index] = {
      ...autofilledConfigs[index],
      ...(autoSettings.contextWindowKnown ? { contextWindow: autoSettings.contextWindow } : {}),
      ...(autoSettings.outputCapKnown ? { maxOutputTokens: autoSettings.maxOutputTokens } : {}),
    };

    setSubmitterConfigs(autofilledConfigs);
    onConfigChange({ ...localConfig, submitter_configs: autofilledConfigs.slice(0, numSubmitters) });
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
      
      markProfileAsCustom();
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
      markProfileAsCustom();
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

  const handleSaveProofSettings = async () => {
    const parsedTimeout = parseInt(proofSettingsTimeout, 10);
    const timeout = Number.isFinite(parsedTimeout) ? parsedTimeout : 120;
    const parsedLspIdleTimeout = parseInt(proofSettingsLspIdleTimeout, 10);
    const lspIdleTimeout = Number.isFinite(parsedLspIdleTimeout) ? parsedLspIdleTimeout : 600;
    const parsedSmtTimeout = parseInt(proofSettingsSmtTimeout, 10);
    const smtTimeout = Number.isFinite(parsedSmtTimeout) ? parsedSmtTimeout : 30;

    try {
      setSavingProofSettings(true);
      setProofSettingsMessage('');
      const status = await autonomousAPI.updateProofSettings({
        enabled: proofSettingsEnabled,
        timeout,
        lean4_lsp_enabled: proofSettingsLspEnabled,
        lean4_lsp_idle_timeout: lspIdleTimeout,
        smt_enabled: proofSettingsSmtEnabled,
        z3_path: proofSettingsZ3Path,
        smt_timeout: smtTimeout,
      });
      setProofStatus(status);
      setProofSettingsEnabled(Boolean(status.lean4_enabled));
      setProofSettingsTimeout(String(status.lean4_proof_timeout ?? timeout));
      setProofSettingsLspEnabled(Boolean(status.lean4_lsp_enabled));
      setProofSettingsLspIdleTimeout(String(status.lean4_lsp_idle_timeout ?? lspIdleTimeout));
      setProofSettingsSmtEnabled(Boolean(status.smt_enabled));
      setProofSettingsZ3Path(status.z3_path || '');
      setProofSettingsSmtTimeout(String(status.smt_timeout ?? smtTimeout));
      setProofSettingsMessage('Lean 4 / SMT proof settings saved.');
    } catch (err) {
      setProofSettingsMessage(`Failed to save Lean 4 / SMT proof settings: ${err.message}`);
    } finally {
      setSavingProofSettings(false);
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
    try {
      const { profile, settings, config: nextConfig } = await applyAutonomousProfileSelection(profileKey, userProfiles);
      const isRecommended = profileKey.startsWith('recommended_');

      console.log(`Applying profile: ${profile.name} (${isRecommended ? 'recommended' : 'user'})`);

      setNumSubmitters(settings.numSubmitters);
      setSubmitterConfigs(settings.submitterConfigs);
      setLocalConfig(settings.localConfig);
      setFreeOnly(settings.freeOnly);
      setFreeModelLooping(settings.freeModelLooping);
      setFreeModelAutoSelector(settings.freeModelAutoSelector);
      setTier3Enabled(settings.tier3Enabled);
      setModelProviders(settings.modelProviders || {});
      setSelectedProfile(settings.selectedProfile);
      onConfigChange(nextConfig);
    } catch (err) {
      console.error(err.message || 'Failed to apply profile:', err);
    }
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
    localStorage.setItem(AUTONOMOUS_PROFILES_STORAGE_KEY, JSON.stringify(updatedProfiles));
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
    localStorage.setItem(AUTONOMOUS_PROFILES_STORAGE_KEY, JSON.stringify(updatedProfiles));
    
    if (selectedProfile === profileKey) {
      setSelectedProfile('');
    }
  };

  return (
    <div className="autonomous-settings-layout">
      {/* Left Sidebar - Known Compatible Models */}
      <div className="settings-left-sidebar">
        <div className="known-models-sidebar">
          <h3 className="flex-row-center">
            <span>Highlighted Models</span>
            <div className="help-tooltip-anchor">
              <button
                type="button"
                className="help-tooltip-btn"
                aria-label="Learn about highlighted models"
                onMouseEnter={() => setShowTestedModelsTooltip(true)}
                onMouseLeave={() => setShowTestedModelsTooltip(false)}
                onFocus={() => setShowTestedModelsTooltip(true)}
                onBlur={() => setShowTestedModelsTooltip(false)}
              >
                ?
              </button>
              {showTestedModelsTooltip && (
                /* sidebar-escape: fixed positioning so the tooltip breaks out of the
                   322px sidebar and renders freely. See index.css for coords. */
                <div className="help-tooltip-popup help-tooltip-popup--sidebar-escape">
                  The models and hosts listed here are not affiliated with MOTO or Intrafere LLC. This chart reflects developer-tested configurations intended to help guide model selection. All statements regarding pricing, performance, roles, rankings, or capabilities are speculative and based on individual testing experience. Intrafere LLC and the MOTO development team make no guarantees about the accuracy of this chart. MOTO is compatible with the majority of models, including many not listed here.
                </div>
              )}
            </div>
          </h3>
          <p className="hint-text hint-text--dim" style={{ marginLeft: '20px', marginBottom: '0.45rem' }}>
            Note: Most models over 20 billion parameters are compatible with MOTO.
          </p>
          <div className="models-list">
            {/* Podium - Top 3 */}
            <div className="models-podium">
              <div className="models-podium-label">Leaderboard</div>
              <div className="model-item model-item--ranked model-item--gold model-item--os">
                <OsTag />
                <div className="flex-row-center">
                  <div className="model-item-name">Kimi K2.6</div>
                  <div
                    className="help-tooltip-anchor"
                    style={{ zIndex: 100 }}
                    aria-label="Learn about the King of the Hill ranking"
                    onMouseEnter={() => setShowKothTooltip(true)}
                    onMouseLeave={() => setShowKothTooltip(false)}
                    onFocus={() => setShowKothTooltip(true)}
                    onBlur={() => setShowKothTooltip(false)}
                    tabIndex={0}
                  >
                    <div className="ranking-badge ranking-badge--gold">👑 KING OF THE HILL</div>
                    {showKothTooltip && (
                      <div
                        className="help-tooltip-popup"
                        style={{ top: 'auto', bottom: 'calc(100% + 10px)', left: 'calc(100% + 10px)', right: 'auto' }}
                      >
                        This model was chosen by the Intrafere developers as the best overall performer in the MOTO harness, optimized for cost, speed, and knowledge.
                      </div>
                    )}
                  </div>
                </div>
                <div className="model-item-badge">Highly knowledgeable and balanced cost</div>
              </div>

              <div className="model-item model-item--ranked model-item--silver">
                <div className="flex-row-center">
                  <div className="model-item-name">Grok 4.1 Fast</div>
                  <div className="ranking-badge ranking-badge--silver">🥈 SILVER</div>
                </div>
                <div className="model-item-badge">Fast validator</div>
              </div>

              <div className="model-item model-item--ranked model-item--bronze model-item--os">
                <OsTag />
                <div className="flex-row-center">
                  <div className="model-item-name">GPT OSS 120B</div>
                  <div className="ranking-badge ranking-badge--bronze">🥉 BRONZE</div>
                </div>
                <div className="model-item-badge">Balanced knowledge and speed at low cost</div>
              </div>
            </div>

            {/* Alphabetical list (rest of models) */}

            <div className="model-item">
              <div className="model-item-name">Arcee AI's Trinity Large</div>
              <div className="model-item-badge">Highly knowledgeable</div>
            </div>
            
            <div className="model-item">
              <div className="model-item-name">Amazon Nova Pro/Premier</div>
              <div className="model-item-badge">Highly knowledgeable</div>
            </div>
            
            <div className="model-item">
              <div className="model-item-name">Claude Opus/Sonnet</div>
              <div className="model-item-badge">Highly knowledgeable</div>
            </div>
            
            <div className="model-item model-item--os">
              <OsTag />
              <div className="model-item-name">DeepSeek</div>
              <div className="model-item-badge">Highly knowledgeable</div>
            </div>
            
            <div className="model-item">
              <div className="model-item-name">Gemini Flash</div>
              <div className="model-item-badge">Fast validator</div>
            </div>
            
            <div className="model-item">
              <div className="model-item-name">Gemini Pro</div>
              <div className="model-item-badge">Highly knowledgeable</div>
            </div>
            
            <div className="model-item model-item--os">
              <OsTag />
              <div className="model-item-name">Google's Gemma</div>
              <div className="model-item-badge">Balanced knowledge and speed</div>
            </div>
            
            <div className="model-item model-item--os">
              <OsTag />
              <div className="model-item-name">GLM</div>
              <div className="model-item-badge">Highly knowledgeable</div>
            </div>
            
            <div className="model-item model-item--os">
              <OsTag />
              <div className="model-item-name">GLM Turbo</div>
              <div className="model-item-badge">Fast validator</div>
            </div>
            
            <div className="model-item">
              <div className="model-item-name">GPT Codex</div>
              <div className="model-item-badge">Computer science</div>
            </div>
            
            <div className="model-item model-item--os">
              <OsTag />
              <div className="model-item-name">OpenAI's GPT OSS</div>
              <div className="model-item-badge">Balanced knowledge and speed</div>
            </div>
            
            <div className="model-item">
              <div className="model-item-name">Grok</div>
              <div className="model-item-badge">Highly knowledgeable</div>
            </div>
            
            <div className="model-item">
              <div className="model-item-name">ChatGPT</div>
              <div className="model-item-badge">Highly knowledgeable</div>
            </div>

            <div className="model-item">
              <div className="model-item-name">Inception's Mercury</div>
              <div className="model-item-badge">Rapid knowledge</div>
            </div>

            <div className="model-item model-item--os">
              <OsTag />
              <div className="model-item-name">Nemotron Super</div>
              <div className="model-item-badge">Balanced knowledge and speed</div>
            </div>

            <div className="model-item model-item--os">
              <OsTag />
              <div className="model-item-name">Nous Hermes</div>
              <div className="model-item-badge">Highly knowledgeable</div>
            </div>
            
            <div className="model-item">
              <div className="model-item-name">Perplexity's Sonar</div>
              <div className="model-item-badge">Native internet search capability</div>
            </div>
            
            <div className="model-item model-item--os">
              <OsTag />
              <div className="model-item-name">Microsoft's Phi</div>
              <div className="model-item-badge">Balanced knowledge and speed</div>
            </div>

            <div className="model-item">
              <div className="model-item-name">MiniMax</div>
              <div className="model-item-badge">Highly knowledgeable</div>
            </div>
            
            <div className="model-item model-item--os">
              <OsTag />
              <div className="model-item-name">Qwen Coder</div>
              <div className="model-item-badge">Computer science</div>
            </div>
            
            <div className="model-item model-item--os">
              <OsTag />
              <div className="model-item-name">Qwen</div>
              <div className="model-item-badge">Highly knowledgeable</div>
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
          Load one of the preselected example profiles as a starting point, or create your own custom profile. (These models and hosts are not affiliated with MOTO/Intrafere)
        </p>
        
        <div className="settings-row">
          <label>
            Select Profile
            <HelpTooltip
              label="Learn how profile selection works"
              anchorClassName="help-tooltip-anchor--inline"
              buttonClassName="help-tooltip-btn--green"
              useFixedPosition
            >
              <strong>Profile menu guide</strong>
              <br /><br />
              <code>-- Custom Settings --</code> means no saved profile is currently loaded, so you are editing the settings manually.
              <br /><br />
              <code>Recommended Profiles</code> are preselected example profiles you can load as starting points.
              <br /><br />
              <code>My Profiles</code> contains any custom profiles you save from your current settings.
            </HelpTooltip>
          </label>
          <select
            value={selectedProfile}
            onChange={(e) => {
              const value = e.target.value;
              if (!value) {
                setSelectedProfile('');
                return;
              }

              if (!hasOpenRouterKey) {
                alert('OpenRouter API key required to use profiles. Please set your API key first.');
                return;
              }
              if (openRouterModels.length === 0) {
                alert('Please wait for OpenRouter models to load, or click "Refresh OpenRouter Models" button below.');
                return;
              }
              applyProfile(value);
            }}
            disabled={isRunning}
          >
            <option value="">-- Custom Settings --</option>
            <optgroup label="Recommended Profiles">
              {RECOMMENDED_PROFILE_KEYS
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
            className="secondary ml-05"
            onClick={() => setShowSaveDialog(true)}
            disabled={isRunning}
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
        <div className="inline-modal-overlay">
          <div className="inline-modal-content">
            <h3 style={{ marginTop: 0 }}>Save Profile</h3>
            <p className="label--muted">
              Enter a name for this profile. Current settings will be saved.
            </p>
            <input
              type="text"
              value={newProfileName}
              onChange={(e) => setNewProfileName(e.target.value)}
              placeholder="Profile name..."
              className="input-dark"
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
                className="btn-success-sm"
                onClick={saveCurrentAsProfile}
              >
                Save Profile
              </button>
            </div>
          </div>
        </div>
      )}

      {/* OpenRouter Status Banner */}
      {!hasOpenRouterKey && (
        <div className="openrouter-banner openrouter-banner--orange">
          <p className="openrouter-banner__text">
            <strong>💡 OpenRouter Available:</strong> Set your OpenRouter API key in the header to enable cloud model selection for any role.
          </p>
        </div>
      )}

      {/* Show only free models + model refresh controls — grouped at top */}
      <div className="model-refresh-controls">
        {lmStudioEnabled && (
          <button 
            className="secondary"
            onClick={async () => {
              try {
                const freshModels = await api.getModels();
                setLmStudioModels(freshModels.models || freshModels || []);
              } catch (err) {
                console.error('Failed to refresh LM Studio models:', err);
              }
            }}
            disabled={isRunning}
          >
            Refresh LM Studio Models
          </button>
        )}
        {hasOpenRouterKey && (
          <>
            <button 
              className="secondary"
              onClick={() => fetchOpenRouterModels(freeOnly)}
              disabled={isRunning || loadingOpenRouter}
            >
              {loadingOpenRouter ? 'Loading...' : 'Refresh OpenRouter Models'}
            </button>
            <button
              className="secondary"
              onClick={() => window.open('https://openrouter.ai/models', '_blank', 'noopener,noreferrer')}
              title="Browse all available OpenRouter models"
            >
              🔗 OpenRouter Model List
            </button>
            <label
              className="settings-checkbox-label model-refresh-controls__toggle"
              style={{ cursor: isRunning ? 'not-allowed' : 'pointer' }}
            >
              <input
                type="checkbox"
                checked={freeOnly}
                onChange={(e) => setFreeOnly(e.target.checked)}
                disabled={isRunning}
                style={{ marginRight: '0.5rem' }}
              />
              Free models only
            </label>
          </>
        )}
      </div>

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
          (() => {
            const effectiveProvider = lmStudioEnabled ? cfg.provider : 'openrouter';
            return (
          <div 
            key={idx} 
            className={`submitter-config-section${effectiveProvider === 'openrouter' ? ' role-config-card--openrouter-orange' : (idx === 0 ? ' role-config-card--main' : '')}`}
          >
            <h5 className={effectiveProvider === 'openrouter' ? 'card-title--orange' : (idx === 0 ? 'card-title--green' : '')}>
              {idx === 0 ? 'Submitter 1 (Main Submitter)' : `Submitter ${idx + 1}`}
              {effectiveProvider === 'openrouter' && <span className="provider-badge-inline">[OpenRouter]</span>}
            </h5>
            
            <ModelSelector
              provider={cfg.provider}
              modelId={cfg.modelId}
              openrouterProv={cfg.openrouterProvider}
              fallback={cfg.lmStudioFallbackId}
              onProviderChange={(p) => handleSubmitterConfigChange(idx, 'provider', p)}
              onModelChange={(m) => handleSubmitterModelChange(idx, m)}
              onOpenrouterProviderChange={(p) => handleSubmitterOpenRouterProviderChange(idx, p)}
              onFallbackChange={(f) => handleSubmitterConfigChange(idx, 'lmStudioFallbackId', f)}
              lmStudioModels={lmStudioModels}
              openRouterModels={openRouterModels}
              modelProviders={modelProviders}
              hasOpenRouterKey={hasOpenRouterKey}
              isRunning={isRunning}
              lmStudioEnabled={lmStudioEnabled}
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
                max={50000000}
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
                max={50000000}
                step={1000}
              />
            </div>
          </div>
            );
          })()
        ))}
      </div>

      {/* Validator (Single) */}
      <div className="settings-group">
        <h4>Validator (Single Instance)</h4>
        <p className="settings-info">
          This single validator model is the gatekeeper of what gets accepted. This model's speed will be your biggest bottleneck for the system, however its knowledge capability is also very important. Choose this model wisely, about half of all API calls will be to this model so it will also greatly control system cost. This is the model that will reject wrong answers, off-track answers, etc. at all stages of solution creation and all solutions run through a single instance to ensure user alignment (markov-chain style bottleneck).
        </p>

        <RoleConfig
          title="Validator"
          rolePrefix="validator"
          borderColor="#ff6b6b"
          localConfig={localConfig}
          handleProviderChange={handleProviderChange}
          handleModelChange={handleModelChange}
          handleOpenRouterProviderChange={handleOpenRouterProviderChange}
          handleChange={handleChange}
          handleNumericBlur={handleNumericBlur}
          isRunning={isRunning}
          lmStudioModels={lmStudioModels}
          openRouterModels={openRouterModels}
          modelProviders={modelProviders}
          hasOpenRouterKey={hasOpenRouterKey}
          lmStudioEnabled={lmStudioEnabled}
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
          handleOpenRouterProviderChange={handleOpenRouterProviderChange}
          handleChange={handleChange}
          handleNumericBlur={handleNumericBlur}
          isRunning={isRunning}
          lmStudioModels={lmStudioModels}
          openRouterModels={openRouterModels}
          modelProviders={modelProviders}
          hasOpenRouterKey={hasOpenRouterKey}
          lmStudioEnabled={lmStudioEnabled}
        />

        <RoleConfig
          title="High-Parameter Submitter"
          hint="Handles mathematical rigor enhancement."
          rolePrefix="high_param"
          borderColor="#2a2a2a"
          localConfig={localConfig}
          handleProviderChange={handleProviderChange}
          handleModelChange={handleModelChange}
          handleOpenRouterProviderChange={handleOpenRouterProviderChange}
          handleChange={handleChange}
          handleNumericBlur={handleNumericBlur}
          isRunning={isRunning}
          lmStudioModels={lmStudioModels}
          openRouterModels={openRouterModels}
          modelProviders={modelProviders}
          hasOpenRouterKey={hasOpenRouterKey}
          lmStudioEnabled={lmStudioEnabled}
        />

        <RoleConfig
          title="Critique Submitter"
          hint="Handles post-body peer review feedback and rewrite decisions."
          rolePrefix="critique_submitter"
          borderColor="#e74c3c"
          localConfig={localConfig}
          handleProviderChange={handleProviderChange}
          handleModelChange={handleModelChange}
          handleOpenRouterProviderChange={handleOpenRouterProviderChange}
          handleChange={handleChange}
          handleNumericBlur={handleNumericBlur}
          isRunning={isRunning}
          lmStudioModels={lmStudioModels}
          openRouterModels={openRouterModels}
          modelProviders={modelProviders}
          hasOpenRouterKey={hasOpenRouterKey}
          lmStudioEnabled={lmStudioEnabled}
        />
      </div>

      <div className="settings-group">
        <div
          className="collapsible-trigger settings-trigger--multiline"
          onClick={() => setAdvancedSettingsExpanded(prev => !prev)}
          onKeyDown={(event) => handleCollapsibleKeyDown(event, () => setAdvancedSettingsExpanded(prev => !prev))}
          role="button"
          tabIndex={0}
          aria-expanded={advancedSettingsExpanded}
          aria-controls="advanced-settings-panel"
          style={{ marginBottom: advancedSettingsExpanded ? '1rem' : 0 }}
        >
          <div className="settings-heading-stack">
            <h4 className="form-group--compact">Advanced Settings</h4>
            <p className="settings-subsection-description">
              Optional integrations, Stage 3 controls, prompt customization, interface polish, and OpenRouter fallback behavior.
            </p>
          </div>
          <span className={`collapse-chevron${advancedSettingsExpanded ? ' collapse-chevron--open' : ''}`}>▼</span>
        </div>

        {advancedSettingsExpanded && (
          <div className="collapsible-body settings-advanced-content" id="advanced-settings-panel">
            {isRunning && (
              <div className="settings-notice">
                Settings cannot be changed while autonomous research is running.
              </div>
            )}

            {/* Wolfram Alpha Integration */}
            <div className="settings-subsection">
              <div className="settings-subsection-header">
                <h5 className="settings-subsection-title">Integrations</h5>
                <p className="settings-subsection-description">
                  Optional external verification tools used by rigor mode.
                </p>
              </div>

              {showLean4Settings && (
                <div style={{ marginBottom: '1.5rem' }}>
                  <h4 className="form-group--compact">Lean 4 Proof Solver</h4>
                  <small className="hint-text">
                    Desktop-only controls for the automatic proof checker, manual proof runs, and certificate export.
                  </small>

                  <div className="settings-row">
                    <label>Lean 4 Status</label>
                    <div>
                      <strong>{proofStatus ? (proofStatus.lean4_enabled ? 'Enabled' : 'Disabled') : 'Starting…'}</strong>
                      <small className="settings-hint" style={{ display: 'block', marginTop: '0.35rem' }}>
                        Workspace: {proofStatus ? (proofStatus.workspace_ready ? 'Ready' : 'Not ready yet') : 'Starting…'}
                      </small>
                    </div>
                  </div>

                  <div className="settings-row">
                    <label>Lean Version</label>
                    <div>{proofStatus?.lean4_version || 'Unavailable'}</div>
                  </div>

                  <div className="settings-row">
                    <label>Mathlib Revision</label>
                    <div>{proofStatus?.mathlib_commit || 'Unavailable'}</div>
                  </div>

                  <div className="settings-row">
                    <label>Lean Binary</label>
                    <div>{proofStatus?.lean4_path || 'Launcher-managed / not detected yet'}</div>
                  </div>

                  <div className="settings-row">
                    <label>Workspace Directory</label>
                    <div>{proofStatus?.lean4_workspace_dir || 'Unavailable'}</div>
                  </div>

                  <div className="settings-row">
                    <label>Persistent LSP Status</label>
                    <div>
                      {proofStatus?.lsp_active
                        ? 'Active'
                        : proofStatus?.lsp_available
                          ? 'Available'
                          : 'Disabled'}
                    </div>
                  </div>

                  <div className="settings-row">
                    <label>Z3 Status</label>
                    <div>
                      <strong>{proofStatus?.smt_available ? 'Ready' : 'Unavailable'}</strong>
                      <small className="settings-hint" style={{ display: 'block', marginTop: '0.35rem' }}>
                        {proofStatus?.z3_version || 'No Z3 version detected yet'}
                      </small>
                    </div>
                  </div>

                  <label className="settings-checkbox-label settings-checkbox-label--stacked" style={{ cursor: isRunning ? 'not-allowed' : 'pointer', marginTop: '1rem' }}>
                    <input
                      type="checkbox"
                      checked={proofSettingsEnabled}
                      onChange={(e) => setProofSettingsEnabled(e.target.checked)}
                      disabled={isRunning || savingProofSettings}
                    />
                    <span className="settings-option-copy">
                      <span className="settings-option-title">Enable Lean 4 proof verification</span>
                      <span className="settings-option-description">
                        Turns on automatic proof checks after brainstorm and paper completion plus manual proof checks from the Proofs tab.
                      </span>
                    </span>
                  </label>

                  <label className="settings-checkbox-label settings-checkbox-label--stacked" style={{ cursor: isRunning ? 'not-allowed' : 'pointer', marginTop: '1rem' }}>
                    <input
                      type="checkbox"
                      checked={proofSettingsLspEnabled}
                      onChange={(e) => setProofSettingsLspEnabled(e.target.checked)}
                      disabled={isRunning || savingProofSettings}
                    />
                    <span className="settings-option-copy">
                      <span className="settings-option-title">Enable persistent Lean LSP mode</span>
                      <span className="settings-option-description">
                        Keeps a warm Lean server available for lower-latency proof verification while preserving subprocess fallback.
                      </span>
                    </span>
                  </label>

                  <div className="settings-row">
                    <label>Proof Timeout (seconds)</label>
                    <input
                      type="number"
                      value={proofSettingsTimeout}
                      onChange={(e) => setProofSettingsTimeout(e.target.value)}
                      disabled={isRunning || savingProofSettings}
                      min={10}
                      max={3600}
                      step={5}
                    />
                  </div>

                  <div className="settings-row">
                    <label>LSP Idle Timeout (seconds)</label>
                    <input
                      type="number"
                      value={proofSettingsLspIdleTimeout}
                      onChange={(e) => setProofSettingsLspIdleTimeout(e.target.value)}
                      disabled={isRunning || savingProofSettings}
                      min={60}
                      max={7200}
                      step={30}
                    />
                  </div>

                  <div style={{ marginTop: '1rem' }}>
                    <h5 className="form-group--compact">SMT (Z3) Integration</h5>
                    <small className="hint-text">
                      Optional early theorem classification and Lean tactic hinting for arithmetic-friendly proof goals. Lean 4 remains authoritative for every stored proof.
                    </small>
                  </div>

                  <label className="settings-checkbox-label settings-checkbox-label--stacked" style={{ cursor: isRunning ? 'not-allowed' : 'pointer', marginTop: '1rem' }}>
                    <input
                      type="checkbox"
                      checked={proofSettingsSmtEnabled}
                      onChange={(e) => setProofSettingsSmtEnabled(e.target.checked)}
                      disabled={isRunning || savingProofSettings}
                    />
                    <span className="settings-option-copy">
                      <span className="settings-option-title">Enable SMT-assisted proof guidance</span>
                      <span className="settings-option-description">
                        Runs Z3 on conservative SMT-amenable goals and feeds any successful result back into Lean proof prompting as hints only.
                      </span>
                    </span>
                  </label>

                  <div className="settings-row">
                    <label>Z3 Binary Path</label>
                    <input
                      type="text"
                      value={proofSettingsZ3Path}
                      onChange={(e) => setProofSettingsZ3Path(e.target.value)}
                      disabled={isRunning || savingProofSettings}
                      placeholder="Optional explicit z3 path"
                    />
                  </div>

                  <div className="settings-row">
                    <label>SMT Timeout (seconds)</label>
                    <input
                      type="number"
                      value={proofSettingsSmtTimeout}
                      onChange={(e) => setProofSettingsSmtTimeout(e.target.value)}
                      disabled={isRunning || savingProofSettings}
                      min={1}
                      max={600}
                      step={1}
                    />
                  </div>

                  <div className="actions-row">
                    <button
                      className="btn-success-sm"
                      onClick={handleSaveProofSettings}
                      disabled={isRunning || savingProofSettings}
                    >
                      {savingProofSettings ? 'Saving...' : 'Save Proof Settings'}
                    </button>
                  </div>

                  {proofSettingsMessage && (
                    <div className={`test-result-banner ${proofSettingsMessage.startsWith('Failed') ? 'test-result-banner--error' : 'test-result-banner--success'}`}>
                      {proofSettingsMessage}
                    </div>
                  )}
                </div>
              )}

              <h4 className="form-group--compact">Wolfram Alpha Integration (Optional)</h4>
              <small className="hint-text">
                Enable Wolfram Alpha API for computational verification in rigor mode. When selecting your key select "full results" for your key type, then copy your APP ID and save it here. This key is also shared with the manual compiler mode.
                Get your API key from <a href="https://products.wolframalpha.com/api" target="_blank" rel="noopener noreferrer">developer.wolframalpha.com</a>
              </small>

              <label className="settings-checkbox-label settings-checkbox-label--stacked">
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
                <span className="settings-option-copy">
                  <span className="settings-option-title">Enable Wolfram Alpha Verification in Rigor Mode</span>
                  <span className="settings-option-description">
                    Lets rigor mode request computational verification for equations, properties, and theorem checks.
                  </span>
                </span>
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
                    />
                    {hasStoredWolframKey && !wolframApiKey && (
                      <small className="hint-text">
                        {genericMode
                          ? 'A Wolfram Alpha key is already loaded in the current backend session.'
                          : 'A Wolfram Alpha key is already stored securely on the backend for this machine.'}
                      </small>
                    )}
                  </div>

                  <div style={{ display: 'flex', gap: '0.75rem', marginTop: '1rem' }}>
                    <button
                      className="btn-success-sm"
                      onClick={handleTestWolframConnection}
                      disabled={testingWolfram}
                      style={testingWolfram ? { cursor: 'wait', opacity: 0.6 } : undefined}
                    >
                      {testingWolfram ? 'Testing...' : 'Test Connection'}
                    </button>

                    <button
                      className="btn-ghost"
                      onClick={handleClearWolframKey}
                    >
                      Clear Key
                    </button>
                  </div>

                  {wolframTestResult && (
                    <div className={`test-result-banner ${wolframTestResult.includes('✓') ? 'test-result-banner--success' : 'test-result-banner--error'}`}>
                      {wolframTestResult}
                    </div>
                  )}

                  <small className="hint-text">
                    In rigor mode, the AI can request Wolfram Alpha verification of mathematical claims.
                    This enables computational checking of theorems, solving equations, and verifying properties.
                    This setting is shared with the manual compiler mode.
                  </small>
                </div>
              )}
            </div>

            {/* Tier 3 Final Answer Toggle */}
            <div className="settings-subsection settings-subsection--accent-danger">
              <div className="settings-subsection-header">
                <h5 className="settings-subsection-title">Advanced / Ending Options</h5>
              </div>
              <h4 className="form-group--compact">Stage 3: Final Answer Generation</h4>
              <p className="settings-info">
                Feature in construction. Enabling this is optional and not recommended. Stage 3 is an in-development mode. Most users should not enable this feature — it is expensive and wasteful at this current stage of development. When enabled, the system will automatically synthesize all completed Stage 2 papers into a final answer that is often book-length or greater. This feature is highly hallucinatory — Stage 2 papers are the recommended final output. Disabled by default; final paper quality is currently much lower than Stage 2 papers. Once optimized and better-functioning, this mode will be advertised more.
              </p>
              <label className="settings-checkbox-label settings-checkbox-label--stacked" style={{ cursor: isRunning ? 'not-allowed' : 'pointer' }}>
                <input
                  type="checkbox"
                  checked={tier3Enabled}
                  onChange={(e) => setTier3Enabled(e.target.checked)}
                  disabled={isRunning}
                />
                <span className="settings-option-copy">
                  <span className="settings-option-title">Enable Stage 3 Final Answer Generation (In Development)</span>
                  <span className="settings-option-description">
                    Allows the system to synthesize completed Stage 2 papers into a final answer after enough papers accumulate.
                  </span>
                </span>
              </label>
            </div>

            {/* Validator Critique Prompt Editor */}
            <div className="settings-subsection">
              <div className="settings-subsection-header">
                <h5 className="settings-subsection-title">Prompt Customization</h5>
                <p className="settings-subsection-description">
                  Optional tweaks for the user-facing paper critique prompt only.
                </p>
              </div>

              <div
                className="collapsible-trigger settings-trigger--multiline"
                onClick={() => setCritiquePromptExpanded(prev => !prev)}
                onKeyDown={(event) => handleCollapsibleKeyDown(event, () => setCritiquePromptExpanded(prev => !prev))}
                role="button"
                tabIndex={0}
                aria-expanded={critiquePromptExpanded}
                aria-controls="critique-prompt-panel"
                style={{ marginBottom: critiquePromptExpanded ? '1rem' : 0 }}
              >
                <div className="settings-trigger-copy">
                  <div className="settings-trigger-title-row">
                    <h4 className="form-group--compact settings-trigger-title">Edit Validator Critique Prompt</h4>
                    {isUsingCustomCritiquePrompt && (
                      <span className="tag-badge tag-badge--purple">CUSTOM</span>
                    )}
                  </div>
                  <p className="settings-subsection-description">
                    Optional prompt customization for the user-facing paper critique mode only. This does not affect the internal critique submitter used during autonomous research.
                  </p>
                </div>
                <span className={`collapse-chevron${critiquePromptExpanded ? ' collapse-chevron--open' : ''}`}>▼</span>
              </div>

              {critiquePromptExpanded && (
                <div className="collapsible-body" id="critique-prompt-panel">
                  <p className="hint-text">
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
                      className="btn-ghost"
                      onClick={handleRestoreCritiquePrompt}
                    >
                      Restore to Default
                    </button>

                    <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
                      {critiquePromptSaved && (
                        <span className="status-success-text">✓ Saved!</span>
                      )}
                      <button
                        className="btn-accent-purple"
                        onClick={handleSaveCritiquePrompt}
                      >
                        Save Prompt
                      </button>
                    </div>
                  </div>
                </div>
              )}
            </div>

            <div className="settings-subsection">
              <div className="settings-subsection-header">
                <h5 className="settings-subsection-title">Interface</h5>
                <p className="settings-subsection-description">
                  Display-only controls for the autonomous research UI.
                </p>
              </div>

              <label className="settings-checkbox-label settings-checkbox-label--stacked">
                <input
                  type="checkbox"
                  checked={(() => {
                    const saved = localStorage.getItem('banner_shimmer_enabled');
                    return saved !== null ? JSON.parse(saved) : true;
                  })()}
                  onChange={(e) => {
                    localStorage.setItem('banner_shimmer_enabled', JSON.stringify(e.target.checked));
                    window.location.reload();
                  }}
                />
                <span className="settings-option-copy">
                  <span className="settings-option-title">Enable shimmer accents</span>
                  <span className="settings-option-description">
                    Keeps the animated banner shimmer and subtle active-tab border sheen on. Disable this when recording video to reduce motion and visual noise.
                  </span>
                </span>
              </label>
            </div>

            {/* Free model looping/auto-selector options */}
            {hasOpenRouterKey && (
              <div className="settings-subsection">
                <div className="settings-subsection-header">
                  <h5 className="settings-subsection-title">OpenRouter Fallback</h5>
                  <p className="settings-subsection-description">
                    Fallback behavior for OpenRouter free-model rate limits.
                  </p>
                </div>

                <div className="checkbox-group-col">
                  <label className="settings-checkbox-label settings-checkbox-label--stacked">
                    <input
                      type="checkbox"
                      checked={freeModelLooping}
                      onChange={(e) => {
                        setFreeModelLooping(e.target.checked);
                        openRouterAPI.setFreeModelSettings(e.target.checked, freeModelAutoSelector).catch(() => {});
                      }}
                    />
                    <span className="settings-option-copy">
                      <span className="settings-option-title">
                        Enable Free Model Looping
                        <HelpTooltip
                          label="Learn about free model looping"
                          anchorClassName="help-tooltip-anchor--inline"
                          popupStyle={{ top: 'auto', bottom: 'calc(100% + 10px)', left: 'calc(100% + 10px)', right: 'auto' }}
                        >
                          When a free model is rate-limited, automatically try the next available free model sorted by highest context limit. Prevents workflow stalls from rate limits.
                        </HelpTooltip>
                      </span>
                      <span className="settings-option-description">
                        Automatically rotate to the next selected free model when one hits a rate limit.
                      </span>
                    </span>
                  </label>
                  <label className="settings-checkbox-label settings-checkbox-label--stacked">
                    <input
                      type="checkbox"
                      checked={freeModelAutoSelector}
                      onChange={(e) => {
                        setFreeModelAutoSelector(e.target.checked);
                        openRouterAPI.setFreeModelSettings(freeModelLooping, e.target.checked).catch(() => {});
                      }}
                    />
                    <span className="settings-option-copy">
                      <span className="settings-option-title">
                        Use OpenRouter Free Models Auto-Selector as Backup
                        <HelpTooltip
                          label="Learn about the free models auto-selector backup"
                          anchorClassName="help-tooltip-anchor--inline"
                          popupStyle={{ top: 'auto', bottom: 'calc(100% + 10px)', left: 'calc(100% + 10px)', right: 'auto' }}
                        >
                          When all selected free models are rate-limited, use OpenRouter&apos;s Free Models Router (`openrouter/free`) as a last resort backup. Works independently of Free Model Looping.
                        </HelpTooltip>
                      </span>
                      <span className="settings-option-description">
                        Falls back to OpenRouter&apos;s free router when every selected free model is temporarily exhausted.
                      </span>
                    </span>
                  </label>
                </div>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
    </div>
  );
};

export default AutonomousResearchSettings;
