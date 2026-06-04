/**
 * AutonomousResearchSettings - Settings panel for autonomous research mode.
 * Supports configurable multi-submitter brainstorm aggregation (1-10 submitters).
 * Compiler settings (high-context, high-param) remain separate.
 * Now supports per-role OpenRouter model selection with provider and fallback options.
 */
import React, { useState, useEffect } from 'react';
import { cloudAccessAPI, openRouterAPI, api, autonomousAPI } from '../../services/api';
import {
  computeCodexAutoSettings,
  computeOpenRouterAutoSettings,
  DEFAULT_CONTEXT_WINDOW,
  DEFAULT_MAX_OUTPUT_TOKENS,
  DEFAULT_OPENROUTER_REASONING_EFFORT,
  findOpenRouterModel,
  getProviderNames,
  getReasoningSupportInfo,
  hasEndpointMetadata,
  normalizeOpenRouterReasoningEffort,
  OPENROUTER_REASONING_EFFORT_OPTIONS,
} from '../../utils/openRouterSelection';
import {
  AUTONOMOUS_SETTINGS_STORAGE_KEY,
  AUTONOMOUS_PROFILES_STORAGE_KEY,
  RECOMMENDED_PROFILE_KEYS,
  RECOMMENDED_PROFILES,
  applyAutonomousProfileSelection,
  getStoredAutonomousSettings,
  persistAutonomousSettings,
  settingsToAutonomousConfig,
} from '../../utils/autonomousProfiles';
import HelpTooltip from '../HelpTooltip';
import HighlightedModelsSidebar from '../HighlightedModelsSidebar';
import ProofStrengthBadge from '../ProofStrengthBadge';
import RawSettingsEditor from '../RawSettingsEditor';
import './AutonomousResearch.css';
import '../settings-common.css';

const DEFAULT_SUBMITTER_CONFIG = {
  submitterId: 1,
  provider: 'lm_studio',
  modelId: '',
  openrouterProvider: null,
  openrouterReasoningEffort: DEFAULT_OPENROUTER_REASONING_EFFORT,
  lmStudioFallbackId: null,
  contextWindow: DEFAULT_CONTEXT_WINDOW,
  maxOutputTokens: DEFAULT_MAX_OUTPUT_TOKENS,
  superchargeEnabled: false
};

const RAW_VIEW_EXIT_WARNING = 'Switching back to the GUI view will restore your last GUI settings/profile and discard raw-only changes. Continue?';
const formatRawSettings = (value) => JSON.stringify(value, null, 2);
const SUPERCHARGE_TOOLTIP = 'Supercharge makes this role generate 4 full answer attempts, then run a 5th same-model call to choose or synthesize the best final answer. It uses 5x the API calls, so it is about 5x slower and 5x more costly, but can produce more intelligent answers.';

// ModelSelector component - extracted outside to prevent recreation on every render
const ModelSelector = ({
  provider,
  modelId,
  openrouterProv,
  openrouterReasoningEffort,
  fallback,
  onProviderChange,
  onModelChange,
  onOpenrouterProviderChange,
  onOpenrouterReasoningEffortChange,
  onFallbackChange,
  lmStudioModels,
  openRouterModels,
  openAICodexModels,
  modelProviders,
  hasOpenRouterKey,
  hasOpenAICodexLogin,
  isRunning,
  lmStudioEnabled,
}) => {
  const effectiveProvider = lmStudioEnabled ? provider : 'openrouter';
  const currentModels = effectiveProvider === 'openrouter'
    ? openRouterModels
    : (effectiveProvider === 'openai_codex_oauth' ? openAICodexModels : lmStudioModels);
  const providers = modelId && effectiveProvider === 'openrouter'
    ? getProviderNames(modelProviders[modelId])
    : [];
  const reasoningInfo = effectiveProvider === 'openrouter'
    ? getReasoningSupportInfo(modelProviders[modelId], openrouterProv || null)
    : { hasEndpointMetadata: false, supportsReasoning: false };

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
            <button
              type="button"
              className={`provider-toggle-btn${provider === 'openai_codex_oauth' ? ' active-or-orange' : ''}`}
              onClick={() => hasOpenAICodexLogin && onProviderChange('openai_codex_oauth')}
              disabled={isRunning || !hasOpenAICodexLogin}
              style={!hasOpenAICodexLogin ? { color: '#666' } : undefined}
              title={!hasOpenAICodexLogin ? 'Set OpenAI Codex login in Cloud Access & Keys first' : 'Use OpenAI Codex'}
            >
              OpenAI Codex
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

      {effectiveProvider === 'openrouter' && modelId && (
        <div className="settings-row">
          <label>Reasoning Effort</label>
          <select
            value={normalizeOpenRouterReasoningEffort(openrouterReasoningEffort)}
            onChange={(e) => onOpenrouterReasoningEffortChange(e.target.value)}
            disabled={isRunning}
          >
            {OPENROUTER_REASONING_EFFORT_OPTIONS.map(option => (
              <option key={option.value} value={option.value}>{option.label}</option>
            ))}
          </select>
          <small className="settings-hint">
            {reasoningInfo.hasEndpointMetadata && !reasoningInfo.supportsReasoning
              ? 'This selected host does not advertise reasoning support; OpenRouter may ignore the setting.'
              : 'Auto sends OpenRouter max reasoning effort by default.'}
          </small>
        </div>
      )}

      {/* LM Studio Fallback (if cloud provider) */}
      {effectiveProvider !== 'lm_studio' && lmStudioEnabled && (
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
          <small className="settings-hint">Used if cloud provider access fails or credits run out</small>
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
  localConfig,
  handleProviderChange,
  handleModelChange,
  handleOpenRouterProviderChange,
  handleChange,
  handleNumericBlur,
  isRunning,
  lmStudioModels,
  openRouterModels,
  openAICodexModels,
  modelProviders,
  hasOpenRouterKey,
  hasOpenAICodexLogin,
  lmStudioEnabled,
  developerModeEnabled = false,
  showProofStrengthBadge = false,
}) => {
  const storedProvider = localConfig[`${rolePrefix}_provider`] || 'lm_studio';
  const provider = lmStudioEnabled ? storedProvider : 'openrouter';
  const modelId = localConfig[`${rolePrefix}_model`] || '';
  const openrouterProv = localConfig[`${rolePrefix}_openrouter_provider`];
  const openrouterReasoningEffort = localConfig[`${rolePrefix}_openrouter_reasoning_effort`];
  const fallback = localConfig[`${rolePrefix}_lm_studio_fallback`];
  const contextWindow = localConfig[`${rolePrefix}_context_window`] ?? DEFAULT_CONTEXT_WINDOW;
  const maxTokens = localConfig[`${rolePrefix}_max_tokens`] ?? DEFAULT_MAX_OUTPUT_TOKENS;
  const superchargeEnabled = Boolean(localConfig[`${rolePrefix}_supercharge_enabled`]);

  return (
    <div className={`submitter-config-section${provider === 'openrouter' ? ' role-config-card--openrouter-orange' : ''}`}>
      <h5 className={provider === 'openrouter' ? 'card-title--orange' : ''}>
        <span className="role-title-with-badges">
          <span>{title}</span>
          {showProofStrengthBadge && <ProofStrengthBadge />}
        </span>
        {provider === 'openrouter' && <span className="provider-badge-inline">[OpenRouter]</span>}
      </h5>
      {hint && <p className="settings-hint">{hint}</p>}

      <ModelSelector
        provider={provider}
        modelId={modelId}
        openrouterProv={openrouterProv}
        openrouterReasoningEffort={openrouterReasoningEffort}
        fallback={fallback}
        onProviderChange={(p) => handleProviderChange(rolePrefix, p)}
        onModelChange={(m) => handleModelChange(rolePrefix, m)}
        onOpenrouterProviderChange={(p) => handleOpenRouterProviderChange(rolePrefix, p)}
        onOpenrouterReasoningEffortChange={(effort) => handleChange(`${rolePrefix}_openrouter_reasoning_effort`, normalizeOpenRouterReasoningEffort(effort))}
        onFallbackChange={(f) => handleChange(`${rolePrefix}_lm_studio_fallback`, f)}
        lmStudioModels={lmStudioModels}
        openRouterModels={openRouterModels}
        openAICodexModels={openAICodexModels}
        modelProviders={modelProviders}
        hasOpenRouterKey={hasOpenRouterKey}
        hasOpenAICodexLogin={hasOpenAICodexLogin}
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

      {developerModeEnabled && (
        <div className="settings-row settings-row--inline-checkbox">
          <label className="settings-checkbox-label settings-checkbox-label--supercharge">
            <input
              type="checkbox"
              checked={superchargeEnabled}
              onChange={(e) => handleChange(`${rolePrefix}_supercharge_enabled`, e.target.checked)}
              disabled={isRunning}
            />
            <HelpTooltip
              label="Learn about Supercharge"
              buttonContent="Supercharge"
              buttonClassName="help-tooltip-btn--text"
              popupClassName="help-tooltip-popup--fixed"
              useFixedPosition
            >
              {SUPERCHARGE_TOOLTIP}
            </HelpTooltip>
          </label>
        </div>
      )}
    </div>
  );
};

const AutonomousResearchSettings = ({
  config,
  onConfigChange,
  models,
  capabilities,
  isRunning,
  developerModeEnabled = false,
}) => {
  // Models and OpenRouter state
  const [lmStudioModels, setLmStudioModels] = useState(models || []);
  const [openRouterModels, setOpenRouterModels] = useState([]);
  const [openAICodexModels, setOpenAICodexModels] = useState([]);
  const [modelProviders, setModelProviders] = useState({});
  const [hasOpenRouterKey, setHasOpenRouterKey] = useState(false);
  const [hasOpenAICodexLogin, setHasOpenAICodexLogin] = useState(false);
  const [openAICodexModelError, setOpenAICodexModelError] = useState('');
  const [loadingOpenRouter, setLoadingOpenRouter] = useState(false);
  const [freeOnly, setFreeOnly] = useState(false);
  const [freeModelLooping, setFreeModelLooping] = useState(true);
  const [freeModelAutoSelector, setFreeModelAutoSelector] = useState(true);
  const [tier3Enabled, setTier3Enabled] = useState(false);
  const [isLoadedFromStorage, setIsLoadedFromStorage] = useState(false);

  // Profile management state
  const [userProfiles, setUserProfiles] = useState({});
  const [selectedProfile, setSelectedProfile] = useState('');
  const [showSaveDialog, setShowSaveDialog] = useState(false);
  const [newProfileName, setNewProfileName] = useState('');
  const [editRawSettings, setEditRawSettings] = useState(false);
  const [rawSettingsText, setRawSettingsText] = useState('');
  const [rawSettingsMessage, setRawSettingsMessage] = useState('');
  const [guiSettingsBeforeRaw, setGuiSettingsBeforeRaw] = useState(null);

  // Wolfram Alpha settings (shared with compiler)
  const [wolframEnabled, setWolframEnabled] = useState(false);
  const [wolframApiKey, setWolframApiKey] = useState('');
  const [hasStoredWolframKey, setHasStoredWolframKey] = useState(false);
  const [wolframTestResult, setWolframTestResult] = useState('');
  const [testingWolfram, setTestingWolfram] = useState(false);
  const [proofStatus, setProofStatus] = useState(null);
  const [proofSettingsEnabled, setProofSettingsEnabled] = useState(false);
  const [proofSettingsTimeout, setProofSettingsTimeout] = useState('600');
  const [proofSettingsLspEnabled, setProofSettingsLspEnabled] = useState(false);
  const [proofSettingsLspIdleTimeout, setProofSettingsLspIdleTimeout] = useState('600');
  const [proofSettingsMaxParallelCandidates, setProofSettingsMaxParallelCandidates] = useState('6');
  const [proofSettingsSmtEnabled, setProofSettingsSmtEnabled] = useState(false);
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

  useEffect(() => {
    if (!developerModeEnabled && editRawSettings) {
      setEditRawSettings(false);
      setRawSettingsMessage('');
    }
  }, [developerModeEnabled, editRawSettings]);

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
        ...c,
        openrouterReasoningEffort: normalizeOpenRouterReasoningEffort(c.openrouterReasoningEffort || c.openrouter_reasoning_effort),
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
    validator_openrouter_reasoning_effort: DEFAULT_OPENROUTER_REASONING_EFFORT,
    validator_lm_studio_fallback: null,
    validator_context_window: DEFAULT_CONTEXT_WINDOW,
    validator_max_tokens: DEFAULT_MAX_OUTPUT_TOKENS,
    validator_supercharge_enabled: false,
    // High-Context
    high_context_provider: 'lm_studio',
    high_context_model: '',
    high_context_openrouter_provider: null,
    high_context_openrouter_reasoning_effort: DEFAULT_OPENROUTER_REASONING_EFFORT,
    high_context_lm_studio_fallback: null,
    high_context_context_window: DEFAULT_CONTEXT_WINDOW,
    high_context_max_tokens: DEFAULT_MAX_OUTPUT_TOKENS,
    high_context_supercharge_enabled: false,
    // High-Param
    high_param_provider: 'lm_studio',
    high_param_model: '',
    high_param_openrouter_provider: null,
    high_param_openrouter_reasoning_effort: DEFAULT_OPENROUTER_REASONING_EFFORT,
    high_param_lm_studio_fallback: null,
    high_param_context_window: DEFAULT_CONTEXT_WINDOW,
    high_param_max_tokens: DEFAULT_MAX_OUTPUT_TOKENS,
    high_param_supercharge_enabled: false,
    // Critique Submitter
    critique_submitter_provider: 'lm_studio',
    critique_submitter_model: '',
    critique_submitter_openrouter_provider: null,
    critique_submitter_openrouter_reasoning_effort: DEFAULT_OPENROUTER_REASONING_EFFORT,
    critique_submitter_lm_studio_fallback: null,
    critique_submitter_context_window: DEFAULT_CONTEXT_WINDOW,
    critique_submitter_max_tokens: DEFAULT_MAX_OUTPUT_TOKENS,
    critique_submitter_supercharge_enabled: false,
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
            normalized_submitter.openrouterReasoningEffort = normalizeOpenRouterReasoningEffort(normalized.submitters[0].openrouterReasoningEffort);
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
        const freeModelSettings = await openRouterAPI.getFreeModelSettings();
        setFreeModelLooping(freeModelSettings.looping_enabled ?? true);
        setFreeModelAutoSelector(freeModelSettings.auto_selector_enabled ?? true);
      } catch (err) {
        console.error('Failed to load free model settings:', err);
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
      try {
        const codexStatus = await cloudAccessAPI.getOpenAICodexStatus();
        const configured = Boolean(codexStatus.status?.configured);
        setHasOpenAICodexLogin(configured);
        if (configured) {
          fetchOpenAICodexModels();
        } else {
          setOpenAICodexModelError('');
        }
      } catch (err) {
        console.error('Failed to check OpenAI Codex login:', err);
        setHasOpenAICodexLogin(false);
        setOpenAICodexModelError(`OpenAI Codex OAuth status could not be checked: ${err.message || 'unknown error'}.`);
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
        setProofSettingsTimeout(String(status.lean4_proof_timeout ?? 600));
        setProofSettingsLspEnabled(Boolean(status.lean4_lsp_enabled));
        setProofSettingsLspIdleTimeout(String(status.lean4_lsp_idle_timeout ?? 600));
        setProofSettingsMaxParallelCandidates(String(status.proof_max_parallel_candidates ?? 6));
        setProofSettingsSmtEnabled(Boolean(status.smt_enabled));
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
        openrouterReasoningEffort: normalizeOpenRouterReasoningEffort(submitterConfig.openrouterReasoningEffort),
        lmStudioFallbackId: null,
      };
    });

    const normalizedLocalConfig = { ...localConfig };
    ['validator', 'high_context', 'high_param', 'critique_submitter'].forEach((rolePrefix) => {
      const providerKey = `${rolePrefix}_provider`;
      const modelKey = `${rolePrefix}_model`;
      const openRouterProviderKey = `${rolePrefix}_openrouter_provider`;
      const reasoningEffortKey = `${rolePrefix}_openrouter_reasoning_effort`;
      const fallbackKey = `${rolePrefix}_lm_studio_fallback`;
      const keepOpenRouterState = normalizedLocalConfig[providerKey] === 'openrouter';

      normalizedLocalConfig[providerKey] = 'openrouter';
      normalizedLocalConfig[modelKey] = keepOpenRouterState ? (normalizedLocalConfig[modelKey] || '') : '';
      normalizedLocalConfig[openRouterProviderKey] = keepOpenRouterState
        ? (normalizedLocalConfig[openRouterProviderKey] || null)
        : null;
      normalizedLocalConfig[reasoningEffortKey] = normalizeOpenRouterReasoningEffort(normalizedLocalConfig[reasoningEffortKey]);
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
      allow_mathematical_proofs: config?.allow_mathematical_proofs ?? true,
      allow_research_papers: config?.allow_research_papers ?? true,
      tier3_enabled: tier3Enabled,
    };
    const nextConfig = {
      ...normalizedLocalConfig,
      submitter_configs: normalizedSubmitters.slice(0, numSubmitters),
      allow_mathematical_proofs: config?.allow_mathematical_proofs ?? true,
      allow_research_papers: config?.allow_research_papers ?? true,
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
    onConfigChange({
      ...localConfig,
      submitter_configs: submitterConfigs.slice(0, numSubmitters),
      allow_mathematical_proofs: config?.allow_mathematical_proofs ?? true,
      allow_research_papers: config?.allow_research_papers ?? true,
      tier3_enabled: tier3Enabled
    });
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

  const fetchOpenAICodexModels = async () => {
    try {
      const result = await cloudAccessAPI.getOpenAICodexModels();
      const models = result.models || [];
      setOpenAICodexModels(models);
      setHasOpenAICodexLogin(models.length > 0);
      setOpenAICodexModelError(models.length > 0
        ? ''
        : 'OpenAI Codex OAuth is connected, but no Codex models were returned. Reconnect OAuth or check account access.'
      );
    } catch (err) {
      console.error('Failed to fetch OpenAI Codex models:', err);
      setOpenAICodexModels([]);
      setHasOpenAICodexLogin(false);
      setOpenAICodexModelError(`OpenAI Codex OAuth is connected, but models could not be loaded: ${err.message || 'unknown error'}.`);
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

  const getCodexAutoSettingsForModel = (modelId) => {
    const model = openAICodexModels.find((item) => item.id === modelId);
    if (!model) {
      console.debug('[AutonomousCodexAutoFill] model not in loaded list, skipping auto-fill', { modelId });
      return null;
    }
    const autoSettings = computeCodexAutoSettings(model);
    if (autoSettings.warnings.length > 0) {
      console.warn('[AutonomousCodexAutoFill] auto-settings fallback used:', autoSettings.warnings);
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
      const finalValue = isNaN(parsed) ? '' : parsed;
      
      const newConfig = { ...localConfig, [field]: finalValue };
      markProfileAsCustom();
      setLocalConfig(newConfig);
      onConfigChange({ ...newConfig, submitter_configs: submitterConfigs.slice(0, numSubmitters) });
    }
  };

  // Handle provider change for a role (keeps existing model settings)
  const handleProviderChange = (rolePrefix, provider) => {
    const updates = {
      [`${rolePrefix}_provider`]: provider,
      [`${rolePrefix}_openrouter_reasoning_effort`]: DEFAULT_OPENROUTER_REASONING_EFFORT
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
      [`${rolePrefix}_openrouter_reasoning_effort`]: DEFAULT_OPENROUTER_REASONING_EFFORT,
    };
    markProfileAsCustom();
    setLocalConfig(newConfig);
    onConfigChange({ ...newConfig, submitter_configs: submitterConfigs.slice(0, numSubmitters) });

    const provider = localConfig[`${rolePrefix}_provider`];
    if (!modelId || !['openrouter', 'openai_codex_oauth'].includes(provider)) {
      return;
    }

    const autoSettings = provider === 'openrouter'
      ? await getAutoSettingsForModel(modelId, null)
      : getCodexAutoSettingsForModel(modelId);
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
        openrouterReasoningEffort: DEFAULT_OPENROUTER_REASONING_EFFORT,
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
      openrouterReasoningEffort: DEFAULT_OPENROUTER_REASONING_EFFORT,
    };

    markProfileAsCustom();
    setSubmitterConfigs(newConfigs);
    onConfigChange({ ...localConfig, submitter_configs: newConfigs.slice(0, numSubmitters) });

    if (!modelId || !['openrouter', 'openai_codex_oauth'].includes(newConfigs[index].provider)) {
      return;
    }

    const autoSettings = newConfigs[index].provider === 'openrouter'
      ? await getAutoSettingsForModel(modelId, null)
      : getCodexAutoSettingsForModel(modelId);
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
      const finalValue = isNaN(parsed) ? '' : parsed;
      
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
        openrouterReasoningEffort: normalizeOpenRouterReasoningEffort(main.openrouterReasoningEffort),
        lmStudioFallbackId: main.lmStudioFallbackId,
        contextWindow: main.contextWindow,
        maxOutputTokens: main.maxOutputTokens,
        superchargeEnabled: Boolean(main.superchargeEnabled)
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
    const timeout = Number.isFinite(parsedTimeout) ? parsedTimeout : 600;
    const parsedLspIdleTimeout = parseInt(proofSettingsLspIdleTimeout, 10);
    const lspIdleTimeout = Number.isFinite(parsedLspIdleTimeout) ? parsedLspIdleTimeout : 600;
    const parsedMaxParallelCandidates = parseInt(proofSettingsMaxParallelCandidates, 10);
    const maxParallelCandidates = Number.isFinite(parsedMaxParallelCandidates)
      ? Math.max(0, parsedMaxParallelCandidates)
      : 6;
    const parsedSmtTimeout = parseInt(proofSettingsSmtTimeout, 10);
    const smtTimeout = Number.isFinite(parsedSmtTimeout) ? parsedSmtTimeout : 30;

    try {
      setSavingProofSettings(true);
      setProofSettingsMessage('');
      const latestProofStatus = await autonomousAPI.getProofStatus().catch(() => null);
      const leanEnabled = latestProofStatus
        ? Boolean(latestProofStatus.lean4_enabled)
        : proofSettingsEnabled;
      const status = await autonomousAPI.updateProofSettings({
        enabled: leanEnabled,
        timeout,
        lean4_lsp_enabled: proofSettingsLspEnabled,
        lean4_lsp_idle_timeout: lspIdleTimeout,
        max_parallel_candidates: maxParallelCandidates,
        smt_enabled: proofSettingsSmtEnabled,
        smt_timeout: smtTimeout,
      });
      setProofStatus(status);
      setProofSettingsEnabled(Boolean(status.lean4_enabled));
      setProofSettingsTimeout(String(status.lean4_proof_timeout ?? timeout));
      setProofSettingsLspEnabled(Boolean(status.lean4_lsp_enabled));
      setProofSettingsLspIdleTimeout(String(status.lean4_lsp_idle_timeout ?? lspIdleTimeout));
      setProofSettingsMaxParallelCandidates(String(status.proof_max_parallel_candidates ?? maxParallelCandidates));
      setProofSettingsSmtEnabled(Boolean(status.smt_enabled));
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
        openrouterReasoningEffort: normalizeOpenRouterReasoningEffort(cfg.openrouterReasoningEffort),
        lmStudioFallbackId: cfg.lmStudioFallbackId,
        contextWindow: cfg.contextWindow,
        maxOutputTokens: cfg.maxOutputTokens,
        superchargeEnabled: Boolean(cfg.superchargeEnabled)
      })),
      validator: {
        modelId: localConfig.validator_model,
        provider: localConfig.validator_provider,
        openrouterProvider: localConfig.validator_openrouter_provider,
        openrouterReasoningEffort: normalizeOpenRouterReasoningEffort(localConfig.validator_openrouter_reasoning_effort),
        lmStudioFallbackId: localConfig.validator_lm_studio_fallback,
        contextWindow: localConfig.validator_context_window,
        maxOutputTokens: localConfig.validator_max_tokens,
        superchargeEnabled: Boolean(localConfig.validator_supercharge_enabled)
      },
      highContext: {
        modelId: localConfig.high_context_model,
        provider: localConfig.high_context_provider,
        openrouterProvider: localConfig.high_context_openrouter_provider,
        openrouterReasoningEffort: normalizeOpenRouterReasoningEffort(localConfig.high_context_openrouter_reasoning_effort),
        lmStudioFallbackId: localConfig.high_context_lm_studio_fallback,
        contextWindow: localConfig.high_context_context_window,
        maxOutputTokens: localConfig.high_context_max_tokens,
        superchargeEnabled: Boolean(localConfig.high_context_supercharge_enabled)
      },
      highParam: {
        modelId: localConfig.high_param_model,
        provider: localConfig.high_param_provider,
        openrouterProvider: localConfig.high_param_openrouter_provider,
        openrouterReasoningEffort: normalizeOpenRouterReasoningEffort(localConfig.high_param_openrouter_reasoning_effort),
        lmStudioFallbackId: localConfig.high_param_lm_studio_fallback,
        contextWindow: localConfig.high_param_context_window,
        maxOutputTokens: localConfig.high_param_max_tokens,
        superchargeEnabled: Boolean(localConfig.high_param_supercharge_enabled)
      },
      critique: {
        modelId: localConfig.critique_submitter_model,
        provider: localConfig.critique_submitter_provider,
        openrouterProvider: localConfig.critique_submitter_openrouter_provider,
        openrouterReasoningEffort: normalizeOpenRouterReasoningEffort(localConfig.critique_submitter_openrouter_reasoning_effort),
        lmStudioFallbackId: localConfig.critique_submitter_lm_studio_fallback,
        contextWindow: localConfig.critique_submitter_context_window,
        maxOutputTokens: localConfig.critique_submitter_max_tokens,
        superchargeEnabled: Boolean(localConfig.critique_submitter_supercharge_enabled)
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

  const getAutonomousRawSettings = () => ({
    numSubmitters,
    submitterConfigs: submitterConfigs.slice(0, numSubmitters),
    localConfig,
    freeOnly,
    freeModelLooping,
    freeModelAutoSelector,
    allowMathematicalProofs: config?.allow_mathematical_proofs ?? true,
    allowResearchPapers: config?.allow_research_papers ?? true,
    tier3Enabled,
    creativityEmphasisBoostEnabled: config?.creativity_emphasis_boost_enabled ?? false,
    modelProviders,
    selectedProfile,
  });

  const applyAutonomousRawSettings = (rawSettings, { updateRawText = true } = {}) => {
    const nextSettings = persistAutonomousSettings({
      numSubmitters: rawSettings.numSubmitters,
      submitterConfigs: rawSettings.submitterConfigs,
      localConfig: rawSettings.localConfig,
      freeOnly: rawSettings.freeOnly,
      freeModelLooping: rawSettings.freeModelLooping,
      freeModelAutoSelector: rawSettings.freeModelAutoSelector,
      allowMathematicalProofs: rawSettings.allowMathematicalProofs,
      allowResearchPapers: rawSettings.allowResearchPapers,
      tier3Enabled: rawSettings.tier3Enabled,
      creativityEmphasisBoostEnabled: rawSettings.creativityEmphasisBoostEnabled,
      modelProviders: rawSettings.modelProviders,
      selectedProfile: rawSettings.selectedProfile,
    });

    setNumSubmitters(nextSettings.numSubmitters);
    setSubmitterConfigs(nextSettings.submitterConfigs);
    setLocalConfig(nextSettings.localConfig);
    setFreeOnly(nextSettings.freeOnly);
    setFreeModelLooping(nextSettings.freeModelLooping);
    setFreeModelAutoSelector(nextSettings.freeModelAutoSelector);
    setTier3Enabled(nextSettings.tier3Enabled);
    setModelProviders(nextSettings.modelProviders || {});
    setSelectedProfile(nextSettings.selectedProfile || '');
    openRouterAPI
      .setFreeModelSettings(nextSettings.freeModelLooping, nextSettings.freeModelAutoSelector)
      .catch(() => {});
    onConfigChange(settingsToAutonomousConfig(nextSettings));

    if (updateRawText) {
      setRawSettingsText(formatRawSettings(nextSettings));
    }
  };

  const handleRawEditToggle = (checked) => {
    if (checked) {
      const currentSettings = getAutonomousRawSettings();
      setGuiSettingsBeforeRaw(currentSettings);
      setRawSettingsText(formatRawSettings(currentSettings));
      setRawSettingsMessage('');
      setEditRawSettings(true);
      return;
    }

    if (!confirm(RAW_VIEW_EXIT_WARNING)) {
      return;
    }

    if (guiSettingsBeforeRaw) {
      applyAutonomousRawSettings(guiSettingsBeforeRaw, { updateRawText: false });
    }
    setRawSettingsMessage('');
    setEditRawSettings(false);
  };

  const saveRawSettings = () => {
    try {
      const parsed = JSON.parse(rawSettingsText);
      applyAutonomousRawSettings(parsed);
      setRawSettingsMessage('Saved raw settings.');
    } catch (error) {
      setRawSettingsMessage(`Invalid JSON: ${error.message}`);
    }
  };

  return (
    <div className="autonomous-settings-layout">
      <HighlightedModelsSidebar />

      {/* Main Content Area */}
      <div className="autonomous-settings">
      {/* Profile Selection Section */}
      <div className="settings-group" style={{ marginBottom: '1.5rem' }}>
        <h4>Profile Selection</h4>
        <p className="settings-info">
          Load one of the preselected example profiles as a starting point, or create your own custom profile. Expect MOTO to run for at least 3 or more hours before seeing the first completed stage 2 paper. MOTO does a lot of research seeking novel discoveries before writing.
        </p>
        
        <div className="settings-row">
          <label>Select Profile</label>
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
      {openAICodexModelError && (
        <div className="test-result-banner test-result-banner--error" style={{ marginBottom: '1rem' }}>
          {openAICodexModelError}
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
        {developerModeEnabled ? (
          <label
            className="settings-checkbox-label model-refresh-controls__toggle"
            style={{ cursor: isRunning ? 'not-allowed' : 'pointer' }}
          >
            <input
              type="checkbox"
              checked={editRawSettings}
              onChange={(e) => handleRawEditToggle(e.target.checked)}
              disabled={isRunning}
            />
            Edit Raw
          </label>
        ) : (
          <span className="settings-developer-mode-hint">
            Developer mode: press Shift + Z + X to toggle raw JSON settings.
          </span>
        )}
      </div>

      {editRawSettings ? (
        <RawSettingsEditor
          value={rawSettingsText}
          onChange={setRawSettingsText}
          onSave={saveRawSettings}
          message={rawSettingsMessage}
          disabled={isRunning}
        />
      ) : (
        <>
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
            className={`submitter-config-section${effectiveProvider === 'openrouter' ? ' role-config-card--openrouter-orange' : ''}`}
          >
            <h5 className={effectiveProvider === 'openrouter' ? 'card-title--orange' : ''}>
              <span className="role-title-with-badges">
                <span>{idx === 0 ? 'Submitter 1 (Main Submitter)' : `Submitter ${idx + 1}`}</span>
                {idx === 0 && <ProofStrengthBadge />}
              </span>
              {effectiveProvider === 'openrouter' && <span className="provider-badge-inline">[OpenRouter]</span>}
            </h5>
            
            <ModelSelector
              provider={cfg.provider}
              modelId={cfg.modelId}
              openrouterProv={cfg.openrouterProvider}
              openrouterReasoningEffort={cfg.openrouterReasoningEffort}
              fallback={cfg.lmStudioFallbackId}
              onProviderChange={(p) => handleSubmitterConfigChange(idx, 'provider', p)}
              onModelChange={(m) => handleSubmitterModelChange(idx, m)}
              onOpenrouterProviderChange={(p) => handleSubmitterOpenRouterProviderChange(idx, p)}
              onOpenrouterReasoningEffortChange={(effort) => handleSubmitterConfigChange(idx, 'openrouterReasoningEffort', normalizeOpenRouterReasoningEffort(effort))}
              onFallbackChange={(f) => handleSubmitterConfigChange(idx, 'lmStudioFallbackId', f)}
              lmStudioModels={lmStudioModels}
              openRouterModels={openRouterModels}
              openAICodexModels={openAICodexModels}
              modelProviders={modelProviders}
              hasOpenRouterKey={hasOpenRouterKey}
              hasOpenAICodexLogin={hasOpenAICodexLogin}
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

            {developerModeEnabled && (
              <div className="settings-row settings-row--inline-checkbox">
                <label className="settings-checkbox-label settings-checkbox-label--supercharge">
                  <input
                    type="checkbox"
                    checked={Boolean(cfg.superchargeEnabled)}
                    onChange={(e) => handleSubmitterConfigChange(idx, 'superchargeEnabled', e.target.checked)}
                    disabled={isRunning}
                  />
                  <HelpTooltip
                    label="Learn about Supercharge"
                    buttonContent="Supercharge"
                    buttonClassName="help-tooltip-btn--text"
                    popupClassName="help-tooltip-popup--fixed"
                    useFixedPosition
                  >
                    {SUPERCHARGE_TOOLTIP}
                  </HelpTooltip>
                </label>
              </div>
            )}
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
          localConfig={localConfig}
          handleProviderChange={handleProviderChange}
          handleModelChange={handleModelChange}
          handleOpenRouterProviderChange={handleOpenRouterProviderChange}
          handleChange={handleChange}
          handleNumericBlur={handleNumericBlur}
          isRunning={isRunning}
          lmStudioModels={lmStudioModels}
          openRouterModels={openRouterModels}
          openAICodexModels={openAICodexModels}
          modelProviders={modelProviders}
          hasOpenRouterKey={hasOpenRouterKey}
          hasOpenAICodexLogin={hasOpenAICodexLogin}
          lmStudioEnabled={lmStudioEnabled}
          developerModeEnabled={developerModeEnabled}
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
          localConfig={localConfig}
          handleProviderChange={handleProviderChange}
          handleModelChange={handleModelChange}
          handleOpenRouterProviderChange={handleOpenRouterProviderChange}
          handleChange={handleChange}
          handleNumericBlur={handleNumericBlur}
          isRunning={isRunning}
          lmStudioModels={lmStudioModels}
          openRouterModels={openRouterModels}
          openAICodexModels={openAICodexModels}
          modelProviders={modelProviders}
          hasOpenRouterKey={hasOpenRouterKey}
          hasOpenAICodexLogin={hasOpenAICodexLogin}
          lmStudioEnabled={lmStudioEnabled}
          developerModeEnabled={developerModeEnabled}
          showProofStrengthBadge
        />

        <RoleConfig
          title="High-Parameter Submitter"
          hint="Handles mathematical rigor enhancement."
          rolePrefix="high_param"
          localConfig={localConfig}
          handleProviderChange={handleProviderChange}
          handleModelChange={handleModelChange}
          handleOpenRouterProviderChange={handleOpenRouterProviderChange}
          handleChange={handleChange}
          handleNumericBlur={handleNumericBlur}
          isRunning={isRunning}
          lmStudioModels={lmStudioModels}
          openRouterModels={openRouterModels}
          openAICodexModels={openAICodexModels}
          modelProviders={modelProviders}
          hasOpenRouterKey={hasOpenRouterKey}
          hasOpenAICodexLogin={hasOpenAICodexLogin}
          lmStudioEnabled={lmStudioEnabled}
          developerModeEnabled={developerModeEnabled}
          showProofStrengthBadge
        />

        <RoleConfig
          title="Critique Submitter"
          hint="Handles post-body peer review feedback for the AI self-review section."
          rolePrefix="critique_submitter"
          localConfig={localConfig}
          handleProviderChange={handleProviderChange}
          handleModelChange={handleModelChange}
          handleOpenRouterProviderChange={handleOpenRouterProviderChange}
          handleChange={handleChange}
          handleNumericBlur={handleNumericBlur}
          isRunning={isRunning}
          lmStudioModels={lmStudioModels}
          openRouterModels={openRouterModels}
          openAICodexModels={openAICodexModels}
          modelProviders={modelProviders}
          hasOpenRouterKey={hasOpenRouterKey}
          hasOpenAICodexLogin={hasOpenAICodexLogin}
          lmStudioEnabled={lmStudioEnabled}
          developerModeEnabled={developerModeEnabled}
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
                    <label>Concurrent Proof Attempts</label>
                    <div>
                      <input
                        type="number"
                        value={proofSettingsMaxParallelCandidates}
                        onChange={(e) => setProofSettingsMaxParallelCandidates(e.target.value)}
                        disabled={isRunning || savingProofSettings}
                        min={0}
                        max={1000}
                        step={1}
                      />
                      <small className="settings-hint" style={{ display: 'block', marginTop: '0.35rem' }}>
                        Default is 6. Set 0 for unlimited. Positive values run autonomous proof checks in strict batches; rigor mode stays one proof at a time. Setting this number to 0 will make the program faster but more expensive and less efficient.
                      </small>
                    </div>
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
                    <div>
                      <strong>{proofStatus?.z3_path || 'System PATH lookup'}</strong>
                      <small className="settings-hint" style={{ display: 'block', marginTop: '0.35rem' }}>
                        Configure this only through trusted startup environment settings.
                      </small>
                    </div>
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
        </>
      )}
    </div>
    </div>
  );
};

export default AutonomousResearchSettings;
