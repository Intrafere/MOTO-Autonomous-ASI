import React, { useState, useEffect } from 'react';
import { api, cloudAccessAPI, openRouterAPI } from '../../services/api';
import {
  computeCodexAutoSettings,
  computeCloudAccessAutoSettings,
  computeOpenRouterAutoSettings,
  computeSakanaFuguAutoSettings,
  computeXAIGrokAutoSettings,
  DEFAULT_CONTEXT_WINDOW,
  DEFAULT_MAX_OUTPUT_TOKENS,
  DEFAULT_OPENROUTER_REASONING_EFFORT,
  findOpenRouterModel,
  formatOpenRouterProviderLabel,
  getOpenRouterProviderTitle,
  getProviderNames,
  getReasoningSupportInfo,
  hasEndpointMetadata,
  normalizeOpenRouterReasoningEffort,
  OPENROUTER_REASONING_EFFORT_OPTIONS,
  SAKANA_FUGU_REASONING_EFFORT_OPTIONS,
} from '../../utils/openRouterSelection';
import { refreshCredentialProviderState } from '../../utils/credentialProviderRefresh';
import {
  chooseCloudAccessProvider,
  getConfiguredCloudAccessProviders,
  isCloudAccessProvider,
  cloudAccessProviderLabel,
  SAKANA_FUGU_PROVIDER,
  XAI_GROK_PROVIDER,
} from '../../utils/oauthProviders';
import HelpTooltip from '../HelpTooltip';
import HighlightedModelsSidebar from '../HighlightedModelsSidebar';
import OpenRouterFreeModelsControl from '../OpenRouterFreeModelsControl';
import ProofStrengthBadge from '../ProofStrengthBadge';
import RawSettingsEditor from '../RawSettingsEditor';
import '../autonomous/AutonomousResearch.css';
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
const SUPERCHARGE_TOOLTIP = 'Supercharge makes this role generate 4 full answer attempts, then run a 5th same-model call to choose or synthesize the best final answer. It uses 5x the API calls, so it is about 5x slower and 5x more costly, but can produce more intelligent answers.';

const formatRawSettings = (value) => JSON.stringify(value, null, 2);

function AggregatorModelSelector({
  provider,
  modelId,
  openrouterProvider: orProvider,
  openrouterReasoningEffort,
  lmStudioFallbackId,
  onModelChange,
  onProviderChange,
  onOpenrouterProviderChange,
  onOpenrouterReasoningEffortChange,
  onFallbackChange,
  label = 'Model',
  lmStudioEnabled,
  hasOpenRouterKey,
  lmStudioModels,
  openRouterModels,
  oauthModelsByProvider,
  configuredOAuthProviders,
  oauthStatusByProvider,
  modelProviders,
}) {
  const effectiveProvider = lmStudioEnabled ? provider : 'openrouter';
  const models = effectiveProvider === 'openrouter'
    ? openRouterModels
    : (isCloudAccessProvider(effectiveProvider) ? (oauthModelsByProvider[effectiveProvider] || []) : lmStudioModels);
  const providers = modelId && effectiveProvider === 'openrouter'
    ? getProviderNames(modelProviders[modelId])
    : [];
  const reasoningInfo = effectiveProvider === 'openrouter'
    ? getReasoningSupportInfo(modelProviders[modelId], orProvider || null)
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
              onClick={() => onProviderChange('lm_studio')}
              className={`provider-toggle-btn${provider === 'lm_studio' ? ' active-lm' : ''}`}
            >
              LM Studio
            </button>
            <button
              type="button"
              onClick={() => hasOpenRouterKey && onProviderChange('openrouter')}
              disabled={!hasOpenRouterKey}
              className={`provider-toggle-btn${provider === 'openrouter' ? ' active-or-orange' : ''}`}
              title={!hasOpenRouterKey ? 'Set OpenRouter API key first' : 'Use OpenRouter'}
            >
              OpenRouter
            </button>
            <button
              type="button"
              onClick={() => {
                if (configuredOAuthProviders.length > 0) {
                  onProviderChange(chooseCloudAccessProvider(oauthStatusByProvider, provider));
                }
              }}
              disabled={configuredOAuthProviders.length === 0}
              className={`provider-toggle-btn${isCloudAccessProvider(provider) ? ' active-or-orange' : ''}`}
              title={configuredOAuthProviders.length === 0 ? 'Set up a cloud provider login or API key first' : 'Use a configured cloud provider'}
            >
              Cloud
            </button>
            {isCloudAccessProvider(provider) && configuredOAuthProviders.length > 1 && (
              <select
                value={provider}
                onChange={(event) => onProviderChange(event.target.value)}
                title="Select cloud provider"
                className="input-dark"
                style={{ width: 'auto', minWidth: '150px' }}
              >
                {configuredOAuthProviders.map((oauthProvider) => (
                  <option key={oauthProvider.id} value={oauthProvider.id}>
                    {oauthProvider.label}
                  </option>
                ))}
              </select>
            )}
          </div>
        ) : (
          <small className="settings-hint">
            OpenRouter is required in this deployment.
          </small>
        )}
      </div>

      {/* Model Selection */}
      <div className="settings-row">
        <label>{label}</label>
        <select
          value={modelId || ''}
          onChange={(e) => onModelChange(e.target.value)}
        >
          <option value="">Select model...</option>
          {models.map(model => {
            const isFree = effectiveProvider === 'openrouter'
              && model.pricing?.prompt === "0"
              && model.pricing?.completion === "0";
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
      {effectiveProvider === 'openrouter' && modelId && (
        <div className="settings-row">
          <label>Host Provider (optional)</label>
          <select
            className="openrouter-host-provider-select"
            value={orProvider || ''}
            onChange={(e) => onOpenrouterProviderChange(e.target.value || null)}
            title={getOpenRouterProviderTitle(orProvider)}
          >
            <option value="">Auto (let OpenRouter choose)</option>
            {providers.map(p => (
              <option key={p} value={p} title={getOpenRouterProviderTitle(p)}>
                {formatOpenRouterProviderLabel(p)}
              </option>
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

      {effectiveProvider === SAKANA_FUGU_PROVIDER && modelId && (
        <div className="settings-row">
          <label>Reasoning Effort</label>
          <select
            value={normalizeOpenRouterReasoningEffort(openrouterReasoningEffort)}
            onChange={(e) => onOpenrouterReasoningEffortChange(e.target.value)}
          >
            {SAKANA_FUGU_REASONING_EFFORT_OPTIONS.map(option => (
              <option key={option.value} value={option.value}>{option.label}</option>
            ))}
          </select>
          <small className="settings-hint">
            Sakana Fugu supports high and xhigh reasoning effort only; auto maps to xhigh.
          </small>
        </div>
      )}

      {/* LM Studio Fallback (only for cloud providers) */}
      {effectiveProvider !== 'lm_studio' && lmStudioEnabled && (
        <div className="settings-row">
          <label className="label--muted">
            LM Studio Fallback (optional)
          </label>
          <select
            value={lmStudioFallbackId || ''}
            onChange={(e) => onFallbackChange(e.target.value || null)}
          >
            <option value="">No fallback</option>
            {lmStudioModels.map(model => (
              <option key={model.id} value={model.id}>{model.id}</option>
            ))}
          </select>
          <small className="settings-hint" style={{ marginTop: '0.25rem' }}>
            Used if cloud provider access fails or credits run out
          </small>
        </div>
      )}
    </>
  );
}

export default function AggregatorSettings({
  config,
  setConfig,
  capabilities,
  connectivityStatus,
  credentialStatusRefreshToken = 0,
  developerModeEnabled = false,
}) {
  const [lmStudioModels, setLmStudioModels] = useState([]);
  const [openRouterModels, setOpenRouterModels] = useState([]);
  const [openAICodexModels, setOpenAICodexModels] = useState([]);
  const [xaiGrokModels, setXaiGrokModels] = useState([]);
  const [sakanaFuguModels, setSakanaFuguModels] = useState([]);
  const [modelProviders, setModelProviders] = useState({}); // { modelId: { providers: [], endpoints: [] } }
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
  const [validatorMaxOutput, setValidatorMaxOutput] = useState(config.validatorMaxOutput ?? DEFAULT_MAX_OUTPUT_TOKENS);
  
  // Validator OpenRouter state
  const [validatorProvider, setValidatorProvider] = useState(config.validatorProvider || 'lm_studio');
  const [validatorOpenrouterProvider, setValidatorOpenrouterProvider] = useState(config.validatorOpenrouterProvider || null);
  const [validatorOpenrouterReasoningEffort, setValidatorOpenrouterReasoningEffort] = useState(normalizeOpenRouterReasoningEffort(config.validatorOpenrouterReasoningEffort));
  const [validatorLmStudioFallback, setValidatorLmStudioFallback] = useState(config.validatorLmStudioFallback || null);
  const [validatorSuperchargeEnabled, setValidatorSuperchargeEnabled] = useState(Boolean(config.validatorSuperchargeEnabled));
  
  // OpenRouter API key status
  const [hasOpenRouterKey, setHasOpenRouterKey] = useState(false);
  const [hasOpenAICodexLogin, setHasOpenAICodexLogin] = useState(false);
  const [hasXAIGrokLogin, setHasXAIGrokLogin] = useState(false);
  const [hasSakanaFuguKey, setHasSakanaFuguKey] = useState(false);
  const [openAICodexModelError, setOpenAICodexModelError] = useState('');
  const [xaiGrokModelError, setXaiGrokModelError] = useState('');
  const [sakanaFuguModelError, setSakanaFuguModelError] = useState('');
  const [loadingOpenRouter, setLoadingOpenRouter] = useState(false);
  const [freeOnly, setFreeOnly] = useState(false);
  const [freeModelLooping, setFreeModelLooping] = useState(false);
  const [freeModelAutoSelector, setFreeModelAutoSelector] = useState(false);
  const [isLoaded, setIsLoaded] = useState(false);
  const [editRawSettings, setEditRawSettings] = useState(false);
  const [rawSettingsText, setRawSettingsText] = useState('');
  const [rawSettingsMessage, setRawSettingsMessage] = useState('');
  const [guiSettingsBeforeRaw, setGuiSettingsBeforeRaw] = useState(null);
  const lmStudioEnabled = capabilities?.lmStudioEnabled !== false;
  const genericMode = Boolean(capabilities?.genericMode);
  const openAICodexOauthAvailable = !genericMode && capabilities?.openAICodexOauthAvailable !== false;
  const xaiGrokOauthAvailable = !genericMode && capabilities?.xaiGrokOauthAvailable !== false;
  const sakanaFuguAvailable = !genericMode && capabilities?.sakanaFuguAvailable !== false;
  const assistantMemoryEnabled = connectivityStatus?.skills?.agent_conversation_memory?.enabled === true;
  const oauthStatusByProvider = {
    openai_codex_oauth: { configured: hasOpenAICodexLogin },
    [XAI_GROK_PROVIDER]: { configured: hasXAIGrokLogin },
    [SAKANA_FUGU_PROVIDER]: { configured: hasSakanaFuguKey },
  };
  const configuredOAuthProviders = getConfiguredCloudAccessProviders(oauthStatusByProvider);

  useEffect(() => {
    if (!developerModeEnabled && editRawSettings) {
      setEditRawSettings(false);
      setRawSettingsMessage('');
    }
  }, [developerModeEnabled, editRawSettings]);

  // Load settings from localStorage on mount
  useEffect(() => {
    const loadSettings = async () => {
      const savedSettings = localStorage.getItem('aggregator_settings');
      if (savedSettings) {
        try {
          const settings = JSON.parse(savedSettings);
          // Restore all state variables
          if (settings.numSubmitters) setNumSubmitters(settings.numSubmitters);
          if (settings.submitterConfigs) {
            setSubmitterConfigs(settings.submitterConfigs.map((item) => ({
              ...item,
              openrouterReasoningEffort: normalizeOpenRouterReasoningEffort(item.openrouterReasoningEffort),
            })));
          }
          if (settings.validatorProvider) setValidatorProvider(settings.validatorProvider);
          if (settings.validatorOpenrouterProvider) setValidatorOpenrouterProvider(settings.validatorOpenrouterProvider);
          if (settings.validatorOpenrouterReasoningEffort) setValidatorOpenrouterReasoningEffort(normalizeOpenRouterReasoningEffort(settings.validatorOpenrouterReasoningEffort));
          if (settings.validatorLmStudioFallback) setValidatorLmStudioFallback(settings.validatorLmStudioFallback);
          if (settings.validatorSuperchargeEnabled !== undefined) setValidatorSuperchargeEnabled(settings.validatorSuperchargeEnabled);
          if (settings.validatorMaxOutput) setValidatorMaxOutput(settings.validatorMaxOutput);
          if (settings.freeOnly !== undefined) setFreeOnly(settings.freeOnly);
          if (settings.freeModelLooping !== undefined) setFreeModelLooping(settings.freeModelLooping);
          if (settings.freeModelAutoSelector !== undefined) setFreeModelAutoSelector(settings.freeModelAutoSelector);
          if (settings.modelProviders) setModelProviders(settings.modelProviders);
        } catch (error) {
          console.error('Failed to load aggregator settings:', error);
        }
      }
      try {
        const freeModelSettings = await openRouterAPI.getFreeModelSettings();
        setFreeModelLooping(freeModelSettings.looping_enabled ?? false);
        setFreeModelAutoSelector(freeModelSettings.auto_selector_enabled ?? false);
      } catch (error) {
        console.error('Failed to load free model settings:', error);
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
    const assistantProvider = config.assistantProvider || validatorProvider;
    const assistantModel = config.assistantModel || config.validatorModel;
    if (assistantProvider === 'openrouter' && assistantModel) {
      fetchProvidersForModel(assistantModel);
    }
  }, [isLoaded, hasOpenRouterKey, submitterConfigs, validatorProvider, config.validatorModel, config.assistantProvider, config.assistantModel]);

  // Save settings to localStorage whenever values change
  useEffect(() => {
    if (!isLoaded) return;
    
    const settings = {
      userPrompt: config.userPrompt || '',
      numSubmitters,
      submitterConfigs,
      validatorModel: config.validatorModel || '',
      validatorProvider,
      validatorOpenrouterProvider,
      validatorOpenrouterReasoningEffort,
      validatorLmStudioFallback,
      validatorSuperchargeEnabled,
      validatorContextSize: config.validatorContextSize ?? DEFAULT_CONTEXT_WINDOW,
      validatorMaxOutput,
      assistantModel: config.assistantModel || config.validatorModel || '',
      assistantProvider: config.assistantProvider || validatorProvider,
      assistantOpenrouterProvider: config.assistantOpenrouterProvider || null,
      assistantOpenrouterReasoningEffort: normalizeOpenRouterReasoningEffort(config.assistantOpenrouterReasoningEffort || validatorOpenrouterReasoningEffort),
      assistantLmStudioFallback: config.assistantLmStudioFallback || null,
      assistantContextSize: config.assistantContextSize || config.validatorContextSize || DEFAULT_CONTEXT_WINDOW,
      assistantMaxOutput: config.assistantMaxOutput || validatorMaxOutput,
      assistantSuperchargeEnabled: Boolean(config.assistantSuperchargeEnabled),
      freeOnly,
      freeModelLooping,
      freeModelAutoSelector,
      modelProviders,
      creativityEmphasisBoostEnabled: config.creativityEmphasisBoostEnabled,
      uploadedFiles: config.uploadedFiles || [],
    };
    localStorage.setItem('aggregator_settings', JSON.stringify(settings));
  }, [isLoaded, config.userPrompt, config.validatorModel, config.validatorContextSize, config.assistantModel, config.assistantProvider, config.assistantOpenrouterProvider, config.assistantOpenrouterReasoningEffort, config.assistantLmStudioFallback, config.assistantContextSize, config.assistantMaxOutput, config.assistantSuperchargeEnabled, numSubmitters, submitterConfigs, validatorProvider, validatorOpenrouterProvider, validatorOpenrouterReasoningEffort, validatorLmStudioFallback, validatorSuperchargeEnabled, validatorMaxOutput, freeOnly, freeModelLooping, freeModelAutoSelector, modelProviders, config.creativityEmphasisBoostEnabled, config.uploadedFiles]);

  useEffect(() => {
    if (lmStudioEnabled) {
      fetchModels();
    } else {
      setLmStudioModels([]);
      setLoading(false);
    }
    checkOpenRouterKey();
  }, [lmStudioEnabled]);

  useEffect(() => {
    if (credentialStatusRefreshToken === 0) {
      return;
    }

    let isCurrent = true;
    refreshCredentialProviderState({
      freeOnly,
      openAICodexOauthAvailable,
      xaiGrokOauthAvailable,
      sakanaFuguAvailable,
      setHasOpenRouterKey,
      setOpenRouterModels,
      setLoadingOpenRouter,
      setHasOpenAICodexLogin,
      setOpenAICodexModels,
      setOpenAICodexModelError,
      setHasXAIGrokLogin,
      setXaiGrokModels,
      setXaiGrokModelError,
      setHasSakanaFuguKey,
      setSakanaFuguModels,
      setSakanaFuguModelError,
      shouldApply: () => isCurrent,
      logContext: 'Aggregator settings',
    });
    return () => {
      isCurrent = false;
    };
  }, [credentialStatusRefreshToken]);

  useEffect(() => {
    if (lmStudioEnabled) {
      return;
    }

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
    const keepValidatorOpenRouterState = validatorProvider === 'openrouter';

    if (JSON.stringify(normalizedSubmitters) !== JSON.stringify(submitterConfigs)) {
      setSubmitterConfigs(normalizedSubmitters);
    }
    if (validatorProvider !== 'openrouter') {
      setValidatorProvider('openrouter');
    }
    if (validatorOpenrouterProvider !== (keepValidatorOpenRouterState ? (validatorOpenrouterProvider || null) : null)) {
      setValidatorOpenrouterProvider(keepValidatorOpenRouterState ? (validatorOpenrouterProvider || null) : null);
    }
    if (validatorLmStudioFallback !== null) {
      setValidatorLmStudioFallback(null);
    }

    setConfig((prev) => {
      const assistantProvider = prev.assistantProvider || validatorProvider;
      const keepAssistantOpenRouterState = assistantProvider === 'openrouter';
      const hasAssistantModelField = Object.prototype.hasOwnProperty.call(prev, 'assistantModel');
      const next = {
        ...prev,
        submitterConfigs: normalizedSubmitters,
        validatorProvider: 'openrouter',
        validatorModel: keepValidatorOpenRouterState ? (prev.validatorModel || '') : '',
        validatorOpenrouterProvider: keepValidatorOpenRouterState
          ? (prev.validatorOpenrouterProvider || null)
          : null,
        validatorOpenrouterReasoningEffort: normalizeOpenRouterReasoningEffort(prev.validatorOpenrouterReasoningEffort),
        validatorLmStudioFallback: null,
        assistantProvider: 'openrouter',
        assistantModel: keepAssistantOpenRouterState
          ? (hasAssistantModelField ? (prev.assistantModel || '') : (prev.validatorModel || ''))
          : '',
        assistantOpenrouterProvider: keepAssistantOpenRouterState
          ? (prev.assistantOpenrouterProvider || null)
          : null,
        assistantOpenrouterReasoningEffort: normalizeOpenRouterReasoningEffort(prev.assistantOpenrouterReasoningEffort),
        assistantLmStudioFallback: null,
      };
      return JSON.stringify(next) === JSON.stringify(prev) ? prev : next;
    });
  }, [
    lmStudioEnabled,
    submitterConfigs,
    validatorProvider,
    validatorOpenrouterProvider,
    validatorLmStudioFallback,
    config.assistantProvider,
    setConfig,
  ]);

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
    if (openAICodexOauthAvailable) {
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
        console.error('Failed to check OpenAI Codex login status:', err);
        setOpenAICodexModelError(`OpenAI Codex OAuth status could not be checked: ${err.message || 'unknown error'}.`);
      }
    } else {
      setHasOpenAICodexLogin(false);
      setOpenAICodexModels([]);
      setOpenAICodexModelError('');
    }
    if (xaiGrokOauthAvailable) {
      try {
        const xaiStatus = await cloudAccessAPI.getXAIGrokStatus();
        const configured = Boolean(xaiStatus.status?.configured);
        setHasXAIGrokLogin(configured);
        if (configured) {
          fetchXAIGrokModels();
        } else {
          setXaiGrokModelError('');
        }
      } catch (err) {
        console.error('Failed to check xAI Grok login status:', err);
        setXaiGrokModelError(`xAI Grok OAuth status could not be checked: ${err.message || 'unknown error'}.`);
      }
    } else {
      setHasXAIGrokLogin(false);
      setXaiGrokModels([]);
      setXaiGrokModelError('');
    }
    if (sakanaFuguAvailable) {
      try {
        const sakanaStatus = await cloudAccessAPI.getSakanaFuguStatus();
        const configured = Boolean(sakanaStatus.status?.configured);
        setHasSakanaFuguKey(configured);
        if (configured) {
          fetchSakanaFuguModels();
        } else {
          setSakanaFuguModelError('');
        }
      } catch (err) {
        console.error('Failed to check Sakana Fugu API key status:', err);
        setSakanaFuguModelError(`Sakana Fugu status could not be checked: ${err.message || 'unknown error'}.`);
      }
    } else {
      setHasSakanaFuguKey(false);
      setSakanaFuguModels([]);
      setSakanaFuguModelError('');
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

  const fetchOpenAICodexModels = async () => {
    if (!openAICodexOauthAvailable) {
      setOpenAICodexModels([]);
      setHasOpenAICodexLogin(false);
      setOpenAICodexModelError('');
      return;
    }
    try {
      const result = await cloudAccessAPI.getOpenAICodexModels();
      const models = result.models || [];
      setOpenAICodexModels(models);
      setHasOpenAICodexLogin(true);
      setOpenAICodexModelError(models.length > 0
        ? ''
        : 'OpenAI Codex OAuth is connected, but no Codex models were returned. Reconnect OAuth or check account access.'
      );
    } catch (err) {
      console.error('Failed to fetch OpenAI Codex models:', err);
      setOpenAICodexModels([]);
      setHasOpenAICodexLogin(true);
      setOpenAICodexModelError(`OpenAI Codex OAuth is connected, but models could not be loaded: ${err.message || 'unknown error'}.`);
    }
  };

  const fetchXAIGrokModels = async () => {
    if (!xaiGrokOauthAvailable) {
      setXaiGrokModels([]);
      setHasXAIGrokLogin(false);
      setXaiGrokModelError('');
      return;
    }
    try {
      const result = await cloudAccessAPI.getXAIGrokModels();
      const models = result.models || [];
      setXaiGrokModels(models);
      setHasXAIGrokLogin(true);
      setXaiGrokModelError(models.length > 0
        ? ''
        : 'xAI Grok OAuth is connected, but no Grok models were returned. Reconnect OAuth or check account access.'
      );
    } catch (err) {
      console.error('Failed to fetch xAI Grok models:', err);
      setXaiGrokModels([]);
      setHasXAIGrokLogin(true);
      setXaiGrokModelError(`xAI Grok OAuth is connected, but models could not be loaded: ${err.message || 'unknown error'}.`);
    }
  };

  const fetchSakanaFuguModels = async () => {
    if (!sakanaFuguAvailable) {
      setSakanaFuguModels([]);
      setHasSakanaFuguKey(false);
      setSakanaFuguModelError('');
      return;
    }
    try {
      const result = await cloudAccessAPI.getSakanaFuguModels();
      const models = result.models || [];
      setSakanaFuguModels(models);
      setHasSakanaFuguKey(true);
      setSakanaFuguModelError(models.length > 0
        ? ''
        : 'Sakana Fugu API key is saved, but no Fugu models were returned. Check your Sakana subscription access.'
      );
    } catch (err) {
      console.error('Failed to fetch Sakana Fugu models:', err);
      setSakanaFuguModels([]);
      setHasSakanaFuguKey(true);
      setSakanaFuguModelError(`Sakana Fugu API key is saved, but models could not be loaded: ${err.message || 'unknown error'}.`);
    }
  };

  // Refetch models when free-only toggle changes
  useEffect(() => {
    if (hasOpenRouterKey && isLoaded) {
      fetchOpenRouterModels(freeOnly);
    }
  }, [freeOnly]);

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
      console.debug('[AggregatorAutoFill] model not in loaded list, skipping auto-fill', { modelId });
      return null;
    }

    const providerData = await fetchProvidersForModel(modelId);
    const autoSettings = computeOpenRouterAutoSettings(model, providerData, selectedProvider);
    if (autoSettings) {
      console.debug('[AggregatorAutoFill] computed auto-settings', {
        modelId,
        selectedProvider,
        source: autoSettings.source,
        contextWindow: autoSettings.contextWindow,
        maxOutputTokens: autoSettings.maxOutputTokens,
        warnings: autoSettings.warnings,
      });
      if (autoSettings.warnings && autoSettings.warnings.length > 0) {
        console.warn('[AggregatorAutoFill] auto-settings fallback used:', autoSettings.warnings);
      }
    }
    return autoSettings;
  };

  const getCodexAutoSettingsForModel = (modelId) => {
    const model = openAICodexModels.find((item) => item.id === modelId);
    if (!model) {
      console.debug('[AggregatorCodexAutoFill] model not in loaded list, skipping auto-fill', { modelId });
      return null;
    }
    const autoSettings = computeCodexAutoSettings(model);
    if (autoSettings.warnings.length > 0) {
      console.warn('[AggregatorCodexAutoFill] auto-settings fallback used:', autoSettings.warnings);
    }
    return autoSettings;
  };

  const getCloudAccessAutoSettingsForModel = (provider, modelId) => {
    if (provider === 'openai_codex_oauth') {
      return getCodexAutoSettingsForModel(modelId);
    }
    const oauthModelsByProvider = {
      openai_codex_oauth: openAICodexModels,
      [XAI_GROK_PROVIDER]: xaiGrokModels,
      [SAKANA_FUGU_PROVIDER]: sakanaFuguModels,
    };
    const model = (oauthModelsByProvider[provider] || []).find((item) => item.id === modelId);
    if (!model) {
      console.debug('[AggregatorOAuthAutoFill] model not in loaded list, skipping auto-fill', { provider, modelId });
      return null;
    }
    const autoSettings = provider === XAI_GROK_PROVIDER
      ? computeXAIGrokAutoSettings(model)
      : (provider === SAKANA_FUGU_PROVIDER
        ? computeSakanaFuguAutoSettings(model)
        : computeCloudAccessAutoSettings(model, cloudAccessProviderLabel(provider)));
    if (autoSettings.warnings.length > 0) {
      console.warn('[AggregatorOAuthAutoFill] auto-settings fallback used:', autoSettings.warnings);
    }
    return autoSettings;
  };

  const handleSubmitterModelChange = async (submitterId, modelId) => {
    const baseConfigs = submitterConfigs.map(c =>
      c.submitterId === submitterId
        ? { ...c, modelId, openrouterProvider: null, openrouterReasoningEffort: DEFAULT_OPENROUTER_REASONING_EFFORT }
        : c
    );
    setSubmitterConfigs(baseConfigs);
    setConfig(prev => ({ ...prev, submitterConfigs: baseConfigs }));

    const targetConfig = baseConfigs.find(c => c.submitterId === submitterId);
    if (!modelId || !(targetConfig?.provider === 'openrouter' || isCloudAccessProvider(targetConfig?.provider))) {
      return;
    }

    const autoSettings = targetConfig.provider === 'openrouter'
      ? await getAutoSettingsForModel(modelId, null)
      : getCloudAccessAutoSettingsForModel(targetConfig.provider, modelId);
    if (!autoSettings) {
      return;
    }

    const nextConfigs = baseConfigs.map(c =>
      c.submitterId === submitterId
        ? {
            ...c,
            ...(autoSettings.contextWindowKnown ? { contextWindow: autoSettings.contextWindow } : {}),
            ...(autoSettings.outputCapKnown ? { maxOutputTokens: autoSettings.maxOutputTokens } : {}),
          }
        : c
    );
    setSubmitterConfigs(nextConfigs);
    setConfig(prev => ({ ...prev, submitterConfigs: nextConfigs }));
  };

  const handleSubmitterOpenRouterProviderChange = async (submitterId, providerName) => {
    const baseConfigs = submitterConfigs.map(c =>
      c.submitterId === submitterId
        ? { ...c, openrouterProvider: providerName }
        : c
    );
    setSubmitterConfigs(baseConfigs);
    setConfig(prev => ({ ...prev, submitterConfigs: baseConfigs }));

    const targetConfig = baseConfigs.find(c => c.submitterId === submitterId);
    if (!targetConfig?.modelId) {
      return;
    }

    const autoSettings = await getAutoSettingsForModel(targetConfig.modelId, providerName);
    if (!autoSettings) {
      return;
    }

    const nextConfigs = baseConfigs.map(c =>
      c.submitterId === submitterId
        ? {
            ...c,
            ...(autoSettings.contextWindowKnown ? { contextWindow: autoSettings.contextWindow } : {}),
            ...(autoSettings.outputCapKnown ? { maxOutputTokens: autoSettings.maxOutputTokens } : {}),
          }
        : c
    );
    setSubmitterConfigs(nextConfigs);
    setConfig(prev => ({ ...prev, submitterConfigs: nextConfigs }));
  };

  const handleValidatorModelChange = async (modelId) => {
    setConfig(prev => ({
      ...prev,
      validatorModel: modelId,
      validatorOpenrouterProvider: null,
      validatorOpenrouterReasoningEffort: DEFAULT_OPENROUTER_REASONING_EFFORT,
    }));
    setValidatorOpenrouterProvider(null);
    setValidatorOpenrouterReasoningEffort(DEFAULT_OPENROUTER_REASONING_EFFORT);

    if (!modelId || !(validatorProvider === 'openrouter' || isCloudAccessProvider(validatorProvider))) {
      return;
    }

    const autoSettings = validatorProvider === 'openrouter'
      ? await getAutoSettingsForModel(modelId, null)
      : getCloudAccessAutoSettingsForModel(validatorProvider, modelId);
    if (!autoSettings) {
      return;
    }

    if (autoSettings.outputCapKnown) {
      setValidatorMaxOutput(autoSettings.maxOutputTokens);
    }
    setConfig(prev => ({
      ...prev,
      validatorModel: modelId,
      validatorOpenrouterProvider: null,
      validatorOpenrouterReasoningEffort: DEFAULT_OPENROUTER_REASONING_EFFORT,
      ...(autoSettings.contextWindowKnown ? { validatorContextSize: autoSettings.contextWindow } : {}),
      ...(autoSettings.outputCapKnown ? { validatorMaxOutput: autoSettings.maxOutputTokens } : {}),
    }));
  };

  const handleValidatorOpenRouterProviderChange = async (providerName) => {
    setValidatorOpenrouterProvider(providerName);
    setConfig(prev => ({
      ...prev,
      validatorOpenrouterProvider: providerName,
    }));

    if (!config.validatorModel) {
      return;
    }

    const autoSettings = await getAutoSettingsForModel(config.validatorModel, providerName);
    if (!autoSettings) {
      return;
    }

    if (autoSettings.outputCapKnown) {
      setValidatorMaxOutput(autoSettings.maxOutputTokens);
    }
    setConfig(prev => ({
      ...prev,
      ...(autoSettings.contextWindowKnown ? { validatorContextSize: autoSettings.contextWindow } : {}),
      ...(autoSettings.outputCapKnown ? { validatorMaxOutput: autoSettings.maxOutputTokens } : {}),
    }));
  };

  // Handle number of submitters change - expand/contract configs
  const handleNumSubmittersChange = (newNum) => {
    const parsed = parseInt(newNum, 10);
    const num = Number.isFinite(parsed) ? Math.min(10, Math.max(1, parsed)) : 1;
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
          openrouterReasoningEffort: normalizeOpenRouterReasoningEffort(template.openrouterReasoningEffort),
          lmStudioFallbackId: template.lmStudioFallbackId,
          contextWindow: template.contextWindow,
          maxOutputTokens: template.maxOutputTokens,
          superchargeEnabled: Boolean(template.superchargeEnabled)
        });
      }
    }
    setSubmitterConfigs(newConfigs);
    setConfig(prev => ({ ...prev, submitterConfigs: newConfigs }));
  };

  const fetchModels = async () => {
    if (!lmStudioEnabled) {
      setLmStudioModels([]);
      setLoading(false);
      return;
    }

    try {
      const data = await api.getModels();
      const nextModels = data.models || data || [];
      setLmStudioModels(nextModels);
      
      // Auto-select first model if none selected (only for LM Studio provider)
      if (nextModels.length > 0) {
        const firstModelId = nextModels[0].id;
        
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
      safeValue = DEFAULT_CONTEXT_WINDOW;
    } else if (field === 'maxOutputTokens' && isNaN(value)) {
      safeValue = DEFAULT_MAX_OUTPUT_TOKENS;
    }
    
    const newConfigs = submitterConfigs.map(c => {
      if (c.submitterId !== submitterId) return c;
      
      const updated = { ...c, [field]: safeValue };
      
      // If switching provider, reset model-specific fields
      if (field === 'provider') {
        updated.modelId = '';
        updated.openrouterProvider = null;
        updated.openrouterReasoningEffort = DEFAULT_OPENROUTER_REASONING_EFFORT;
        updated.lmStudioFallbackId = null;
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
      openrouterReasoningEffort: normalizeOpenRouterReasoningEffort(source.openrouterReasoningEffort),
      lmStudioFallbackId: source.lmStudioFallbackId,
      contextWindow: source.contextWindow,
      maxOutputTokens: source.maxOutputTokens,
      superchargeEnabled: Boolean(source.superchargeEnabled)
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
      setValidatorOpenrouterReasoningEffort(DEFAULT_OPENROUTER_REASONING_EFFORT);
      setValidatorLmStudioFallback(null);
    }
    setConfig(prev => ({
      ...prev,
      validatorProvider: provider,
      validatorModel: '',
      validatorOpenrouterProvider: null,
      validatorOpenrouterReasoningEffort: DEFAULT_OPENROUTER_REASONING_EFFORT,
      validatorLmStudioFallback: null
    }));
  };

  const getAggregatorRawSettings = () => ({
    userPrompt: config.userPrompt || '',
    numSubmitters,
    submitterConfigs,
    validatorModel: config.validatorModel || '',
    validatorProvider,
    validatorOpenrouterProvider,
    validatorOpenrouterReasoningEffort,
    validatorLmStudioFallback,
    validatorSuperchargeEnabled,
    validatorContextSize: config.validatorContextSize ?? DEFAULT_CONTEXT_WINDOW,
    validatorMaxOutput,
    assistantModel: config.assistantModel || config.validatorModel || '',
    assistantProvider: config.assistantProvider || validatorProvider,
    assistantOpenrouterProvider: config.assistantOpenrouterProvider || null,
    assistantOpenrouterReasoningEffort: normalizeOpenRouterReasoningEffort(config.assistantOpenrouterReasoningEffort || validatorOpenrouterReasoningEffort),
    assistantLmStudioFallback: config.assistantLmStudioFallback || null,
    assistantContextSize: config.assistantContextSize || config.validatorContextSize || DEFAULT_CONTEXT_WINDOW,
    assistantMaxOutput: config.assistantMaxOutput || validatorMaxOutput,
    assistantSuperchargeEnabled: Boolean(config.assistantSuperchargeEnabled),
    freeOnly,
    freeModelLooping,
    freeModelAutoSelector,
    modelProviders,
  });

  const applyAggregatorRawSettings = (rawSettings, { updateRawText = true } = {}) => {
    const nextSubmitters = Array.isArray(rawSettings.submitterConfigs) && rawSettings.submitterConfigs.length > 0
      ? rawSettings.submitterConfigs.map((item) => ({
          ...item,
          openrouterReasoningEffort: normalizeOpenRouterReasoningEffort(item.openrouterReasoningEffort),
        }))
      : submitterConfigs;
    const nextNumSubmitters = Number(rawSettings.numSubmitters || nextSubmitters.length || 3);
    const nextValidatorProvider = rawSettings.validatorProvider || 'lm_studio';
    const nextValidatorOpenrouterProvider = rawSettings.validatorOpenrouterProvider || null;
    const nextValidatorOpenrouterReasoningEffort = normalizeOpenRouterReasoningEffort(rawSettings.validatorOpenrouterReasoningEffort);
    const nextValidatorLmStudioFallback = rawSettings.validatorLmStudioFallback || null;
    const nextValidatorSuperchargeEnabled = Boolean(rawSettings.validatorSuperchargeEnabled);
    const nextValidatorContextSize = rawSettings.validatorContextSize ?? DEFAULT_CONTEXT_WINDOW;
    const nextValidatorMaxOutput = rawSettings.validatorMaxOutput ?? DEFAULT_MAX_OUTPUT_TOKENS;
    const nextModelProviders = rawSettings.modelProviders || {};

    setNumSubmitters(nextNumSubmitters);
    setSubmitterConfigs(nextSubmitters);
    setValidatorProvider(nextValidatorProvider);
    setValidatorOpenrouterProvider(nextValidatorOpenrouterProvider);
    setValidatorOpenrouterReasoningEffort(nextValidatorOpenrouterReasoningEffort);
    setValidatorLmStudioFallback(nextValidatorLmStudioFallback);
    setValidatorSuperchargeEnabled(nextValidatorSuperchargeEnabled);
    setValidatorMaxOutput(nextValidatorMaxOutput);
    setFreeOnly(rawSettings.freeOnly ?? false);
    setFreeModelLooping(rawSettings.freeModelLooping ?? false);
    setFreeModelAutoSelector(rawSettings.freeModelAutoSelector ?? false);
    setModelProviders(nextModelProviders);
    openRouterAPI
      .setFreeModelSettings(rawSettings.freeModelLooping ?? false, rawSettings.freeModelAutoSelector ?? false)
      .catch(() => {});

    const hasRawUserPrompt = Object.prototype.hasOwnProperty.call(rawSettings, 'userPrompt');
    const nextConfig = {
      ...config,
      userPrompt: hasRawUserPrompt ? (rawSettings.userPrompt || '') : (config.userPrompt || ''),
      submitterConfigs: nextSubmitters,
      validatorModel: rawSettings.validatorModel || '',
      validatorProvider: nextValidatorProvider,
      validatorOpenrouterProvider: nextValidatorOpenrouterProvider,
      validatorOpenrouterReasoningEffort: nextValidatorOpenrouterReasoningEffort,
      validatorLmStudioFallback: nextValidatorLmStudioFallback,
      validatorSuperchargeEnabled: nextValidatorSuperchargeEnabled,
      validatorContextSize: nextValidatorContextSize,
      validatorMaxOutput: nextValidatorMaxOutput,
      assistantModel: rawSettings.assistantModel || rawSettings.validatorModel || '',
      assistantProvider: rawSettings.assistantProvider || nextValidatorProvider,
      assistantOpenrouterProvider: rawSettings.assistantOpenrouterProvider || null,
      assistantOpenrouterReasoningEffort: normalizeOpenRouterReasoningEffort(rawSettings.assistantOpenrouterReasoningEffort || rawSettings.validatorOpenrouterReasoningEffort),
      assistantLmStudioFallback: rawSettings.assistantLmStudioFallback || null,
      assistantContextSize: rawSettings.assistantContextSize || nextValidatorContextSize,
      assistantMaxOutput: rawSettings.assistantMaxOutput || nextValidatorMaxOutput,
      assistantSuperchargeEnabled: Boolean(rawSettings.assistantSuperchargeEnabled),
    };
    setConfig(nextConfig);

    if (updateRawText) {
      setRawSettingsText(formatRawSettings({
        ...rawSettings,
        ...nextConfig,
        numSubmitters: nextNumSubmitters,
        freeOnly: rawSettings.freeOnly ?? false,
        validatorOpenrouterReasoningEffort: nextValidatorOpenrouterReasoningEffort,
        validatorSuperchargeEnabled: nextValidatorSuperchargeEnabled,
        freeModelLooping: rawSettings.freeModelLooping ?? false,
        freeModelAutoSelector: rawSettings.freeModelAutoSelector ?? false,
        modelProviders: nextModelProviders,
      }));
    }
  };

  const handleRawEditToggle = (checked) => {
    if (checked) {
      const currentSettings = getAggregatorRawSettings();
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
      applyAggregatorRawSettings(guiSettingsBeforeRaw, { updateRawText: false });
    }
    setRawSettingsMessage('');
    setEditRawSettings(false);
  };

  const saveRawSettings = () => {
    try {
      const parsed = JSON.parse(rawSettingsText);
      applyAggregatorRawSettings(parsed);
      setRawSettingsMessage('Saved raw settings.');
    } catch (error) {
      setRawSettingsMessage(`Invalid JSON: ${error.message}`);
    }
  };

  return (
    <div className="autonomous-settings-layout">
      <HighlightedModelsSidebar />
      <div className="autonomous-settings">
          <div className="settings-header-row">
            <h2>Aggregator Settings</h2>
            {saveMessage && (
              <div className="save-message">
                {saveMessage}
              </div>
            )}
          </div>
          {openAICodexModelError && (
            <div className="test-result-banner test-result-banner--error" style={{ marginBottom: '1rem' }}>
              {openAICodexModelError}
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

      {loading ? (
        <div>Loading models...</div>
      ) : lmStudioEnabled && lmStudioModels.length === 0 && !hasOpenRouterKey ? (
        <div className="error-text">
          <p>No models found. Make sure LM Studio is running on http://127.0.0.1:1234 or configure OpenRouter.</p>
          <button onClick={fetchModels} className="secondary">Retry</button>
        </div>
      ) : !lmStudioEnabled && !hasOpenRouterKey ? (
        <div className="error-text">
          <p>This deployment disables LM Studio. Set an OpenRouter API key in the header to configure models.</p>
        </div>
      ) : (
        <>
          <div className="model-refresh-controls">
            {lmStudioEnabled && (
              <button onClick={fetchModels} className="secondary">
                Refresh LM Studio Models
              </button>
            )}
            {hasOpenRouterKey && (
              <>
                <button onClick={() => fetchOpenRouterModels(freeOnly)} className="secondary" disabled={loadingOpenRouter}>
                  {loadingOpenRouter ? 'Loading...' : 'Refresh OpenRouter Models'}
                </button>
                <OpenRouterFreeModelsControl
                  checked={freeOnly}
                  onChange={setFreeOnly}
                />
              </>
            )}
            {developerModeEnabled && (
              <>
              {hasOpenRouterKey && <span className="model-refresh-controls__divider" aria-hidden="true" />}
              <label className="settings-checkbox-label model-refresh-controls__toggle">
                <input
                  type="checkbox"
                  checked={editRawSettings}
                  onChange={(e) => handleRawEditToggle(e.target.checked)}
                />
                Edit Raw
              </label>
              </>
            )}
          </div>

          {editRawSettings ? (
            <RawSettingsEditor
              value={rawSettingsText}
              onChange={setRawSettingsText}
              onSave={saveRawSettings}
              message={rawSettingsMessage}
            />
          ) : (
            <>
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
                min={1}
                max={10}
                step={1}
                value={numSubmitters}
                onChange={(e) => handleNumSubmittersChange(e.target.value)}
              />
              {numSubmitters > 1 && (
                <button
                  type="button"
                  className="copy-btn"
                  onClick={() => applyToAll(1)}
                  title="Copy Main Submitter settings to all others"
                >
                  Copy Main to All
                </button>
              )}
            </div>
            
            {submitterConfigs.map((cfg, idx) => (
              (() => {
                const effectiveProvider = lmStudioEnabled ? cfg.provider : 'openrouter';
                return (
              <div 
                key={cfg.submitterId}
                className={`submitter-config-section${effectiveProvider === 'openrouter' ? ' role-config-card--openrouter-orange' : ''}`}
              >
                <h5 className={effectiveProvider === 'openrouter' ? 'card-title--orange' : ''}>
                  <span className="role-title-with-badges">
                    <span>{idx === 0 ? 'Submitter 1 (Main Submitter)' : `Submitter ${idx + 1}`}</span>
                    {idx === 0 && <ProofStrengthBadge />}
                  </span>
                  {effectiveProvider === 'openrouter' && <span className="provider-badge-inline">[OpenRouter]</span>}
                </h5>

                  <AggregatorModelSelector
                    provider={cfg.provider}
                    modelId={cfg.modelId}
                    openrouterProvider={cfg.openrouterProvider}
                    openrouterReasoningEffort={cfg.openrouterReasoningEffort}
                    lmStudioFallbackId={cfg.lmStudioFallbackId}
                    onProviderChange={(p) => updateSubmitterConfig(cfg.submitterId, 'provider', p)}
                    onModelChange={(m) => handleSubmitterModelChange(cfg.submitterId, m)}
                    onOpenrouterProviderChange={(p) => handleSubmitterOpenRouterProviderChange(cfg.submitterId, p)}
                    onOpenrouterReasoningEffortChange={(effort) => updateSubmitterConfig(cfg.submitterId, 'openrouterReasoningEffort', effort)}
                    onFallbackChange={(f) => updateSubmitterConfig(cfg.submitterId, 'lmStudioFallbackId', f)}
                    lmStudioEnabled={lmStudioEnabled}
                    hasOpenRouterKey={hasOpenRouterKey}
                    lmStudioModels={lmStudioModels}
                    openRouterModels={openRouterModels}
                    oauthModelsByProvider={{
                      openai_codex_oauth: openAICodexModels,
                      [XAI_GROK_PROVIDER]: xaiGrokModels,
                      [SAKANA_FUGU_PROVIDER]: sakanaFuguModels,
                    }}
                    configuredOAuthProviders={configuredOAuthProviders}
                    oauthStatusByProvider={oauthStatusByProvider}
                    modelProviders={modelProviders}
                  />

                  <div className="settings-row">
                    <label>Context Window</label>
                    <input
                      type="number"
                      value={cfg.contextWindow}
                      onChange={(e) => updateSubmitterConfig(cfg.submitterId, 'contextWindow', parseInt(e.target.value))}
                      min="4096"
                      max="50000000"
                      step="1024"
                    />
                  </div>

                  <div className="settings-row">
                    <label>Max Output Tokens</label>
                    <input
                      type="number"
                      value={cfg.maxOutputTokens}
                      onChange={(e) => updateSubmitterConfig(cfg.submitterId, 'maxOutputTokens', parseInt(e.target.value))}
                      min="1000"
                      max="50000000"
                      step="1000"
                    />
                  </div>

                  {developerModeEnabled && (
                    <div className="settings-row settings-row--inline-checkbox">
                      <label className="settings-checkbox-label settings-checkbox-label--supercharge">
                        <input
                          type="checkbox"
                          checked={Boolean(cfg.superchargeEnabled)}
                          onChange={(e) => updateSubmitterConfig(cfg.submitterId, 'superchargeEnabled', e.target.checked)}
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

          {/* Validator Configuration (Single) */}
          <div className="settings-group">
            <h4>Validator (Single Instance)</h4>
            <p className="settings-info">
              Only one validator is allowed to maintain a single Markov chain evolution of the database.
            </p>

            <div
              className={`submitter-config-section${validatorProvider === 'openrouter' ? ' role-config-card--openrouter-orange' : ''}`}
              style={{ borderColor: validatorProvider === 'openrouter' ? undefined : '#ff6b6b' }}
            >
              <h5 className={validatorProvider === 'openrouter' ? 'card-title--orange' : ''} style={validatorProvider === 'openrouter' ? undefined : { color: '#ff6b6b' }}>
                Validator
                {validatorProvider === 'openrouter' && <span className="provider-badge-inline">[OpenRouter]</span>}
              </h5>

              <AggregatorModelSelector
                provider={validatorProvider}
                modelId={config.validatorModel}
                openrouterProvider={validatorOpenrouterProvider}
                openrouterReasoningEffort={validatorOpenrouterReasoningEffort}
                lmStudioFallbackId={validatorLmStudioFallback}
                onProviderChange={updateValidatorProvider}
                onModelChange={handleValidatorModelChange}
                onOpenrouterProviderChange={handleValidatorOpenRouterProviderChange}
                onOpenrouterReasoningEffortChange={(effort) => {
                  const normalized = normalizeOpenRouterReasoningEffort(effort);
                  setValidatorOpenrouterReasoningEffort(normalized);
                  setConfig({ ...config, validatorOpenrouterReasoningEffort: normalized });
                }}
                onFallbackChange={(f) => {
                  setValidatorLmStudioFallback(f);
                  setConfig({ ...config, validatorLmStudioFallback: f });
                }}
                label="Validator Model"
                lmStudioEnabled={lmStudioEnabled}
                hasOpenRouterKey={hasOpenRouterKey}
                lmStudioModels={lmStudioModels}
                openRouterModels={openRouterModels}
                oauthModelsByProvider={{
                  openai_codex_oauth: openAICodexModels,
                  [XAI_GROK_PROVIDER]: xaiGrokModels,
                  [SAKANA_FUGU_PROVIDER]: sakanaFuguModels,
                }}
                configuredOAuthProviders={configuredOAuthProviders}
                oauthStatusByProvider={oauthStatusByProvider}
                modelProviders={modelProviders}
              />

              <div className="settings-row">
                <label>Context Window</label>
                <input
                  type="number"
                  value={config.validatorContextSize}
                  onChange={(e) => {
                    const parsed = parseInt(e.target.value);
                    setConfig({ ...config, validatorContextSize: isNaN(parsed) ? '' : parsed });
                  }}
                  min="4096"
                  max="50000000"
                  step="1024"
                />
              </div>

              <div className="settings-row">
                <label>
                  Max Output Tokens{' '}
                  <HelpTooltip
                    label="Learn about validator max output tokens"
                    anchorClassName="help-tooltip-anchor--inline"
                    buttonContent="?"
                  >
                    Uses the max output token setting you enter here. OpenRouter and Codex selections auto-fill from provider metadata when available.
                  </HelpTooltip>
                </label>
                <input
                  type="number"
                  value={validatorMaxOutput}
                  onChange={(e) => {
                    const parsed = parseInt(e.target.value);
                    const value = isNaN(parsed) ? '' : parsed;
                    setValidatorMaxOutput(value);
                    setConfig({ ...config, validatorMaxOutput: value });
                  }}
                  min="1000"
                  max="50000000"
                  step="1000"
                />
              </div>

              {developerModeEnabled && (
                <div className="settings-row settings-row--inline-checkbox">
                  <label className="settings-checkbox-label settings-checkbox-label--supercharge">
                    <input
                      type="checkbox"
                      checked={validatorSuperchargeEnabled}
                      onChange={(e) => {
                        setValidatorSuperchargeEnabled(e.target.checked);
                        setConfig({ ...config, validatorSuperchargeEnabled: e.target.checked });
                      }}
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
          </div>

          <div className="settings-group">
            <h4>Assistant</h4>
            <p className="settings-info">
              Runs in parallel during brainstorming and proof work to retrieve up to 7 relevant verified proof-memory supports from Session History Memory and SyntheticLib4 when enabled. Validators and critique phases never receive Assistant context.
            </p>
            <div
              className={`submitter-config-section${(config.assistantProvider || validatorProvider) === 'openrouter' ? ' role-config-card--openrouter-orange' : ''}`}
              aria-disabled={!assistantMemoryEnabled}
              style={!assistantMemoryEnabled ? { opacity: 0.55, pointerEvents: 'none' } : undefined}
            >
              <h5 className={(config.assistantProvider || validatorProvider) === 'openrouter' ? 'card-title--orange' : ''}>
                Assistant
                {(config.assistantProvider || validatorProvider) === 'openrouter' && <span className="provider-badge-inline">[OpenRouter]</span>}
              </h5>
              {!assistantMemoryEnabled && (
                <p className="settings-hint">
                  Assistant requires Session History Memory. Enable it from Connectivity to edit or run this role.
                </p>
              )}
              <fieldset disabled={!assistantMemoryEnabled} style={{ border: 0, margin: 0, padding: 0, minWidth: 0 }}>
                <AggregatorModelSelector
                  provider={config.assistantProvider || validatorProvider}
                  modelId={config.assistantModel || config.validatorModel || ''}
                  openrouterProvider={config.assistantOpenrouterProvider || null}
                  openrouterReasoningEffort={config.assistantOpenrouterReasoningEffort || validatorOpenrouterReasoningEffort}
                  lmStudioFallbackId={config.assistantLmStudioFallback || null}
                  onProviderChange={(provider) => setConfig({
                    ...config,
                    assistantProvider: provider,
                    assistantModel: '',
                    assistantOpenrouterProvider: null,
                    assistantOpenrouterReasoningEffort: DEFAULT_OPENROUTER_REASONING_EFFORT,
                    assistantLmStudioFallback: null,
                  })}
                  onModelChange={async (modelId) => {
                    const next = {
                      ...config,
                      assistantModel: modelId,
                      assistantOpenrouterProvider: null,
                      assistantOpenrouterReasoningEffort: DEFAULT_OPENROUTER_REASONING_EFFORT,
                    };
                    const provider = config.assistantProvider || validatorProvider;
                    if (modelId && (provider === 'openrouter' || isCloudAccessProvider(provider))) {
                      const autoSettings = provider === 'openrouter'
                        ? await getAutoSettingsForModel(modelId, null)
                        : getCloudAccessAutoSettingsForModel(provider, modelId);
                      if (autoSettings?.contextWindowKnown) next.assistantContextSize = autoSettings.contextWindow;
                      if (autoSettings?.outputCapKnown) next.assistantMaxOutput = autoSettings.maxOutputTokens;
                    }
                    setConfig(next);
                  }}
                  onOpenrouterProviderChange={async (providerName) => {
                    const next = { ...config, assistantOpenrouterProvider: providerName };
                    if (config.assistantModel) {
                      const autoSettings = await getAutoSettingsForModel(config.assistantModel, providerName);
                      if (autoSettings?.contextWindowKnown) next.assistantContextSize = autoSettings.contextWindow;
                      if (autoSettings?.outputCapKnown) next.assistantMaxOutput = autoSettings.maxOutputTokens;
                    }
                    setConfig(next);
                  }}
                  onOpenrouterReasoningEffortChange={(effort) => setConfig({
                    ...config,
                    assistantOpenrouterReasoningEffort: normalizeOpenRouterReasoningEffort(effort),
                  })}
                  onFallbackChange={(fallback) => setConfig({ ...config, assistantLmStudioFallback: fallback })}
                  label="Assistant Model"
                  lmStudioEnabled={lmStudioEnabled}
                  hasOpenRouterKey={hasOpenRouterKey}
                  lmStudioModels={lmStudioModels}
                  openRouterModels={openRouterModels}
                  oauthModelsByProvider={{
                    openai_codex_oauth: openAICodexModels,
                    [XAI_GROK_PROVIDER]: xaiGrokModels,
                    [SAKANA_FUGU_PROVIDER]: sakanaFuguModels,
                  }}
                  configuredOAuthProviders={configuredOAuthProviders}
                  oauthStatusByProvider={oauthStatusByProvider}
                  modelProviders={modelProviders}
                />
                <div className="settings-row">
                  <label>Context Window</label>
                  <input
                    type="number"
                    value={config.assistantContextSize || config.validatorContextSize || DEFAULT_CONTEXT_WINDOW}
                    onChange={(e) => setConfig({ ...config, assistantContextSize: parseInt(e.target.value) || '' })}
                    min="4096"
                    max="50000000"
                    step="1024"
                  />
                </div>
                <div className="settings-row">
                  <label>Max Output Tokens</label>
                  <input
                    type="number"
                    value={config.assistantMaxOutput || validatorMaxOutput}
                    onChange={(e) => setConfig({ ...config, assistantMaxOutput: parseInt(e.target.value) || '' })}
                    min="1000"
                    max="50000000"
                    step="1000"
                  />
                </div>
                {developerModeEnabled && (
                  <div className="settings-row settings-row--inline-checkbox">
                    <label className="settings-checkbox-label settings-checkbox-label--supercharge">
                      <input
                        type="checkbox"
                        checked={Boolean(config.assistantSuperchargeEnabled)}
                        onChange={(e) => setConfig({ ...config, assistantSuperchargeEnabled: e.target.checked })}
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
              </fieldset>
            </div>
          </div>

          {hasOpenRouterKey && (
            <div className="settings-group">
              <h4>OpenRouter Fallback</h4>
              <p className="settings-info">
                Fallback behavior for OpenRouter free-model rate limits.
              </p>
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
            </>
          )}
        </>
      )}

      {/* Current Configuration Summary */}
      <div className="settings-panel mt-1">
        <h3>Current Configuration Summary</h3>
        <pre className="config-summary-pre">
          {JSON.stringify({
            numSubmitters: submitterConfigs.length,
            submitterConfigs: submitterConfigs.map(s => ({
              id: s.submitterId,
              provider: s.provider,
              model: s.modelId?.split('/').pop() || 'Not selected',
              host: s.provider === 'openrouter' ? (s.openrouterProvider || 'Auto') : 'N/A',
              fallback: s.provider === 'openrouter' ? (s.lmStudioFallbackId?.split('/').pop() || 'None') : 'N/A',
              context: s.contextWindow,
              maxOutput: s.maxOutputTokens,
              supercharge: Boolean(s.superchargeEnabled)
            })),
            validator: {
              provider: validatorProvider,
              model: config.validatorModel?.split('/').pop() || 'Not selected',
              host: validatorProvider === 'openrouter' ? (validatorOpenrouterProvider || 'Auto') : 'N/A',
              fallback: validatorProvider === 'openrouter' ? (validatorLmStudioFallback?.split('/').pop() || 'None') : 'N/A',
              context: config.validatorContextSize,
              maxOutput: validatorMaxOutput,
              supercharge: validatorSuperchargeEnabled
            },
            uploadedFiles: config.uploadedFiles?.length || 0
          }, null, 2)}
        </pre>
      </div>
      </div>
    </div>
  );
}
