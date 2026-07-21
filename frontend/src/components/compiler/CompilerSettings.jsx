import React, { useState, useEffect } from 'react';
import { cloudAccessAPI, openRouterAPI, api, aggregatorAPI, compilerAPI } from '../../services/api';
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

const SETTINGS_KEY = 'compiler_settings';
const RAW_VIEW_EXIT_WARNING = 'Switching back to the GUI view will restore your last GUI settings/profile and discard raw-only changes. Continue?';
const formatRawSettings = (value) => JSON.stringify(value, null, 2);
const SUPERCHARGE_TOOLTIP = 'Supercharge makes this role generate 4 full answer attempts, then run a 5th same-model call to choose or synthesize the best final answer. It uses 5x the API calls, so it is about 5x slower and 5x more costly, but can produce more intelligent answers.';
const LEGACY_WRITER_CAMEL_PREFIX = ['high', 'Context'].join('');

const isMeaningfulWriterSetting = (value, suffix) => {
  if (value === undefined || value === null) return false;
  if (typeof value === 'string' && value.trim() === '') return false;
  if ((suffix === 'ContextSize' || suffix === 'MaxOutput') && Number(value) <= 0) return false;
  return true;
};

const readWriterSetting = (settings, suffix) => {
  const current = settings[`writer${suffix}`];
  if (isMeaningfulWriterSetting(current, suffix)) {
    return current;
  }
  return settings[`${LEGACY_WRITER_CAMEL_PREFIX}${suffix}`];
};

function CompilerSettings({
  capabilities,
  connectivityStatus,
  credentialStatusRefreshToken = 0,
  developerModeEnabled = false,
}) {
  // LM Studio and OpenRouter models
  const [lmStudioModels, setLmStudioModels] = useState([]);
  const [openRouterModels, setOpenRouterModels] = useState([]);
  const [openAICodexModels, setOpenAICodexModels] = useState([]);
  const [xaiGrokModels, setXaiGrokModels] = useState([]);
  const [sakanaFuguModels, setSakanaFuguModels] = useState([]);
  const [modelProviders, setModelProviders] = useState({});
  const [hasOpenRouterKey, setHasOpenRouterKey] = useState(false);
  const [hasOpenAICodexLogin, setHasOpenAICodexLogin] = useState(false);
  const [hasXAIGrokLogin, setHasXAIGrokLogin] = useState(false);
  const [hasSakanaFuguKey, setHasSakanaFuguKey] = useState(false);
  const [openAICodexModelError, setOpenAICodexModelError] = useState('');
  const [xaiGrokModelError, setXaiGrokModelError] = useState('');
  const [sakanaFuguModelError, setSakanaFuguModelError] = useState('');
  const [loadingModels, setLoadingModels] = useState(true);
  const [freeOnly, setFreeOnly] = useState(false);
  const [freeModelLooping, setFreeModelLooping] = useState(false);
  const [freeModelAutoSelector, setFreeModelAutoSelector] = useState(false);

  // Validator settings
  const [validatorProvider, setValidatorProvider] = useState('lm_studio');
  const [validatorModel, setValidatorModel] = useState('');
  const [validatorOpenrouterProvider, setValidatorOpenrouterProvider] = useState(null);
  const [validatorOpenrouterReasoningEffort, setValidatorOpenrouterReasoningEffort] = useState(DEFAULT_OPENROUTER_REASONING_EFFORT);
  const [validatorLmStudioFallback, setValidatorLmStudioFallback] = useState(null);
  const [validatorContextSize, setValidatorContextSize] = useState(DEFAULT_CONTEXT_WINDOW);
  const [validatorMaxOutput, setValidatorMaxOutput] = useState(DEFAULT_MAX_OUTPUT_TOKENS);
  const [validatorSuperchargeEnabled, setValidatorSuperchargeEnabled] = useState(false);

  // Assistant settings (defaults to Validator)
  const [assistantProvider, setAssistantProvider] = useState('lm_studio');
  const [assistantModel, setAssistantModel] = useState('');
  const [assistantOpenrouterProvider, setAssistantOpenrouterProvider] = useState(null);
  const [assistantOpenrouterReasoningEffort, setAssistantOpenrouterReasoningEffort] = useState(DEFAULT_OPENROUTER_REASONING_EFFORT);
  const [assistantLmStudioFallback, setAssistantLmStudioFallback] = useState(null);
  const [assistantContextSize, setAssistantContextSize] = useState(DEFAULT_CONTEXT_WINDOW);
  const [assistantMaxOutput, setAssistantMaxOutput] = useState(DEFAULT_MAX_OUTPUT_TOKENS);
  const [assistantSuperchargeEnabled, setAssistantSuperchargeEnabled] = useState(false);

  // Writing settings
  const [writerProvider, setWritingProvider] = useState('lm_studio');
  const [writerModel, setWritingModel] = useState('');
  const [writerOpenrouterProvider, setWritingOpenrouterProvider] = useState(null);
  const [writerOpenrouterReasoningEffort, setWritingOpenrouterReasoningEffort] = useState(DEFAULT_OPENROUTER_REASONING_EFFORT);
  const [writerLmStudioFallback, setWritingLmStudioFallback] = useState(null);
  const [writerContextSize, setWritingContextSize] = useState(DEFAULT_CONTEXT_WINDOW);
  const [writerMaxOutput, setWritingMaxOutput] = useState(DEFAULT_MAX_OUTPUT_TOKENS);
  const [writerSuperchargeEnabled, setWritingSuperchargeEnabled] = useState(false);

  // Rigor & Proofs settings (legacy highParam keys)
  const [highParamProvider, setHighParamProvider] = useState('lm_studio');
  const [highParamModel, setHighParamModel] = useState('');
  const [highParamOpenrouterProvider, setHighParamOpenrouterProvider] = useState(null);
  const [highParamOpenrouterReasoningEffort, setHighParamOpenrouterReasoningEffort] = useState(DEFAULT_OPENROUTER_REASONING_EFFORT);
  const [highParamLmStudioFallback, setHighParamLmStudioFallback] = useState(null);
  const [highParamContextSize, setHighParamContextSize] = useState(DEFAULT_CONTEXT_WINDOW);
  const [highParamMaxOutput, setHighParamMaxOutput] = useState(DEFAULT_MAX_OUTPUT_TOKENS);
  const [highParamSuperchargeEnabled, setHighParamSuperchargeEnabled] = useState(false);

  // Deprecated critique compatibility state; saved values mirror Rigor & Proofs.
  const [critiqueSubmitterProvider, setCritiqueSubmitterProvider] = useState('lm_studio');
  const [critiqueSubmitterModel, setCritiqueSubmitterModel] = useState('');
  const [critiqueSubmitterOpenrouterProvider, setCritiqueSubmitterOpenrouterProvider] = useState(null);
  const [critiqueSubmitterOpenrouterReasoningEffort, setCritiqueSubmitterOpenrouterReasoningEffort] = useState(DEFAULT_OPENROUTER_REASONING_EFFORT);
  const [critiqueSubmitterLmStudioFallback, setCritiqueSubmitterLmStudioFallback] = useState(null);
  const [critiqueSubmitterContextSize, setCritiqueSubmitterContextSize] = useState(DEFAULT_CONTEXT_WINDOW);
  const [critiqueSubmitterMaxOutput, setCritiqueSubmitterMaxOutput] = useState(DEFAULT_MAX_OUTPUT_TOKENS);
  const [critiqueSubmitterSuperchargeEnabled, setCritiqueSubmitterSuperchargeEnabled] = useState(false);

  const [saveStatus, setSaveStatus] = useState('');
  const [isLoaded, setIsLoaded] = useState(false);
  const [editRawSettings, setEditRawSettings] = useState(false);
  const [rawSettingsText, setRawSettingsText] = useState('');
  const [rawSettingsMessage, setRawSettingsMessage] = useState('');
  const [guiSettingsBeforeRaw, setGuiSettingsBeforeRaw] = useState(null);

  // Critique prompt editor state
  const [critiquePromptExpanded, setCritiquePromptExpanded] = useState(false);
  const [customCritiquePrompt, setCustomCritiquePrompt] = useState('');
  const [critiquePromptSaved, setCritiquePromptSaved] = useState(false);
  const [defaultCritiquePrompt, setDefaultCritiquePrompt] = useState('');
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

  const normalizeRoleState = (provider, model, openrouterProvider, reasoningEffort) => {
    const keepOpenRouterState = provider === 'openrouter';
    return {
      provider: 'openrouter',
      model: keepOpenRouterState ? (model || '') : '',
      openrouterProvider: keepOpenRouterState ? (openrouterProvider || null) : null,
      openrouterReasoningEffort: normalizeOpenRouterReasoningEffort(reasoningEffort),
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
          console.error('Failed to check OpenAI Codex login:', err);
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
          console.error('Failed to check xAI Grok login:', err);
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
          console.error('Failed to check Sakana Fugu API key:', err);
          setSakanaFuguModelError(`Sakana Fugu status could not be checked: ${err.message || 'unknown error'}.`);
        }
      } else {
        setHasSakanaFuguKey(false);
        setSakanaFuguModels([]);
        setSakanaFuguModelError('');
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
          if (settings.validatorOpenrouterReasoningEffort) setValidatorOpenrouterReasoningEffort(normalizeOpenRouterReasoningEffort(settings.validatorOpenrouterReasoningEffort));
          if (settings.validatorLmStudioFallback) setValidatorLmStudioFallback(settings.validatorLmStudioFallback);
          if (settings.validatorContextSize) setValidatorContextSize(settings.validatorContextSize);
          if (settings.validatorMaxOutput) setValidatorMaxOutput(settings.validatorMaxOutput);
          if (settings.validatorSuperchargeEnabled !== undefined) setValidatorSuperchargeEnabled(settings.validatorSuperchargeEnabled);
          // Assistant
          setAssistantProvider(settings.assistantProvider || settings.validatorProvider || 'lm_studio');
          setAssistantModel(settings.assistantModel || settings.validatorModel || '');
          setAssistantOpenrouterProvider(settings.assistantOpenrouterProvider || settings.validatorOpenrouterProvider || null);
          setAssistantOpenrouterReasoningEffort(normalizeOpenRouterReasoningEffort(settings.assistantOpenrouterReasoningEffort || settings.validatorOpenrouterReasoningEffort));
          setAssistantLmStudioFallback(settings.assistantLmStudioFallback || settings.validatorLmStudioFallback || null);
          setAssistantContextSize(settings.assistantContextSize || settings.validatorContextSize || DEFAULT_CONTEXT_WINDOW);
          setAssistantMaxOutput(settings.assistantMaxOutput || settings.validatorMaxOutput || DEFAULT_MAX_OUTPUT_TOKENS);
          setAssistantSuperchargeEnabled(
            settings.assistantModel
              ? Boolean(settings.assistantSuperchargeEnabled)
              : Boolean(settings.validatorSuperchargeEnabled)
          );
          // Writing
          if (readWriterSetting(settings, 'Provider')) setWritingProvider(readWriterSetting(settings, 'Provider'));
          if (readWriterSetting(settings, 'Model')) setWritingModel(readWriterSetting(settings, 'Model'));
          if (readWriterSetting(settings, 'OpenrouterProvider')) setWritingOpenrouterProvider(readWriterSetting(settings, 'OpenrouterProvider'));
          if (readWriterSetting(settings, 'OpenrouterReasoningEffort')) setWritingOpenrouterReasoningEffort(normalizeOpenRouterReasoningEffort(readWriterSetting(settings, 'OpenrouterReasoningEffort')));
          if (readWriterSetting(settings, 'LmStudioFallback')) setWritingLmStudioFallback(readWriterSetting(settings, 'LmStudioFallback'));
          if (readWriterSetting(settings, 'ContextSize')) setWritingContextSize(readWriterSetting(settings, 'ContextSize'));
          if (readWriterSetting(settings, 'MaxOutput')) setWritingMaxOutput(readWriterSetting(settings, 'MaxOutput'));
          if (readWriterSetting(settings, 'SuperchargeEnabled') !== undefined) setWritingSuperchargeEnabled(readWriterSetting(settings, 'SuperchargeEnabled'));
          // Rigor & Proofs
          if (settings.highParamProvider) setHighParamProvider(settings.highParamProvider);
          if (settings.highParamModel) setHighParamModel(settings.highParamModel);
          if (settings.highParamOpenrouterProvider) setHighParamOpenrouterProvider(settings.highParamOpenrouterProvider);
          if (settings.highParamOpenrouterReasoningEffort) setHighParamOpenrouterReasoningEffort(normalizeOpenRouterReasoningEffort(settings.highParamOpenrouterReasoningEffort));
          if (settings.highParamLmStudioFallback) setHighParamLmStudioFallback(settings.highParamLmStudioFallback);
          if (settings.highParamContextSize) setHighParamContextSize(settings.highParamContextSize);
          if (settings.highParamMaxOutput) setHighParamMaxOutput(settings.highParamMaxOutput);
          if (settings.highParamSuperchargeEnabled !== undefined) setHighParamSuperchargeEnabled(settings.highParamSuperchargeEnabled);
          // Deprecated critique compatibility
          if (settings.critiqueSubmitterProvider) setCritiqueSubmitterProvider(settings.critiqueSubmitterProvider);
          if (settings.critiqueSubmitterModel) setCritiqueSubmitterModel(settings.critiqueSubmitterModel);
          if (settings.critiqueSubmitterOpenrouterProvider) setCritiqueSubmitterOpenrouterProvider(settings.critiqueSubmitterOpenrouterProvider);
          if (settings.critiqueSubmitterOpenrouterReasoningEffort) setCritiqueSubmitterOpenrouterReasoningEffort(normalizeOpenRouterReasoningEffort(settings.critiqueSubmitterOpenrouterReasoningEffort));
          if (settings.critiqueSubmitterLmStudioFallback) setCritiqueSubmitterLmStudioFallback(settings.critiqueSubmitterLmStudioFallback);
          if (settings.critiqueSubmitterContextSize) setCritiqueSubmitterContextSize(settings.critiqueSubmitterContextSize);
          if (settings.critiqueSubmitterMaxOutput) setCritiqueSubmitterMaxOutput(settings.critiqueSubmitterMaxOutput);
          if (settings.critiqueSubmitterSuperchargeEnabled !== undefined) setCritiqueSubmitterSuperchargeEnabled(settings.critiqueSubmitterSuperchargeEnabled);
          // Free-only toggle
          if (settings.freeOnly !== undefined) setFreeOnly(settings.freeOnly);
          if (settings.freeModelLooping !== undefined) setFreeModelLooping(settings.freeModelLooping);
          if (settings.freeModelAutoSelector !== undefined) setFreeModelAutoSelector(settings.freeModelAutoSelector);
          if (settings.modelProviders) setModelProviders(settings.modelProviders);
        } catch (error) {
          console.error('Failed to load compiler settings:', error);
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
      validatorOpenrouterProvider,
      validatorOpenrouterReasoningEffort
    );
    const nextWriting = normalizeRoleState(
      writerProvider,
      writerModel,
      writerOpenrouterProvider,
      writerOpenrouterReasoningEffort
    );
    const nextHighParam = normalizeRoleState(
      highParamProvider,
      highParamModel,
      highParamOpenrouterProvider,
      highParamOpenrouterReasoningEffort
    );
    const nextCritique = normalizeRoleState(
      critiqueSubmitterProvider,
      critiqueSubmitterModel,
      critiqueSubmitterOpenrouterProvider,
      critiqueSubmitterOpenrouterReasoningEffort
    );
    const nextAssistant = normalizeRoleState(
      assistantProvider,
      assistantModel,
      assistantOpenrouterProvider,
      assistantOpenrouterReasoningEffort
    );

    if (validatorProvider !== nextValidator.provider) setValidatorProvider(nextValidator.provider);
    if (validatorModel !== nextValidator.model) setValidatorModel(nextValidator.model);
    if (validatorOpenrouterProvider !== nextValidator.openrouterProvider) {
      setValidatorOpenrouterProvider(nextValidator.openrouterProvider);
    }
    if (validatorOpenrouterReasoningEffort !== nextValidator.openrouterReasoningEffort) {
      setValidatorOpenrouterReasoningEffort(nextValidator.openrouterReasoningEffort);
    }
    if (validatorLmStudioFallback !== null) setValidatorLmStudioFallback(null);

    if (writerProvider !== nextWriting.provider) setWritingProvider(nextWriting.provider);
    if (writerModel !== nextWriting.model) setWritingModel(nextWriting.model);
    if (writerOpenrouterProvider !== nextWriting.openrouterProvider) {
      setWritingOpenrouterProvider(nextWriting.openrouterProvider);
    }
    if (writerOpenrouterReasoningEffort !== nextWriting.openrouterReasoningEffort) {
      setWritingOpenrouterReasoningEffort(nextWriting.openrouterReasoningEffort);
    }
    if (writerLmStudioFallback !== null) setWritingLmStudioFallback(null);

    if (highParamProvider !== nextHighParam.provider) setHighParamProvider(nextHighParam.provider);
    if (highParamModel !== nextHighParam.model) setHighParamModel(nextHighParam.model);
    if (highParamOpenrouterProvider !== nextHighParam.openrouterProvider) {
      setHighParamOpenrouterProvider(nextHighParam.openrouterProvider);
    }
    if (highParamOpenrouterReasoningEffort !== nextHighParam.openrouterReasoningEffort) {
      setHighParamOpenrouterReasoningEffort(nextHighParam.openrouterReasoningEffort);
    }
    if (highParamLmStudioFallback !== null) setHighParamLmStudioFallback(null);

    if (critiqueSubmitterProvider !== nextCritique.provider) {
      setCritiqueSubmitterProvider(nextCritique.provider);
    }
    if (critiqueSubmitterModel !== nextCritique.model) setCritiqueSubmitterModel(nextCritique.model);
    if (critiqueSubmitterOpenrouterProvider !== nextCritique.openrouterProvider) {
      setCritiqueSubmitterOpenrouterProvider(nextCritique.openrouterProvider);
    }
    if (critiqueSubmitterOpenrouterReasoningEffort !== nextCritique.openrouterReasoningEffort) {
      setCritiqueSubmitterOpenrouterReasoningEffort(nextCritique.openrouterReasoningEffort);
    }
    if (critiqueSubmitterLmStudioFallback !== null) setCritiqueSubmitterLmStudioFallback(null);
    if (assistantProvider !== nextAssistant.provider) setAssistantProvider(nextAssistant.provider);
    if (assistantModel !== nextAssistant.model) setAssistantModel(nextAssistant.model);
    if (assistantOpenrouterProvider !== nextAssistant.openrouterProvider) {
      setAssistantOpenrouterProvider(nextAssistant.openrouterProvider);
    }
    if (assistantOpenrouterReasoningEffort !== nextAssistant.openrouterReasoningEffort) {
      setAssistantOpenrouterReasoningEffort(nextAssistant.openrouterReasoningEffort);
    }
    if (assistantLmStudioFallback !== null) setAssistantLmStudioFallback(null);
  }, [
    lmStudioEnabled,
    validatorProvider,
    validatorModel,
    validatorOpenrouterProvider,
    validatorOpenrouterReasoningEffort,
    validatorLmStudioFallback,
    writerProvider,
    writerModel,
    writerOpenrouterProvider,
    writerOpenrouterReasoningEffort,
    writerLmStudioFallback,
    highParamProvider,
    highParamModel,
    highParamOpenrouterProvider,
    highParamOpenrouterReasoningEffort,
    highParamLmStudioFallback,
    critiqueSubmitterProvider,
    critiqueSubmitterModel,
    critiqueSubmitterOpenrouterProvider,
    critiqueSubmitterOpenrouterReasoningEffort,
    critiqueSubmitterLmStudioFallback,
    assistantProvider,
    assistantModel,
    assistantOpenrouterProvider,
    assistantOpenrouterReasoningEffort,
    assistantLmStudioFallback,
  ]);

  // Fetch providers for any OpenRouter models after settings are loaded
  useEffect(() => {
    if (!isLoaded || !hasOpenRouterKey) return;
    
    // Fetch providers for validator
    if (validatorProvider === 'openrouter' && validatorModel) {
      fetchProvidersForModel(validatorModel);
    }
    
    // Fetch providers for writer
    if (writerProvider === 'openrouter' && writerModel) {
      fetchProvidersForModel(writerModel);
    }
    
    // Fetch providers for Rigor & Proofs
    if (highParamProvider === 'openrouter' && highParamModel) {
      fetchProvidersForModel(highParamModel);
    }
    
    // Fetch providers for deprecated critique compatibility fields
    if (critiqueSubmitterProvider === 'openrouter' && critiqueSubmitterModel) {
      fetchProvidersForModel(critiqueSubmitterModel);
    }

    // Fetch providers for Assistant
    if (assistantProvider === 'openrouter' && assistantModel) {
      fetchProvidersForModel(assistantModel);
    }
  }, [isLoaded, hasOpenRouterKey, validatorProvider, validatorModel, writerProvider, writerModel, highParamProvider, highParamModel, critiqueSubmitterProvider, critiqueSubmitterModel, assistantProvider, assistantModel]);

  // Save settings to localStorage whenever values change
  useEffect(() => {
    if (!isLoaded) return;
    
    const settings = {
      validatorProvider, validatorModel, validatorOpenrouterProvider, validatorOpenrouterReasoningEffort, validatorLmStudioFallback,
      validatorContextSize, validatorMaxOutput, validatorSuperchargeEnabled,
      assistantProvider, assistantModel, assistantOpenrouterProvider, assistantOpenrouterReasoningEffort, assistantLmStudioFallback,
      assistantContextSize, assistantMaxOutput, assistantSuperchargeEnabled,
      writerProvider, writerModel, writerOpenrouterProvider, writerOpenrouterReasoningEffort, writerLmStudioFallback,
      writerContextSize, writerMaxOutput, writerSuperchargeEnabled,
      highParamProvider, highParamModel, highParamOpenrouterProvider, highParamOpenrouterReasoningEffort, highParamLmStudioFallback,
      highParamContextSize, highParamMaxOutput, highParamSuperchargeEnabled,
      critiqueSubmitterProvider: highParamProvider,
      critiqueSubmitterModel: highParamModel,
      critiqueSubmitterOpenrouterProvider: highParamOpenrouterProvider,
      critiqueSubmitterOpenrouterReasoningEffort: highParamOpenrouterReasoningEffort,
      critiqueSubmitterLmStudioFallback: highParamLmStudioFallback,
      critiqueSubmitterContextSize: highParamContextSize,
      critiqueSubmitterMaxOutput: highParamMaxOutput,
      critiqueSubmitterSuperchargeEnabled: highParamSuperchargeEnabled,
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
    isLoaded, validatorProvider, validatorModel, validatorOpenrouterProvider, validatorOpenrouterReasoningEffort, validatorLmStudioFallback,
    validatorContextSize, validatorMaxOutput, validatorSuperchargeEnabled,
    assistantProvider, assistantModel, assistantOpenrouterProvider, assistantOpenrouterReasoningEffort, assistantLmStudioFallback,
    assistantContextSize, assistantMaxOutput, assistantSuperchargeEnabled,
    writerProvider, writerModel, writerOpenrouterProvider, writerOpenrouterReasoningEffort, writerLmStudioFallback,
    writerContextSize, writerMaxOutput, writerSuperchargeEnabled,
    highParamProvider, highParamModel, highParamOpenrouterProvider, highParamOpenrouterReasoningEffort, highParamLmStudioFallback,
    highParamContextSize, highParamMaxOutput, highParamSuperchargeEnabled,
    critiqueSubmitterProvider, critiqueSubmitterModel, critiqueSubmitterOpenrouterProvider, critiqueSubmitterOpenrouterReasoningEffort, critiqueSubmitterLmStudioFallback,
    critiqueSubmitterContextSize, critiqueSubmitterMaxOutput, critiqueSubmitterSuperchargeEnabled,
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
      logContext: 'Compiler settings',
    });
    return () => {
      isCurrent = false;
    };
  }, [credentialStatusRefreshToken]);

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

  const getCodexAutoSettingsForModel = (modelId) => {
    const model = openAICodexModels.find((item) => item.id === modelId);
    if (!model) {
      console.debug('[CompilerCodexAutoFill] model not in loaded list, skipping auto-fill', { modelId });
      return null;
    }
    const autoSettings = computeCodexAutoSettings(model);
    if (autoSettings.warnings.length > 0) {
      console.warn('[CompilerCodexAutoFill] auto-settings fallback used:', autoSettings.warnings);
    }
    return autoSettings;
  };

  const getOAuthModels = (provider) => {
    if (provider === 'openai_codex_oauth') return openAICodexModels;
    if (provider === XAI_GROK_PROVIDER) return xaiGrokModels;
    if (provider === SAKANA_FUGU_PROVIDER) return sakanaFuguModels;
    return [];
  };

  const getCloudAccessAutoSettingsForModel = (provider, modelId) => {
    if (provider === 'openai_codex_oauth') {
      return getCodexAutoSettingsForModel(modelId);
    }
    const model = getOAuthModels(provider).find((item) => item.id === modelId);
    if (!model) {
      console.debug('[CompilerOAuthAutoFill] model not in loaded list, skipping auto-fill', { provider, modelId });
      return null;
    }
    const autoSettings = provider === XAI_GROK_PROVIDER
      ? computeXAIGrokAutoSettings(model)
      : (provider === SAKANA_FUGU_PROVIDER
        ? computeSakanaFuguAutoSettings(model)
        : computeCloudAccessAutoSettings(model, cloudAccessProviderLabel(provider)));
    if (autoSettings.warnings.length > 0) {
      console.warn('[CompilerOAuthAutoFill] auto-settings fallback used:', autoSettings.warnings);
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
        setValidatorOpenrouterReasoningEffort(DEFAULT_OPENROUTER_REASONING_EFFORT);
        setValidatorLmStudioFallback(null);
        
        setWritingProvider('lm_studio');
        setWritingModel(settings.submitter_model);
        setWritingOpenrouterProvider(null);
        setWritingOpenrouterReasoningEffort(DEFAULT_OPENROUTER_REASONING_EFFORT);
        setWritingLmStudioFallback(null);
        
        setHighParamProvider('lm_studio');
        setHighParamModel(settings.submitter_model);
        setHighParamOpenrouterProvider(null);
        setHighParamOpenrouterReasoningEffort(DEFAULT_OPENROUTER_REASONING_EFFORT);
        setHighParamLmStudioFallback(null);
        
        setCritiqueSubmitterProvider('lm_studio');
        setCritiqueSubmitterModel(settings.submitter_model);
        setCritiqueSubmitterOpenrouterProvider(null);
        setCritiqueSubmitterOpenrouterReasoningEffort(DEFAULT_OPENROUTER_REASONING_EFFORT);
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

  const getCompilerRawSettings = () => ({
    validatorProvider,
    validatorModel,
    validatorOpenrouterProvider,
    validatorOpenrouterReasoningEffort,
    validatorLmStudioFallback,
    validatorContextSize,
    validatorMaxOutput,
    validatorSuperchargeEnabled,
    assistantProvider,
    assistantModel,
    assistantOpenrouterProvider,
    assistantOpenrouterReasoningEffort,
    assistantLmStudioFallback,
    assistantContextSize,
    assistantMaxOutput,
    assistantSuperchargeEnabled,
    writerProvider,
    writerModel,
    writerOpenrouterProvider,
    writerOpenrouterReasoningEffort,
    writerLmStudioFallback,
    writerContextSize,
    writerMaxOutput,
    writerSuperchargeEnabled,
    highParamProvider,
    highParamModel,
    highParamOpenrouterProvider,
    highParamOpenrouterReasoningEffort,
    highParamLmStudioFallback,
    highParamContextSize,
    highParamMaxOutput,
    highParamSuperchargeEnabled,
    critiqueSubmitterProvider,
    critiqueSubmitterModel,
    critiqueSubmitterOpenrouterProvider,
    critiqueSubmitterOpenrouterReasoningEffort,
    critiqueSubmitterLmStudioFallback,
    critiqueSubmitterContextSize,
    critiqueSubmitterMaxOutput,
    critiqueSubmitterSuperchargeEnabled,
    freeOnly,
    freeModelLooping,
    freeModelAutoSelector,
    modelProviders,
  });

  const applyCompilerRawSettings = (rawSettings, { updateRawText = true } = {}) => {
    setValidatorProvider(rawSettings.validatorProvider || 'lm_studio');
    setValidatorModel(rawSettings.validatorModel || '');
    setValidatorOpenrouterProvider(rawSettings.validatorOpenrouterProvider || null);
    setValidatorOpenrouterReasoningEffort(normalizeOpenRouterReasoningEffort(rawSettings.validatorOpenrouterReasoningEffort));
    setValidatorLmStudioFallback(rawSettings.validatorLmStudioFallback || null);
    setValidatorContextSize(rawSettings.validatorContextSize ?? DEFAULT_CONTEXT_WINDOW);
    setValidatorMaxOutput(rawSettings.validatorMaxOutput ?? DEFAULT_MAX_OUTPUT_TOKENS);
    setValidatorSuperchargeEnabled(Boolean(rawSettings.validatorSuperchargeEnabled));
    setAssistantProvider(rawSettings.assistantProvider || rawSettings.validatorProvider || 'lm_studio');
    setAssistantModel(rawSettings.assistantModel || rawSettings.validatorModel || '');
    setAssistantOpenrouterProvider(rawSettings.assistantOpenrouterProvider || rawSettings.validatorOpenrouterProvider || null);
    setAssistantOpenrouterReasoningEffort(normalizeOpenRouterReasoningEffort(rawSettings.assistantOpenrouterReasoningEffort || rawSettings.validatorOpenrouterReasoningEffort));
    setAssistantLmStudioFallback(rawSettings.assistantLmStudioFallback || rawSettings.validatorLmStudioFallback || null);
    setAssistantContextSize(rawSettings.assistantContextSize || rawSettings.validatorContextSize || DEFAULT_CONTEXT_WINDOW);
    setAssistantMaxOutput(rawSettings.assistantMaxOutput || rawSettings.validatorMaxOutput || DEFAULT_MAX_OUTPUT_TOKENS);
    setAssistantSuperchargeEnabled(
      rawSettings.assistantModel
        ? Boolean(rawSettings.assistantSuperchargeEnabled)
        : Boolean(rawSettings.validatorSuperchargeEnabled)
    );
    setWritingProvider(readWriterSetting(rawSettings, 'Provider') || 'lm_studio');
    setWritingModel(readWriterSetting(rawSettings, 'Model') || '');
    setWritingOpenrouterProvider(readWriterSetting(rawSettings, 'OpenrouterProvider') || null);
    setWritingOpenrouterReasoningEffort(normalizeOpenRouterReasoningEffort(readWriterSetting(rawSettings, 'OpenrouterReasoningEffort')));
    setWritingLmStudioFallback(readWriterSetting(rawSettings, 'LmStudioFallback') || null);
    setWritingContextSize(readWriterSetting(rawSettings, 'ContextSize') ?? DEFAULT_CONTEXT_WINDOW);
    setWritingMaxOutput(readWriterSetting(rawSettings, 'MaxOutput') ?? DEFAULT_MAX_OUTPUT_TOKENS);
    setWritingSuperchargeEnabled(Boolean(readWriterSetting(rawSettings, 'SuperchargeEnabled')));
    setHighParamProvider(rawSettings.highParamProvider || 'lm_studio');
    setHighParamModel(rawSettings.highParamModel || '');
    setHighParamOpenrouterProvider(rawSettings.highParamOpenrouterProvider || null);
    setHighParamOpenrouterReasoningEffort(normalizeOpenRouterReasoningEffort(rawSettings.highParamOpenrouterReasoningEffort));
    setHighParamLmStudioFallback(rawSettings.highParamLmStudioFallback || null);
    setHighParamContextSize(rawSettings.highParamContextSize ?? DEFAULT_CONTEXT_WINDOW);
    setHighParamMaxOutput(rawSettings.highParamMaxOutput ?? DEFAULT_MAX_OUTPUT_TOKENS);
    setHighParamSuperchargeEnabled(Boolean(rawSettings.highParamSuperchargeEnabled));
    setCritiqueSubmitterProvider(rawSettings.critiqueSubmitterProvider || 'lm_studio');
    setCritiqueSubmitterModel(rawSettings.critiqueSubmitterModel || '');
    setCritiqueSubmitterOpenrouterProvider(rawSettings.critiqueSubmitterOpenrouterProvider || null);
    setCritiqueSubmitterOpenrouterReasoningEffort(normalizeOpenRouterReasoningEffort(rawSettings.critiqueSubmitterOpenrouterReasoningEffort));
    setCritiqueSubmitterLmStudioFallback(rawSettings.critiqueSubmitterLmStudioFallback || null);
    setCritiqueSubmitterContextSize(rawSettings.critiqueSubmitterContextSize ?? DEFAULT_CONTEXT_WINDOW);
    setCritiqueSubmitterMaxOutput(rawSettings.critiqueSubmitterMaxOutput ?? DEFAULT_MAX_OUTPUT_TOKENS);
    setCritiqueSubmitterSuperchargeEnabled(Boolean(rawSettings.critiqueSubmitterSuperchargeEnabled));
    setFreeOnly(rawSettings.freeOnly ?? false);
    setFreeModelLooping(rawSettings.freeModelLooping ?? false);
    setFreeModelAutoSelector(rawSettings.freeModelAutoSelector ?? false);
    setModelProviders(rawSettings.modelProviders || {});
    openRouterAPI
      .setFreeModelSettings(rawSettings.freeModelLooping ?? false, rawSettings.freeModelAutoSelector ?? false)
      .catch(() => {});

    if (updateRawText) {
      setRawSettingsText(formatRawSettings({
        ...rawSettings,
        validatorProvider: rawSettings.validatorProvider || 'lm_studio',
        validatorModel: rawSettings.validatorModel || '',
        validatorOpenrouterReasoningEffort: normalizeOpenRouterReasoningEffort(rawSettings.validatorOpenrouterReasoningEffort),
        assistantProvider: rawSettings.assistantProvider || rawSettings.validatorProvider || 'lm_studio',
        assistantModel: rawSettings.assistantModel || rawSettings.validatorModel || '',
        assistantOpenrouterReasoningEffort: normalizeOpenRouterReasoningEffort(rawSettings.assistantOpenrouterReasoningEffort || rawSettings.validatorOpenrouterReasoningEffort),
        assistantContextSize: rawSettings.assistantContextSize || rawSettings.validatorContextSize || DEFAULT_CONTEXT_WINDOW,
        assistantMaxOutput: rawSettings.assistantMaxOutput || rawSettings.validatorMaxOutput || DEFAULT_MAX_OUTPUT_TOKENS,
        assistantSuperchargeEnabled: rawSettings.assistantModel
          ? Boolean(rawSettings.assistantSuperchargeEnabled)
          : Boolean(rawSettings.validatorSuperchargeEnabled),
        writerProvider: readWriterSetting(rawSettings, 'Provider') || 'lm_studio',
        writerModel: readWriterSetting(rawSettings, 'Model') || '',
        writerOpenrouterReasoningEffort: normalizeOpenRouterReasoningEffort(readWriterSetting(rawSettings, 'OpenrouterReasoningEffort')),
        highParamProvider: rawSettings.highParamProvider || 'lm_studio',
        highParamModel: rawSettings.highParamModel || '',
        highParamOpenrouterReasoningEffort: normalizeOpenRouterReasoningEffort(rawSettings.highParamOpenrouterReasoningEffort),
        critiqueSubmitterProvider: rawSettings.critiqueSubmitterProvider || 'lm_studio',
        critiqueSubmitterModel: rawSettings.critiqueSubmitterModel || '',
        critiqueSubmitterOpenrouterReasoningEffort: normalizeOpenRouterReasoningEffort(rawSettings.critiqueSubmitterOpenrouterReasoningEffort),
        freeOnly: rawSettings.freeOnly ?? false,
        freeModelLooping: rawSettings.freeModelLooping ?? false,
        freeModelAutoSelector: rawSettings.freeModelAutoSelector ?? false,
        modelProviders: rawSettings.modelProviders || {},
      }));
    }
  };

  const handleRawEditToggle = (checked) => {
    if (checked) {
      const currentSettings = getCompilerRawSettings();
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
      applyCompilerRawSettings(guiSettingsBeforeRaw, { updateRawText: false });
    }
    setRawSettingsMessage('');
    setEditRawSettings(false);
  };

  const saveRawSettings = () => {
    try {
      const parsed = JSON.parse(rawSettingsText);
      applyCompilerRawSettings(parsed);
      setRawSettingsMessage('Saved raw settings.');
    } catch (error) {
      setRawSettingsMessage(`Invalid JSON: ${error.message}`);
    }
  };

  // Reusable role renderer. Keep this as a plain render helper instead of a nested
  // component so periodic parent refreshes do not remount open native selects.
  const renderRoleConfig = ({
    title, 
    description, 
    provider, setProvider,
    model, setModel,
    openrouterProv, setOpenrouterProv,
    openrouterReasoningEffort, setOpenrouterReasoningEffort,
    fallback, setFallback,
    contextSize, setContextSize,
    maxOutput, setMaxOutput,
    superchargeEnabled, setSuperchargeEnabled,
    showProofStrengthBadge = false,
    disabled = false,
  }) => {
    const effectiveProvider = lmStudioEnabled ? provider : 'openrouter';
    const models = effectiveProvider === 'openrouter'
      ? openRouterModels
      : (isCloudAccessProvider(effectiveProvider) ? getOAuthModels(effectiveProvider) : lmStudioModels);
    const providers = model && effectiveProvider === 'openrouter'
      ? getProviderNames(modelProviders[model])
      : [];
    const reasoningInfo = effectiveProvider === 'openrouter'
      ? getReasoningSupportInfo(modelProviders[model], openrouterProv || null)
      : { hasEndpointMetadata: false, supportsReasoning: false };

    return (
      <div
        className={`submitter-config-section${effectiveProvider === 'openrouter' ? ' role-config-card--openrouter-orange' : ''}`}
        aria-disabled={disabled}
        style={disabled ? { opacity: 0.55, pointerEvents: 'none' } : undefined}
      >
        <h5 className={effectiveProvider === 'openrouter' ? 'card-title--orange' : ''}>
          <span className="role-title-with-badges">
            <span>{title}</span>
            {showProofStrengthBadge && <ProofStrengthBadge />}
          </span>
          {effectiveProvider === 'openrouter' && <span className="provider-badge-inline">[OpenRouter]</span>}
        </h5>
        <p className="settings-hint">{description}</p>
        {disabled && (
          <p className="settings-hint">
            Assistant requires Session History Memory. Enable it from Connectivity to edit or run this role.
          </p>
        )}
        <fieldset disabled={disabled} style={{ border: 0, margin: 0, padding: 0, minWidth: 0 }}>

        {/* Provider Toggle */}
        <div className="settings-row">
          <label>Provider</label>
          {lmStudioEnabled ? (
            <div className="provider-toggle-group">
              <button
                type="button"
                onClick={() => {
                  setProvider('lm_studio');
                  setModel('');
                  setOpenrouterProv(null);
                  setOpenrouterReasoningEffort(DEFAULT_OPENROUTER_REASONING_EFFORT);
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
                    setOpenrouterReasoningEffort(DEFAULT_OPENROUTER_REASONING_EFFORT);
                    setFallback(null);
                  }
                }}
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
                    setProvider(chooseCloudAccessProvider(oauthStatusByProvider, provider));
                    setModel('');
                    setOpenrouterProv(null);
                    setOpenrouterReasoningEffort(DEFAULT_OPENROUTER_REASONING_EFFORT);
                    setFallback(null);
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
                  onChange={(event) => {
                    setProvider(event.target.value);
                    setModel('');
                    setOpenrouterProv(null);
                    setOpenrouterReasoningEffort(DEFAULT_OPENROUTER_REASONING_EFFORT);
                    setFallback(null);
                  }}
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
          <label>Model</label>
          <select
            value={model || ''}
            onChange={async (e) => {
              const m = e.target.value;
              setModel(m);
              setOpenrouterProv(null);
              setOpenrouterReasoningEffort(DEFAULT_OPENROUTER_REASONING_EFFORT);
              if ((effectiveProvider === 'openrouter' || isCloudAccessProvider(effectiveProvider)) && m) {
                const autoSettings = effectiveProvider === 'openrouter'
                  ? await getAutoSettingsForModel(m, null)
                  : getCloudAccessAutoSettingsForModel(effectiveProvider, m);
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
          <div className="settings-row">
            <label>Host Provider (optional)</label>
            <select
              className="openrouter-host-provider-select"
              value={openrouterProv || ''}
              title={getOpenRouterProviderTitle(openrouterProv)}
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
                <option key={p} value={p} title={getOpenRouterProviderTitle(p)}>
                  {formatOpenRouterProviderLabel(p)}
                </option>
              ))}
            </select>
          </div>
        )}

        {effectiveProvider === 'openrouter' && model && (
          <div className="settings-row">
            <label>Reasoning Effort</label>
            <select
              value={normalizeOpenRouterReasoningEffort(openrouterReasoningEffort)}
              onChange={(e) => setOpenrouterReasoningEffort(e.target.value)}
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

        {effectiveProvider === SAKANA_FUGU_PROVIDER && model && (
          <div className="settings-row">
            <label>Reasoning Effort</label>
            <select
              value={normalizeOpenRouterReasoningEffort(openrouterReasoningEffort)}
              onChange={(e) => setOpenrouterReasoningEffort(e.target.value)}
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

        {/* LM Studio Fallback (if cloud provider) */}
        {effectiveProvider !== 'lm_studio' && lmStudioEnabled && (
          <div className="settings-row">
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
            <small className="settings-hint">Used if cloud provider access fails or credits run out</small>
          </div>
        )}

        <div className="settings-row">
          <label>Context Window</label>
          <input
            type="number"
            value={contextSize}
            onChange={(e) => {
              const parsed = parseInt(e.target.value, 10);
              setContextSize(isNaN(parsed) ? '' : parsed);
            }}
            min={4096}
            max={50000000}
            step={1024}
          />
        </div>

        <div className="settings-row">
          <label>Max Output Tokens</label>
          <input
            type="number"
            value={maxOutput}
            onChange={(e) => {
              const parsed = parseInt(e.target.value, 10);
              setMaxOutput(isNaN(parsed) ? '' : parsed);
            }}
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
                checked={Boolean(superchargeEnabled)}
                onChange={(e) => setSuperchargeEnabled(e.target.checked)}
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
    );
  };

  if (loadingModels) {
    return <div>Loading models...</div>;
  }

  return (
    <div className="autonomous-settings-layout">
      <HighlightedModelsSidebar />
      <div className="autonomous-settings">
          <div className="settings-header-row">
            <h2>Compiler Settings</h2>
            {saveStatus && (
              <div className="save-message">
                {saveStatus}
              </div>
            )}
          </div>

      {/* OpenRouter Status Banner */}
      {!hasOpenRouterKey && (
        <div className="openrouter-banner">
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
      {xaiGrokModelError && (
        <div className="test-result-banner test-result-banner--error" style={{ marginBottom: '1rem' }}>
          {xaiGrokModelError}
        </div>
      )}

      <div className="model-refresh-controls">
        {lmStudioEnabled && (
          <>
            <button
              onClick={handleUseAggregatorModels}
              className="secondary"
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
            <button
              className="secondary"
              onClick={() => window.open('https://openrouter.ai/models', '_blank', 'noopener,noreferrer')}
              title="Browse all available OpenRouter models"
            >
              🔗 OpenRouter Model List
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
        <h4>Model Configuration</h4>
        <p className="settings-info">
          Configure the validator and compiler roles used by manual paper compilation.
        </p>
        
        {renderRoleConfig({
          title: 'Validator',
          description: 'Validates all submissions for coherence, rigor, placement, and non-redundancy.',
          provider: validatorProvider,
          setProvider: setValidatorProvider,
          model: validatorModel,
          setModel: setValidatorModel,
          openrouterProv: validatorOpenrouterProvider,
          setOpenrouterProv: setValidatorOpenrouterProvider,
          openrouterReasoningEffort: validatorOpenrouterReasoningEffort,
          setOpenrouterReasoningEffort: setValidatorOpenrouterReasoningEffort,
          fallback: validatorLmStudioFallback,
          setFallback: setValidatorLmStudioFallback,
          contextSize: validatorContextSize,
          setContextSize: setValidatorContextSize,
          maxOutput: validatorMaxOutput,
          setMaxOutput: setValidatorMaxOutput,
          superchargeEnabled: validatorSuperchargeEnabled,
          setSuperchargeEnabled: setValidatorSuperchargeEnabled,
        })}

        {renderRoleConfig({
          title: 'Assistant',
          description: 'Runs in parallel during outline, writing, review, and proof work to retrieve up to 7 relevant verified proof-memory supports from Session History Memory and SyntheticLib4 when enabled. Validators and critique phases do not receive Assistant context.',
          provider: assistantProvider,
          setProvider: setAssistantProvider,
          model: assistantModel || validatorModel,
          setModel: setAssistantModel,
          openrouterProv: assistantOpenrouterProvider,
          setOpenrouterProv: setAssistantOpenrouterProvider,
          openrouterReasoningEffort: assistantOpenrouterReasoningEffort,
          setOpenrouterReasoningEffort: setAssistantOpenrouterReasoningEffort,
          fallback: assistantLmStudioFallback,
          setFallback: setAssistantLmStudioFallback,
          contextSize: assistantContextSize,
          setContextSize: setAssistantContextSize,
          maxOutput: assistantMaxOutput,
          setMaxOutput: setAssistantMaxOutput,
          superchargeEnabled: assistantSuperchargeEnabled,
          setSuperchargeEnabled: setAssistantSuperchargeEnabled,
          disabled: !assistantMemoryEnabled,
        })}

        {renderRoleConfig({
          title: 'Writing Submitter',
          description: 'Handles construction, outline creation/updates, and review modes. Needs large context for comprehensive outlines.',
          provider: writerProvider,
          setProvider: setWritingProvider,
          model: writerModel,
          setModel: setWritingModel,
          openrouterProv: writerOpenrouterProvider,
          setOpenrouterProv: setWritingOpenrouterProvider,
          openrouterReasoningEffort: writerOpenrouterReasoningEffort,
          setOpenrouterReasoningEffort: setWritingOpenrouterReasoningEffort,
          fallback: writerLmStudioFallback,
          setFallback: setWritingLmStudioFallback,
          contextSize: writerContextSize,
          setContextSize: setWritingContextSize,
          maxOutput: writerMaxOutput,
          setMaxOutput: setWritingMaxOutput,
          superchargeEnabled: writerSuperchargeEnabled,
          setSuperchargeEnabled: setWritingSuperchargeEnabled,
        })}

        {renderRoleConfig({
          title: 'Rigor & Proofs Submitter',
          description: 'Handles Lean/proof work, rigor theorem discovery and placement, and post-body critique generation.',
          provider: highParamProvider,
          setProvider: setHighParamProvider,
          model: highParamModel,
          setModel: setHighParamModel,
          openrouterProv: highParamOpenrouterProvider,
          setOpenrouterProv: setHighParamOpenrouterProvider,
          openrouterReasoningEffort: highParamOpenrouterReasoningEffort,
          setOpenrouterReasoningEffort: setHighParamOpenrouterReasoningEffort,
          fallback: highParamLmStudioFallback,
          setFallback: setHighParamLmStudioFallback,
          contextSize: highParamContextSize,
          setContextSize: setHighParamContextSize,
          maxOutput: highParamMaxOutput,
          setMaxOutput: setHighParamMaxOutput,
          superchargeEnabled: highParamSuperchargeEnabled,
          setSuperchargeEnabled: setHighParamSuperchargeEnabled,
          showProofStrengthBadge: true,
        })}

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

      <div className="settings-group">
        <h4>Workflow Configuration</h4>
        
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
      <div className="settings-group">
        <div 
          onClick={() => setCritiquePromptExpanded(!critiquePromptExpanded)}
          className="collapsible-trigger settings-trigger--multiline"
        >
          <div className="settings-trigger-copy">
            <div className="settings-trigger-title-row">
              <h4 className="form-group--compact settings-trigger-title">Edit Validator Critique Prompt</h4>
              {isUsingCustomCritiquePrompt && (
                <span className="tag-badge tag-badge--purple">CUSTOM</span>
              )}
            </div>
            <p className="settings-subsection-description">
              Optional prompt customization for the user-facing paper critique mode.
            </p>
          </div>
          <span className={`collapse-chevron${critiquePromptExpanded ? ' collapse-chevron--open' : ''}`}>▼</span>
        </div>

        {critiquePromptExpanded && (
          <div className="collapsible-body" style={{ marginTop: '1rem' }}>
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
                <span className={`status-success-text status-success-text--reserved${critiquePromptSaved ? '' : ' status-success-text--hidden'}`}>
                  ✓ Saved!
                </span>
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
      <div className="settings-group">
        <h4>Current Configuration Summary</h4>
        <pre className="config-summary-pre">
          {JSON.stringify({
            validator: {
              provider: validatorProvider,
              model: validatorModel?.split('/').pop() || 'Not selected',
              host: validatorProvider === 'openrouter' ? (validatorOpenrouterProvider || 'Auto') : 'N/A',
              fallback: validatorProvider === 'openrouter' ? (validatorLmStudioFallback?.split('/').pop() || 'None') : 'N/A',
              context: validatorContextSize,
              maxOutput: validatorMaxOutput,
              supercharge: validatorSuperchargeEnabled
            },
            writer: {
              provider: writerProvider,
              model: writerModel?.split('/').pop() || 'Not selected',
              host: writerProvider === 'openrouter' ? (writerOpenrouterProvider || 'Auto') : 'N/A',
              fallback: writerProvider === 'openrouter' ? (writerLmStudioFallback?.split('/').pop() || 'None') : 'N/A',
              context: writerContextSize,
              maxOutput: writerMaxOutput,
              supercharge: writerSuperchargeEnabled
            },
            rigorAndProofs: {
              provider: highParamProvider,
              model: highParamModel?.split('/').pop() || 'Not selected',
              host: highParamProvider === 'openrouter' ? (highParamOpenrouterProvider || 'Auto') : 'N/A',
              fallback: highParamProvider === 'openrouter' ? (highParamLmStudioFallback?.split('/').pop() || 'None') : 'N/A',
              context: highParamContextSize,
              maxOutput: highParamMaxOutput,
              supercharge: highParamSuperchargeEnabled
            }
          }, null, 2)}
        </pre>
      </div>
        </>
      )}
      </div>
    </div>
  );
}

export default CompilerSettings;
