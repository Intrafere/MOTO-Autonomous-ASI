import React, { useState, useEffect } from 'react';
import { openRouterAPI, api, aggregatorAPI, compilerAPI } from '../../services/api';
import {
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
import HelpTooltip from '../HelpTooltip';
import HighlightedModelsSidebar from '../HighlightedModelsSidebar';
import ProofStrengthBadge from '../ProofStrengthBadge';
import RawSettingsEditor from '../RawSettingsEditor';
import '../autonomous/AutonomousResearch.css';
import '../settings-common.css';

const SETTINGS_KEY = 'compiler_settings';
const RAW_VIEW_EXIT_WARNING = 'Switching back to the GUI view will restore your last GUI settings/profile and discard raw-only changes. Continue?';
const formatRawSettings = (value) => JSON.stringify(value, null, 2);
const SUPERCHARGE_TOOLTIP = 'Supercharge makes this role generate 4 full answer attempts, then run a 5th same-model call to choose or synthesize the best final answer. It uses 5x the API calls, so it is about 5x slower and 5x more costly, but can produce more intelligent answers.';

function CompilerSettings({ capabilities, developerModeEnabled = false }) {
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
  const [validatorOpenrouterReasoningEffort, setValidatorOpenrouterReasoningEffort] = useState(DEFAULT_OPENROUTER_REASONING_EFFORT);
  const [validatorLmStudioFallback, setValidatorLmStudioFallback] = useState(null);
  const [validatorContextSize, setValidatorContextSize] = useState(DEFAULT_CONTEXT_WINDOW);
  const [validatorMaxOutput, setValidatorMaxOutput] = useState(DEFAULT_MAX_OUTPUT_TOKENS);
  const [validatorSuperchargeEnabled, setValidatorSuperchargeEnabled] = useState(false);

  // High-Context settings
  const [highContextProvider, setHighContextProvider] = useState('lm_studio');
  const [highContextModel, setHighContextModel] = useState('');
  const [highContextOpenrouterProvider, setHighContextOpenrouterProvider] = useState(null);
  const [highContextOpenrouterReasoningEffort, setHighContextOpenrouterReasoningEffort] = useState(DEFAULT_OPENROUTER_REASONING_EFFORT);
  const [highContextLmStudioFallback, setHighContextLmStudioFallback] = useState(null);
  const [highContextContextSize, setHighContextContextSize] = useState(DEFAULT_CONTEXT_WINDOW);
  const [highContextMaxOutput, setHighContextMaxOutput] = useState(DEFAULT_MAX_OUTPUT_TOKENS);
  const [highContextSuperchargeEnabled, setHighContextSuperchargeEnabled] = useState(false);

  // High-Param settings
  const [highParamProvider, setHighParamProvider] = useState('lm_studio');
  const [highParamModel, setHighParamModel] = useState('');
  const [highParamOpenrouterProvider, setHighParamOpenrouterProvider] = useState(null);
  const [highParamOpenrouterReasoningEffort, setHighParamOpenrouterReasoningEffort] = useState(DEFAULT_OPENROUTER_REASONING_EFFORT);
  const [highParamLmStudioFallback, setHighParamLmStudioFallback] = useState(null);
  const [highParamContextSize, setHighParamContextSize] = useState(DEFAULT_CONTEXT_WINDOW);
  const [highParamMaxOutput, setHighParamMaxOutput] = useState(DEFAULT_MAX_OUTPUT_TOKENS);
  const [highParamSuperchargeEnabled, setHighParamSuperchargeEnabled] = useState(false);

  // Critique Submitter settings
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
          // High-Context
          if (settings.highContextProvider) setHighContextProvider(settings.highContextProvider);
          if (settings.highContextModel) setHighContextModel(settings.highContextModel);
          if (settings.highContextOpenrouterProvider) setHighContextOpenrouterProvider(settings.highContextOpenrouterProvider);
          if (settings.highContextOpenrouterReasoningEffort) setHighContextOpenrouterReasoningEffort(normalizeOpenRouterReasoningEffort(settings.highContextOpenrouterReasoningEffort));
          if (settings.highContextLmStudioFallback) setHighContextLmStudioFallback(settings.highContextLmStudioFallback);
          if (settings.highContextContextSize) setHighContextContextSize(settings.highContextContextSize);
          if (settings.highContextMaxOutput) setHighContextMaxOutput(settings.highContextMaxOutput);
          if (settings.highContextSuperchargeEnabled !== undefined) setHighContextSuperchargeEnabled(settings.highContextSuperchargeEnabled);
          // High-Param
          if (settings.highParamProvider) setHighParamProvider(settings.highParamProvider);
          if (settings.highParamModel) setHighParamModel(settings.highParamModel);
          if (settings.highParamOpenrouterProvider) setHighParamOpenrouterProvider(settings.highParamOpenrouterProvider);
          if (settings.highParamOpenrouterReasoningEffort) setHighParamOpenrouterReasoningEffort(normalizeOpenRouterReasoningEffort(settings.highParamOpenrouterReasoningEffort));
          if (settings.highParamLmStudioFallback) setHighParamLmStudioFallback(settings.highParamLmStudioFallback);
          if (settings.highParamContextSize) setHighParamContextSize(settings.highParamContextSize);
          if (settings.highParamMaxOutput) setHighParamMaxOutput(settings.highParamMaxOutput);
          if (settings.highParamSuperchargeEnabled !== undefined) setHighParamSuperchargeEnabled(settings.highParamSuperchargeEnabled);
          // Critique Submitter
          if (settings.critiqueSubmitterProvider) setCritiqueSubmitterProvider(settings.critiqueSubmitterProvider);
          if (settings.critiqueSubmitterModel) setCritiqueSubmitterModel(settings.critiqueSubmitterModel);
          if (settings.critiqueSubmitterOpenrouterProvider) setCritiqueSubmitterOpenrouterProvider(settings.critiqueSubmitterOpenrouterProvider);
          if (settings.critiqueSubmitterOpenrouterReasoningEffort) setCritiqueSubmitterOpenrouterReasoningEffort(normalizeOpenRouterReasoningEffort(settings.critiqueSubmitterOpenrouterReasoningEffort));
          if (settings.critiqueSubmitterLmStudioFallback) setCritiqueSubmitterLmStudioFallback(settings.critiqueSubmitterLmStudioFallback);
          if (settings.critiqueSubmitterContextSize) setCritiqueSubmitterContextSize(settings.critiqueSubmitterContextSize);
          if (settings.critiqueSubmitterMaxOutput) setCritiqueSubmitterMaxOutput(settings.critiqueSubmitterMaxOutput);
          if (settings.critiqueSubmitterSuperchargeEnabled !== undefined) setCritiqueSubmitterSuperchargeEnabled(settings.critiqueSubmitterSuperchargeEnabled);
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
      validatorOpenrouterProvider,
      validatorOpenrouterReasoningEffort
    );
    const nextHighContext = normalizeRoleState(
      highContextProvider,
      highContextModel,
      highContextOpenrouterProvider,
      highContextOpenrouterReasoningEffort
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

    if (validatorProvider !== nextValidator.provider) setValidatorProvider(nextValidator.provider);
    if (validatorModel !== nextValidator.model) setValidatorModel(nextValidator.model);
    if (validatorOpenrouterProvider !== nextValidator.openrouterProvider) {
      setValidatorOpenrouterProvider(nextValidator.openrouterProvider);
    }
    if (validatorOpenrouterReasoningEffort !== nextValidator.openrouterReasoningEffort) {
      setValidatorOpenrouterReasoningEffort(nextValidator.openrouterReasoningEffort);
    }
    if (validatorLmStudioFallback !== null) setValidatorLmStudioFallback(null);

    if (highContextProvider !== nextHighContext.provider) setHighContextProvider(nextHighContext.provider);
    if (highContextModel !== nextHighContext.model) setHighContextModel(nextHighContext.model);
    if (highContextOpenrouterProvider !== nextHighContext.openrouterProvider) {
      setHighContextOpenrouterProvider(nextHighContext.openrouterProvider);
    }
    if (highContextOpenrouterReasoningEffort !== nextHighContext.openrouterReasoningEffort) {
      setHighContextOpenrouterReasoningEffort(nextHighContext.openrouterReasoningEffort);
    }
    if (highContextLmStudioFallback !== null) setHighContextLmStudioFallback(null);

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
  }, [
    lmStudioEnabled,
    validatorProvider,
    validatorModel,
    validatorOpenrouterProvider,
    validatorOpenrouterReasoningEffort,
    validatorLmStudioFallback,
    highContextProvider,
    highContextModel,
    highContextOpenrouterProvider,
    highContextOpenrouterReasoningEffort,
    highContextLmStudioFallback,
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
      validatorProvider, validatorModel, validatorOpenrouterProvider, validatorOpenrouterReasoningEffort, validatorLmStudioFallback,
      validatorContextSize, validatorMaxOutput, validatorSuperchargeEnabled,
      highContextProvider, highContextModel, highContextOpenrouterProvider, highContextOpenrouterReasoningEffort, highContextLmStudioFallback,
      highContextContextSize, highContextMaxOutput, highContextSuperchargeEnabled,
      highParamProvider, highParamModel, highParamOpenrouterProvider, highParamOpenrouterReasoningEffort, highParamLmStudioFallback,
      highParamContextSize, highParamMaxOutput, highParamSuperchargeEnabled,
      critiqueSubmitterProvider, critiqueSubmitterModel, critiqueSubmitterOpenrouterProvider, critiqueSubmitterOpenrouterReasoningEffort, critiqueSubmitterLmStudioFallback,
      critiqueSubmitterContextSize, critiqueSubmitterMaxOutput, critiqueSubmitterSuperchargeEnabled,
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
    isLoaded, validatorProvider, validatorModel, validatorOpenrouterProvider, validatorOpenrouterReasoningEffort, validatorLmStudioFallback,
    validatorContextSize, validatorMaxOutput, validatorSuperchargeEnabled,
    highContextProvider, highContextModel, highContextOpenrouterProvider, highContextOpenrouterReasoningEffort, highContextLmStudioFallback,
    highContextContextSize, highContextMaxOutput, highContextSuperchargeEnabled,
    highParamProvider, highParamModel, highParamOpenrouterProvider, highParamOpenrouterReasoningEffort, highParamLmStudioFallback,
    highParamContextSize, highParamMaxOutput, highParamSuperchargeEnabled,
    critiqueSubmitterProvider, critiqueSubmitterModel, critiqueSubmitterOpenrouterProvider, critiqueSubmitterOpenrouterReasoningEffort, critiqueSubmitterLmStudioFallback,
    critiqueSubmitterContextSize, critiqueSubmitterMaxOutput, critiqueSubmitterSuperchargeEnabled,
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
        setValidatorOpenrouterReasoningEffort(DEFAULT_OPENROUTER_REASONING_EFFORT);
        setValidatorLmStudioFallback(null);
        
        setHighContextProvider('lm_studio');
        setHighContextModel(settings.submitter_model);
        setHighContextOpenrouterProvider(null);
        setHighContextOpenrouterReasoningEffort(DEFAULT_OPENROUTER_REASONING_EFFORT);
        setHighContextLmStudioFallback(null);
        
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
    highContextProvider,
    highContextModel,
    highContextOpenrouterProvider,
    highContextOpenrouterReasoningEffort,
    highContextLmStudioFallback,
    highContextContextSize,
    highContextMaxOutput,
    highContextSuperchargeEnabled,
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
    wolframEnabled,
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
    setValidatorContextSize(Number(rawSettings.validatorContextSize || DEFAULT_CONTEXT_WINDOW));
    setValidatorMaxOutput(Number(rawSettings.validatorMaxOutput || DEFAULT_MAX_OUTPUT_TOKENS));
    setValidatorSuperchargeEnabled(Boolean(rawSettings.validatorSuperchargeEnabled));
    setHighContextProvider(rawSettings.highContextProvider || 'lm_studio');
    setHighContextModel(rawSettings.highContextModel || '');
    setHighContextOpenrouterProvider(rawSettings.highContextOpenrouterProvider || null);
    setHighContextOpenrouterReasoningEffort(normalizeOpenRouterReasoningEffort(rawSettings.highContextOpenrouterReasoningEffort));
    setHighContextLmStudioFallback(rawSettings.highContextLmStudioFallback || null);
    setHighContextContextSize(Number(rawSettings.highContextContextSize || DEFAULT_CONTEXT_WINDOW));
    setHighContextMaxOutput(Number(rawSettings.highContextMaxOutput || DEFAULT_MAX_OUTPUT_TOKENS));
    setHighContextSuperchargeEnabled(Boolean(rawSettings.highContextSuperchargeEnabled));
    setHighParamProvider(rawSettings.highParamProvider || 'lm_studio');
    setHighParamModel(rawSettings.highParamModel || '');
    setHighParamOpenrouterProvider(rawSettings.highParamOpenrouterProvider || null);
    setHighParamOpenrouterReasoningEffort(normalizeOpenRouterReasoningEffort(rawSettings.highParamOpenrouterReasoningEffort));
    setHighParamLmStudioFallback(rawSettings.highParamLmStudioFallback || null);
    setHighParamContextSize(Number(rawSettings.highParamContextSize || DEFAULT_CONTEXT_WINDOW));
    setHighParamMaxOutput(Number(rawSettings.highParamMaxOutput || DEFAULT_MAX_OUTPUT_TOKENS));
    setHighParamSuperchargeEnabled(Boolean(rawSettings.highParamSuperchargeEnabled));
    setCritiqueSubmitterProvider(rawSettings.critiqueSubmitterProvider || 'lm_studio');
    setCritiqueSubmitterModel(rawSettings.critiqueSubmitterModel || '');
    setCritiqueSubmitterOpenrouterProvider(rawSettings.critiqueSubmitterOpenrouterProvider || null);
    setCritiqueSubmitterOpenrouterReasoningEffort(normalizeOpenRouterReasoningEffort(rawSettings.critiqueSubmitterOpenrouterReasoningEffort));
    setCritiqueSubmitterLmStudioFallback(rawSettings.critiqueSubmitterLmStudioFallback || null);
    setCritiqueSubmitterContextSize(Number(rawSettings.critiqueSubmitterContextSize || DEFAULT_CONTEXT_WINDOW));
    setCritiqueSubmitterMaxOutput(Number(rawSettings.critiqueSubmitterMaxOutput || DEFAULT_MAX_OUTPUT_TOKENS));
    setCritiqueSubmitterSuperchargeEnabled(Boolean(rawSettings.critiqueSubmitterSuperchargeEnabled));
    setWolframEnabled(rawSettings.wolframEnabled ?? false);
    setFreeOnly(rawSettings.freeOnly ?? false);
    setFreeModelLooping(rawSettings.freeModelLooping ?? true);
    setFreeModelAutoSelector(rawSettings.freeModelAutoSelector ?? true);
    setModelProviders(rawSettings.modelProviders || {});

    if (updateRawText) {
      setRawSettingsText(formatRawSettings({
        ...rawSettings,
        validatorProvider: rawSettings.validatorProvider || 'lm_studio',
        validatorModel: rawSettings.validatorModel || '',
        validatorOpenrouterReasoningEffort: normalizeOpenRouterReasoningEffort(rawSettings.validatorOpenrouterReasoningEffort),
        highContextProvider: rawSettings.highContextProvider || 'lm_studio',
        highContextModel: rawSettings.highContextModel || '',
        highContextOpenrouterReasoningEffort: normalizeOpenRouterReasoningEffort(rawSettings.highContextOpenrouterReasoningEffort),
        highParamProvider: rawSettings.highParamProvider || 'lm_studio',
        highParamModel: rawSettings.highParamModel || '',
        highParamOpenrouterReasoningEffort: normalizeOpenRouterReasoningEffort(rawSettings.highParamOpenrouterReasoningEffort),
        critiqueSubmitterProvider: rawSettings.critiqueSubmitterProvider || 'lm_studio',
        critiqueSubmitterModel: rawSettings.critiqueSubmitterModel || '',
        critiqueSubmitterOpenrouterReasoningEffort: normalizeOpenRouterReasoningEffort(rawSettings.critiqueSubmitterOpenrouterReasoningEffort),
        wolframEnabled: rawSettings.wolframEnabled ?? false,
        freeOnly: rawSettings.freeOnly ?? false,
        freeModelLooping: rawSettings.freeModelLooping ?? true,
        freeModelAutoSelector: rawSettings.freeModelAutoSelector ?? true,
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

  // Reusable Role Configuration Component
  const RoleConfig = ({ 
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
    borderColor = '#333',
    showProofStrengthBadge = false
  }) => {
    const effectiveProvider = lmStudioEnabled ? provider : 'openrouter';
    const models = effectiveProvider === 'openrouter' ? openRouterModels : lmStudioModels;
    const providers = model && effectiveProvider === 'openrouter'
      ? getProviderNames(modelProviders[model])
      : [];
    const reasoningInfo = effectiveProvider === 'openrouter'
      ? getReasoningSupportInfo(modelProviders[model], openrouterProv || null)
      : { hasEndpointMetadata: false, supportsReasoning: false };

    return (
      <div
        className={`submitter-config-section${effectiveProvider === 'openrouter' ? ' role-config-card--openrouter-orange' : ''}`}
        style={{ borderColor: effectiveProvider === 'openrouter' ? undefined : borderColor }}
      >
        <h5 className={effectiveProvider === 'openrouter' ? 'card-title--orange' : ''} style={effectiveProvider === 'openrouter' ? undefined : { color: borderColor }}>
          <span className="role-title-with-badges">
            <span>{title}</span>
            {showProofStrengthBadge && <ProofStrengthBadge />}
          </span>
          {effectiveProvider === 'openrouter' && <span className="provider-badge-inline">[OpenRouter]</span>}
        </h5>
        <p className="settings-hint">{description}</p>

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
          <div className="settings-row">
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

        {/* LM Studio Fallback (if OpenRouter) */}
        {effectiveProvider === 'openrouter' && lmStudioEnabled && (
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
            <small className="settings-hint">Used if OpenRouter credits run out</small>
          </div>
        )}

        <div className="settings-row">
          <label>Context Window</label>
          <input
            type="number"
            value={contextSize}
            onChange={(e) => {
              const parsed = parseInt(e.target.value, 10);
              setContextSize(isNaN(parsed) ? DEFAULT_CONTEXT_WINDOW : parsed);
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
              setMaxOutput(isNaN(parsed) ? DEFAULT_MAX_OUTPUT_TOKENS : parsed);
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
            <label className="settings-checkbox-label model-refresh-controls__toggle">
              <input
                type="checkbox"
                checked={freeOnly}
                onChange={(e) => setFreeOnly(e.target.checked)}
              />
              Free models only
            </label>
          </>
        )}
        {developerModeEnabled ? (
          <label className="settings-checkbox-label model-refresh-controls__toggle">
            <input
              type="checkbox"
              checked={editRawSettings}
              onChange={(e) => handleRawEditToggle(e.target.checked)}
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
        />
      ) : (
        <>
      <div className="settings-group">
        <h4>Model Configuration</h4>
        <p className="settings-info">
          Configure the validator and compiler roles used by manual paper compilation.
        </p>
        
        <RoleConfig
          title="Validator"
          description="Validates all submissions for coherence, rigor, placement, and non-redundancy."
          borderColor="#ff6b6b"
          provider={validatorProvider} setProvider={setValidatorProvider}
          model={validatorModel} setModel={setValidatorModel}
          openrouterProv={validatorOpenrouterProvider} setOpenrouterProv={setValidatorOpenrouterProvider}
          openrouterReasoningEffort={validatorOpenrouterReasoningEffort} setOpenrouterReasoningEffort={setValidatorOpenrouterReasoningEffort}
          fallback={validatorLmStudioFallback} setFallback={setValidatorLmStudioFallback}
          contextSize={validatorContextSize} setContextSize={setValidatorContextSize}
          maxOutput={validatorMaxOutput} setMaxOutput={setValidatorMaxOutput}
          superchargeEnabled={validatorSuperchargeEnabled} setSuperchargeEnabled={setValidatorSuperchargeEnabled}
        />

        <RoleConfig
          title="High-Context Model"
          description="Handles construction, outline creation/updates, and review modes. Needs large context for comprehensive outlines."
          borderColor="#4CAF50"
          provider={highContextProvider} setProvider={setHighContextProvider}
          model={highContextModel} setModel={setHighContextModel}
          openrouterProv={highContextOpenrouterProvider} setOpenrouterProv={setHighContextOpenrouterProvider}
          openrouterReasoningEffort={highContextOpenrouterReasoningEffort} setOpenrouterReasoningEffort={setHighContextOpenrouterReasoningEffort}
          fallback={highContextLmStudioFallback} setFallback={setHighContextLmStudioFallback}
          contextSize={highContextContextSize} setContextSize={setHighContextContextSize}
          maxOutput={highContextMaxOutput} setMaxOutput={setHighContextMaxOutput}
          superchargeEnabled={highContextSuperchargeEnabled} setSuperchargeEnabled={setHighContextSuperchargeEnabled}
          showProofStrengthBadge
        />

        <RoleConfig
          title="High-Parameter Model"
          description="Rigor enhancement mode: adds citations, strengthens methodology, and clarifies assumptions."
          borderColor="#2a2a2a"
          provider={highParamProvider} setProvider={setHighParamProvider}
          model={highParamModel} setModel={setHighParamModel}
          openrouterProv={highParamOpenrouterProvider} setOpenrouterProv={setHighParamOpenrouterProvider}
          openrouterReasoningEffort={highParamOpenrouterReasoningEffort} setOpenrouterReasoningEffort={setHighParamOpenrouterReasoningEffort}
          fallback={highParamLmStudioFallback} setFallback={setHighParamLmStudioFallback}
          contextSize={highParamContextSize} setContextSize={setHighParamContextSize}
          maxOutput={highParamMaxOutput} setMaxOutput={setHighParamMaxOutput}
          superchargeEnabled={highParamSuperchargeEnabled} setSuperchargeEnabled={setHighParamSuperchargeEnabled}
          showProofStrengthBadge
        />

        <RoleConfig
          title="Critique Submitter"
          description="Generates validated peer review critiques for the paper's AI self-review section."
          borderColor="#e74c3c"
          provider={critiqueSubmitterProvider} setProvider={setCritiqueSubmitterProvider}
          model={critiqueSubmitterModel} setModel={setCritiqueSubmitterModel}
          openrouterProv={critiqueSubmitterOpenrouterProvider} setOpenrouterProv={setCritiqueSubmitterOpenrouterProvider}
          openrouterReasoningEffort={critiqueSubmitterOpenrouterReasoningEffort} setOpenrouterReasoningEffort={setCritiqueSubmitterOpenrouterReasoningEffort}
          fallback={critiqueSubmitterLmStudioFallback} setFallback={setCritiqueSubmitterLmStudioFallback}
          contextSize={critiqueSubmitterContextSize} setContextSize={setCritiqueSubmitterContextSize}
          maxOutput={critiqueSubmitterMaxOutput} setMaxOutput={setCritiqueSubmitterMaxOutput}
          superchargeEnabled={critiqueSubmitterSuperchargeEnabled} setSuperchargeEnabled={setCritiqueSubmitterSuperchargeEnabled}
        />
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

      {/* Wolfram Alpha Integration */}
      <div className="settings-group">
        <h4>Wolfram Alpha Integration (Optional)</h4>
        <p className="settings-info">
          Enable Wolfram Alpha API for computational verification in rigor mode. 
          Get your API key from <a href="https://products.wolframalpha.com/api" target="_blank" rel="noopener noreferrer">developer.wolframalpha.com</a>
        </p>
        
        <label className="settings-checkbox-label settings-checkbox-label--stacked" style={{ marginBottom: '1rem' }}>
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
            highContext: {
              provider: highContextProvider,
              model: highContextModel?.split('/').pop() || 'Not selected',
              host: highContextProvider === 'openrouter' ? (highContextOpenrouterProvider || 'Auto') : 'N/A',
              fallback: highContextProvider === 'openrouter' ? (highContextLmStudioFallback?.split('/').pop() || 'None') : 'N/A',
              context: highContextContextSize,
              maxOutput: highContextMaxOutput,
              supercharge: highContextSuperchargeEnabled
            },
            highParam: {
              provider: highParamProvider,
              model: highParamModel?.split('/').pop() || 'Not selected',
              host: highParamProvider === 'openrouter' ? (highParamOpenrouterProvider || 'Auto') : 'N/A',
              fallback: highParamProvider === 'openrouter' ? (highParamLmStudioFallback?.split('/').pop() || 'None') : 'N/A',
              context: highParamContextSize,
              maxOutput: highParamMaxOutput,
              supercharge: highParamSuperchargeEnabled
            },
            critiqueSubmitter: {
              provider: critiqueSubmitterProvider,
              model: critiqueSubmitterModel?.split('/').pop() || 'Not selected',
              host: critiqueSubmitterProvider === 'openrouter' ? (critiqueSubmitterOpenrouterProvider || 'Auto') : 'N/A',
              fallback: critiqueSubmitterProvider === 'openrouter' ? (critiqueSubmitterLmStudioFallback?.split('/').pop() || 'None') : 'N/A',
              context: critiqueSubmitterContextSize,
              maxOutput: critiqueSubmitterMaxOutput,
              supercharge: critiqueSubmitterSuperchargeEnabled
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
