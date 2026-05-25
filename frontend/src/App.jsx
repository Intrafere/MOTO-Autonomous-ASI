import React, { useState, useEffect, useRef, useCallback } from 'react';
import AggregatorInterface from './components/aggregator/AggregatorInterface';
import AggregatorSettings from './components/aggregator/AggregatorSettings';
import AggregatorLogs from './components/aggregator/AggregatorLogs';
import LiveResults from './components/aggregator/LiveResults';
import CompilerInterface from './components/compiler/CompilerInterface';
import CompilerSettings from './components/compiler/CompilerSettings';
import CompilerLogs from './components/compiler/CompilerLogs';
import LivePaper from './components/compiler/LivePaper';
import {
  AutonomousResearchInterface,
  BrainstormList,
  PaperLibrary,
  Stage2PaperHistory,
  AutonomousResearchSettings,
  AutonomousResearchLogs,
  FinalAnswerView,
  FinalAnswerLibrary,
  MathematicalProofs,
  ProofLibrary
} from './components/autonomous';
import {
  LeanOJBrainstorms,
  LeanOJInterface,
  LeanOJLogs,
  LeanOJMasterProof,
  LeanOJMathematicalProofs,
  LeanOJProofLibrary,
  LeanOJSettings,
} from './components/leanoj';
import WorkflowPanel from './components/WorkflowPanel';
import BoostControlModal from './components/BoostControlModal';
import StartupProviderSetupModal from './components/StartupProviderSetupModal';
import OpenRouterApiKeyModal from './components/OpenRouterApiKeyModal';
import OpenRouterPrivacyWarningModal from './components/OpenRouterPrivacyWarningModal';
import CritiqueNotificationStack from './components/CritiqueNotificationStack';
import ProofNotificationStack from './components/autonomous/ProofNotificationStack';
import CreditExhaustionNotificationStack from './components/CreditExhaustionNotificationStack';
import UpdateNotificationBanner from './components/UpdateNotificationBanner';
import PaperCritiqueModal from './components/PaperCritiqueModal';
import { websocket } from './services/websocket';
import { api, autonomousAPI, cloudAccessAPI, leanojAPI, openRouterAPI } from './services/api';
import {
  LM_STUDIO_STARTUP_CHOICE,
  RECOMMENDED_PROFILE_KEY,
  STARTUP_PROVIDER_CHOICE_STORAGE_KEY,
  applyAutonomousProfileSelection,
  applyLmStudioStartupDefaults,
  getStoredAutonomousSettings,
  settingsToAutonomousConfig,
  persistAutonomousSettings,
} from './utils/autonomousProfiles';
import {
  getStoredLeanOJSettings,
  persistLeanOJSettings,
} from './utils/leanojProfiles';
import {
  DEFAULT_CONTEXT_WINDOW,
  DEFAULT_MAX_OUTPUT_TOKENS,
} from './utils/openRouterSelection';

const DEVELOPER_MODE_STORAGE_KEY = 'developerModeSettingsEnabled';
const DEPRECATED_SCREEN_STATE_STORAGE_KEYS = [
  'appMode',
  'singlePaperWriterExpanded',
  'autonomousActiveTab',
  'completedWorksSubTab',
  'manualActiveTab',
  'leanojActiveTab',
];
const EMBEDDING_MODEL_HINTS = ['embed', 'embedding', 'nomic', 'bge', 'e5', 'gte'];
const AUTONOMOUS_ROLE_PREFIXES = ['validator', 'high_context', 'high_param', 'critique_submitter'];
const HIGH_SCORE_CRITIQUE_THRESHOLD = 6.25;
const SEEN_HIGH_SCORE_CRITIQUES_STORAGE_KEY = 'seenHighScoreCritiqueNotifications';
const MAX_SEEN_HIGH_SCORE_CRITIQUES = 500;
const MAX_LIVE_ACTIVITY_EVENTS = 5000;
const MAX_PROOF_NOTIFICATIONS = 20;
const UPDATE_NOTICE_POLL_INTERVAL_MS = 4 * 60 * 60 * 1000;
const DEFAULT_CAPABILITIES = Object.freeze({
  genericMode: false,
  lmStudioEnabled: true,
  pdfDownloadAvailable: true,
  version: '',
  buildCommit: '',
  updateChannel: 'main',
  apiContractVersion: '',
});

function readDeveloperModeEnabled() {
  return localStorage.getItem(DEVELOPER_MODE_STORAGE_KEY) === 'true';
}

function getHighScoreCritiqueNotificationKey(paperId, averageRating) {
  const rating = Number(averageRating);
  if (!paperId || !Number.isFinite(rating)) {
    return null;
  }
  return `${paperId}:${rating.toFixed(1)}`;
}

function readSeenHighScoreCritiques() {
  if (typeof window === 'undefined') {
    return new Set();
  }

  try {
    const raw = window.localStorage.getItem(SEEN_HIGH_SCORE_CRITIQUES_STORAGE_KEY);
    const values = raw ? JSON.parse(raw) : [];
    return new Set(Array.isArray(values) ? values.filter(value => typeof value === 'string') : []);
  } catch (error) {
    console.warn('Could not read seen high-score critique notifications:', error);
    return new Set();
  }
}

function persistSeenHighScoreCritiques(seenSet) {
  if (typeof window === 'undefined') {
    return;
  }

  try {
    const values = Array.from(seenSet).slice(-MAX_SEEN_HIGH_SCORE_CRITIQUES);
    window.localStorage.setItem(SEEN_HIGH_SCORE_CRITIQUES_STORAGE_KEY, JSON.stringify(values));
  } catch (error) {
    console.warn('Could not save seen high-score critique notifications:', error);
  }
}

const createDefaultAggregatorSubmitterConfigs = () => (
  [1, 2, 3].map((submitterId) => ({
    submitterId,
    provider: 'lm_studio',
    modelId: '',
    openrouterProvider: null,
    openrouterReasoningEffort: 'auto',
    lmStudioFallbackId: null,
    contextWindow: DEFAULT_CONTEXT_WINDOW,
    maxOutputTokens: DEFAULT_MAX_OUTPUT_TOKENS,
    superchargeEnabled: false,
  }))
);

function normalizeLoadedLmStudioModelId(modelId = '') {
  return String(modelId).replace(/:\d+$/, '');
}

function isLikelyEmbeddingModel(modelId = '') {
  const normalizedModelId = normalizeLoadedLmStudioModelId(modelId).toLowerCase();
  return EMBEDDING_MODEL_HINTS.some((hint) => normalizedModelId.includes(hint));
}

function getUsableLoadedLmStudioChatModelId(loadedModels = []) {
  for (const loadedModelId of loadedModels) {
    const normalizedModelId = normalizeLoadedLmStudioModelId(loadedModelId);
    if (!normalizedModelId || isLikelyEmbeddingModel(normalizedModelId)) {
      continue;
    }
    return normalizedModelId;
  }

  return '';
}

function normalizeFeaturesPayload(payload = {}) {
  return {
    genericMode: Boolean(payload.generic_mode),
    lmStudioEnabled: payload.lm_studio_enabled !== false,
    pdfDownloadAvailable: payload.pdf_download_available !== false,
    version: payload.version || '',
    buildCommit: payload.build_commit || '',
    updateChannel: payload.update_channel || 'main',
    apiContractVersion: payload.api_contract_version || '',
  };
}

function normalizeRuntimeProvider(provider, lmStudioEnabled) {
  return lmStudioEnabled ? (provider || 'lm_studio') : 'openrouter';
}

function normalizeRuntimeModelConfig(config = {}, lmStudioEnabled) {
  const originalProvider = config.provider || 'lm_studio';
  const shouldResetLmState = !lmStudioEnabled && originalProvider !== 'openrouter';

  return {
    ...config,
    provider: normalizeRuntimeProvider(config.provider, lmStudioEnabled),
    modelId: shouldResetLmState ? '' : (config.modelId || ''),
    openrouterProvider: shouldResetLmState ? null : (config.openrouterProvider || null),
    openrouterReasoningEffort: config.openrouterReasoningEffort || 'auto',
    lmStudioFallbackId: lmStudioEnabled ? (config.lmStudioFallbackId || null) : null,
  };
}

function normalizeAggregatorConfigForCapabilities(config, lmStudioEnabled) {
  const originalValidatorProvider = config.validatorProvider || 'lm_studio';
  const shouldResetValidator = !lmStudioEnabled && originalValidatorProvider !== 'openrouter';

  return {
    ...config,
    submitterConfigs: (config.submitterConfigs || []).map((submitterConfig) =>
      normalizeRuntimeModelConfig(submitterConfig, lmStudioEnabled)
    ),
    validatorProvider: normalizeRuntimeProvider(config.validatorProvider, lmStudioEnabled),
    validatorModel: shouldResetValidator ? '' : (config.validatorModel || ''),
    validatorOpenrouterProvider: shouldResetValidator
      ? null
      : (config.validatorOpenrouterProvider || null),
    validatorOpenrouterReasoningEffort: config.validatorOpenrouterReasoningEffort || 'auto',
    validatorLmStudioFallback: lmStudioEnabled ? (config.validatorLmStudioFallback || null) : null,
  };
}

function normalizeAutonomousConfigForCapabilities(config, lmStudioEnabled) {
  const nextConfig = {
    ...config,
    submitter_configs: (config.submitter_configs || []).map((submitterConfig) =>
      normalizeRuntimeModelConfig(submitterConfig, lmStudioEnabled)
    ),
  };

  AUTONOMOUS_ROLE_PREFIXES.forEach((rolePrefix) => {
    const providerKey = `${rolePrefix}_provider`;
    const modelKey = `${rolePrefix}_model`;
    const openRouterProviderKey = `${rolePrefix}_openrouter_provider`;
    const fallbackKey = `${rolePrefix}_lm_studio_fallback`;
    const originalProvider = nextConfig[providerKey] || 'lm_studio';
    const shouldResetRole = !lmStudioEnabled && originalProvider !== 'openrouter';

    nextConfig[providerKey] = normalizeRuntimeProvider(nextConfig[providerKey], lmStudioEnabled);
    nextConfig[modelKey] = shouldResetRole ? '' : (nextConfig[modelKey] || '');
    nextConfig[openRouterProviderKey] = shouldResetRole
      ? null
      : (nextConfig[openRouterProviderKey] || null);
    nextConfig[`${rolePrefix}_openrouter_reasoning_effort`] = nextConfig[`${rolePrefix}_openrouter_reasoning_effort`] || 'auto';
    nextConfig[fallbackKey] = lmStudioEnabled ? (nextConfig[fallbackKey] || null) : null;
  });

  return nextConfig;
}

function App() {
  const [appMode, setAppMode] = useState('autonomous');
  const [autonomousActiveTab, setAutonomousActiveTab] = useState('auto-interface');
  const [manualActiveTab, setManualActiveTab] = useState('aggregator-interface');
  const [leanojActiveTab, setLeanojActiveTab] = useState('leanoj-interface');
  const [completedWorksSubTab, setCompletedWorksSubTab] = useState('stage2-history');
  const activeTab = appMode === 'manual'
    ? manualActiveTab
    : appMode === 'leanoj'
      ? leanojActiveTab
      : autonomousActiveTab;
  const shimmerAccentsEnabled = (() => {
    const saved = localStorage.getItem('banner_shimmer_enabled');
    return saved !== null ? JSON.parse(saved) : true;
  })();
  
  // Models list (fetched from API)
  const [models, setModels] = useState([]);
  
  // Boost modal state
  const [showBoostModal, setShowBoostModal] = useState(false);
  const [showApiBoostTooltip, setShowApiBoostTooltip] = useState(false);
  
  // OpenRouter API Key modal state
  const [showOpenRouterKeyModal, setShowOpenRouterKeyModal] = useState(false);
  const [openRouterKeyReason, setOpenRouterKeyReason] = useState('setup');
  
  // LM Studio availability state (for determining default provider)
  const [lmStudioAvailable, setLmStudioAvailable] = useState(true);
  const [lmStudioStatus, setLmStudioStatus] = useState({
    available: true,
    has_models: false,
    model_count: 0,
    models: [],
    error: null,
    usable_chat_model_id: '',
    has_usable_chat_model: false,
  });
  // Tri-state: null = unknown (backend unreachable / cold-start in progress),
  // true = key stored in backend, false = confirmed no key. The UI treats
  // "unknown" neutrally (does NOT open the startup setup modal, does NOT flash
  // a red "Set OpenRouter Key" chip) so that a slow-to-boot backend can never
  // make a stored key look like it "disappeared".
  const [hasOpenRouterKey, setHasOpenRouterKey] = useState(null);
  const [hasCloudAccess, setHasCloudAccess] = useState(null);
  const [capabilities, setCapabilities] = useState(DEFAULT_CAPABILITIES);
  
  // Track if any workflow is running (for WorkflowPanel visibility)
  const [anyWorkflowRunning, setAnyWorkflowRunning] = useState(false);
  
  // Track WorkflowPanel collapse state for sliding boost buttons
  const [workflowPanelCollapsed, setWorkflowPanelCollapsed] = useState(() => {
    const savedState = localStorage.getItem('workflow_panel_collapsed');
    return savedState !== 'false';
  });
  const [developerModeEnabled, setDeveloperModeEnabled] = useState(() => {
    return readDeveloperModeEnabled();
  });

  // Update notice banner state (dismissible per session, re-appears on restart)
  const [updateNotice, setUpdateNotice] = useState(null);
  const [updateNoticeDismissed, setUpdateNoticeDismissed] = useState(false);

  useEffect(() => {
    DEPRECATED_SCREEN_STATE_STORAGE_KEYS.forEach((key) => {
      localStorage.removeItem(key);
    });
  }, []);

  useEffect(() => {
    if (!developerModeEnabled && appMode === 'leanoj') {
      setAppMode('autonomous');
    }
  }, [developerModeEnabled, appMode]);

  useEffect(() => {
    const pressedCodes = new Set();
    let shortcutChordActive = false;

    const toggleDeveloperMode = () => {
      setDeveloperModeEnabled((currentValue) => {
        const nextValue = !currentValue;
        localStorage.setItem(DEVELOPER_MODE_STORAGE_KEY, String(nextValue));
        return nextValue;
      });
    };

    const getShortcutCode = (event) => {
      if (event.code?.startsWith('Shift') || event.key === 'Shift') {
        return 'Shift';
      }
      if (event.code === 'KeyZ' || event.key?.toLowerCase() === 'z') {
        return 'KeyZ';
      }
      if (event.code === 'KeyX' || event.key?.toLowerCase() === 'x') {
        return 'KeyX';
      }
      return null;
    };

    const hasDeveloperShortcutChord = () => (
      pressedCodes.has('Shift') &&
      pressedCodes.has('KeyZ') &&
      pressedCodes.has('KeyX')
    );

    const handleKeyDown = (event) => {
      const shortcutCode = getShortcutCode(event);
      if (!shortcutCode) {
        return;
      }

      pressedCodes.add(shortcutCode);
      if (hasDeveloperShortcutChord() && !shortcutChordActive) {
        shortcutChordActive = true;
        event.preventDefault();
        toggleDeveloperMode();
      }
    };

    const handleKeyUp = (event) => {
      const shortcutCode = getShortcutCode(event);
      if (shortcutCode) {
        pressedCodes.delete(shortcutCode);
      }
      if (!hasDeveloperShortcutChord()) {
        shortcutChordActive = false;
      }
    };

    const clearPressedCodes = () => {
      pressedCodes.clear();
      shortcutChordActive = false;
    };

    window.addEventListener('keydown', handleKeyDown, true);
    window.addEventListener('keyup', handleKeyUp, true);
    window.addEventListener('blur', clearPressedCodes);

    return () => {
      window.removeEventListener('keydown', handleKeyDown, true);
      window.removeEventListener('keyup', handleKeyUp, true);
      window.removeEventListener('blur', clearPressedCodes);
    };
  }, []);
  
  // Initialize config from localStorage or use defaults
  // CRITICAL: Read from 'aggregator_settings' (used by AggregatorSettings component)
  const [config, setConfig] = useState(() => {
    // Try to load from the settings component key first
    const settingsConfig = localStorage.getItem('aggregator_settings');
    if (settingsConfig) {
      try {
        const settings = JSON.parse(settingsConfig);
        return {
          userPrompt: settings.userPrompt || '',
          submitterConfigs: settings.submitterConfigs || createDefaultAggregatorSubmitterConfigs(),
          validatorModel: settings.validatorModel || '',
          validatorProvider: settings.validatorProvider || 'lm_studio',
          validatorOpenrouterProvider: settings.validatorOpenrouterProvider || null,
          validatorOpenrouterReasoningEffort: settings.validatorOpenrouterReasoningEffort || 'auto',
          validatorLmStudioFallback: settings.validatorLmStudioFallback || null,
          validatorContextSize: settings.validatorContextSize ?? DEFAULT_CONTEXT_WINDOW,
          validatorMaxOutput: settings.validatorMaxOutput ?? DEFAULT_MAX_OUTPUT_TOKENS,
          validatorSuperchargeEnabled: Boolean(settings.validatorSuperchargeEnabled),
          creativityEmphasisBoostEnabled: Boolean(settings.creativityEmphasisBoostEnabled),
          uploadedFiles: [],
        };
      } catch (e) {
        console.error('Failed to parse aggregator_settings:', e);
      }
    }
    
    // Fallback to old key for backward compatibility
    const savedConfig = localStorage.getItem('aggregatorConfig');
    if (savedConfig) {
      try {
        const parsed = JSON.parse(savedConfig);
        return {
          userPrompt: parsed.userPrompt || '',
          submitterConfigs: parsed.submitterConfigs || createDefaultAggregatorSubmitterConfigs(),
          validatorModel: parsed.validatorModel || '',
          validatorProvider: parsed.validatorProvider || 'lm_studio',
          validatorOpenrouterProvider: parsed.validatorOpenrouterProvider || null,
          validatorOpenrouterReasoningEffort: parsed.validatorOpenrouterReasoningEffort || 'auto',
          validatorLmStudioFallback: parsed.validatorLmStudioFallback || null,
          validatorContextSize: parsed.validatorContextSize ?? DEFAULT_CONTEXT_WINDOW,
          validatorMaxOutput: parsed.validatorMaxOutput ?? DEFAULT_MAX_OUTPUT_TOKENS,
          validatorSuperchargeEnabled: Boolean(parsed.validatorSuperchargeEnabled),
          creativityEmphasisBoostEnabled: Boolean(parsed.creativityEmphasisBoostEnabled),
          uploadedFiles: [],
        };
      } catch (e) {
        console.error('Failed to parse saved config:', e);
      }
    }
    return {
      userPrompt: '',
      submitterConfigs: createDefaultAggregatorSubmitterConfigs(),
      validatorModel: '',
      validatorProvider: 'lm_studio',
      validatorOpenrouterProvider: null,
      validatorOpenrouterReasoningEffort: 'auto',
      validatorLmStudioFallback: null,
      validatorContextSize: DEFAULT_CONTEXT_WINDOW,
      validatorMaxOutput: DEFAULT_MAX_OUTPUT_TOKENS,
      validatorSuperchargeEnabled: false,
      creativityEmphasisBoostEnabled: false,
      uploadedFiles: [],
    };
  });

  // Save config to localStorage whenever it changes (excluding transient data)
  // CRITICAL: Save to BOTH keys to maintain backward compatibility
  useEffect(() => {
    const configToSave = {
      userPrompt: config.userPrompt,
      submitterConfigs: config.submitterConfigs,
      validatorModel: config.validatorModel,
      validatorProvider: config.validatorProvider,
      validatorOpenrouterProvider: config.validatorOpenrouterProvider,
      validatorOpenrouterReasoningEffort: config.validatorOpenrouterReasoningEffort,
      validatorLmStudioFallback: config.validatorLmStudioFallback,
      validatorContextSize: config.validatorContextSize,
      validatorMaxOutput: config.validatorMaxOutput,
      validatorSuperchargeEnabled: config.validatorSuperchargeEnabled,
      creativityEmphasisBoostEnabled: config.creativityEmphasisBoostEnabled,
    };
    // Save to both old and new keys
    localStorage.setItem('aggregatorConfig', JSON.stringify(configToSave));
    localStorage.setItem('aggregator_settings', JSON.stringify(configToSave));
  }, [config.userPrompt, config.submitterConfigs, config.validatorModel, config.validatorProvider, config.validatorOpenrouterProvider, config.validatorOpenrouterReasoningEffort, config.validatorLmStudioFallback, config.validatorContextSize, config.validatorMaxOutput, config.validatorSuperchargeEnabled, config.creativityEmphasisBoostEnabled]);

  // Autonomous mode state
  const [autonomousRunning, setAutonomousRunning] = useState(false);
  const [autonomousStopping, setAutonomousStopping] = useState(false);
  const [autonomousStatus, setAutonomousStatus] = useState(null);
  const [autonomousActivity, setAutonomousActivity] = useState([]);
  const [brainstorms, setBrainstorms] = useState([]);
  const [papers, setPapers] = useState([]);
  const [autonomousStats, setAutonomousStats] = useState(null);

  // LeanOJ mode state
  const [leanojRunning, setLeanojRunning] = useState(false);
  const [leanojStatus, setLeanojStatus] = useState(null);
  const [leanojActivity, setLeanojActivity] = useState([]);
  const [leanojSettings, setLeanojSettings] = useState(() => getStoredLeanOJSettings());
  const [leanojProofRefreshToken, setLeanojProofRefreshToken] = useState(0);
  
  // Disclaimer modal state (shows on every app load)
  const [showDisclaimer, setShowDisclaimer] = useState(true);
  const [showStartupSetupModal, setShowStartupSetupModal] = useState(false);
  const [startupSetupMessage, setStartupSetupMessage] = useState('');
  const [checkingLmStudioStartupChoice, setCheckingLmStudioStartupChoice] = useState(false);
  
  // OpenRouter privacy warning modal state
  const [showPrivacyWarning, setShowPrivacyWarning] = useState(false);
  const [privacyWarningData, setPrivacyWarningData] = useState(null);
  
  // OpenRouter rate limit tracking
  const [rateLimitedModels, setRateLimitedModels] = useState(new Map());
  
  // Critique notification stack state
  const [critiqueNotifications, setCritiqueNotifications] = useState([]);
  const [selectedCritiquePaper, setSelectedCritiquePaper] = useState(null);
  const [showCritiqueModal, setShowCritiqueModal] = useState(false);
  const [proofNotifications, setProofNotifications] = useState([]);
  const [selectedProofId, setSelectedProofId] = useState(null);
  const [proofRefreshToken, setProofRefreshToken] = useState(0);
  const [latestProofDependencyEvent, setLatestProofDependencyEvent] = useState(null);

  // Credit exhaustion notification state (persistent until dismissed)
  const [creditExhaustionNotifications, setCreditExhaustionNotifications] = useState([]);

  // Live refs used by websocket listeners (which are registered once)
  const autonomousRunningRef = useRef(autonomousRunning);
  const autonomousTierRef = useRef(autonomousStatus?.current_tier || null);
  const openRouterKeyJustSavedRef = useRef(false);
  const seenHighScoreCritiquesRef = useRef(null);
  const shownHighScoreCritiquesRef = useRef(null);
  if (seenHighScoreCritiquesRef.current === null) {
    seenHighScoreCritiquesRef.current = readSeenHighScoreCritiques();
    shownHighScoreCritiquesRef.current = new Set(seenHighScoreCritiquesRef.current);
  }

  useEffect(() => {
    autonomousRunningRef.current = autonomousRunning;
  }, [autonomousRunning]);

  useEffect(() => {
    autonomousTierRef.current = autonomousStatus?.current_tier || null;
  }, [autonomousStatus]);

  const markHighScoreCritiqueSeen = useCallback((seenKey) => {
    if (!seenKey) {
      return;
    }

    const seen = seenHighScoreCritiquesRef.current;
    if (seen.has(seenKey)) {
      return;
    }

    seen.add(seenKey);
    persistSeenHighScoreCritiques(seen);
  }, []);

  // Autonomous config with localStorage persistence
  // CRITICAL: Read from 'autonomous_research_settings' (used by AutonomousResearchSettings component)
  const [autonomousConfig, setAutonomousConfig] = useState(() => {
    return settingsToAutonomousConfig(getStoredAutonomousSettings());
  });

  // Save autonomous config to localStorage
  useEffect(() => {
    const existingSettings = getStoredAutonomousSettings();
    persistAutonomousSettings({
      ...existingSettings,
      numSubmitters: autonomousConfig.submitter_configs?.length || existingSettings.numSubmitters || 3,
      submitterConfigs: autonomousConfig.submitter_configs || existingSettings.submitterConfigs,
      localConfig: {
        ...existingSettings.localConfig,
        validator_provider: autonomousConfig.validator_provider,
        validator_model: autonomousConfig.validator_model,
        validator_openrouter_provider: autonomousConfig.validator_openrouter_provider,
        validator_lm_studio_fallback: autonomousConfig.validator_lm_studio_fallback,
        validator_context_window: autonomousConfig.validator_context_window,
        validator_max_tokens: autonomousConfig.validator_max_tokens,
        validator_supercharge_enabled: autonomousConfig.validator_supercharge_enabled,
        high_context_provider: autonomousConfig.high_context_provider,
        high_context_model: autonomousConfig.high_context_model,
        high_context_openrouter_provider: autonomousConfig.high_context_openrouter_provider,
        high_context_lm_studio_fallback: autonomousConfig.high_context_lm_studio_fallback,
        high_context_context_window: autonomousConfig.high_context_context_window,
        high_context_max_tokens: autonomousConfig.high_context_max_tokens,
        high_context_supercharge_enabled: autonomousConfig.high_context_supercharge_enabled,
        high_param_provider: autonomousConfig.high_param_provider,
        high_param_model: autonomousConfig.high_param_model,
        high_param_openrouter_provider: autonomousConfig.high_param_openrouter_provider,
        high_param_lm_studio_fallback: autonomousConfig.high_param_lm_studio_fallback,
        high_param_context_window: autonomousConfig.high_param_context_window,
        high_param_max_tokens: autonomousConfig.high_param_max_tokens,
        high_param_supercharge_enabled: autonomousConfig.high_param_supercharge_enabled,
        critique_submitter_provider: autonomousConfig.critique_submitter_provider,
        critique_submitter_model: autonomousConfig.critique_submitter_model,
        critique_submitter_openrouter_provider: autonomousConfig.critique_submitter_openrouter_provider,
        critique_submitter_lm_studio_fallback: autonomousConfig.critique_submitter_lm_studio_fallback,
        critique_submitter_context_window: autonomousConfig.critique_submitter_context_window,
        critique_submitter_max_tokens: autonomousConfig.critique_submitter_max_tokens,
        critique_submitter_supercharge_enabled: autonomousConfig.critique_submitter_supercharge_enabled,
      },
      allowMathematicalProofs: autonomousConfig.allow_mathematical_proofs ?? existingSettings.allowMathematicalProofs ?? true,
      allowResearchPapers: autonomousConfig.allow_research_papers ?? existingSettings.allowResearchPapers ?? true,
      tier3Enabled: autonomousConfig.tier3_enabled ?? existingSettings.tier3Enabled ?? false,
      creativityEmphasisBoostEnabled: autonomousConfig.creativity_emphasis_boost_enabled ?? existingSettings.creativityEmphasisBoostEnabled ?? false,
    });
  }, [autonomousConfig]);

  useEffect(() => {
    persistLeanOJSettings(leanojSettings);
  }, [leanojSettings]);

  const syncProviderAvailability = useCallback(async () => {
    let nextCapabilities = DEFAULT_CAPABILITIES;
    try {
      const featuresPayload = await api.getFeatures();
      nextCapabilities = normalizeFeaturesPayload(featuresPayload);
    } catch (err) {
      console.error('Failed to fetch runtime feature flags:', err);
    }

    setCapabilities(nextCapabilities);

    let lmResult = {
      available: false,
      has_models: false,
      model_count: 0,
      models: [],
      error: nextCapabilities.lmStudioEnabled
        ? null
        : (nextCapabilities.genericMode
            ? 'LM Studio is intentionally disabled in this hosted deployment.'
            : null),
      generic_mode: nextCapabilities.genericMode,
    };

    if (nextCapabilities.lmStudioEnabled) {
      try {
        lmResult = await openRouterAPI.checkLMStudioAvailability();
      } catch (err) {
        console.error('Failed to check LM Studio availability:', err);
        lmResult = {
          available: false,
          has_models: false,
          model_count: 0,
          models: [],
          error: err.message || 'Failed to check LM Studio availability.',
          generic_mode: nextCapabilities.genericMode,
        };
      }
    }

    const usableLmStudioChatModelId = getUsableLoadedLmStudioChatModelId(lmResult.models || []);
    const hasUsableLmStudioChatModel = Boolean(usableLmStudioChatModelId);
    const nextLmStudioStatus = {
      ...lmResult,
      usable_chat_model_id: usableLmStudioChatModelId,
      has_usable_chat_model: hasUsableLmStudioChatModel,
    };
    const lmAvailable = nextCapabilities.lmStudioEnabled && Boolean(lmResult.available && lmResult.has_models);
    setLmStudioStatus(nextLmStudioStatus);
    setLmStudioAvailable(lmAvailable);

    let keyStatus = { has_key: false };
    let keyStatusOk = false;
    // Retry aggressively (up to ~20s) to cover backend cold-start. The
    // `/api/openrouter/api-key-status` endpoint is trivial (memory lookup),
    // so any failure here means the backend literally is not yet accepting
    // HTTP — we must NOT declare "no key" to the UI on that basis, because
    // the real state is "unknown" and declaring it false would incorrectly
    // open the startup provider setup modal over a stored key.
    const delays = [200, 400, 800, 1200, 1500, 2000, 2000, 2500, 2500, 3000, 3000];
    for (let attempt = 0; attempt < delays.length; attempt += 1) {
      try {
        keyStatus = await openRouterAPI.getApiKeyStatus();
        keyStatusOk = true;
        break;
      } catch (err) {
        if (attempt === delays.length - 1) {
          console.warn('OpenRouter key-status probe still unreachable after initial cold-start window; background poller will retry.', err);
        }
        await new Promise((resolve) => setTimeout(resolve, delays[attempt]));
      }
    }

    let codexConfigured = false;
    try {
      const cloudStatus = await cloudAccessAPI.getStatus();
      codexConfigured = Boolean(cloudStatus.providers?.openai_codex_oauth?.configured);
    } catch {
      codexConfigured = false;
    }

    const finalHasOpenRouterKey = Boolean(keyStatus.has_key);
    if (keyStatusOk) {
      setHasOpenRouterKey(finalHasOpenRouterKey);
      setHasCloudAccess(finalHasOpenRouterKey || codexConfigured);
    }

    let availableModels = [];
    if (nextCapabilities.lmStudioEnabled && lmAvailable) {
      try {
        const data = await api.getModels();
        availableModels = data.models || data || [];
        setModels(availableModels);
      } catch (err) {
        console.error('Failed to fetch LM Studio models:', err);
        setModels([]);
      }
    } else {
      setModels([]);
    }

    return {
      capabilities: nextCapabilities,
      lmAvailable,
      hasOpenRouterKey: finalHasOpenRouterKey,
      hasCloudAccess: finalHasOpenRouterKey || codexConfigured,
      keyStatusReachable: keyStatusOk,
      hasUsableLmStudioChatModel,
      lmStudioStatus: nextLmStudioStatus,
      defaultLmStudioModelId: usableLmStudioChatModelId,
    };
  }, []);

  useEffect(() => {
    syncProviderAvailability();
  }, [syncProviderAvailability]);

  // Fetch update notices on mount, then every 4 hours until one is shown or dismissed.
  useEffect(() => {
    if (updateNoticeDismissed || updateNotice?.update_available) {
      return undefined;
    }

    let cancelled = false;
    const fetchUpdateNotice = async () => {
      try {
        const notice = await api.getUpdateNotice();
        if (!cancelled && notice && notice.update_available) {
          setUpdateNotice(notice);
        }
      } catch {
        // Backend unreachable, skip this cycle.
      }
    };

    fetchUpdateNotice();
    const intervalId = window.setInterval(fetchUpdateNotice, UPDATE_NOTICE_POLL_INTERVAL_MS);

    return () => {
      cancelled = true;
      window.clearInterval(intervalId);
    };
  }, [updateNoticeDismissed, updateNotice]);

  useEffect(() => {
    if (capabilities.lmStudioEnabled) {
      return;
    }

    setConfig((prev) => {
      const next = normalizeAggregatorConfigForCapabilities(prev, false);
      return JSON.stringify(next) === JSON.stringify(prev) ? prev : next;
    });

    setAutonomousConfig((prev) => {
      const next = normalizeAutonomousConfigForCapabilities(prev, false);
      return JSON.stringify(next) === JSON.stringify(prev) ? prev : next;
    });

    if (localStorage.getItem(STARTUP_PROVIDER_CHOICE_STORAGE_KEY) === LM_STUDIO_STARTUP_CHOICE) {
      localStorage.removeItem(STARTUP_PROVIDER_CHOICE_STORAGE_KEY);
    }
  }, [capabilities.lmStudioEnabled]);

  // Periodically re-check cloud access status to keep indicator in sync.
  // We poll aggressively (5s) because the state mostly flips from "unknown"
  // to "known" shortly after backend startup, and users notice any delay as
  // "my key didn't save."
  useEffect(() => {
    const interval = setInterval(async () => {
      try {
        const keyStatus = await openRouterAPI.getApiKeyStatus();
        const hasKey = Boolean(keyStatus.has_key);
        setHasOpenRouterKey(hasKey);
        try {
          const cloudStatus = await cloudAccessAPI.getStatus();
          setHasCloudAccess(hasKey || Boolean(cloudStatus.providers?.openai_codex_oauth?.configured));
        } catch {
          setHasCloudAccess(hasKey);
        }
      } catch {
        // Backend unreachable, skip this cycle
      }
    }, 5000);

    return () => clearInterval(interval);
  }, []);

  // Periodically re-check LM Studio availability so the header indicator
  // recovers when LM Studio finishes starting after the initial page load
  // (e.g. MOTO launches the browser before LM Studio's local server is
  // ready to serve /v1/models). Without this, the first check returns
  // unavailable and the "LM Studio Offline" badge sticks for the entire
  // session even while nomic embedding calls are succeeding.
  useEffect(() => {
    if (!capabilities.lmStudioEnabled) {
      return undefined;
    }

    const interval = setInterval(() => {
      if (typeof document !== 'undefined' && document.visibilityState === 'hidden') {
        return;
      }
      syncProviderAvailability().catch(() => {
        // Backend unreachable or transient failure, skip this cycle
      });
    }, 15000);

    return () => clearInterval(interval);
  }, [capabilities.lmStudioEnabled, syncProviderAvailability]);

  // Re-sync provider availability immediately when the tab becomes visible
  // again. Users often switch to LM Studio to load a model and then return;
  // waiting up to 15s for the next interval tick feels broken.
  useEffect(() => {
    if (!capabilities.lmStudioEnabled) {
      return undefined;
    }
    if (typeof document === 'undefined') {
      return undefined;
    }

    const handleVisibilityChange = () => {
      if (document.visibilityState === 'visible') {
        syncProviderAvailability().catch(() => {});
      }
    };

    document.addEventListener('visibilitychange', handleVisibilityChange);
    return () => {
      document.removeEventListener('visibilitychange', handleVisibilityChange);
    };
  }, [capabilities.lmStudioEnabled, syncProviderAvailability]);

  // Check autonomous research status on mount (handles page refresh while running)
  // CRITICAL: Always load all data (brainstorms, papers, stats) on startup,
  // even when not running. This ensures users see existing data immediately
  // without having to click Start first.
  useEffect(() => {
    const checkInitialStatus = async () => {
      try {
        const [status, brainstormsData, papersData, stats] = await Promise.all([
          autonomousAPI.getStatus(),
          autonomousAPI.getBrainstorms(),
          autonomousAPI.getPapers(),
          autonomousAPI.getStats()
        ]);
        
        // ALWAYS load brainstorms, papers, and stats regardless of running state
        // This ensures data is visible on app startup without needing to click Start
        setBrainstorms(brainstormsData.brainstorms || []);
        setPapers(papersData.papers || []);
        setAutonomousStats(stats);
        setAutonomousStatus(status);
        
        // If backend reports running, also sync the running state
        if (status.is_running) {
          console.log('Autonomous research detected as running, syncing state...');
          setAutonomousRunning(true);
          setAnyWorkflowRunning(true);
        }
      } catch (error) {
        console.error('Failed to check initial autonomous status:', error);
      }
    };
    
    checkInitialStatus();
  }, []);

  // Recover high-score critique popups from persisted paper metadata. WebSocket
  // events are best-effort, so a sleeping/closed browser can miss the live event.
  useEffect(() => {
    if (!papers || papers.length === 0) {
      return;
    }

    const recoveredNotifications = [];
    for (const paper of papers) {
      const averageRating = Number(paper.critique_avg);
      if (!Number.isFinite(averageRating) || averageRating < HIGH_SCORE_CRITIQUE_THRESHOLD) {
        continue;
      }

      const seenKey = getHighScoreCritiqueNotificationKey(paper.paper_id, averageRating);
      if (!seenKey || shownHighScoreCritiquesRef.current.has(seenKey)) {
        continue;
      }

      shownHighScoreCritiquesRef.current.add(seenKey);
      recoveredNotifications.push({
        id: `critique_recovered_${seenKey}_${Date.now()}`,
        paper_id: paper.paper_id,
        paper_title: paper.title || paper.paper_title || paper.paper_id,
        average_rating: averageRating,
        timestamp: paper.created_at || new Date().toISOString(),
        seenKey,
        recovered: true,
      });
    }

    if (recoveredNotifications.length === 0) {
      return;
    }

    setCritiqueNotifications(prev => {
      const existingSeenKeys = new Set(prev.map(notification => notification.seenKey).filter(Boolean));
      const newNotifications = recoveredNotifications.filter(notification => !existingSeenKeys.has(notification.seenKey));
      if (newNotifications.length === 0) {
        return prev;
      }

      const newStack = [...prev, ...newNotifications];
      return newStack.length > 3 ? newStack.slice(-3) : newStack;
    });
  }, [papers]);

  useEffect(() => {
    const checkLeanOJStatus = async () => {
      try {
        const status = await leanojAPI.getStatus();
        setLeanojStatus(status);
        if (status.is_running) {
          setLeanojRunning(true);
          setAnyWorkflowRunning(true);
        }
      } catch (error) {
        console.error('Failed to check initial Proof Solver status:', error);
      }
    };
    checkLeanOJStatus();
  }, []);

  // WebSocket connection
  useEffect(() => {
    // Connect to WebSocket
    websocket.connect();

    return () => {
      websocket.disconnect();
    };
  }, []);

  // Autonomous WebSocket event listeners
  useEffect(() => {
    const unsubscribers = [];
    
    // Helper to add activity with limit (prevents unbounded array growth causing UI freeze)
    // Helper to get timestamp from server or fallback to client time
    const getTimestamp = (data) => data?._serverTimestamp || new Date().toISOString();
    const addActivity = (event) => {
      setAutonomousActivity(prev => [...prev, event].slice(-MAX_LIVE_ACTIVITY_EVENTS));
    };
    const formatHungConnectionMessage = (data = {}) => {
      const model = data.model || 'model';
      const provider = data.provider || 'provider';
      const elapsed = data.elapsed_minutes || 15;
      return `Possible hung model call: ${model} via ${provider} (${elapsed}+ min). It may still be thinking; you can keep waiting or lower reasoning effort in Settings if this repeats.`;
    };
    const addLeanOJActivityFromGlobalAlert = (event) => {
      setLeanojActivity(prev => [...prev, event].slice(-MAX_LIVE_ACTIVITY_EVENTS));
    };
    const shouldAddHungAlertToAutonomousFeed = (data = {}) => {
      const roleId = String(data.role_id || '').toLowerCase();
      if (roleId.startsWith('leanoj_')) {
        return false;
      }
      if (roleId.startsWith('autonomous_') || roleId.startsWith('proof_')) {
        return true;
      }
      if (autonomousRunningRef.current && (
        roleId.startsWith('aggregator_') ||
        roleId.startsWith('compiler_')
      )) {
        return true;
      }
      return false;
    };
    const isAutonomousTier2Active = () =>
      autonomousRunningRef.current && autonomousTierRef.current === 'tier2_paper_writing';
    const formatCompilerMode = (mode) => {
      switch (mode) {
        case 'outline_create':
          return 'Outline creation';
        case 'construction':
          return 'Construction';
        case 'outline_update':
          return 'Outline update';
        case 'review':
          return 'Review';
        case 'rigor':
          return 'Rigor';
        default:
          return mode || 'Compiler';
      }
    };
    const formatReason = (reasoning, maxLen = 140) => {
      if (!reasoning) return '';
      const cleaned = String(reasoning).replace(/\s+/g, ' ').trim();
      if (!cleaned) return '';
      return cleaned.length > maxLen ? `${cleaned.slice(0, maxLen)}...` : cleaned;
    };
    const proofName = (data = {}) => (data.proof_label ? `Proof ${data.proof_label}` : 'Proof');
    const proofTarget = (data = {}) => data.theorem_statement || data.theorem_id || '';
    const proofLeanResponse = (data = {}) => {
      if (data.lean_response) return data.lean_response;
      if (data.proof_verified === true) return 'Lean 4 response: proof verified.';
      const error = formatReason(data.error_summary || data.error_output || data.reason || '', 960);
      return error ? `Lean 4 response: ${error} - proof not verified.` : 'Lean 4 response: proof not verified.';
    };
    const formatProofNoveltyTier = (tier) => {
      switch (tier) {
        case 'major_mathematical_discovery':
          return 'Major mathematical discovery';
        case 'mathematical_discovery':
          return 'Mathematical discovery';
        case 'novel_variant':
          return 'Novel variant';
        case 'novel_formulation':
          return 'Novel formulation';
        case 'not_novel':
          return 'Not novel';
        case 'novel':
          return 'Novel';
        default:
          return tier ? String(tier).replace(/_/g, ' ') : 'Not rated';
      }
    };
    const proofNoveltyMessage = (data = {}) => {
      const tierLabel = formatProofNoveltyTier(data.novelty_tier || (data.is_novel ? 'novel' : 'not_novel'));
      const duplicateNote = data.duplicate ? ' (duplicate proof reused)' : '';
      const reason = formatReason(data.novelty_reasoning || data.reasoning || '', 240);
      const target = proofTarget(data);
      return `${proofName(data)} Lean 4 novelty validator rating: ${tierLabel}${duplicateNote}${reason ? ` - ${reason}` : ''}${target ? ` (${target})` : ''}`;
    };
    const isLeanOJProofEvent = (data = {}) => {
      const sourceType = String(data.source_type || '');
      const sourceId = String(data.source_id || '');
      const trigger = String(data.trigger || '');
      return sourceType === 'leanoj_final'
        || sourceType === 'leanoj_subproof'
        || sourceId.startsWith('leanoj_')
        || trigger.startsWith('leanoj');
    };
    const shouldShowAutonomousProofNovelty = (data = {}) => {
      if (isLeanOJProofEvent(data)) return false;
      if (data.source_type === 'compiler_rigor' && !isAutonomousTier2Active()) return false;
      return true;
    };
    const formatProofCheckCompleteMessage = (data = {}) => {
      const verified = data.verified_count ?? 0;
      const novel = data.novel_count ?? 0;
      const hasTotal = data.total_candidates !== undefined && data.total_candidates !== null;
      const base = hasTotal
        ? `Proof check complete: ${verified}/${data.total_candidates} candidates verified, ${novel} novel`
        : `Proof check complete: ${verified} verified`;
      const detail = formatReason(data.message, 220);
      return detail ? `${base} - ${detail}` : base;
    };
    
    // Topic exploration events (pre-brainstorm candidate collection)
    unsubscribers.push(websocket.on('topic_exploration_started', (data) => {
      addActivity({
        event: 'topic_exploration_started',
        timestamp: getTimestamp(data),
        message: `Topic exploration started (target: ${data.target || 5} candidates${data.resumed_count ? `, resuming with ${data.resumed_count}` : ''})`,
        data
      });
    }));
    
    unsubscribers.push(websocket.on('topic_exploration_progress', (data) => {
      addActivity({
        event: 'topic_exploration_progress',
        timestamp: getTimestamp(data),
        message: `Exploration candidate ${data.accepted}/${data.target} accepted: ${data.latest_question ? data.latest_question.substring(0, 100) + '...' : ''}`,
        data
      });
    }));
    
    unsubscribers.push(websocket.on('topic_exploration_complete', (data) => {
      addActivity({
        event: 'topic_exploration_complete',
        timestamp: getTimestamp(data),
        message: `Topic exploration complete: ${data.accepted_count} candidates collected from ${data.total_attempts} attempts`,
        data
      });
    }));
    
    // Paper title exploration events (pre-title-selection candidate collection)
    unsubscribers.push(websocket.on('paper_title_exploration_started', (data) => {
      addActivity({
        event: 'paper_title_exploration_started',
        timestamp: getTimestamp(data),
        message: `Title exploration started (target: ${data.target || 5} candidate titles)`,
        data
      });
    }));
    
    unsubscribers.push(websocket.on('paper_title_exploration_progress', (data) => {
      addActivity({
        event: 'paper_title_exploration_progress',
        timestamp: getTimestamp(data),
        message: `Title candidate ${data.accepted}/${data.target} accepted`,
        data
      });
    }));
    
    unsubscribers.push(websocket.on('paper_title_exploration_complete', (data) => {
      addActivity({
        event: 'paper_title_exploration_complete',
        timestamp: getTimestamp(data),
        message: `Title exploration complete: ${data.accepted_count} candidates collected from ${data.total_attempts} attempts`,
        data
      });
    }));
    
    // Topic selection events
    unsubscribers.push(websocket.on('topic_selected', (data) => {
      addActivity({
        event: 'topic_selected',
        timestamp: getTimestamp(data),
        message: `Topic selected: ${data.topic_prompt}`,
        data
      });
    }));
    
    unsubscribers.push(websocket.on('topic_selection_rejected', (data) => {
      addActivity({
        event: 'topic_selection_rejected',
        timestamp: getTimestamp(data),
        message: `Topic selection rejected`,
        data
      });
    }));
    
    // Aggregator's direct submission events (per-submission with individual submitter_id)
    unsubscribers.push(websocket.on('submission_accepted', (data) => {
      const modelName = data.submitter_model ? (data.submitter_model.split('/')[1] || data.submitter_model.substring(0, 15)) : 'N/A';
      const creativityPrefix = data.creativity_emphasized ? '(Creativity Emphasized) ' : '';
      addActivity({
        event: 'submission_accepted',
        timestamp: getTimestamp(data),
        message: `${creativityPrefix}Submitter ${data.submitter_id} [${modelName}]: ✓ ACCEPTED (total: ${data.total_acceptances})`,
        data
      });
    }));
    
    unsubscribers.push(websocket.on('submission_rejected', (data) => {
      const modelName = data.submitter_model ? (data.submitter_model.split('/')[1] || data.submitter_model.substring(0, 15)) : 'N/A';
      const creativityPrefix = data.creativity_emphasized ? '(Creativity Emphasized) ' : '';
      addActivity({
        event: 'submission_rejected',
        timestamp: getTimestamp(data),
        message: `${creativityPrefix}Submitter ${data.submitter_id} [${modelName}]: ✗ REJECTED (total: ${data.total_rejections})`,
        data
      });
    }));
    
    // Completion review events
    unsubscribers.push(websocket.on('completion_review_started', (data) => {
      addActivity({
        event: 'completion_review_started',
        timestamp: getTimestamp(data),
        message: `Completion review started`,
        data
      });
    }));
    
    unsubscribers.push(websocket.on('completion_review_result', (data) => {
      addActivity({
        event: 'completion_review_result',
        timestamp: getTimestamp(data),
        message: `Decision: ${data.decision}`,
        data
      });
    }));
    
    unsubscribers.push(websocket.on('manual_paper_writing_triggered', (data) => {
      addActivity({
        event: 'manual_paper_writing_triggered',
        timestamp: getTimestamp(data),
        message: `Manual override: Forcing paper writing for ${data.topic_id} (${data.submission_count} submissions)`,
        data
      });
    }));
    
    // Paper events
    unsubscribers.push(websocket.on('paper_writing_started', (data) => {
      autonomousTierRef.current = 'tier2_paper_writing';
      addActivity({
        event: 'paper_writing_started',
        timestamp: getTimestamp(data),
        message: `Paper writing started: ${data.title}`,
        data
      });
    }));

    // Compiler writing activity events (Tier 2 paper writing internals)
    unsubscribers.push(websocket.on('compiler_acceptance', (data) => {
      if (!isAutonomousTier2Active()) return;
      const modeLabel = formatCompilerMode(data.mode);
      const iterationSuffix = data.iteration ? ` (iteration ${data.iteration})` : '';
      addActivity({
        event: 'compiler_acceptance',
        timestamp: getTimestamp(data),
        message: `${modeLabel}: ✓ ACCEPTED${iterationSuffix}`,
        data
      });
    }));

    unsubscribers.push(websocket.on('compiler_rejection', (data) => {
      if (!isAutonomousTier2Active()) return;
      const modeLabel = formatCompilerMode(data.mode);
      const iterationSuffix = data.iteration ? ` (iteration ${data.iteration})` : '';
      const reason = formatReason(data.reasoning);
      addActivity({
        event: 'compiler_rejection',
        timestamp: getTimestamp(data),
        message: `${modeLabel}: ✗ REJECTED${iterationSuffix}${reason ? ` - ${reason}` : ''}`,
        data
      });
    }));

    unsubscribers.push(websocket.on('compiler_decline', (data) => {
      if (!isAutonomousTier2Active()) return;
      const modeLabel = formatCompilerMode(data.mode);
      const reason = formatReason(data.reasoning, 100);
      addActivity({
        event: 'compiler_decline',
        timestamp: getTimestamp(data),
        message: `${modeLabel}: ↷ DECLINED${reason ? ` - ${reason}` : ''}`,
        data
      });
    }));

    unsubscribers.push(websocket.on('outline_locked', (data) => {
      if (!isAutonomousTier2Active()) return;
      addActivity({
        event: 'outline_locked',
        timestamp: getTimestamp(data),
        message: `Outline locked after ${data.total_iterations || data.iteration || '?'} iteration(s)`,
        data
      });
    }));
    
    // Critique phase events (paper writing substages)
    unsubscribers.push(websocket.on('critique_phase_started', (data) => {
      addActivity({
        event: 'critique_phase_started',
        timestamp: getTimestamp(data),
        message: `Critique phase started (Paper v${data.paper_version || '?'}, target: ${data.target_critiques || 3} attempts)`,
        data
      });
    }));
    
    unsubscribers.push(websocket.on('critique_progress', (data) => {
      // Only log every few updates to avoid spam
      if (data.total_attempts % 2 === 0 || data.total_attempts >= data.target) {
        addActivity({
          event: 'critique_progress',
          timestamp: getTimestamp(data),
          message: `Critique progress: ${data.acceptances} accepted, ${data.rejections} rejected (${data.total_attempts}/${data.target} attempts)`,
          data
        });
      }
    }));
    
    unsubscribers.push(websocket.on('self_review_appended', (data) => {
      addActivity({
        event: 'self_review_appended',
        timestamp: getTimestamp(data),
        message: `AI self-review appended (${data.critique_count || 0} accepted critique${data.critique_count === 1 ? '' : 's'})`,
        data
      });
    }));
    
    unsubscribers.push(websocket.on('critique_phase_ended', (data) => {
      addActivity({
        event: 'critique_phase_ended',
        timestamp: getTimestamp(data),
        message: `Critique phase complete (self-review appended: ${data.self_review_appended ? 'yes' : 'no'})`,
        data
      });
    }));
    
    // Phase transitions during paper writing
    unsubscribers.push(websocket.on('phase_transition', (data) => {
      const fromPhase = data.from_phase || '?';
      const toPhase = data.to_phase || '?';
      const trigger = data.trigger || 'complete';
      addActivity({
        event: 'phase_transition',
        timestamp: getTimestamp(data),
        message: `Phase transition: ${fromPhase} → ${toPhase} (${trigger})`,
        data
      });
    }));
    
    unsubscribers.push(websocket.on('paper_completed', (data) => {
      addActivity({
        event: 'paper_completed',
        timestamp: getTimestamp(data),
        message: `Paper completed: ${data.title}`,
        data
      });
      // Refresh papers list
      autonomousAPI.getPapers().then(res => setPapers(res.papers || [])).catch(console.error);
    }));
    
    unsubscribers.push(websocket.on('paper_redundancy_review', (data) => {
      addActivity({
        event: 'paper_redundancy_review',
        timestamp: getTimestamp(data),
        message: `Redundancy review: ${data.should_remove ? 'Removing paper' : 'No removal'}`,
        data
      });
    }));

    unsubscribers.push(websocket.on('proof_framing_decided', (data) => {
      addActivity({
        event: 'proof_framing_decided',
        timestamp: getTimestamp(data),
        message: data.is_proof_amenable
          ? 'Proof framing enabled for this research run'
          : 'Proof framing not applied for this research run',
        data
      });
    }));

    unsubscribers.push(websocket.on('proof_check_started', (data) => {
      setProofRefreshToken((prev) => prev + 1);
    }));

    unsubscribers.push(websocket.on('proof_retry_scheduled', (data) => {
      setProofRefreshToken((prev) => prev + 1);
    }));

    unsubscribers.push(websocket.on('proof_retry_started', (data) => {
      setProofRefreshToken((prev) => prev + 1);
    }));

    unsubscribers.push(websocket.on('proof_check_no_candidates', (data) => {
      setProofRefreshToken((prev) => prev + 1);
    }));

    unsubscribers.push(websocket.on('proof_check_candidates_found', (data) => {
      setProofRefreshToken((prev) => prev + 1);
    }));

    unsubscribers.push(websocket.on('proof_attempt_started', (data) => {
      addActivity({
        event: 'proof_attempt_started',
        timestamp: getTimestamp(data),
        message: `${proofName(data)}, Attempt ${data.attempt || 1} started: ${proofTarget(data)}`,
        data
      });
    }));

    unsubscribers.push(websocket.on('smt_check_error', (data) => {
      addActivity({
        event: 'smt_check_error',
        timestamp: getTimestamp(data),
        message: `${proofName(data)} SMT error: ${formatReason(data.error_summary, 960) || proofTarget(data)}`,
        data
      });
    }));

    unsubscribers.push(websocket.on('proof_attempt_failed', (data) => {
      addActivity({
        event: 'proof_attempt_failed',
        timestamp: getTimestamp(data),
        message: `${proofName(data)}, Attempt ${data.attempt || '?'} final: ${proofLeanResponse(data)}`,
        data
      });
    }));

    unsubscribers.push(websocket.on('proof_verified', (data) => {
      setProofRefreshToken((prev) => prev + 1);
    }));

    unsubscribers.push(websocket.on('proof_lean_accepted', (data) => {
      addActivity({
        event: 'proof_lean_accepted',
        timestamp: getTimestamp(data),
        message: `${proofName(data)}, Attempt ${data.attempt || '?'} final: ${proofLeanResponse(data)}`,
        data
      });
    }));

    unsubscribers.push(websocket.on('proof_integrity_rejected', (data) => {
      addActivity({
        event: 'proof_integrity_rejected',
        timestamp: getTimestamp(data),
        message: `${proofName(data)} error: integrity rejected - ${formatReason(data.reason, 960) || proofTarget(data)}`,
        data
      });
    }));

    unsubscribers.push(websocket.on('proof_attempts_exhausted', (data) => {
      addActivity({
        event: 'proof_attempts_exhausted',
        timestamp: getTimestamp(data),
        message: `${proofName(data)} terminated: proof attempts exhausted for ${proofTarget(data)}`,
        data
      });
    }));

    unsubscribers.push(websocket.on('novel_proof_discovered', (data) => {
      setProofRefreshToken((prev) => prev + 1);
      if (shouldShowAutonomousProofNovelty(data)) {
        addActivity({
          event: 'novel_proof_discovered',
          timestamp: getTimestamp(data),
          message: proofNoveltyMessage(data),
          data
        });
      }
      setProofNotifications((prev) => {
        const next = [
          ...prev,
          {
            id: `proof_${data.proof_id}_${Date.now()}`,
            proof_id: data.proof_id,
            theorem_statement: data.theorem_statement,
            source_type: data.source_type,
            source_id: data.source_id,
            novelty_tier: data.novelty_tier || 'mathematical_discovery',
            timestamp: getTimestamp(data),
          }
        ];
        return next.length > MAX_PROOF_NOTIFICATIONS
          ? next.slice(-MAX_PROOF_NOTIFICATIONS)
          : next;
      });
    }));

    unsubscribers.push(websocket.on('known_proof_verified', (data) => {
      setProofRefreshToken((prev) => prev + 1);
      if (shouldShowAutonomousProofNovelty(data)) {
        addActivity({
          event: 'known_proof_verified',
          timestamp: getTimestamp(data),
          message: proofNoveltyMessage(data),
          data
        });
      }
    }));

    unsubscribers.push(websocket.on('proof_registration_duplicate', (data) => {
      setProofRefreshToken((prev) => prev + 1);
      if (shouldShowAutonomousProofNovelty(data)) {
        addActivity({
          event: 'proof_registration_duplicate',
          timestamp: getTimestamp(data),
          message: proofNoveltyMessage({ ...data, duplicate: true }),
          data: { ...data, duplicate: true }
        });
      }
    }));

    unsubscribers.push(websocket.on('proof_dependency_added', (data) => {
      setLatestProofDependencyEvent(data);
      setProofRefreshToken((prev) => prev + 1);
    }));

    unsubscribers.push(websocket.on('proof_check_complete', (data) => {
      if (isLeanOJProofEvent(data)) return;
      if (data.source_type === 'compiler_rigor' && !isAutonomousTier2Active()) return;

      setProofRefreshToken((prev) => prev + 1);
      const message = formatProofCheckCompleteMessage(data);

      addActivity({
        event: 'proof_check_complete',
        timestamp: getTimestamp(data),
        message,
        data
      });
    }));
    
    unsubscribers.push(websocket.on('auto_research_started', () => {
      setAutonomousActivity([]);
      setAutonomousRunning(true);
      setAnyWorkflowRunning(true);
      setAutonomousStopping(false);
    }));
    
    unsubscribers.push(websocket.on('auto_research_resumed', (data) => {
      // Handle resume after crash/restart - sync running state
      console.log('Autonomous research resumed:', data);
      setAutonomousRunning(true);
      setAnyWorkflowRunning(true);
      setAutonomousStopping(false);
      if (data?.tier) {
        autonomousTierRef.current = data.tier;
      }
      addActivity({
        event: 'auto_research_resumed',
        timestamp: getTimestamp(data),
        message: `Research resumed (${data?.tier || 'unknown tier'})`,
        data
      });
      // Fetch latest status
      autonomousAPI.getStatus().then(status => {
        setAutonomousStatus(status);
      }).catch(console.error);
    }));
    
    unsubscribers.push(websocket.on('auto_research_stopped', () => {
      setAutonomousRunning(false);
      setAutonomousStopping(false);
      setAnyWorkflowRunning(false);
      autonomousTierRef.current = null;
    }));
    
    // Tier 3 events
    unsubscribers.push(websocket.on('tier3_started', (data) => {
      autonomousTierRef.current = 'tier3_final_answer';
      addActivity({
        event: 'tier3_started',
        timestamp: getTimestamp(data),
        message: `Tier 3 Final Answer generation started`,
        data
      });
      // Refresh status to update tier3 info
      autonomousAPI.getStatus().then(setAutonomousStatus).catch(console.error);
    }));
    
    // tier3_result - Backend sends this for certainty assessment result
    unsubscribers.push(websocket.on('tier3_result', (data) => {
      let message;
      if (data.result === 'continue_research') {
        // AI determined that existing papers don't provide a definitive answer yet
        message = 'AI assessment: No definitive answer can be derived from current research. Resuming autonomous research to generate more papers before attempting final answer again.';
      } else {
        message = `Certainty assessment: ${data.certainty_level || 'complete'} - Proceeding to generate final answer`;
      }
      addActivity({
        event: 'tier3_result',
        timestamp: getTimestamp(data),
        message,
        data
      });
    }));
    
    unsubscribers.push(websocket.on('tier3_format_selected', (data) => {
      addActivity({
        event: 'tier3_format_selected',
        timestamp: getTimestamp(data),
        message: `Answer format: ${data.format === 'short_form' ? 'Short Form (Single Paper)' : 'Long Form (Volume)'}`,
        data
      });
    }));
    
    // tier3_volume_organized - Backend sends this event name (not tier3_volume_organization_complete)
    unsubscribers.push(websocket.on('tier3_volume_organized', (data) => {
      addActivity({
        event: 'tier3_volume_organized',
        timestamp: getTimestamp(data),
        message: `Volume organized: "${data.title}" (${data.chapters?.length || 0} chapters)`,
        data
      });
    }));
    
    unsubscribers.push(websocket.on('tier3_chapter_started', (data) => {
      addActivity({
        event: 'tier3_chapter_started',
        timestamp: getTimestamp(data),
        message: `Writing chapter ${data.chapter_order}: ${data.title}`,
        data
      });
    }));
    
    unsubscribers.push(websocket.on('tier3_chapter_complete', (data) => {
      addActivity({
        event: 'tier3_chapter_complete',
        timestamp: getTimestamp(data),
        message: `Chapter ${data.chapter_order} complete: ${data.title}`,
        data
      });
    }));
    
    unsubscribers.push(websocket.on('tier3_rejection', (data) => {
      addActivity({
        event: 'tier3_rejection',
        timestamp: getTimestamp(data),
        message: `Tier 3 submission rejected: ${data.phase || 'unknown phase'}`,
        data
      });
    }));
    
    unsubscribers.push(websocket.on('tier3_complete', (data) => {
      addActivity({
        event: 'tier3_complete',
        timestamp: getTimestamp(data),
        message: `🏆 FINAL ANSWER COMPLETE! ${data.format === 'short_form' ? 'Paper' : 'Volume'}: "${data.title}"`,
        data
      });
      // Refresh status to update tier3 info
      autonomousAPI.getStatus().then(setAutonomousStatus).catch(console.error);
    }));
    
    // Reference selection events
    unsubscribers.push(websocket.on('reference_selection_started', (data) => {
      addActivity({
        event: 'reference_selection_started',
        timestamp: getTimestamp(data),
        message: `Reference selection started (${data.mode})`,
        data
      });
    }));
    
    unsubscribers.push(websocket.on('reference_selection_complete', (data) => {
      addActivity({
        event: 'reference_selection_complete',
        timestamp: getTimestamp(data),
        message: `Reference selection complete: ${data.selected_count} papers selected`,
        data
      });
    }));
    
    // Paper writing resumed (after crash recovery)
    unsubscribers.push(websocket.on('paper_writing_resumed', (data) => {
      autonomousTierRef.current = 'tier2_paper_writing';
      addActivity({
        event: 'paper_writing_resumed',
        timestamp: getTimestamp(data),
        message: `Paper writing resumed: ${data.title}`,
        data
      });
    }));
    
    // Tier 3 additional events
    unsubscribers.push(websocket.on('tier3_forced', (data) => {
      addActivity({
        event: 'tier3_forced',
        timestamp: getTimestamp(data),
        message: `Tier 3 forced with mode: ${data.mode} (${data.completed_papers} papers available)`,
        data
      });
    }));
    
    unsubscribers.push(websocket.on('tier3_phase_changed', (data) => {
      addActivity({
        event: 'tier3_phase_changed',
        timestamp: getTimestamp(data),
        message: `Tier 3 phase: ${data.description || data.phase}`,
        data
      });
    }));
    
    unsubscribers.push(websocket.on('tier3_paper_started', (data) => {
      addActivity({
        event: 'tier3_paper_started',
        timestamp: getTimestamp(data),
        message: `Writing final answer paper: ${data.title}`,
        data
      });
    }));
    
    unsubscribers.push(websocket.on('tier3_short_form_complete', (data) => {
      addActivity({
        event: 'tier3_short_form_complete',
        timestamp: getTimestamp(data),
        message: `Short form paper complete: ${data.title}`,
        data
      });
    }));
    
    unsubscribers.push(websocket.on('tier3_long_form_complete', (data) => {
      addActivity({
        event: 'tier3_long_form_complete',
        timestamp: getTimestamp(data),
        message: `Long form volume complete: ${data.title} (${data.total_chapters} chapters)`,
        data
      });
    }));
    
    // OpenRouter privacy error event
    unsubscribers.push(websocket.on('openrouter_privacy_error', (data) => {
      console.error('OpenRouter privacy policy error:', data);
      setPrivacyWarningData(data);
      setShowPrivacyWarning(true);
      
      // Also add to activity log
      addActivity({
        event: 'openrouter_privacy_error',
        timestamp: getTimestamp(data),
        ...data
      });
    }));
    
    // OpenRouter rate limit event
    unsubscribers.push(websocket.on('openrouter_rate_limit', (data) => {
      console.warn('OpenRouter rate limit hit:', data);
      
      // Add to rate-limited models tracking
      setRateLimitedModels(prev => {
        const newMap = new Map(prev);
        newMap.set(data.model, new Date(data.retry_after));
        return newMap;
      });
      
      // Also add to activity log
      addActivity({
        event: 'openrouter_rate_limit',
        timestamp: getTimestamp(data),
        message: `⏳ Rate limit: ${data.model} (retry in 1 hour)`,
        ...data
      });
    }));

    unsubscribers.push(websocket.on('free_model_rotated', (data) => {
      console.info('Free model rotated:', data);
      addActivity({
        event: 'free_model_rotated',
        timestamp: getTimestamp(data),
        message: `🔄 Model rotated: ${data.from_model} → ${data.to_model} (${data.role_id})`,
        ...data
      });
    }));

    unsubscribers.push(websocket.on('free_model_auto_selector_used', (data) => {
      console.info('Free model auto-selector used:', data);
      addActivity({
        event: 'free_model_auto_selector_used',
        timestamp: getTimestamp(data),
        message: `🔄 Auto-selector backup: openrouter/free used for ${data.role_id}`,
        ...data
      });
    }));

    unsubscribers.push(websocket.on('free_models_exhausted', (data) => {
      console.error('All free models exhausted:', data);
      addActivity({
        event: 'free_models_exhausted',
        timestamp: getTimestamp(data),
        message: `❌ All free models exhausted: ${data.message}`,
        ...data
      });
    }));

    unsubscribers.push(websocket.on('account_credits_exhausted', (data) => {
      console.error('Account credits exhausted:', data);
      addActivity({
        event: 'account_credits_exhausted',
        timestamp: getTimestamp(data),
        message: `❌ Account free credits depleted: ${data.message}`,
        ...data
      });
      setCreditExhaustionNotifications(prev => {
        const roleId = data.role_id || 'Account';
        if (prev.some(n => n.role_id === roleId && n.reason === 'account_credits_exhausted')) return prev;
        return [...prev, {
          id: `account_exhausted_${Date.now()}`,
          role_id: roleId,
          reason: 'account_credits_exhausted',
          message: data.message || 'Account free credits depleted.',
          timestamp: getTimestamp(data)
        }];
      });
    }));

    // OpenRouter fallback event (credit exhaustion triggered fallback to LM Studio)
    unsubscribers.push(websocket.on('openrouter_fallback', (data) => {
      console.warn('OpenRouter fallback triggered:', data);
      addActivity({
        event: 'openrouter_fallback',
        timestamp: getTimestamp(data),
        message: `⚠️ OpenRouter credits exhausted for ${data.role_id} — fell back to ${data.fallback_model || 'LM Studio'}`,
        ...data
      });
      setCreditExhaustionNotifications(prev => {
        const reason = data.reason || 'credit_exhaustion';
        if (prev.some(n => n.role_id === data.role_id && n.reason === reason)) return prev;
        return [...prev, {
          id: `fallback_${data.role_id}_${Date.now()}`,
          role_id: data.role_id,
          reason,
          message: data.message,
          fallback_model: data.fallback_model,
          timestamp: getTimestamp(data)
        }];
      });
    }));

    // OpenRouter fallback failed (no fallback configured — role stopped)
    unsubscribers.push(websocket.on('openrouter_fallback_failed', (data) => {
      console.error('OpenRouter fallback failed:', data);
      addActivity({
        event: 'openrouter_fallback_failed',
        timestamp: getTimestamp(data),
        message: `🛑 OpenRouter credits exhausted for ${data.role_id} — NO FALLBACK configured!`,
        ...data
      });
      setCreditExhaustionNotifications(prev => {
        if (prev.some(n => n.role_id === data.role_id && n.reason === 'no_fallback_configured')) return prev;
        return [...prev, {
          id: `fallback_failed_${data.role_id}_${Date.now()}`,
          role_id: data.role_id,
          reason: 'no_fallback_configured',
          message: data.message,
          timestamp: getTimestamp(data)
        }];
      });
    }));

    unsubscribers.push(websocket.on('leanoj_provider_paused', (data) => {
      console.warn('Proof Solver paused for provider credits:', data);
      addActivity({
        event: 'leanoj_provider_paused',
        timestamp: getTimestamp(data),
        message: `Proof Solver paused until OpenRouter credits are reset: ${data.message || data.role_id || 'provider credits exhausted'}`,
        ...data
      });
      setCreditExhaustionNotifications(prev => {
        const roleId = data.role_id || 'Proof Solver';
        if (prev.some(n => n.role_id === roleId && n.reason === 'provider_paused')) return prev;
        return [...prev, {
          id: `leanoj_provider_paused_${roleId}_${Date.now()}`,
          role_id: roleId,
          reason: 'provider_paused',
          message: data.message || 'Proof Solver is paused until OpenRouter credits are reset.',
          timestamp: getTimestamp(data)
        }];
      });
    }));

    unsubscribers.push(websocket.on('leanoj_provider_resumed', (data) => {
      console.info('Proof Solver provider pause resumed:', data);
      addActivity({
        event: 'leanoj_provider_resumed',
        timestamp: getTimestamp(data),
        message: 'Proof Solver resumed after OpenRouter reset.',
        ...data
      });
    }));

    unsubscribers.push(websocket.on('autonomous_proof_provider_paused', (data) => {
      console.warn('Autonomous proof verification paused for provider credits:', data);
      addActivity({
        event: 'autonomous_proof_provider_paused',
        timestamp: getTimestamp(data),
        message: `Autonomous proof verification paused until OpenRouter credits are reset: ${data.message || data.source_id || 'provider credits exhausted'}`,
        ...data
      });
      setCreditExhaustionNotifications(prev => {
        const roleId = `Autonomous Proof (${data.source_id || data.source_type || 'checkpoint'})`;
        if (prev.some(n => n.role_id === roleId && n.reason === 'provider_paused')) return prev;
        return [...prev, {
          id: `auto_proof_provider_paused_${Date.now()}`,
          role_id: roleId,
          reason: 'provider_paused',
          message: data.message || 'Autonomous proof verification is paused until OpenRouter credits are reset.',
          timestamp: getTimestamp(data)
        }];
      });
    }));

    unsubscribers.push(websocket.on('autonomous_proof_provider_resumed', (data) => {
      console.info('Autonomous proof verification resumed:', data);
      addActivity({
        event: 'autonomous_proof_provider_resumed',
        timestamp: getTimestamp(data),
        message: 'Autonomous proof verification resumed after OpenRouter reset.',
        ...data
      });
    }));

    // Boost credits exhausted
    unsubscribers.push(websocket.on('boost_credits_exhausted', (data) => {
      console.warn('Boost credits exhausted:', data);
      addActivity({
        event: 'boost_credits_exhausted',
        timestamp: getTimestamp(data),
        message: `⚠️ Boost credits exhausted for task ${data.task_id}`,
        ...data
      });
      setCreditExhaustionNotifications(prev => {
        if (prev.some(n => n.reason === 'boost_credits_exhausted')) return prev;
        return [...prev, {
          id: `boost_exhausted_${Date.now()}`,
          role_id: `Boost (${data.task_id || 'unknown'})`,
          reason: 'boost_credits_exhausted',
          message: data.message || 'Boost API credits exhausted. Falling back to primary model.',
          timestamp: getTimestamp(data)
        }];
      });
    }));

    unsubscribers.push(websocket.on('openrouter_fallbacks_reset', (data) => {
      console.info('OpenRouter fallbacks reset:', data);
      addActivity({
        event: 'openrouter_fallbacks_reset',
        timestamp: getTimestamp(data),
        message: `OpenRouter reset: ${data.message}`,
        ...data
      });
      setCreditExhaustionNotifications([]);
    }));

    unsubscribers.push(websocket.on('hung_connection_alert', (data) => {
      console.warn('Hung connection alert:', data);
      const event = {
        event: 'hung_connection_alert',
        timestamp: getTimestamp(data),
        message: formatHungConnectionMessage(data),
        data
      };
      if (String(data.role_id || '').toLowerCase().startsWith('leanoj_')) {
        addLeanOJActivityFromGlobalAlert(event);
      } else if (shouldAddHungAlertToAutonomousFeed(data)) {
        addActivity(event);
      }
    }));

    unsubscribers.push(websocket.on('final_answer_complete', (data) => {
      addActivity({
        event: 'final_answer_complete',
        timestamp: getTimestamp(data),
        message: `Final answer complete! Format: ${data.format}`,
        data
      });
      // Refresh status
      autonomousAPI.getStatus().then(setAutonomousStatus).catch(console.error);
    }));
    
    // Paper critique completed event (always fires, updates badge)
    unsubscribers.push(websocket.on('paper_critique_completed', (data) => {
      console.log('Paper critique completed:', data);
      // Refresh papers list to show updated critique rating badge on tile
      autonomousAPI.getPapers().then(res => setPapers(res.papers || [])).catch(console.error);
    }));
    
    // High-score critique notification event (only fires for ratings >= 6.25, shows popup)
    unsubscribers.push(websocket.on('high_score_critique', (data) => {
      console.log('High-score critique received:', data);
      
      // Add to notification stack (max 3, FIFO)
      setCritiqueNotifications(prev => {
        const seenKey = getHighScoreCritiqueNotificationKey(data.paper_id, data.average_rating);
        if (seenKey && (seenHighScoreCritiquesRef.current.has(seenKey) || prev.some(notification => notification.seenKey === seenKey))) {
          return prev;
        }
        if (seenKey) {
          shownHighScoreCritiquesRef.current.add(seenKey);
        }

        const newNotification = {
          id: `critique_${data.paper_id}_${Date.now()}`,
          paper_id: data.paper_id,
          paper_title: data.paper_title,
          average_rating: data.average_rating,
          novelty_rating: data.novelty_rating,
          correctness_rating: data.correctness_rating,
          impact_rating: data.impact_rating,
          timestamp: data.timestamp,
          seenKey
        };
        
        // Add to stack, keep max 3 (remove oldest if full)
        const newStack = [...prev, newNotification];
        if (newStack.length > 3) {
          return newStack.slice(-3); // Keep last 3
        }
        return newStack;
      });
      
      // Also add to activity log
      addActivity({
        event: 'high_score_critique',
        timestamp: getTimestamp(data),
        message: `⭐ High-score critique: ${data.paper_title} (avg: ${data.average_rating})`,
        data
      });
    }));
    
    return () => {
      unsubscribers.forEach(unsub => unsub());
    };
  }, []);

  useEffect(() => {
    const getTimestamp = (data = {}) => data?._serverTimestamp || data?.timestamp || new Date().toISOString();
    const shouldTrackLeanOJModelCall = (data = {}) => {
      const taskId = String(data.task_id || '');
      const roleId = String(data.role_id || '');
      const summary = String(data.result_summary || data.message || '').toLowerCase();
      return !(
        taskId === 'leanoj_sufficiency' ||
        taskId === 'leanoj_path' ||
        taskId === 'leanoj_path_val' ||
        summary.startsWith('sufficiency result:') ||
        summary.startsWith('path result:') ||
        (
          roleId === 'leanoj_path_validator' &&
          (summary.startsWith('decision: accept') || summary.startsWith('decision: reject'))
        )
      );
    };
    const addLeanOJActivity = (event, data = {}, message = '') => {
      setLeanojActivity(prev => [
        ...prev,
        {
          event,
          timestamp: getTimestamp(data),
          message: message || data.message || data.reasoning || data.decision || data.phase || 'Proof Solver update',
          data,
        },
      ].slice(-MAX_LIVE_ACTIVITY_EVENTS));
    };
    const summarizeLeanOJText = (text = '', limit = 220) => {
      const cleaned = String(text || '').replace(/\s+/g, ' ').trim();
      return cleaned.length > limit ? `${cleaned.slice(0, limit)}...` : cleaned;
    };
    const formatModelName = (modelId = '') => {
      const cleaned = String(modelId || '').trim();
      if (!cleaned) return '';
      const displayName = cleaned.split('/').pop() || cleaned;
      return displayName.length > 32 ? `${displayName.slice(0, 32)}...` : displayName;
    };
    const formatLeanOJRole = (roleId = '') => {
      const cleaned = String(roleId || '').replace(/^leanoj_/, '').replace(/_/g, ' ').trim();
      return cleaned ? cleaned.replace(/\b\w/g, (char) => char.toUpperCase()) : 'Proof Solver Model';
    };
    const formatLeanOJDuration = (durationMs) => {
      if (durationMs === null || durationMs === undefined || Number.isNaN(Number(durationMs))) return '';
      const seconds = Number(durationMs) / 1000;
      return seconds >= 60 ? `${(seconds / 60).toFixed(1)}m` : `${seconds.toFixed(1)}s`;
    };
    const formatLeanOJCallResult = (data = {}) => {
      const role = formatLeanOJRole(data.role_id);
      const modelName = formatModelName(data.model) || 'model';
      const summary = summarizeLeanOJText(data.result_summary || data.message || '', 220);
      const attemptSuffix = Number(data.attempt || 1) > 1 ? `, attempt ${data.attempt}` : '';
      const duration = formatLeanOJDuration(data.duration_ms);
      const durationSuffix = duration ? `, ${duration}` : '';
      return `${role} [${modelName}]: ✓ RESULT${attemptSuffix}${durationSuffix}${summary ? ` - ${summary}` : ''}`;
    };
    const formatLeanOJBrainstormMessage = (data = {}, accepted = true) => {
      const submitterId = data.submitter_id ?? data.submitter ?? '?';
      const modelName = formatModelName(data.submitter_model || data.model) || 'N/A';
      const creativityPrefix = data.creativity_emphasized ? '(Creativity Emphasized) ' : '';
      const totalValue = accepted ? data.total_acceptances : data.total_rejections;
      const total = totalValue !== undefined ? ` (total: ${totalValue})` : '';
      const detail = accepted
        ? summarizeLeanOJText(data.submission_preview || data.submission, 160)
        : summarizeLeanOJText(
          data.rejection_reason
            || data.validator_summary
            || data.validator_reasoning
            || data.submission_preview
            || data.submission,
          160
        );
      return `${creativityPrefix}Brainstorm Submitter ${submitterId} [${modelName}]: ${accepted ? '✓ ACCEPTED' : '✗ REJECTED'}${total}${detail ? ` - ${detail}` : ''}`;
    };
    const formatLeanOJTopicValidationMessage = (data = {}, accepted = true) => {
      const submitterId = data.submitter_id ?? data.submitter ?? '?';
      const modelName = formatModelName(data.submitter_model || data.model) || 'N/A';
      const creativityPrefix = data.creativity_emphasized ? '(Creativity Emphasized) ' : '';
      const count = data.accepted_topics !== undefined && data.target_topics !== undefined
        ? ` (${data.accepted_topics}/${data.target_topics})`
        : '';
      const detail = summarizeLeanOJText(data.topic, 160);
      return `${creativityPrefix}Topic Submitter ${submitterId} [${modelName}]: ${accepted ? '✓ ACCEPTED' : '✗ REJECTED'}${count}${detail ? ` - ${detail}` : ''}`;
    };
    const leanOJProofName = (data = {}) => {
      const attempt = data.attempt || {};
      if (data.proof_label) return `Proof ${data.proof_label}`;
      if (data.source_type === 'leanoj_final' || attempt.target === 'final') return 'Final proof';
      if (data.source_type === 'leanoj_subproof' || data.subproof_id || data.subproof || attempt.target === 'subproof') return 'Proof fragment';
      return 'Proof';
    };
    const leanOJProofTarget = (data = {}) => {
      const attempt = data.attempt || {};
      const subproof = data.subproof || {};
      return data.theorem_statement
        || data.theorem_id
        || subproof.theorem_or_lemma
        || subproof.request
        || attempt.request
        || data.subproof_id
        || '';
    };
    const leanOJLeanResponse = (data = {}) => {
      const attempt = data.attempt || {};
      if (data.lean_response) return data.lean_response;
      if (data.proof_verified === true || attempt.success === true) return 'Lean 4 response: proof verified.';
      const error = summarizeLeanOJText(
        attempt.error_output || data.error_summary || data.error_output || data.reason || data.message || '',
        960
      );
      return error ? `Lean 4 response: ${error} - proof not verified.` : 'Lean 4 response: proof not verified.';
    };
    const formatLeanOJNoveltyTier = (tier) => {
      switch (tier) {
        case 'major_mathematical_discovery':
          return 'Major mathematical discovery';
        case 'mathematical_discovery':
          return 'Mathematical discovery';
        case 'novel_variant':
          return 'Novel variant';
        case 'novel_formulation':
          return 'Novel formulation';
        case 'not_novel':
          return 'Not novel';
        case 'novel':
          return 'Novel';
        default:
          return tier ? String(tier).replace(/_/g, ' ') : 'Not rated';
      }
    };
    const leanOJNoveltyMessage = (data = {}) => {
      const tierLabel = formatLeanOJNoveltyTier(data.novelty_tier || (data.is_novel ? 'novel' : 'not_novel'));
      const duplicateNote = data.duplicate ? ' (duplicate proof reused)' : '';
      const reason = summarizeLeanOJText(data.novelty_reasoning || data.reasoning || '', 240);
      const target = leanOJProofTarget(data);
      return `${leanOJProofName(data)} Lean 4 novelty validator rating: ${tierLabel}${duplicateNote}${reason ? ` - ${reason}` : ''}${target ? ` (${target})` : ''}`;
    };
    const leanOJAttemptStartedMessage = (data = {}) => {
      const attemptNumber = data.attempt?.attempt || data.attempt || 1;
      const target = leanOJProofTarget(data);
      return `${leanOJProofName(data)}, Attempt ${attemptNumber} started${target ? `: ${target}` : ''}`;
    };
    const leanOJAttemptFinalMessage = (data = {}) => {
      const attemptNumber = data.attempt?.attempt || data.attempt || '?';
      return `${leanOJProofName(data)}, Attempt ${attemptNumber} final: ${leanOJLeanResponse(data)}`;
    };
    const isLeanOJProofEvent = (data = {}) => {
      const sourceType = String(data.source_type || '');
      const sourceId = String(data.source_id || '');
      const trigger = String(data.trigger || '');
      return sourceType === 'leanoj_final'
        || sourceType === 'leanoj_subproof'
        || sourceId.startsWith('leanoj_')
        || trigger.startsWith('leanoj');
    };
    const addLeanOJSharedProofActivity = (event, data = {}, messageFactory) => {
      if (!isLeanOJProofEvent(data)) return;
      setLeanojProofRefreshToken((prev) => prev + 1);
      addLeanOJActivity(event, data, messageFactory(data));
    };

    const handlers = [
      ['leanoj_started', (data) => {
        setLeanojRunning(true);
        addLeanOJActivity('leanoj_started', data, 'Proof Solver started');
      }],
      ['leanoj_stopped', (data) => {
        setLeanojRunning(false);
        setAnyWorkflowRunning(false);
        addLeanOJActivity('leanoj_stopped', data, 'Proof Solver stopped');
        leanojAPI.getStatus().then(setLeanojStatus).catch(console.error);
      }],
      ['leanoj_status_updated', (data) => setLeanojStatus(data)],
      ['leanoj_phase_changed', (data) => addLeanOJActivity('leanoj_phase_changed', data, `Proof Solver phase: ${data.phase || 'unknown'}`)],
      ['leanoj_model_call_completed', (data) => {
        if (shouldTrackLeanOJModelCall(data)) {
          addLeanOJActivity('leanoj_model_call_completed', data, formatLeanOJCallResult(data));
        }
      }],
      ['leanoj_model_call_failed', (data) => addLeanOJActivity('leanoj_model_call_failed', data, `${formatLeanOJRole(data.role_id)} call failed${data.retryable ? '; retrying' : ''}: ${summarizeLeanOJText(data.message, 160)}`)],
      ['leanoj_role_json_retrying', (data) => addLeanOJActivity('leanoj_role_json_retrying', data, `Proof Solver role ${data.role_id || 'model'} returned invalid JSON; retrying attempt ${data.attempt || '?'}`)],
      ['leanoj_skip_brainstorm_requested', (data) => addLeanOJActivity('leanoj_skip_brainstorm_requested', data, 'Skip brainstorm requested')],
      ['leanoj_brainstorm_skip_deferred', (data) => addLeanOJActivity('leanoj_brainstorm_skip_deferred', data, 'Brainstorm skip queued after topic setup')],
      ['leanoj_brainstorm_skipped', (data) => addLeanOJActivity('leanoj_brainstorm_skipped', data, 'Brainstorm skipped; proceeding directly to proof solving')],
      ['leanoj_force_brainstorm_requested', (data) => addLeanOJActivity('leanoj_force_brainstorm_requested', data, 'Force recursive brainstorm requested')],
      ['leanoj_brainstorm_forced', (data) => addLeanOJActivity('leanoj_brainstorm_forced', data, 'Returning to recursive brainstorm with the current proof preserved')],
      ['leanoj_topic_submitters_started', (data) => addLeanOJActivity('leanoj_topic_submitters_started', data, `Topic submitters started (${data.submitter_count || 0} parallel submitters)`)],
      ['leanoj_topic_generation_started', (data) => addLeanOJActivity('leanoj_topic_generation_started', data, `Submitter ${data.submitter_id ?? data.submitter ?? '?'} generating topic ${data.topic_index || '?'}/${data.target_topics || 5}`)],
      ['leanoj_topic_empty', (data) => addLeanOJActivity('leanoj_topic_empty', data, `Topic submitter ${data.submitter_id ?? data.submitter ?? '?'} returned empty output on attempt ${data.attempt || '?'}`)],
      ['leanoj_topic_candidate_queued', (data) => addLeanOJActivity('leanoj_topic_candidate_queued', data, `Submitter ${data.submitter_id ?? data.submitter ?? '?'} queued topic for validation: ${summarizeLeanOJText(data.topic_preview, 140)}`)],
      ['leanoj_topic_batch_validation_started', (data) => addLeanOJActivity('leanoj_topic_batch_validation_started', data, `Topic validator reviewing batch of ${data.batch_size || 0} topic(s)`)],
      ['leanoj_topic_validated', (data) => addLeanOJActivity('leanoj_topic_validated', data, formatLeanOJTopicValidationMessage(data, true))],
      ['leanoj_topic_rejected', (data) => addLeanOJActivity('leanoj_topic_rejected', data, formatLeanOJTopicValidationMessage(data, false))],
      ['leanoj_recursive_brainstorm_started', (data) => addLeanOJActivity('leanoj_recursive_brainstorm_started', data, `Recursive brainstorm cycle ${data.cycle || '?'} ${data.resumed ? 'resumed' : 'started'}; targeting the current proof attempt`)],
      ['leanoj_topic_submitter_failed', (data) => addLeanOJActivity('leanoj_topic_submitter_failed', data, `Topic submitter ${data.submitter || '?'} failed: ${summarizeLeanOJText(data.message, 160)}`)],
      ['leanoj_recursive_brainstorm_completed', (data) => addLeanOJActivity('leanoj_recursive_brainstorm_completed', data, `Recursive brainstorm cycle ${data.cycle || '?'} completed with ${data.accepted_delta || 0} new accepted ideas`)],
      ['leanoj_initial_topic_selected', (data) => addLeanOJActivity('leanoj_initial_topic_selected', data, `Initial topic: ${summarizeLeanOJText(data.topic, 140)}`)],
      ['leanoj_brainstorm_submitters_started', (data) => addLeanOJActivity('leanoj_brainstorm_submitters_started', data, `Brainstorm submitters started for ${data.phase || 'brainstorm'} (${data.submitter_count || 0} parallel submitters)`)],
      ['leanoj_brainstorm_submission_queued', (data) => addLeanOJActivity('leanoj_brainstorm_submission_queued', data, `Submitter ${data.submitter_id ?? data.submitter ?? '?'} queued brainstorm idea for validation: ${summarizeLeanOJText(data.submission_preview, 140)}`)],
      ['leanoj_brainstorm_submitter_failed', (data) => addLeanOJActivity('leanoj_brainstorm_submitter_failed', data, `Brainstorm submitter ${data.submitter || '?'} failed: ${summarizeLeanOJText(data.message, 160)}`)],
      ['leanoj_brainstorm_batch_validation_started', (data) => addLeanOJActivity('leanoj_brainstorm_batch_validation_started', data, `Brainstorm validator reviewing batch of ${data.batch_size || 0} submission(s)`)],
      ['leanoj_brainstorm_accepted', (data) => addLeanOJActivity('leanoj_brainstorm_accepted', data, formatLeanOJBrainstormMessage(data, true))],
      ['leanoj_brainstorm_rejected', (data) => addLeanOJActivity('leanoj_brainstorm_rejected', data, formatLeanOJBrainstormMessage(data, false))],
      ['leanoj_brainstorm_phase_limit_reached', (data) => addLeanOJActivity('leanoj_brainstorm_phase_limit_reached', data, `Brainstorm phase limit reached for ${data.phase || 'brainstorm'} (${data.accepted_delta || 0}/${data.max_accepts || '?'})`)],
      ['leanoj_brainstorm_prune_review_complete', (data) => addLeanOJActivity('leanoj_brainstorm_prune_review_complete', data, 'Brainstorm prune review complete: no removal needed')],
      ['leanoj_brainstorm_prune_rejected', (data) => addLeanOJActivity('leanoj_brainstorm_prune_rejected', data, `Brainstorm prune rejected: ${summarizeLeanOJText(data.reasoning || data.reason, 140)}`)],
      ['leanoj_brainstorm_prune_applied', (data) => addLeanOJActivity('leanoj_brainstorm_prune_applied', data, `Brainstorm prune applied: ${summarizeLeanOJText(data.reasoning || data.reason, 140)}`)],
      ['leanoj_brainstorm_prune_apply_failed', (data) => addLeanOJActivity('leanoj_brainstorm_prune_apply_failed', data, 'Brainstorm prune apply failed')],
      ['leanoj_brainstorm_prune_error', (data) => addLeanOJActivity('leanoj_brainstorm_prune_error', data, data.message || 'Brainstorm prune review error')],
      ['leanoj_brainstorm_proof_failed', (data) => addLeanOJActivity('leanoj_brainstorm_proof_failed', data, `Brainstorm proof failed Lean gate: ${summarizeLeanOJText(data.feedback?.error_summary, 180)}`)],
      ['leanoj_brainstorm_proof_registration_failed', (data) => addLeanOJActivity('leanoj_brainstorm_proof_registration_failed', data, `Brainstorm proof registration failed: ${summarizeLeanOJText(data.error, 180)}`)],
      ['leanoj_brainstorm_proof_verified', (data) => {
        setLeanojProofRefreshToken((prev) => prev + 1);
        addLeanOJActivity('leanoj_brainstorm_proof_verified', data, `Brainstorm proof verified and accepted: ${leanOJProofTarget(data)}`);
      }],
      ['leanoj_path_decided', (data) => addLeanOJActivity('leanoj_path_decided', data, `Path decision: ${data.decision || ''}`)],
      ['leanoj_partial_proof_saved', (data) => addLeanOJActivity('leanoj_partial_proof_saved', data, `Partial proof saved: ${data.partial_proof?.request || data.partial_proof?.target || ''}`)],
      ['leanoj_master_proof_initialized', (data) => addLeanOJActivity('leanoj_master_proof_initialized', data, 'Proof Solver master proof initialized')],
      ['leanoj_master_proof_edit_started', (data) => addLeanOJActivity('leanoj_master_proof_edit_started', data, `Master proof edit started for final attempt ${data.next_verification_attempt || '?'}`)],
      ['leanoj_master_proof_edit_validation_started', (data) => addLeanOJActivity('leanoj_master_proof_edit_validation_started', data, `Master proof shortening validation started (${data.line_delta_removed || 0} line(s), ${data.char_delta_removed || 0} char(s) removed)`)],
      ['leanoj_master_proof_edit_applied', (data) => addLeanOJActivity('leanoj_master_proof_edit_applied', data, `Master proof edit accepted (version ${data.master_proof_version || '?'})`)],
      ['leanoj_master_proof_edit_rejected', (data) => addLeanOJActivity('leanoj_master_proof_edit_rejected', data, `Master proof edit rejected: ${summarizeLeanOJText(data.validator_feedback || data.error_summary || data.message, 180)}`)],
      ['leanoj_master_proof_stuck', (data) => addLeanOJActivity('leanoj_master_proof_stuck', data, data.continuing_final_cycle ? `Master proof stuck; continuing final cycle (${data.attempts_in_cycle || '?'} / ${data.max_attempts || '?'})` : `Master proof stuck; path requested: ${data.requested_path || 'unknown'}`)],
      ['leanoj_master_proof_progress_watchdog', (data) => addLeanOJActivity('leanoj_master_proof_progress_watchdog', data, data.continuing_final_cycle ? `Master proof watchdog fired; continuing final cycle (${data.attempts_in_cycle || '?'} / ${data.max_attempts || '?'})` : `Master proof watchdog returned to ${data.requested_path || 'path planning'}`)],
      ['leanoj_final_attempt_started', (data) => addLeanOJActivity('leanoj_final_attempt_started', data, leanOJAttemptStartedMessage(data))],
      ['leanoj_final_attempt_failed', (data) => addLeanOJActivity('leanoj_final_attempt_failed', data, leanOJAttemptFinalMessage(data))],
      ['leanoj_final_attempt_cycle_exhausted', (data) => addLeanOJActivity('leanoj_final_attempt_cycle_exhausted', data, data.message || 'Final attempt cycle exhausted; returning to path planning')],
      ['leanoj_final_verified', (data) => {
        setLeanojRunning(false);
        setAnyWorkflowRunning(false);
        setLeanojProofRefreshToken((prev) => prev + 1);
        addLeanOJActivity('leanoj_final_verified', data, `${leanOJProofName(data)} verified and accepted: ${leanOJProofTarget(data) || 'final Proof Solver submission'}`);
        leanojAPI.getStatus().then(setLeanojStatus).catch(console.error);
      }],
      ['proof_check_started', (data) => addLeanOJSharedProofActivity('proof_check_started', data, (eventData) => `Proof check started for ${eventData.source_type} ${eventData.source_id}`)],
      ['proof_check_no_candidates', (data) => addLeanOJSharedProofActivity('proof_check_no_candidates', data, (eventData) => `No formal theorem candidates found in ${eventData.source_type} ${eventData.source_id}`)],
      ['proof_check_candidates_found', (data) => addLeanOJSharedProofActivity('proof_check_candidates_found', data, (eventData) => `Proof candidates found: ${eventData.count || 0}`)],
      ['proof_attempt_started', (data) => addLeanOJSharedProofActivity('proof_attempt_started', data, leanOJAttemptStartedMessage)],
      ['proof_attempt_failed', (data) => addLeanOJSharedProofActivity('proof_attempt_failed', data, leanOJAttemptFinalMessage)],
      ['proof_lean_accepted', (data) => addLeanOJSharedProofActivity('proof_lean_accepted', data, leanOJAttemptFinalMessage)],
      ['proof_integrity_rejected', (data) => addLeanOJSharedProofActivity('proof_integrity_rejected', data, (eventData) => `${leanOJProofName(eventData)} error: integrity rejected - ${summarizeLeanOJText(eventData.reason || leanOJProofTarget(eventData), 960)}`)],
      ['proof_verified', (data) => addLeanOJSharedProofActivity('proof_verified', data, (eventData) => `${leanOJProofName(eventData)} verified and accepted: ${leanOJProofTarget(eventData)}`)],
      ['proof_attempts_exhausted', (data) => addLeanOJSharedProofActivity('proof_attempts_exhausted', data, (eventData) => `${leanOJProofName(eventData)} terminated: proof attempts exhausted for ${leanOJProofTarget(eventData)}`)],
      ['novel_proof_discovered', (data) => addLeanOJSharedProofActivity('novel_proof_discovered', data, leanOJNoveltyMessage)],
      ['known_proof_verified', (data) => addLeanOJSharedProofActivity('known_proof_verified', data, leanOJNoveltyMessage)],
      ['proof_registration_duplicate', (data) => addLeanOJSharedProofActivity('proof_registration_duplicate', data, (eventData) => leanOJNoveltyMessage({ ...eventData, duplicate: true }))],
      ['proof_dependency_added', (data) => addLeanOJSharedProofActivity('proof_dependency_added', data, () => 'Proof Solver proof dependency added')],
      ['proof_check_complete', (data) => addLeanOJSharedProofActivity('proof_check_complete', data, (eventData) => `Proof check complete: ${eventData.verified_count || 0} verified, ${eventData.novel_count || 0} novel`)],
      ['leanoj_error', (data) => addLeanOJActivity('leanoj_error', data, data.message || 'Proof Solver error')],
      ['leanoj_cleared', (data) => {
        setLeanojRunning(false);
        setAnyWorkflowRunning(false);
        setLeanojActivity([]);
        setLeanojStatus(data);
        setLeanojProofRefreshToken((prev) => prev + 1);
      }],
    ];

    handlers.forEach(([event, handler]) => websocket.on(event, handler));
    return () => handlers.forEach(([event, handler]) => websocket.off(event, handler));
  }, []);

  // Poll for autonomous data while running
  useEffect(() => {
    if (!autonomousRunning) return;
    
    const interval = setInterval(async () => {
      try {
        const [status, brainstormsData, papersData, stats] = await Promise.all([
          autonomousAPI.getStatus(),
          autonomousAPI.getBrainstorms(),
          autonomousAPI.getPapers(),
          autonomousAPI.getStats()
        ]);
        
        setAutonomousStatus(status);
        setBrainstorms(brainstormsData.brainstorms || []);
        setPapers(papersData.papers || []);
        setAutonomousStats(stats);
      } catch (error) {
        console.error('Failed to poll autonomous data:', error);
      }
    }, 3000);
    
    return () => clearInterval(interval);
  }, [autonomousRunning]);

  useEffect(() => {
    if (!leanojRunning) return;

    const interval = setInterval(async () => {
      try {
        const status = await leanojAPI.getStatus();
        setLeanojStatus(status);
        if (!status.is_running) {
          setLeanojRunning(false);
        }
      } catch (error) {
        console.error('Failed to poll Proof Solver status:', error);
      }
    }, 3000);

    return () => clearInterval(interval);
  }, [leanojRunning]);
  
  // Clean up expired rate limits every minute
  useEffect(() => {
    const interval = setInterval(() => {
      setRateLimitedModels(prev => {
        const now = new Date();
        const newMap = new Map();
        
        for (const [model, retryAfter] of prev.entries()) {
          if (retryAfter > now) {
            newMap.set(model, retryAfter);
          }
        }
        
        return newMap;
      });
    }, 60000); // Check every minute
    
    return () => clearInterval(interval);
  }, []);

  // Autonomous handlers
  const handleAutonomousStart = async (researchPrompt) => {
    try {
      const lmStudioEnabled = capabilities.lmStudioEnabled;
      const superchargeAllowed = developerModeEnabled;

      // Convert frontend camelCase to backend snake_case for submitter_configs (includes OpenRouter fields)
      const submitterConfigs = autonomousConfig.submitter_configs?.map(cfg => ({
        submitter_id: cfg.submitterId,
        provider: normalizeRuntimeProvider(cfg.provider, lmStudioEnabled),
        model_id: cfg.modelId,
        openrouter_provider: cfg.openrouterProvider || null,
        openrouter_reasoning_effort: cfg.openrouterReasoningEffort || 'auto',
        lm_studio_fallback_id: lmStudioEnabled ? (cfg.lmStudioFallbackId || null) : null,
        context_window: cfg.contextWindow,
        max_output_tokens: cfg.maxOutputTokens,
        supercharge_enabled: superchargeAllowed && Boolean(cfg.superchargeEnabled || cfg.supercharge_enabled)
      })) || [];

      await autonomousAPI.start({
        user_research_prompt: researchPrompt,
        submitter_configs: submitterConfigs,
        creativity_emphasis_boost_enabled: developerModeEnabled && Boolean(autonomousConfig.creativity_emphasis_boost_enabled),
        // Validator config with OpenRouter support
        validator_provider: normalizeRuntimeProvider(
          autonomousConfig.validator_provider,
          lmStudioEnabled
        ),
        validator_model: autonomousConfig.validator_model,
        validator_openrouter_provider: autonomousConfig.validator_openrouter_provider,
        validator_openrouter_reasoning_effort: autonomousConfig.validator_openrouter_reasoning_effort || 'auto',
        validator_lm_studio_fallback: lmStudioEnabled
          ? autonomousConfig.validator_lm_studio_fallback
          : null,
        validator_context_window: autonomousConfig.validator_context_window,
        validator_max_tokens: autonomousConfig.validator_max_tokens,
        validator_supercharge_enabled: superchargeAllowed && Boolean(autonomousConfig.validator_supercharge_enabled),
        // High-context submitter config with OpenRouter support
        high_context_provider: normalizeRuntimeProvider(
          autonomousConfig.high_context_provider,
          lmStudioEnabled
        ),
        high_context_model: autonomousConfig.high_context_model,
        high_context_openrouter_provider: autonomousConfig.high_context_openrouter_provider,
        high_context_openrouter_reasoning_effort: autonomousConfig.high_context_openrouter_reasoning_effort || 'auto',
        high_context_lm_studio_fallback: lmStudioEnabled
          ? autonomousConfig.high_context_lm_studio_fallback
          : null,
        high_context_context_window: autonomousConfig.high_context_context_window,
        high_context_max_tokens: autonomousConfig.high_context_max_tokens,
        high_context_supercharge_enabled: superchargeAllowed && Boolean(autonomousConfig.high_context_supercharge_enabled),
        // High-param submitter config with OpenRouter support
        high_param_provider: normalizeRuntimeProvider(
          autonomousConfig.high_param_provider,
          lmStudioEnabled
        ),
        high_param_model: autonomousConfig.high_param_model,
        high_param_openrouter_provider: autonomousConfig.high_param_openrouter_provider,
        high_param_openrouter_reasoning_effort: autonomousConfig.high_param_openrouter_reasoning_effort || 'auto',
        high_param_lm_studio_fallback: lmStudioEnabled
          ? autonomousConfig.high_param_lm_studio_fallback
          : null,
        high_param_context_window: autonomousConfig.high_param_context_window,
        high_param_max_tokens: autonomousConfig.high_param_max_tokens,
        high_param_supercharge_enabled: superchargeAllowed && Boolean(autonomousConfig.high_param_supercharge_enabled),
        // Critique submitter config with OpenRouter support
        critique_submitter_provider: normalizeRuntimeProvider(
          autonomousConfig.critique_submitter_provider,
          lmStudioEnabled
        ),
        critique_submitter_model: autonomousConfig.critique_submitter_model,
        critique_submitter_openrouter_provider: autonomousConfig.critique_submitter_openrouter_provider,
        critique_submitter_openrouter_reasoning_effort: autonomousConfig.critique_submitter_openrouter_reasoning_effort || 'auto',
        critique_submitter_lm_studio_fallback: lmStudioEnabled
          ? autonomousConfig.critique_submitter_lm_studio_fallback
          : null,
        critique_submitter_context_window: autonomousConfig.critique_submitter_context_window,
        critique_submitter_max_tokens: autonomousConfig.critique_submitter_max_tokens,
        critique_submitter_supercharge_enabled: superchargeAllowed && Boolean(autonomousConfig.critique_submitter_supercharge_enabled),
        allow_mathematical_proofs: !capabilities.genericMode && (autonomousConfig.allow_mathematical_proofs ?? true),
        allow_research_papers: autonomousConfig.allow_research_papers ?? true,
        tier3_enabled: autonomousConfig.tier3_enabled ?? false
      });
      setAutonomousRunning(true);
      setAutonomousStopping(false);
      setAutonomousActivity([]);
      setAnyWorkflowRunning(true);
    } catch (error) {
      alert(`Failed to start autonomous research: ${error.details || error.message}`);
    }
  };

  const handleAutonomousStop = async () => {
    if (autonomousStopping) {
      return;
    }

    setAutonomousStopping(true);
    try {
      await autonomousAPI.stop();
      setAutonomousRunning(false);
      setAnyWorkflowRunning(false);
      const status = await autonomousAPI.getStatus();
      setAutonomousStatus(status);
    } catch (error) {
      alert(`Failed to stop autonomous research: ${error.message}`);
    } finally {
      setAutonomousStopping(false);
    }
  };

  const handleAutonomousClear = async () => {
    if (!window.confirm('Clear all autonomous research data? This cannot be undone.')) {
      return;
    }
    try {
      const result = await autonomousAPI.clear();
      
      // Success - clear frontend state
      setBrainstorms([]);
      setPapers([]);
      setAutonomousActivity([]);
      setAutonomousStats(null);
      
      // Show success message
      if (result.warnings) {
        alert(`Data cleared successfully.\n\nNote: ${result.warnings}`);
      } else {
        alert('All autonomous research data cleared successfully.');
      }
    } catch (error) {
      // Show detailed error message
      const errorMsg = error.details || error.message || 'Unknown error';
      alert(`Failed to clear data:\n\n${errorMsg}\n\nThis may be due to Windows file locking. Try closing file explorer and any programs that may have files open, then try again.`);
    }
  };

  const normalizeLeanOJRoleForCapabilities = (roleConfig = {}) => {
    const lmStudioEnabled = capabilities.lmStudioEnabled;
    const provider = normalizeRuntimeProvider(roleConfig.provider, lmStudioEnabled);
    const shouldResetLmState = !lmStudioEnabled && roleConfig.provider !== 'openrouter';
    return {
      ...roleConfig,
      provider,
      model_id: shouldResetLmState ? '' : (roleConfig.model_id || ''),
      openrouter_provider: shouldResetLmState ? null : (roleConfig.openrouter_provider || null),
      lm_studio_fallback_id: lmStudioEnabled ? (roleConfig.lm_studio_fallback_id || null) : null,
      supercharge_enabled: developerModeEnabled && Boolean(roleConfig.supercharge_enabled),
    };
  };

  const normalizeLeanOJRequestForCapabilities = (request) => ({
    ...request,
    creativity_emphasis_boost_enabled: developerModeEnabled && Boolean(request.creativity_emphasis_boost_enabled),
    topic_generator: normalizeLeanOJRoleForCapabilities(request.topic_generator),
    topic_validator: normalizeLeanOJRoleForCapabilities(request.topic_validator),
    brainstorm_submitters: (request.brainstorm_submitters || []).map(normalizeLeanOJRoleForCapabilities),
    brainstorm_validator: normalizeLeanOJRoleForCapabilities(request.brainstorm_validator),
    path_decider: normalizeLeanOJRoleForCapabilities(request.path_decider || request.final_solver),
    final_solver: normalizeLeanOJRoleForCapabilities(request.final_solver),
  });

  const handleLeanOJStart = async (request) => {
    try {
      await leanojAPI.start(normalizeLeanOJRequestForCapabilities(request));
      setLeanojRunning(true);
      setLeanojActivity([]);
      const status = await leanojAPI.getStatus();
      setLeanojStatus(status);
      setLeanojProofRefreshToken((prev) => prev + 1);
      setAnyWorkflowRunning(true);
    } catch (error) {
      alert(`Failed to start Proof Solver: ${error.details || error.message}`);
    }
  };

  const handleLeanOJStop = async () => {
    try {
      await leanojAPI.stop();
      setLeanojRunning(false);
      setAnyWorkflowRunning(false);
      const status = await leanojAPI.getStatus();
      setLeanojStatus(status);
    } catch (error) {
      alert(`Failed to stop Proof Solver: ${error.message}`);
    }
  };

  const handleLeanOJClear = async () => {
    if (!window.confirm('Clear all saved Proof Solver progress?')) {
      return;
    }
    try {
      const result = await leanojAPI.clear();
      setLeanojRunning(false);
      setAnyWorkflowRunning(false);
      setLeanojActivity([]);
      setLeanojStatus(result.status || null);
      setLeanojProofRefreshToken((prev) => prev + 1);
    } catch (error) {
      alert(`Failed to clear Proof Solver progress: ${error.message}`);
    }
  };

  const handleLeanOJSkipBrainstorm = async () => {
    try {
      const result = await leanojAPI.skipBrainstorm();
      if (result.status) {
        setLeanojStatus(result.status);
      }
    } catch (error) {
      alert(`Failed to skip Proof Solver brainstorming: ${error.message}`);
    }
  };

  const handleLeanOJForceBrainstorm = async () => {
    try {
      const result = await leanojAPI.forceBrainstorm();
      if (result.status) {
        setLeanojStatus(result.status);
      }
    } catch (error) {
      alert(`Failed to force Proof Solver recursive brainstorming: ${error.message}`);
    }
  };

  const refreshBrainstorms = async () => {
    try {
      const data = await autonomousAPI.getBrainstorms();
      setBrainstorms(data.brainstorms || []);
    } catch (error) {
      console.error('Failed to refresh brainstorms:', error);
    }
  };

  const refreshPapers = async () => {
    try {
      const [data, stats] = await Promise.all([
        autonomousAPI.getPapers(),
        autonomousAPI.getStats(),
      ]);
      setPapers(data.papers || []);
      setAutonomousStats(stats);
    } catch (error) {
      console.error('Failed to refresh papers:', error);
    }
  };

  // Determine Final Answer tab label based on Tier 3 status
  const getFinalAnswerLabel = () => {
    if (autonomousStatus?.is_tier3_active) {
      return 'Autonomous Stage 3: FINAL ANSWER IN PROGRESS';
    }
    if (autonomousStatus?.tier3_status === 'complete') {
      return 'Stage 3: FINAL ANSWER COMPLETE ✓';
    }
    return 'Stage 3: Final Answer';
  };
  
  // Critique notification handlers
  const handleDismissNotification = (notificationId) => {
    const notification = critiqueNotifications.find(item => item.id === notificationId);
    markHighScoreCritiqueSeen(notification?.seenKey);
    setCritiqueNotifications(prev => prev.filter(n => n.id !== notificationId));
  };
  
  const handleClickNotification = (paperId, paperTitle, seenKey) => {
    markHighScoreCritiqueSeen(seenKey);
    setSelectedCritiquePaper({ paper_id: paperId, paper_title: paperTitle });
    setShowCritiqueModal(true);
  };
  
  const handleCloseCritiqueModal = () => {
    setShowCritiqueModal(false);
    setSelectedCritiquePaper(null);
  };

  const handleDismissProofNotification = (notificationId) => {
    setProofNotifications(prev => prev.filter(n => n.id !== notificationId));
  };

  const handleClickProofNotification = (proofId) => {
    setSelectedProofId(proofId);
    handleAutonomousTabSelect('auto-proofs');
  };

  const handleModeChange = (nextMode) => {
    setAppMode(nextMode);
  };

  const handleAutonomousTabSelect = (tabId) => {
    setAutonomousActiveTab(tabId);
    if (appMode !== 'autonomous') {
      setAppMode('autonomous');
    }
  };

  const handleManualTabSelect = (tabId) => {
    setManualActiveTab(tabId);
    if (appMode !== 'manual') {
      setAppMode('manual');
    }
  };

  const handleLeanOJTabSelect = (tabId) => {
    setLeanojActiveTab(tabId);
    if (appMode !== 'leanoj') {
      setAppMode('leanoj');
    }
  };

  // Credit exhaustion notification handler
  const handleDismissCreditNotification = (notificationId) => {
    setCreditExhaustionNotifications(prev => prev.filter(n => n.id !== notificationId));
  };

  // Critique modal API functions
  const handleGenerateCritique = async (customPrompt, validatorConfig) => {
    if (!selectedCritiquePaper) return;
    
    const response = await autonomousAPI.generatePaperCritique(
      selectedCritiquePaper.paper_id,
      customPrompt,
      validatorConfig
    );
    return response;
  };
  
  const handleGetCritiques = async () => {
    if (!selectedCritiquePaper) return { critiques: [] };
    
    const response = await autonomousAPI.getPaperCritiques(selectedCritiquePaper.paper_id);
    return response;
  };

  const handleDisclaimerAcknowledge = async () => {
    setShowDisclaimer(false);
    setStartupSetupMessage('');

    const {
      capabilities: nextCapabilities,
      lmAvailable,
      hasOpenRouterKey: keyPresent,
      hasCloudAccess: cloudAccessPresent,
      keyStatusReachable,
      hasUsableLmStudioChatModel,
    } = await syncProviderAvailability();
    if (keyPresent || cloudAccessPresent) {
      return;
    }

    if (!keyStatusReachable) {
      // Backend is still booting (e.g. Lean 4 warm start on a cold Mathlib
      // cache can push this past 20s even though uvicorn itself should be up
      // within seconds). Avoid opening the startup provider setup modal with
      // stale "no key" info — the periodic poller re-checks every 5s and
      // will surface the real state without forcing the user to re-enter a
      // key that is already persisted.
      return;
    }

    const startupChoice = localStorage.getItem(STARTUP_PROVIDER_CHOICE_STORAGE_KEY);
    if (!nextCapabilities.lmStudioEnabled) {
      if (startupChoice === LM_STUDIO_STARTUP_CHOICE) {
        setStartupSetupMessage(
          'This deployment runs in hosted web mode, so LM Studio is intentionally disabled here. Configure OpenRouter to continue.'
        );
      }
      setShowStartupSetupModal(true);
      return;
    }

    if (startupChoice === LM_STUDIO_STARTUP_CHOICE && lmAvailable && hasUsableLmStudioChatModel) {
      return;
    }

    if (startupChoice === LM_STUDIO_STARTUP_CHOICE && (!lmAvailable || !hasUsableLmStudioChatModel)) {
      setStartupSetupMessage(
        'LM Studio was previously selected, but it is not fully ready. Start LM Studio, load nomic-ai/nomic-embed-text-v1.5 and at least one usable local chat model, then try again.'
      );
    }

    setShowStartupSetupModal(true);
  };

  const handleStartupOpenRouterChoice = () => {
    setStartupSetupMessage('');
    setShowStartupSetupModal(false);
    setOpenRouterKeyReason('startup_setup');
    setShowOpenRouterKeyModal(true);
  };

  const handleCloseOpenRouterKeyModal = () => {
    const keyWasJustSaved = openRouterKeyJustSavedRef.current;
    const shouldReturnToStartup = openRouterKeyReason === 'startup_setup' && !keyWasJustSaved && !hasCloudAccess;
    openRouterKeyJustSavedRef.current = false;
    setShowOpenRouterKeyModal(false);

    if (shouldReturnToStartup) {
      setShowStartupSetupModal(true);
    }
  };

  const handleStartupLmStudioChoice = async () => {
    if (!capabilities.lmStudioEnabled) {
      setStartupSetupMessage(
        'LM Studio is intentionally disabled in this deployment. Configure OpenRouter to continue.'
      );
      return;
    }

    setCheckingLmStudioStartupChoice(true);
    setStartupSetupMessage('');

    try {
      const { lmAvailable, hasUsableLmStudioChatModel, defaultLmStudioModelId } = await syncProviderAvailability();

      if (!lmAvailable) {
        setStartupSetupMessage(
          'LM Studio is not detected with a loaded model yet. Install LM Studio, start the local server, load nomic-ai/nomic-embed-text-v1.5, and then try again.'
        );
        return;
      }

      if (!hasUsableLmStudioChatModel || !defaultLmStudioModelId) {
        setStartupSetupMessage(
          'LM Studio is running, but no usable chat model is currently loaded. Load at least one local chat model in addition to nomic-ai/nomic-embed-text-v1.5, then try again.'
        );
        return;
      }

      const { config: nextAutonomousConfig } = applyLmStudioStartupDefaults(defaultLmStudioModelId);
      setAutonomousConfig(nextAutonomousConfig);
      localStorage.setItem(STARTUP_PROVIDER_CHOICE_STORAGE_KEY, LM_STUDIO_STARTUP_CHOICE);
      setShowStartupSetupModal(false);
    } finally {
      setCheckingLmStudioStartupChoice(false);
    }
  };

  const handleOpenRouterKeySet = async () => {
    if (openRouterKeyReason === 'startup_setup') {
      const { config: nextAutonomousConfig } = await applyAutonomousProfileSelection(RECOMMENDED_PROFILE_KEY);
      setAutonomousConfig(nextAutonomousConfig);
      setShowStartupSetupModal(false);
      setStartupSetupMessage('');
    }

    openRouterKeyJustSavedRef.current = true;
    setHasOpenRouterKey(true);
    setHasCloudAccess(true);
    console.log('OpenRouter API key set successfully');
  };

  const mainTabs = [
    { id: 'auto-interface', label: 'Start Here: Autonomous Deep Research Controller', group: 'autonomous-main' },
    { id: 'auto-brainstorms', label: 'Autonomous Stage 1: Brainstorms', group: 'autonomous-main' },
    { id: 'auto-papers', label: 'Autonomous Stage 2: Papers', group: 'autonomous-main' },
    { id: 'auto-proofs', label: 'Mathematical Proofs', group: 'autonomous-main' },
    ...(autonomousConfig.tier3_enabled ? [
      { id: 'auto-final-answer', label: getFinalAnswerLabel(), subtext: '(In Development / Highly Hallucinatory)', group: 'autonomous-main' },
    ] : []),
  ];

  const autonomousSettingsTabs = [
    { id: 'auto-completed-works', label: 'Your Completed Works Library', group: 'autonomous-settings' },
    { id: 'auto-logs', label: 'API Call Logs', group: 'autonomous-settings' },
    { id: 'auto-settings', label: 'Autonomous Model Selection & Settings', group: 'autonomous-settings' },
  ];

  const manualTabs = [
    { id: 'aggregator-interface', label: 'Aggregator', subtext: 'Part 1', subtextClass: 'green', group: 'aggregator' },
    { id: 'aggregator-settings', label: 'Aggregator Settings', group: 'aggregator' },
    { id: 'aggregator-logs', label: 'Aggregator Logs', group: 'aggregator' },
    { id: 'aggregator-results', label: 'Live Results', subtext: 'Part 1 Live Results', subtextClass: 'green', group: 'aggregator' },
    { id: 'compiler-interface', label: 'Compiler', subtext: 'Part 2', subtextClass: 'green', group: 'compiler' },
    { id: 'compiler-settings', label: 'Compiler Settings', group: 'compiler' },
    { id: 'compiler-logs', label: 'Compiler Logs', group: 'compiler' },
    { id: 'compiler-live-paper', label: 'Live Paper', subtext: 'Part 2 Live Results', subtextClass: 'green', group: 'compiler' },
  ];

  const leanojMainTabs = [
    { id: 'leanoj-interface', label: 'Proof Solver', group: 'leanoj-main' },
    { id: 'leanoj-brainstorms', label: 'Brainstorms', group: 'leanoj-main' },
    { id: 'leanoj-master-proof', label: 'Master Proof Draft', group: 'leanoj-main' },
    { id: 'leanoj-proofs', label: 'Mathematical Proofs', group: 'leanoj-main' },
  ];

  const leanojSettingsTabs = [
    { id: 'leanoj-completed-proof-works', label: 'Your Completed Proof Works Library', group: 'leanoj-settings' },
    { id: 'leanoj-logs', label: 'API Call Logs', group: 'leanoj-settings' },
    { id: 'leanoj-settings', label: 'Proof Solver Model Profiles & Settings', group: 'leanoj-settings' },
  ];

  useEffect(() => {
    if (!autonomousConfig.tier3_enabled && autonomousActiveTab === 'auto-final-answer') {
      setAutonomousActiveTab('auto-interface');
    }
  }, [autonomousConfig.tier3_enabled, autonomousActiveTab]);

  // Sync with WorkflowPanel collapse state (stored in localStorage)
  useEffect(() => {
    const handleStorageChange = () => {
      const savedState = localStorage.getItem('workflow_panel_collapsed');
      setWorkflowPanelCollapsed(savedState !== 'false');
    };
    
    const interval = setInterval(handleStorageChange, 500);
    return () => clearInterval(interval);
  }, []);

  // Check if any workflow is running
  useEffect(() => {
    const checkWorkflowStatus = async () => {
      try {
        const [aggStatus, compStatus, autoStatus, leanojCurrentStatus] = await Promise.all([
          api.get('/api/aggregator/status').catch(() => ({ is_running: false })),
          api.get('/api/compiler/status').catch(() => ({ is_running: false })),
          autonomousAPI.getStatus().catch(() => ({ is_running: false })),
          leanojAPI.getStatus().catch(() => ({ is_running: false }))
        ]);
        
        const running = aggStatus.is_running || compStatus.is_running || autoStatus.is_running || leanojCurrentStatus.is_running;
        setAnyWorkflowRunning(running);
      } catch (error) {
        console.error('Failed to check workflow status:', error);
      }
    };
    
    checkWorkflowStatus();
    const interval = setInterval(checkWorkflowStatus, 5000);
    return () => clearInterval(interval);
  }, []);

  return (
    <div className={`app ${workflowPanelCollapsed ? 'workflow-panel-collapsed' : 'workflow-panel-expanded'}`}>
      {/* Banner Section */}
      <div className={`app-banner ${shimmerAccentsEnabled ? '' : 'no-shimmer'}`}>
        <div className="banner-content">
          <h1 className="banner-title">
            <span className="banner-moto" aria-label="M.O.T.O.">M.O.T.O.</span>
            <span className="banner-subtitle">Autonomous ASI</span>
          </h1>
          <p className="banner-company">By Intrafere Research Group</p>
          <p className="banner-variant">A Prototype Artificial Superintelligence - Novelty Seeking Autonomous S.T.E.M. Researcher For Automated Theorem Generation</p>
          <p
            className={`banner-mode-subtitle ${appMode === 'manual' || appMode === 'leanoj' ? '' : 'banner-mode-subtitle--hidden'}`}
            aria-hidden={appMode !== 'manual' && appMode !== 'leanoj'}
          >
            {appMode === 'manual' ? 'MANUAL S.T.E.M. WRITER' : 'Proof Solver Mode'}
          </p>
        </div>
      </div>

      {/* Update Notice Banner — dismissible per session, reappears on restart */}
      {updateNotice && !updateNoticeDismissed && (
        <UpdateNotificationBanner
          notice={updateNotice}
          onDismiss={() => setUpdateNoticeDismissed(true)}
        />
      )}
      
      {/* CRITICAL: Boost buttons are ETERNAL - they NEVER disappear */}
      {/* These buttons are fixed-position, high z-index, and unconditionally rendered */}
      {/* They are visible at program launch and stay visible forever */}
      {/* Slide with WorkflowPanel collapse/expand animation */}
      <div className={`app-header ${workflowPanelCollapsed ? 'panel-collapsed' : ''}`}>
        <div className="mode-switch-control">
          <label className="mode-switch-label" htmlFor="app-mode-select">
            Change Mode
          </label>
          <select
            id="app-mode-select"
            className="mode-switch-select"
            value={appMode}
            onChange={(e) => handleModeChange(e.target.value)}
          >
            <option value="autonomous">Autonomous S.T.E.M. ASI</option>
            <option value="manual">Advanced Manual S.T.E.M. ASI</option>
            {developerModeEnabled && (
              <option value="leanoj">LeanOJ Proof Solver</option>
            )}
          </select>
        </div>
        <div className="boost-control-row">
          <div className="help-tooltip-anchor">
            <button
              type="button"
              className="help-tooltip-btn"
              aria-label="Learn about API Boost"
              onMouseEnter={() => setShowApiBoostTooltip(true)}
              onMouseLeave={() => setShowApiBoostTooltip(false)}
              onFocus={() => setShowApiBoostTooltip(true)}
              onBlur={() => setShowApiBoostTooltip(false)}
            >
              ?
            </button>
            {showApiBoostTooltip && (
              <div className="help-tooltip-popup">
                Use this mode to change your model selections mid-run. It is a good way to use your free daily OpenRouter credits without interrupting your research run. For the easiest setup, select your free model and enable "Use boost as next API call when available." Some free models may be more rate-limited on OpenRouter than others.
              </div>
            )}
          </div>
          <button
            className="boost-btn"
            onClick={() => setShowBoostModal(true)}
            title="Configure API Boost"
          >
            API Boost
          </button>
        </div>
        <button
          className={`header-status-chip ${
            hasCloudAccess === true
              ? 'header-status-chip--ready'
              : hasCloudAccess === false
                ? 'header-status-chip--inactive'
                : 'header-status-chip--pending'
          }`}
          onClick={() => {
            setOpenRouterKeyReason('setup');
            setShowOpenRouterKeyModal(true);
          }}
          title={
            hasCloudAccess === true
              ? 'Cloud access is configured'
              : hasCloudAccess === false
                ? 'Configure Cloud Access & Keys'
                : 'Checking cloud access status...'
          }
        >
          {hasCloudAccess === true
            ? 'Cloud Access & Keys ✓'
            : hasCloudAccess === false
              ? 'Cloud Access & Keys'
              : 'Cloud Access…'}
        </button>
        {capabilities.lmStudioEnabled ? (
          <span
            className={`header-status-chip ${
              lmStudioAvailable ? 'header-status-chip--ready' : 'header-status-chip--inactive'
            }`}
            title={lmStudioAvailable
              ? `LM Studio is online (${lmStudioStatus.model_count || 0} model${(lmStudioStatus.model_count || 0) === 1 ? '' : 's'} loaded)`
              : (lmStudioStatus.error || 'LM Studio server is not reachable at 127.0.0.1:1234')}
          >
            {lmStudioAvailable ? 'LM Studio ✓' : 'LM Studio Offline'}
          </span>
        ) : capabilities.genericMode && (
          <span className="header-status-chip header-status-chip--hosted">
            Hosted Web Mode
          </span>
        )}
        {developerModeEnabled && (
          <span
            className="header-status-chip header-status-chip--ready"
            title="Developer mode settings are enabled. Raw JSON editors are available in settings pages."
          >
            Developer Mode
          </span>
        )}
      </div>
      
      <div className={`tabs ${appMode === 'manual' ? 'tabs-manual' : ''} ${appMode === 'leanoj' ? 'tabs-leanoj' : ''} ${shimmerAccentsEnabled ? 'tabs-shimmer-enabled' : ''}`}>
        {appMode === 'autonomous' ? (
          <>
            {mainTabs.map((tab, index) => {
              const prevTab = mainTabs[index - 1];
              const showSeparator = prevTab && prevTab.group !== tab.group;
              
              // Special styling for Final Answer tab
              const isFinalAnswerTab = tab.id === 'auto-final-answer';
              const tier3Classes = isFinalAnswerTab 
                ? (autonomousStatus?.tier3_status === 'complete' 
                    ? 'tab-tier3-complete' 
                    : (autonomousStatus?.is_tier3_active ? 'tab-tier3-active' : ''))
                : '';
              
              return (
                <React.Fragment key={tab.id}>
                  {showSeparator && <div className="tab-separator" />}
                  <button
                    className={`tab ${activeTab === tab.id ? 'active' : ''} tab-${tab.group} ${tier3Classes} ${tab.subtext ? 'tab-with-subtext' : ''}`}
                    onClick={() => handleAutonomousTabSelect(tab.id)}
                  >
                    <div className="tab-content-wrapper">
                      <span className="tab-main-label">{tab.label}</span>
                      {tab.subtext && <span className={`tab-subtext ${tab.subtextClass || ''}`}>{tab.subtext}</span>}
                    </div>
                  </button>
                </React.Fragment>
              );
            })}
            
            {/* Large spacer for settings group */}
            <div className="tab-group-spacer-large"></div>
            
            {autonomousSettingsTabs.map(tab => {
              return (
                <React.Fragment key={tab.id}>
                  <button
                    className={`tab ${activeTab === tab.id ? 'active' : ''} tab-${tab.group}`}
                    onClick={() => handleAutonomousTabSelect(tab.id)}
                  >
                    {tab.label}
                  </button>
                </React.Fragment>
              );
            })}
          </>
        ) : appMode === 'leanoj' ? (
          <>
            {leanojMainTabs.map((tab) => (
              <React.Fragment key={tab.id}>
                <button
                  className={`tab ${activeTab === tab.id ? 'active' : ''} tab-${tab.group}`}
                  onClick={() => handleLeanOJTabSelect(tab.id)}
                >
                  {tab.label}
                </button>
              </React.Fragment>
            ))}

            <div className="tab-group-spacer-large"></div>

            {leanojSettingsTabs.map((tab) => (
              <React.Fragment key={tab.id}>
                <button
                  className={`tab ${activeTab === tab.id ? 'active' : ''} tab-${tab.group}`}
                  onClick={() => handleLeanOJTabSelect(tab.id)}
                >
                  {tab.label}
                </button>
              </React.Fragment>
            ))}
          </>
        ) : (
          <>
            {manualTabs.map((tab, index) => {
              const prevTab = manualTabs[index - 1];
              const showSeparator = prevTab && prevTab.group !== tab.group;

              return (
                <React.Fragment key={tab.id}>
                  {showSeparator && <div className="tab-separator" />}
                  <button
                    className={`tab ${activeTab === tab.id ? 'active' : ''} tab-${tab.group} ${tab.subtext ? 'tab-with-subtext' : ''}`}
                    onClick={() => handleManualTabSelect(tab.id)}
                  >
                    <div className="tab-content-wrapper">
                      <span className="tab-main-label">{tab.label}</span>
                      {tab.subtext && <span className={`tab-subtext ${tab.subtextClass || ''}`}>{tab.subtext}</span>}
                    </div>
                  </button>
                </React.Fragment>
              );
            })}
          </>
        )}
      </div>
      
      <div className="tab-content">
        <div className="container">
          {/* Rate Limit Warning Banner - Global indicator */}
          {rateLimitedModels.size > 0 && (
            <div className="rate-limit-warning-banner">
              <div className="rate-limit-header">
                <span className="rate-limit-icon">⏳</span>
                <span className="rate-limit-title">
                  OpenRouter free model usage limits in effect - {rateLimitedModels.size} model(s) paused and retrying
                </span>
              </div>
              <div className="rate-limit-details">
                {Array.from(rateLimitedModels.entries()).map(([model, retryAfter]) => {
                  const now = new Date();
                  const minutesRemaining = Math.max(0, Math.ceil((retryAfter - now) / 60000));
                  return (
                    <div key={model} className="rate-limit-model">
                      <span className="rate-limit-model-name">{model}</span>
                      <span className="rate-limit-countdown">
                        Retry in {minutesRemaining} minute{minutesRemaining !== 1 ? 's' : ''}
                      </span>
                    </div>
                  );
                })}
              </div>
            </div>
          )}
          
          {/* Main Tabs Content */}
          {activeTab === 'auto-interface' && (
            <AutonomousResearchInterface
              isRunning={autonomousRunning}
              isStopping={autonomousStopping}
              anyWorkflowRunning={anyWorkflowRunning}
              status={autonomousStatus}
              activity={autonomousActivity}
              onStart={handleAutonomousStart}
              onStop={handleAutonomousStop}
              onClear={handleAutonomousClear}
              config={autonomousConfig}
              onConfigChange={setAutonomousConfig}
              developerModeEnabled={developerModeEnabled}
              capabilities={capabilities}
              api={autonomousAPI}
            />
          )}
          {activeTab === 'auto-brainstorms' && (
            <BrainstormList
              brainstorms={brainstorms}
              onRefresh={refreshBrainstorms}
              api={{ 
                getBrainstorm: autonomousAPI.getBrainstorm,
                deleteBrainstorm: autonomousAPI.deleteBrainstorm
              }}
            />
          )}
          {activeTab === 'auto-papers' && (
            <PaperLibrary
              papers={papers}
              onRefresh={refreshPapers}
              archivedCount={autonomousStats?.paper_counts?.pruned || autonomousStats?.paper_counts?.archived || 0}
              capabilities={capabilities}
              api={{ 
                getAutonomousPaper: autonomousAPI.getAutonomousPaper,
                deletePaper: autonomousAPI.deletePaper,
                deleteAllPrunedPapers: autonomousAPI.deleteAllPrunedPapers
              }}
            />
          )}
          {activeTab === 'auto-proofs' && (
            <MathematicalProofs
              api={autonomousAPI}
              refreshToken={proofRefreshToken}
              selectedProofId={selectedProofId}
              latestDependencyEvent={latestProofDependencyEvent}
            />
          )}
          {activeTab === 'auto-final-answer' && (
            <FinalAnswerView
              api={autonomousAPI}
              isRunning={autonomousRunning}
              status={autonomousStatus}
              capabilities={capabilities}
            />
          )}
          {activeTab === 'auto-completed-works' && (
            <div className="completed-works-library">
              <div className="completed-works-header">
                <h2 className="completed-works-title">Your Completed Works Library</h2>
                <p className="completed-works-subtitle">
                  Browse all research outputs across every session — papers, final answers, and verified proofs.
                </p>
              </div>
              <div className="completed-works-sub-tabs">
                <button
                  className={`completed-works-sub-tab ${completedWorksSubTab === 'stage2-history' ? 'active' : ''}`}
                  onClick={() => setCompletedWorksSubTab('stage2-history')}
                >
                  Stage 2 Papers History
                </button>
                <button
                  className={`completed-works-sub-tab ${completedWorksSubTab === 'stage3-history' ? 'active' : ''}`}
                  onClick={() => setCompletedWorksSubTab('stage3-history')}
                >
                  Stage 3 Final Answers History
                </button>
                <button
                  className={`completed-works-sub-tab ${completedWorksSubTab === 'proof-library' ? 'active' : ''}`}
                  onClick={() => setCompletedWorksSubTab('proof-library')}
                >
                  Proof Library
                </button>
              </div>
              <div className="completed-works-content">
                {completedWorksSubTab === 'stage2-history' && (
                  <Stage2PaperHistory
                    capabilities={capabilities}
                    onCurrentSessionDataChanged={async () => {
                      await Promise.all([refreshPapers(), refreshBrainstorms()]);
                    }}
                  />
                )}
                {completedWorksSubTab === 'stage3-history' && (
                  <FinalAnswerLibrary capabilities={capabilities} />
                )}
                {completedWorksSubTab === 'proof-library' && (
                  <ProofLibrary />
                )}
              </div>
            </div>
          )}
          {activeTab === 'auto-logs' && (
            <AutonomousResearchLogs
              stats={autonomousStats}
              events={autonomousActivity}
            />
          )}

          {activeTab === 'leanoj-interface' && (
            <LeanOJInterface
              isRunning={leanojRunning}
              anyWorkflowRunning={anyWorkflowRunning}
              status={leanojStatus}
              activity={leanojActivity}
              settings={leanojSettings}
              onSettingsChange={setLeanojSettings}
              onStart={handleLeanOJStart}
              onStop={handleLeanOJStop}
              onClear={handleLeanOJClear}
              onSkipBrainstorm={handleLeanOJSkipBrainstorm}
              onForceBrainstorm={handleLeanOJForceBrainstorm}
              developerModeEnabled={developerModeEnabled}
            />
          )}
          {activeTab === 'leanoj-brainstorms' && (
            <LeanOJBrainstorms status={leanojStatus} />
          )}
          {activeTab === 'leanoj-proofs' && (
            <LeanOJMathematicalProofs
              api={leanojAPI}
              status={leanojStatus}
              refreshToken={leanojProofRefreshToken}
            />
          )}
          {activeTab === 'leanoj-master-proof' && (
            <LeanOJMasterProof
              api={leanojAPI}
              status={leanojStatus}
              refreshToken={leanojProofRefreshToken}
            />
          )}
          {activeTab === 'leanoj-completed-proof-works' && (
            <LeanOJProofLibrary
              api={leanojAPI}
              refreshToken={leanojProofRefreshToken}
            />
          )}
          {activeTab === 'leanoj-logs' && (
            <LeanOJLogs />
          )}
          {/* Full-width settings screens with model sidebars are rendered outside the padded tab container. */}
          
          {activeTab === 'aggregator-interface' && (
            <AggregatorInterface
              config={config}
              setConfig={setConfig}
              capabilities={capabilities}
              anyWorkflowRunning={anyWorkflowRunning}
              onWorkflowRunningChange={setAnyWorkflowRunning}
              developerModeEnabled={developerModeEnabled}
            />
          )}
          {/* Full-width settings screens with model sidebars are rendered outside the padded tab container. */}
          {activeTab === 'aggregator-logs' && <AggregatorLogs />}
          {activeTab === 'aggregator-results' && <LiveResults />}
          
          {activeTab === 'compiler-interface' && (
            <CompilerInterface
              activeTab={activeTab}
              capabilities={capabilities}
              anyWorkflowRunning={anyWorkflowRunning}
              onWorkflowRunningChange={setAnyWorkflowRunning}
              developerModeEnabled={developerModeEnabled}
            />
          )}
          {/* Full-width settings screens with model sidebars are rendered outside the padded tab container. */}
          {activeTab === 'compiler-logs' && <CompilerLogs />}
          {activeTab === 'compiler-live-paper' && <LivePaper capabilities={capabilities} />}
        </div>
      </div>
      
      {/* Autonomous Settings - Rendered OUTSIDE tab-content to allow full-width sidebar layout */}
      {activeTab === 'auto-settings' && (
        <AutonomousResearchSettings
          config={autonomousConfig}
          onConfigChange={setAutonomousConfig}
          models={models}
          capabilities={capabilities}
          isRunning={autonomousRunning}
          developerModeEnabled={developerModeEnabled}
        />
      )}

      {activeTab === 'leanoj-settings' && (
        <LeanOJSettings
          settings={leanojSettings}
          onSettingsChange={setLeanojSettings}
          capabilities={capabilities}
          isRunning={leanojRunning}
          developerModeEnabled={developerModeEnabled}
        />
      )}

      {activeTab === 'aggregator-settings' && (
        <AggregatorSettings
          config={config}
          setConfig={setConfig}
          capabilities={capabilities}
          developerModeEnabled={developerModeEnabled}
        />
      )}

      {activeTab === 'compiler-settings' && (
        <CompilerSettings
          capabilities={capabilities}
          developerModeEnabled={developerModeEnabled}
        />
      )}
      
      {/* WorkflowPanel is ETERNAL - always visible for boost controls */}
      {/* The panel shows workflow tasks when running, but boost controls are ALWAYS accessible */}
      {/* Users can configure boost (set next count, toggle categories) at any time */}
      <WorkflowPanel isRunning={anyWorkflowRunning} />
      
      {/* Disclaimer Modal - Shows on every app load */}
      {showDisclaimer && (
        <>
          <div className="disclaimer-overlay" onClick={(e) => e.stopPropagation()} />
          <div className="disclaimer-modal">
            <div className="disclaimer-content">
              <h2 style={{ marginTop: 0, marginBottom: '1.5rem', color: '#1eff1c' }}>
                  Disclaimer & Quickstart
              </h2>
              <p style={{ fontSize: '1.1rem', lineHeight: '1.6', marginBottom: '1.5rem', color: '#1eff1c' }}>
                {capabilities.lmStudioEnabled ? (
                  <>
                    <strong>QUICKSTART:</strong> In LM Studio, load the embedding model <code>nomic-ai/nomic-embed-text-v1.5</code> by <strong>Nomic AI</strong> (optional but recommended), or use only an OpenRouter API key instead of LM Studio. You must leave your PC on and awake during runtime, the program will often run for days without interruption.
                  </>
                ) : (
                  <>
                    <strong>QUICKSTART:</strong> This hosted deployment uses OpenRouter-only inference. Set your OpenRouter API key, choose a profile or role models, and then begin your research run. LM Studio is intentionally disabled in this environment.
                  </>
                )}
              </p>
              <div
                style={{
                  marginBottom: '1.5rem',
                  padding: '1rem 1.1rem',
                  border: '1px solid rgba(30, 255, 28, 0.24)',
                  borderRadius: '10px',
                  background: 'rgba(30, 255, 28, 0.05)',
                }}
              >
                <p
                  style={{
                    margin: '0 0 0.75rem 0',
                    fontSize: '0.82rem',
                    fontWeight: 700,
                    letterSpacing: '0.06em',
                    textTransform: 'uppercase',
                    color: '#1eff1c',
                  }}
                >
                  Legal Disclaimer
                </p>
                <p style={{ fontSize: '0.95rem', lineHeight: '1.5', margin: 0 }}>
                  MOTO is a prototype system under active development. It directs selected AI models to generate novel solution attempts in response to your prompt. Outputs may be incorrect, incomplete, misleading, fabricated, poorly reasoned, or otherwise unsuitable for reliance without independent review, especially for high-stakes, academic, financial, legal, medical, engineering, or operational use.
                  <br />
                  <br />
                  This software and all generated content are provided as-is and at your own risk. By using MOTO, you acknowledge that you are solely responsible for reviewing, validating, and deciding how to use any output, and that the developers, operators, and contributors are not responsible or liable for incorrect solutions, hallucinations, omissions, formatting issues, infinite loops, wasted API calls, model or provider failures, data loss, third-party charges, or any direct or indirect loss, damage, cost, or liability resulting from use of the program or its outputs.
                </p>
              </div>
              <button 
                className="disclaimer-acknowledge-btn"
                onClick={handleDisclaimerAcknowledge}
              >
                I Have Read and Acknowledge This Disclaimer
              </button>
            </div>
          </div>
        </>
      )}

      <StartupProviderSetupModal
        isOpen={showStartupSetupModal}
        capabilities={capabilities}
        lmStudioAvailable={lmStudioAvailable}
        hasUsableLmStudioChatModel={Boolean(lmStudioStatus.has_usable_chat_model)}
        lmStudioModelCount={lmStudioStatus.model_count || 0}
        lmStudioError={lmStudioStatus.error || ''}
        statusMessage={startupSetupMessage}
        isCheckingLmStudio={checkingLmStudioStartupChoice}
        onChooseOpenRouter={handleStartupOpenRouterChoice}
        onConfirmLmStudio={handleStartupLmStudioChoice}
      />
      
      {/* Boost Control Modal */}
      <BoostControlModal 
        isOpen={showBoostModal}
        onClose={() => setShowBoostModal(false)}
        capabilities={capabilities}
        developerModeEnabled={developerModeEnabled}
      />
      
      {/* OpenRouter API Key Modal */}
      <OpenRouterApiKeyModal
        isOpen={showOpenRouterKeyModal}
        onClose={handleCloseOpenRouterKeyModal}
        onKeySet={handleOpenRouterKeySet}
        onCloudAccessChanged={(configured) => setHasCloudAccess(Boolean(configured) || Boolean(hasOpenRouterKey))}
        reason={openRouterKeyReason}
        capabilities={capabilities}
      />
      
      {/* OpenRouter Privacy Warning Modal */}
      <OpenRouterPrivacyWarningModal
        isOpen={showPrivacyWarning}
        onClose={() => setShowPrivacyWarning(false)}
        errorData={privacyWarningData}
        capabilities={capabilities}
      />

      <ProofNotificationStack
        notifications={proofNotifications}
        onDismiss={handleDismissProofNotification}
        onClickNotification={handleClickProofNotification}
        panelCollapsed={workflowPanelCollapsed}
      />
      
      {/* Critique Notification Stack - Persists across all screens */}
      <CritiqueNotificationStack
        notifications={critiqueNotifications}
        onDismiss={handleDismissNotification}
        onClickNotification={handleClickNotification}
        panelCollapsed={workflowPanelCollapsed}
      />
      
      {/* Credit Exhaustion Notification Stack - Persists until user dismisses */}
      <CreditExhaustionNotificationStack
        notifications={creditExhaustionNotifications}
        onDismiss={handleDismissCreditNotification}
        onDismissAll={() => setCreditExhaustionNotifications([])}
      />

      {/* Critique Modal - Opens when notification is clicked */}
      {showCritiqueModal && selectedCritiquePaper && (
        <PaperCritiqueModal
          isOpen={showCritiqueModal}
          onClose={handleCloseCritiqueModal}
          paperType="autonomous_paper"
          paperId={selectedCritiquePaper.paper_id}
          paperTitle={selectedCritiquePaper.paper_title}
          onGenerateCritique={handleGenerateCritique}
          onGetCritiques={handleGetCritiques}
          developerModeEnabled={developerModeEnabled}
        />
      )}
      
      {/* Footer Section */}
      <footer className="app-footer">
        <div className="footer-content">
          <div className="footer-section footer-license">
            <span>MIT License</span>
            <span className="footer-divider">|</span>
            <span className="footer-copyright">© 2026 Intrafere LLC</span>
          </div>
          
          <div className="footer-section footer-links">
            <a
              href="https://intrafere.com/structured-brainstorming-validated-feedback/"
              target="_blank"
              rel="noopener noreferrer"
              className="footer-link footer-link-github"
            >
              How MOTO's Superintelligence Works
            </a>
            <a
              href="https://intrafere.com/order-a-custom-orchestrator/"
              target="_blank"
              rel="noopener noreferrer"
              className="footer-link footer-link-purchase"
            >
              Purchase Custom Industrial-Grade ASI Programs
            </a>
            <a
              href="https://github.com/Intrafere/MOTO-Autonomous-ASI"
              target="_blank"
              rel="noopener noreferrer"
              className="footer-link footer-link-github"
            >
              Star Our GitHub!
            </a>
          </div>
        </div>
      </footer>
    </div>
  );
}

export default App;

