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
  MathematicalProofs
} from './components/autonomous';
import WorkflowPanel from './components/WorkflowPanel';
import BoostControlModal from './components/BoostControlModal';
import StartupProviderSetupModal from './components/StartupProviderSetupModal';
import OpenRouterApiKeyModal from './components/OpenRouterApiKeyModal';
import OpenRouterPrivacyWarningModal from './components/OpenRouterPrivacyWarningModal';
import CritiqueNotificationStack from './components/CritiqueNotificationStack';
import ProofNotificationStack from './components/autonomous/ProofNotificationStack';
import CreditExhaustionNotificationStack from './components/CreditExhaustionNotificationStack';
import HungConnectionNotificationStack from './components/HungConnectionNotificationStack';
import PaperCritiqueModal from './components/PaperCritiqueModal';
import { websocket } from './services/websocket';
import { api, autonomousAPI, openRouterAPI } from './services/api';
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

const APP_MODE_STORAGE_KEY = 'appMode';
const AUTONOMOUS_TAB_STORAGE_KEY = 'autonomousActiveTab';
const MANUAL_TAB_STORAGE_KEY = 'manualActiveTab';
const LEGACY_SINGLE_PAPER_WRITER_STORAGE_KEY = 'singlePaperWriterExpanded';
const EMBEDDING_MODEL_HINTS = ['embed', 'embedding', 'nomic', 'bge', 'e5', 'gte'];
const AUTONOMOUS_ROLE_PREFIXES = ['validator', 'high_context', 'high_param', 'critique_submitter'];
const DEFAULT_CAPABILITIES = Object.freeze({
  genericMode: false,
  lmStudioEnabled: true,
  pdfDownloadAvailable: true,
  version: '',
  buildCommit: '',
  updateChannel: 'main',
  apiContractVersion: '',
});

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
    nextConfig[fallbackKey] = lmStudioEnabled ? (nextConfig[fallbackKey] || null) : null;
  });

  return nextConfig;
}

function App() {
  const [appMode, setAppMode] = useState(() => {
    const savedMode = localStorage.getItem(APP_MODE_STORAGE_KEY);
    if (savedMode === 'autonomous' || savedMode === 'manual') {
      return savedMode;
    }

    const legacyExpanded = localStorage.getItem(LEGACY_SINGLE_PAPER_WRITER_STORAGE_KEY);
    if (!legacyExpanded) {
      return 'autonomous';
    }

    try {
      return JSON.parse(legacyExpanded) ? 'manual' : 'autonomous';
    } catch {
      return 'autonomous';
    }
  });
  const [autonomousActiveTab, setAutonomousActiveTab] = useState('auto-interface');
  const [manualActiveTab, setManualActiveTab] = useState('aggregator-interface');
  const activeTab = appMode === 'manual' ? manualActiveTab : autonomousActiveTab;
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
  const [capabilities, setCapabilities] = useState(DEFAULT_CAPABILITIES);
  
  // Track if any workflow is running (for WorkflowPanel visibility)
  const [anyWorkflowRunning, setAnyWorkflowRunning] = useState(false);
  
  // Track WorkflowPanel collapse state for sliding boost buttons
  const [workflowPanelCollapsed, setWorkflowPanelCollapsed] = useState(() => {
    const savedState = localStorage.getItem('workflow_panel_collapsed');
    return savedState === 'true';
  });

  // Update notice banner state (dismissible per session, re-appears on restart)
  const [updateNotice, setUpdateNotice] = useState(null);
  const [updateNoticeDismissed, setUpdateNoticeDismissed] = useState(false);

  useEffect(() => {
    localStorage.setItem(APP_MODE_STORAGE_KEY, appMode);
    localStorage.setItem(
      LEGACY_SINGLE_PAPER_WRITER_STORAGE_KEY,
      JSON.stringify(appMode === 'manual')
    );
  }, [appMode]);

  useEffect(() => {
    localStorage.setItem(AUTONOMOUS_TAB_STORAGE_KEY, autonomousActiveTab);
  }, [autonomousActiveTab]);

  useEffect(() => {
    localStorage.setItem(MANUAL_TAB_STORAGE_KEY, manualActiveTab);
  }, [manualActiveTab]);
  
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
          submitterConfigs: settings.submitterConfigs || [
            { submitterId: 1, provider: 'lm_studio', modelId: '', openrouterProvider: null, lmStudioFallbackId: null, contextWindow: 131072, maxOutputTokens: 25000 },
            { submitterId: 2, provider: 'lm_studio', modelId: '', openrouterProvider: null, lmStudioFallbackId: null, contextWindow: 131072, maxOutputTokens: 25000 },
            { submitterId: 3, provider: 'lm_studio', modelId: '', openrouterProvider: null, lmStudioFallbackId: null, contextWindow: 131072, maxOutputTokens: 25000 }
          ],
          validatorModel: settings.validatorModel || '',
          validatorProvider: settings.validatorProvider || 'lm_studio',
          validatorOpenrouterProvider: settings.validatorOpenrouterProvider || null,
          validatorLmStudioFallback: settings.validatorLmStudioFallback || null,
          validatorContextSize: settings.validatorContextSize || 131072,
          validatorMaxOutput: settings.validatorMaxOutput || 25000,
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
          submitterConfigs: parsed.submitterConfigs || [
            { submitterId: 1, provider: 'lm_studio', modelId: '', openrouterProvider: null, lmStudioFallbackId: null, contextWindow: 131072, maxOutputTokens: 25000 },
            { submitterId: 2, provider: 'lm_studio', modelId: '', openrouterProvider: null, lmStudioFallbackId: null, contextWindow: 131072, maxOutputTokens: 25000 },
            { submitterId: 3, provider: 'lm_studio', modelId: '', openrouterProvider: null, lmStudioFallbackId: null, contextWindow: 131072, maxOutputTokens: 25000 }
          ],
          validatorModel: parsed.validatorModel || '',
          validatorProvider: parsed.validatorProvider || 'lm_studio',
          validatorOpenrouterProvider: parsed.validatorOpenrouterProvider || null,
          validatorLmStudioFallback: parsed.validatorLmStudioFallback || null,
          validatorContextSize: parsed.validatorContextSize || 131072,
          validatorMaxOutput: parsed.validatorMaxOutput || 25000,
          uploadedFiles: [],
        };
      } catch (e) {
        console.error('Failed to parse saved config:', e);
      }
    }
    return {
      userPrompt: '',
      submitterConfigs: [
        { submitterId: 1, provider: 'lm_studio', modelId: '', openrouterProvider: null, lmStudioFallbackId: null, contextWindow: 131072, maxOutputTokens: 25000 },
        { submitterId: 2, provider: 'lm_studio', modelId: '', openrouterProvider: null, lmStudioFallbackId: null, contextWindow: 131072, maxOutputTokens: 25000 },
        { submitterId: 3, provider: 'lm_studio', modelId: '', openrouterProvider: null, lmStudioFallbackId: null, contextWindow: 131072, maxOutputTokens: 25000 }
      ],
      validatorModel: '',
      validatorProvider: 'lm_studio',
      validatorOpenrouterProvider: null,
      validatorLmStudioFallback: null,
      validatorContextSize: 131072,
      validatorMaxOutput: 25000,
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
      validatorLmStudioFallback: config.validatorLmStudioFallback,
      validatorContextSize: config.validatorContextSize,
      validatorMaxOutput: config.validatorMaxOutput,
    };
    // Save to both old and new keys
    localStorage.setItem('aggregatorConfig', JSON.stringify(configToSave));
    localStorage.setItem('aggregator_settings', JSON.stringify(configToSave));
  }, [config.userPrompt, config.submitterConfigs, config.validatorModel, config.validatorProvider, config.validatorOpenrouterProvider, config.validatorLmStudioFallback, config.validatorContextSize, config.validatorMaxOutput]);

  // Autonomous mode state
  const [autonomousRunning, setAutonomousRunning] = useState(false);
  const [autonomousStatus, setAutonomousStatus] = useState(null);
  const [autonomousActivity, setAutonomousActivity] = useState([]);
  const [brainstorms, setBrainstorms] = useState([]);
  const [papers, setPapers] = useState([]);
  const [autonomousStats, setAutonomousStats] = useState(null);
  
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

  // Hung connection notification state (persistent until dismissed)
  const [hungConnectionNotifications, setHungConnectionNotifications] = useState([]);

  // Live refs used by websocket listeners (which are registered once)
  const autonomousRunningRef = useRef(autonomousRunning);
  const autonomousTierRef = useRef(autonomousStatus?.current_tier || null);
  const openRouterKeyJustSavedRef = useRef(false);

  useEffect(() => {
    autonomousRunningRef.current = autonomousRunning;
  }, [autonomousRunning]);

  useEffect(() => {
    autonomousTierRef.current = autonomousStatus?.current_tier || null;
  }, [autonomousStatus]);

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
        high_context_provider: autonomousConfig.high_context_provider,
        high_context_model: autonomousConfig.high_context_model,
        high_context_openrouter_provider: autonomousConfig.high_context_openrouter_provider,
        high_context_lm_studio_fallback: autonomousConfig.high_context_lm_studio_fallback,
        high_context_context_window: autonomousConfig.high_context_context_window,
        high_context_max_tokens: autonomousConfig.high_context_max_tokens,
        high_param_provider: autonomousConfig.high_param_provider,
        high_param_model: autonomousConfig.high_param_model,
        high_param_openrouter_provider: autonomousConfig.high_param_openrouter_provider,
        high_param_lm_studio_fallback: autonomousConfig.high_param_lm_studio_fallback,
        high_param_context_window: autonomousConfig.high_param_context_window,
        high_param_max_tokens: autonomousConfig.high_param_max_tokens,
        critique_submitter_provider: autonomousConfig.critique_submitter_provider,
        critique_submitter_model: autonomousConfig.critique_submitter_model,
        critique_submitter_openrouter_provider: autonomousConfig.critique_submitter_openrouter_provider,
        critique_submitter_lm_studio_fallback: autonomousConfig.critique_submitter_lm_studio_fallback,
        critique_submitter_context_window: autonomousConfig.critique_submitter_context_window,
        critique_submitter_max_tokens: autonomousConfig.critique_submitter_max_tokens,
      },
      tier3Enabled: autonomousConfig.tier3_enabled ?? existingSettings.tier3Enabled ?? false,
    });
  }, [autonomousConfig]);

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

    const finalHasOpenRouterKey = Boolean(keyStatus.has_key);
    if (keyStatusOk) {
      setHasOpenRouterKey(finalHasOpenRouterKey);
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
      keyStatusReachable: keyStatusOk,
      hasUsableLmStudioChatModel,
      lmStudioStatus: nextLmStudioStatus,
      defaultLmStudioModelId: usableLmStudioChatModelId,
    };
  }, []);

  useEffect(() => {
    syncProviderAvailability();
  }, [syncProviderAvailability]);

  // Fetch update notice from the backend on mount
  useEffect(() => {
    api.getUpdateNotice()
      .then((notice) => {
        if (notice && notice.update_available) {
          setUpdateNotice(notice);
        }
      })
      .catch(() => {});
  }, []);

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

  // Periodically re-check OpenRouter key status to keep indicator in sync.
  // We poll aggressively (5s) because the state mostly flips from "unknown"
  // to "known" shortly after backend startup, and users notice any delay as
  // "my key didn't save."
  useEffect(() => {
    const interval = setInterval(async () => {
      try {
        const keyStatus = await openRouterAPI.getApiKeyStatus();
        setHasOpenRouterKey(Boolean(keyStatus.has_key));
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
        }
      } catch (error) {
        console.error('Failed to check initial autonomous status:', error);
      }
    };
    
    checkInitialStatus();
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
    const MAX_ACTIVITY_EVENTS = 500;
    // Helper to get timestamp from server or fallback to client time
    const getTimestamp = (data) => data?._serverTimestamp || new Date().toISOString();
    const addActivity = (event) => {
      setAutonomousActivity(prev => [...prev, event].slice(-MAX_ACTIVITY_EVENTS));
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
      addActivity({
        event: 'submission_accepted',
        timestamp: getTimestamp(data),
        message: `Submitter ${data.submitter_id} [${modelName}]: ✓ ACCEPTED (total: ${data.total_acceptances})`,
        data
      });
    }));
    
    unsubscribers.push(websocket.on('submission_rejected', (data) => {
      const modelName = data.submitter_model ? (data.submitter_model.split('/')[1] || data.submitter_model.substring(0, 15)) : 'N/A';
      addActivity({
        event: 'submission_rejected',
        timestamp: getTimestamp(data),
        message: `Submitter ${data.submitter_id} [${modelName}]: ✗ REJECTED (total: ${data.total_rejections})`,
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
        message: `Critique phase started (Paper v${data.paper_version || '?'}, target: ${data.target_critiques || 5} critiques)`,
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
    
    unsubscribers.push(websocket.on('body_rewrite_started', (data) => {
      addActivity({
        event: 'body_rewrite_started',
        timestamp: getTimestamp(data),
        message: `REWRITE PHASE: Total rewrite started for Paper v${data.version || '?'}${data.title_changed ? ' (Title updated)' : ''}`,
        data
      });
    }));
    
    unsubscribers.push(websocket.on('partial_revision_complete', (data) => {
      addActivity({
        event: 'partial_revision_complete',
        timestamp: getTimestamp(data),
        message: `PARTIAL REVISION: Applied ${data.edits_applied || 0} targeted edits (Paper v${data.version || '?'})${data.title_changed ? ' (Title updated)' : ''}`,
        data
      });
    }));
    
    unsubscribers.push(websocket.on('critique_phase_ended', (data) => {
      addActivity({
        event: 'critique_phase_ended',
        timestamp: getTimestamp(data),
        message: `Critique phase complete (${data.decision || 'unknown'})`,
        data
      });
    }));
    
    unsubscribers.push(websocket.on('critique_phase_skipped', (data) => {
      addActivity({
        event: 'critique_phase_skipped',
        timestamp: getTimestamp(data),
        message: `Critique phase skipped: ${data.reason || 'user override'}`,
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
      const prefix = data.trigger === 'manual'
        ? 'Manual proof check started'
        : data.trigger === 'retry'
          ? 'Paper-stage proof retry started'
          : 'Proof check started';
      addActivity({
        event: 'proof_check_started',
        timestamp: getTimestamp(data),
        message: `${prefix} for ${data.source_type} ${data.source_id}`,
        data
      });
    }));

    unsubscribers.push(websocket.on('proof_retry_scheduled', (data) => {
      addActivity({
        event: 'proof_retry_scheduled',
        timestamp: getTimestamp(data),
        message: `Scheduled ${data.count || 0} proof retry candidate(s) for paper ${data.source_id}`,
        data
      });
    }));

    unsubscribers.push(websocket.on('proof_retry_started', (data) => {
      addActivity({
        event: 'proof_retry_started',
        timestamp: getTimestamp(data),
        message: `Retrying ${data.count || 0} failed proof candidate(s) against paper ${data.source_id}`,
        data
      });
    }));

    unsubscribers.push(websocket.on('proof_check_no_candidates', (data) => {
      addActivity({
        event: 'proof_check_no_candidates',
        timestamp: getTimestamp(data),
        message: `No formal proof candidates found in ${data.source_type} ${data.source_id}`,
        data
      });
    }));

    unsubscribers.push(websocket.on('proof_check_candidates_found', (data) => {
      addActivity({
        event: 'proof_check_candidates_found',
        timestamp: getTimestamp(data),
        message: `Proof check found ${data.count || 0} theorem candidate(s)`,
        data
      });
    }));

    unsubscribers.push(websocket.on('proof_attempt_started', (data) => {
      addActivity({
        event: 'proof_attempt_started',
        timestamp: getTimestamp(data),
        message: `Proof attempt ${data.attempt || 1} started: ${data.theorem_statement || data.theorem_id}`,
        data
      });
    }));

    unsubscribers.push(websocket.on('smt_check_started', (data) => {
      addActivity({
        event: 'smt_check_started',
        timestamp: getTimestamp(data),
        message: `SMT check started: ${data.theorem_statement || data.theorem_id}`,
        data
      });
    }));

    unsubscribers.push(websocket.on('smt_check_complete', (data) => {
      addActivity({
        event: 'smt_check_complete',
        timestamp: getTimestamp(data),
        message: `SMT check complete (${data.result || 'unknown'}): ${data.theorem_statement || data.theorem_id}`,
        data
      });
    }));

    unsubscribers.push(websocket.on('proof_attempt_failed', (data) => {
      addActivity({
        event: 'proof_attempt_failed',
        timestamp: getTimestamp(data),
        message: `Proof attempt ${data.attempt || '?'} failed: ${formatReason(data.error_summary, 960) || data.theorem_statement || data.theorem_id}`,
        data
      });
    }));

    unsubscribers.push(websocket.on('proof_verified', (data) => {
      addActivity({
        event: 'proof_verified',
        timestamp: getTimestamp(data),
        message: `Lean 4 verified: ${data.theorem_statement || data.theorem_id}`,
        data
      });
    }));

    unsubscribers.push(websocket.on('proof_attempts_exhausted', (data) => {
      addActivity({
        event: 'proof_attempts_exhausted',
        timestamp: getTimestamp(data),
        message: `Proof attempts exhausted: ${data.theorem_statement || data.theorem_id}`,
        data
      });
    }));

    unsubscribers.push(websocket.on('novel_proof_discovered', (data) => {
      setProofRefreshToken((prev) => prev + 1);
      setProofNotifications((prev) => {
        const next = [
          ...prev,
          {
            id: `proof_${data.proof_id}_${Date.now()}`,
            proof_id: data.proof_id,
            theorem_statement: data.theorem_statement,
            source_type: data.source_type,
            source_id: data.source_id,
            timestamp: getTimestamp(data),
          }
        ];
        return next.length > 3 ? next.slice(-3) : next;
      });
      addActivity({
        event: 'novel_proof_discovered',
        timestamp: getTimestamp(data),
        message: `Novel proof discovered: ${data.theorem_statement}`,
        data
      });
    }));

    unsubscribers.push(websocket.on('known_proof_verified', (data) => {
      setProofRefreshToken((prev) => prev + 1);
      addActivity({
        event: 'known_proof_verified',
        timestamp: getTimestamp(data),
        message: `Verified known proof recorded for ${data.source_type} ${data.source_id}`,
        data
      });
    }));

    unsubscribers.push(websocket.on('proof_dependency_added', (data) => {
      setLatestProofDependencyEvent(data);
      setProofRefreshToken((prev) => prev + 1);
      addActivity({
        event: 'proof_dependency_added',
        timestamp: getTimestamp(data),
        message: `Dependency graph updated for ${data.theorem_name || data.proof_id}`,
        data
      });
    }));

    unsubscribers.push(websocket.on('proof_check_complete', (data) => {
      setProofRefreshToken((prev) => prev + 1);
      addActivity({
        event: 'proof_check_complete',
        timestamp: getTimestamp(data),
        message: `Proof check complete: ${data.verified_count || 0} verified, ${data.novel_count || 0} novel`,
        data
      });
    }));
    
    unsubscribers.push(websocket.on('auto_research_started', () => {
      setAutonomousActivity([]);
      setAutonomousRunning(true);
    }));
    
    unsubscribers.push(websocket.on('auto_research_resumed', (data) => {
      // Handle resume after crash/restart - sync running state
      console.log('Autonomous research resumed:', data);
      setAutonomousRunning(true);
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
      autonomousTierRef.current = null;
      setHungConnectionNotifications([]);
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

    unsubscribers.push(websocket.on('serial_bottleneck_paused', (data) => {
      console.warn('Serial bottleneck - workflow paused:', data);
      addActivity({
        event: 'serial_bottleneck_paused',
        timestamp: getTimestamp(data),
        message: `⏸️ SERIAL BOTTLENECK: ${data.role_id} paused for ${Math.round((data.wait_seconds || 0) / 60)} min`,
        ...data
      });
    }));

    unsubscribers.push(websocket.on('serial_bottleneck_resumed', (data) => {
      console.info('Serial bottleneck resolved:', data);
      addActivity({
        event: 'serial_bottleneck_resumed',
        timestamp: getTimestamp(data),
        message: `▶️ SERIAL BOTTLENECK resolved: ${data.role_id} resumed`,
        ...data
      });
    }));

    unsubscribers.push(websocket.on('all_free_models_exhausted', (data) => {
      console.error('All free models exhausted:', data);
      addActivity({
        event: 'all_free_models_exhausted',
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
      setHungConnectionNotifications([]);
    }));

    unsubscribers.push(websocket.on('hung_connection_alert', (data) => {
      console.warn('Hung connection alert:', data);
      addLog({
        type: 'warning',
        message: `⏳ Possible hung connection: ${data.model} via ${data.provider} (${data.elapsed_minutes}+ min)`,
        ...data
      });
      setHungConnectionNotifications(prev => {
        if (prev.some(n => n.role_id === data.role_id)) return prev;
        return [...prev, {
          id: `hung_${data.role_id}_${Date.now()}`,
          ...data,
          timestamp: Date.now()
        }];
      });
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
        const newNotification = {
          id: `critique_${data.paper_id}_${Date.now()}`,
          paper_id: data.paper_id,
          paper_title: data.paper_title,
          average_rating: data.average_rating,
          novelty_rating: data.novelty_rating,
          correctness_rating: data.correctness_rating,
          impact_rating: data.impact_rating,
          timestamp: data.timestamp
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

      // Convert frontend camelCase to backend snake_case for submitter_configs (includes OpenRouter fields)
      const submitterConfigs = autonomousConfig.submitter_configs?.map(cfg => ({
        submitter_id: cfg.submitterId,
        provider: normalizeRuntimeProvider(cfg.provider, lmStudioEnabled),
        model_id: cfg.modelId,
        openrouter_provider: cfg.openrouterProvider || null,
        lm_studio_fallback_id: lmStudioEnabled ? (cfg.lmStudioFallbackId || null) : null,
        context_window: cfg.contextWindow,
        max_output_tokens: cfg.maxOutputTokens
      })) || [];

      await autonomousAPI.start({
        user_research_prompt: researchPrompt,
        submitter_configs: submitterConfigs,
        // Validator config with OpenRouter support
        validator_provider: normalizeRuntimeProvider(
          autonomousConfig.validator_provider,
          lmStudioEnabled
        ),
        validator_model: autonomousConfig.validator_model,
        validator_openrouter_provider: autonomousConfig.validator_openrouter_provider,
        validator_lm_studio_fallback: lmStudioEnabled
          ? autonomousConfig.validator_lm_studio_fallback
          : null,
        validator_context_window: autonomousConfig.validator_context_window,
        validator_max_tokens: autonomousConfig.validator_max_tokens,
        // High-context submitter config with OpenRouter support
        high_context_provider: normalizeRuntimeProvider(
          autonomousConfig.high_context_provider,
          lmStudioEnabled
        ),
        high_context_model: autonomousConfig.high_context_model,
        high_context_openrouter_provider: autonomousConfig.high_context_openrouter_provider,
        high_context_lm_studio_fallback: lmStudioEnabled
          ? autonomousConfig.high_context_lm_studio_fallback
          : null,
        high_context_context_window: autonomousConfig.high_context_context_window,
        high_context_max_tokens: autonomousConfig.high_context_max_tokens,
        // High-param submitter config with OpenRouter support
        high_param_provider: normalizeRuntimeProvider(
          autonomousConfig.high_param_provider,
          lmStudioEnabled
        ),
        high_param_model: autonomousConfig.high_param_model,
        high_param_openrouter_provider: autonomousConfig.high_param_openrouter_provider,
        high_param_lm_studio_fallback: lmStudioEnabled
          ? autonomousConfig.high_param_lm_studio_fallback
          : null,
        high_param_context_window: autonomousConfig.high_param_context_window,
        high_param_max_tokens: autonomousConfig.high_param_max_tokens,
        // Critique submitter config with OpenRouter support
        critique_submitter_provider: normalizeRuntimeProvider(
          autonomousConfig.critique_submitter_provider,
          lmStudioEnabled
        ),
        critique_submitter_model: autonomousConfig.critique_submitter_model,
        critique_submitter_openrouter_provider: autonomousConfig.critique_submitter_openrouter_provider,
        critique_submitter_lm_studio_fallback: lmStudioEnabled
          ? autonomousConfig.critique_submitter_lm_studio_fallback
          : null,
        critique_submitter_context_window: autonomousConfig.critique_submitter_context_window,
        critique_submitter_max_tokens: autonomousConfig.critique_submitter_max_tokens,
        tier3_enabled: autonomousConfig.tier3_enabled ?? false
      });
      setAutonomousRunning(true);
      setAutonomousActivity([]);
    } catch (error) {
      alert(`Failed to start autonomous research: ${error.details || error.message}`);
    }
  };

  const handleAutonomousStop = async () => {
    try {
      await autonomousAPI.stop();
      setAutonomousRunning(false);
    } catch (error) {
      alert(`Failed to stop autonomous research: ${error.message}`);
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
      const data = await autonomousAPI.getPapers();
      setPapers(data.papers || []);
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
    setCritiqueNotifications(prev => prev.filter(n => n.id !== notificationId));
  };
  
  const handleClickNotification = (paperId, paperTitle) => {
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

  // Credit exhaustion notification handler
  const handleDismissCreditNotification = (notificationId) => {
    setCreditExhaustionNotifications(prev => prev.filter(n => n.id !== notificationId));
  };

  // Hung connection notification handler
  const handleDismissHungNotification = (notificationId) => {
    setHungConnectionNotifications(prev => prev.filter(n => n.id !== notificationId));
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
      keyStatusReachable,
      hasUsableLmStudioChatModel,
    } = await syncProviderAvailability();
    if (keyPresent) {
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
    const shouldReturnToStartup = openRouterKeyReason === 'startup_setup' && !keyWasJustSaved && !hasOpenRouterKey;
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
    { id: 'auto-stage2-history', label: 'Stage 2 Final Answers History', group: 'autonomous-settings' },
    { id: 'auto-final-answer-library', label: 'Stage 3 Final Answers History', subtext: '(In Development / Highly Hallucinatory)', group: 'autonomous-settings' },
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

  useEffect(() => {
    if (!autonomousConfig.tier3_enabled && autonomousActiveTab === 'auto-final-answer') {
      setAutonomousActiveTab('auto-interface');
    }
  }, [autonomousConfig.tier3_enabled, autonomousActiveTab]);

  // Sync with WorkflowPanel collapse state (stored in localStorage)
  useEffect(() => {
    const handleStorageChange = () => {
      const savedState = localStorage.getItem('workflow_panel_collapsed');
      setWorkflowPanelCollapsed(savedState === 'true');
    };
    
    const interval = setInterval(handleStorageChange, 500);
    return () => clearInterval(interval);
  }, []);

  // Check if any workflow is running
  useEffect(() => {
    const checkWorkflowStatus = async () => {
      try {
        const [aggStatus, compStatus, autoStatus] = await Promise.all([
          api.get('/api/aggregator/status').catch(() => ({ is_running: false })),
          api.get('/api/compiler/status').catch(() => ({ is_running: false })),
          autonomousAPI.getStatus().catch(() => ({ is_running: false }))
        ]);
        
        const running = aggStatus.is_running || compStatus.is_running || autoStatus.is_running;
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
        </div>
      </div>

      {/* Update Notice Banner — dismissible per session, reappears on restart */}
      {updateNotice && !updateNoticeDismissed && (
        <div className="update-notice-banner">
          <div className="update-notice-content">
            <span className="update-notice-icon">&#9432;</span>
            <span className="update-notice-text">
              <strong>Update available:</strong>{' '}
              {updateNotice.installed_version} ({updateNotice.installed_commit})
              {' '}&rarr;{' '}
              {updateNotice.available_version} ({updateNotice.available_commit})
              {' '}&mdash;{' '}
              <span className="update-notice-detail">
                {updateNotice.can_auto_apply
                  ? 'Restart the launcher to apply this update.'
                  : `Install layout: ${updateNotice.install_layout}. Pull the latest from GitHub main to update.`}
              </span>
            </span>
          </div>
          <button
            className="update-notice-dismiss"
            onClick={() => setUpdateNoticeDismissed(true)}
            aria-label="Dismiss update notice"
            title="Dismiss"
          >
            &#10005;
          </button>
        </div>
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
            hasOpenRouterKey === true
              ? 'header-status-chip--ready'
              : hasOpenRouterKey === false
                ? 'header-status-chip--inactive'
                : 'header-status-chip--pending'
          }`}
          onClick={() => {
            setOpenRouterKeyReason('setup');
            setShowOpenRouterKeyModal(true);
          }}
          title={
            hasOpenRouterKey === true
              ? 'OpenRouter API key is configured'
              : hasOpenRouterKey === false
                ? 'Configure OpenRouter API Key'
                : 'Checking OpenRouter key status...'
          }
        >
          {hasOpenRouterKey === true
            ? 'OpenRouter ✓'
            : hasOpenRouterKey === false
              ? 'Set OpenRouter Key'
              : 'OpenRouter…'}
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
      </div>
      
      <div className={`tabs ${appMode === 'manual' ? 'tabs-manual' : ''} ${shimmerAccentsEnabled ? 'tabs-shimmer-enabled' : ''}`}>
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
              anyWorkflowRunning={anyWorkflowRunning}
              status={autonomousStatus}
              activity={autonomousActivity}
              onStart={handleAutonomousStart}
              onStop={handleAutonomousStop}
              onClear={handleAutonomousClear}
              config={autonomousConfig}
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
              archivedCount={autonomousStats?.paper_counts?.archived || 0}
              api={{ 
                getAutonomousPaper: autonomousAPI.getAutonomousPaper,
                deletePaper: autonomousAPI.deletePaper
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
            />
          )}
          {activeTab === 'auto-stage2-history' && (
            <Stage2PaperHistory
              onCurrentSessionDataChanged={async () => {
                await Promise.all([refreshPapers(), refreshBrainstorms()]);
              }}
            />
          )}
          {activeTab === 'auto-final-answer-library' && (
            <FinalAnswerLibrary />
          )}
          {activeTab === 'auto-logs' && (
            <AutonomousResearchLogs
              stats={autonomousStats}
              events={autonomousActivity}
            />
          )}
          
          {activeTab === 'aggregator-interface' && (
            <AggregatorInterface
              config={config}
              setConfig={setConfig}
              capabilities={capabilities}
              anyWorkflowRunning={anyWorkflowRunning}
            />
          )}
          {activeTab === 'aggregator-settings' && (
            <AggregatorSettings
              config={config}
              setConfig={setConfig}
              capabilities={capabilities}
            />
          )}
          {activeTab === 'aggregator-logs' && <AggregatorLogs />}
          {activeTab === 'aggregator-results' && <LiveResults />}
          
          {activeTab === 'compiler-interface' && (
            <CompilerInterface
              activeTab={activeTab}
              capabilities={capabilities}
              anyWorkflowRunning={anyWorkflowRunning}
            />
          )}
          {activeTab === 'compiler-settings' && (
            <CompilerSettings capabilities={capabilities} />
          )}
          {activeTab === 'compiler-logs' && <CompilerLogs />}
          {activeTab === 'compiler-live-paper' && <LivePaper />}
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
      />
      
      {/* OpenRouter API Key Modal */}
      <OpenRouterApiKeyModal
        isOpen={showOpenRouterKeyModal}
        onClose={handleCloseOpenRouterKeyModal}
        onKeySet={handleOpenRouterKeySet}
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
      />
      
      {/* Critique Notification Stack - Persists across all screens */}
      <CritiqueNotificationStack
        notifications={critiqueNotifications}
        onDismiss={handleDismissNotification}
        onClickNotification={handleClickNotification}
      />
      
      {/* Credit Exhaustion Notification Stack - Persists until user dismisses */}
      <CreditExhaustionNotificationStack
        notifications={creditExhaustionNotifications}
        onDismiss={handleDismissCreditNotification}
        onDismissAll={() => setCreditExhaustionNotifications([])}
      />

      {/* Hung Connection Notification Stack - Persists until user dismisses */}
      <HungConnectionNotificationStack
        notifications={hungConnectionNotifications}
        onDismiss={handleDismissHungNotification}
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
              <span className="footer-icon">ℹ️</span>
              How MOTO's Superintelligence Works
            </a>
            <a
              href="https://intrafere.com/order-a-custom-orchestrator/"
              target="_blank"
              rel="noopener noreferrer"
              className="footer-link footer-link-purchase"
            >
              Purchase a Custom ASI Program
            </a>
            <a
              href="https://github.com/"
              target="_blank"
              rel="noopener noreferrer"
              className="footer-link footer-link-github"
            >
              <span className="footer-icon">⭐</span>
              Visit MOTO's GitHub (Star Us for More ASI Programs)
            </a>
          </div>
        </div>
      </footer>
    </div>
  );
}

export default App;

