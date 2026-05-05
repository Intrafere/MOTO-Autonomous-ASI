import { loadModelCache, getModelApiId } from './modelCache';

export const AUTONOMOUS_SETTINGS_STORAGE_KEY = 'autonomous_research_settings';
export const AUTONOMOUS_PROFILES_STORAGE_KEY = 'autonomous_research_profiles';
export const STARTUP_PROVIDER_CHOICE_STORAGE_KEY = 'startup_provider_choice';
export const LM_STUDIO_STARTUP_CHOICE = 'lm_studio';
export const RECOMMENDED_PROFILE_KEY = 'recommended_slower_affordable_higher_knowledge';
export const RECOMMENDED_ALTERNATE_PROFILE_KEY = 'recommended_fast_affordable_mid';
export const RECOMMENDED_LAB_FAST_PROFILE_KEY = 'recommended_lab_fast_costly_extra_high';
export const RECOMMENDED_LAB_MAX_PROFILE_KEY = 'recommended_lab_slow_costly_max';
export const RECOMMENDED_ENTRY_LAB_PROFILE_KEY = 'recommended_entry_lab_fast_less_affordable';
export const RECOMMENDED_PROFILE_KEYS = [
  RECOMMENDED_PROFILE_KEY,
  RECOMMENDED_ALTERNATE_PROFILE_KEY,
  RECOMMENDED_ENTRY_LAB_PROFILE_KEY,
  RECOMMENDED_LAB_FAST_PROFILE_KEY,
  RECOMMENDED_LAB_MAX_PROFILE_KEY,
];

const DEFAULT_SUBMITTER_CONFIG = {
  submitterId: 1,
  provider: 'lm_studio',
  modelId: '',
  openrouterProvider: null,
  lmStudioFallbackId: null,
  contextWindow: 131072,
  maxOutputTokens: 25000,
};

// NOTE: DEFAULT_OPENROUTER_SUBMITTER_CONFIGS and DEFAULT_LOCAL_CONFIG are derived
// from RECOMMENDED_PROFILES[RECOMMENDED_PROFILE_KEY] further below so the "default"
// startup configuration and the selectable recommended profile stay in sync.
// Update the recommended profile below to change what a fresh install runs with.

const DEFAULT_LM_LOCAL_CONFIG = {
  validator_provider: 'lm_studio',
  validator_model: '',
  validator_openrouter_provider: null,
  validator_lm_studio_fallback: null,
  validator_context_window: 131072,
  validator_max_tokens: 25000,
  high_context_provider: 'lm_studio',
  high_context_model: '',
  high_context_openrouter_provider: null,
  high_context_lm_studio_fallback: null,
  high_context_context_window: 131072,
  high_context_max_tokens: 25000,
  high_param_provider: 'lm_studio',
  high_param_model: '',
  high_param_openrouter_provider: null,
  high_param_lm_studio_fallback: null,
  high_param_context_window: 131072,
  high_param_max_tokens: 25000,
  critique_submitter_provider: 'lm_studio',
  critique_submitter_model: '',
  critique_submitter_openrouter_provider: null,
  critique_submitter_lm_studio_fallback: null,
  critique_submitter_context_window: 131072,
  critique_submitter_max_tokens: 25000,
};

const createDefaultSubmitterConfigs = (modelId = '') => (
  [1, 2, 3].map((submitterId) => ({
    ...DEFAULT_SUBMITTER_CONFIG,
    submitterId,
    modelId,
  }))
);

export const RECOMMENDED_PROFILES = {
  [RECOMMENDED_PROFILE_KEY]: {
    name: 'Slow, less affordable, higher knowledge',
    numSubmitters: 3,
    submitters: [
      {
        modelId: 'google/gemini-3.1-pro-preview',
        provider: 'openrouter',
        openrouterProvider: null,
        lmStudioFallbackId: null,
        contextWindow: 1048576,
        maxOutputTokens: 65500,
      },
      {
        modelId: 'moonshotai/kimi-k2.6',
        provider: 'openrouter',
        openrouterProvider: null,
        lmStudioFallbackId: null,
        contextWindow: 262000,
        maxOutputTokens: 40000,
      },
      {
        modelId: 'deepseek/deepseek-v4-pro',
        provider: 'openrouter',
        openrouterProvider: null,
        lmStudioFallbackId: null,
        contextWindow: 1048576,
        maxOutputTokens: 65500,
      },
    ],
    validator: {
      modelId: 'moonshotai/kimi-k2.6',
      provider: 'openrouter',
      openrouterProvider: null,
      lmStudioFallbackId: null,
      contextWindow: 262000,
      maxOutputTokens: 40000,
    },
    highContext: {
      modelId: 'google/gemini-3.1-pro-preview',
      provider: 'openrouter',
      openrouterProvider: null,
      lmStudioFallbackId: null,
      contextWindow: 1048576,
      maxOutputTokens: 65500,
    },
    highParam: {
      modelId: 'google/gemini-3.1-pro-preview',
      provider: 'openrouter',
      openrouterProvider: null,
      lmStudioFallbackId: null,
      contextWindow: 1048576,
      maxOutputTokens: 65500,
    },
    critique: {
      modelId: 'z-ai/glm-5.1',
      provider: 'openrouter',
      openrouterProvider: null,
      lmStudioFallbackId: null,
      contextWindow: 202752,
      maxOutputTokens: 65500,
    },
  },
  [RECOMMENDED_ALTERNATE_PROFILE_KEY]: {
    name: 'Fast, affordable, mid-tier knowledge',
    numSubmitters: 4,
    submitters: [
      {
        modelId: 'moonshotai/kimi-k2.6',
        provider: 'openrouter',
        openrouterProvider: null,
        lmStudioFallbackId: null,
        contextWindow: 262000,
        maxOutputTokens: 40000,
      },
      {
        modelId: 'openai/gpt-oss-120b',
        provider: 'openrouter',
        openrouterProvider: 'Groq',
        lmStudioFallbackId: null,
        contextWindow: 131072,
        maxOutputTokens: 25000,
      },
      {
        modelId: 'deepseek/deepseek-v4-pro',
        provider: 'openrouter',
        openrouterProvider: null,
        lmStudioFallbackId: null,
        contextWindow: 1048576,
        maxOutputTokens: 65500,
      },
      {
        modelId: 'inception/mercury-2',
        provider: 'openrouter',
        openrouterProvider: null,
        lmStudioFallbackId: null,
        contextWindow: 128000,
        maxOutputTokens: 25000,
      },
    ],
    validator: {
      modelId: 'qwen/qwen3.5-flash-02-23',
      provider: 'openrouter',
      openrouterProvider: null,
      lmStudioFallbackId: null,
      contextWindow: 1048576,
      maxOutputTokens: 65500,
    },
    highContext: {
      modelId: 'moonshotai/kimi-k2.6',
      provider: 'openrouter',
      openrouterProvider: null,
      lmStudioFallbackId: null,
      contextWindow: 262000,
      maxOutputTokens: 40000,
    },
    highParam: {
      modelId: 'google/gemini-3.1-pro-preview',
      provider: 'openrouter',
      openrouterProvider: null,
      lmStudioFallbackId: null,
      contextWindow: 1048576,
      maxOutputTokens: 65500,
    },
    critique: {
      modelId: 'google/gemini-3.1-pro-preview',
      provider: 'openrouter',
      openrouterProvider: null,
      lmStudioFallbackId: null,
      contextWindow: 1048576,
      maxOutputTokens: 65500,
    },
  },
  [RECOMMENDED_ENTRY_LAB_PROFILE_KEY]: {
    name: 'Fast, less affordable, higher knowledge',
    numSubmitters: 3,
    submitters: [
      {
        modelId: 'x-ai/grok-4.3',
        provider: 'openrouter',
        openrouterProvider: null,
        lmStudioFallbackId: null,
        contextWindow: 1000000,
        maxOutputTokens: 128000,
      },
      {
        modelId: 'moonshotai/kimi-k2.6',
        provider: 'openrouter',
        openrouterProvider: null,
        lmStudioFallbackId: null,
        contextWindow: 262000,
        maxOutputTokens: 40000,
      },
      {
        modelId: 'x-ai/grok-4.3',
        provider: 'openrouter',
        openrouterProvider: null,
        lmStudioFallbackId: null,
        contextWindow: 1000000,
        maxOutputTokens: 128000,
      },
    ],
    validator: {
      modelId: 'x-ai/grok-4.1-fast',
      provider: 'openrouter',
      openrouterProvider: null,
      lmStudioFallbackId: null,
      contextWindow: 2000000,
      maxOutputTokens: 30000,
    },
    highContext: {
      modelId: 'x-ai/grok-4.3',
      provider: 'openrouter',
      openrouterProvider: null,
      lmStudioFallbackId: null,
      contextWindow: 1000000,
      maxOutputTokens: 128000,
    },
    highParam: {
      modelId: 'x-ai/grok-4.3',
      provider: 'openrouter',
      openrouterProvider: null,
      lmStudioFallbackId: null,
      contextWindow: 1000000,
      maxOutputTokens: 128000,
    },
    critique: {
      modelId: 'google/gemini-3.1-pro-preview',
      provider: 'openrouter',
      openrouterProvider: null,
      lmStudioFallbackId: null,
      contextWindow: 1048576,
      maxOutputTokens: 65500,
    },
  },
  [RECOMMENDED_LAB_FAST_PROFILE_KEY]: {
    name: 'Lab grade, fast, costly (starts at ~$10 per hour), extra-high knowledge',
    numSubmitters: 3,
    submitters: [
      {
        modelId: 'openai/gpt-5.5',
        provider: 'openrouter',
        openrouterProvider: null,
        lmStudioFallbackId: null,
        contextWindow: 1050000,
        maxOutputTokens: 128000,
      },
      {
        modelId: 'moonshotai/kimi-k2.6',
        provider: 'openrouter',
        openrouterProvider: null,
        lmStudioFallbackId: null,
        contextWindow: 262000,
        maxOutputTokens: 40000,
      },
      {
        modelId: 'deepseek/deepseek-v4-pro',
        provider: 'openrouter',
        openrouterProvider: null,
        lmStudioFallbackId: null,
        contextWindow: 1048576,
        maxOutputTokens: 65500,
      },
    ],
    validator: {
      modelId: 'x-ai/grok-4.1-fast',
      provider: 'openrouter',
      openrouterProvider: null,
      lmStudioFallbackId: null,
      contextWindow: 2000000,
      maxOutputTokens: 30000,
    },
    highContext: {
      modelId: 'openai/gpt-5.5',
      provider: 'openrouter',
      openrouterProvider: null,
      lmStudioFallbackId: null,
      contextWindow: 1050000,
      maxOutputTokens: 128000,
    },
    highParam: {
      modelId: 'anthropic/claude-opus-4.7',
      provider: 'openrouter',
      openrouterProvider: null,
      lmStudioFallbackId: null,
      contextWindow: 1000000,
      maxOutputTokens: 128000,
    },
    critique: {
      modelId: 'google/gemini-3.1-pro-preview',
      provider: 'openrouter',
      openrouterProvider: null,
      lmStudioFallbackId: null,
      contextWindow: 1048576,
      maxOutputTokens: 65500,
    },
  },
  [RECOMMENDED_LAB_MAX_PROFILE_KEY]: {
    name: 'Lab grade, SOTA models, slower, costly (starts at ~$20 per hour), max knowledge',
    numSubmitters: 4,
    submitters: [
      {
        modelId: 'anthropic/claude-opus-4.7',
        provider: 'openrouter',
        openrouterProvider: null,
        lmStudioFallbackId: null,
        contextWindow: 1000000,
        maxOutputTokens: 128000,
      },
      {
        modelId: 'openai/gpt-5.5-pro',
        provider: 'openrouter',
        openrouterProvider: null,
        lmStudioFallbackId: null,
        contextWindow: 1050000,
        maxOutputTokens: 128000,
      },
      {
        modelId: 'x-ai/grok-4.20-multi-agent',
        provider: 'openrouter',
        openrouterProvider: null,
        lmStudioFallbackId: null,
        contextWindow: 2000000,
        maxOutputTokens: 65500,
      },
      {
        modelId: 'moonshotai/kimi-k2.6',
        provider: 'openrouter',
        openrouterProvider: null,
        lmStudioFallbackId: null,
        contextWindow: 262000,
        maxOutputTokens: 40000,
      },
    ],
    validator: {
      modelId: 'openai/gpt-5.5-pro',
      provider: 'openrouter',
      openrouterProvider: null,
      lmStudioFallbackId: null,
      contextWindow: 1050000,
      maxOutputTokens: 128000,
    },
    highContext: {
      modelId: 'anthropic/claude-opus-4.7',
      provider: 'openrouter',
      openrouterProvider: null,
      lmStudioFallbackId: null,
      contextWindow: 1000000,
      maxOutputTokens: 128000,
    },
    highParam: {
      modelId: 'anthropic/claude-opus-4.7',
      provider: 'openrouter',
      openrouterProvider: null,
      lmStudioFallbackId: null,
      contextWindow: 1000000,
      maxOutputTokens: 128000,
    },
    critique: {
      modelId: 'x-ai/grok-4.20-multi-agent',
      provider: 'openrouter',
      openrouterProvider: null,
      lmStudioFallbackId: null,
      contextWindow: 2000000,
      maxOutputTokens: 65500,
    },
  },
};

// Derive the startup/fallback OpenRouter defaults directly from the default
// recommended profile so there is a single source of truth. Changing the
// RECOMMENDED_PROFILE_KEY profile above automatically updates what a fresh
// install (or any settings reset) runs with.
const DEFAULT_RECOMMENDED_PROFILE = RECOMMENDED_PROFILES[RECOMMENDED_PROFILE_KEY];

const submitterFromRecommended = (submitter, submitterId) => ({
  submitterId,
  provider: submitter.provider || 'openrouter',
  modelId: submitter.modelId || '',
  openrouterProvider: submitter.openrouterProvider || null,
  lmStudioFallbackId: submitter.lmStudioFallbackId || null,
  contextWindow: submitter.contextWindow,
  maxOutputTokens: submitter.maxOutputTokens,
});

const DEFAULT_OPENROUTER_SUBMITTER_CONFIGS = DEFAULT_RECOMMENDED_PROFILE.submitters.map(
  (submitter, index) => submitterFromRecommended(submitter, index + 1)
);

const DEFAULT_LOCAL_CONFIG = {
  validator_provider: DEFAULT_RECOMMENDED_PROFILE.validator.provider || 'openrouter',
  validator_model: DEFAULT_RECOMMENDED_PROFILE.validator.modelId || '',
  validator_openrouter_provider: DEFAULT_RECOMMENDED_PROFILE.validator.openrouterProvider || null,
  validator_lm_studio_fallback: DEFAULT_RECOMMENDED_PROFILE.validator.lmStudioFallbackId || null,
  validator_context_window: DEFAULT_RECOMMENDED_PROFILE.validator.contextWindow,
  validator_max_tokens: DEFAULT_RECOMMENDED_PROFILE.validator.maxOutputTokens,
  high_context_provider: DEFAULT_RECOMMENDED_PROFILE.highContext.provider || 'openrouter',
  high_context_model: DEFAULT_RECOMMENDED_PROFILE.highContext.modelId || '',
  high_context_openrouter_provider: DEFAULT_RECOMMENDED_PROFILE.highContext.openrouterProvider || null,
  high_context_lm_studio_fallback: DEFAULT_RECOMMENDED_PROFILE.highContext.lmStudioFallbackId || null,
  high_context_context_window: DEFAULT_RECOMMENDED_PROFILE.highContext.contextWindow,
  high_context_max_tokens: DEFAULT_RECOMMENDED_PROFILE.highContext.maxOutputTokens,
  high_param_provider: DEFAULT_RECOMMENDED_PROFILE.highParam.provider || 'openrouter',
  high_param_model: DEFAULT_RECOMMENDED_PROFILE.highParam.modelId || '',
  high_param_openrouter_provider: DEFAULT_RECOMMENDED_PROFILE.highParam.openrouterProvider || null,
  high_param_lm_studio_fallback: DEFAULT_RECOMMENDED_PROFILE.highParam.lmStudioFallbackId || null,
  high_param_context_window: DEFAULT_RECOMMENDED_PROFILE.highParam.contextWindow,
  high_param_max_tokens: DEFAULT_RECOMMENDED_PROFILE.highParam.maxOutputTokens,
  critique_submitter_provider: DEFAULT_RECOMMENDED_PROFILE.critique.provider || 'openrouter',
  critique_submitter_model: DEFAULT_RECOMMENDED_PROFILE.critique.modelId || '',
  critique_submitter_openrouter_provider: DEFAULT_RECOMMENDED_PROFILE.critique.openrouterProvider || null,
  critique_submitter_lm_studio_fallback: DEFAULT_RECOMMENDED_PROFILE.critique.lmStudioFallbackId || null,
  critique_submitter_context_window: DEFAULT_RECOMMENDED_PROFILE.critique.contextWindow,
  critique_submitter_max_tokens: DEFAULT_RECOMMENDED_PROFILE.critique.maxOutputTokens,
};

const DEFAULT_AUTONOMOUS_SETTINGS = {
  numSubmitters: DEFAULT_RECOMMENDED_PROFILE.numSubmitters || DEFAULT_OPENROUTER_SUBMITTER_CONFIGS.length,
  submitterConfigs: DEFAULT_OPENROUTER_SUBMITTER_CONFIGS,
  localConfig: DEFAULT_LOCAL_CONFIG,
  freeOnly: false,
  freeModelLooping: true,
  freeModelAutoSelector: true,
  tier3Enabled: false,
  modelProviders: {},
  selectedProfile: RECOMMENDED_PROFILE_KEY,
};

function normalizeStoredSettings(settings = {}) {
  const submitterConfigs = Array.isArray(settings.submitterConfigs) && settings.submitterConfigs.length > 0
    ? settings.submitterConfigs.map((cfg, index) => ({
        ...DEFAULT_SUBMITTER_CONFIG,
        ...cfg,
        submitterId: cfg.submitterId || index + 1,
      }))
    : DEFAULT_AUTONOMOUS_SETTINGS.submitterConfigs;

  return {
    ...DEFAULT_AUTONOMOUS_SETTINGS,
    ...settings,
    numSubmitters: settings.numSubmitters || submitterConfigs.length || DEFAULT_AUTONOMOUS_SETTINGS.numSubmitters,
    submitterConfigs,
    localConfig: {
      ...DEFAULT_LOCAL_CONFIG,
      ...(settings.localConfig || {}),
    },
    freeOnly: settings.freeOnly ?? DEFAULT_AUTONOMOUS_SETTINGS.freeOnly,
    freeModelLooping: settings.freeModelLooping ?? DEFAULT_AUTONOMOUS_SETTINGS.freeModelLooping,
    freeModelAutoSelector: settings.freeModelAutoSelector ?? DEFAULT_AUTONOMOUS_SETTINGS.freeModelAutoSelector,
    tier3Enabled: settings.tier3Enabled ?? DEFAULT_AUTONOMOUS_SETTINGS.tier3Enabled,
    modelProviders: settings.modelProviders || DEFAULT_AUTONOMOUS_SETTINGS.modelProviders,
    selectedProfile: settings.selectedProfile ?? DEFAULT_AUTONOMOUS_SETTINGS.selectedProfile,
  };
}

export function getStoredAutonomousSettings() {
  try {
    const raw = localStorage.getItem(AUTONOMOUS_SETTINGS_STORAGE_KEY);
    if (!raw) {
      return normalizeStoredSettings();
    }

    return normalizeStoredSettings(JSON.parse(raw));
  } catch (error) {
    console.error('Failed to load autonomous research settings:', error);
    return normalizeStoredSettings();
  }
}

export function persistAutonomousSettings(settings) {
  const normalized = normalizeStoredSettings(settings);
  localStorage.setItem(AUTONOMOUS_SETTINGS_STORAGE_KEY, JSON.stringify(normalized));
  return normalized;
}

export function settingsToAutonomousConfig(settings) {
  const normalized = normalizeStoredSettings(settings);
  const localConfig = normalized.localConfig || {};

  return {
    submitter_configs: normalized.submitterConfigs.slice(0, normalized.numSubmitters),
    validator_provider: localConfig.validator_provider,
    validator_model: localConfig.validator_model,
    validator_openrouter_provider: localConfig.validator_openrouter_provider,
    validator_lm_studio_fallback: localConfig.validator_lm_studio_fallback,
    validator_context_window: localConfig.validator_context_window,
    validator_max_tokens: localConfig.validator_max_tokens,
    high_context_provider: localConfig.high_context_provider,
    high_context_model: localConfig.high_context_model,
    high_context_openrouter_provider: localConfig.high_context_openrouter_provider,
    high_context_lm_studio_fallback: localConfig.high_context_lm_studio_fallback,
    high_context_context_window: localConfig.high_context_context_window,
    high_context_max_tokens: localConfig.high_context_max_tokens,
    high_param_provider: localConfig.high_param_provider,
    high_param_model: localConfig.high_param_model,
    high_param_openrouter_provider: localConfig.high_param_openrouter_provider,
    high_param_lm_studio_fallback: localConfig.high_param_lm_studio_fallback,
    high_param_context_window: localConfig.high_param_context_window,
    high_param_max_tokens: localConfig.high_param_max_tokens,
    critique_submitter_provider: localConfig.critique_submitter_provider,
    critique_submitter_model: localConfig.critique_submitter_model,
    critique_submitter_openrouter_provider: localConfig.critique_submitter_openrouter_provider,
    critique_submitter_lm_studio_fallback: localConfig.critique_submitter_lm_studio_fallback,
    critique_submitter_context_window: localConfig.critique_submitter_context_window,
    critique_submitter_max_tokens: localConfig.critique_submitter_max_tokens,
    tier3_enabled: normalized.tier3Enabled ?? false,
  };
}

function buildLocalConfigFromLmStudio(modelId = '') {
  return {
    ...DEFAULT_LM_LOCAL_CONFIG,
    validator_model: modelId,
    high_context_model: modelId,
    high_param_model: modelId,
    critique_submitter_model: modelId,
  };
}

export function applyLmStudioStartupDefaults(modelId = '') {
  const currentSettings = getStoredAutonomousSettings();
  const nextSettings = persistAutonomousSettings({
    ...currentSettings,
    numSubmitters: 3,
    submitterConfigs: createDefaultSubmitterConfigs(modelId),
    localConfig: {
      ...currentSettings.localConfig,
      ...buildLocalConfigFromLmStudio(modelId),
    },
    selectedProfile: '',
  });

  return {
    settings: nextSettings,
    config: settingsToAutonomousConfig(nextSettings),
  };
}

export async function applyAutonomousProfileSelection(profileKey, userProfiles = {}) {
  const isRecommended = profileKey.startsWith('recommended_');
  const profile = isRecommended
    ? RECOMMENDED_PROFILES[profileKey]
    : userProfiles[profileKey];

  if (!profile) {
    throw new Error(`Profile not found: ${profileKey}`);
  }

  await loadModelCache();

  const convertToApiId = (displayNameOrId) => {
    if (!displayNameOrId) return '';
    return getModelApiId(displayNameOrId);
  };

  const submitterConfigs = profile.submitters.map((submitterProfile, index) => ({
    submitterId: index + 1,
    provider: submitterProfile.provider || 'openrouter',
    modelId: isRecommended
      ? convertToApiId(submitterProfile.modelId || '')
      : (submitterProfile.modelId || ''),
    openrouterProvider: submitterProfile.openrouterProvider || null,
    lmStudioFallbackId: isRecommended ? null : (submitterProfile.lmStudioFallbackId || null),
    contextWindow: submitterProfile.contextWindow,
    maxOutputTokens: submitterProfile.maxOutputTokens,
  }));

  const getModelId = (roleProfile = {}) => (
    isRecommended
      ? convertToApiId(roleProfile.modelId || '')
      : (roleProfile.modelId || '')
  );

  const getOpenRouterProvider = (roleProfile = {}) => roleProfile.openrouterProvider || null;

  const currentSettings = getStoredAutonomousSettings();
  const nextSettings = persistAutonomousSettings({
    ...currentSettings,
    numSubmitters: profile.numSubmitters,
    submitterConfigs,
    localConfig: {
      ...currentSettings.localConfig,
      validator_provider: isRecommended ? 'openrouter' : (profile.validator.provider || 'openrouter'),
      validator_model: getModelId(profile.validator),
      validator_openrouter_provider: getOpenRouterProvider(profile.validator),
      validator_lm_studio_fallback: isRecommended ? null : (profile.validator.lmStudioFallbackId || null),
      validator_context_window: profile.validator.contextWindow,
      validator_max_tokens: profile.validator.maxOutputTokens,
      high_context_provider: isRecommended ? 'openrouter' : (profile.highContext.provider || 'openrouter'),
      high_context_model: getModelId(profile.highContext),
      high_context_openrouter_provider: getOpenRouterProvider(profile.highContext),
      high_context_lm_studio_fallback: isRecommended ? null : (profile.highContext.lmStudioFallbackId || null),
      high_context_context_window: profile.highContext.contextWindow,
      high_context_max_tokens: profile.highContext.maxOutputTokens,
      high_param_provider: isRecommended ? 'openrouter' : (profile.highParam.provider || 'openrouter'),
      high_param_model: getModelId(profile.highParam),
      high_param_openrouter_provider: getOpenRouterProvider(profile.highParam),
      high_param_lm_studio_fallback: isRecommended ? null : (profile.highParam.lmStudioFallbackId || null),
      high_param_context_window: profile.highParam.contextWindow,
      high_param_max_tokens: profile.highParam.maxOutputTokens,
      critique_submitter_provider: isRecommended ? 'openrouter' : (profile.critique.provider || 'openrouter'),
      critique_submitter_model: getModelId(profile.critique),
      critique_submitter_openrouter_provider: getOpenRouterProvider(profile.critique),
      critique_submitter_lm_studio_fallback: isRecommended ? null : (profile.critique.lmStudioFallbackId || null),
      critique_submitter_context_window: profile.critique.contextWindow,
      critique_submitter_max_tokens: profile.critique.maxOutputTokens,
    },
    selectedProfile: profileKey,
  });

  return {
    profile,
    settings: nextSettings,
    config: settingsToAutonomousConfig(nextSettings),
  };
}
