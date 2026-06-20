import { loadModelCache, getModelApiId } from './modelCache';
import {
  computeCodexAutoSettings,
  computeCloudAccessAutoSettings,
  computeXAIGrokAutoSettings,
  DEFAULT_CONTEXT_WINDOW,
  DEFAULT_MAX_OUTPUT_TOKENS,
  DEFAULT_OPENROUTER_REASONING_EFFORT,
  normalizeOpenRouterReasoningEffort,
} from './openRouterSelection';

export const AUTONOMOUS_SETTINGS_STORAGE_KEY = 'autonomous_research_settings';
export const AUTONOMOUS_PROFILES_STORAGE_KEY = 'autonomous_research_profiles';
export const STARTUP_PROVIDER_CHOICE_STORAGE_KEY = 'startup_provider_choice';
export const LM_STUDIO_STARTUP_CHOICE = 'lm_studio';
export const OPENAI_CODEX_STARTUP_CHOICE = 'openai_codex_oauth';
export const XAI_GROK_STARTUP_CHOICE = 'xai_grok_oauth';
export const RECOMMENDED_PROFILE_KEY = 'recommended_slower_affordable_higher_knowledge';
export const RECOMMENDED_LAB_FAST_PROFILE_KEY = 'recommended_lab_fast_costly_extra_high';
export const RECOMMENDED_LAB_MAX_PROFILE_KEY = 'recommended_lab_slow_costly_max';
export const RECOMMENDED_PROFILE_KEYS = [
  RECOMMENDED_PROFILE_KEY,
  RECOMMENDED_LAB_FAST_PROFILE_KEY,
  RECOMMENDED_LAB_MAX_PROFILE_KEY,
];
const LEGACY_WRITER_SNAKE_PREFIX = ['high', 'context'].join('_');
const LEGACY_WRITER_PROFILE_KEY = ['high', 'Context'].join('');

const isMeaningfulWriterLocal = (value, suffix) => {
  if (value === undefined || value === null) return false;
  if (typeof value === 'string' && value.trim() === '') return false;
  if ((suffix === 'context_window' || suffix === 'max_tokens') && Number(value) <= 0) return false;
  return true;
};

const readWriterLocal = (localConfig = {}, suffix) => {
  const current = localConfig[`writer_${suffix}`];
  if (isMeaningfulWriterLocal(current, suffix)) {
    return current;
  }
  return localConfig[`${LEGACY_WRITER_SNAKE_PREFIX}_${suffix}`];
};

const isMeaningfulProfileValue = (value) => (
  value !== undefined
  && value !== null
  && !(typeof value === 'string' && value.trim() === '')
);

const normalizeProfileWriter = (profile = {}) => {
  const current = profile.writer || {};
  const legacy = profile[LEGACY_WRITER_PROFILE_KEY] || {};
  if (!isMeaningfulProfileValue(current.modelId) && isMeaningfulProfileValue(legacy.modelId)) {
    return {
      ...current,
      ...Object.fromEntries(
        Object.entries(legacy).filter(([, value]) => isMeaningfulProfileValue(value))
      ),
    };
  }
  return current;
};

const normalizeProfileRigor = (profile = {}) => {
  const current = profile.highParam || {};
  const legacyCritique = profile.critique || {};
  if (!isMeaningfulProfileValue(current.modelId) && isMeaningfulProfileValue(legacyCritique.modelId)) {
    return {
      ...current,
      ...Object.fromEntries(
        Object.entries(legacyCritique).filter(([, value]) => isMeaningfulProfileValue(value))
      ),
    };
  }
  return current;
};

const DEFAULT_SUBMITTER_CONFIG = {
  submitterId: 1,
  provider: 'lm_studio',
  modelId: '',
  openrouterProvider: null,
  openrouterReasoningEffort: DEFAULT_OPENROUTER_REASONING_EFFORT,
  lmStudioFallbackId: null,
  contextWindow: DEFAULT_CONTEXT_WINDOW,
  maxOutputTokens: DEFAULT_MAX_OUTPUT_TOKENS,
  superchargeEnabled: false,
};

// NOTE: DEFAULT_OPENROUTER_SUBMITTER_CONFIGS and DEFAULT_LOCAL_CONFIG are derived
// from RECOMMENDED_PROFILES[RECOMMENDED_PROFILE_KEY] further below so the "default"
// startup configuration and the selectable recommended profile stay in sync.
// Update the recommended profile below to change what a fresh install runs with.

const GEMINI_FLASH_LATEST_PROFILE_CONFIG = {
  modelId: '~google/gemini-flash-latest',
  provider: 'openrouter',
  openrouterProvider: null,
  lmStudioFallbackId: null,
  contextWindow: 1048576,
  maxOutputTokens: 65536,
};

const MINIMAX_M3_PROFILE_CONFIG = {
  modelId: 'minimax/minimax-m3',
  provider: 'openrouter',
  openrouterProvider: null,
  lmStudioFallbackId: null,
  contextWindow: 1048576,
  maxOutputTokens: 131072,
};

const DEFAULT_LM_LOCAL_CONFIG = {
  validator_provider: 'lm_studio',
  validator_model: '',
  validator_openrouter_provider: null,
  validator_openrouter_reasoning_effort: DEFAULT_OPENROUTER_REASONING_EFFORT,
  validator_lm_studio_fallback: null,
  validator_context_window: DEFAULT_CONTEXT_WINDOW,
  validator_max_tokens: DEFAULT_MAX_OUTPUT_TOKENS,
  validator_supercharge_enabled: false,
  assistant_provider: 'lm_studio',
  assistant_model: '',
  assistant_openrouter_provider: null,
  assistant_openrouter_reasoning_effort: DEFAULT_OPENROUTER_REASONING_EFFORT,
  assistant_lm_studio_fallback: null,
  assistant_context_window: DEFAULT_CONTEXT_WINDOW,
  assistant_max_tokens: DEFAULT_MAX_OUTPUT_TOKENS,
  assistant_supercharge_enabled: false,
  writer_provider: 'lm_studio',
  writer_model: '',
  writer_openrouter_provider: null,
  writer_openrouter_reasoning_effort: DEFAULT_OPENROUTER_REASONING_EFFORT,
  writer_lm_studio_fallback: null,
  writer_context_window: DEFAULT_CONTEXT_WINDOW,
  writer_max_tokens: DEFAULT_MAX_OUTPUT_TOKENS,
  writer_supercharge_enabled: false,
  high_param_provider: 'lm_studio',
  high_param_model: '',
  high_param_openrouter_provider: null,
  high_param_openrouter_reasoning_effort: DEFAULT_OPENROUTER_REASONING_EFFORT,
  high_param_lm_studio_fallback: null,
  high_param_context_window: DEFAULT_CONTEXT_WINDOW,
  high_param_max_tokens: DEFAULT_MAX_OUTPUT_TOKENS,
  high_param_supercharge_enabled: false,
};

const DEFAULT_CODEX_STARTUP_MODEL = Object.freeze({
  id: 'gpt-5.5',
  name: 'gpt-5.5',
  context_length: 400000,
  max_output_tokens: 128000,
});
const PUBLIC_CODEX_STARTUP_MODELS = Object.freeze([
  DEFAULT_CODEX_STARTUP_MODEL,
  Object.freeze({
    id: 'gpt-5.5-mini',
    name: 'gpt-5.5-mini',
    context_length: 400000,
    max_output_tokens: 128000,
  }),
  Object.freeze({
    id: 'gpt-5.4',
    name: 'gpt-5.4',
    context_length: 400000,
    max_output_tokens: 128000,
  }),
  Object.freeze({
    id: 'gpt-5.4-mini',
    name: 'gpt-5.4-mini',
    context_length: 400000,
    max_output_tokens: 128000,
  }),
]);
const DEFAULT_XAI_GROK_STARTUP_MODEL = Object.freeze({
  id: 'grok-4.3',
  name: 'grok-4.3',
  context_length: 1000000,
  max_output_tokens: 131072,
});
const PUBLIC_XAI_GROK_STARTUP_MODELS = Object.freeze([
  DEFAULT_XAI_GROK_STARTUP_MODEL,
  Object.freeze({
    id: 'grok-4.2',
    name: 'grok-4.2',
    context_length: 1000000,
    max_output_tokens: 131072,
  }),
  Object.freeze({
    id: 'grok-4',
    name: 'grok-4',
    context_length: 1000000,
    max_output_tokens: 131072,
  }),
]);

const createDefaultSubmitterConfigs = (modelId = '') => (
  [1, 2, 3].map((submitterId) => ({
    ...DEFAULT_SUBMITTER_CONFIG,
    submitterId,
    modelId,
  }))
);

export const RECOMMENDED_PROFILES = {
  [RECOMMENDED_PROFILE_KEY]: {
    name: 'Slow, affordable, high knowledge',
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
        ...MINIMAX_M3_PROFILE_CONFIG,
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
      ...MINIMAX_M3_PROFILE_CONFIG,
    },
    writer: {
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
        ...MINIMAX_M3_PROFILE_CONFIG,
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
      ...GEMINI_FLASH_LATEST_PROFILE_CONFIG,
    },
    writer: {
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
  },
  [RECOMMENDED_LAB_MAX_PROFILE_KEY]: {
    name: 'Lab grade, SOTA models, slower, very costly (starts at ~$40 per hour), max knowledge, too expensive and overkill for most home users',
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
        ...MINIMAX_M3_PROFILE_CONFIG,
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
    writer: {
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
  validator_openrouter_reasoning_effort: normalizeOpenRouterReasoningEffort(DEFAULT_RECOMMENDED_PROFILE.validator.openrouterReasoningEffort),
  validator_lm_studio_fallback: DEFAULT_RECOMMENDED_PROFILE.validator.lmStudioFallbackId || null,
  validator_context_window: DEFAULT_RECOMMENDED_PROFILE.validator.contextWindow,
  validator_max_tokens: DEFAULT_RECOMMENDED_PROFILE.validator.maxOutputTokens,
  validator_supercharge_enabled: Boolean(DEFAULT_RECOMMENDED_PROFILE.validator.superchargeEnabled),
  assistant_provider: DEFAULT_RECOMMENDED_PROFILE.validator.provider || 'openrouter',
  assistant_model: DEFAULT_RECOMMENDED_PROFILE.validator.modelId || '',
  assistant_openrouter_provider: DEFAULT_RECOMMENDED_PROFILE.validator.openrouterProvider || null,
  assistant_openrouter_reasoning_effort: normalizeOpenRouterReasoningEffort(DEFAULT_RECOMMENDED_PROFILE.validator.openrouterReasoningEffort),
  assistant_lm_studio_fallback: DEFAULT_RECOMMENDED_PROFILE.validator.lmStudioFallbackId || null,
  assistant_context_window: DEFAULT_RECOMMENDED_PROFILE.validator.contextWindow,
  assistant_max_tokens: DEFAULT_RECOMMENDED_PROFILE.validator.maxOutputTokens,
  assistant_supercharge_enabled: Boolean(DEFAULT_RECOMMENDED_PROFILE.validator.superchargeEnabled),
  writer_provider: DEFAULT_RECOMMENDED_PROFILE.writer.provider || 'openrouter',
  writer_model: DEFAULT_RECOMMENDED_PROFILE.writer.modelId || '',
  writer_openrouter_provider: DEFAULT_RECOMMENDED_PROFILE.writer.openrouterProvider || null,
  writer_openrouter_reasoning_effort: normalizeOpenRouterReasoningEffort(DEFAULT_RECOMMENDED_PROFILE.writer.openrouterReasoningEffort),
  writer_lm_studio_fallback: DEFAULT_RECOMMENDED_PROFILE.writer.lmStudioFallbackId || null,
  writer_context_window: DEFAULT_RECOMMENDED_PROFILE.writer.contextWindow,
  writer_max_tokens: DEFAULT_RECOMMENDED_PROFILE.writer.maxOutputTokens,
  writer_supercharge_enabled: Boolean(DEFAULT_RECOMMENDED_PROFILE.writer.superchargeEnabled),
  high_param_provider: DEFAULT_RECOMMENDED_PROFILE.highParam.provider || 'openrouter',
  high_param_model: DEFAULT_RECOMMENDED_PROFILE.highParam.modelId || '',
  high_param_openrouter_provider: DEFAULT_RECOMMENDED_PROFILE.highParam.openrouterProvider || null,
  high_param_openrouter_reasoning_effort: normalizeOpenRouterReasoningEffort(DEFAULT_RECOMMENDED_PROFILE.highParam.openrouterReasoningEffort),
  high_param_lm_studio_fallback: DEFAULT_RECOMMENDED_PROFILE.highParam.lmStudioFallbackId || null,
  high_param_context_window: DEFAULT_RECOMMENDED_PROFILE.highParam.contextWindow,
  high_param_max_tokens: DEFAULT_RECOMMENDED_PROFILE.highParam.maxOutputTokens,
  high_param_supercharge_enabled: Boolean(DEFAULT_RECOMMENDED_PROFILE.highParam.superchargeEnabled),
  critique_submitter_provider: DEFAULT_RECOMMENDED_PROFILE.highParam.provider || 'openrouter',
  critique_submitter_model: DEFAULT_RECOMMENDED_PROFILE.highParam.modelId || '',
  critique_submitter_openrouter_provider: DEFAULT_RECOMMENDED_PROFILE.highParam.openrouterProvider || null,
  critique_submitter_openrouter_reasoning_effort: normalizeOpenRouterReasoningEffort(DEFAULT_RECOMMENDED_PROFILE.highParam.openrouterReasoningEffort),
  critique_submitter_lm_studio_fallback: DEFAULT_RECOMMENDED_PROFILE.highParam.lmStudioFallbackId || null,
  critique_submitter_context_window: DEFAULT_RECOMMENDED_PROFILE.highParam.contextWindow,
  critique_submitter_max_tokens: DEFAULT_RECOMMENDED_PROFILE.highParam.maxOutputTokens,
  critique_submitter_supercharge_enabled: Boolean(DEFAULT_RECOMMENDED_PROFILE.highParam.superchargeEnabled),
};

const DEFAULT_AUTONOMOUS_SETTINGS = {
  numSubmitters: DEFAULT_RECOMMENDED_PROFILE.numSubmitters || DEFAULT_OPENROUTER_SUBMITTER_CONFIGS.length,
  submitterConfigs: DEFAULT_OPENROUTER_SUBMITTER_CONFIGS,
  localConfig: DEFAULT_LOCAL_CONFIG,
  freeOnly: false,
  freeModelLooping: true,
  freeModelAutoSelector: true,
  allowMathematicalProofs: true,
  allowResearchPapers: true,
  tier3Enabled: false,
  creativityEmphasisBoostEnabled: false,
  modelProviders: {},
  selectedProfile: RECOMMENDED_PROFILE_KEY,
};

const PUBLIC_ROLE_STORAGE_KEYS = [
  'provider',
  'modelId',
  'openrouterProvider',
  'openrouterReasoningEffort',
  'lmStudioFallbackId',
  'contextWindow',
  'maxOutputTokens',
  'superchargeEnabled',
];

const PUBLIC_LOCAL_CONFIG_STORAGE_KEYS = [
  'validator_provider',
  'validator_model',
  'validator_openrouter_provider',
  'validator_openrouter_reasoning_effort',
  'validator_lm_studio_fallback',
  'validator_context_window',
  'validator_max_tokens',
  'validator_supercharge_enabled',
  'assistant_provider',
  'assistant_model',
  'assistant_openrouter_provider',
  'assistant_openrouter_reasoning_effort',
  'assistant_lm_studio_fallback',
  'assistant_context_window',
  'assistant_max_tokens',
  'assistant_supercharge_enabled',
  'writer_provider',
  'writer_model',
  'writer_openrouter_provider',
  'writer_openrouter_reasoning_effort',
  'writer_lm_studio_fallback',
  'writer_context_window',
  'writer_max_tokens',
  'writer_supercharge_enabled',
  'high_param_provider',
  'high_param_model',
  'high_param_openrouter_provider',
  'high_param_openrouter_reasoning_effort',
  'high_param_lm_studio_fallback',
  'high_param_context_window',
  'high_param_max_tokens',
  'high_param_supercharge_enabled',
  'critique_submitter_provider',
  'critique_submitter_model',
  'critique_submitter_openrouter_provider',
  'critique_submitter_openrouter_reasoning_effort',
  'critique_submitter_lm_studio_fallback',
  'critique_submitter_context_window',
  'critique_submitter_max_tokens',
  'critique_submitter_supercharge_enabled',
];

const SECRET_STORAGE_KEY_PATTERN = /(?:api[_-]?key|access[_-]?token|refresh[_-]?token|id[_-]?token|authorization|bearer|password|secret|credential|session|cookie)/i;

function stripSecretLikeStorageFields(value) {
  if (Array.isArray(value)) {
    return value.map(stripSecretLikeStorageFields);
  }
  if (!value || typeof value !== 'object') {
    return value;
  }
  return Object.fromEntries(
    Object.entries(value)
      .filter(([key]) => !SECRET_STORAGE_KEY_PATTERN.test(key))
      .map(([key, nestedValue]) => [key, stripSecretLikeStorageFields(nestedValue)])
  );
}

function pickPublicFields(source = {}, keys = []) {
  return Object.fromEntries(
    keys
      .filter((key) => Object.prototype.hasOwnProperty.call(source, key))
      .map((key) => [key, source[key]])
  );
}

function publicSubmitterConfigForStorage(config = {}, index = 0) {
  return {
    ...pickPublicFields(config, PUBLIC_ROLE_STORAGE_KEYS),
    submitterId: config.submitterId || index + 1,
    openrouterReasoningEffort: normalizeOpenRouterReasoningEffort(config.openrouterReasoningEffort),
    superchargeEnabled: Boolean(config.superchargeEnabled),
  };
}

function publicRoleProfileForStorage(profile = {}) {
  return {
    ...pickPublicFields(profile, PUBLIC_ROLE_STORAGE_KEYS),
    openrouterReasoningEffort: normalizeOpenRouterReasoningEffort(profile.openrouterReasoningEffort),
    superchargeEnabled: Boolean(profile.superchargeEnabled),
  };
}

function publicLocalConfigForStorage(localConfig = {}) {
  return {
    ...pickPublicFields(localConfig, PUBLIC_LOCAL_CONFIG_STORAGE_KEYS),
    validator_openrouter_reasoning_effort: normalizeOpenRouterReasoningEffort(localConfig.validator_openrouter_reasoning_effort),
    assistant_openrouter_reasoning_effort: normalizeOpenRouterReasoningEffort(localConfig.assistant_openrouter_reasoning_effort),
    writer_openrouter_reasoning_effort: normalizeOpenRouterReasoningEffort(localConfig.writer_openrouter_reasoning_effort),
    high_param_openrouter_reasoning_effort: normalizeOpenRouterReasoningEffort(localConfig.high_param_openrouter_reasoning_effort),
    critique_submitter_openrouter_reasoning_effort: normalizeOpenRouterReasoningEffort(localConfig.critique_submitter_openrouter_reasoning_effort),
    validator_supercharge_enabled: Boolean(localConfig.validator_supercharge_enabled),
    assistant_supercharge_enabled: Boolean(localConfig.assistant_supercharge_enabled),
    writer_supercharge_enabled: Boolean(localConfig.writer_supercharge_enabled),
    high_param_supercharge_enabled: Boolean(localConfig.high_param_supercharge_enabled),
    critique_submitter_supercharge_enabled: Boolean(localConfig.critique_submitter_supercharge_enabled),
  };
}

export function publicAutonomousSettingsForStorage(settings = {}) {
  const normalized = normalizeStoredSettings(settings);
  return {
    numSubmitters: normalized.numSubmitters,
    submitterConfigs: normalized.submitterConfigs
      .slice(0, normalized.numSubmitters)
      .map(publicSubmitterConfigForStorage),
    localConfig: publicLocalConfigForStorage(normalized.localConfig),
    freeOnly: Boolean(normalized.freeOnly),
    freeModelLooping: Boolean(normalized.freeModelLooping),
    freeModelAutoSelector: Boolean(normalized.freeModelAutoSelector),
    allowMathematicalProofs: Boolean(normalized.allowMathematicalProofs),
    allowResearchPapers: Boolean(normalized.allowResearchPapers),
    tier3Enabled: Boolean(normalized.tier3Enabled),
    creativityEmphasisBoostEnabled: Boolean(normalized.creativityEmphasisBoostEnabled),
    selectedProfile: normalizeSelectedProfile(normalized.selectedProfile),
  };
}

export function publicAutonomousProfilesForStorage(profiles = {}) {
  return Object.fromEntries(
    Object.entries(profiles).map(([profileKey, profile = {}]) => {
      const writerProfile = normalizeProfileWriter(profile);
      const rigorProfile = normalizeProfileRigor(profile);
      const publicProfile = {
        name: typeof profile.name === 'string' ? profile.name : '',
        numSubmitters: profile.numSubmitters,
        submitters: Array.isArray(profile.submitters)
          ? profile.submitters.map(publicRoleProfileForStorage)
          : [],
        validator: publicRoleProfileForStorage(profile.validator),
        assistant: profile.assistant ? publicRoleProfileForStorage(profile.assistant) : undefined,
        writer: publicRoleProfileForStorage(writerProfile),
        highParam: publicRoleProfileForStorage(rigorProfile),
      };
      return [profileKey, stripSecretLikeStorageFields(publicProfile)];
    })
  );
}

function hasOwnSetting(settings = {}, key) {
  return Object.prototype.hasOwnProperty.call(settings, key);
}

function normalizeSelectedProfile(selectedProfile) {
  if (selectedProfile === undefined || selectedProfile === null) {
    return DEFAULT_AUTONOMOUS_SETTINGS.selectedProfile;
  }
  if (selectedProfile === '') {
    return '';
  }
  if (typeof selectedProfile !== 'string') {
    return DEFAULT_AUTONOMOUS_SETTINGS.selectedProfile;
  }
  if (selectedProfile.startsWith('recommended_') && !RECOMMENDED_PROFILE_KEYS.includes(selectedProfile)) {
    return '';
  }
  return selectedProfile;
}

function mirrorCritiqueFromRigor(localConfig = {}) {
  return {
    critique_submitter_provider: localConfig.high_param_provider,
    critique_submitter_model: localConfig.high_param_model,
    critique_submitter_openrouter_provider: localConfig.high_param_openrouter_provider,
    critique_submitter_openrouter_reasoning_effort: normalizeOpenRouterReasoningEffort(localConfig.high_param_openrouter_reasoning_effort),
    critique_submitter_lm_studio_fallback: localConfig.high_param_lm_studio_fallback,
    critique_submitter_context_window: localConfig.high_param_context_window,
    critique_submitter_max_tokens: localConfig.high_param_max_tokens,
    critique_submitter_supercharge_enabled: Boolean(localConfig.high_param_supercharge_enabled),
  };
}

function normalizeStoredSettings(settings = {}) {
  const submitterConfigs = Array.isArray(settings.submitterConfigs) && settings.submitterConfigs.length > 0
    ? settings.submitterConfigs.map((cfg, index) => ({
        ...DEFAULT_SUBMITTER_CONFIG,
        ...cfg,
        submitterId: cfg.submitterId || index + 1,
        openrouterReasoningEffort: normalizeOpenRouterReasoningEffort(cfg.openrouterReasoningEffort),
      }))
    : DEFAULT_AUTONOMOUS_SETTINGS.submitterConfigs;

  const inputLocalConfig = settings.localConfig || {};
  const migratedWriterLocalConfig = {
    writer_provider: readWriterLocal(inputLocalConfig, 'provider'),
    writer_model: readWriterLocal(inputLocalConfig, 'model'),
    writer_openrouter_provider: readWriterLocal(inputLocalConfig, 'openrouter_provider'),
    writer_openrouter_reasoning_effort: readWriterLocal(inputLocalConfig, 'openrouter_reasoning_effort'),
    writer_lm_studio_fallback: readWriterLocal(inputLocalConfig, 'lm_studio_fallback'),
    writer_context_window: readWriterLocal(inputLocalConfig, 'context_window'),
    writer_max_tokens: readWriterLocal(inputLocalConfig, 'max_tokens'),
    writer_supercharge_enabled: readWriterLocal(inputLocalConfig, 'supercharge_enabled'),
  };
  const baseLocalConfig = {
    ...DEFAULT_LOCAL_CONFIG,
    ...inputLocalConfig,
    ...Object.fromEntries(
      Object.entries(migratedWriterLocalConfig).filter(([, value]) => value !== undefined)
    ),
    validator_openrouter_reasoning_effort: normalizeOpenRouterReasoningEffort(inputLocalConfig.validator_openrouter_reasoning_effort),
    assistant_provider: inputLocalConfig.assistant_provider || inputLocalConfig.validator_provider || DEFAULT_LOCAL_CONFIG.assistant_provider,
    assistant_model: inputLocalConfig.assistant_model || inputLocalConfig.validator_model || DEFAULT_LOCAL_CONFIG.assistant_model,
    assistant_openrouter_provider: inputLocalConfig.assistant_openrouter_provider ?? inputLocalConfig.validator_openrouter_provider ?? DEFAULT_LOCAL_CONFIG.assistant_openrouter_provider,
    assistant_openrouter_reasoning_effort: normalizeOpenRouterReasoningEffort(inputLocalConfig.assistant_openrouter_reasoning_effort || inputLocalConfig.validator_openrouter_reasoning_effort),
    assistant_lm_studio_fallback: inputLocalConfig.assistant_lm_studio_fallback ?? inputLocalConfig.validator_lm_studio_fallback ?? DEFAULT_LOCAL_CONFIG.assistant_lm_studio_fallback,
    assistant_context_window: inputLocalConfig.assistant_context_window || inputLocalConfig.validator_context_window || DEFAULT_LOCAL_CONFIG.assistant_context_window,
    assistant_max_tokens: inputLocalConfig.assistant_max_tokens || inputLocalConfig.validator_max_tokens || DEFAULT_LOCAL_CONFIG.assistant_max_tokens,
    assistant_supercharge_enabled: inputLocalConfig.assistant_model
      ? Boolean(inputLocalConfig.assistant_supercharge_enabled)
      : Boolean(inputLocalConfig.validator_supercharge_enabled ?? DEFAULT_LOCAL_CONFIG.assistant_supercharge_enabled),
    writer_openrouter_reasoning_effort: normalizeOpenRouterReasoningEffort(
      migratedWriterLocalConfig.writer_openrouter_reasoning_effort
    ),
    high_param_openrouter_reasoning_effort: normalizeOpenRouterReasoningEffort(inputLocalConfig.high_param_openrouter_reasoning_effort),
  };

  return {
    ...DEFAULT_AUTONOMOUS_SETTINGS,
    ...settings,
    numSubmitters: settings.numSubmitters || submitterConfigs.length || DEFAULT_AUTONOMOUS_SETTINGS.numSubmitters,
    submitterConfigs,
    localConfig: {
      ...baseLocalConfig,
      ...mirrorCritiqueFromRigor(baseLocalConfig),
    },
    freeOnly: settings.freeOnly ?? DEFAULT_AUTONOMOUS_SETTINGS.freeOnly,
    freeModelLooping: settings.freeModelLooping ?? DEFAULT_AUTONOMOUS_SETTINGS.freeModelLooping,
    freeModelAutoSelector: settings.freeModelAutoSelector ?? DEFAULT_AUTONOMOUS_SETTINGS.freeModelAutoSelector,
    allowMathematicalProofs: settings.allowMathematicalProofs ?? DEFAULT_AUTONOMOUS_SETTINGS.allowMathematicalProofs,
    allowResearchPapers: settings.allowResearchPapers ?? DEFAULT_AUTONOMOUS_SETTINGS.allowResearchPapers,
    tier3Enabled: settings.tier3Enabled ?? DEFAULT_AUTONOMOUS_SETTINGS.tier3Enabled,
    creativityEmphasisBoostEnabled: settings.creativityEmphasisBoostEnabled ?? DEFAULT_AUTONOMOUS_SETTINGS.creativityEmphasisBoostEnabled,
    modelProviders: settings.modelProviders || DEFAULT_AUTONOMOUS_SETTINGS.modelProviders,
    selectedProfile: normalizeSelectedProfile(settings.selectedProfile),
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
  const inputSettings = settings || {};
  const existingSettings = getStoredAutonomousSettings();
  const normalized = normalizeStoredSettings({
    ...inputSettings,
    allowMathematicalProofs: hasOwnSetting(inputSettings, 'allowMathematicalProofs')
      ? inputSettings.allowMathematicalProofs
      : existingSettings.allowMathematicalProofs,
    allowResearchPapers: hasOwnSetting(inputSettings, 'allowResearchPapers')
      ? inputSettings.allowResearchPapers
      : existingSettings.allowResearchPapers,
    creativityEmphasisBoostEnabled: hasOwnSetting(inputSettings, 'creativityEmphasisBoostEnabled')
      ? inputSettings.creativityEmphasisBoostEnabled
      : existingSettings.creativityEmphasisBoostEnabled,
  });
  localStorage.setItem(AUTONOMOUS_SETTINGS_STORAGE_KEY, JSON.stringify(publicAutonomousSettingsForStorage(normalized)));
  return normalized;
}

export function persistAutonomousProfiles(profiles) {
  const publicProfiles = publicAutonomousProfilesForStorage(profiles);
  localStorage.setItem(AUTONOMOUS_PROFILES_STORAGE_KEY, JSON.stringify(publicProfiles));
  return publicProfiles;
}

export function settingsToAutonomousConfig(settings) {
  const normalized = normalizeStoredSettings(settings);
  const localConfig = normalized.localConfig || {};

  return {
    submitter_configs: normalized.submitterConfigs.slice(0, normalized.numSubmitters).map(cfg => ({
      ...cfg,
      openrouter_reasoning_effort: normalizeOpenRouterReasoningEffort(cfg.openrouterReasoningEffort),
      supercharge_enabled: Boolean(cfg.superchargeEnabled),
    })),
    creativity_emphasis_boost_enabled: Boolean(normalized.creativityEmphasisBoostEnabled),
    validator_provider: localConfig.validator_provider,
    validator_model: localConfig.validator_model,
    validator_openrouter_provider: localConfig.validator_openrouter_provider,
    validator_openrouter_reasoning_effort: normalizeOpenRouterReasoningEffort(localConfig.validator_openrouter_reasoning_effort),
    validator_lm_studio_fallback: localConfig.validator_lm_studio_fallback,
    validator_context_window: localConfig.validator_context_window,
    validator_max_tokens: localConfig.validator_max_tokens,
    validator_supercharge_enabled: Boolean(localConfig.validator_supercharge_enabled),
    assistant_provider: localConfig.assistant_provider || localConfig.validator_provider,
    assistant_model: localConfig.assistant_model || localConfig.validator_model,
    assistant_openrouter_provider: localConfig.assistant_openrouter_provider ?? localConfig.validator_openrouter_provider,
    assistant_openrouter_reasoning_effort: normalizeOpenRouterReasoningEffort(localConfig.assistant_openrouter_reasoning_effort || localConfig.validator_openrouter_reasoning_effort),
    assistant_lm_studio_fallback: localConfig.assistant_lm_studio_fallback ?? localConfig.validator_lm_studio_fallback,
    assistant_context_window: localConfig.assistant_context_window || localConfig.validator_context_window,
    assistant_max_tokens: localConfig.assistant_max_tokens || localConfig.validator_max_tokens,
    assistant_supercharge_enabled: Boolean(localConfig.assistant_supercharge_enabled),
    writer_provider: localConfig.writer_provider,
    writer_model: localConfig.writer_model,
    writer_openrouter_provider: localConfig.writer_openrouter_provider,
    writer_openrouter_reasoning_effort: normalizeOpenRouterReasoningEffort(localConfig.writer_openrouter_reasoning_effort),
    writer_lm_studio_fallback: localConfig.writer_lm_studio_fallback,
    writer_context_window: localConfig.writer_context_window,
    writer_max_tokens: localConfig.writer_max_tokens,
    writer_supercharge_enabled: Boolean(localConfig.writer_supercharge_enabled),
    high_param_provider: localConfig.high_param_provider,
    high_param_model: localConfig.high_param_model,
    high_param_openrouter_provider: localConfig.high_param_openrouter_provider,
    high_param_openrouter_reasoning_effort: normalizeOpenRouterReasoningEffort(localConfig.high_param_openrouter_reasoning_effort),
    high_param_lm_studio_fallback: localConfig.high_param_lm_studio_fallback,
    high_param_context_window: localConfig.high_param_context_window,
    high_param_max_tokens: localConfig.high_param_max_tokens,
    high_param_supercharge_enabled: Boolean(localConfig.high_param_supercharge_enabled),
    critique_submitter_provider: localConfig.high_param_provider,
    critique_submitter_model: localConfig.high_param_model,
    critique_submitter_openrouter_provider: localConfig.high_param_openrouter_provider,
    critique_submitter_openrouter_reasoning_effort: normalizeOpenRouterReasoningEffort(localConfig.high_param_openrouter_reasoning_effort),
    critique_submitter_lm_studio_fallback: localConfig.high_param_lm_studio_fallback,
    critique_submitter_context_window: localConfig.high_param_context_window,
    critique_submitter_max_tokens: localConfig.high_param_max_tokens,
    critique_submitter_supercharge_enabled: Boolean(localConfig.high_param_supercharge_enabled),
    allow_mathematical_proofs: Boolean(normalized.allowMathematicalProofs),
    allow_research_papers: Boolean(normalized.allowResearchPapers),
    tier3_enabled: normalized.tier3Enabled ?? false,
  };
}

function buildLocalConfigFromLmStudio(modelId = '') {
  return {
    ...DEFAULT_LM_LOCAL_CONFIG,
    validator_model: modelId,
    assistant_model: modelId,
    writer_model: modelId,
    high_param_model: modelId,
  };
}

function choosePublicStartupModel(availableModels = [], publicModels = []) {
  const modelList = Array.isArray(availableModels) ? availableModels : [];
  for (const publicModel of publicModels) {
    if (modelList.some((model) => model?.id === publicModel.id)) {
      return publicModel;
    }
  }
  return publicModels[0];
}

function chooseCloudAccessStartupModel(providerId = OPENAI_CODEX_STARTUP_CHOICE, models = []) {
  // Use live OAuth model lists only as availability signals. Persisted defaults
  // are reconstructed from public constants so account-scoped response objects
  // never flow into localStorage.
  return providerId === XAI_GROK_STARTUP_CHOICE
    ? choosePublicStartupModel(models, PUBLIC_XAI_GROK_STARTUP_MODELS)
    : choosePublicStartupModel(models, PUBLIC_CODEX_STARTUP_MODELS);
}

function buildStartupRoleDefaults(providerId = OPENAI_CODEX_STARTUP_CHOICE, model = DEFAULT_CODEX_STARTUP_MODEL) {
  const autoSettings = providerId === OPENAI_CODEX_STARTUP_CHOICE
    ? computeCodexAutoSettings(model)
    : providerId === XAI_GROK_STARTUP_CHOICE
      ? computeXAIGrokAutoSettings(model)
      : computeCloudAccessAutoSettings(model, 'OpenRouter/OAuth');
  return {
    provider: providerId,
    modelId: model.id || (providerId === XAI_GROK_STARTUP_CHOICE ? DEFAULT_XAI_GROK_STARTUP_MODEL.id : DEFAULT_CODEX_STARTUP_MODEL.id),
    openrouterProvider: null,
    openrouterReasoningEffort: DEFAULT_OPENROUTER_REASONING_EFFORT,
    lmStudioFallbackId: null,
    contextWindow: autoSettings.contextWindowKnown ? autoSettings.contextWindow : DEFAULT_CONTEXT_WINDOW,
    maxOutputTokens: autoSettings.outputCapKnown ? autoSettings.maxOutputTokens : DEFAULT_MAX_OUTPUT_TOKENS,
    superchargeEnabled: false,
  };
}

function buildStartupLocalConfig(roleDefaults = buildStartupRoleDefaults()) {
  return {
    validator_provider: roleDefaults.provider,
    validator_model: roleDefaults.modelId,
    validator_openrouter_provider: null,
    validator_openrouter_reasoning_effort: DEFAULT_OPENROUTER_REASONING_EFFORT,
    validator_lm_studio_fallback: null,
    validator_context_window: roleDefaults.contextWindow,
    validator_max_tokens: roleDefaults.maxOutputTokens,
    validator_supercharge_enabled: false,
    assistant_provider: roleDefaults.provider,
    assistant_model: roleDefaults.modelId,
    assistant_openrouter_provider: null,
    assistant_openrouter_reasoning_effort: DEFAULT_OPENROUTER_REASONING_EFFORT,
    assistant_lm_studio_fallback: null,
    assistant_context_window: roleDefaults.contextWindow,
    assistant_max_tokens: roleDefaults.maxOutputTokens,
    assistant_supercharge_enabled: false,
    writer_provider: roleDefaults.provider,
    writer_model: roleDefaults.modelId,
    writer_openrouter_provider: null,
    writer_openrouter_reasoning_effort: DEFAULT_OPENROUTER_REASONING_EFFORT,
    writer_lm_studio_fallback: null,
    writer_context_window: roleDefaults.contextWindow,
    writer_max_tokens: roleDefaults.maxOutputTokens,
    writer_supercharge_enabled: false,
    high_param_provider: roleDefaults.provider,
    high_param_model: roleDefaults.modelId,
    high_param_openrouter_provider: null,
    high_param_openrouter_reasoning_effort: DEFAULT_OPENROUTER_REASONING_EFFORT,
    high_param_lm_studio_fallback: null,
    high_param_context_window: roleDefaults.contextWindow,
    high_param_max_tokens: roleDefaults.maxOutputTokens,
    high_param_supercharge_enabled: false,
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

export function applyCodexStartupDefaults(models = []) {
  return applyCloudAccessStartupDefaults(OPENAI_CODEX_STARTUP_CHOICE, models);
}

export function applyCloudAccessStartupDefaults(providerId = OPENAI_CODEX_STARTUP_CHOICE, models = []) {
  const selectedModel = chooseCloudAccessStartupModel(providerId, models);
  const roleDefaults = buildStartupRoleDefaults(providerId, selectedModel);
  const submitterConfigs = [1, 2, 3].map((submitterId) => ({
    ...roleDefaults,
    submitterId,
  }));
  const currentSettings = getStoredAutonomousSettings();
  const nextSettings = persistAutonomousSettings({
    ...currentSettings,
    numSubmitters: 3,
    submitterConfigs,
    localConfig: {
      ...currentSettings.localConfig,
      ...buildStartupLocalConfig(roleDefaults),
    },
    selectedProfile: '',
  });

  return {
    settings: nextSettings,
    config: settingsToAutonomousConfig(nextSettings),
    modelId: selectedModel.id || (providerId === XAI_GROK_STARTUP_CHOICE ? DEFAULT_XAI_GROK_STARTUP_MODEL.id : DEFAULT_CODEX_STARTUP_MODEL.id),
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
    openrouterReasoningEffort: normalizeOpenRouterReasoningEffort(submitterProfile.openrouterReasoningEffort),
    lmStudioFallbackId: isRecommended ? null : (submitterProfile.lmStudioFallbackId || null),
    contextWindow: submitterProfile.contextWindow,
    maxOutputTokens: submitterProfile.maxOutputTokens,
    superchargeEnabled: Boolean(submitterProfile.superchargeEnabled),
  }));

  const getModelId = (roleProfile = {}) => (
    isRecommended
      ? convertToApiId(roleProfile.modelId || '')
      : (roleProfile.modelId || '')
  );

  const getOpenRouterProvider = (roleProfile = {}) => roleProfile.openrouterProvider || null;
  const getOpenRouterReasoningEffort = (roleProfile = {}) => normalizeOpenRouterReasoningEffort(roleProfile.openrouterReasoningEffort);
  const writerProfile = normalizeProfileWriter(profile);
  const rigorProfile = normalizeProfileRigor(profile);

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
      validator_openrouter_reasoning_effort: getOpenRouterReasoningEffort(profile.validator),
      validator_lm_studio_fallback: isRecommended ? null : (profile.validator.lmStudioFallbackId || null),
      validator_context_window: profile.validator.contextWindow,
      validator_max_tokens: profile.validator.maxOutputTokens,
      validator_supercharge_enabled: Boolean(profile.validator.superchargeEnabled),
      assistant_provider: isRecommended ? 'openrouter' : (profile.assistant?.provider || profile.validator.provider || 'openrouter'),
      assistant_model: getModelId(profile.assistant || profile.validator),
      assistant_openrouter_provider: getOpenRouterProvider(profile.assistant || profile.validator),
      assistant_openrouter_reasoning_effort: getOpenRouterReasoningEffort(profile.assistant || profile.validator),
      assistant_lm_studio_fallback: isRecommended ? null : ((profile.assistant || profile.validator).lmStudioFallbackId || null),
      assistant_context_window: (profile.assistant || profile.validator).contextWindow,
      assistant_max_tokens: (profile.assistant || profile.validator).maxOutputTokens,
      assistant_supercharge_enabled: Boolean((profile.assistant || profile.validator).superchargeEnabled),
      writer_provider: isRecommended ? 'openrouter' : (writerProfile.provider || 'openrouter'),
      writer_model: getModelId(writerProfile),
      writer_openrouter_provider: getOpenRouterProvider(writerProfile),
      writer_openrouter_reasoning_effort: getOpenRouterReasoningEffort(writerProfile),
      writer_lm_studio_fallback: isRecommended ? null : (writerProfile.lmStudioFallbackId || null),
      writer_context_window: writerProfile.contextWindow,
      writer_max_tokens: writerProfile.maxOutputTokens,
      writer_supercharge_enabled: Boolean(writerProfile.superchargeEnabled),
      high_param_provider: isRecommended ? 'openrouter' : (rigorProfile.provider || 'openrouter'),
      high_param_model: getModelId(rigorProfile),
      high_param_openrouter_provider: getOpenRouterProvider(rigorProfile),
      high_param_openrouter_reasoning_effort: getOpenRouterReasoningEffort(rigorProfile),
      high_param_lm_studio_fallback: isRecommended ? null : (rigorProfile.lmStudioFallbackId || null),
      high_param_context_window: rigorProfile.contextWindow,
      high_param_max_tokens: rigorProfile.maxOutputTokens,
      high_param_supercharge_enabled: Boolean(rigorProfile.superchargeEnabled),
      critique_submitter_provider: isRecommended ? 'openrouter' : (rigorProfile.provider || 'openrouter'),
      critique_submitter_model: getModelId(rigorProfile),
      critique_submitter_openrouter_provider: getOpenRouterProvider(rigorProfile),
      critique_submitter_openrouter_reasoning_effort: getOpenRouterReasoningEffort(rigorProfile),
      critique_submitter_lm_studio_fallback: isRecommended ? null : (rigorProfile.lmStudioFallbackId || null),
      critique_submitter_context_window: rigorProfile.contextWindow,
      critique_submitter_max_tokens: rigorProfile.maxOutputTokens,
      critique_submitter_supercharge_enabled: Boolean(rigorProfile.superchargeEnabled),
    },
    selectedProfile: profileKey,
  });

  return {
    profile,
    settings: nextSettings,
    config: settingsToAutonomousConfig(nextSettings),
  };
}
