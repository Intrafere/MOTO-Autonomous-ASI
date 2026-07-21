import { loadModelCache, getModelApiId } from './modelCache';
import {
  DEFAULT_CONTEXT_WINDOW,
  DEFAULT_MAX_OUTPUT_TOKENS,
  DEFAULT_OPENROUTER_REASONING_EFFORT,
  normalizeOpenRouterReasoningEffort,
} from './openRouterSelection';

export const LEANOJ_SETTINGS_STORAGE_KEY = 'leanoj_solver_settings';
export const LEANOJ_PROFILES_STORAGE_KEY = 'leanoj_solver_profiles';
export const LEANOJ_RECOMMENDED_PROFILE_KEY = 'leanoj_recommended_balanced_proof';
export const LEANOJ_LAB_GRADE_PROFILE_KEY = 'leanoj_recommended_lab_grade_solver';

export const LEANOJ_ROLE_KEYS = [
  'topic_generator',
  'topic_validator',
  'brainstorm_validator',
  'final_solver',
  'assistant',
];

const GEMINI_FLASH_LATEST_MODEL = '~google/gemini-flash-latest';
const GEMINI_FLASH_LATEST_CONTEXT_WINDOW = 1048576;
const GEMINI_FLASH_LATEST_MAX_OUTPUT_TOKENS = 65536;
const MINIMAX_M3_MODEL = 'minimax/minimax-m3';
const MINIMAX_M3_CONTEXT_WINDOW = 1048576;
const MINIMAX_M3_MAX_OUTPUT_TOKENS = 131072;
const GPT_55_MODEL = 'openai/gpt-5.5';
const GPT_55_CONTEXT_WINDOW = 400000;
const GPT_55_MAX_OUTPUT_TOKENS = 65536;
const CLAUDE_OPUS_LATEST_MODEL = '~anthropic/claude-opus-latest';

const DEFAULT_ROLE_CONFIG = {
  provider: 'lm_studio',
  modelId: '',
  openrouterProvider: null,
  openrouterReasoningEffort: DEFAULT_OPENROUTER_REASONING_EFFORT,
  lmStudioFallbackId: null,
  contextWindow: DEFAULT_CONTEXT_WINDOW,
  maxOutputTokens: DEFAULT_MAX_OUTPUT_TOKENS,
  superchargeEnabled: false,
};

const DEFAULT_SUBMITTER_CONFIG = {
  submitterId: 1,
  ...DEFAULT_ROLE_CONFIG,
};

const role = (modelId, contextWindow = DEFAULT_CONTEXT_WINDOW, maxOutputTokens = DEFAULT_MAX_OUTPUT_TOKENS) => ({
  provider: 'openrouter',
  modelId,
  openrouterProvider: null,
  openrouterReasoningEffort: DEFAULT_OPENROUTER_REASONING_EFFORT,
  lmStudioFallbackId: null,
  contextWindow,
  maxOutputTokens,
});

const geminiFlashLatestRole = () => role(
  GEMINI_FLASH_LATEST_MODEL,
  GEMINI_FLASH_LATEST_CONTEXT_WINDOW,
  GEMINI_FLASH_LATEST_MAX_OUTPUT_TOKENS
);

const minimaxM3Role = () => role(
  MINIMAX_M3_MODEL,
  MINIMAX_M3_CONTEXT_WINDOW,
  MINIMAX_M3_MAX_OUTPUT_TOKENS
);

const gpt55Role = () => role(
  GPT_55_MODEL,
  GPT_55_CONTEXT_WINDOW,
  GPT_55_MAX_OUTPUT_TOKENS
);

export const LEANOJ_RECOMMENDED_PROFILES = {
  [LEANOJ_RECOMMENDED_PROFILE_KEY]: {
    name: 'Balanced Proof Solver',
    numSubmitters: 3,
    submitters: [
      minimaxM3Role(),
      role('deepseek/deepseek-v4-pro', 1048576, 65500),
      role('google/gemini-3.1-pro-preview', 1048576, 65500),
    ],
    roles: {
      topic_generator: minimaxM3Role(),
      topic_validator: geminiFlashLatestRole(),
      brainstorm_validator: geminiFlashLatestRole(),
      assistant: geminiFlashLatestRole(),
      final_solver: role('google/gemini-3.1-pro-preview', 1048576, 65500),
    },
  },
  [LEANOJ_LAB_GRADE_PROFILE_KEY]: {
    name: 'Lab Grade Solver',
    numSubmitters: 3,
    submitters: [
      gpt55Role(),
      role('deepseek/deepseek-v4-pro', 1048576, 65500),
      role(CLAUDE_OPUS_LATEST_MODEL, 1048576, 65500),
    ],
    roles: {
      topic_generator: gpt55Role(),
      topic_validator: geminiFlashLatestRole(),
      brainstorm_validator: geminiFlashLatestRole(),
      assistant: geminiFlashLatestRole(),
      final_solver: role(CLAUDE_OPUS_LATEST_MODEL, 1048576, 65500),
    },
  },
};

const DEFAULT_PROFILE = LEANOJ_RECOMMENDED_PROFILES[LEANOJ_RECOMMENDED_PROFILE_KEY];

const createDefaultSubmitters = (modelId = '') => (
  [1, 2, 3].map((submitterId) => ({
    ...DEFAULT_SUBMITTER_CONFIG,
    submitterId,
    modelId,
  }))
);

const createDefaultRoles = (modelId = '') => (
  LEANOJ_ROLE_KEYS.reduce((acc, roleKey) => {
    acc[roleKey] = {
      ...DEFAULT_ROLE_CONFIG,
      modelId,
    };
    return acc;
  }, {})
);

const profileSubmitters = DEFAULT_PROFILE.submitters.map((submitter, index) => ({
  ...DEFAULT_SUBMITTER_CONFIG,
  ...submitter,
  submitterId: index + 1,
}));

const DEFAULT_SETTINGS = {
  prompt: '',
  leanTemplate: '',
  numSubmitters: DEFAULT_PROFILE.numSubmitters,
  submitterConfigs: profileSubmitters,
  roles: DEFAULT_PROFILE.roles,
  maxInitialBrainstormAccepts: 30,
  maxRecursiveBrainstormAccepts: 10,
  finalAttemptsPerCycle: 30,
  freeOnly: false,
  freeModelLooping: false,
  freeModelAutoSelector: false,
  creativityEmphasisBoostEnabled: false,
  modelProviders: {},
  selectedProfile: LEANOJ_RECOMMENDED_PROFILE_KEY,
};

function normalizeRoleConfig(config = {}) {
  return {
    ...DEFAULT_ROLE_CONFIG,
    ...config,
    openrouterProvider: config.openrouterProvider || null,
    openrouterReasoningEffort: normalizeOpenRouterReasoningEffort(config.openrouterReasoningEffort),
    lmStudioFallbackId: config.lmStudioFallbackId || null,
  };
}

function normalizeSubmitterConfig(config = {}, index = 0) {
  return {
    ...DEFAULT_SUBMITTER_CONFIG,
    ...normalizeRoleConfig(config),
    submitterId: config.submitterId || index + 1,
  };
}

export function normalizeLeanOJSettings(settings = {}) {
  const submitterConfigs = Array.isArray(settings.submitterConfigs) && settings.submitterConfigs.length > 0
    ? settings.submitterConfigs.map(normalizeSubmitterConfig)
    : DEFAULT_SETTINGS.submitterConfigs;

  const roles = LEANOJ_ROLE_KEYS.reduce((acc, roleKey) => {
    const rawRoles = settings.roles || {};
    acc[roleKey] = normalizeRoleConfig(
      rawRoles[roleKey]
      || (roleKey === 'assistant' ? rawRoles.topic_validator : null)
      || DEFAULT_SETTINGS.roles[roleKey]
      || DEFAULT_SETTINGS.roles.topic_validator
    );
    return acc;
  }, {});

  return {
    ...DEFAULT_SETTINGS,
    ...settings,
    numSubmitters: settings.numSubmitters || submitterConfigs.length || DEFAULT_SETTINGS.numSubmitters,
    submitterConfigs,
    roles,
    maxInitialBrainstormAccepts: settings.maxInitialBrainstormAccepts ?? DEFAULT_SETTINGS.maxInitialBrainstormAccepts,
    maxRecursiveBrainstormAccepts: settings.maxRecursiveBrainstormAccepts ?? DEFAULT_SETTINGS.maxRecursiveBrainstormAccepts,
    finalAttemptsPerCycle: Math.max(30, Number(settings.finalAttemptsPerCycle ?? DEFAULT_SETTINGS.finalAttemptsPerCycle)),
    creativityEmphasisBoostEnabled: settings.creativityEmphasisBoostEnabled ?? DEFAULT_SETTINGS.creativityEmphasisBoostEnabled,
    modelProviders: settings.modelProviders || DEFAULT_SETTINGS.modelProviders,
    selectedProfile: settings.selectedProfile ?? DEFAULT_SETTINGS.selectedProfile,
  };
}

export function getStoredLeanOJSettings() {
  try {
    const raw = localStorage.getItem(LEANOJ_SETTINGS_STORAGE_KEY);
    if (!raw) return normalizeLeanOJSettings();
    return normalizeLeanOJSettings(JSON.parse(raw));
  } catch (error) {
    console.error('Failed to load Proof Solver settings:', error);
    return normalizeLeanOJSettings();
  }
}

export function persistLeanOJSettings(settings) {
  const normalized = normalizeLeanOJSettings(settings);
  localStorage.setItem(LEANOJ_SETTINGS_STORAGE_KEY, JSON.stringify(normalized));
  return normalized;
}

function positiveRoleSetting(value, label) {
  const parsed = Number(value);
  if (!Number.isFinite(parsed) || parsed <= 0) {
    throw new Error(`${label} must be configured as a positive number in Proof Solver Settings.`);
  }
  return Math.floor(parsed);
}

const roleToApi = (config = {}, label = 'Role') => ({
  provider: config.provider || 'lm_studio',
  model_id: config.modelId || '',
  openrouter_provider: config.openRouterProvider || config.openrouterProvider || null,
  openrouter_reasoning_effort: normalizeOpenRouterReasoningEffort(config.openrouterReasoningEffort),
  lm_studio_fallback_id: config.lmStudioFallbackId || null,
  context_window: positiveRoleSetting(config.contextWindow, `${label} context window`),
  max_output_tokens: positiveRoleSetting(config.maxOutputTokens, `${label} max output tokens`),
  supercharge_enabled: Boolean(config.superchargeEnabled),
});

export function settingsToLeanOJRequest(settings, prompt, leanTemplate, options = {}) {
  const normalized = normalizeLeanOJSettings(settings);
  const roles = normalized.roles;
  const topicGenerator = normalized.submitterConfigs[0] || roles.topic_generator;
  const assistantEnabled = options.assistantEnabled !== false;
  const assistantConfig = assistantEnabled
    ? roleToApi(roles.assistant || roles.topic_validator, 'Assistant')
    : {
        ...roleToApi(roles.topic_validator, 'Topic validator'),
        model_id: '',
        openrouter_provider: null,
        lm_studio_fallback_id: null,
      };
  return {
    user_prompt: prompt ?? normalized.prompt ?? '',
    lean_template: leanTemplate ?? normalized.leanTemplate ?? '',
    creativity_emphasis_boost_enabled: Boolean(normalized.creativityEmphasisBoostEnabled),
    topic_generator: roleToApi(topicGenerator, 'Topic generator'),
    topic_validator: roleToApi(roles.topic_validator, 'Topic validator'),
    brainstorm_submitters: normalized.submitterConfigs.map((config, index) => roleToApi(config, `Brainstorm submitter ${index + 1}`)),
    brainstorm_validator: roleToApi(roles.brainstorm_validator, 'Brainstorm validator'),
    path_decider: roleToApi(roles.final_solver, 'Final proof solver'),
    final_solver: roleToApi(roles.final_solver, 'Final proof solver'),
    assistant: assistantConfig,
    max_initial_brainstorm_accepts: Number(normalized.maxInitialBrainstormAccepts || 30),
    max_recursive_brainstorm_accepts: Number(normalized.maxRecursiveBrainstormAccepts || 10),
    final_attempts_per_cycle: Number(normalized.finalAttemptsPerCycle || 30),
  };
}

export function applyLeanOJLmStudioDefaults(modelId = '') {
  const current = getStoredLeanOJSettings();
  const next = persistLeanOJSettings({
    ...current,
    numSubmitters: 3,
    submitterConfigs: createDefaultSubmitters(modelId),
    roles: createDefaultRoles(modelId),
    selectedProfile: '',
  });
  return next;
}

export async function applyLeanOJProfileSelection(profileKey, userProfiles = {}) {
  const isRecommended = profileKey.startsWith('leanoj_recommended_');
  const profile = isRecommended
    ? LEANOJ_RECOMMENDED_PROFILES[profileKey]
    : userProfiles[profileKey];
  if (!profile) {
    throw new Error(`Proof Solver profile not found: ${profileKey}`);
  }

  await loadModelCache();
  const toModelId = (modelId = '') => (
    isRecommended ? (getModelApiId(modelId) || modelId) : modelId
  );
  const convertRole = (config = {}) => ({
    ...normalizeRoleConfig(config),
    provider: isRecommended ? 'openrouter' : (config.provider || 'openrouter'),
    modelId: toModelId(config.modelId || ''),
    lmStudioFallbackId: isRecommended ? null : (config.lmStudioFallbackId || null),
  });

  const current = getStoredLeanOJSettings();
  const roles = LEANOJ_ROLE_KEYS.reduce((acc, roleKey) => {
    acc[roleKey] = convertRole((profile.roles || {})[roleKey] || (roleKey === 'assistant' ? (profile.roles || {}).topic_validator : {}));
    return acc;
  }, {});
  const submitterConfigs = (profile.submitters || []).map((submitter, index) => ({
    ...convertRole(submitter),
    submitterId: index + 1,
  }));

  const nextSettings = persistLeanOJSettings({
    ...current,
    numSubmitters: profile.numSubmitters || submitterConfigs.length,
    submitterConfigs,
    roles,
    maxInitialBrainstormAccepts: profile.maxInitialBrainstormAccepts ?? current.maxInitialBrainstormAccepts,
    maxRecursiveBrainstormAccepts: profile.maxRecursiveBrainstormAccepts ?? current.maxRecursiveBrainstormAccepts,
    finalAttemptsPerCycle: profile.finalAttemptsPerCycle ?? current.finalAttemptsPerCycle,
    selectedProfile: profileKey,
  });

  return {
    profile,
    settings: nextSettings,
  };
}
