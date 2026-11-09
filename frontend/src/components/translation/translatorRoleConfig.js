import {
  DEFAULT_OPENROUTER_REASONING_EFFORT,
  normalizeOpenRouterReasoningEffort,
} from '../../utils/openRouterSelection';

// Translator role helpers for the Auto Top-p Translate modal. The backend
// settings contract is snake_case (`model_id`, `context_window`, ...); the
// shared RoleModelEditor edits camelCase configs (`modelId`, `contextWindow`, ...).

export const TRANSLATOR_SUBMITTER_TITLE = 'Translator Submitter';
export const TRANSLATOR_VALIDATOR_TITLE = 'Translator Validator';
export const GENERIC_MODE_PROVIDER_MESSAGE = 'Hosted mode supports OpenRouter translator models only.';

// Readable message for a failed translation request. Validation errors (400/422)
// drop the "HTTP 422 …:" prefix so the backend's safe message reads directly.
export function describeTranslationError(error, fallback = 'The request failed.') {
  const message = typeof error?.message === 'string' ? error.message.trim() : '';
  if (!message) {
    return fallback;
  }
  if (error?.status === 400 || error?.status === 422) {
    return message.replace(/^HTTP \d{3}[^:]*:\s*/, '') || message;
  }
  return message;
}

function cleanString(value) {
  return typeof value === 'string' ? value.trim() : '';
}

function optionalString(value) {
  return cleanString(value) || null;
}

function positiveIntegerOrBlank(value) {
  const parsed = typeof value === 'number' ? value : parseInt(cleanString(String(value ?? '')), 10);
  return Number.isFinite(parsed) && parsed > 0 ? Math.floor(parsed) : '';
}

export function createEmptyTranslatorRole({ lmStudioEnabled = true } = {}) {
  return {
    provider: lmStudioEnabled ? 'lm_studio' : 'openrouter',
    modelId: '',
    openrouterProvider: null,
    openrouterReasoningEffort: DEFAULT_OPENROUTER_REASONING_EFFORT,
    lmStudioFallbackId: null,
    contextWindow: '',
    maxOutputTokens: '',
  };
}

// Hosted (generic) mode supports OpenRouter only: any other provider resets to
// a blank OpenRouter role and LM Studio fallbacks are dropped.
export function normalizeRoleForMode(role, { lmStudioEnabled = true } = {}) {
  if (lmStudioEnabled) {
    return role;
  }
  if (role.provider === 'openrouter') {
    return role.lmStudioFallbackId ? { ...role, lmStudioFallbackId: null } : role;
  }
  return {
    ...role,
    provider: 'openrouter',
    modelId: '',
    openrouterProvider: null,
    openrouterReasoningEffort: DEFAULT_OPENROUTER_REASONING_EFFORT,
    lmStudioFallbackId: null,
  };
}

// snake_case backend role -> camelCase editor config.
export function roleFromSettings(role, options = {}) {
  const { lmStudioEnabled = true } = options;
  if (!role || typeof role !== 'object') {
    return createEmptyTranslatorRole(options);
  }
  const provider = cleanString(role.provider) || (lmStudioEnabled ? 'lm_studio' : 'openrouter');
  return normalizeRoleForMode({
    provider,
    modelId: cleanString(role.model_id),
    openrouterProvider: optionalString(role.openrouter_provider),
    openrouterReasoningEffort: normalizeOpenRouterReasoningEffort(role.openrouter_reasoning_effort, provider),
    lmStudioFallbackId: optionalString(role.lm_studio_fallback_id),
    contextWindow: positiveIntegerOrBlank(role.context_window),
    maxOutputTokens: positiveIntegerOrBlank(role.max_output_tokens),
  }, options);
}

// camelCase editor config -> snake_case backend role (exactly the backend fields).
export function roleToSettings(role, options = {}) {
  const normalized = normalizeRoleForMode({ ...createEmptyTranslatorRole(options), ...(role || {}) }, options);
  const provider = cleanString(normalized.provider) || 'openrouter';
  return {
    provider,
    model_id: cleanString(normalized.modelId),
    openrouter_provider: provider === 'openrouter' ? optionalString(normalized.openrouterProvider) : null,
    openrouter_reasoning_effort: normalizeOpenRouterReasoningEffort(normalized.openrouterReasoningEffort, provider),
    lm_studio_fallback_id: provider === 'lm_studio' ? null : optionalString(normalized.lmStudioFallbackId),
    context_window: positiveIntegerOrBlank(normalized.contextWindow) || 0,
    max_output_tokens: positiveIntegerOrBlank(normalized.maxOutputTokens) || 0,
  };
}

// Keep only the backend role fields (the settings model forbids extras).
export function sanitizeRolePayload(role) {
  if (!role || typeof role !== 'object') {
    return null;
  }
  return {
    provider: role.provider,
    model_id: role.model_id,
    openrouter_provider: role.openrouter_provider ?? null,
    openrouter_reasoning_effort: role.openrouter_reasoning_effort ?? DEFAULT_OPENROUTER_REASONING_EFFORT,
    lm_studio_fallback_id: role.lm_studio_fallback_id ?? null,
    context_window: role.context_window,
    max_output_tokens: role.max_output_tokens,
  };
}

export function sanitizeLanguagePayload(language) {
  return {
    key: String(language?.key || ''),
    name: String(language?.name || ''),
    custom: Boolean(language?.custom),
  };
}

export function hasSavedTranslatorModels(settings) {
  return Boolean(settings?.translator) && Boolean(settings?.validator);
}

// Stable identity of the saved translator roles (changes only when roles change).
export function savedRolesSignature(settings) {
  return JSON.stringify({
    translator: settings?.translator || null,
    validator: settings?.validator || null,
  });
}

function roleOrEmpty(role, options) {
  return role ? roleFromSettings(role, options) : createEmptyTranslatorRole(options);
}

// Editable draft of the two translator roles from saved settings.
export function draftFromSettings(settings, options = {}) {
  return {
    translator: roleOrEmpty(settings?.translator, options),
    validator: roleOrEmpty(settings?.validator, options),
  };
}

export function draftToSettingsRoles(draft, options = {}) {
  return {
    translator: roleToSettings(draft?.translator, options),
    validator: roleToSettings(draft?.validator, options),
  };
}

export function isDraftDirty(draft, baseline, options = {}) {
  return JSON.stringify(draftToSettingsRoles(draft, options))
    !== JSON.stringify(draftToSettingsRoles(baseline, options));
}

// Map stored Autonomous Research settings (utils/autonomousProfiles) onto the
// translator roles: Submitter 1 (`submitterConfigs[0]`, camelCase) becomes the
// translator submitter and the `localConfig.validator_*` fields become the
// translator validator. A role with no model is not copied; returns null when
// neither role names a model.
export function rolesFromAutonomousSettings(stored, options = {}) {
  const submitterOne = Array.isArray(stored?.submitterConfigs) ? stored.submitterConfigs[0] : null;
  const translator = roleFromSettings({
    provider: submitterOne?.provider,
    model_id: submitterOne?.modelId,
    openrouter_provider: submitterOne?.openrouterProvider,
    openrouter_reasoning_effort: submitterOne?.openrouterReasoningEffort,
    lm_studio_fallback_id: submitterOne?.lmStudioFallbackId,
    context_window: submitterOne?.contextWindow,
    max_output_tokens: submitterOne?.maxOutputTokens,
  }, options);
  const local = stored?.localConfig || {};
  const validator = roleFromSettings({
    provider: local.validator_provider,
    model_id: local.validator_model,
    openrouter_provider: local.validator_openrouter_provider,
    openrouter_reasoning_effort: local.validator_openrouter_reasoning_effort,
    lm_studio_fallback_id: local.validator_lm_studio_fallback,
    context_window: local.validator_context_window,
    max_output_tokens: local.validator_max_tokens,
  }, options);
  if (!translator.modelId && !validator.modelId) {
    return null;
  }
  return {
    translator: translator.modelId ? translator : null,
    validator: validator.modelId ? validator : null,
  };
}

// Apply copied Autonomous roles over a draft; a role the copy lacks stays as it is.
export function applyCopiedRoles(draft, copied) {
  if (!copied) {
    return draft;
  }
  return {
    translator: copied.translator || draft.translator,
    validator: copied.validator || draft.validator,
  };
}

export function validateTranslatorDraft(draft, { lmStudioEnabled = true } = {}) {
  const roles = [
    [TRANSLATOR_SUBMITTER_TITLE, draft?.translator],
    [TRANSLATOR_VALIDATOR_TITLE, draft?.validator],
  ];
  const errors = [];
  roles.forEach(([title, role]) => {
    const config = role || {};
    if (!lmStudioEnabled && config.provider && config.provider !== 'openrouter') {
      errors.push(`${title}: ${GENERIC_MODE_PROVIDER_MESSAGE}`);
    }
    if (!cleanString(config.modelId)) {
      errors.push(`Select a model for ${title}.`);
    }
    const contextWindow = positiveIntegerOrBlank(config.contextWindow);
    const maxOutputTokens = positiveIntegerOrBlank(config.maxOutputTokens);
    if (!contextWindow) {
      errors.push(`Enter a context window greater than 0 for ${title}.`);
    }
    if (!maxOutputTokens) {
      errors.push(`Enter max output tokens greater than 0 for ${title}.`);
    }
    if (contextWindow && maxOutputTokens && maxOutputTokens >= contextWindow) {
      errors.push(`${title}: max output tokens must be smaller than the context window.`);
    }
  });
  return errors;
}
