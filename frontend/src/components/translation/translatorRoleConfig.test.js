import {
  GENERIC_MODE_PROVIDER_MESSAGE,
  TRANSLATOR_SUBMITTER_TITLE,
  TRANSLATOR_VALIDATOR_TITLE,
  applyCopiedRoles,
  createEmptyTranslatorRole,
  describeTranslationError,
  draftFromSettings,
  draftToSettingsRoles,
  hasSavedTranslatorModels,
  isDraftDirty,
  roleFromSettings,
  roleToSettings,
  rolesFromAutonomousSettings,
  sanitizeLanguagePayload,
  sanitizeRolePayload,
  savedRolesSignature,
  validateTranslatorDraft,
} from './translatorRoleConfig';

const SNAKE_ROLE = {
  provider: 'openrouter',
  model_id: 'openrouter/alpha',
  openrouter_provider: 'Anthropic',
  openrouter_reasoning_effort: 'high',
  lm_studio_fallback_id: 'lm-local',
  context_window: 131072,
  max_output_tokens: 25000,
};

const CAMEL_ROLE = {
  provider: 'openrouter',
  modelId: 'openrouter/alpha',
  openrouterProvider: 'Anthropic',
  openrouterReasoningEffort: 'high',
  lmStudioFallbackId: 'lm-local',
  contextWindow: 131072,
  maxOutputTokens: 25000,
};

describe('role conversion', () => {
  test('converts a backend role to the editor shape and back without loss', () => {
    expect(roleFromSettings(SNAKE_ROLE)).toEqual(CAMEL_ROLE);
    expect(roleToSettings(CAMEL_ROLE)).toEqual(SNAKE_ROLE);
  });

  test('drops fields that do not apply to the provider and coerces numbers', () => {
    expect(roleToSettings({
      provider: 'lm_studio',
      modelId: '  lm-one ',
      openrouterProvider: 'Anthropic',
      openrouterReasoningEffort: 'not-a-level',
      lmStudioFallbackId: 'lm-two',
      contextWindow: '32768',
      maxOutputTokens: '4096',
    })).toEqual({
      provider: 'lm_studio',
      model_id: 'lm-one',
      openrouter_provider: null,
      openrouter_reasoning_effort: 'auto',
      lm_studio_fallback_id: null,
      context_window: 32768,
      max_output_tokens: 4096,
    });
  });

  test('keeps max reasoning effort only for providers that support it', () => {
    expect(roleToSettings({ ...CAMEL_ROLE, openrouterReasoningEffort: 'max' }).openrouter_reasoning_effort).toBe('auto');
    expect(roleToSettings({
      ...CAMEL_ROLE,
      provider: 'openai_codex_oauth',
      openrouterReasoningEffort: 'max',
    }).openrouter_reasoning_effort).toBe('max');
  });

  test('hosted mode resets non-OpenRouter roles and removes LM Studio fallbacks', () => {
    const hosted = { lmStudioEnabled: false };
    expect(roleFromSettings({ ...SNAKE_ROLE, provider: 'lm_studio', model_id: 'lm-one' }, hosted)).toMatchObject({
      provider: 'openrouter',
      modelId: '',
      lmStudioFallbackId: null,
    });
    expect(roleToSettings(CAMEL_ROLE, hosted).lm_studio_fallback_id).toBeNull();
    expect(createEmptyTranslatorRole(hosted).provider).toBe('openrouter');
    expect(createEmptyTranslatorRole().provider).toBe('lm_studio');
  });

  test('payload sanitizers keep exactly the backend fields', () => {
    expect(sanitizeRolePayload({ ...SNAKE_ROLE, supercharge_enabled: true, extra: 1 })).toEqual(SNAKE_ROLE);
    expect(sanitizeRolePayload(null)).toBeNull();
    expect(sanitizeLanguagePayload({ key: 'fr', name: 'French', proficiency: 'uncertain' })).toEqual({
      key: 'fr', name: 'French', custom: false,
    });
  });
});

describe('drafts', () => {
  test('a fresh config starts with a blank translator and a blank validator', () => {
    const draft = draftFromSettings({ languages: [], translator: null, validator: null });
    expect(Object.keys(draft)).toEqual(['translator', 'validator']);
    expect(draft.translator).toEqual(createEmptyTranslatorRole());
    expect(draft.validator).toEqual(createEmptyTranslatorRole());
    expect(hasSavedTranslatorModels({ translator: null, validator: null })).toBe(false);
    expect(hasSavedTranslatorModels({ translator: SNAKE_ROLE, validator: null })).toBe(false);
    expect(hasSavedTranslatorModels({ translator: null, validator: SNAKE_ROLE })).toBe(false);
  });

  test('saved roles round-trip through the draft and detect edits', () => {
    const settings = { translator: SNAKE_ROLE, validator: { ...SNAKE_ROLE, model_id: 'openrouter/beta' } };
    const draft = draftFromSettings(settings);
    expect(draftToSettingsRoles(draft)).toEqual(settings);
    expect(hasSavedTranslatorModels(settings)).toBe(true);
    expect(isDraftDirty(draft, draftFromSettings(settings))).toBe(false);
    const edited = { ...draft, validator: { ...draft.validator, maxOutputTokens: 8000 } };
    expect(isDraftDirty(edited, draft)).toBe(true);
    expect(isDraftDirty({ ...draft, translator: { ...draft.translator, modelId: 'other' } }, draft)).toBe(true);
  });

  test('the roles signature follows the roles only', () => {
    const settings = { translator: SNAKE_ROLE, validator: SNAKE_ROLE, batch_translation: false };
    expect(savedRolesSignature(settings)).toBe(savedRolesSignature({
      ...settings,
      languages: [{ key: 'fr' }],
      batch_translation: true,
    }));
    expect(savedRolesSignature(settings)).not.toBe(savedRolesSignature({ ...settings, translator: null }));
    expect(savedRolesSignature(null)).toBe(savedRolesSignature({ translator: null, validator: null }));
  });

  test('role titles', () => {
    expect(TRANSLATOR_SUBMITTER_TITLE).toBe('Translator Submitter');
    expect(TRANSLATOR_VALIDATOR_TITLE).toBe('Translator Validator');
  });
});

describe('copy from Autonomous Research settings', () => {
  const stored = {
    submitterConfigs: [
      {
        submitterId: 1,
        provider: 'openrouter',
        modelId: 'openrouter/one',
        openrouterProvider: 'Google',
        openrouterReasoningEffort: 'medium',
        lmStudioFallbackId: null,
        contextWindow: 200000,
        maxOutputTokens: 40000,
        superchargeEnabled: true,
      },
      {
        submitterId: 2,
        provider: 'lm_studio',
        modelId: 'lm-two',
        openrouterProvider: null,
        openrouterReasoningEffort: 'auto',
        lmStudioFallbackId: null,
        contextWindow: 32768,
        maxOutputTokens: 8192,
      },
    ],
    localConfig: {
      validator_provider: 'openrouter',
      validator_model: 'openrouter/validator',
      validator_openrouter_provider: null,
      validator_openrouter_reasoning_effort: 'xhigh',
      validator_lm_studio_fallback: 'lm-fallback',
      validator_context_window: 400000,
      validator_max_tokens: 64000,
      validator_supercharge_enabled: true,
      writer_model: 'ignored',
    },
  };

  test('maps Submitter 1 to the translator and validator_* fields to the validator', () => {
    const copied = rolesFromAutonomousSettings(stored);
    expect(copied).toEqual({
      translator: {
        provider: 'openrouter',
        modelId: 'openrouter/one',
        openrouterProvider: 'Google',
        openrouterReasoningEffort: 'medium',
        lmStudioFallbackId: null,
        contextWindow: 200000,
        maxOutputTokens: 40000,
      },
      validator: {
        provider: 'openrouter',
        modelId: 'openrouter/validator',
        openrouterProvider: null,
        openrouterReasoningEffort: 'xhigh',
        lmStudioFallbackId: 'lm-fallback',
        contextWindow: 400000,
        maxOutputTokens: 64000,
      },
    });
  });

  test('only Submitter 1 is copied, even when it has no model', () => {
    const noFirstModel = {
      ...stored,
      submitterConfigs: [{ ...stored.submitterConfigs[0], modelId: '' }, stored.submitterConfigs[1]],
    };
    const copied = rolesFromAutonomousSettings(noFirstModel);
    expect(copied.translator).toBeNull();
    expect(copied.validator.modelId).toBe('openrouter/validator');
  });

  test('hosted mode keeps only OpenRouter routes', () => {
    const lmFirst = { ...stored, submitterConfigs: [stored.submitterConfigs[1], stored.submitterConfigs[0]] };
    const hosted = rolesFromAutonomousSettings(lmFirst, { lmStudioEnabled: false });
    expect(hosted.translator).toBeNull();
    expect(hosted.validator.lmStudioFallbackId).toBeNull();
    expect(rolesFromAutonomousSettings(stored, { lmStudioEnabled: false }).translator.modelId).toBe('openrouter/one');
  });

  test('returns null when nothing is configured and applies only what was copied', () => {
    expect(rolesFromAutonomousSettings({ submitterConfigs: [], localConfig: { validator_model: '' } })).toBeNull();
    expect(rolesFromAutonomousSettings(null)).toBeNull();
    const draft = { translator: CAMEL_ROLE, validator: { ...CAMEL_ROLE, modelId: 'keep' } };
    const onlyTranslator = applyCopiedRoles(draft, { translator: { ...CAMEL_ROLE, modelId: 'x' }, validator: null });
    expect(onlyTranslator.translator.modelId).toBe('x');
    expect(onlyTranslator.validator.modelId).toBe('keep');
    const onlyValidator = applyCopiedRoles(draft, { translator: null, validator: { ...CAMEL_ROLE, modelId: 'y' } });
    expect(onlyValidator.translator).toBe(draft.translator);
    expect(onlyValidator.validator.modelId).toBe('y');
    expect(applyCopiedRoles(draft, null)).toBe(draft);
  });
});

describe('client validation', () => {
  const valid = { translator: CAMEL_ROLE, validator: CAMEL_ROLE };

  test('accepts a complete draft', () => {
    expect(validateTranslatorDraft(valid)).toEqual([]);
  });

  test('requires a model and positive limits with max output below context', () => {
    expect(validateTranslatorDraft({
      translator: { ...CAMEL_ROLE, modelId: '', contextWindow: '', maxOutputTokens: 0 },
      validator: { ...CAMEL_ROLE, contextWindow: 8000, maxOutputTokens: 8000 },
    })).toEqual([
      'Select a model for Translator Submitter.',
      'Enter a context window greater than 0 for Translator Submitter.',
      'Enter max output tokens greater than 0 for Translator Submitter.',
      'Translator Validator: max output tokens must be smaller than the context window.',
    ]);
    expect(validateTranslatorDraft({})).toContain('Select a model for Translator Validator.');
  });

  test('hosted mode rejects non-OpenRouter roles', () => {
    const errors = validateTranslatorDraft(
      { translator: { ...CAMEL_ROLE, provider: 'lm_studio' }, validator: CAMEL_ROLE },
      { lmStudioEnabled: false },
    );
    expect(errors).toContain(`Translator Submitter: ${GENERIC_MODE_PROVIDER_MESSAGE}`);
  });
});

test('describeTranslationError strips the HTTP prefix from validation errors only', () => {
  expect(describeTranslationError({
    status: 422,
    message: 'HTTP 422 Unprocessable Entity: Hosted mode supports OpenRouter translator models only.',
  })).toBe('Hosted mode supports OpenRouter translator models only.');
  expect(describeTranslationError({ status: 500, message: 'HTTP 500: boom' })).toBe('HTTP 500: boom');
  expect(describeTranslationError(null, 'Fallback.')).toBe('Fallback.');
});
