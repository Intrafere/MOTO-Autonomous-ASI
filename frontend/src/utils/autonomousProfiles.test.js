import {
  OPENAI_CODEX_STARTUP_CHOICE,
  XAI_GROK_STARTUP_CHOICE,
  applyCloudAccessStartupDefaults,
  persistAutonomousSettings,
  RECOMMENDED_PROFILE_KEYS,
  RECOMMENDED_PROFILES,
  settingsToAutonomousConfig,
} from './autonomousProfiles';

beforeEach(() => {
  localStorage.clear();
});

test('recommended profiles do not carry a standalone critique role', () => {
  for (const profileKey of RECOMMENDED_PROFILE_KEYS) {
    expect(RECOMMENDED_PROFILES[profileKey]).toBeTruthy();
    expect(RECOMMENDED_PROFILES[profileKey].highParam).toBeTruthy();
    expect(RECOMMENDED_PROFILES[profileKey].critique).toBeUndefined();
  }
});

test('recommended autonomous profiles use MiniMax M3 instead of Kimi K2.6', () => {
  const recommendedRoles = RECOMMENDED_PROFILE_KEYS.flatMap((profileKey) => {
    const profile = RECOMMENDED_PROFILES[profileKey];
    return [
      ...(profile.submitters || []),
      profile.validator,
      profile.assistant,
      profile.writer,
      profile.highParam,
    ].filter(Boolean);
  });
  const minimaxRoles = recommendedRoles.filter((role) => role.modelId === 'minimax/minimax-m3');

  expect(recommendedRoles.map((role) => role.modelId)).not.toContain('moonshotai/kimi-k2.6');
  expect(minimaxRoles.length).toBeGreaterThan(0);
  minimaxRoles.forEach((role) => {
    expect(role.provider).toBe('openrouter');
    expect(role.contextWindow).toBe(1048576);
    expect(role.maxOutputTokens).toBe(131072);
  });
});

test('legacy user profile with only critique role migrates it to Rigor & Proofs', async () => {
  const { config } = await import('./autonomousProfiles').then(({ applyAutonomousProfileSelection }) => (
    applyAutonomousProfileSelection('user_legacy_critique_only', {
      user_legacy_critique_only: {
        name: 'Legacy Critique Only',
        numSubmitters: 1,
        submitters: [
          {
            modelId: 'submitter-model',
            provider: 'openrouter',
            contextWindow: 4096,
            maxOutputTokens: 512,
          },
        ],
        validator: {
          modelId: 'validator-model',
          provider: 'openrouter',
          contextWindow: 4096,
          maxOutputTokens: 512,
        },
        writer: {
          modelId: 'writer-model',
          provider: 'openrouter',
          contextWindow: 8192,
          maxOutputTokens: 1024,
        },
        critique: {
          modelId: 'legacy-rigor-model',
          provider: 'openrouter',
          openrouterProvider: 'LegacyHost',
          contextWindow: 16384,
          maxOutputTokens: 2048,
        },
      },
    })
  ));

  expect(config.high_param_model).toBe('legacy-rigor-model');
  expect(config.high_param_openrouter_provider).toBe('LegacyHost');
  expect(config.high_param_context_window).toBe(16384);
  expect(config.high_param_max_tokens).toBe(2048);
  expect(config.critique_submitter_model).toBe('legacy-rigor-model');
});

test('normalizes deprecated autonomous critique fields from Rigor & Proofs settings', () => {
  const settings = persistAutonomousSettings({
    localConfig: {
      validator_provider: 'openrouter',
      validator_model: 'validator-model',
      validator_context_window: 4096,
      validator_max_tokens: 512,
      writer_provider: 'openrouter',
      writer_model: 'writer-model',
      writer_context_window: 4096,
      writer_max_tokens: 512,
      high_param_provider: 'openrouter',
      high_param_model: 'rigor-model',
      high_param_openrouter_provider: 'RigorHost',
      high_param_openrouter_reasoning_effort: 'xhigh',
      high_param_lm_studio_fallback: 'rigor-fallback',
      high_param_context_window: 8192,
      high_param_max_tokens: 1024,
      high_param_supercharge_enabled: true,
      critique_submitter_provider: 'openai_codex_oauth',
      critique_submitter_model: 'stale-critique-model',
      critique_submitter_openrouter_provider: 'StaleHost',
      critique_submitter_openrouter_reasoning_effort: 'low',
      critique_submitter_lm_studio_fallback: 'stale-fallback',
      critique_submitter_context_window: 1234,
      critique_submitter_max_tokens: 123,
      critique_submitter_supercharge_enabled: false,
    },
  });

  expect(settings.localConfig.critique_submitter_provider).toBe('openrouter');
  expect(settings.localConfig.critique_submitter_model).toBe('rigor-model');
  expect(settings.localConfig.critique_submitter_openrouter_provider).toBe('RigorHost');
  expect(settings.localConfig.critique_submitter_openrouter_reasoning_effort).toBe('xhigh');
  expect(settings.localConfig.critique_submitter_lm_studio_fallback).toBe('rigor-fallback');
  expect(settings.localConfig.critique_submitter_context_window).toBe(8192);
  expect(settings.localConfig.critique_submitter_max_tokens).toBe(1024);
  expect(settings.localConfig.critique_submitter_supercharge_enabled).toBe(true);
});

test('autonomous start config sends deprecated critique fields mirrored from Rigor & Proofs', () => {
  const config = settingsToAutonomousConfig({
    submitterConfigs: [
      {
        submitterId: 1,
        provider: 'openrouter',
        modelId: 'submitter-model',
        contextWindow: 4096,
        maxOutputTokens: 512,
      },
    ],
    localConfig: {
      validator_provider: 'openrouter',
      validator_model: 'validator-model',
      validator_context_window: 4096,
      validator_max_tokens: 512,
      writer_provider: 'openrouter',
      writer_model: 'writer-model',
      writer_context_window: 4096,
      writer_max_tokens: 512,
      high_param_provider: 'openrouter',
      high_param_model: 'rigor-model',
      high_param_openrouter_provider: 'RigorHost',
      high_param_context_window: 8192,
      high_param_max_tokens: 1024,
      high_param_supercharge_enabled: true,
      critique_submitter_provider: 'lm_studio',
      critique_submitter_model: 'stale-critique-model',
      critique_submitter_context_window: 999,
      critique_submitter_max_tokens: 99,
    },
  });

  expect(config.high_param_model).toBe('rigor-model');
  expect(config.critique_submitter_provider).toBe(config.high_param_provider);
  expect(config.critique_submitter_model).toBe(config.high_param_model);
  expect(config.critique_submitter_openrouter_provider).toBe(config.high_param_openrouter_provider);
  expect(config.critique_submitter_context_window).toBe(config.high_param_context_window);
  expect(config.critique_submitter_max_tokens).toBe(config.high_param_max_tokens);
  expect(config.critique_submitter_supercharge_enabled).toBe(true);
});

test('migrates legacy autonomous writer local config into current writer fields', () => {
  const legacyPrefix = ['high', 'context'].join('_');
  const config = settingsToAutonomousConfig({
    submitterConfigs: [
      {
        submitterId: 1,
        provider: 'openrouter',
        modelId: 'submitter-model',
        contextWindow: 4096,
        maxOutputTokens: 512,
      },
    ],
    localConfig: {
      validator_provider: 'openrouter',
      validator_model: 'validator-model',
      validator_context_window: 4096,
      validator_max_tokens: 512,
      writer_provider: 'openrouter',
      writer_model: '',
      writer_context_window: 0,
      writer_max_tokens: 0,
      [`${legacyPrefix}_provider`]: 'openrouter',
      [`${legacyPrefix}_model`]: 'legacy-writer-model',
      [`${legacyPrefix}_context_window`]: 12345,
      [`${legacyPrefix}_max_tokens`]: 1234,
      high_param_provider: 'openrouter',
      high_param_model: 'rigor-model',
      high_param_context_window: 8192,
      high_param_max_tokens: 1024,
    },
  });

  expect(config.writer_provider).toBe('openrouter');
  expect(config.writer_model).toBe('legacy-writer-model');
  expect(config.writer_context_window).toBe(12345);
  expect(config.writer_max_tokens).toBe(1234);
});

test('omitted autonomous Assistant supercharge follows Validator unless Assistant is explicit', () => {
  const inherited = persistAutonomousSettings({
    localConfig: {
      validator_model: 'validator-model',
      validator_supercharge_enabled: true,
      high_param_model: 'rigor-model',
    },
  });

  expect(inherited.localConfig.assistant_model).toBe('validator-model');
  expect(inherited.localConfig.assistant_supercharge_enabled).toBe(true);

  const explicit = persistAutonomousSettings({
    localConfig: {
      validator_model: 'validator-model',
      validator_supercharge_enabled: true,
      assistant_model: 'assistant-model',
      assistant_supercharge_enabled: false,
      high_param_model: 'rigor-model',
    },
  });

  expect(explicit.localConfig.assistant_model).toBe('assistant-model');
  expect(explicit.localConfig.assistant_supercharge_enabled).toBe(false);
});

test('persisted autonomous settings store only public non-secret configuration', () => {
  persistAutonomousSettings({
    numSubmitters: 1,
    submitterConfigs: [
      {
        submitterId: 1,
        provider: 'openrouter',
        modelId: 'public-model',
        contextWindow: 4096,
        maxOutputTokens: 512,
        apiKey: 'should-not-persist',
      },
    ],
    localConfig: {
      validator_provider: 'openrouter',
      validator_model: 'validator-model',
      validator_context_window: 4096,
      validator_max_tokens: 512,
      access_token: 'should-not-persist',
      openrouter_api_key: 'should-not-persist',
    },
    modelProviders: {
      'public-model': {
        provider: 'public-host',
        api_key: 'should-not-persist',
      },
    },
  });

  const rawStoredSettings = localStorage.getItem('autonomous_research_settings');
  expect(rawStoredSettings).toContain('public-model');
  expect(rawStoredSettings).not.toContain('should-not-persist');
  expect(rawStoredSettings).not.toContain('apiKey');
  expect(rawStoredSettings).not.toContain('access_token');
  expect(rawStoredSettings).not.toContain('modelProviders');
});

test('partial autonomous settings saves preserve existing output and creativity toggles', () => {
  persistAutonomousSettings({
    allowMathematicalProofs: false,
    allowResearchPapers: true,
    creativityEmphasisBoostEnabled: true,
  });

  const updated = persistAutonomousSettings({
    localConfig: {
      validator_model: 'validator-model',
      high_param_model: 'rigor-model',
    },
  });

  expect(updated.allowMathematicalProofs).toBe(false);
  expect(updated.allowResearchPapers).toBe(true);
  expect(updated.creativityEmphasisBoostEnabled).toBe(true);

  const rawStoredSettings = JSON.parse(localStorage.getItem('autonomous_research_settings'));
  expect(rawStoredSettings.allowMathematicalProofs).toBe(false);
  expect(rawStoredSettings.allowResearchPapers).toBe(true);
  expect(rawStoredSettings.creativityEmphasisBoostEnabled).toBe(true);
});

test('OAuth startup defaults use known public model metadata from availability only', () => {
  const codex = applyCloudAccessStartupDefaults(OPENAI_CODEX_STARTUP_CHOICE, [
    { id: 'gpt-5.4', accountScopedField: 'must-not-persist' },
  ]);
  expect(codex.modelId).toBe('gpt-5.4');
  expect(codex.config.validator_model).toBe('gpt-5.4');

  const xai = applyCloudAccessStartupDefaults(XAI_GROK_STARTUP_CHOICE, [
    { id: 'grok-4', accountScopedField: 'must-not-persist' },
  ]);
  expect(xai.modelId).toBe('grok-4');
  expect(xai.config.validator_model).toBe('grok-4');

  const rawStoredSettings = localStorage.getItem('autonomous_research_settings');
  expect(rawStoredSettings).toContain('grok-4');
  expect(rawStoredSettings).not.toContain('accountScopedField');
  expect(rawStoredSettings).not.toContain('must-not-persist');
});
