import {
  LEANOJ_RECOMMENDED_PROFILES,
  normalizeLeanOJSettings,
  settingsToLeanOJRequest,
} from './leanojProfiles';

test('recommended LeanOJ profiles use MiniMax M3 instead of Kimi K2.6', () => {
  const recommendedRoles = Object.values(LEANOJ_RECOMMENDED_PROFILES).flatMap((profile) => [
    ...(profile.submitters || []),
    ...Object.values(profile.roles || {}),
  ]);
  const minimaxRoles = recommendedRoles.filter((role) => role.modelId === 'minimax/minimax-m3');

  expect(recommendedRoles.map((role) => role.modelId)).not.toContain('moonshotai/kimi-k2.6');
  expect(minimaxRoles.length).toBeGreaterThan(0);
  minimaxRoles.forEach((role) => {
    expect(role.provider).toBe('openrouter');
    expect(role.contextWindow).toBe(1048576);
    expect(role.maxOutputTokens).toBe(131072);
  });
});

test('migrates missing LeanOJ Assistant role from topic validator settings', () => {
  const settings = normalizeLeanOJSettings({
    roles: {
      topic_validator: {
        provider: 'openrouter',
        modelId: 'openrouter/topic-validator',
        openrouterProvider: 'ValidatorHost',
        openrouterReasoningEffort: 'xhigh',
        lmStudioFallbackId: 'validator-fallback',
        contextWindow: 7777,
        maxOutputTokens: 777,
        superchargeEnabled: true,
      },
    },
  });

  expect(settings.roles.assistant.provider).toBe('openrouter');
  expect(settings.roles.assistant.modelId).toBe('openrouter/topic-validator');
  expect(settings.roles.assistant.openrouterProvider).toBe('ValidatorHost');
  expect(settings.roles.assistant.openrouterReasoningEffort).toBe('xhigh');
  expect(settings.roles.assistant.lmStudioFallbackId).toBe('validator-fallback');
  expect(settings.roles.assistant.contextWindow).toBe(7777);
  expect(settings.roles.assistant.maxOutputTokens).toBe(777);
  expect(settings.roles.assistant.superchargeEnabled).toBe(true);
});

test('preserves explicit LeanOJ Assistant role through request conversion', () => {
  const request = settingsToLeanOJRequest(
    {
      submitterConfigs: [
        {
          provider: 'openrouter',
          modelId: 'openrouter/topic-generator',
          contextWindow: 4096,
          maxOutputTokens: 512,
        },
      ],
      roles: {
        topic_validator: {
          provider: 'openrouter',
          modelId: 'openrouter/topic-validator',
          contextWindow: 7777,
          maxOutputTokens: 777,
        },
        brainstorm_validator: {
          provider: 'openrouter',
          modelId: 'openrouter/brainstorm-validator',
          contextWindow: 4096,
          maxOutputTokens: 512,
        },
        final_solver: {
          provider: 'openrouter',
          modelId: 'openrouter/final-solver',
          contextWindow: 8192,
          maxOutputTokens: 1024,
        },
        assistant: {
          provider: 'openrouter',
          modelId: 'openrouter/assistant',
          openrouterProvider: 'AssistantHost',
          openrouterReasoningEffort: 'medium',
          contextWindow: 9999,
          maxOutputTokens: 999,
          superchargeEnabled: true,
        },
      },
    },
    'Solve the template.',
    'theorem target : True := by\n  trivial'
  );

  expect(request.assistant.provider).toBe('openrouter');
  expect(request.assistant.model_id).toBe('openrouter/assistant');
  expect(request.assistant.openrouter_provider).toBe('AssistantHost');
  expect(request.assistant.openrouter_reasoning_effort).toBe('medium');
  expect(request.assistant.context_window).toBe(9999);
  expect(request.assistant.max_output_tokens).toBe(999);
  expect(request.assistant.supercharge_enabled).toBe(true);
});

test('disabled LeanOJ Assistant does not require valid Assistant role limits', () => {
  const request = settingsToLeanOJRequest(
    {
      submitterConfigs: [
        {
          provider: 'openrouter',
          modelId: 'openrouter/topic-generator',
          contextWindow: 4096,
          maxOutputTokens: 512,
        },
      ],
      roles: {
        topic_validator: {
          provider: 'openrouter',
          modelId: 'openrouter/topic-validator',
          contextWindow: 7777,
          maxOutputTokens: 777,
        },
        brainstorm_validator: {
          provider: 'openrouter',
          modelId: 'openrouter/brainstorm-validator',
          contextWindow: 4096,
          maxOutputTokens: 512,
        },
        final_solver: {
          provider: 'openrouter',
          modelId: 'openrouter/final-solver',
          contextWindow: 8192,
          maxOutputTokens: 1024,
        },
        assistant: {
          provider: 'openrouter',
          modelId: '',
          contextWindow: '',
          maxOutputTokens: '',
        },
      },
    },
    'Solve the template.',
    'theorem target : True := by\n  trivial',
    { assistantEnabled: false }
  );

  expect(request.assistant.model_id).toBe('');
  expect(request.assistant.context_window).toBe(7777);
  expect(request.assistant.max_output_tokens).toBe(777);
});
