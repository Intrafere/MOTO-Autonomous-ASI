import { persistAutonomousSettings, settingsToAutonomousConfig } from './autonomousProfiles';
import { normalizeLeanOJSettings, settingsToLeanOJRequest, LEANOJ_ROLE_KEYS } from './leanojProfiles';

const role = { provider: 'openai_codex_oauth', modelId: 'catalog-model', openrouterReasoningEffort: 'max', contextWindow: 400000, maxOutputTokens: 32000 };

test('Autonomous persisted roles, competition, and request preserve Codex max', () => {
  localStorage.clear();
  const localConfig = {};
  for (const prefix of ['validator', 'assistant', 'writer', 'high_param']) {
    Object.assign(localConfig, { [`${prefix}_provider`]: role.provider, [`${prefix}_model`]: role.modelId, [`${prefix}_openrouter_reasoning_effort`]: 'max', [`${prefix}_context_window`]: role.contextWindow, [`${prefix}_max_tokens`]: role.maxOutputTokens });
  }
  localConfig.proof_competition = { enabled: true, secondaries: [{ provider: role.provider, model_id: role.modelId, openrouter_reasoning_effort: 'max', context_window: role.contextWindow, max_tokens: role.maxOutputTokens }] };
  const saved = persistAutonomousSettings({ numSubmitters: 1, submitterConfigs: [role], localConfig });
  const request = settingsToAutonomousConfig(saved);
  expect(request.submitter_configs[0].openrouter_reasoning_effort).toBe('max');
  for (const prefix of ['validator', 'assistant', 'writer', 'high_param', 'critique_submitter']) expect(request[`${prefix}_openrouter_reasoning_effort`]).toBe('max');
  expect(request.proof_competition.secondaries[0].openrouter_reasoning_effort).toBe('max');
});

test('LeanOJ all role normalization and requests preserve Codex max', () => {
  const settings = normalizeLeanOJSettings({ numSubmitters: 1, submitterConfigs: [role], roles: Object.fromEntries(LEANOJ_ROLE_KEYS.map(key => [key, role])) });
  const request = settingsToLeanOJRequest(settings, 'Goal', 'template');
  for (const key of [...LEANOJ_ROLE_KEYS, 'path_decider']) expect(request[key].openrouter_reasoning_effort).toBe('max');
  expect(request.brainstorm_submitters[0].openrouter_reasoning_effort).toBe('max');
});
