import {
  chooseCloudAccessProvider,
  CLOUD_ACCESS_PROVIDERS,
  getAvailableCloudAccessProviders,
  isApiKeyCloudProvider,
  isCloudProviderAvailable,
  KIMI_CODE_PROVIDER,
  SAKANA_FUGU_PROVIDER,
} from './oauthProviders';
import { computeCloudProviderAutoSettings, normalizeOpenRouterReasoningEffort } from './openRouterSelection';

test('every registry entry has unique identity and the fields shared UI relies on', () => {
  for (const field of ['id', 'apiSlug', 'capabilityKey', 'featureFlag']) {
    const values = CLOUD_ACCESS_PROVIDERS.map((provider) => provider[field]);
    expect(new Set(values).size).toBe(values.length);
  }
  for (const provider of CLOUD_ACCESS_PROVIDERS) {
    expect(['oauth_pkce', 'api_key']).toContain(provider.authKind);
    expect(provider.label && provider.shortLabel && provider.loginLabel && provider.noModelsMessage).toBeTruthy();
    expect(['codex', 'options']).toContain(provider.reasoning.control);
    if (provider.authKind === 'api_key') {
      expect(provider.apiKey.inputId).toMatch(/-api-key$/);
      expect(provider.apiKey.consoleUrl).toMatch(/^https:\/\//);
    }
  }
});

test('availability follows hosted mode and explicit capability flags', () => {
  expect(isCloudProviderAvailable({ genericMode: false }, KIMI_CODE_PROVIDER)).toBe(true);
  expect(isCloudProviderAvailable({ genericMode: false, kimiCodeAvailable: false }, KIMI_CODE_PROVIDER)).toBe(false);
  expect(isCloudProviderAvailable({ genericMode: true, kimiCodeAvailable: true }, KIMI_CODE_PROVIDER)).toBe(false);
  expect(getAvailableCloudAccessProviders({ genericMode: true })).toEqual([]);
  expect(isApiKeyCloudProvider(SAKANA_FUGU_PROVIDER)).toBe(true);
  expect(isApiKeyCloudProvider('openai_codex_oauth')).toBe(false);
});

test('keeps a configured current provider and otherwise picks the only configured one', () => {
  const status = { [KIMI_CODE_PROVIDER]: { configured: true } };
  expect(chooseCloudAccessProvider(status, KIMI_CODE_PROVIDER)).toBe(KIMI_CODE_PROVIDER);
  expect(chooseCloudAccessProvider(status, SAKANA_FUGU_PROVIDER)).toBe(KIMI_CODE_PROVIDER);
});

test('normalizes reasoning effort to each provider\'s offered choices', () => {
  expect(normalizeOpenRouterReasoningEffort('xhigh', KIMI_CODE_PROVIDER)).toBe('max');
  expect(normalizeOpenRouterReasoningEffort('medium', KIMI_CODE_PROVIDER)).toBe('high');
  expect(normalizeOpenRouterReasoningEffort('none', KIMI_CODE_PROVIDER)).toBe('none');
  expect(normalizeOpenRouterReasoningEffort('bogus', KIMI_CODE_PROVIDER)).toBe('auto');
  // Sakana offers only auto/xhigh/high; unsupported values no longer pass through.
  expect(normalizeOpenRouterReasoningEffort('medium', SAKANA_FUGU_PROVIDER)).toBe('auto');
  expect(normalizeOpenRouterReasoningEffort('max', SAKANA_FUGU_PROVIDER)).toBe('xhigh');
  expect(normalizeOpenRouterReasoningEffort('max', 'openai_codex_oauth')).toBe('max');
  expect(normalizeOpenRouterReasoningEffort('max', 'openrouter')).toBe('auto');
});

test('auto-fill uses only known metadata and never invents an output cap', () => {
  const autoSettings = computeCloudProviderAutoSettings(KIMI_CODE_PROVIDER, { id: 'k3', context_length: 1048576 });
  expect(autoSettings.contextWindow).toBe(1048576);
  expect(autoSettings.outputCapKnown).toBe(false);
});
