import {
  AMAZON_BEDROCK_PROVIDER,
  chooseCloudAccessProvider,
  CLAUDE_API_PROVIDER,
  CLOUD_ACCESS_PROVIDERS,
  CLOUD_SUBSCRIPTION_PROVIDERS,
  describeLocalFallback,
  getAvailableCloudAccessProviders,
  getConfiguredCloudAccessProviders,
  isApiKeyCloudProvider,
  isCloudAccessProvider,
  isCloudProviderAvailable,
  isCloudSubscriptionProvider,
  isLocalServerProvider,
  KIMI_CODE_PROVIDER,
  LOCAL_SERVER_PROVIDERS,
  localFallbackValue,
  SAKANA_FUGU_PROVIDER,
  VLLM_PROVIDER,
} from './oauthProviders';
import { computeCloudProviderAutoSettings, normalizeOpenRouterReasoningEffort } from './openRouterSelection';

test('every registry entry has unique identity and the fields shared UI relies on', () => {
  for (const field of ['id', 'apiSlug', 'capabilityKey', 'featureFlag']) {
    const values = CLOUD_ACCESS_PROVIDERS.map((provider) => provider[field]);
    expect(new Set(values).size).toBe(values.length);
  }
  for (const provider of CLOUD_ACCESS_PROVIDERS) {
    expect(['oauth_pkce', 'api_key', 'local_server']).toContain(provider.authKind);
    expect(provider.label && provider.shortLabel && provider.loginLabel && provider.noModelsMessage).toBeTruthy();
    if (provider.authKind === 'local_server') {
      // A local server takes no reasoning-effort setting and has no API-key section.
      expect(provider.reasoning).toBeUndefined();
      expect(provider.apiKey).toBeUndefined();
      expect(provider.unavailableTitle).toMatch(/Inference Connectivity/);
      continue;
    }
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

test('vLLM is a local server role provider kept out of the cloud lists', () => {
  expect(LOCAL_SERVER_PROVIDERS.map((provider) => provider.id)).toEqual([VLLM_PROVIDER]);
  expect(CLOUD_SUBSCRIPTION_PROVIDERS.some((provider) => provider.id === VLLM_PROVIDER)).toBe(false);
  expect(isCloudAccessProvider(VLLM_PROVIDER)).toBe(true);
  expect(isLocalServerProvider(VLLM_PROVIDER)).toBe(true);
  expect(isCloudSubscriptionProvider(VLLM_PROVIDER)).toBe(false);
  expect(isCloudSubscriptionProvider(KIMI_CODE_PROVIDER)).toBe(true);
  expect(isApiKeyCloudProvider(VLLM_PROVIDER)).toBe(false);

  expect(isCloudProviderAvailable({ genericMode: false }, VLLM_PROVIDER)).toBe(true);
  expect(isCloudProviderAvailable({ genericMode: true, vllmAvailable: true }, VLLM_PROVIDER)).toBe(false);
  const status = { [VLLM_PROVIDER]: { configured: true }, [KIMI_CODE_PROVIDER]: { configured: true } };
  expect(getConfiguredCloudAccessProviders(status).map((provider) => provider.id)).toEqual([KIMI_CODE_PROVIDER]);
  expect(getAvailableCloudAccessProviders({ genericMode: false }).some((provider) => provider.id === VLLM_PROVIDER))
    .toBe(false);
  // The "Cloud" button never lands on vLLM, even when it is the only configured entry.
  expect(chooseCloudAccessProvider({ [VLLM_PROVIDER]: { configured: true } }, VLLM_PROVIDER)).not.toBe(VLLM_PROVIDER);
});

test('a local fallback value names LM Studio by default and tags a local server model', () => {
  expect(localFallbackValue('lm_studio', 'model:2')).toBe('model:2');
  expect(localFallbackValue(VLLM_PROVIDER, 'Qwen/Qwen2.5')).toBe('vllm::Qwen/Qwen2.5');
  expect(describeLocalFallback('vllm::Qwen/Qwen2.5')).toEqual({
    providerId: VLLM_PROVIDER, providerLabel: 'vLLM', modelId: 'Qwen/Qwen2.5', text: 'vLLM: Qwen/Qwen2.5',
  });
  // LM Studio ids keep their own colons, and an empty tag is not a vLLM model.
  expect(describeLocalFallback('model:2').providerId).toBe('lm_studio');
  expect(describeLocalFallback('vllm::')).toMatchObject({ providerId: 'lm_studio', modelId: 'vllm::' });
  expect(describeLocalFallback(null)).toMatchObject({ providerId: 'lm_studio', modelId: '', text: '' });
});

test('vLLM auto-fill uses the detected context window and a 20% output cap', () => {
  const detected = computeCloudProviderAutoSettings(VLLM_PROVIDER, { id: 'Qwen/Qwen2.5', context_length: 32768 });
  expect(detected.contextWindowKnown).toBe(true);
  expect(detected.contextWindow).toBe(32768);
  expect(detected.outputCapKnown).toBe(true);
  expect(detected.maxOutputTokens).toBe(6553);
  expect(detected.outputCapSource).toBe('context-window-share');
  expect(detected.warnings).toEqual([]);

  // A server that does not report a context window leaves both settings untouched.
  const unknown = computeCloudProviderAutoSettings(VLLM_PROVIDER, { id: 'mystery' });
  expect(unknown.contextWindowKnown).toBe(false);
  expect(unknown.outputCapKnown).toBe(false);
  expect(normalizeOpenRouterReasoningEffort('high', VLLM_PROVIDER)).toBe('high');
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

test('Claude and Bedrock offer Claude effort levels and map unsupported choices downward', () => {
  for (const provider of [CLAUDE_API_PROVIDER, AMAZON_BEDROCK_PROVIDER]) {
    expect(isApiKeyCloudProvider(provider)).toBe(true);
    expect(isCloudProviderAvailable({ genericMode: false }, provider)).toBe(true);
    expect(isCloudProviderAvailable({ genericMode: true }, provider)).toBe(false);
    expect(normalizeOpenRouterReasoningEffort('max', provider)).toBe('max');
    expect(normalizeOpenRouterReasoningEffort('xhigh', provider)).toBe('xhigh');
    expect(normalizeOpenRouterReasoningEffort('medium', provider)).toBe('medium');
    // Thinking cannot be switched off on several current Claude models.
    expect(normalizeOpenRouterReasoningEffort('none', provider)).toBe('low');
    expect(normalizeOpenRouterReasoningEffort('minimal', provider)).toBe('low');
    expect(normalizeOpenRouterReasoningEffort('bogus', provider)).toBe('auto');
  }
  const autoSettings = computeCloudProviderAutoSettings(CLAUDE_API_PROVIDER, {
    id: 'claude-opus-5-5', context_length: 1000000, max_output_tokens: 128000,
  });
  expect(autoSettings.contextWindow).toBe(1000000);
  expect(autoSettings.maxOutputTokens).toBe(128000);
  // A Bedrock model with no published limits keeps the user's current settings.
  const unknown = computeCloudProviderAutoSettings(AMAZON_BEDROCK_PROVIDER, { id: 'openai.gpt-oss-120b' });
  expect(unknown.contextWindowKnown).toBe(false);
  expect(unknown.outputCapKnown).toBe(false);
});

test('auto-fill uses only known metadata and never invents an output cap', () => {
  const autoSettings = computeCloudProviderAutoSettings(KIMI_CODE_PROVIDER, { id: 'k3', context_length: 1048576 });
  expect(autoSettings.contextWindow).toBe(1048576);
  expect(autoSettings.outputCapKnown).toBe(false);
});
