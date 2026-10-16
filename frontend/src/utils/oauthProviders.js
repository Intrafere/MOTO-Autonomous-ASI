// Registry of desktop cloud subscription providers (OAuth logins and subscription
// API keys). Every settings panel, the OpenRouter/OAuth modal, and the API helpers
// read provider details from here, so adding a provider is one entry plus its
// backend profile. It mirrors backend/shared/subscription_providers/metadata.py.

export const OPENAI_CODEX_PROVIDER = 'openai_codex_oauth';
export const XAI_GROK_PROVIDER = 'xai_grok_oauth';
export const SAKANA_FUGU_PROVIDER = 'sakana_fugu';
export const KIMI_CODE_PROVIDER = 'kimi_code';

export const SAKANA_FUGU_REASONING_EFFORT_OPTIONS = [
  { value: 'auto', label: 'Auto (xhigh)' },
  { value: 'xhigh', label: 'xhigh (maximum)' },
  { value: 'high', label: 'high' },
];
export const XAI_GROK_REASONING_EFFORT_OPTIONS = [
  { value: 'auto', label: 'Auto (high)' },
  { value: 'high', label: 'high' },
  { value: 'medium', label: 'medium' },
  { value: 'low', label: 'low' },
  { value: 'none', label: 'none / disabled' },
];
export const KIMI_CODE_REASONING_EFFORT_OPTIONS = [
  { value: 'auto', label: 'Auto (model default)' },
  { value: 'max', label: 'max' },
  { value: 'high', label: 'high' },
  { value: 'low', label: 'low' },
  { value: 'none', label: 'none / thinking off' },
];

export const CLOUD_ACCESS_PROVIDERS = [
  {
    id: OPENAI_CODEX_PROVIDER,
    label: 'OpenAI Codex',
    shortLabel: 'Codex',
    authKind: 'oauth_pkce',
    apiSlug: 'openai-codex',
    capabilityKey: 'openAICodexOauthAvailable',
    featureFlag: 'openai_codex_oauth_available',
    loginLabel: 'OpenAI Codex OAuth',
    unavailableTitle: 'Set OpenAI Codex login in OpenRouter/OAuth first',
    modelErrorLabel: 'OpenAI Codex OAuth',
    noModelsMessage: 'OpenAI Codex OAuth is connected, but no Codex models were returned. Reconnect OAuth or check account access.',
    reasoning: { control: 'codex' },
    autoSettings: 'codex',
  },
  {
    id: XAI_GROK_PROVIDER,
    label: 'xAI Grok',
    shortLabel: 'Grok',
    authKind: 'oauth_pkce',
    apiSlug: 'xai-grok',
    capabilityKey: 'xaiGrokOauthAvailable',
    featureFlag: 'xai_grok_oauth_available',
    loginLabel: 'xAI Grok OAuth',
    unavailableTitle: 'Set xAI Grok login in OpenRouter/OAuth first',
    modelErrorLabel: 'xAI Grok OAuth',
    noModelsMessage: 'xAI Grok OAuth is connected, but no Grok models were returned. Reconnect OAuth or check account access.',
    reasoning: {
      control: 'options',
      ariaLabel: 'xAI Reasoning Effort',
      options: XAI_GROK_REASONING_EFFORT_OPTIONS,
      aliases: { xhigh: 'high', max: 'high', maximum: 'high', highest: 'high', minimal: 'low' },
      hint: 'Auto sends high reasoning effort; disabled omits the xAI reasoning setting.',
    },
    autoSettings: 'xai',
  },
  {
    id: SAKANA_FUGU_PROVIDER,
    label: 'Sakana Fugu',
    shortLabel: 'Fugu',
    authKind: 'api_key',
    apiSlug: 'sakana-fugu',
    capabilityKey: 'sakanaFuguAvailable',
    featureFlag: 'sakana_fugu_available',
    loginLabel: 'Sakana Fugu API Key',
    unavailableTitle: 'Set Sakana Fugu API key in OpenRouter/OAuth first',
    modelErrorLabel: 'Sakana Fugu API',
    noModelsMessage: 'Sakana Fugu API key is saved, but no Fugu models were returned. Check your Sakana subscription access.',
    reasoning: {
      control: 'options',
      ariaLabel: 'Sakana Fugu Reasoning Effort',
      options: SAKANA_FUGU_REASONING_EFFORT_OPTIONS,
      aliases: { max: 'xhigh', maximum: 'xhigh', highest: 'xhigh' },
      hint: 'Sakana Fugu supports high and xhigh reasoning effort only; auto maps to xhigh.',
    },
    autoSettings: 'generic',
    apiKey: {
      inputId: 'sakana-fugu-api-key',
      inputName: 'moto-sakana-fugu-api-key',
      consoleUrl: 'https://console.sakana.ai',
      consoleLabel: 'console.sakana.ai',
      description: 'Save your Sakana Fugu subscription API key to use Fugu or Fugu Ultra directly as a MOTO role provider.',
      supportsUsage: false,
    },
  },
  {
    id: KIMI_CODE_PROVIDER,
    label: 'Kimi Code',
    shortLabel: 'Kimi',
    authKind: 'api_key',
    apiSlug: 'kimi-code',
    capabilityKey: 'kimiCodeAvailable',
    featureFlag: 'kimi_code_available',
    loginLabel: 'Kimi Code API Key',
    unavailableTitle: 'Set Kimi Code API key in OpenRouter/OAuth first',
    modelErrorLabel: 'Kimi Code API',
    noModelsMessage: 'Kimi Code API key is saved, but no Kimi Code models were returned. Check your Kimi Code membership.',
    reasoning: {
      control: 'options',
      ariaLabel: 'Kimi Code Reasoning Effort',
      options: KIMI_CODE_REASONING_EFFORT_OPTIONS,
      aliases: { xhigh: 'max', maximum: 'max', highest: 'max', medium: 'high', minimal: 'low' },
      hint: 'Auto uses the model default (high for K3, max for Kimi for Coding); none turns thinking off.',
      noEffortModels: ['kimi-for-coding-highspeed'],
    },
    autoSettings: 'generic',
    apiKey: {
      inputId: 'kimi-code-api-key',
      inputName: 'moto-kimi-code-api-key',
      consoleUrl: 'https://www.kimi.com/code/',
      consoleLabel: 'kimi.com/code',
      description: (
        'Use an API key from the Kimi Code console (Kimi membership). A Kimi Platform pay-as-you-go '
        + 'key will not work here. MOTO identifies itself to Kimi as MOTO, and requests share the '
        + 'quota of your other Kimi Code devices and keys.'
      ),
      supportsUsage: true,
    },
  },
];

const CLOUD_ACCESS_PROVIDER_BY_ID = new Map(CLOUD_ACCESS_PROVIDERS.map((provider) => [provider.id, provider]));

export function isCloudAccessProvider(providerId) {
  return CLOUD_ACCESS_PROVIDER_BY_ID.has(providerId);
}

export function getCloudAccessProvider(providerId) {
  return CLOUD_ACCESS_PROVIDER_BY_ID.get(providerId) || null;
}

export function cloudAccessProviderLabel(providerId) {
  return getCloudAccessProvider(providerId)?.label || 'Cloud Provider';
}

export function cloudAccessProviderShortLabel(providerId) {
  return getCloudAccessProvider(providerId)?.shortLabel || cloudAccessProviderLabel(providerId);
}

export function isApiKeyCloudProvider(providerId) {
  return getCloudAccessProvider(providerId)?.authKind === 'api_key';
}

// Desktop-only providers are available unless hosted mode is active or the backend
// reports the capability off; a missing flag (older backend) counts as available.
export function isCloudProviderAvailable(capabilities, providerId) {
  const provider = getCloudAccessProvider(providerId);
  if (!provider || capabilities?.genericMode) {
    return false;
  }
  return capabilities?.[provider.capabilityKey] !== false;
}

export function getAvailableCloudAccessProviders(capabilities) {
  return CLOUD_ACCESS_PROVIDERS.filter((provider) => isCloudProviderAvailable(capabilities, provider.id));
}

export function getConfiguredCloudAccessProviders(statusByProvider = {}) {
  return CLOUD_ACCESS_PROVIDERS.filter((provider) => Boolean(statusByProvider?.[provider.id]?.configured));
}

export function chooseDefaultCloudAccessProvider(statusByProvider = {}) {
  return getConfiguredCloudAccessProviders(statusByProvider)[0]?.id || CLOUD_ACCESS_PROVIDERS[0].id;
}

export function chooseCloudAccessProvider(statusByProvider = {}, currentProvider = '') {
  const configured = getConfiguredCloudAccessProviders(statusByProvider);
  if (currentProvider && configured.some((provider) => provider.id === currentProvider)) {
    return currentProvider;
  }
  if (configured.length === 1) {
    return configured[0].id;
  }
  return configured[0]?.id || CLOUD_ACCESS_PROVIDERS[0].id;
}
