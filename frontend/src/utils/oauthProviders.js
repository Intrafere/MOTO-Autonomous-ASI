// Registry of desktop role providers beyond LM Studio and OpenRouter: cloud
// subscriptions (OAuth logins and subscription API keys) and local inference
// servers (vLLM). Every settings panel, the OpenRouter/OAuth modal, and the API
// helpers read provider details from here, so adding a provider is one entry plus
// its backend profile. It mirrors backend/shared/subscription_providers/metadata.py.

export const OPENAI_CODEX_PROVIDER = 'openai_codex_oauth';
export const XAI_GROK_PROVIDER = 'xai_grok_oauth';
export const SAKANA_FUGU_PROVIDER = 'sakana_fugu';
export const KIMI_CODE_PROVIDER = 'kimi_code';
export const CLAUDE_API_PROVIDER = 'claude_api';
export const AMAZON_BEDROCK_PROVIDER = 'amazon_bedrock';
export const VLLM_PROVIDER = 'vllm';

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
export const CLAUDE_REASONING_EFFORT_OPTIONS = [
  { value: 'auto', label: 'Auto (model default)' },
  { value: 'max', label: 'max' },
  { value: 'xhigh', label: 'xhigh' },
  { value: 'high', label: 'high' },
  { value: 'medium', label: 'medium' },
  { value: 'low', label: 'low' },
];
const CLAUDE_REASONING = {
  control: 'options',
  options: CLAUDE_REASONING_EFFORT_OPTIONS,
  aliases: { maximum: 'max', highest: 'max', minimal: 'low', none: 'low' },
};

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
  {
    id: CLAUDE_API_PROVIDER,
    label: 'Anthropic Claude',
    shortLabel: 'Claude',
    authKind: 'api_key',
    apiSlug: 'claude-api',
    capabilityKey: 'claudeApiAvailable',
    featureFlag: 'claude_api_available',
    loginLabel: 'Anthropic Claude API Key',
    unavailableTitle: 'Set Anthropic Claude API key in OpenRouter/OAuth first',
    modelErrorLabel: 'Anthropic Claude API',
    noModelsMessage: 'Anthropic Claude API key is saved, but no Claude models were returned. Check the key in the Claude Console.',
    reasoning: {
      ...CLAUDE_REASONING,
      ariaLabel: 'Claude Reasoning Effort',
      hint: 'Auto uses the model\'s own default effort. A level a model does not offer falls to the next lower one.',
    },
    autoSettings: 'generic',
    apiKey: {
      inputId: 'claude-api-key',
      inputName: 'moto-claude-api-key',
      consoleUrl: 'https://platform.claude.com/settings/keys',
      consoleLabel: 'platform.claude.com',
      description: (
        'Use a Claude API key from the Claude Console to run Claude models directly as a MOTO role '
        + 'provider. This is pay-as-you-go API billing, not a Claude.ai or Claude Code subscription login.'
      ),
      supportsUsage: false,
    },
  },
  {
    id: AMAZON_BEDROCK_PROVIDER,
    label: 'Amazon Bedrock',
    shortLabel: 'Bedrock',
    authKind: 'api_key',
    apiSlug: 'amazon-bedrock',
    capabilityKey: 'amazonBedrockAvailable',
    featureFlag: 'amazon_bedrock_available',
    loginLabel: 'Amazon Bedrock API Key',
    unavailableTitle: 'Set Amazon Bedrock API key in OpenRouter/OAuth first',
    modelErrorLabel: 'Amazon Bedrock API',
    noModelsMessage: 'Amazon Bedrock API key is saved, but no models were returned. Check the key, its region, and model access in the Bedrock console.',
    reasoning: {
      ...CLAUDE_REASONING,
      ariaLabel: 'Amazon Bedrock Reasoning Effort',
      hint: 'Applies to Claude models; Auto uses the model\'s own default effort. Other Bedrock models ignore this setting.',
    },
    autoSettings: 'generic',
    apiKey: {
      inputId: 'amazon-bedrock-api-key',
      inputName: 'moto-amazon-bedrock-api-key',
      consoleUrl: 'https://console.aws.amazon.com/bedrock/home#/api-keys',
      consoleLabel: 'the Amazon Bedrock console (API keys)',
      description: (
        'Use an Amazon Bedrock API key and choose the AWS region it belongs to. Short-term keys expire '
        + 'within 12 hours and only work in the region that created them; a long-term key lasts until its '
        + 'expiry date. Usage is billed to your AWS account.'
      ),
      supportsUsage: false,
    },
  },
  {
    id: VLLM_PROVIDER,
    label: 'vLLM',
    shortLabel: 'vLLM',
    authKind: 'local_server',
    apiSlug: 'vllm',
    capabilityKey: 'vllmAvailable',
    featureFlag: 'vllm_available',
    loginLabel: 'vLLM server',
    unavailableTitle: 'Add your vLLM server in Inference Connectivity first',
    modelErrorLabel: 'vLLM',
    noModelsMessage: 'vLLM is set up, but its servers list no chat models. Start a vLLM server with a chat model, then refresh.',
    autoSettings: 'local_server',
  },
];

// Cloud subscriptions only: the providers behind the "Cloud" role button and the
// OpenRouter/OAuth modal. Local servers have their own button and setup window.
export const CLOUD_SUBSCRIPTION_PROVIDERS = CLOUD_ACCESS_PROVIDERS.filter(
  (provider) => provider.authKind !== 'local_server',
);
export const LOCAL_SERVER_PROVIDERS = CLOUD_ACCESS_PROVIDERS.filter(
  (provider) => provider.authKind === 'local_server',
);

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

export function isLocalServerProvider(providerId) {
  return getCloudAccessProvider(providerId)?.authKind === 'local_server';
}

export function isCloudSubscriptionProvider(providerId) {
  return isCloudAccessProvider(providerId) && !isLocalServerProvider(providerId);
}

// A role's local fallback is an LM Studio model id, or "<local server id>::<model>"
// for a model on a local server (vLLM). Mirrors split_local_fallback in the backend.
export const LOCAL_FALLBACK_SEPARATOR = '::';

export function localFallbackValue(providerId, modelId) {
  return providerId === 'lm_studio' ? modelId : `${providerId}${LOCAL_FALLBACK_SEPARATOR}${modelId}`;
}

export function describeLocalFallback(value) {
  const text = String(value || '').trim();
  for (const provider of LOCAL_SERVER_PROVIDERS) {
    const prefix = `${provider.id}${LOCAL_FALLBACK_SEPARATOR}`;
    if (text.startsWith(prefix) && text.length > prefix.length) {
      const modelId = text.slice(prefix.length);
      return { providerId: provider.id, providerLabel: provider.label, modelId, text: `${provider.label}: ${modelId}` };
    }
  }
  return { providerId: 'lm_studio', providerLabel: 'LM Studio', modelId: text, text };
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
  return CLOUD_SUBSCRIPTION_PROVIDERS.filter((provider) => isCloudProviderAvailable(capabilities, provider.id));
}

export function getConfiguredCloudAccessProviders(statusByProvider = {}) {
  return CLOUD_SUBSCRIPTION_PROVIDERS.filter((provider) => Boolean(statusByProvider?.[provider.id]?.configured));
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
