export const OPENAI_CODEX_PROVIDER = 'openai_codex_oauth';
export const XAI_GROK_PROVIDER = 'xai_grok_oauth';
export const SAKANA_FUGU_PROVIDER = 'sakana_fugu';

export const CLOUD_ACCESS_PROVIDERS = [
  {
    id: OPENAI_CODEX_PROVIDER,
    label: 'OpenAI Codex',
    shortLabel: 'Codex',
    loginLabel: 'OpenAI Codex OAuth',
    unavailableTitle: 'Set OpenAI Codex login in OpenRouter/OAuth first',
    modelErrorLabel: 'OpenAI Codex OAuth',
  },
  {
    id: XAI_GROK_PROVIDER,
    label: 'xAI Grok',
    shortLabel: 'Grok',
    loginLabel: 'xAI Grok OAuth',
    unavailableTitle: 'Set xAI Grok login in OpenRouter/OAuth first',
    modelErrorLabel: 'xAI Grok OAuth',
  },
  {
    id: SAKANA_FUGU_PROVIDER,
    label: 'Sakana Fugu',
    shortLabel: 'Fugu',
    loginLabel: 'Sakana Fugu API Key',
    unavailableTitle: 'Set Sakana Fugu API key in OpenRouter/OAuth first',
    modelErrorLabel: 'Sakana Fugu API',
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
