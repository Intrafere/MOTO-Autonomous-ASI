const DEFAULT_CONTEXT_WINDOW = 131072;
const CONTEXT_BUFFER_TOKENS = 500;

function toPositiveInteger(value) {
  const parsed = Number(value);
  if (!Number.isFinite(parsed) || parsed <= 0) {
    return null;
  }
  return Math.floor(parsed);
}

export function findOpenRouterModel(models, modelId) {
  if (!Array.isArray(models) || !modelId) {
    return null;
  }
  return models.find((model) => model.id === modelId) || null;
}

export function hasEndpointMetadata(providerData) {
  return Boolean(
    providerData &&
    !Array.isArray(providerData) &&
    Array.isArray(providerData.endpoints)
  );
}

export function normalizeProviderData(providerData) {
  if (Array.isArray(providerData)) {
    return {
      providers: providerData,
      endpoints: [],
    };
  }

  if (!providerData || typeof providerData !== 'object') {
    return {
      providers: [],
      endpoints: [],
    };
  }

  return {
    providers: Array.isArray(providerData.providers) ? providerData.providers : [],
    endpoints: Array.isArray(providerData.endpoints) ? providerData.endpoints : [],
  };
}

export function getProviderNames(providerData) {
  return normalizeProviderData(providerData).providers;
}

export function computeOpenRouterAutoSettings(model, providerData, selectedProvider = null) {
  const { endpoints } = normalizeProviderData(providerData);

  const relevantEndpoints = selectedProvider
    ? endpoints.filter((endpoint) => endpoint?.provider_name === selectedProvider)
    : endpoints;

  if (relevantEndpoints.length === 0) {
    return null;
  }

  const hasCompleteEndpointContexts = relevantEndpoints.every(
    (endpoint) => toPositiveInteger(endpoint?.context_length) !== null
  );
  const hasCompleteEndpointOutputCaps = relevantEndpoints.every(
    (endpoint) => toPositiveInteger(endpoint?.max_completion_tokens) !== null
  );

  if (!hasCompleteEndpointContexts || !hasCompleteEndpointOutputCaps) {
    return null;
  }

  const endpointContexts = relevantEndpoints
    .map((endpoint) => toPositiveInteger(endpoint?.context_length))
    .filter((value) => value !== null);

  const endpointOutputCaps = relevantEndpoints
    .map((endpoint) => toPositiveInteger(endpoint?.max_completion_tokens))
    .filter((value) => value !== null);

  const endpointPromptCaps = relevantEndpoints
    .map((endpoint) => toPositiveInteger(endpoint?.max_prompt_tokens))
    .filter((value) => value !== null);

  const smallestEndpointContext = Math.min(...endpointContexts);
  const smallestEndpointOutputCap = Math.min(...endpointOutputCaps);
  const smallestEndpointPromptCap = endpointPromptCaps.length > 0
    ? Math.min(...endpointPromptCaps)
    : null;

  let contextWindow = smallestEndpointContext;
  let twentyPercentOutputCap = Math.max(1, Math.floor(contextWindow * 0.2));

  if (smallestEndpointPromptCap !== null) {
    const promptLimitedOutputCap = Math.max(
      1,
      Math.floor((smallestEndpointPromptCap + CONTEXT_BUFFER_TOKENS) / 4)
    );
    twentyPercentOutputCap = Math.min(twentyPercentOutputCap, promptLimitedOutputCap);
  }

  const maxOutputTokens = Math.min(smallestEndpointOutputCap, twentyPercentOutputCap);

  if (smallestEndpointPromptCap !== null) {
    contextWindow = Math.min(
      contextWindow,
      smallestEndpointPromptCap + maxOutputTokens + CONTEXT_BUFFER_TOKENS
    );
  }

  return {
    contextWindow,
    maxOutputTokens,
    twentyPercentOutputCap,
    smallestEndpointOutputCap,
    smallestEndpointContext,
    smallestEndpointPromptCap,
    fallbackModelContext: toPositiveInteger(model?.context_length) || DEFAULT_CONTEXT_WINDOW,
  };
}
