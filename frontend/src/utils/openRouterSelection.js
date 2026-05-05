const DEFAULT_CONTEXT_WINDOW = 131072;
const CONTEXT_BUFFER_TOKENS = 500;
const KNOWN_NO_OUTPUT_CAP_DEFAULTS = {
  'x-ai/grok-4.3': 128000,
};

function toPositiveInteger(value) {
  const parsed = Number(value);
  if (!Number.isFinite(parsed) || parsed <= 0) {
    return null;
  }
  return Math.floor(parsed);
}

function getKnownNoOutputCapDefault(model) {
  const modelId = typeof model?.id === 'string' ? model.id.toLowerCase() : '';
  return KNOWN_NO_OUTPUT_CAP_DEFAULTS[modelId] || null;
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

/**
 * Compute auto-fill context window + max output tokens for an OpenRouter model.
 *
 * Returns a valid object and marks which values are metadata-backed:
 *   1. Best: full endpoint metadata for the relevant provider(s)
 *   2. Partial: use explicit `max_completion_tokens` when present
 *   3. No-cap: use vetted model-specific defaults for known no-cap models
 *   4. Unknown: fill context only and ask callers not to overwrite output
 *
 * The `source` field reports which tier produced the answer, and `warnings`
 * is a list of human-readable diagnostics for logging.
 */
export function computeOpenRouterAutoSettings(model, providerData, selectedProvider = null) {
  const { endpoints } = normalizeProviderData(providerData);
  const warnings = [];

  const relevantEndpoints = selectedProvider
    ? endpoints.filter((endpoint) => endpoint?.provider_name === selectedProvider)
    : endpoints;

  if (selectedProvider && relevantEndpoints.length === 0 && endpoints.length > 0) {
    warnings.push(
      `Selected provider "${selectedProvider}" not present in endpoint list; falling back to model-level context.`
    );
  }

  const modelContext = toPositiveInteger(model?.context_length);
  const knownNoOutputCapDefault = getKnownNoOutputCapDefault(model);

  if (relevantEndpoints.length === 0) {
    const contextWindow = modelContext || DEFAULT_CONTEXT_WINDOW;
    const contextWindowKnown = modelContext !== null;
    const maxOutputTokens = knownNoOutputCapDefault;

    if (!modelContext) {
      warnings.push(
        `No endpoint metadata and no model.context_length; using default ${DEFAULT_CONTEXT_WINDOW}.`
      );
    } else {
      warnings.push(
        `No endpoint metadata available; falling back to model.context_length=${modelContext}.`
      );
    }

    if (maxOutputTokens === null) {
      warnings.push(
        'No endpoint metadata exposed max_completion_tokens; preserving the current max output setting.'
      );
    } else {
      warnings.push(
        `No endpoint metadata exposed max_completion_tokens; using known no-cap default ${maxOutputTokens}.`
      );
    }

    return {
      contextWindow,
      contextWindowKnown,
      maxOutputTokens,
      outputCapKnown: maxOutputTokens !== null,
      outputCapSource: maxOutputTokens !== null ? 'known-no-cap-default' : 'unknown',
      smallestEndpointOutputCap: null,
      smallestEndpointContext: null,
      smallestEndpointPromptCap: null,
      fallbackModelContext: modelContext || DEFAULT_CONTEXT_WINDOW,
      source: modelContext ? 'model-context-length' : 'hardcoded-default',
      warnings,
    };
  }

  // Filter endpoints to only those that expose a usable context_length.
  const endpointsWithContext = relevantEndpoints.filter(
    (endpoint) => toPositiveInteger(endpoint?.context_length) !== null
  );

  const endpointContexts = endpointsWithContext
    .map((endpoint) => toPositiveInteger(endpoint.context_length))
    .filter((value) => value !== null);

  const endpointOutputCaps = relevantEndpoints
    .map((endpoint) => toPositiveInteger(endpoint?.max_completion_tokens))
    .filter((value) => value !== null);

  const endpointPromptCaps = relevantEndpoints
    .map((endpoint) => toPositiveInteger(endpoint?.max_prompt_tokens))
    .filter((value) => value !== null);

  // Choose a base context: smallest endpoint context, then model context, then default.
  let contextWindow;
  let contextWindowKnown = true;
  if (endpointContexts.length > 0) {
    contextWindow = Math.min(...endpointContexts);
    if (endpointContexts.length < relevantEndpoints.length) {
      warnings.push(
        `${relevantEndpoints.length - endpointContexts.length}/${relevantEndpoints.length} endpoints missing context_length; using min of remaining.`
      );
    }
  } else if (modelContext) {
    contextWindow = modelContext;
    warnings.push(
      'No endpoints exposed context_length; falling back to model.context_length.'
    );
  } else {
    contextWindow = DEFAULT_CONTEXT_WINDOW;
    contextWindowKnown = false;
    warnings.push(
      `No endpoint or model context_length; using default ${DEFAULT_CONTEXT_WINDOW}.`
    );
  }

  const smallestEndpointContext = endpointContexts.length > 0 ? Math.min(...endpointContexts) : null;
  const smallestEndpointOutputCap = endpointOutputCaps.length > 0 ? Math.min(...endpointOutputCaps) : null;
  const smallestEndpointPromptCap = endpointPromptCaps.length > 0 ? Math.min(...endpointPromptCaps) : null;

  // Determine max output tokens.
  // If at least one endpoint provides max_completion_tokens, honor the smallest.
  // If none do, use only vetted model-specific defaults; otherwise preserve
  // the user's current setting instead of guessing from context length.
  let maxOutputTokens;
  let outputCapSource;
  if (smallestEndpointOutputCap !== null) {
    maxOutputTokens = smallestEndpointOutputCap;
    outputCapSource = 'endpoint-metadata';
    if (endpointOutputCaps.length < relevantEndpoints.length) {
      warnings.push(
        `${relevantEndpoints.length - endpointOutputCaps.length}/${relevantEndpoints.length} endpoints missing max_completion_tokens; using min of remaining.`
      );
    }
  } else if (knownNoOutputCapDefault !== null) {
    maxOutputTokens = knownNoOutputCapDefault;
    outputCapSource = 'known-no-cap-default';
    warnings.push(
      `No endpoints exposed max_completion_tokens; using known no-cap default ${maxOutputTokens}.`
    );
  } else {
    maxOutputTokens = null;
    outputCapSource = 'unknown';
    warnings.push(
      'No endpoints exposed max_completion_tokens; preserving the current max output setting.'
    );
  }

  if (smallestEndpointPromptCap !== null && maxOutputTokens !== null) {
    contextWindow = Math.min(
      contextWindow,
      smallestEndpointPromptCap + maxOutputTokens + CONTEXT_BUFFER_TOKENS
    );
  }

  const source = smallestEndpointContext !== null && smallestEndpointOutputCap !== null
    ? 'endpoint-metadata'
    : 'partial-endpoint-metadata';

  return {
    contextWindow,
    contextWindowKnown,
    maxOutputTokens,
    outputCapKnown: maxOutputTokens !== null,
    outputCapSource,
    smallestEndpointOutputCap,
    smallestEndpointContext,
    smallestEndpointPromptCap,
    fallbackModelContext: modelContext || DEFAULT_CONTEXT_WINDOW,
    source,
    warnings,
  };
}
