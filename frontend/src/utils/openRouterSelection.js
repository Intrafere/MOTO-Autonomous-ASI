export const DEFAULT_CONTEXT_WINDOW = '';
export const DEFAULT_MAX_OUTPUT_TOKENS = '';
export const DEFAULT_OPENROUTER_REASONING_EFFORT = 'auto';
export const USA_HOST_TOOLTIP = 'We manually mark USA based companies so researchers can more easily distinguish which hosts may have more strict data protection laws. Check OpenRouter terms to be certain.';
export const OPENROUTER_REASONING_EFFORT_OPTIONS = [
  { value: 'auto', label: 'Auto (max supported)' },
  { value: 'xhigh', label: 'xhigh (maximum)' },
  { value: 'high', label: 'high' },
  { value: 'medium', label: 'medium' },
  { value: 'low', label: 'low' },
  { value: 'minimal', label: 'minimal' },
  { value: 'none', label: 'none / disabled' },
];
export const SAKANA_FUGU_REASONING_EFFORT_OPTIONS = [
  { value: 'auto', label: 'Auto (xhigh)' },
  { value: 'xhigh', label: 'xhigh (maximum)' },
  { value: 'high', label: 'high' },
];
const AUTO_ENDPOINT_OUTLIER_RATIO = 0.75;
const AUTO_MIN_CAPABLE_OUTPUT_TOKENS = 32768;
const KNOWN_WEAK_AUTO_PROVIDERS = new Set([
  'venice',
]);
const USA_OPENROUTER_HOST_PROVIDERS = new Set([
  'amazon',
  'amazon bedrock',
  'anthropic',
  'anyscale',
  'aws',
  'aws bedrock',
  'azure',
  'baseten',
  'cerebras',
  'cerebras systems',
  'cloudflare',
  'databricks',
  'deepinfra',
  'fireworks',
  'fireworks ai',
  'google',
  'google ai',
  'google ai studio',
  'google vertex',
  'groq',
  'hyperbolic',
  'inference.net',
  'lambda',
  'lambda labs',
  'lepton',
  'lepton ai',
  'meta',
  'microsoft',
  'microsoft azure',
  'modal',
  'nvidia',
  'nvidia nim',
  'openai',
  'openrouter',
  'parasail',
  'perplexity',
  'replicate',
  'sambanova',
  'sambanova systems',
  'snowflake',
  'together',
  'together ai',
  'xai',
]);
const XAI_GROK_KNOWN_MODEL_LIMITS = {
  'grok-4': {
    contextLength: 1000000,
    maxOutputTokens: 131072,
  },
  'grok-4.2': {
    contextLength: 1000000,
    maxOutputTokens: 131072,
  },
  'grok-4.3': {
    contextLength: 1000000,
    maxOutputTokens: 131072,
  },
};

function toPositiveInteger(value) {
  const parsed = Number(value);
  if (!Number.isFinite(parsed) || parsed <= 0) {
    return null;
  }
  return Math.floor(parsed);
}

function getModelContext(model) {
  return (
    toPositiveInteger(model?.context_length) ||
    toPositiveInteger(model?.top_provider?.context_length)
  );
}

function normalizeProviderName(providerName) {
  return typeof providerName === 'string' ? providerName.trim().toLowerCase().replace(/\s+/g, ' ') : '';
}

function getEndpointProviderName(endpoint) {
  return (
    endpoint?.provider_name ||
    endpoint?.provider ||
    endpoint?.name ||
    endpoint?.id ||
    ''
  );
}

function getCloudAccessModelContext(model) {
  return (
    toPositiveInteger(model?.context_length) ||
    toPositiveInteger(model?.context_window) ||
    toPositiveInteger(model?.contextTokens) ||
    toPositiveInteger(model?.input_context_window) ||
    toPositiveInteger(model?.effective_input_context_window)
  );
}

function getCloudAccessModelOutputCap(model) {
  return (
    toPositiveInteger(model?.max_output_tokens) ||
    toPositiveInteger(model?.max_completion_tokens) ||
    toPositiveInteger(model?.output_tokens) ||
    toPositiveInteger(model?.completion_tokens)
  );
}

function getKnownXAIGrokLimits(model) {
  const candidates = [
    model?.id,
    model?.name,
    model?.title,
    model?.slug,
  ]
    .filter(Boolean)
    .map((value) => String(value).trim().toLowerCase().replace(/_/g, '-'));

  const knownModelId = candidates.find((candidate) => (
    Object.prototype.hasOwnProperty.call(XAI_GROK_KNOWN_MODEL_LIMITS, candidate)
  ));
  return knownModelId ? XAI_GROK_KNOWN_MODEL_LIMITS[knownModelId] : null;
}

function uniqueSorted(values) {
  return Array.from(new Set(values.filter(Boolean))).sort((a, b) => a.localeCompare(b));
}

function median(values) {
  const sorted = values
    .filter((value) => Number.isFinite(value))
    .slice()
    .sort((a, b) => a - b);
  if (sorted.length === 0) {
    return null;
  }
  const midpoint = Math.floor(sorted.length / 2);
  if (sorted.length % 2 === 1) {
    return sorted[midpoint];
  }
  return Math.floor((sorted[midpoint - 1] + sorted[midpoint]) / 2);
}

function buildEndpointMetric(endpoint) {
  return {
    endpoint,
    providerName: getEndpointProviderName(endpoint),
    normalizedProviderName: normalizeProviderName(getEndpointProviderName(endpoint)),
    contextLength: toPositiveInteger(endpoint?.context_length),
    outputCap: toPositiveInteger(endpoint?.max_completion_tokens),
    promptCap: toPositiveInteger(endpoint?.max_prompt_tokens),
  };
}

function selectAutoEndpointSet(endpoints, modelContext = null) {
  if (!Array.isArray(endpoints) || endpoints.length <= 1) {
    return {
      endpoints: endpoints || [],
      ignoredEndpoints: [],
      recommendedProvider: null,
      capableProviderNames: uniqueSorted((endpoints || []).map(getEndpointProviderName)),
      ignoredProviderNames: [],
      providerSelectionRecommended: false,
    };
  }

  const metrics = endpoints.map(buildEndpointMetric);
  const outputCaps = metrics.map((metric) => metric.outputCap).filter((value) => value !== null);
  const contextLengths = metrics.map((metric) => metric.contextLength).filter((value) => value !== null);
  const medianOutputCap = median(outputCaps);
  const outputThreshold = medianOutputCap && medianOutputCap >= AUTO_MIN_CAPABLE_OUTPUT_TOKENS
    ? Math.max(AUTO_MIN_CAPABLE_OUTPUT_TOKENS, Math.floor(medianOutputCap * AUTO_ENDPOINT_OUTLIER_RATIO))
    : null;
  const contextThresholdFromModel = modelContext
    ? Math.floor(modelContext * AUTO_ENDPOINT_OUTLIER_RATIO)
    : null;
  const hasEndpointNearModelContext = contextThresholdFromModel !== null && metrics.some(
    (metric) => metric.contextLength !== null && metric.contextLength >= contextThresholdFromModel
  );
  const contextThreshold = hasEndpointNearModelContext ? contextThresholdFromModel : null;

  const annotatedMetrics = metrics.map((metric) => {
    const reasons = [];
    if (KNOWN_WEAK_AUTO_PROVIDERS.has(metric.normalizedProviderName)) {
      reasons.push('known weak auto-routing host');
    }
    if (outputThreshold !== null && metric.outputCap === null && outputCaps.length > 0) {
      reasons.push('missing max_completion_tokens while other endpoints expose output caps');
    }
    if (outputThreshold !== null && metric.outputCap !== null && metric.outputCap < outputThreshold) {
      reasons.push(`max_completion_tokens=${metric.outputCap} below capable threshold ${outputThreshold}`);
    }
    if (contextThreshold !== null && metric.contextLength !== null && metric.contextLength < contextThreshold) {
      reasons.push(`context_length=${metric.contextLength} below capable threshold ${contextThreshold}`);
    }
    if (contextThreshold !== null && metric.contextLength === null && contextLengths.length > 0) {
      reasons.push('missing context_length while other endpoints match model context');
    }
    return { ...metric, reasons };
  });

  const capableMetrics = annotatedMetrics.filter((metric) => metric.reasons.length === 0);
  if (capableMetrics.length === 0) {
    return {
      endpoints,
      ignoredEndpoints: [],
      recommendedProvider: null,
      capableProviderNames: uniqueSorted(metrics.map((metric) => metric.providerName)),
      ignoredProviderNames: [],
      providerSelectionRecommended: false,
    };
  }

  const ignoredMetrics = annotatedMetrics.filter((metric) => metric.reasons.length > 0);
  if (ignoredMetrics.length === 0) {
    return {
      endpoints,
      ignoredEndpoints: [],
      recommendedProvider: null,
      capableProviderNames: uniqueSorted(capableMetrics.map((metric) => metric.providerName)),
      ignoredProviderNames: [],
      providerSelectionRecommended: false,
    };
  }

  const capableProviderNames = uniqueSorted(capableMetrics.map((metric) => metric.providerName));

  return {
    endpoints: capableMetrics.map((metric) => metric.endpoint),
    ignoredEndpoints: ignoredMetrics,
    recommendedProvider: null,
    capableProviderNames,
    ignoredProviderNames: uniqueSorted(ignoredMetrics.map((metric) => metric.providerName)),
    providerSelectionRecommended: false,
  };
}

export function findOpenRouterModel(models, modelId) {
  if (!Array.isArray(models) || !modelId) {
    return null;
  }
  return models.find((model) => model.id === modelId) || null;
}

export function computeCloudAccessAutoSettings(model, providerLabel = 'OpenRouter/OAuth') {
  const warnings = [];
  const contextWindow = getCloudAccessModelContext(model);
  const maxOutputTokens = getCloudAccessModelOutputCap(model);

  if (!contextWindow) {
    warnings.push(`${providerLabel} model metadata did not expose a known context window; preserving the current context setting.`);
  }
  if (!maxOutputTokens) {
    warnings.push(`${providerLabel} model metadata did not expose a max output cap; preserving the current max output setting.`);
  }

  return {
    contextWindow,
    contextWindowKnown: contextWindow !== null,
    maxOutputTokens,
    outputCapKnown: maxOutputTokens !== null,
    outputCapSource: maxOutputTokens !== null ? 'cloud-access-model-metadata' : 'unknown',
    source: 'cloud-access-model-metadata',
    inputContextWindow: toPositiveInteger(model?.input_context_window),
    effectiveInputContextWindow: toPositiveInteger(model?.effective_input_context_window),
    warnings,
  };
}

export function computeCodexAutoSettings(model) {
  return computeCloudAccessAutoSettings(model, 'Codex');
}

export function computeXAIGrokAutoSettings(model) {
  const knownLimits = getKnownXAIGrokLimits(model);
  const metadataContextWindow = getCloudAccessModelContext(model);
  const metadataOutputCap = getCloudAccessModelOutputCap(model);
  const modelWithKnownLimits = knownLimits
    ? {
        ...model,
        context_length: metadataContextWindow || knownLimits.contextLength,
        max_output_tokens: metadataOutputCap || knownLimits.maxOutputTokens,
      }
    : model;
  const autoSettings = computeCloudAccessAutoSettings(modelWithKnownLimits, 'xAI Grok');
  if (!knownLimits || (metadataContextWindow && metadataOutputCap)) {
    return autoSettings;
  }
  return {
    ...autoSettings,
    source: `${autoSettings.source}+xai-grok-known-limits`,
    warnings: autoSettings.warnings.filter((warning) => (
      !warning.includes('xAI Grok model metadata did not expose')
    )),
  };
}

export function computeSakanaFuguAutoSettings(model) {
  return computeCloudAccessAutoSettings(model, 'Sakana Fugu');
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

export function isOpenRouterUsHostProvider(providerName) {
  return USA_OPENROUTER_HOST_PROVIDERS.has(normalizeProviderName(providerName));
}

export function formatOpenRouterProviderLabel(providerName) {
  return isOpenRouterUsHostProvider(providerName) ? `🇺🇸 ${providerName}` : providerName;
}

export function getOpenRouterProviderTitle(providerName) {
  return isOpenRouterUsHostProvider(providerName) ? USA_HOST_TOOLTIP : undefined;
}

export function normalizeOpenRouterReasoningEffort(value) {
  const normalized = typeof value === 'string' ? value.trim().toLowerCase() : '';
  if (OPENROUTER_REASONING_EFFORT_OPTIONS.some((option) => option.value === normalized)) {
    return normalized;
  }
  return DEFAULT_OPENROUTER_REASONING_EFFORT;
}

export function getReasoningSupportInfo(providerData, selectedProvider = null) {
  const { endpoints } = normalizeProviderData(providerData);
  const normalizedSelectedProvider = normalizeProviderName(selectedProvider);
  const relevantEndpoints = selectedProvider
    ? endpoints.filter((endpoint) => normalizeProviderName(getEndpointProviderName(endpoint)) === normalizedSelectedProvider)
    : endpoints;
  const supportedParameters = relevantEndpoints.flatMap((endpoint) => (
    Array.isArray(endpoint?.supported_parameters) ? endpoint.supported_parameters : []
  ));
  const normalizedParams = supportedParameters.map((param) => String(param).toLowerCase());
  const supportsReasoning = normalizedParams.some((param) => (
    param === 'reasoning' ||
    param === 'reasoning_effort' ||
    param === 'reasoning.effort' ||
    param === 'include_reasoning'
  ));

  return {
    supportsReasoning,
    hasEndpointMetadata: relevantEndpoints.length > 0,
    supportedParameters,
  };
}

/**
 * Compute auto-fill context window + max output tokens for an OpenRouter model.
 *
 * Returns a valid object and marks which values are metadata-backed:
 *   1. Context: OpenRouter model.context_length is the source of truth
 *   2. Output: largest non-outlier endpoint max_completion_tokens
 *   3. Safety: cap output at 20% of model.context_length
 *   4. Unknown: ask callers not to overwrite values when metadata is missing
 *
 * The `source` field reports which tier produced the answer, and `warnings`
 * is a list of human-readable diagnostics for logging.
 */
export function computeOpenRouterAutoSettings(model, providerData, selectedProvider = null) {
  const { endpoints } = normalizeProviderData(providerData);
  const warnings = [];
  const modelContext = getModelContext(model);
  const normalizedSelectedProvider = normalizeProviderName(selectedProvider);

  const initialRelevantEndpoints = selectedProvider
    ? endpoints.filter((endpoint) => normalizeProviderName(getEndpointProviderName(endpoint)) === normalizedSelectedProvider)
    : endpoints;
  const autoEndpointSelection = selectedProvider
    ? {
        endpoints: initialRelevantEndpoints,
        ignoredEndpoints: [],
        recommendedProvider: null,
        capableProviderNames: uniqueSorted(initialRelevantEndpoints.map(getEndpointProviderName)),
        ignoredProviderNames: [],
        providerSelectionRecommended: false,
      }
    : selectAutoEndpointSet(initialRelevantEndpoints, modelContext);
  const relevantEndpoints = autoEndpointSelection.endpoints;

  if (selectedProvider && initialRelevantEndpoints.length === 0 && endpoints.length > 0) {
    warnings.push(
      `Selected provider "${selectedProvider}" not present in endpoint list; falling back to model-level context.`
    );
  }

  if (!selectedProvider && autoEndpointSelection.ignoredEndpoints.length > 0) {
    const ignoredSummary = autoEndpointSelection.ignoredEndpoints
      .map((metric) => `${metric.providerName || 'unknown'} (${metric.reasons.join('; ')})`)
      .join(', ');
    warnings.push(
      `Ignored weak OpenRouter auto-routing endpoint(s): ${ignoredSummary}.`
    );
  }

  if (relevantEndpoints.length === 0) {
    const contextWindow = modelContext;
    const contextWindowKnown = modelContext !== null;
    const maxOutputTokens = null;
    const outputCapSource = 'unknown';

    if (!modelContext) {
      warnings.push(
        'No endpoint metadata and no model.context_length; preserving the current context setting.'
      );
    } else {
      warnings.push(
        `No endpoint metadata available; falling back to model.context_length=${modelContext}.`
      );
    }

    warnings.push(
      'No endpoint metadata exposed max_completion_tokens; preserving the current max output setting.'
    );

    return {
      contextWindow,
      contextWindowKnown,
      maxOutputTokens,
      outputCapKnown: maxOutputTokens !== null,
      outputCapSource,
      smallestEndpointOutputCap: null,
      smallestEndpointContext: null,
      smallestEndpointPromptCap: null,
      largestEndpointOutputCap: null,
      largestEndpointContext: null,
      largestEndpointPromptCap: null,
      recommendedProvider: autoEndpointSelection.recommendedProvider,
      providerSelectionRecommended: autoEndpointSelection.providerSelectionRecommended,
      capableProviderNames: autoEndpointSelection.capableProviderNames,
      ignoredProviderNames: autoEndpointSelection.ignoredProviderNames,
      fallbackModelContext: modelContext,
      source: modelContext ? 'model-context-length' : 'unknown-context',
      warnings,
    };
  }

  const endpointContexts = relevantEndpoints
    .map((endpoint) => toPositiveInteger(endpoint.context_length))
    .filter((value) => value !== null);

  const endpointOutputCaps = relevantEndpoints
    .map((endpoint) => toPositiveInteger(endpoint?.max_completion_tokens))
    .filter((value) => value !== null);

  const endpointPromptCaps = relevantEndpoints
    .map((endpoint) => toPositiveInteger(endpoint?.max_prompt_tokens))
    .filter((value) => value !== null);

  const smallestEndpointContext = endpointContexts.length > 0 ? Math.min(...endpointContexts) : null;
  const smallestEndpointOutputCap = endpointOutputCaps.length > 0 ? Math.min(...endpointOutputCaps) : null;
  const smallestEndpointPromptCap = endpointPromptCaps.length > 0 ? Math.min(...endpointPromptCaps) : null;
  const largestEndpointContext = endpointContexts.length > 0 ? Math.max(...endpointContexts) : null;
  const largestEndpointOutputCap = endpointOutputCaps.length > 0 ? Math.max(...endpointOutputCaps) : null;
  const largestEndpointPromptCap = endpointPromptCaps.length > 0 ? Math.max(...endpointPromptCaps) : null;

  // The model-level OpenRouter context is the total context source of truth.
  // Endpoint context rows are provider diagnostics only; they must not shrink
  // the configured model context after weak providers have been filtered out.
  const contextWindow = modelContext;
  const contextWindowKnown = modelContext !== null;
  let contextSource;
  if (modelContext) {
    contextSource = 'model-context-length';
  } else {
    contextSource = 'unknown-context';
    warnings.push(
      'No OpenRouter model.context_length; preserving the current context setting.'
    );
  }

  // Determine max output tokens from non-outlier endpoint caps, capped at 20%
  // of the OpenRouter model context. In OpenRouter auto mode, use the smallest
  // capable endpoint cap so OpenRouter can choose any remaining host safely.
  let maxOutputTokens;
  let outputCapSource;
  const endpointOutputCap = selectedProvider ? largestEndpointOutputCap : smallestEndpointOutputCap;
  const contextBasedOutputCap = modelContext ? Math.floor(modelContext * 0.2) : null;
  if (endpointOutputCap !== null && contextBasedOutputCap !== null) {
    maxOutputTokens = Math.min(contextBasedOutputCap, endpointOutputCap);
    outputCapSource = 'endpoint-metadata';
  } else {
    maxOutputTokens = null;
    outputCapSource = 'unknown';
    if (endpointOutputCap !== null && contextBasedOutputCap === null) {
      warnings.push(
        'Endpoint metadata exposed max_completion_tokens but model.context_length is unknown; preserving the current max output setting.'
      );
    } else {
      warnings.push(
        'No endpoints exposed max_completion_tokens; preserving the current max output setting.'
      );
    }
  }

  const source = `${contextSource}+${outputCapSource}`;

  return {
    contextWindow,
    contextWindowKnown,
    maxOutputTokens,
    outputCapKnown: maxOutputTokens !== null,
    outputCapSource,
    smallestEndpointOutputCap,
    smallestEndpointContext,
    smallestEndpointPromptCap,
    largestEndpointOutputCap,
    largestEndpointContext,
    largestEndpointPromptCap,
    recommendedProvider: autoEndpointSelection.recommendedProvider,
    providerSelectionRecommended: autoEndpointSelection.providerSelectionRecommended,
    capableProviderNames: autoEndpointSelection.capableProviderNames,
    ignoredProviderNames: autoEndpointSelection.ignoredProviderNames,
    fallbackModelContext: modelContext,
    source,
    warnings,
  };
}
