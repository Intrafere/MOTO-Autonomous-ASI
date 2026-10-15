import { cloudAccessAPI, openRouterAPI } from '../services/api';
import {
  CLOUD_ACCESS_PROVIDERS,
  getCloudAccessProvider,
  isCloudProviderAvailable,
} from './oauthProviders';

const noop = () => {};
const alwaysCurrent = () => true;

function applyIfCurrent(shouldApply, setter, ...args) {
  if (shouldApply()) {
    setter(...args);
  }
}

export const EMPTY_CLOUD_PROVIDER_STATE = Object.freeze({ configured: false, models: [], error: '' });

function connectedLabel(provider) {
  return provider.authKind === 'api_key'
    ? `${provider.label} API key is saved`
    : `${provider.loginLabel} is connected`;
}

// Load one cloud provider's credential status and, when configured, its model list.
export async function loadCloudProviderState(providerId, { logContext = 'settings' } = {}) {
  const provider = getCloudAccessProvider(providerId);
  if (!provider) {
    return EMPTY_CLOUD_PROVIDER_STATE;
  }
  let configured = false;
  try {
    const status = await cloudAccessAPI.getProviderStatus(providerId);
    configured = Boolean(status.status?.configured);
  } catch (error) {
    console.error(`Failed to check ${provider.label} status for ${logContext}:`, error);
    return {
      configured: false,
      models: [],
      error: `${provider.loginLabel} status could not be checked: ${error.message || 'unknown error'}.`,
    };
  }
  if (!configured) {
    return EMPTY_CLOUD_PROVIDER_STATE;
  }
  try {
    const result = await cloudAccessAPI.getProviderModels(providerId);
    const models = result.models || [];
    return { configured: true, models, error: models.length > 0 ? '' : provider.noModelsMessage };
  } catch (error) {
    console.error(`Failed to fetch ${provider.label} models for ${logContext}:`, error);
    return {
      configured: true,
      models: [],
      error: `${connectedLabel(provider)}, but models could not be loaded: ${error.message || 'unknown error'}.`,
    };
  }
}

// Refresh every registry provider; unavailable providers are reset without requests.
export async function refreshCloudProviderStates({
  capabilities,
  setCloudProviderState = noop,
  shouldApply = alwaysCurrent,
  logContext = 'settings',
} = {}) {
  await Promise.all(CLOUD_ACCESS_PROVIDERS.map(async (provider) => {
    if (!isCloudProviderAvailable(capabilities, provider.id)) {
      applyIfCurrent(shouldApply, setCloudProviderState, provider.id, EMPTY_CLOUD_PROVIDER_STATE);
      return;
    }
    const state = await loadCloudProviderState(provider.id, { logContext });
    applyIfCurrent(shouldApply, setCloudProviderState, provider.id, state);
  }));
}

export async function refreshOpenRouterCredentialState({
  freeOnly = false,
  setHasOpenRouterKey = noop,
  setOpenRouterModels = noop,
  setLoadingOpenRouter = noop,
  shouldApply = alwaysCurrent,
  logContext = 'settings',
} = {}) {
  try {
    const status = await openRouterAPI.getApiKeyStatus();
    if (!shouldApply()) return;
    const configured = Boolean(status.has_key);
    applyIfCurrent(shouldApply, setHasOpenRouterKey, configured);
    if (configured) {
      applyIfCurrent(shouldApply, setLoadingOpenRouter, true);
      try {
        const result = await openRouterAPI.getModels(null, freeOnly);
        applyIfCurrent(shouldApply, setOpenRouterModels, result.models || []);
      } catch (error) {
        if (!shouldApply()) return;
        console.error(`Failed to refresh OpenRouter models for ${logContext}:`, error);
      } finally {
        applyIfCurrent(shouldApply, setLoadingOpenRouter, false);
      }
    } else {
      applyIfCurrent(shouldApply, setOpenRouterModels, []);
    }
  } catch (error) {
    if (!shouldApply()) return;
    console.error(`Failed to refresh OpenRouter state for ${logContext}:`, error);
    applyIfCurrent(shouldApply, setHasOpenRouterKey, false);
    applyIfCurrent(shouldApply, setOpenRouterModels, []);
    applyIfCurrent(shouldApply, setLoadingOpenRouter, false);
  }
}
