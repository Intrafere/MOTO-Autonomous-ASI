import { useCallback, useEffect, useMemo, useState } from 'react';
import { EMPTY_CLOUD_PROVIDER_STATE, refreshCloudProviderStates } from '../utils/credentialProviderRefresh';
import {
  CLOUD_ACCESS_PROVIDERS,
  getConfiguredCloudAccessProviders,
  isCloudProviderAvailable,
} from '../utils/oauthProviders';

function initialStates() {
  return Object.fromEntries(CLOUD_ACCESS_PROVIDERS.map((provider) => [provider.id, EMPTY_CLOUD_PROVIDER_STATE]));
}

// Read-only view over per-provider { configured, models, error } states that
// settings panels and model selectors consume.
export function buildCloudProvidersView(states = {}) {
  const statusByProvider = {};
  const modelsByProvider = {};
  const modelErrors = [];
  for (const provider of CLOUD_ACCESS_PROVIDERS) {
    const state = { ...EMPTY_CLOUD_PROVIDER_STATE, ...(states[provider.id] || {}) };
    statusByProvider[provider.id] = { configured: Boolean(state.configured) };
    modelsByProvider[provider.id] = state.models || [];
    if (state.error) {
      modelErrors.push({ providerId: provider.id, label: provider.label, message: state.error });
    }
  }
  return {
    states,
    statusByProvider,
    modelsByProvider,
    modelErrors,
    configuredProviders: getConfiguredCloudAccessProviders(statusByProvider),
    isConfigured: (providerId) => Boolean(statusByProvider[providerId]?.configured),
    modelsFor: (providerId) => modelsByProvider[providerId] || [],
  };
}

// Credential status, models and model errors for every registry cloud provider.
// Loads on mount, whenever provider availability changes, and on each credential
// refresh token so saving or removing a key updates mounted settings panels.
export default function useCloudAccessProviders({
  capabilities,
  refreshToken = 0,
  logContext = 'settings',
} = {}) {
  const [states, setStates] = useState(initialStates);
  const availabilityKey = CLOUD_ACCESS_PROVIDERS
    .map((provider) => (isCloudProviderAvailable(capabilities, provider.id) ? '1' : '0'))
    .join('');

  const setCloudProviderState = useCallback((providerId, state) => {
    setStates((previous) => ({ ...previous, [providerId]: { ...EMPTY_CLOUD_PROVIDER_STATE, ...state } }));
  }, []);

  useEffect(() => {
    let isCurrent = true;
    refreshCloudProviderStates({
      capabilities,
      setCloudProviderState,
      shouldApply: () => isCurrent,
      logContext,
    });
    return () => {
      isCurrent = false;
    };
  }, [availabilityKey, refreshToken, logContext, setCloudProviderState]); // eslint-disable-line react-hooks/exhaustive-deps

  return useMemo(() => buildCloudProvidersView(states), [states]);
}
