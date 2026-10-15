import {
  refreshCloudProviderStates,
  refreshOpenRouterCredentialState,
} from './credentialProviderRefresh';
import { cloudAccessAPI, openRouterAPI } from '../services/api';
import { CLOUD_ACCESS_PROVIDERS } from './oauthProviders';

vi.mock('../services/api', () => ({
  cloudAccessAPI: {
    getProviderStatus: vi.fn(),
    getProviderModels: vi.fn(),
  },
  openRouterAPI: {
    getApiKeyStatus: vi.fn(),
    getModels: vi.fn(),
  },
}));

const DESKTOP = { genericMode: false };

function statesFrom(setCloudProviderState) {
  return Object.fromEntries(setCloudProviderState.mock.calls.map(([providerId, state]) => [providerId, state]));
}

beforeEach(() => {
  vi.clearAllMocks();
  vi.spyOn(console, 'error').mockImplementation(() => {});
  cloudAccessAPI.getProviderStatus.mockResolvedValue({ status: { configured: false } });
  cloudAccessAPI.getProviderModels.mockResolvedValue({ models: [] });
});

afterEach(() => {
  vi.restoreAllMocks();
});

test('keeps OpenRouter enabled when model refresh fails after a valid key check', async () => {
  const setHasOpenRouterKey = vi.fn();
  const setOpenRouterModels = vi.fn();
  const setLoadingOpenRouter = vi.fn();

  openRouterAPI.getApiKeyStatus.mockResolvedValue({ has_key: true });
  openRouterAPI.getModels.mockRejectedValue(new Error('models unavailable'));

  await refreshOpenRouterCredentialState({
    setHasOpenRouterKey,
    setOpenRouterModels,
    setLoadingOpenRouter,
  });

  expect(setHasOpenRouterKey).toHaveBeenCalledWith(true);
  expect(setOpenRouterModels).not.toHaveBeenCalled();
  expect(setLoadingOpenRouter).toHaveBeenNthCalledWith(1, true);
  expect(setLoadingOpenRouter).toHaveBeenLastCalledWith(false);
});

test('refreshes every registry provider and keeps a configured OAuth provider when models fail', async () => {
  const setCloudProviderState = vi.fn();
  cloudAccessAPI.getProviderStatus.mockImplementation(async (providerId) => ({
    status: { configured: providerId === 'openai_codex_oauth' },
  }));
  cloudAccessAPI.getProviderModels.mockRejectedValue(new Error('oauth model error'));

  await refreshCloudProviderStates({ capabilities: DESKTOP, setCloudProviderState });

  const states = statesFrom(setCloudProviderState);
  expect(Object.keys(states).sort()).toEqual(CLOUD_ACCESS_PROVIDERS.map((provider) => provider.id).sort());
  expect(states.openai_codex_oauth.configured).toBe(true);
  expect(states.openai_codex_oauth.models).toEqual([]);
  expect(states.openai_codex_oauth.error).toContain('OpenAI Codex OAuth is connected, but models could not be loaded');
  expect(states.kimi_code).toEqual({ configured: false, models: [], error: '' });
});

test('keeps configured subscription provider enabled when no models are returned', async () => {
  const setCloudProviderState = vi.fn();
  cloudAccessAPI.getProviderStatus.mockImplementation(async (providerId) => ({
    status: { configured: providerId === 'sakana_fugu' },
  }));

  await refreshCloudProviderStates({ capabilities: DESKTOP, setCloudProviderState });

  const sakana = statesFrom(setCloudProviderState).sakana_fugu;
  expect(sakana.configured).toBe(true);
  expect(sakana.models).toEqual([]);
  expect(sakana.error).toContain('no Fugu models were returned');
});

test('hosted mode resets every provider without requests', async () => {
  const setCloudProviderState = vi.fn();

  await refreshCloudProviderStates({ capabilities: { genericMode: true }, setCloudProviderState });

  expect(cloudAccessAPI.getProviderStatus).not.toHaveBeenCalled();
  expect(Object.values(statesFrom(setCloudProviderState)).every((state) => !state.configured)).toBe(true);
});

test('a provider reported unavailable is reset without requests', async () => {
  const setCloudProviderState = vi.fn();

  await refreshCloudProviderStates({
    capabilities: { genericMode: false, kimiCodeAvailable: false },
    setCloudProviderState,
  });

  expect(cloudAccessAPI.getProviderStatus).not.toHaveBeenCalledWith('kimi_code');
  expect(cloudAccessAPI.getProviderStatus).toHaveBeenCalledWith('sakana_fugu');
});

test('does not update state after shouldApply returns false', async () => {
  const setHasOpenRouterKey = vi.fn();
  const setOpenRouterModels = vi.fn();
  let current = true;

  openRouterAPI.getApiKeyStatus.mockImplementation(async () => {
    current = false;
    return { has_key: true };
  });

  await refreshOpenRouterCredentialState({
    setHasOpenRouterKey,
    setOpenRouterModels,
    shouldApply: () => current,
  });

  expect(setHasOpenRouterKey).not.toHaveBeenCalled();
  expect(setOpenRouterModels).not.toHaveBeenCalled();
});
