import { refreshCredentialProviderState } from './credentialProviderRefresh';
import { cloudAccessAPI, openRouterAPI } from '../services/api';

vi.mock('../services/api', () => ({
  cloudAccessAPI: {
    getOpenAICodexStatus: vi.fn(),
    getOpenAICodexModels: vi.fn(),
    getXAIGrokStatus: vi.fn(),
    getXAIGrokModels: vi.fn(),
    getSakanaFuguStatus: vi.fn(),
    getSakanaFuguModels: vi.fn(),
  },
  openRouterAPI: {
    getApiKeyStatus: vi.fn(),
    getModels: vi.fn(),
  },
}));

beforeEach(() => {
  vi.clearAllMocks();
  vi.spyOn(console, 'error').mockImplementation(() => {});
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

  await refreshCredentialProviderState({
    setHasOpenRouterKey,
    setOpenRouterModels,
    setLoadingOpenRouter,
  });

  expect(setHasOpenRouterKey).toHaveBeenCalledWith(true);
  expect(setOpenRouterModels).not.toHaveBeenCalled();
  expect(setLoadingOpenRouter).toHaveBeenNthCalledWith(1, true);
  expect(setLoadingOpenRouter).toHaveBeenLastCalledWith(false);
});

test('keeps configured OAuth provider enabled when model loading fails', async () => {
  const setHasOpenAICodexLogin = vi.fn();
  const setOpenAICodexModels = vi.fn();
  const setOpenAICodexModelError = vi.fn();

  openRouterAPI.getApiKeyStatus.mockResolvedValue({ has_key: false });
  cloudAccessAPI.getOpenAICodexStatus.mockResolvedValue({ status: { configured: true } });
  cloudAccessAPI.getOpenAICodexModels.mockRejectedValue(new Error('oauth model error'));

  await refreshCredentialProviderState({
    openAICodexOauthAvailable: true,
    setHasOpenAICodexLogin,
    setOpenAICodexModels,
    setOpenAICodexModelError,
  });

  expect(setHasOpenAICodexLogin).toHaveBeenCalledWith(true);
  expect(setOpenAICodexModels).toHaveBeenCalledWith([]);
  expect(setOpenAICodexModelError).toHaveBeenCalledWith(
    expect.stringContaining('OpenAI Codex OAuth models could not be loaded')
  );
});

test('keeps configured subscription provider enabled when no models are returned', async () => {
  const setHasSakanaFuguKey = vi.fn();
  const setSakanaFuguModels = vi.fn();
  const setSakanaFuguModelError = vi.fn();

  openRouterAPI.getApiKeyStatus.mockResolvedValue({ has_key: false });
  cloudAccessAPI.getSakanaFuguStatus.mockResolvedValue({ status: { configured: true } });
  cloudAccessAPI.getSakanaFuguModels.mockResolvedValue({ models: [] });

  await refreshCredentialProviderState({
    sakanaFuguAvailable: true,
    setHasSakanaFuguKey,
    setSakanaFuguModels,
    setSakanaFuguModelError,
  });

  expect(setHasSakanaFuguKey).toHaveBeenCalledWith(true);
  expect(setHasSakanaFuguKey).not.toHaveBeenCalledWith(false);
  expect(setSakanaFuguModels).toHaveBeenCalledWith([]);
  expect(setSakanaFuguModelError).toHaveBeenCalledWith(
    expect.stringContaining('no Fugu models were returned')
  );
});

test('does not update state after shouldApply returns false', async () => {
  const setHasOpenRouterKey = vi.fn();
  const setOpenRouterModels = vi.fn();
  let current = true;

  openRouterAPI.getApiKeyStatus.mockImplementation(async () => {
    current = false;
    return { has_key: true };
  });

  await refreshCredentialProviderState({
    setHasOpenRouterKey,
    setOpenRouterModels,
    shouldApply: () => current,
  });

  expect(setHasOpenRouterKey).not.toHaveBeenCalled();
  expect(setOpenRouterModels).not.toHaveBeenCalled();
});
