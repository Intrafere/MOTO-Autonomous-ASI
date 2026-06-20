import React from 'react';
import { render, waitFor } from '@testing-library/react';
import LeanOJSettings from './LeanOJSettings';
import { api, cloudAccessAPI, openRouterAPI } from '../../services/api';
import { normalizeLeanOJSettings } from '../../utils/leanojProfiles';

vi.mock('../../services/api', () => ({
  api: {
    getModels: vi.fn(),
  },
  cloudAccessAPI: {
    getOpenAICodexStatus: vi.fn(),
    getXAIGrokStatus: vi.fn(),
    getOpenAICodexModels: vi.fn(),
    getXAIGrokModels: vi.fn(),
  },
  openRouterAPI: {
    getApiKeyStatus: vi.fn(),
    getModels: vi.fn(),
    getProviders: vi.fn(),
    setFreeModelSettings: vi.fn(),
  },
}));

beforeEach(() => {
  vi.clearAllMocks();
  localStorage.clear();
  openRouterAPI.getApiKeyStatus.mockResolvedValue({ has_key: true });
  openRouterAPI.getModels.mockResolvedValue({
    models: [{ id: 'openrouter/final', name: 'OpenRouter Final', context_length: 65536 }],
  });
  openRouterAPI.getProviders.mockResolvedValue({ providers: [], endpoints: [] });
  api.getModels.mockResolvedValue({ models: [{ id: 'lm-final' }] });
  cloudAccessAPI.getOpenAICodexStatus.mockResolvedValue({ status: { configured: false } });
  cloudAccessAPI.getXAIGrokStatus.mockResolvedValue({ status: { configured: false } });
  cloudAccessAPI.getOpenAICodexModels.mockResolvedValue({ models: [] });
  cloudAccessAPI.getXAIGrokModels.mockResolvedValue({ models: [] });
});

test('does not call desktop OAuth endpoints in hosted OpenRouter-only mode', async () => {
  render(
    <LeanOJSettings
      settings={normalizeLeanOJSettings()}
      onSettingsChange={vi.fn()}
      capabilities={{ lmStudioEnabled: false, genericMode: true }}
      connectivityStatus={{
        skills: {
          agent_conversation_memory: {
            enabled: true,
          },
        },
      }}
      isRunning={false}
    />
  );

  await waitFor(() => {
    expect(openRouterAPI.getApiKeyStatus).toHaveBeenCalled();
  });
  expect(cloudAccessAPI.getOpenAICodexStatus).not.toHaveBeenCalled();
  expect(cloudAccessAPI.getXAIGrokStatus).not.toHaveBeenCalled();
  expect(cloudAccessAPI.getOpenAICodexModels).not.toHaveBeenCalled();
  expect(cloudAccessAPI.getXAIGrokModels).not.toHaveBeenCalled();
});
