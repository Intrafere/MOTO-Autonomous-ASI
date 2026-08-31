import React from 'react';
import { fireEvent, render, screen, waitFor } from '@testing-library/react';
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

test('keeps configured models visible while running when discovery returns no models', async () => {
  const settings = normalizeLeanOJSettings();
  settings.roles.final_solver = {
    ...settings.roles.final_solver,
    provider: 'openrouter',
    modelId: 'openrouter/configured-final',
  };
  openRouterAPI.getModels.mockResolvedValueOnce({ models: [] });

  render(
    <LeanOJSettings
      settings={settings}
      onSettingsChange={vi.fn()}
      capabilities={{ lmStudioEnabled: true, genericMode: false }}
      connectivityStatus={{
        skills: {
          agent_conversation_memory: {
            enabled: true,
          },
        },
      }}
      isRunning
    />
  );

  const configuredOption = await screen.findByRole('option', {
    name: 'openrouter/configured-final (configured)',
  });
  expect(configuredOption.parentElement).toHaveValue('openrouter/configured-final');
  expect(configuredOption.parentElement).toBeDisabled();
});

test('clears a stale model when switching between LM Studio and OpenRouter', async () => {
  const settings = normalizeLeanOJSettings();
  settings.roles.final_solver = {
    ...settings.roles.final_solver,
    provider: 'lm_studio',
    modelId: 'lm-final',
  };
  const onSettingsChange = vi.fn();

  render(
    <LeanOJSettings
      settings={settings}
      onSettingsChange={onSettingsChange}
      capabilities={{ lmStudioEnabled: true, genericMode: false }}
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

  const finalSolverSection = screen.getByText('Final Proof Solver').closest('.submitter-config-section');
  await waitFor(() => {
    expect(finalSolverSection.querySelector('button[title="Use OpenRouter"]')).toBeEnabled();
  });
  fireEvent.click(finalSolverSection.querySelector('button[title="Use OpenRouter"]'));

  expect(onSettingsChange).toHaveBeenLastCalledWith(
    expect.objectContaining({
      roles: expect.objectContaining({
        final_solver: expect.objectContaining({
          provider: 'openrouter',
          modelId: '',
          openrouterProvider: null,
        }),
      }),
    })
  );
});

test('normalizes stale LM Studio configurations in hosted OpenRouter-only mode', async () => {
  const settings = normalizeLeanOJSettings();
  settings.submitterConfigs[0] = {
    ...settings.submitterConfigs[0],
    provider: 'lm_studio',
    modelId: 'lm-submitter',
  };
  settings.roles.final_solver = {
    ...settings.roles.final_solver,
    provider: 'lm_studio',
    modelId: 'lm-final',
  };
  const onSettingsChange = vi.fn();

  render(
    <LeanOJSettings
      settings={settings}
      onSettingsChange={onSettingsChange}
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
    expect(onSettingsChange).toHaveBeenCalledWith(
      expect.objectContaining({
        submitterConfigs: expect.arrayContaining([
          expect.objectContaining({
            provider: 'openrouter',
            modelId: '',
          }),
        ]),
        roles: expect.objectContaining({
          final_solver: expect.objectContaining({
            provider: 'openrouter',
            modelId: '',
          }),
        }),
      })
    );
  });
});

test('re-clicking the active provider preserves model routing settings', async () => {
  const settings = normalizeLeanOJSettings();
  settings.roles.final_solver = {
    ...settings.roles.final_solver,
    provider: 'openrouter',
    modelId: 'openrouter/final',
    openrouterProvider: 'ExplicitHost',
    openrouterReasoningEffort: 'medium',
  };
  const onSettingsChange = vi.fn();

  render(
    <LeanOJSettings
      settings={settings}
      onSettingsChange={onSettingsChange}
      capabilities={{ lmStudioEnabled: true, genericMode: false }}
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

  const finalSolverSection = screen.getByText('Final Proof Solver').closest('.submitter-config-section');
  await waitFor(() => {
    expect(finalSolverSection.querySelector('button[title="Use OpenRouter"]')).toBeEnabled();
  });
  onSettingsChange.mockClear();
  fireEvent.click(finalSolverSection.querySelector('button[title="Use OpenRouter"]'));

  expect(onSettingsChange).not.toHaveBeenCalled();
});
