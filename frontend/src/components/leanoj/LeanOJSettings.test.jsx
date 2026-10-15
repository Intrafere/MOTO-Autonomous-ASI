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
    getProviderStatus: vi.fn(),
    getProviderModels: vi.fn(),
  },
  openRouterAPI: {
    getApiKeyStatus: vi.fn(),
    getModels: vi.fn(),
    getProviders: vi.fn(),
    setFreeModelSettings: vi.fn(),
  },
}));

const cloudProviderMocks = {};
function mockCloudProvider(providerId, { configured, modelsResponse } = {}) {
  const current = cloudProviderMocks[providerId] || { configured: false, models: [] };
  cloudProviderMocks[providerId] = {
    configured: configured ?? current.configured,
    models: modelsResponse ? (modelsResponse.models || []) : current.models,
  };
  cloudAccessAPI.getProviderStatus.mockImplementation(async (id) => ({
    status: { configured: Boolean(cloudProviderMocks[id]?.configured) },
  }));
  cloudAccessAPI.getProviderModels.mockImplementation(async (id) => ({ models: cloudProviderMocks[id]?.models || [] }));
}

test.each([false, true])('Codex effort covers every visible LeanOJ role with running=%s', async isRunning => {
  mockCloudProvider('openai_codex_oauth', { configured: true });
  mockCloudProvider('openai_codex_oauth', { modelsResponse: { models: [{ id: 'codex', supported_reasoning_levels: ['high', 'max'] }] } });
  const role = { provider: 'openai_codex_oauth', modelId: 'codex', openrouterReasoningEffort: 'max', contextWindow: 400000, maxOutputTokens: 32000 };
  const settings = normalizeLeanOJSettings({ numSubmitters: 1, submitterConfigs: [role], roles: Object.fromEntries(['topic_generator', 'topic_validator', 'brainstorm_validator', 'final_solver', 'assistant'].map(key => [key, role])) });
  render(<LeanOJSettings settings={settings} onSettingsChange={vi.fn()} capabilities={{ lmStudioEnabled: true }} connectivityStatus={{ skills: { agent_conversation_memory: { enabled: true } } }} isRunning={isRunning} />);
  await waitFor(() => expect(screen.getAllByRole('combobox', { name: 'Codex Reasoning Effort' }).length).toBe(4));
  expect(screen.getAllByRole('combobox', { name: 'Codex Reasoning Effort' }).every(node => node.value === 'max')).toBe(true);
  await waitFor(() => expect(screen.getAllByRole('combobox', { name: 'Codex Reasoning Effort' }).every(node => node.disabled === isRunning)).toBe(true));
});

test.each([false, true])('xAI effort covers every visible LeanOJ role with running=%s', async isRunning => {
  mockCloudProvider('xai_grok_oauth', { configured: true });
  mockCloudProvider('xai_grok_oauth', { modelsResponse: { models: [{ id: 'grok-4.3' }] } });
  const role = { provider: 'xai_grok_oauth', modelId: 'grok-4.3', openrouterReasoningEffort: 'low', contextWindow: 1000000, maxOutputTokens: 131072 };
  const settings = normalizeLeanOJSettings({ numSubmitters: 1, submitterConfigs: [role], roles: Object.fromEntries(['topic_generator', 'topic_validator', 'brainstorm_validator', 'final_solver', 'assistant'].map(key => [key, role])) });
  render(<LeanOJSettings settings={settings} onSettingsChange={vi.fn()} capabilities={{ lmStudioEnabled: true }} connectivityStatus={{ skills: { agent_conversation_memory: { enabled: true } } }} isRunning={isRunning} />);
  const controls = await screen.findAllByRole('combobox', { name: 'xAI Reasoning Effort' });
  expect(controls).toHaveLength(4);
  expect(controls.every(node => node.value === 'low' && node.disabled === isRunning)).toBe(true);
});

beforeEach(() => {
  vi.clearAllMocks();
  localStorage.clear();
  openRouterAPI.getApiKeyStatus.mockResolvedValue({ has_key: true });
  openRouterAPI.getModels.mockResolvedValue({
    models: [{ id: 'openrouter/final', name: 'OpenRouter Final', context_length: 65536 }],
  });
  openRouterAPI.getProviders.mockResolvedValue({ providers: [], endpoints: [] });
  api.getModels.mockResolvedValue({ models: [{ id: 'lm-final' }] });
  mockCloudProvider('openai_codex_oauth', { configured: false });
  mockCloudProvider('xai_grok_oauth', { configured: false });
  mockCloudProvider('openai_codex_oauth', { modelsResponse: { models: [] } });
  mockCloudProvider('xai_grok_oauth', { modelsResponse: { models: [] } });
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
  expect(cloudAccessAPI.getProviderStatus).not.toHaveBeenCalled();
  expect(cloudAccessAPI.getProviderModels).not.toHaveBeenCalled();
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
