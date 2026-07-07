import React from 'react';
import { fireEvent, render, screen, waitFor } from '@testing-library/react';
import AutonomousResearchSettings from './AutonomousResearchSettings';
import { api, autonomousAPI, cloudAccessAPI, openRouterAPI } from '../../services/api';

vi.mock('../../services/api', () => ({
  api: {
    getModels: vi.fn(),
  },
  autonomousAPI: {
    getProofStatus: vi.fn(),
    getDefaultCritiquePrompt: vi.fn(),
  },
  cloudAccessAPI: {
    getOpenAICodexStatus: vi.fn(),
    getXAIGrokStatus: vi.fn(),
    getOpenAICodexModels: vi.fn(),
    getXAIGrokModels: vi.fn(),
  },
  openRouterAPI: {
    getApiKeyStatus: vi.fn(),
    getFreeModelSettings: vi.fn(),
    getModels: vi.fn(),
    getProviders: vi.fn(),
    setFreeModelSettings: vi.fn(),
  },
}));

const baseConfig = {
  validator_provider: 'openrouter',
  validator_model: 'openrouter/validator',
  validator_context_window: 7777,
  validator_max_tokens: 777,
  assistant_provider: 'openrouter',
  assistant_model: 'openrouter/assistant',
  assistant_context_window: 9999,
  assistant_max_tokens: 999,
  writer_provider: 'openrouter',
  writer_model: 'openrouter/writer',
  writer_context_window: 8192,
  writer_max_tokens: 1024,
  high_param_provider: 'openrouter',
  high_param_model: 'openrouter/rigor',
  high_param_context_window: 8192,
  high_param_max_tokens: 1024,
  submitter_configs: [
    {
      submitterId: 1,
      provider: 'openrouter',
      modelId: 'openrouter/submitter',
      contextWindow: 4096,
      maxOutputTokens: 512,
    },
  ],
};

function renderSettings({
  config = baseConfig,
  lmStudioEnabled = true,
  memoryEnabled = true,
} = {}) {
  return render(
    <AutonomousResearchSettings
      config={config}
      onConfigChange={vi.fn()}
      models={[{ id: 'lm-validator' }]}
      capabilities={{ lmStudioEnabled, genericMode: !lmStudioEnabled }}
      connectivityStatus={{
        skills: {
          agent_conversation_memory: {
            enabled: memoryEnabled,
          },
        },
      }}
      isRunning={false}
    />
  );
}

beforeEach(() => {
  vi.clearAllMocks();
  localStorage.clear();
  openRouterAPI.getApiKeyStatus.mockResolvedValue({ has_key: true });
  openRouterAPI.getFreeModelSettings.mockResolvedValue({
    looping_enabled: true,
    auto_selector_enabled: true,
  });
  openRouterAPI.getModels.mockResolvedValue({
    models: [
      { id: 'openrouter/validator', name: 'OpenRouter Validator', context_length: 65536 },
      { id: 'openrouter/assistant', name: 'OpenRouter Assistant', context_length: 65536 },
      { id: 'openrouter/writer', name: 'OpenRouter Writer', context_length: 65536 },
      { id: 'openrouter/rigor', name: 'OpenRouter Rigor', context_length: 65536 },
      { id: 'openrouter/submitter', name: 'OpenRouter Submitter', context_length: 65536 },
    ],
  });
  openRouterAPI.getProviders.mockResolvedValue({ providers: [], endpoints: [] });
  api.getModels.mockResolvedValue({ models: [{ id: 'lm-validator' }] });
  autonomousAPI.getProofStatus.mockResolvedValue({
    lean4_enabled: false,
    lean4_path: '',
    lean4_proof_timeout: 900,
    lean4_lsp_enabled: false,
    proof_max_parallel_candidates: 6,
    smt_enabled: false,
    smt_timeout: 30,
  });
  autonomousAPI.getDefaultCritiquePrompt.mockResolvedValue({
    data: {
      prompt: 'Default critique prompt.',
    },
  });
  cloudAccessAPI.getOpenAICodexStatus.mockResolvedValue({ status: { configured: false } });
  cloudAccessAPI.getXAIGrokStatus.mockResolvedValue({ status: { configured: false } });
  cloudAccessAPI.getOpenAICodexModels.mockResolvedValue({ models: [] });
  cloudAccessAPI.getXAIGrokModels.mockResolvedValue({ models: [] });
});

test('renders Assistant role and greys it out when Session History Memory is disabled', async () => {
  renderSettings({ memoryEnabled: false });

  expect(
    await screen.findByText(/Assistant requires Session History Memory/i)
  ).toBeInTheDocument();
  const assistantSection = screen
    .getByText(/Assistant requires Session History Memory/i)
    .closest('.submitter-config-section');
  expect(assistantSection).toHaveAttribute('aria-disabled', 'true');
  expect(assistantSection.querySelector('select')).toBeDisabled();
  expect(assistantSection.querySelector('input')).toBeDisabled();
});

test('prefetches OpenRouter provider metadata for Assistant model', async () => {
  renderSettings();

  await waitFor(() => {
    expect(openRouterAPI.getProviders).toHaveBeenCalledWith('openrouter/assistant');
  });
});

test('does not call desktop OAuth endpoints in hosted OpenRouter-only mode', async () => {
  renderSettings({ lmStudioEnabled: false });

  await waitFor(() => {
    expect(openRouterAPI.getApiKeyStatus).toHaveBeenCalled();
  });
  expect(cloudAccessAPI.getOpenAICodexStatus).not.toHaveBeenCalled();
  expect(cloudAccessAPI.getXAIGrokStatus).not.toHaveBeenCalled();
  expect(cloudAccessAPI.getOpenAICodexModels).not.toHaveBeenCalled();
  expect(cloudAccessAPI.getXAIGrokModels).not.toHaveBeenCalled();
});

test('applies a recommended profile even when OpenRouter model list is empty', async () => {
  openRouterAPI.getModels.mockResolvedValueOnce({ models: [] });
  const onConfigChange = vi.fn();

  render(
    <AutonomousResearchSettings
      config={baseConfig}
      onConfigChange={onConfigChange}
      models={[{ id: 'lm-validator' }]}
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

  await waitFor(() => {
    expect(openRouterAPI.getModels).toHaveBeenCalled();
  });

  const profileSelect = screen.getByText('Select Profile').parentElement.querySelector('select');
  fireEvent.change(profileSelect, {
    target: { value: 'recommended_lab_fast_costly_extra_high' },
  });

  await waitFor(() => {
    expect(profileSelect.value).toBe('recommended_lab_fast_costly_extra_high');
  });
  expect(onConfigChange).toHaveBeenCalledWith(
    expect.objectContaining({
      validator_provider: 'openrouter',
    })
  );
});
