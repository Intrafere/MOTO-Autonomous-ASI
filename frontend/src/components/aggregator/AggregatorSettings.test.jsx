import React, { useState } from 'react';
import { render, screen, waitFor } from '@testing-library/react';
import AggregatorSettings from './AggregatorSettings';
import { api, cloudAccessAPI, openRouterAPI } from '../../services/api';

vi.mock('../../services/api', () => ({
  api: {
    getModels: vi.fn(),
  },
  cloudAccessAPI: {
    getOpenAICodexStatus: vi.fn(),
    getXAIGrokStatus: vi.fn(),
    getSakanaFuguStatus: vi.fn(),
    getOpenAICodexModels: vi.fn(),
    getXAIGrokModels: vi.fn(),
    getSakanaFuguModels: vi.fn(),
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
  userPrompt: 'Aggregate.',
  validatorProvider: 'openrouter',
  validatorModel: 'openrouter/validator',
  validatorOpenrouterProvider: 'ValidatorHost',
  validatorOpenrouterReasoningEffort: 'xhigh',
  validatorLmStudioFallback: 'validator-fallback',
  validatorContextSize: 7777,
  validatorMaxOutput: 777,
  submitterConfigs: [
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
  initialConfig = baseConfig,
  lmStudioEnabled = true,
  memoryEnabled = true,
  capabilities: capabilityOverrides = {},
} = {}) {
  let observedConfig = initialConfig;

  function Harness() {
    const [config, setConfigState] = useState(initialConfig);
    observedConfig = config;
    const setConfig = (updater) => {
      setConfigState((previous) => {
        const next = typeof updater === 'function' ? updater(previous) : updater;
        observedConfig = next;
        return next;
      });
    };

    return (
      <AggregatorSettings
        config={config}
        setConfig={setConfig}
        capabilities={{
          lmStudioEnabled,
          genericMode: !lmStudioEnabled,
          ...capabilityOverrides,
        }}
        connectivityStatus={{
          skills: {
            agent_conversation_memory: {
              enabled: memoryEnabled,
            },
          },
        }}
      />
    );
  }

  return {
    ...render(<Harness />),
    getObservedConfig: () => observedConfig,
  };
}

beforeEach(() => {
  vi.clearAllMocks();
  localStorage.clear();
  openRouterAPI.getApiKeyStatus.mockResolvedValue({ has_key: true });
  openRouterAPI.getFreeModelSettings.mockResolvedValue({
    looping_enabled: false,
    auto_selector_enabled: false,
  });
  openRouterAPI.getModels.mockResolvedValue({
    models: [
      { id: 'openrouter/validator', name: 'OpenRouter Validator', context_length: 65536 },
      { id: 'openrouter/assistant', name: 'OpenRouter Assistant', context_length: 65536 },
      { id: 'openrouter/submitter', name: 'OpenRouter Submitter', context_length: 65536 },
    ],
  });
  openRouterAPI.getProviders.mockResolvedValue({ providers: [], endpoints: [] });
  api.getModels.mockResolvedValue({ models: [{ id: 'lm-validator' }, { id: 'lm-assistant' }] });
  cloudAccessAPI.getOpenAICodexStatus.mockResolvedValue({ status: { configured: false } });
  cloudAccessAPI.getXAIGrokStatus.mockResolvedValue({ status: { configured: false } });
  cloudAccessAPI.getSakanaFuguStatus.mockResolvedValue({ status: { configured: false } });
  cloudAccessAPI.getOpenAICodexModels.mockResolvedValue({ models: [] });
  cloudAccessAPI.getXAIGrokModels.mockResolvedValue({ models: [] });
  cloudAccessAPI.getSakanaFuguModels.mockResolvedValue({ models: [] });
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

test('persists omitted Assistant settings from Validator and preserves explicit Assistant settings', async () => {
  const { unmount } = renderSettings();

  await waitFor(() => {
    const saved = JSON.parse(localStorage.getItem('aggregator_settings'));
    expect(saved.assistantProvider).toBe('openrouter');
    expect(saved.assistantModel).toBe('openrouter/validator');
    expect(saved.assistantOpenrouterReasoningEffort).toBe('xhigh');
    expect(saved.assistantContextSize).toBe(7777);
    expect(saved.assistantMaxOutput).toBe(777);
  });

  unmount();
  localStorage.clear();

  renderSettings({
    initialConfig: {
      ...baseConfig,
      assistantProvider: 'openrouter',
      assistantModel: 'openrouter/assistant',
      assistantOpenrouterProvider: 'AssistantHost',
      assistantOpenrouterReasoningEffort: 'medium',
      assistantContextSize: 9999,
      assistantMaxOutput: 999,
    },
  });

  await waitFor(() => {
    const saved = JSON.parse(localStorage.getItem('aggregator_settings'));
    expect(saved.assistantModel).toBe('openrouter/assistant');
    expect(saved.assistantOpenrouterProvider).toBe('AssistantHost');
    expect(saved.assistantOpenrouterReasoningEffort).toBe('medium');
    expect(saved.assistantContextSize).toBe(9999);
    expect(saved.assistantMaxOutput).toBe(999);
  });
});

test('normalizes Assistant to OpenRouter when LM Studio is unavailable', async () => {
  const { getObservedConfig } = renderSettings({
    lmStudioEnabled: false,
    initialConfig: {
      ...baseConfig,
      assistantProvider: 'lm_studio',
      assistantModel: 'lm-assistant',
      assistantLmStudioFallback: 'lm-fallback',
    },
  });

  await waitFor(() => {
    const config = getObservedConfig();
    expect(config.assistantProvider).toBe('openrouter');
    expect(config.assistantModel).toBe('');
    expect(config.assistantLmStudioFallback).toBeNull();
  });
});

test('does not call desktop OAuth endpoints in hosted OpenRouter-only mode', async () => {
  renderSettings({ lmStudioEnabled: false });

  await waitFor(() => {
    expect(openRouterAPI.getApiKeyStatus).toHaveBeenCalled();
  });
  expect(cloudAccessAPI.getOpenAICodexStatus).not.toHaveBeenCalled();
  expect(cloudAccessAPI.getXAIGrokStatus).not.toHaveBeenCalled();
  expect(cloudAccessAPI.getSakanaFuguStatus).not.toHaveBeenCalled();
  expect(cloudAccessAPI.getOpenAICodexModels).not.toHaveBeenCalled();
  expect(cloudAccessAPI.getXAIGrokModels).not.toHaveBeenCalled();
  expect(cloudAccessAPI.getSakanaFuguModels).not.toHaveBeenCalled();
});

test('loads Sakana Fugu models when desktop capability is enabled and key is configured', async () => {
  cloudAccessAPI.getSakanaFuguStatus.mockResolvedValue({ status: { configured: true } });
  cloudAccessAPI.getSakanaFuguModels.mockResolvedValue({
    models: [{ id: 'fugu', name: 'Fugu', context_length: 1000000, max_output_tokens: 100000 }],
  });

  renderSettings({
    capabilities: { lmStudioEnabled: true, genericMode: false, sakanaFuguAvailable: true },
  });

  await waitFor(() => {
    expect(cloudAccessAPI.getSakanaFuguStatus).toHaveBeenCalled();
    expect(cloudAccessAPI.getSakanaFuguModels).toHaveBeenCalled();
  });
});
