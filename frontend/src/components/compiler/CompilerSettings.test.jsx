import React from 'react';
import { render, screen, waitFor } from '@testing-library/react';
import CompilerSettings from './CompilerSettings';
import { cloudAccessAPI, openRouterAPI, api, compilerAPI } from '../../services/api';

vi.mock('../../services/api', () => ({
  api: {
    getModels: vi.fn(),
  },
  aggregatorAPI: {
    getSettings: vi.fn(),
  },
  compilerAPI: {
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

function renderSettings({
  lmStudioEnabled = true,
  memoryEnabled = true,
  developerModeEnabled = false,
} = {}) {
  return render(
    <CompilerSettings
      capabilities={{ lmStudioEnabled, genericMode: !lmStudioEnabled }}
      connectivityStatus={{
        skills: {
          agent_conversation_memory: {
            enabled: memoryEnabled,
          },
        },
      }}
      developerModeEnabled={developerModeEnabled}
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
      {
        id: 'openrouter/validator',
        name: 'OpenRouter Validator',
        context_length: 65536,
      },
      {
        id: 'openrouter/assistant',
        name: 'OpenRouter Assistant',
        context_length: 65536,
      },
    ],
  });
  openRouterAPI.getProviders.mockResolvedValue({ providers: [], endpoints: [] });
  api.getModels.mockResolvedValue({ models: [{ id: 'lm-validator' }] });
  cloudAccessAPI.getOpenAICodexStatus.mockResolvedValue({ status: { configured: false } });
  cloudAccessAPI.getXAIGrokStatus.mockResolvedValue({ status: { configured: false } });
  cloudAccessAPI.getOpenAICodexModels.mockResolvedValue({ models: [] });
  cloudAccessAPI.getXAIGrokModels.mockResolvedValue({ models: [] });
  compilerAPI.getDefaultCritiquePrompt.mockResolvedValue({
    data: {
      prompt: 'Default critique prompt.',
    },
  });
});

test('renders Assistant role and greys it out when Session History Memory is disabled', async () => {
  renderSettings({ memoryEnabled: false });

  expect(await screen.findByText('Assistant')).toBeInTheDocument();
  expect(
    screen.getByText(/Assistant requires Session History Memory/i)
  ).toBeInTheDocument();
  const assistantSection = screen.getByText('Assistant').closest('.submitter-config-section');
  expect(assistantSection).toHaveAttribute('aria-disabled', 'true');
  expect(assistantSection.querySelector('select')).toBeDisabled();
  expect(assistantSection.querySelector('input')).toBeDisabled();
});

test('hydrates omitted Assistant settings from Validator and preserves explicit Assistant settings', async () => {
  localStorage.setItem(
    'compiler_settings',
    JSON.stringify({
      validatorProvider: 'openrouter',
      validatorModel: 'openrouter/validator',
      validatorOpenrouterProvider: 'ValidatorHost',
      validatorOpenrouterReasoningEffort: 'xhigh',
      validatorContextSize: 7777,
      validatorMaxOutput: 777,
      validatorSuperchargeEnabled: true,
      writerProvider: 'openrouter',
      writerModel: 'openrouter/writer',
      writerContextSize: 8192,
      writerMaxOutput: 1024,
      highParamProvider: 'openrouter',
      highParamModel: 'openrouter/rigor',
      highParamContextSize: 8192,
      highParamMaxOutput: 1024,
      critiqueSubmitterProvider: 'openrouter',
      critiqueSubmitterModel: 'openrouter/critique',
      critiqueSubmitterContextSize: 8192,
      critiqueSubmitterMaxOutput: 1024,
    })
  );

  const { unmount } = renderSettings();

  await waitFor(() => {
    const saved = JSON.parse(localStorage.getItem('compiler_settings'));
    expect(saved.assistantProvider).toBe('openrouter');
    expect(saved.assistantModel).toBe('openrouter/validator');
    expect(saved.assistantOpenrouterProvider).toBe('ValidatorHost');
    expect(saved.assistantOpenrouterReasoningEffort).toBe('xhigh');
    expect(saved.assistantContextSize).toBe(7777);
    expect(saved.assistantMaxOutput).toBe(777);
    expect(saved.assistantSuperchargeEnabled).toBe(true);
  });

  unmount();
  localStorage.clear();
  localStorage.setItem(
    'compiler_settings',
    JSON.stringify({
      validatorProvider: 'openrouter',
      validatorModel: 'openrouter/validator',
      assistantProvider: 'openrouter',
      assistantModel: 'openrouter/assistant',
      assistantOpenrouterProvider: 'AssistantHost',
      assistantOpenrouterReasoningEffort: 'medium',
      assistantContextSize: 9999,
      assistantMaxOutput: 999,
      assistantSuperchargeEnabled: false,
      writerProvider: 'openrouter',
      writerModel: 'openrouter/writer',
      writerContextSize: 8192,
      writerMaxOutput: 1024,
      highParamProvider: 'openrouter',
      highParamModel: 'openrouter/rigor',
      highParamContextSize: 8192,
      highParamMaxOutput: 1024,
      critiqueSubmitterProvider: 'openrouter',
      critiqueSubmitterModel: 'openrouter/critique',
      critiqueSubmitterContextSize: 8192,
      critiqueSubmitterMaxOutput: 1024,
    })
  );

  renderSettings();

  await waitFor(() => {
    const saved = JSON.parse(localStorage.getItem('compiler_settings'));
    expect(saved.assistantModel).toBe('openrouter/assistant');
    expect(saved.assistantOpenrouterProvider).toBe('AssistantHost');
    expect(saved.assistantOpenrouterReasoningEffort).toBe('medium');
    expect(saved.assistantContextSize).toBe(9999);
    expect(saved.assistantMaxOutput).toBe(999);
  });
});

test('normalizes Assistant to OpenRouter when LM Studio is unavailable', async () => {
  localStorage.setItem(
    'compiler_settings',
    JSON.stringify({
      validatorProvider: 'openrouter',
      validatorModel: 'openrouter/validator',
      assistantProvider: 'lm_studio',
      assistantModel: 'lm-assistant',
      assistantContextSize: 4096,
      assistantMaxOutput: 512,
      writerProvider: 'openrouter',
      writerModel: 'openrouter/writer',
      writerContextSize: 8192,
      writerMaxOutput: 1024,
      highParamProvider: 'openrouter',
      highParamModel: 'openrouter/rigor',
      highParamContextSize: 8192,
      highParamMaxOutput: 1024,
      critiqueSubmitterProvider: 'openrouter',
      critiqueSubmitterModel: 'openrouter/critique',
      critiqueSubmitterContextSize: 8192,
      critiqueSubmitterMaxOutput: 1024,
    })
  );

  renderSettings({ lmStudioEnabled: false });

  await waitFor(() => {
    const saved = JSON.parse(localStorage.getItem('compiler_settings'));
    expect(saved.assistantProvider).toBe('openrouter');
    expect(saved.assistantModel).toBe('');
    expect(saved.assistantLmStudioFallback).toBeNull();
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

test('migrates legacy saved compiler writer settings into Writing Submitter fields', async () => {
  const legacyPrefix = ['high', 'Context'].join('');
  localStorage.setItem(
    'compiler_settings',
    JSON.stringify({
      validatorProvider: 'openrouter',
      validatorModel: 'openrouter/validator',
      writerProvider: 'openrouter',
      writerModel: '',
      writerContextSize: 0,
      writerMaxOutput: 0,
      [`${legacyPrefix}Provider`]: 'openrouter',
      [`${legacyPrefix}Model`]: 'openrouter/legacy-writer',
      [`${legacyPrefix}ContextSize`]: 12345,
      [`${legacyPrefix}MaxOutput`]: 1234,
      highParamProvider: 'openrouter',
      highParamModel: 'openrouter/rigor',
      highParamContextSize: 8192,
      highParamMaxOutput: 1024,
    })
  );

  renderSettings();

  await waitFor(() => {
    const saved = JSON.parse(localStorage.getItem('compiler_settings'));
    expect(saved.writerProvider).toBe('openrouter');
    expect(saved.writerModel).toBe('openrouter/legacy-writer');
    expect(saved.writerContextSize).toBe(12345);
    expect(saved.writerMaxOutput).toBe(1234);
  });
});
