import React from 'react';
import { fireEvent, render, screen, waitFor, within } from '@testing-library/react';
import AutonomousResearchSettings from './AutonomousResearchSettings';
import { persistAutonomousSettings } from '../../utils/autonomousProfiles';
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
    getProviderStatus: vi.fn(),
    getProviderModels: vi.fn(),
  },
  openRouterAPI: {
    getApiKeyStatus: vi.fn(),
    getFreeModelSettings: vi.fn(),
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
  isRunning = false,
  onConfigChange = vi.fn(),
} = {}) {
  return {
    ...render(
    <AutonomousResearchSettings
      config={config}
      onConfigChange={onConfigChange}
      models={[{ id: 'lm-validator' }]}
      capabilities={{ lmStudioEnabled, genericMode: !lmStudioEnabled }}
      connectivityStatus={{
        skills: {
          agent_conversation_memory: {
            enabled: memoryEnabled,
          },
        },
      }}
      isRunning={isRunning}
    />
    ),
    onConfigChange,
  };
}

test.each([false, true])('Codex effort covers all Autonomous roles with running=%s', async isRunning => {
  mockCloudProvider('openai_codex_oauth', { configured: true });
  mockCloudProvider('openai_codex_oauth', { modelsResponse: { models: [{ id: 'codex', supported_reasoning_levels: ['medium', 'max'] }] } });
  const config = { ...baseConfig, submitter_configs: [{ ...baseConfig.submitter_configs[0], provider: 'openai_codex_oauth', modelId: 'codex', openrouterReasoningEffort: 'max' }] };
  for (const prefix of ['validator', 'assistant', 'writer', 'high_param']) Object.assign(config, { [`${prefix}_provider`]: 'openai_codex_oauth', [`${prefix}_model`]: 'codex', [`${prefix}_openrouter_reasoning_effort`]: 'max' });
  renderSettings({ config, isRunning });
  await waitFor(() => expect(screen.getAllByRole('combobox', { name: 'Codex Reasoning Effort' })).toHaveLength(5));
  expect(screen.getAllByRole('combobox', { name: 'Codex Reasoning Effort' }).every(node => node.value === 'max')).toBe(true);
  await waitFor(() => expect(screen.getAllByRole('combobox', { name: 'Codex Reasoning Effort' }).every(node => node.disabled === isRunning)).toBe(true));
});

test.each([false, true])('xAI effort covers all Autonomous roles with running=%s', async isRunning => {
  mockCloudProvider('xai_grok_oauth', { configured: true });
  mockCloudProvider('xai_grok_oauth', { modelsResponse: { models: [{ id: 'grok-4.3' }] } });
  const config = { ...baseConfig, submitter_configs: [{ ...baseConfig.submitter_configs[0], provider: 'xai_grok_oauth', modelId: 'grok-4.3', openrouterReasoningEffort: 'medium' }] };
  for (const prefix of ['validator', 'assistant', 'writer', 'high_param']) Object.assign(config, { [`${prefix}_provider`]: 'xai_grok_oauth', [`${prefix}_model`]: 'grok-4.3', [`${prefix}_openrouter_reasoning_effort`]: 'medium' });
  renderSettings({ config, isRunning });
  const controls = await screen.findAllByRole('combobox', { name: 'xAI Reasoning Effort' });
  expect(controls).toHaveLength(5);
  expect(controls.every(node => node.value === 'medium' && node.disabled === isRunning)).toBe(true);
});

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
    lean4_proof_timeout: 2000,
    lean4_lsp_enabled: false,
    proof_max_parallel_candidates: 6,
    smt_enabled: false,
    smt_timeout: 300,
  });
  autonomousAPI.getDefaultCritiquePrompt.mockResolvedValue({
    data: {
      prompt: 'Default critique prompt.',
    },
  });
  mockCloudProvider('openai_codex_oauth', { configured: false });
  mockCloudProvider('xai_grok_oauth', { configured: false });
  mockCloudProvider('sakana_fugu', { configured: false });
  mockCloudProvider('openai_codex_oauth', { modelsResponse: { models: [] } });
  mockCloudProvider('xai_grok_oauth', { modelsResponse: { models: [] } });
  mockCloudProvider('sakana_fugu', { modelsResponse: { models: [] } });
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

test('keeps configured models visible while running when discovery returns no models', async () => {
  openRouterAPI.getModels.mockResolvedValueOnce({ models: [] });
  renderSettings({ isRunning: true });

  const configuredOptions = await screen.findAllByRole('option', {
    name: 'openrouter/validator (configured)',
  });

  expect(configuredOptions.length).toBeGreaterThan(0);
  expect(configuredOptions[0].parentElement).toHaveValue('openrouter/validator');
  expect(configuredOptions[0].parentElement).toBeDisabled();
});

test('does not call desktop OAuth endpoints in hosted OpenRouter-only mode', async () => {
  renderSettings({ lmStudioEnabled: false });

  await waitFor(() => {
    expect(openRouterAPI.getApiKeyStatus).toHaveBeenCalled();
  });
  expect(cloudAccessAPI.getProviderStatus).not.toHaveBeenCalled();
  expect(cloudAccessAPI.getProviderModels).not.toHaveBeenCalled();
});

test('does not invent extra submitters or LM Studio providers for incomplete saved roles', async () => {
  persistAutonomousSettings({
    numSubmitters: 1,
    selectedProfile: '',
    submitterConfigs: [
      {
        submitterId: 1,
        modelId: 'openrouter/submitter',
        contextWindow: 4096,
        maxOutputTokens: 512,
      },
    ],
    localConfig: {
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
    },
  });
  const incompleteConfig = {
    ...baseConfig,
    num_submitters: 1,
    submitter_configs: [
      {
        submitterId: 1,
        modelId: 'openrouter/submitter',
        contextWindow: 4096,
        maxOutputTokens: 512,
      },
    ],
  };

  renderSettings({ config: incompleteConfig });

  await waitFor(() => {
    expect(screen.getByRole('combobox', { name: 'Number of Submitters' })).toHaveValue('1');
  });
  expect(screen.queryByText(/Submitter 2/i)).not.toBeInTheDocument();
  const lmStudioButtons = screen.getAllByRole('button', { name: 'LM Studio' });
  expect(lmStudioButtons[0].className).not.toContain('active-lm');
});

test('keeps an increased submitter count when the new submitter role is edited', async () => {
  const submitterConfigs = [1, 2, 3].map(submitterId => ({
    ...baseConfig.submitter_configs[0],
    submitterId,
  }));
  persistAutonomousSettings({
    numSubmitters: 3,
    submitterConfigs,
    localConfig: {
      validator_provider: baseConfig.validator_provider,
      validator_model: baseConfig.validator_model,
      validator_context_window: baseConfig.validator_context_window,
      validator_max_tokens: baseConfig.validator_max_tokens,
      assistant_provider: baseConfig.assistant_provider,
      assistant_model: baseConfig.assistant_model,
      assistant_context_window: baseConfig.assistant_context_window,
      assistant_max_tokens: baseConfig.assistant_max_tokens,
      writer_provider: baseConfig.writer_provider,
      writer_model: baseConfig.writer_model,
      writer_context_window: baseConfig.writer_context_window,
      writer_max_tokens: baseConfig.writer_max_tokens,
      high_param_provider: baseConfig.high_param_provider,
      high_param_model: baseConfig.high_param_model,
      high_param_context_window: baseConfig.high_param_context_window,
      high_param_max_tokens: baseConfig.high_param_max_tokens,
    },
  });
  const onConfigChange = vi.fn();
  renderSettings({
    config: { ...baseConfig, num_submitters: 3, submitter_configs: submitterConfigs },
    onConfigChange,
  });

  const countInput = await screen.findByRole('combobox', { name: 'Number of Submitters' });
  fireEvent.change(countInput, { target: { value: '4' } });

  const submitterFour = await screen.findByText('Submitter 4');
  const submitterFourCard = submitterFour.closest('.submitter-config-section');
  fireEvent.click(within(submitterFourCard).getByRole('button', { name: 'LM Studio' }));

  await waitFor(() => {
    const latestConfig = onConfigChange.mock.calls.at(-1)[0];
    expect(latestConfig.num_submitters).toBe(4);
    expect(latestConfig.submitter_configs).toHaveLength(4);
    expect(latestConfig.submitter_configs[3].provider).toBe('lm_studio');
  });

  fireEvent.change(countInput, { target: { value: '2' } });

  await waitFor(() => {
    const saved = JSON.parse(localStorage.getItem('autonomous_research_settings'));
    expect(saved.numSubmitters).toBe(2);
    expect(saved.submitterConfigs).toHaveLength(2);
  });
});

test('publishes a changed submitter count immediately', async () => {
  const submitterConfigs = [1, 2, 3].map(submitterId => ({
    ...baseConfig.submitter_configs[0],
    submitterId,
  }));
  const onConfigChange = vi.fn();
  const { unmount } = renderSettings({
    config: { ...baseConfig, num_submitters: 3, submitter_configs: submitterConfigs },
    onConfigChange,
  });

  const countInput = await screen.findByRole('combobox', { name: 'Number of Submitters' });
  fireEvent.change(countInput, { target: { value: '4' } });

  await waitFor(() => {
    const latestConfig = onConfigChange.mock.calls.at(-1)[0];
    expect(latestConfig.num_submitters).toBe(4);
    expect(latestConfig.submitter_configs).toHaveLength(4);
  });
  unmount();
});

test('does not let delayed model auto-fill restore an older submitter count', async () => {
  let resolveProviders;
  openRouterAPI.getProviders.mockImplementation(() => new Promise(resolve => {
    resolveProviders = resolve;
  }));
  const submitterConfigs = [1, 2, 3].map(submitterId => ({
    ...baseConfig.submitter_configs[0],
    submitterId,
  }));
  const onConfigChange = vi.fn();
  renderSettings({
    config: { ...baseConfig, num_submitters: 3, submitter_configs: submitterConfigs },
    onConfigChange,
  });

  const firstSubmitter = (await screen.findByText('Submitter 1 (Main Submitter)'))
    .closest('.submitter-config-section');
  const modelSelect = within(firstSubmitter).getAllByRole('combobox')[0];
  fireEvent.change(modelSelect, { target: { value: 'openrouter/assistant' } });

  const countInput = screen.getByRole('combobox', { name: 'Number of Submitters' });
  fireEvent.change(countInput, { target: { value: '4' } });
  resolveProviders({ providers: [], endpoints: [] });

  await waitFor(() => {
    const latestConfig = onConfigChange.mock.calls.at(-1)[0];
    expect(latestConfig.num_submitters).toBe(4);
    expect(latestConfig.submitter_configs).toHaveLength(4);
  });
});

test('discards delayed submitter auto-fill after the selected model changes', async () => {
  let resolveProviders;
  openRouterAPI.getProviders.mockImplementation(() => new Promise(resolve => {
    resolveProviders = resolve;
  }));
  const onConfigChange = vi.fn();
  renderSettings({ onConfigChange });

  const firstSubmitter = (await screen.findByText('Submitter 1 (Main Submitter)'))
    .closest('.submitter-config-section');
  const modelSelect = within(firstSubmitter).getAllByRole('combobox')[0];
  fireEvent.change(modelSelect, { target: { value: 'openrouter/assistant' } });
  fireEvent.click(within(firstSubmitter).getByRole('button', { name: 'LM Studio' }));
  resolveProviders({
    providers: [],
    endpoints: [{ context_length: 12345, max_completion_tokens: 1234 }],
  });

  await waitFor(() => {
    const latestConfig = onConfigChange.mock.calls.at(-1)[0];
    expect(latestConfig.submitter_configs[0].provider).toBe('lm_studio');
    expect(latestConfig.submitter_configs[0].modelId).toBe('');
    expect(latestConfig.submitter_configs[0].contextWindow).not.toBe(12345);
  });
});

test('does not let delayed Validator auto-fill restore an older submitter count', async () => {
  let resolveProviders;
  openRouterAPI.getProviders.mockImplementation(() => new Promise(resolve => {
    resolveProviders = resolve;
  }));
  const submitterConfigs = [1, 2, 3].map(submitterId => ({
    ...baseConfig.submitter_configs[0],
    submitterId,
  }));
  const onConfigChange = vi.fn();
  renderSettings({
    config: { ...baseConfig, num_submitters: 3, submitter_configs: submitterConfigs },
    onConfigChange,
  });

  const validatorHeading = await screen.findByText('Validator (Single Instance)');
  const validatorCard = validatorHeading.parentElement.querySelector('.submitter-config-section');
  fireEvent.change(within(validatorCard).getAllByRole('combobox')[0], {
    target: { value: 'openrouter/assistant' },
  });
  fireEvent.change(screen.getByRole('combobox', { name: 'Number of Submitters' }), {
    target: { value: '4' },
  });
  resolveProviders({ providers: [], endpoints: [] });

  await waitFor(() => {
    const latestConfig = onConfigChange.mock.calls.at(-1)[0];
    expect(latestConfig.num_submitters).toBe(4);
    expect(latestConfig.submitter_configs).toHaveLength(4);
  });
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
