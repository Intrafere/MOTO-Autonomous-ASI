import React, { useState } from 'react';
import { act, render, screen, waitFor, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import TranslatorModelsPanel, { TRANSLATOR_MODELS_EXPLANATION } from './TranslatorModelsPanel';
import { api, openRouterAPI } from '../../services/api';
import useCloudAccessProviders from '../../hooks/useCloudAccessProviders';
import { getStoredAutonomousSettings } from '../../utils/autonomousProfiles';

vi.mock('../../services/api', () => ({
  api: { getModels: vi.fn() },
  cloudAccessAPI: { getProviderStatus: vi.fn(), getProviderModels: vi.fn() },
  openRouterAPI: {
    getApiKeyStatus: vi.fn(),
    getModels: vi.fn(),
    getProviders: vi.fn(),
  },
}));

vi.mock('../../hooks/useCloudAccessProviders', () => ({
  default: vi.fn(),
}));

vi.mock('../../utils/autonomousProfiles', () => ({
  getStoredAutonomousSettings: vi.fn(),
}));

const OPENROUTER_MODELS = [
  { id: 'openrouter/alpha', name: 'Alpha', context_length: 200000 },
  { id: 'openrouter/beta', name: 'Beta', context_length: 100000 },
];

const ROLE = (modelId, overrides = {}) => ({
  provider: 'openrouter',
  model_id: modelId,
  openrouter_provider: null,
  openrouter_reasoning_effort: 'auto',
  lm_studio_fallback_id: null,
  context_window: 131072,
  max_output_tokens: 25000,
  ...overrides,
});

const EMPTY_SETTINGS = {
  enabled: false,
  paused: false,
  batch_translation: false,
  languages: [],
  translator: null,
  validator: null,
};

const SAVED_SETTINGS = { ...EMPTY_SETTINGS, translator: ROLE('openrouter/alpha'), validator: ROLE('openrouter/beta') };

function Harness({ initialSettings = EMPTY_SETTINGS, onSaveRoles, ...props }) {
  const [savedSettings, setSavedSettings] = useState(initialSettings);
  const saveRoles = async (roles) => {
    const response = await onSaveRoles(roles);
    setSavedSettings({ ...savedSettings, ...roles });
    return response;
  };
  return (
    <TranslatorModelsPanel
      savedSettings={savedSettings}
      capabilities={{ lmStudioEnabled: true }}
      onSaveRoles={saveRoles}
      {...props}
    />
  );
}

function renderPanel(props = {}) {
  const onSaveRoles = props.onSaveRoles || vi.fn(async () => ({ success: true }));
  const user = userEvent.setup();
  const view = render(<Harness {...props} onSaveRoles={onSaveRoles} />);
  return { ...view, onSaveRoles, user };
}

function card(title) {
  return screen.getByRole('heading', { name: title }).closest('.submitter-config-section');
}

function fieldIn(container, label) {
  const row = within(container).getByText(label, { selector: 'label' }).closest('.settings-row');
  return row.querySelector('select, input');
}

function roleTitles() {
  return screen.getAllByRole('heading', { level: 4 }).map((heading) => heading.textContent);
}

beforeEach(() => {
  vi.clearAllMocks();
  openRouterAPI.getApiKeyStatus.mockResolvedValue({ has_key: true });
  openRouterAPI.getModels.mockResolvedValue({ models: OPENROUTER_MODELS });
  openRouterAPI.getProviders.mockResolvedValue({ providers: [], endpoints: [] });
  api.getModels.mockResolvedValue({ models: [{ id: 'lm-one' }] });
  useCloudAccessProviders.mockReturnValue({
    statusByProvider: {},
    configuredProviders: [],
    modelErrors: [],
    modelsFor: () => [],
  });
  getStoredAutonomousSettings.mockReturnValue({ submitterConfigs: [], localConfig: {} });
});

test('shows exactly the translator submitter and validator cards, without lanes or Supercharge', async () => {
  renderPanel({ developerModeEnabled: true });
  expect(roleTitles()).toEqual(['Translator Submitter', 'Translator Validator']);
  expect(screen.getByText(TRANSLATOR_MODELS_EXPLANATION)).toBeInTheDocument();
  expect(TRANSLATOR_MODELS_EXPLANATION).toBe(
    'The translator works through each document in order, one logical part at a time, and can correct earlier '
      + 'parts. Each part keeps improving until the validator has accepted three attempts, or the translator '
      + 'confirms twice in a row that an accepted attempt is already perfect.',
  );
  expect(screen.queryByRole('combobox', { name: /Number of translator submitters/ })).not.toBeInTheDocument();
  expect(screen.queryByRole('button', { name: /Copy Submitter 1 to all/ })).not.toBeInTheDocument();
  expect(screen.getByRole('button', { name: 'Copy from Autonomous Research settings' })).toBeInTheDocument();
  expect(screen.queryByText('Supercharge')).not.toBeInTheDocument();
  expect(screen.getByRole('button', { name: 'Save translator models' })).toBeDisabled();
  expect(screen.getByRole('button', { name: 'Discard changes' })).toBeDisabled();
  await waitFor(() => expect(api.getModels).toHaveBeenCalledTimes(1));
  expect(useCloudAccessProviders).toHaveBeenCalledWith(expect.objectContaining({ capabilities: { lmStudioEnabled: true } }));
});

test('saved roles load into their cards; edits enable Save and Discard restores them', async () => {
  const { user } = renderPanel({ initialSettings: SAVED_SETTINGS });
  expect(fieldIn(card('Translator Submitter'), 'Context Window')).toHaveValue(131072);
  const validator = card('Translator Validator');
  await user.clear(fieldIn(validator, 'Max Output Tokens'));
  await user.type(fieldIn(validator, 'Max Output Tokens'), '8000');
  expect(screen.getByRole('button', { name: 'Save translator models' })).toBeEnabled();
  expect(screen.getByText('You have unsaved model changes.')).toBeInTheDocument();
  await user.click(screen.getByRole('button', { name: 'Discard changes' }));
  expect(fieldIn(card('Translator Validator'), 'Max Output Tokens')).toHaveValue(25000);
  expect(screen.getByRole('button', { name: 'Save translator models' })).toBeDisabled();
});

test('client validation blocks saving incomplete roles', async () => {
  const { user, onSaveRoles } = renderPanel({ initialSettings: { ...EMPTY_SETTINGS, validator: ROLE('openrouter/beta') } });
  const translator = card('Translator Submitter');
  await user.type(fieldIn(translator, 'Context Window'), '32768');
  await user.click(screen.getByRole('button', { name: 'Save translator models' }));
  const alert = screen.getByRole('alert');
  expect(alert).toHaveTextContent('Select a model for Translator Submitter.');
  expect(alert).toHaveTextContent('Enter max output tokens greater than 0 for Translator Submitter.');
  expect(alert).not.toHaveTextContent('Translator Validator');
  expect(onSaveRoles).not.toHaveBeenCalled();
});

test('max output must stay below the context window', async () => {
  const { user, onSaveRoles } = renderPanel({ initialSettings: SAVED_SETTINGS });
  const validator = card('Translator Validator');
  await user.clear(fieldIn(validator, 'Max Output Tokens'));
  await user.type(fieldIn(validator, 'Max Output Tokens'), '200000');
  await user.click(screen.getByRole('button', { name: 'Save translator models' }));
  expect(screen.getByRole('alert')).toHaveTextContent(
    'Translator Validator: max output tokens must be smaller than the context window.',
  );
  expect(onSaveRoles).not.toHaveBeenCalled();
});

test('saves exactly the translator and validator backend roles', async () => {
  const { user, onSaveRoles } = renderPanel({ initialSettings: SAVED_SETTINGS });
  const translator = card('Translator Submitter');
  await user.clear(fieldIn(translator, 'Max Output Tokens'));
  await user.type(fieldIn(translator, 'Max Output Tokens'), '16000');
  await user.click(screen.getByRole('button', { name: 'Save translator models' }));
  await waitFor(() => expect(onSaveRoles).toHaveBeenCalledTimes(1));
  expect(onSaveRoles).toHaveBeenCalledWith({
    translator: ROLE('openrouter/alpha', { max_output_tokens: 16000 }),
    validator: ROLE('openrouter/beta'),
  });
  expect(await screen.findByText('Translator models saved.')).toBeInTheDocument();
  expect(screen.getByRole('button', { name: 'Save translator models' })).toBeDisabled();
});

test('role cards are locked while a save is in flight, so no edit is lost when the reply arrives', async () => {
  let finishSave;
  const onSaveRoles = vi.fn(() => new Promise((resolve) => { finishSave = resolve; }));
  const { user } = renderPanel({ initialSettings: SAVED_SETTINGS, onSaveRoles });
  const translator = card('Translator Submitter');
  await user.clear(fieldIn(translator, 'Max Output Tokens'));
  await user.type(fieldIn(translator, 'Max Output Tokens'), '16000');
  await user.click(screen.getByRole('button', { name: 'Save translator models' }));

  expect(screen.getByRole('button', { name: 'Saving translator models…' })).toBeDisabled();
  ['Translator Submitter', 'Translator Validator'].forEach((title) => {
    ['Model', 'Context Window', 'Max Output Tokens'].forEach((label) => {
      expect(fieldIn(card(title), label)).toBeDisabled();
    });
  });
  expect(screen.getByRole('button', { name: 'Copy from Autonomous Research settings' })).toBeDisabled();
  await user.type(fieldIn(card('Translator Validator'), 'Max Output Tokens'), '9');
  expect(fieldIn(card('Translator Validator'), 'Max Output Tokens')).toHaveValue(25000);

  await act(async () => { finishSave({ success: true }); });
  expect(await screen.findByText('Translator models saved.')).toBeInTheDocument();
  expect(fieldIn(card('Translator Submitter'), 'Max Output Tokens')).toHaveValue(16000);
  expect(fieldIn(card('Translator Validator'), 'Max Output Tokens')).toHaveValue(25000);
  expect(fieldIn(card('Translator Validator'), 'Max Output Tokens')).toBeEnabled();
  expect(screen.getByRole('button', { name: 'Copy from Autonomous Research settings' })).toBeEnabled();
});

test('copies Autonomous Submitter 1 to the translator and the Validator to the validator', async () => {
  getStoredAutonomousSettings.mockReturnValue({
    submitterConfigs: [
      {
        submitterId: 1,
        provider: 'openrouter',
        modelId: 'openrouter/alpha',
        openrouterProvider: null,
        openrouterReasoningEffort: 'high',
        lmStudioFallbackId: 'lm-one',
        contextWindow: 200000,
        maxOutputTokens: 32000,
        superchargeEnabled: true,
      },
      {
        submitterId: 2,
        provider: 'lm_studio',
        modelId: 'lm-one',
        openrouterReasoningEffort: 'auto',
        contextWindow: 32768,
        maxOutputTokens: 8192,
      },
    ],
    localConfig: {
      validator_provider: 'openrouter',
      validator_model: 'openrouter/beta',
      validator_openrouter_provider: null,
      validator_openrouter_reasoning_effort: 'xhigh',
      validator_lm_studio_fallback: null,
      validator_context_window: 100000,
      validator_max_tokens: 20000,
      validator_supercharge_enabled: true,
    },
  });
  const { user, onSaveRoles } = renderPanel();
  await user.click(screen.getByRole('button', { name: 'Copy from Autonomous Research settings' }));
  expect(screen.getByRole('status')).toHaveTextContent(
    'Copied Submitter 1 to the Translator Submitter and the Validator to the Translator Validator '
      + 'from Autonomous Research settings. Review, then save.',
  );
  expect(roleTitles()).toEqual(['Translator Submitter', 'Translator Validator']);
  await user.click(screen.getByRole('button', { name: 'Save translator models' }));
  await waitFor(() => expect(onSaveRoles).toHaveBeenCalledTimes(1));
  expect(onSaveRoles).toHaveBeenCalledWith({
    translator: ROLE('openrouter/alpha', {
      openrouter_reasoning_effort: 'high',
      lm_studio_fallback_id: 'lm-one',
      context_window: 200000,
      max_output_tokens: 32000,
    }),
    validator: ROLE('openrouter/beta', {
      openrouter_reasoning_effort: 'xhigh',
      context_window: 100000,
      max_output_tokens: 20000,
    }),
  });
});

test('a partial copy keeps the role it lacks and says what was copied', async () => {
  getStoredAutonomousSettings.mockReturnValue({
    submitterConfigs: [],
    localConfig: { validator_provider: 'openrouter', validator_model: 'openrouter/alpha', validator_context_window: 100000, validator_max_tokens: 9000 },
  });
  const { user } = renderPanel({ initialSettings: SAVED_SETTINGS });
  await user.click(screen.getByRole('button', { name: 'Copy from Autonomous Research settings' }));
  expect(screen.getByRole('status')).toHaveTextContent(
    'Copied the Validator to the Translator Validator from Autonomous Research settings.',
  );
  expect(fieldIn(card('Translator Validator'), 'Max Output Tokens')).toHaveValue(9000);
  expect(fieldIn(card('Translator Submitter'), 'Max Output Tokens')).toHaveValue(25000);
});

test('reports when there are no Autonomous Research models to copy', async () => {
  const { user } = renderPanel();
  await user.click(screen.getByRole('button', { name: 'Copy from Autonomous Research settings' }));
  expect(screen.getByRole('status')).toHaveTextContent('No Autonomous Research model settings were found to copy.');
});

test('shows the backend validation message when saving fails', async () => {
  const onSaveRoles = vi.fn(async () => {
    throw Object.assign(
      new Error("HTTP 422 Unprocessable Entity: Reasoning effort 'max' is supported only by kimi_code, openai_codex_oauth"),
      { status: 422 },
    );
  });
  const { user } = renderPanel({ initialSettings: SAVED_SETTINGS, onSaveRoles });
  const translator = card('Translator Submitter');
  await user.clear(fieldIn(translator, 'Max Output Tokens'));
  await user.type(fieldIn(translator, 'Max Output Tokens'), '16000');
  await user.click(screen.getByRole('button', { name: 'Save translator models' }));
  expect(await screen.findByRole('alert')).toHaveTextContent(
    "Reasoning effort 'max' is supported only by kimi_code, openai_codex_oauth",
  );
  expect(screen.getByRole('alert')).not.toHaveTextContent('HTTP 422');
  expect(screen.getByRole('button', { name: 'Save translator models' })).toBeEnabled();
});

test('hosted mode is OpenRouter only and never loads LM Studio models', async () => {
  renderPanel({ capabilities: { lmStudioEnabled: false, genericMode: true } });
  await waitFor(() => expect(openRouterAPI.getModels).toHaveBeenCalled());
  expect(api.getModels).not.toHaveBeenCalled();
  expect(screen.queryByRole('button', { name: 'LM Studio' })).not.toBeInTheDocument();
  expect(screen.getAllByText('OpenRouter is required in this deployment.')).toHaveLength(2);
  expect(screen.queryByText('LM Studio Fallback')).not.toBeInTheDocument();
});

test.each([
  ['Translator Submitter'],
  ['Translator Validator'],
])('auto-fills %s context and max output from OpenRouter metadata and ignores stale results', async (title) => {
  let resolveAlpha;
  openRouterAPI.getProviders.mockImplementation((modelId) => {
    if (modelId === 'openrouter/alpha') {
      return new Promise((resolve) => { resolveAlpha = resolve; });
    }
    return Promise.resolve({
      providers: ['Google'],
      endpoints: [{ provider_name: 'Google', context_length: 100000, max_completion_tokens: 8000 }],
    });
  });
  const { user } = renderPanel({ capabilities: { lmStudioEnabled: false } });
  await waitFor(() => expect(openRouterAPI.getModels).toHaveBeenCalled());
  const role = card(title);
  await waitFor(() => expect(within(fieldIn(role, 'Model')).getByRole('option', { name: /Alpha/ })).toBeInTheDocument());

  await user.selectOptions(fieldIn(role, 'Model'), 'openrouter/alpha');
  await user.selectOptions(fieldIn(role, 'Model'), 'openrouter/beta');
  await waitFor(() => expect(fieldIn(role, 'Context Window')).toHaveValue(100000));
  expect(fieldIn(role, 'Max Output Tokens')).toHaveValue(8000);

  await act(async () => {
    resolveAlpha({
      providers: ['Anthropic'],
      endpoints: [{ provider_name: 'Anthropic', context_length: 200000, max_completion_tokens: 32000 }],
    });
  });
  expect(fieldIn(role, 'Context Window')).toHaveValue(100000);
  expect(fieldIn(role, 'Max Output Tokens')).toHaveValue(8000);
  expect(within(fieldIn(role, 'Host Provider')).getByRole('option', { name: /Google/ })).toBeInTheDocument();
});

const LOADING_LIMITS = 'Loading the context window and max output for the selected model…';
const BETA_ENDPOINTS = {
  providers: ['Google'],
  endpoints: [{ provider_name: 'Google', context_length: 100000, max_completion_tokens: 8000 }],
};

async function selectTranslatorModelWithHeldLimits(user, modelId) {
  let resolveLimits;
  let rejectLimits;
  openRouterAPI.getProviders.mockImplementation(() => new Promise((resolve, reject) => {
    resolveLimits = resolve;
    rejectLimits = reject;
  }));
  await waitFor(() => expect(openRouterAPI.getModels).toHaveBeenCalled());
  const translator = card('Translator Submitter');
  await waitFor(() => expect(within(fieldIn(translator, 'Model')).getByRole('option', { name: /Beta/ })).toBeInTheDocument());
  await user.selectOptions(fieldIn(translator, 'Model'), modelId);
  return { resolveLimits: (value) => resolveLimits(value), rejectLimits: (error) => rejectLimits(error) };
}

test('Save waits for the new model limits, so a saved role never keeps the previous model limits', async () => {
  const { user, onSaveRoles } = renderPanel({ initialSettings: SAVED_SETTINGS, capabilities: { lmStudioEnabled: false } });
  const { resolveLimits } = await selectTranslatorModelWithHeldLimits(user, 'openrouter/beta');
  const translator = card('Translator Submitter');

  expect(screen.getByRole('button', { name: 'Save translator models' })).toBeDisabled();
  expect(screen.getByText(LOADING_LIMITS)).toBeInTheDocument();
  expect(fieldIn(translator, 'Context Window')).toHaveValue(131072);
  await user.click(screen.getByRole('button', { name: 'Save translator models' }));
  expect(onSaveRoles).not.toHaveBeenCalled();

  await act(async () => { resolveLimits(BETA_ENDPOINTS); });
  expect(fieldIn(translator, 'Context Window')).toHaveValue(100000);
  expect(fieldIn(translator, 'Max Output Tokens')).toHaveValue(8000);
  expect(screen.queryByText(LOADING_LIMITS)).not.toBeInTheDocument();
  expect(screen.getByText('You have unsaved model changes.')).toBeInTheDocument();
  await user.click(screen.getByRole('button', { name: 'Save translator models' }));
  await waitFor(() => expect(onSaveRoles).toHaveBeenCalledTimes(1));
  expect(onSaveRoles.mock.calls[0][0].translator).toEqual(
    ROLE('openrouter/beta', { context_window: 100000, max_output_tokens: 8000 }),
  );
  expect(await screen.findByText('Translator models saved.')).toBeInTheDocument();
  expect(fieldIn(card('Translator Submitter'), 'Context Window')).toHaveValue(100000);
});

test('a failed limit lookup releases Save and keeps the limits the user can edit', async () => {
  const consoleError = vi.spyOn(console, 'error').mockImplementation(() => {});
  try {
    const { user } = renderPanel({ initialSettings: SAVED_SETTINGS, capabilities: { lmStudioEnabled: false } });
    const { rejectLimits } = await selectTranslatorModelWithHeldLimits(user, 'openrouter/beta');
    expect(screen.getByRole('button', { name: 'Save translator models' })).toBeDisabled();

    await act(async () => { rejectLimits(new Error('offline')); });
    expect(consoleError).toHaveBeenCalled();
    expect(screen.getByRole('button', { name: 'Save translator models' })).toBeEnabled();
    expect(screen.queryByText(LOADING_LIMITS)).not.toBeInTheDocument();
    expect(fieldIn(card('Translator Submitter'), 'Max Output Tokens')).toHaveValue(25000);
  } finally {
    consoleError.mockRestore();
  }
});

test('Discard cancels a pending limit lookup, so it neither holds Save nor changes the restored role', async () => {
  const { user } = renderPanel({ initialSettings: SAVED_SETTINGS, capabilities: { lmStudioEnabled: false } });
  const { resolveLimits } = await selectTranslatorModelWithHeldLimits(user, 'openrouter/beta');
  await user.click(screen.getByRole('button', { name: 'Discard changes' }));

  const validator = card('Translator Validator');
  await user.clear(fieldIn(validator, 'Max Output Tokens'));
  await user.type(fieldIn(validator, 'Max Output Tokens'), '9000');
  expect(screen.getByRole('button', { name: 'Save translator models' })).toBeEnabled();
  expect(screen.queryByText(LOADING_LIMITS)).not.toBeInTheDocument();

  await act(async () => { resolveLimits(BETA_ENDPOINTS); });
  const translator = card('Translator Submitter');
  expect(fieldIn(translator, 'Model')).toHaveValue('openrouter/alpha');
  expect(fieldIn(translator, 'Context Window')).toHaveValue(131072);
  expect(fieldIn(translator, 'Max Output Tokens')).toHaveValue(25000);
  expect(screen.getByRole('button', { name: 'Save translator models' })).toBeEnabled();
});

test('auto-fill of one role never changes the other role', async () => {
  openRouterAPI.getProviders.mockResolvedValue({
    providers: ['Google'],
    endpoints: [{ provider_name: 'Google', context_length: 100000, max_completion_tokens: 8000 }],
  });
  const { user } = renderPanel({ initialSettings: SAVED_SETTINGS, capabilities: { lmStudioEnabled: false } });
  await waitFor(() => expect(openRouterAPI.getModels).toHaveBeenCalled());
  const translator = card('Translator Submitter');
  await user.selectOptions(fieldIn(translator, 'Model'), 'openrouter/beta');
  await waitFor(() => expect(fieldIn(translator, 'Max Output Tokens')).toHaveValue(8000));
  expect(fieldIn(card('Translator Validator'), 'Max Output Tokens')).toHaveValue(25000);
  expect(fieldIn(card('Translator Validator'), 'Context Window')).toHaveValue(131072);
});

test('a credential refresh token reloads the OpenRouter key state', async () => {
  const { rerender } = render(
    <Harness onSaveRoles={vi.fn()} credentialStatusRefreshToken={0} />,
  );
  await waitFor(() => expect(openRouterAPI.getApiKeyStatus).toHaveBeenCalledTimes(1));
  rerender(<Harness onSaveRoles={vi.fn()} credentialStatusRefreshToken={1} />);
  await waitFor(() => expect(openRouterAPI.getApiKeyStatus).toHaveBeenCalledTimes(2));
});
