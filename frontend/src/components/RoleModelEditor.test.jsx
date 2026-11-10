import React from 'react';
import { fireEvent, render, screen, within } from '@testing-library/react';
import RoleModelEditorDefault, { RoleModelEditor, RoleModelSelector } from './RoleModelEditor';
import { getConfiguredCloudAccessProviders } from '../utils/oauthProviders';

const OPENROUTER_MODELS = [
  { id: 'openrouter/alpha', name: 'Alpha', context_length: 131072 },
  { id: 'openrouter/free', name: 'Free Model', context_length: 32000, pricing: { prompt: '0', completion: '0' } },
];
const LM_STUDIO_MODELS = [{ id: 'lm-one' }, { id: 'lm-two' }];

function cloudProvidersView({ configured = [], models = {} } = {}) {
  const statusByProvider = Object.fromEntries(configured.map((id) => [id, { configured: true }]));
  return {
    statusByProvider,
    configuredProviders: getConfiguredCloudAccessProviders(statusByProvider),
    modelsFor: (id) => models[id] || [],
  };
}

function renderEditor(overrides = {}) {
  const onChange = overrides.onChange || vi.fn();
  const props = {
    title: 'Translator Submitter 1 (Main)',
    config: {
      provider: 'lm_studio',
      modelId: '',
      openrouterProvider: null,
      openrouterReasoningEffort: 'auto',
      lmStudioFallbackId: null,
      contextWindow: 131072,
      maxOutputTokens: 25000,
    },
    lmStudioModels: LM_STUDIO_MODELS,
    openRouterModels: OPENROUTER_MODELS,
    cloudProviders: cloudProvidersView(),
    modelProviders: {},
    hasOpenRouterKey: true,
    lmStudioEnabled: true,
    isRunning: false,
    developerModeEnabled: false,
    ...overrides,
    onChange,
  };
  const view = render(<RoleModelEditor {...props} />);
  return { ...view, onChange, props };
}

const rowFor = (container, label) => {
  const labelNode = within(container).queryByText(label, { selector: 'label' });
  return labelNode ? labelNode.closest('.settings-row') : null;
};
const selectFor = (container, label) => rowFor(container, label)?.querySelector('select') || null;
const inputFor = (container, label) => rowFor(container, label)?.querySelector('input') || null;

test('exports the editor as both a named and default export', () => {
  expect(RoleModelEditorDefault).toBe(RoleModelEditor);
  expect(typeof RoleModelSelector).toBe('function');
});

test('renders the title and caller children before the model selector', () => {
  const { container } = renderEditor({
    children: <p className="settings-info">Translator note</p>,
  });
  const card = container.querySelector('.submitter-config-section');
  const heading = within(card).getByRole('heading', { name: 'Translator Submitter 1 (Main)' });
  const note = within(card).getByText('Translator note');
  const providerRow = rowFor(card, 'Provider');
  expect(heading.compareDocumentPosition(note) & Node.DOCUMENT_POSITION_FOLLOWING).toBeTruthy();
  expect(note.compareDocumentPosition(providerRow) & Node.DOCUMENT_POSITION_FOLLOWING).toBeTruthy();
});

test('provider toggles: OpenRouter needs a key and Cloud needs a configured provider', () => {
  const { onChange } = renderEditor({ hasOpenRouterKey: false });
  expect(screen.getByRole('button', { name: 'LM Studio' })).toBeEnabled();
  const openRouterButton = screen.getByRole('button', { name: 'OpenRouter' });
  expect(openRouterButton).toBeDisabled();
  expect(openRouterButton).toHaveAttribute('title', 'Set OpenRouter API key first');
  const cloudButton = screen.getByRole('button', { name: 'Cloud' });
  expect(cloudButton).toBeDisabled();
  expect(cloudButton).toHaveAttribute('title', 'Set up a cloud provider login or API key first');
  expect(screen.getByRole('button', { name: 'LM Studio' })).toHaveClass('active-lm');
  fireEvent.click(screen.getByRole('button', { name: 'LM Studio' }));
  expect(onChange).not.toHaveBeenCalled();
});

test('provider toggles switch providers and clear the stale model', () => {
  const { onChange } = renderEditor({
    config: { provider: 'lm_studio', modelId: 'lm-one', contextWindow: 8192, maxOutputTokens: 2000 },
    cloudProviders: cloudProvidersView({ configured: ['xai_grok_oauth'] }),
  });
  const openRouterButton = screen.getByRole('button', { name: 'OpenRouter' });
  expect(openRouterButton).toHaveAttribute('title', 'Use OpenRouter');
  fireEvent.click(openRouterButton);
  expect(onChange).toHaveBeenLastCalledWith(expect.objectContaining({
    provider: 'openrouter',
    modelId: '',
    openrouterProvider: null,
    openrouterReasoningEffort: 'auto',
    contextWindow: 8192,
    maxOutputTokens: 2000,
  }));

  const cloudButton = screen.getByRole('button', { name: 'Cloud' });
  expect(cloudButton).toBeEnabled();
  expect(cloudButton).toHaveAttribute('title', 'Use a configured cloud provider');
  fireEvent.click(cloudButton);
  expect(onChange).toHaveBeenLastCalledWith(expect.objectContaining({
    provider: 'xai_grok_oauth',
    modelId: '',
  }));
});

test('model select lists provider models and keeps an unknown configured model', () => {
  const { container, onChange } = renderEditor({
    config: { provider: 'openrouter', modelId: 'openrouter/retired', contextWindow: 8192, maxOutputTokens: 2000 },
  });
  const modelSelect = selectFor(container, 'Model');
  expect(modelSelect).toHaveValue('openrouter/retired');
  expect(within(modelSelect).getByRole('option', { name: 'openrouter/retired (configured)' })).toBeInTheDocument();
  expect(within(modelSelect).getByRole('option', { name: 'Alpha (131K)' })).toBeInTheDocument();
  expect(within(modelSelect).getByRole('option', { name: 'Free Model (32K) [FREE]' })).toBeInTheDocument();

  fireEvent.change(modelSelect, { target: { value: 'openrouter/alpha' } });
  expect(onChange).toHaveBeenLastCalledWith(expect.objectContaining({
    provider: 'openrouter',
    modelId: 'openrouter/alpha',
    openrouterProvider: null,
    openrouterReasoningEffort: 'auto',
  }));
});

test('model select uses LM Studio models for LM Studio roles without a configured option', () => {
  const { container } = renderEditor({
    config: { provider: 'lm_studio', modelId: 'lm-two', contextWindow: 8192, maxOutputTokens: 2000 },
  });
  const modelSelect = selectFor(container, 'Model');
  expect(modelSelect).toHaveValue('lm-two');
  expect(within(modelSelect).queryByText(/\(configured\)/)).not.toBeInTheDocument();
  expect(within(modelSelect).getAllByRole('option').map((option) => option.value)).toEqual(['', 'lm-one', 'lm-two']);
});

test('OpenRouter roles with a model show host provider and reasoning effort rows', () => {
  const { container, onChange } = renderEditor({
    config: {
      provider: 'openrouter',
      modelId: 'openrouter/alpha',
      openrouterProvider: null,
      openrouterReasoningEffort: 'medium',
      contextWindow: 131072,
      maxOutputTokens: 25000,
    },
    modelProviders: {
      'openrouter/alpha': {
        providers: ['HostA', 'HostB'],
        endpoints: [{ provider_name: 'HostA', supported_parameters: ['max_tokens'] }],
      },
    },
  });
  const hostSelect = selectFor(container, 'Host Provider');
  expect(hostSelect).toHaveValue('');
  expect(within(hostSelect).getAllByRole('option').map((option) => option.value)).toEqual(['', 'HostA', 'HostB']);
  fireEvent.change(hostSelect, { target: { value: 'HostB' } });
  expect(onChange).toHaveBeenLastCalledWith(expect.objectContaining({ provider: 'openrouter', openrouterProvider: 'HostB' }));
  fireEvent.change(hostSelect, { target: { value: '' } });
  expect(onChange).toHaveBeenLastCalledWith(expect.objectContaining({ openrouterProvider: null }));

  const reasoningSelect = selectFor(container, 'Reasoning Effort');
  expect(reasoningSelect).toHaveValue('medium');
  expect(within(rowFor(container, 'Reasoning Effort')).getByText(
    'This selected host does not advertise reasoning support; OpenRouter may ignore the setting.'
  )).toBeInTheDocument();
  fireEvent.change(reasoningSelect, { target: { value: 'low' } });
  expect(onChange).toHaveBeenLastCalledWith(expect.objectContaining({ provider: 'openrouter', openrouterReasoningEffort: 'low' }));
});

test('OpenRouter host and reasoning rows stay hidden until a model is chosen', () => {
  const { container } = renderEditor({
    config: { provider: 'openrouter', modelId: '', contextWindow: 8192, maxOutputTokens: 2000 },
  });
  expect(rowFor(container, 'Host Provider')).toBeNull();
  expect(rowFor(container, 'Reasoning Effort')).toBeNull();
});

test('LM Studio fallback row appears only for non-LM providers when LM Studio is enabled', () => {
  const lmView = renderEditor({
    config: { provider: 'lm_studio', modelId: 'lm-one', contextWindow: 8192, maxOutputTokens: 2000 },
  });
  expect(rowFor(lmView.container, 'LM Studio Fallback')).toBeNull();
  lmView.unmount();

  const { container, onChange } = renderEditor({
    config: { provider: 'openrouter', modelId: 'openrouter/alpha', lmStudioFallbackId: null, contextWindow: 8192, maxOutputTokens: 2000 },
  });
  const fallbackSelect = selectFor(container, 'LM Studio Fallback');
  expect(fallbackSelect).toHaveValue('');
  expect(within(fallbackSelect).getAllByRole('option').map((option) => option.value)).toEqual(['', 'lm-one', 'lm-two']);
  fireEvent.change(fallbackSelect, { target: { value: 'lm-two' } });
  expect(onChange).toHaveBeenLastCalledWith(expect.objectContaining({ provider: 'openrouter', lmStudioFallbackId: 'lm-two' }));
  fireEvent.change(fallbackSelect, { target: { value: '' } });
  expect(onChange).toHaveBeenLastCalledWith(expect.objectContaining({ lmStudioFallbackId: null }));
});

test('generic mode shows the OpenRouter-only hint, OpenRouter models, and no fallback row', () => {
  const { container } = renderEditor({
    lmStudioEnabled: false,
    config: { provider: 'lm_studio', modelId: '', contextWindow: 8192, maxOutputTokens: 2000 },
  });
  expect(screen.getByText('OpenRouter is required in this deployment.')).toBeInTheDocument();
  expect(screen.queryByRole('button', { name: 'LM Studio' })).not.toBeInTheDocument();
  expect(screen.queryByRole('button', { name: 'OpenRouter' })).not.toBeInTheDocument();
  expect(screen.queryByRole('button', { name: 'Cloud' })).not.toBeInTheDocument();
  const modelSelect = selectFor(container, 'Model');
  expect(within(modelSelect).getAllByRole('option').map((option) => option.value)).toEqual(['', 'openrouter/alpha', 'openrouter/free']);
  expect(rowFor(container, 'LM Studio Fallback')).toBeNull();
});

test('Supercharge is developer-only and never rendered when showSupercharge is false', () => {
  const hidden = renderEditor({ developerModeEnabled: false });
  expect(screen.queryByRole('button', { name: 'Learn about Supercharge' })).not.toBeInTheDocument();
  hidden.unmount();

  const shown = renderEditor({ developerModeEnabled: true });
  const checkbox = shown.container.querySelector('.settings-checkbox-label--supercharge input[type="checkbox"]');
  expect(checkbox).not.toBeNull();
  fireEvent.click(checkbox);
  expect(shown.onChange).toHaveBeenLastCalledWith(expect.objectContaining({ superchargeEnabled: true }));
  shown.unmount();

  const suppressed = renderEditor({ developerModeEnabled: true, showSupercharge: false });
  expect(suppressed.container.querySelector('.settings-checkbox-label--supercharge')).toBeNull();
  expect(screen.queryByRole('button', { name: 'Learn about Supercharge' })).not.toBeInTheDocument();
});

test('context window and max output inputs report parsed integers', () => {
  const { container, onChange } = renderEditor();
  const contextInput = inputFor(container, 'Context Window');
  const outputInput = inputFor(container, 'Max Output Tokens');
  expect(contextInput).toHaveValue(131072);
  expect(outputInput).toHaveValue(25000);

  fireEvent.change(contextInput, { target: { value: '65536' } });
  expect(onChange).toHaveBeenLastCalledWith(expect.objectContaining({ contextWindow: 65536, maxOutputTokens: 25000 }));
  fireEvent.change(outputInput, { target: { value: '12000' } });
  expect(onChange).toHaveBeenLastCalledWith(expect.objectContaining({ contextWindow: 131072, maxOutputTokens: 12000 }));
  fireEvent.change(outputInput, { target: { value: '' } });
  expect(onChange).toHaveBeenLastCalledWith(expect.objectContaining({ maxOutputTokens: '' }));
  expect(onChange.mock.calls.every(([config]) => typeof config.contextWindow === 'number')).toBe(true);
});

test('running or disabled roles lock every control', () => {
  const running = renderEditor({
    isRunning: true,
    developerModeEnabled: true,
    config: { provider: 'openrouter', modelId: 'openrouter/alpha', contextWindow: 8192, maxOutputTokens: 2000 },
  });
  const runningCard = running.container.querySelector('.submitter-config-section');
  expect(runningCard).toHaveAttribute('aria-disabled', 'false');
  expect(runningCard).toHaveClass('role-config-card--openrouter-orange');
  expect(Array.from(runningCard.querySelectorAll('select, input')).every((node) => node.disabled)).toBe(true);
  running.unmount();

  const disabled = renderEditor({ disabled: true, developerModeEnabled: true });
  const disabledCard = disabled.container.querySelector('.submitter-config-section');
  expect(disabledCard).toHaveAttribute('aria-disabled', 'true');
  expect(disabledCard).not.toHaveClass('role-config-card--openrouter-orange');
  expect(Array.from(disabledCard.querySelectorAll('select, input, button.provider-toggle-btn')).every((node) => node.disabled)).toBe(true);
});

test('RoleModelSelector tolerates a missing cloud provider view', () => {
  render(
    <RoleModelSelector
      config={{ provider: 'lm_studio', modelId: '' }}
      onChange={vi.fn()}
      hasOpenRouterKey
      lmStudioEnabled
      isRunning={false}
    />
  );
  expect(screen.getByRole('button', { name: 'Cloud' })).toBeDisabled();
});
