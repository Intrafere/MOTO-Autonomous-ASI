import React, { useCallback, useState } from 'react';
import { fireEvent, render, screen, waitFor, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import AutoTranslateModal, {
  BATCH_TRANSLATION_HELP,
  TRANSLATE_ALL_HELP,
  TRANSLATION_MODES_HELP,
  TranslationDeveloperModesGate,
  batchCapacityText,
  buildTranslationSettingsPayload,
  statusBannerVariant,
  translateAllStatusText,
} from './AutoTranslateModal';
import {
  buildQueueRows,
  describeActivePart,
  describeTranslationActivity,
} from './TranslationProgressPanel';
import { formatPartRange, translationPercent } from './translationPercent';
import { AutoTranslateContext, INERT_AUTO_TRANSLATE_CONTEXT } from '../../hooks/useAutoTranslate';
import { api, openRouterAPI, translationAPI } from '../../services/api';
import useCloudAccessProviders from '../../hooks/useCloudAccessProviders';
import { expectNoAxeViolations } from '../../test/axe';

vi.mock('../../services/api', () => ({
  api: { getModels: vi.fn() },
  cloudAccessAPI: { getProviderStatus: vi.fn(), getProviderModels: vi.fn() },
  openRouterAPI: {
    getApiKeyStatus: vi.fn(),
    getModels: vi.fn(),
    getProviders: vi.fn(),
  },
  translationAPI: {
    getStatus: vi.fn(),
    getSettings: vi.fn(),
    saveSettings: vi.fn(),
    control: vi.fn(),
    getLanguages: vi.fn(),
    probeLanguages: vi.fn(),
    clearCache: vi.fn(),
  },
}));

vi.mock('../../hooks/useCloudAccessProviders', () => ({
  default: vi.fn(),
}));

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

const ENGLISH = { key: 'en', name: 'English', custom: false };
const FRENCH = { key: 'fr', name: 'French', custom: false };

const STANDARD_MODES = ['autonomous', 'aggregator', 'compiler'];

const SAVED_SETTINGS = {
  enabled: true,
  paused: false,
  batch_translation: false,
  languages: [ENGLISH],
  translator: ROLE('openrouter/alpha'),
  validator: ROLE('openrouter/beta'),
  modes: STANDARD_MODES,
  translate_all: false,
};

const TRANSLATE_ALL_OFF = {
  requested: false,
  state: 'off',
  documents: 0,
  complete_documents: 0,
  translated_units: 0,
  pending_units: 0,
  message: '',
};

const CATALOG = {
  success: true,
  pinned_keys: ['en', 'fr'],
  languages: [
    { key: 'en', name: 'English', native_name: '', aliases: [], pinned: true, proficiency: 'proficient' },
    { key: 'fr', name: 'French', native_name: 'Français', aliases: [], pinned: true, proficiency: 'proficient' },
    { key: 'yo', name: 'Yoruba', native_name: 'Yorùbá', aliases: [], pinned: false, proficiency: 'uncertain' },
  ],
  extras: [],
  probe: {
    status: 'complete',
    probed_at: new Date().toISOString(),
    models: [
      { role: 'translator', provider: 'openrouter', model_id: 'openrouter/alpha', ok: true, error: '' },
      { role: 'validator', provider: 'openrouter', model_id: 'openrouter/beta', ok: true, error: '' },
    ],
    stale: false,
  },
};

function status(overrides = {}) {
  return {
    success: true,
    enabled: true,
    paused: false,
    batch_translation: false,
    max_parallel_jobs: 1,
    state: 'translating',
    status_label: 'translating',
    message: 'Translating 14 unit(s).',
    generation: 3,
    languages: [
      { key: 'fr', name: 'French', custom: false, translated_units: 6, pending_units: 4, pending_documents: 2, complete_documents: 1 },
      { key: 'es', name: 'Spanish', custom: false, translated_units: 0, pending_units: 10, pending_documents: 1, complete_documents: 0 },
    ],
    // Deliberately out of position order: the panel orders rows by position.
    queue: [
      {
        doc_hash: 'h1',
        doc_key: 'proof:autonomous:run:p1',
        kind: 'proof',
        title: 'Main Theorem',
        language_key: 'fr',
        position: 0,
        viewing: true,
        translated_units: 2,
        total_units: 5,
      },
      {
        doc_hash: 'h2',
        doc_key: 'final_answer:session',
        kind: 'final_answer',
        title: 'Final Answer',
        language_key: 'fr',
        position: 2,
        viewing: false,
        translated_units: 1,
        total_units: 9,
      },
      {
        doc_hash: 'h1',
        doc_key: 'proof:autonomous:run:p1',
        kind: 'proof',
        title: 'Main Theorem',
        language_key: 'es',
        position: 1,
        viewing: true,
        translated_units: 0,
        total_units: 5,
      },
    ],
    active: [
      {
        doc_hash: 'h1',
        doc_title: 'Main Theorem',
        language_key: 'fr',
        part_start: 3,
        part_end: 4,
        attempt: 2,
        accepted_count: 1,
        keep_streak: 1,
      },
    ],
    repair: null,
    usage: {
      calls: 42,
      prompt_tokens: 900,
      completion_tokens: 300,
      accepted_attempts: 7,
      rejected_attempts: 5,
      finalized_parts: 3,
      check_failures: 2,
    },
    recent_activity: [],
    translate_all: TRANSLATE_ALL_OFF,
    ...overrides,
  };
}

// Live events carry the full payload; status recent_activity copies keep only
// candidate_id, round_index, keep_streak, and the other kept fields.
const ACTIVITY = [
  {
    activity_id: 'a1',
    event_type: 'translation_attempt_accepted',
    language_key: 'fr',
    language_name: 'French',
    doc_title: 'Main Theorem',
    part_start: 3,
    part_end: 4,
    round_index: 2,
    attempt_number: 2,
    rejudged: false,
    candidate_id: 'h1:fr:3:attempt2',
    accepted_count: 1,
    timestamp: '2026-11-01T12:00:00Z',
  },
  {
    activity_id: 'a2',
    event_type: 'translation_attempt_rejected',
    language_key: 'fr',
    language_name: 'French',
    doc_title: 'Main Theorem',
    part_start: 3,
    part_end: 4,
    round_index: 1,
    candidate_id: 'h1:fr:3:attempt1',
    feedback: 'Keep the theorem name untranslated.',
  },
  {
    activity_id: 'a3',
    event_type: 'translation_attempt_accepted',
    language_key: 'fr',
    doc_title: 'Paper A',
    part_start: 1,
    part_end: 1,
    round_index: 3,
    candidate_id: 'h2:fr:1:attempt1:rejudge3',
    accepted_count: 1,
  },
  {
    activity_id: 'a4',
    event_type: 'translation_attempt_kept',
    language_key: 'fr',
    doc_title: 'Main Theorem',
    part_start: 3,
    part_end: 4,
    round_index: 5,
    keep_attempts: [2],
    keep_streak: 1,
    dispute: false,
  },
  {
    activity_id: 'a5',
    event_type: 'translation_attempt_kept',
    language_key: 'fr',
    doc_title: 'Paper A',
    part_start: 1,
    part_end: 1,
    round_index: 3,
    keep_attempts: [1],
    keep_streak: 0,
  },
  {
    activity_id: 'a6',
    event_type: 'translation_part_finalized',
    language_key: 'fr',
    doc_title: 'Main Theorem',
    part_start: 3,
    part_end: 4,
    final_attempt: 2,
    reason: 'kept_twice',
  },
  {
    activity_id: 'a7',
    event_type: 'translation_attempt_check_failed',
    language_key: 'es',
    doc_title: 'Paper A',
    part_start: 5,
    part_end: 5,
    round_index: 4,
    candidate_id: 'h2:es:5:attempt4',
    problems: ['Placeholder [[M1]] is missing from unit U5.'],
  },
  {
    activity_id: 'a8',
    event_type: 'translation_validator_contract_retry',
    language_key: 'fr',
    doc_title: 'Paper A',
    part_start: 1,
    part_end: 1,
    round_index: 2,
    attempt: 1,
    problem: 'decisions must list every attempt',
  },
  {
    activity_id: 'a9',
    event_type: 'translation_repair_required',
    message: 'Raise Max Output Tokens for the Translator Submitter.',
    route_identity: 'Foreign Language Translation w/ Top-p · Translator Submitter · openrouter/alpha',
  },
  {
    activity_id: 'a10',
    event_type: 'provider_retry_cooldown',
    workflow_mode: 'translation',
    route_identity: 'Foreign Language Translation w/ Top-p · Translator Validator · openrouter/beta',
    retry_after_seconds: 30,
  },
  {
    activity_id: 'a11',
    event_type: 'provider_retry_cooldown',
    workflow_mode: 'translation',
    message: 'OpenRouter is busy; retrying soon.',
  },
  {
    activity_id: 'a12',
    event_type: 'translation_part_finalized',
    language_key: 'fr',
    doc_title: 'Paper A',
    part_start: 1,
    part_end: 1,
    final_attempt: 3,
    reason: 'three_accepted',
  },
];

function contextValue(overrides = {}) {
  const currentStatus = overrides.status === undefined ? status() : overrides.status;
  return {
    ...INERT_AUTO_TRANSLATE_CONTEXT,
    available: true,
    status: currentStatus,
    enabled: Boolean(currentStatus?.enabled),
    languages: currentStatus?.languages || [],
    refreshStatus: vi.fn(async () => {}),
    recentActivity: [],
    ...overrides,
  };
}

function renderModal({ context = contextValue(), ...props } = {}) {
  const user = userEvent.setup();
  const onClose = props.onClose || vi.fn();
  const onStatusChanged = props.onStatusChanged || vi.fn();
  const view = render(
    <AutoTranslateContext.Provider value={context}>
      <AutoTranslateModal
        isOpen
        capabilities={{ lmStudioEnabled: true }}
        credentialStatusRefreshToken={0}
        developerModeEnabled={false}
        {...props}
        onClose={onClose}
        onStatusChanged={onStatusChanged}
      />
    </AutoTranslateContext.Provider>,
  );
  return { ...view, user, onClose, onStatusChanged, context };
}

function card(title) {
  return screen.getByRole('heading', { name: title }).closest('.submitter-config-section');
}

function fieldIn(container, label) {
  const row = within(container).getByText(label, { selector: 'label' }).closest('.settings-row');
  return row.querySelector('select, input');
}

function deferred() {
  let resolve;
  let reject;
  const promise = new Promise((resolvePromise, rejectPromise) => {
    resolve = resolvePromise;
    reject = rejectPromise;
  });
  return { promise, resolve, reject };
}

beforeEach(() => {
  vi.clearAllMocks();
  translationAPI.getSettings.mockResolvedValue({ success: true, settings: SAVED_SETTINGS });
  translationAPI.saveSettings.mockImplementation(async (payload) => ({ success: true, settings: payload, status: status() }));
  translationAPI.control.mockResolvedValue({ success: true, status: status() });
  translationAPI.getLanguages.mockResolvedValue(CATALOG);
  translationAPI.probeLanguages.mockResolvedValue(CATALOG);
  translationAPI.clearCache.mockResolvedValue({ success: true, status: status() });
  openRouterAPI.getApiKeyStatus.mockResolvedValue({ has_key: true });
  openRouterAPI.getModels.mockResolvedValue({ models: [{ id: 'openrouter/alpha', name: 'Alpha', context_length: 200000 }] });
  openRouterAPI.getProviders.mockResolvedValue({ providers: [], endpoints: [] });
  api.getModels.mockResolvedValue({ models: [{ id: 'lm-one' }] });
  useCloudAccessProviders.mockReturnValue({
    statusByProvider: {},
    configuredProviders: [],
    modelErrors: [],
    modelsFor: () => [],
  });
});

describe('dialog', () => {
  test('renders nothing and requests nothing while closed', () => {
    const { container } = render(<AutoTranslateModal isOpen={false} onClose={vi.fn()} />);
    expect(container).toBeEmptyDOMElement();
    expect(translationAPI.getSettings).not.toHaveBeenCalled();
  });

  test('is a labelled modal dialog with plain-language intro that takes focus and closes on Escape', async () => {
    const { user, onClose } = renderModal();
    const dialog = screen.getByRole('dialog', { name: 'Foreign Language Translation w/ Top-p' });
    expect(dialog).toHaveAttribute('aria-modal', 'true');
    expect(within(dialog).getByText(/works through each document one part at a time/i)).toBeInTheDocument();
    expect(within(dialog).getByText(
      /translates the papers, proofs, final answers, and brainstorms from your current runs into the languages/,
    )).toBeInTheDocument();
    expect(within(dialog).queryByText(/several translator models/i)).not.toBeInTheDocument();
    expect(within(dialog).getByText(/Math, code, and Lean proof code stay untranslated/)).toBeInTheDocument();
    // The enable section explains Top-p Exploration and links to its page.
    const topP = within(dialog).getByRole('link', { name: 'Top-p Exploration' });
    expect(topP).toHaveAttribute('href', 'https://intrafere.com/structured-brainstorming-validated-feedback/');
    expect(topP).toHaveAttribute('target', '_blank');
    expect(topP).toHaveAttribute('rel', 'noopener noreferrer');
    expect(topP.closest('p')).toHaveTextContent(/makes MOTO so smart and lets it run autonomously/);
    expect(within(dialog).getByRole('checkbox', { name: 'Enable Foreign Language Translation w/ Top-p' }))
      .toBeInTheDocument();
    const close = screen.getByRole('button', { name: 'Close Foreign Language Translation w/ Top-p' });
    expect(close).toHaveAttribute('type', 'button');
    expect(close).toHaveFocus();
    await user.keyboard('{Escape}');
    expect(onClose).toHaveBeenCalledTimes(1);
    await user.click(close);
    expect(onClose).toHaveBeenCalledTimes(2);
    await screen.findByRole('list', { name: 'Most spoken' });
  });

  test('returns focus to the opener when it closes', async () => {
    const user = userEvent.setup();
    function Opener() {
      const [open, setOpen] = useState(false);
      return (
        <AutoTranslateContext.Provider value={contextValue()}>
          <button type="button" onClick={() => setOpen(true)}>Open translate</button>
          <AutoTranslateModal isOpen={open} onClose={() => setOpen(false)} capabilities={{}} />
        </AutoTranslateContext.Provider>
      );
    }
    render(<Opener />);
    const opener = screen.getByRole('button', { name: 'Open translate' });
    await user.click(opener);
    expect(screen.getByRole('button', { name: 'Close Foreign Language Translation w/ Top-p' })).toHaveFocus();
    await screen.findByRole('list', { name: 'Most spoken' });
    await user.keyboard('{Escape}');
    expect(screen.queryByRole('dialog')).not.toBeInTheDocument();
    expect(opener).toHaveFocus();
  });

  test('tabs follow the tablist pattern with arrow, Home, and End keys', async () => {
    const { user } = renderModal();
    const tablist = screen.getByRole('tablist', { name: 'Foreign Language Translation w/ Top-p sections' });
    const tabs = within(tablist).getAllByRole('tab');
    expect(tabs.map((tab) => tab.textContent)).toEqual(['Languages', 'Translator models', 'Progress']);
    expect(tabs[0]).toHaveAttribute('aria-selected', 'true');
    expect(tabs[0]).toHaveAttribute('tabindex', '0');
    expect(tabs[1]).toHaveAttribute('tabindex', '-1');
    const languagesPanel = screen.getByRole('tabpanel', { name: 'Languages' });
    expect(tabs[0]).toHaveAttribute('aria-controls', languagesPanel.id);

    tabs[0].focus();
    await user.keyboard('{ArrowRight}');
    expect(tabs[1]).toHaveFocus();
    expect(tabs[1]).toHaveAttribute('aria-selected', 'true');
    expect(tabs[0]).toHaveAttribute('aria-selected', 'false');
    expect(screen.getByRole('tabpanel', { name: 'Translator models' })).toBeVisible();
    expect(languagesPanel).not.toBeVisible();

    await user.keyboard('{End}');
    expect(tabs[2]).toHaveFocus();
    expect(screen.getByRole('tabpanel', { name: 'Progress' })).toBeVisible();
    await user.keyboard('{ArrowRight}');
    expect(tabs[0]).toHaveFocus();
    await user.keyboard('{ArrowLeft}');
    expect(tabs[2]).toHaveFocus();
    await user.keyboard('{Home}');
    expect(tabs[0]).toHaveAttribute('aria-selected', 'true');

    await user.click(tabs[1]);
    expect(tabs[1]).toHaveAttribute('aria-selected', 'true');
  });

  test('Tab and Shift+Tab wrap inside the dialog while hidden tab panels stay out of the focus order', async () => {
    const { user } = renderModal();
    await screen.findByRole('list', { name: 'Most spoken' });
    const close = screen.getByRole('button', { name: 'Close Foreign Language Translation w/ Top-p' });
    const add = screen.getByRole('button', { name: 'Add' });
    const panels = screen.getAllByRole('tabpanel', { hidden: true });
    expect(panels.map((panel) => panel.getAttribute('tabindex'))).toEqual(['0', '-1', '-1']);
    expect(close).toHaveFocus();

    await user.tab({ shift: true });
    expect(add).toHaveFocus();

    await user.tab();
    expect(close).toHaveFocus();
  });

  test('shows a settings load failure with a retry', async () => {
    translationAPI.getSettings.mockRejectedValueOnce(new Error('Failed to get Foreign Language Translation w/ Top-p settings: backend unavailable'));
    const { user } = renderModal();
    expect(await screen.findByRole('alert')).toHaveTextContent('backend unavailable');
    expect(screen.getByRole('checkbox', { name: 'Batch translation' })).toBeDisabled();
    await user.click(screen.getByRole('button', { name: 'Try again' }));
    expect(await screen.findByRole('list', { name: 'Most spoken' })).toBeInTheDocument();
    expect(screen.getByRole('checkbox', { name: 'Batch translation' })).toBeEnabled();
    expect(translationAPI.getSettings).toHaveBeenCalledTimes(2);
  });
});

describe('controls', () => {
  test('the default language lists Original and the selected languages and sets the default', async () => {
    const setDisplayLanguage = vi.fn();
    const { user } = renderModal({ context: contextValue({ displayLanguage: 'es', setDisplayLanguage }) });
    const select = screen.getByLabelText('Default language');
    expect(select).toHaveValue('es');
    expect([...select.options].map((option) => option.textContent)).toEqual(['Original', 'French', 'Spanish']);

    await user.selectOptions(select, 'fr');
    expect(setDisplayLanguage).toHaveBeenCalledWith('fr');
    await user.hover(screen.getByRole('button', { name: 'Learn about the default language' }));
    expect((await screen.findAllByText(/Each item's Language menu can switch just that item/)).length).toBeGreaterThan(0);
  });

  test('the enable checkbox enables and disables through control()', async () => {
    const disabledStatus = status({ enabled: false, state: 'disabled', status_label: 'disabled', message: 'Foreign Language Translation w/ Top-p is off.' });
    translationAPI.control.mockResolvedValueOnce({ success: true, status: status({ message: 'Translating now.' }) });
    const { user, context, onStatusChanged } = renderModal({ context: contextValue({ status: disabledStatus }) });
    const checkbox = screen.getByRole('checkbox', { name: 'Enable Foreign Language Translation w/ Top-p' });
    expect(checkbox).not.toBeChecked();
    expect(screen.queryByRole('button', { name: 'Pause' })).not.toBeInTheDocument();
    expect(screen.getByText('Foreign Language Translation w/ Top-p is off.')).toBeInTheDocument();

    await user.click(checkbox);
    expect(translationAPI.control).toHaveBeenLastCalledWith('enable');
    await waitFor(() => expect(checkbox).toBeChecked());
    expect(screen.getByText('Translating now.')).toBeInTheDocument();
    expect(context.refreshStatus).toHaveBeenCalled();
    expect(onStatusChanged).toHaveBeenCalled();

    translationAPI.control.mockResolvedValueOnce({ success: true, status: disabledStatus });
    await user.click(checkbox);
    expect(translationAPI.control).toHaveBeenLastCalledWith('disable');
    await waitFor(() => expect(checkbox).not.toBeChecked());
  });

  test('a language save racing an Enable click never flips the checkbox back', async () => {
    const disabledStatus = status({ enabled: false, state: 'disabled', generation: 4 });
    let finishControl;
    let finishSave;
    translationAPI.control.mockImplementationOnce(() => new Promise((resolve) => { finishControl = resolve; }));
    translationAPI.saveSettings.mockImplementationOnce((payload) => new Promise((resolve) => {
      finishSave = () => resolve({
        success: true,
        settings: payload,
        // Computed before the enable was applied, but delivered after its reply.
        status: status({ enabled: false, state: 'disabled', generation: 5 }),
      });
    }));
    const { user, context } = renderModal({ context: contextValue({ status: disabledStatus }) });
    await screen.findByRole('list', { name: 'Most spoken' });
    const checkbox = screen.getByRole('checkbox', { name: 'Enable Foreign Language Translation w/ Top-p' });

    await user.click(checkbox);
    await user.click(screen.getByRole('checkbox', { name: /^French/ }));
    await waitFor(() => expect(translationAPI.saveSettings).toHaveBeenCalledTimes(1));

    finishControl({ success: true, status: status({ enabled: true, generation: 6 }) });
    await waitFor(() => expect(checkbox).toBeChecked());
    finishSave();
    // Both requests have been handled once each has asked the provider to refresh.
    await waitFor(() => expect(context.refreshStatus).toHaveBeenCalledTimes(2));
    expect(checkbox).toBeChecked();
    expect(screen.getByRole('button', { name: 'Pause' })).toBeInTheDocument();
  });

  test('pause and resume', async () => {
    translationAPI.control.mockResolvedValueOnce({
      success: true,
      status: status({ paused: true, state: 'paused', status_label: 'paused', message: 'Foreign Language Translation w/ Top-p is paused.' }),
    });
    const { user } = renderModal();
    await user.click(screen.getByRole('button', { name: 'Pause' }));
    expect(translationAPI.control).toHaveBeenLastCalledWith('pause');
    const resume = await screen.findByRole('button', { name: 'Resume' });
    await user.click(resume);
    expect(translationAPI.control).toHaveBeenLastCalledWith('resume');
  });

  test('control errors are shown inline', async () => {
    translationAPI.control.mockRejectedValueOnce(new Error('Failed to update Foreign Language Translation w/ Top-p: backend unavailable'));
    const { user } = renderModal();
    await user.click(screen.getByRole('button', { name: 'Pause' }));
    expect(await screen.findByText(/Failed to update Foreign Language Translation w\/ Top-p: backend unavailable/)).toHaveAttribute('role', 'alert');
  });

  test('repair-required status shows the repair message, guidance, route, and Retry', async () => {
    const repairStatus = status({
      state: 'repair_required',
      status_label: 'repair required',
      message: 'Repair needed.',
      repair: {
        reason: 'output_truncated',
        message: 'Translator Submitter kept hitting its max output token limit.',
        guidance: 'Raise Max Output Tokens for the Translator Submitter, then retry.',
        role_id: 'translation_submitter',
        route_identity: 'Foreign Language Translation w/ Top-p · Translator Submitter · openrouter/alpha',
      },
    });
    const { user } = renderModal({ context: contextValue({ status: repairStatus }) });
    const banner = screen.getByText('Translator Submitter kept hitting its max output token limit.').closest('.auto-translate-banner');
    expect(banner).toHaveClass('test-result-banner--error');
    expect(within(banner).getByText('Raise Max Output Tokens for the Translator Submitter, then retry.')).toBeInTheDocument();
    expect(within(banner).getByText('Foreign Language Translation w/ Top-p · Translator Submitter · openrouter/alpha')).toBeInTheDocument();
    await user.click(within(banner).getByRole('button', { name: 'Retry' }));
    expect(translationAPI.control).toHaveBeenCalledWith('retry');
  });

  test('banner variants follow the service state', () => {
    expect(statusBannerVariant('translating')).toBe('success');
    expect(statusBannerVariant('idle')).toBe('success');
    expect(statusBannerVariant('paused')).toBe('warning');
    expect(statusBannerVariant('waiting_for_credits')).toBe('warning');
    expect(statusBannerVariant('repair_required')).toBe('error');
    expect(statusBannerVariant('disabled')).toBe('neutral');
  });
});

describe('batch translation', () => {
  test('the checkbox explains itself and saves immediately with the other saved settings', async () => {
    translationAPI.saveSettings.mockImplementation(async (payload) => ({
      success: true,
      settings: payload,
      status: status({ batch_translation: payload.batch_translation, max_parallel_jobs: payload.batch_translation ? 3 : 1 }),
    }));
    const { user } = renderModal();
    await screen.findByRole('list', { name: 'Most spoken' });
    const checkbox = screen.getByRole('checkbox', { name: 'Batch translation' });
    expect(checkbox).not.toBeChecked();
    expect(checkbox).toBeEnabled();
    expect(checkbox).toHaveAccessibleDescription(BATCH_TRANSLATION_HELP);
    expect(BATCH_TRANSLATION_HELP).toBe(
      'Translate several documents at once when your connection allows it: OpenRouter and cloud providers, or LM '
        + 'Studio with several loaded copies of the model. More documents finish sooner, but it uses more model '
        + 'capacity and can slow down research and the rest of MOTO.',
    );

    await user.click(checkbox);
    expect(checkbox).toBeChecked();
    await waitFor(() => expect(translationAPI.saveSettings).toHaveBeenCalledTimes(1));
    expect(translationAPI.saveSettings).toHaveBeenCalledWith({
      enabled: true,
      paused: false,
      batch_translation: true,
      languages: [ENGLISH],
      translator: ROLE('openrouter/alpha'),
      validator: ROLE('openrouter/beta'),
      modes: STANDARD_MODES,
    });
    expect(await screen.findByText('Up to 3 documents translate at once.')).toBeInTheDocument();
    expect(checkbox).toBeChecked();

    await user.click(checkbox);
    await waitFor(() => expect(translationAPI.saveSettings).toHaveBeenCalledTimes(2));
    expect(translationAPI.saveSettings.mock.calls[1][0]).toMatchObject({ batch_translation: false, languages: [ENGLISH] });
    await waitFor(() => expect(checkbox).not.toBeChecked());
    expect(screen.queryByText(/documents? translate at once|one document at a time/)).not.toBeInTheDocument();
  });

  test('the help button shows the explanation without toggling the setting', async () => {
    const { user } = renderModal();
    await screen.findByRole('list', { name: 'Most spoken' });
    const help = screen.getByRole('button', { name: 'Learn about Batch translation' });
    await user.hover(help);
    expect(screen.getAllByText(BATCH_TRANSLATION_HELP).some((node) => !node.classList.contains('visually-hidden'))).toBe(true);
    await user.click(help);
    expect(screen.getByRole('checkbox', { name: 'Batch translation' })).not.toBeChecked();
    expect(translationAPI.saveSettings).not.toHaveBeenCalled();
  });

  test('is disabled until the saved settings load, then shows the saved value', async () => {
    const settings = deferred();
    translationAPI.getSettings.mockReturnValueOnce(settings.promise);
    renderModal();
    const checkbox = screen.getByRole('checkbox', { name: 'Batch translation' });
    expect(checkbox).toBeDisabled();
    settings.resolve({ success: true, settings: { ...SAVED_SETTINGS, batch_translation: true } });
    await waitFor(() => expect(checkbox).toBeEnabled());
    expect(checkbox).toBeChecked();
  });

  test('a failed save shows the error and the saved value again', async () => {
    translationAPI.saveSettings.mockRejectedValueOnce(Object.assign(
      new Error('HTTP 422 Unprocessable Entity: Settings could not be applied.'),
      { status: 422 },
    ));
    const { user } = renderModal();
    await screen.findByRole('list', { name: 'Most spoken' });
    const checkbox = screen.getByRole('checkbox', { name: 'Batch translation' });
    await user.click(checkbox);
    expect(await screen.findByRole('alert')).toHaveTextContent('Settings could not be applied.');
    expect(checkbox).not.toBeChecked();
  });

  test('quick toggles save in order and the checkbox follows the latest request', async () => {
    const firstSave = deferred();
    translationAPI.saveSettings.mockImplementationOnce((payload) => firstSave.promise.then(() => ({
      success: true,
      settings: payload,
      status: status({ batch_translation: true }),
    })));
    const { user } = renderModal();
    await screen.findByRole('list', { name: 'Most spoken' });
    const checkbox = screen.getByRole('checkbox', { name: 'Batch translation' });
    await user.click(checkbox);
    await user.click(checkbox);
    expect(checkbox).not.toBeChecked();
    expect(translationAPI.saveSettings).toHaveBeenCalledTimes(1);

    firstSave.resolve();
    await waitFor(() => expect(translationAPI.saveSettings).toHaveBeenCalledTimes(2));
    expect(checkbox).not.toBeChecked();
    expect(translationAPI.saveSettings.mock.calls.map(([payload]) => payload.batch_translation)).toEqual([true, false]);
    await waitFor(() => expect(checkbox).not.toBeChecked());
  });

  test('capacity text', () => {
    expect(batchCapacityText(1)).toBe('Your current models translate one document at a time.');
    expect(batchCapacityText(4)).toBe('Up to 4 documents translate at once.');
    expect(batchCapacityText(0)).toBe('');
    expect(batchCapacityText(undefined)).toBe('');
  });
});

function modesTrigger() {
  return screen.getByRole('button', { name: /^Current runs:/ });
}

function modesPanel() {
  return document.getElementById(modesTrigger().getAttribute('aria-controls'));
}

describe('current runs checklist', () => {
  test('lists the standard modes in their workspace groups, all checked, with descriptions and help', async () => {
    const { user } = renderModal();
    await screen.findByRole('list', { name: 'Most spoken' });
    const trigger = modesTrigger();
    expect(trigger).toHaveAccessibleName('Current runs: All modes');
    expect(trigger).toHaveAttribute('aria-expanded', 'false');
    expect(trigger).toHaveAccessibleDescription(TRANSLATION_MODES_HELP);
    expect(TRANSLATION_MODES_HELP).toBe(
      'Choose which modes have their current run translated automatically. New papers, proofs, final answers, '
        + 'completed brainstorms, and results from each checked mode\'s current run are translated as they appear. '
        + 'Older runs of a checked mode translate when you open an item in your language, or all at once with '
        + 'Translate all older items. Items from unchecked modes translate only when you open them. Unchecking a '
        + 'mode keeps the translations already finished.',
    );
    // Rendered inline in the dialog (portaled content would be inert) and hidden while closed.
    expect(screen.getByRole('dialog')).toContainElement(modesPanel());
    expect(modesPanel()).toHaveAttribute('hidden');

    await user.click(trigger);
    expect(trigger).toHaveAttribute('aria-expanded', 'true');
    const list = screen.getByRole('group', { name: 'Translate the current run of' });
    expect(list).toBe(modesPanel());
    const autonomousGroup = within(list).getByRole('group', { name: 'Autonomous S.T.E.M. ASI' });
    const autonomous = within(autonomousGroup).getByRole('checkbox', { name: 'Autonomous Research' });
    expect(autonomous).toBeChecked();
    expect(autonomous).toHaveAccessibleDescription(
      'The active research session: papers, final answer, proofs, and completed brainstorms.',
    );
    const manualGroup = within(list).getByRole('group', { name: 'Advanced Manual S.T.E.M. ASI' });
    expect(within(manualGroup).getAllByRole('checkbox')).toHaveLength(2);
    expect(within(manualGroup).getByRole('checkbox', { name: 'Aggregator' })).toHaveAccessibleDescription(
      'Live Results and proofs found in the Aggregator database (including Single Paper Writer proofs-only runs).',
    );
    expect(within(manualGroup).getByRole('checkbox', { name: 'Single Paper Writer' }))
      .toHaveAccessibleDescription('The Live Paper and the proofs found in it.');
    expect(within(list).getAllByRole('checkbox').every((box) => box.checked)).toBe(true);
    expect(within(list).queryByRole('group', { name: 'LeanOJ' })).not.toBeInTheDocument();
    expect(screen.queryByRole('checkbox', { name: 'LeanOJ Proof Solver' })).not.toBeInTheDocument();
  });

  test('each change saves at once with the other saved settings and the summary follows', async () => {
    const { user } = renderModal();
    await screen.findByRole('list', { name: 'Most spoken' });
    await user.click(modesTrigger());

    await user.click(screen.getByRole('checkbox', { name: 'Aggregator' }));
    await waitFor(() => expect(translationAPI.saveSettings).toHaveBeenCalledTimes(1));
    expect(translationAPI.saveSettings).toHaveBeenCalledWith({
      enabled: true,
      paused: false,
      batch_translation: false,
      languages: [ENGLISH],
      translator: ROLE('openrouter/alpha'),
      validator: ROLE('openrouter/beta'),
      modes: ['autonomous', 'compiler'],
    });
    expect(modesTrigger()).toHaveAccessibleName('Current runs: Autonomous, Single Paper Writer');
    expect(modesTrigger()).toHaveAttribute('aria-expanded', 'true');

    await user.click(screen.getByRole('checkbox', { name: 'Autonomous Research' }));
    await user.click(screen.getByRole('checkbox', { name: 'Single Paper Writer' }));
    await waitFor(() => expect(translationAPI.saveSettings).toHaveBeenCalledTimes(3));
    expect(translationAPI.saveSettings.mock.calls[2][0].modes).toEqual([]);
    expect(modesTrigger()).toHaveAccessibleName('Current runs: None — only items you open');

    await user.click(screen.getByRole('checkbox', { name: 'Single Paper Writer' }));
    await user.click(screen.getByRole('checkbox', { name: 'Aggregator' }));
    await waitFor(() => expect(translationAPI.saveSettings).toHaveBeenCalledTimes(5));
    // Modes are always sent in the backend's order, whatever order they were checked in.
    expect(translationAPI.saveSettings.mock.calls[4][0].modes).toEqual(['aggregator', 'compiler']);
  });

  test('Escape closes only the list and returns focus to it; a second Escape closes the dialog', async () => {
    const { user, onClose } = renderModal();
    await screen.findByRole('list', { name: 'Most spoken' });
    await user.click(modesTrigger());
    const checkbox = screen.getByRole('checkbox', { name: 'Single Paper Writer' });
    checkbox.focus();

    await user.keyboard('{Escape}');
    expect(modesPanel()).toHaveAttribute('hidden');
    expect(modesTrigger()).toHaveAttribute('aria-expanded', 'false');
    expect(modesTrigger()).toHaveFocus();
    expect(onClose).not.toHaveBeenCalled();
    expect(screen.getByRole('dialog')).toBeInTheDocument();

    await user.keyboard('{Escape}');
    expect(onClose).toHaveBeenCalledTimes(1);
  });

  test('a mousedown outside the list closes it; one inside keeps it open', async () => {
    const { user } = renderModal();
    await screen.findByRole('list', { name: 'Most spoken' });
    await user.click(modesTrigger());
    fireEvent.mouseDown(screen.getByText('Translate the current run of'));
    expect(modesPanel()).not.toHaveAttribute('hidden');

    fireEvent.mouseDown(screen.getByRole('heading', { name: 'Foreign Language Translation w/ Top-p' }));
    expect(modesPanel()).toHaveAttribute('hidden');
    expect(translationAPI.saveSettings).not.toHaveBeenCalled();
  });

  test('works from the keyboard: Enter opens, Tab reaches the modes, Space toggles', async () => {
    const { user } = renderModal();
    await screen.findByRole('list', { name: 'Most spoken' });
    modesTrigger().focus();
    await user.keyboard('{Enter}');
    expect(modesTrigger()).toHaveAttribute('aria-expanded', 'true');

    await user.tab();
    expect(screen.getByRole('button', { name: 'Learn about which modes are translated' })).toHaveFocus();
    await user.tab();
    const autonomous = screen.getByRole('checkbox', { name: 'Autonomous Research' });
    expect(autonomous).toHaveFocus();
    await user.keyboard(' ');
    await waitFor(() => expect(translationAPI.saveSettings).toHaveBeenCalledTimes(1));
    expect(translationAPI.saveSettings.mock.calls[0][0].modes).toEqual(['aggregator', 'compiler']);
  });

  test('the help shows its explanation on hover without opening the list', async () => {
    const { user } = renderModal();
    await screen.findByRole('list', { name: 'Most spoken' });
    const help = screen.getByRole('button', { name: 'Learn about which modes are translated' });
    await user.hover(help);
    expect(screen.getAllByText(TRANSLATION_MODES_HELP).some((node) => !node.classList.contains('visually-hidden'))).toBe(true);
    await user.click(help);
    expect(modesTrigger()).toHaveAttribute('aria-expanded', 'false');
    expect(translationAPI.saveSettings).not.toHaveBeenCalled();
  });

  test('developer mode adds LeanOJ in its own group, unchecked by default, and saves it', async () => {
    const { user } = renderModal({ developerModeEnabled: true });
    await screen.findByRole('list', { name: 'Most spoken' });
    expect(modesTrigger()).toHaveAccessibleName('Current runs: Autonomous, Aggregator, Single Paper Writer');
    await user.click(modesTrigger());
    const group = screen.getByRole('group', { name: 'LeanOJ' });
    const leanoj = within(group).getByRole('checkbox', { name: 'LeanOJ Proof Solver' });
    expect(leanoj).not.toBeChecked();
    expect(leanoj).toHaveAccessibleDescription('Proofs from the loaded Proof Solver run.');

    await user.click(leanoj);
    await waitFor(() => expect(translationAPI.saveSettings).toHaveBeenCalledTimes(1));
    expect(translationAPI.saveSettings.mock.calls[0][0].modes).toEqual([...STANDARD_MODES, 'leanoj']);
    expect(modesTrigger()).toHaveAccessibleName('Current runs: All modes');
  });

  test('without developer mode a saved LeanOJ mode is hidden and left out of every save', async () => {
    translationAPI.getSettings.mockResolvedValue({
      success: true,
      settings: { ...SAVED_SETTINGS, modes: [...STANDARD_MODES, 'leanoj'] },
    });
    const { user } = renderModal();
    await screen.findByRole('list', { name: 'Most spoken' });
    expect(modesTrigger()).toHaveAccessibleName('Current runs: All modes');
    await user.click(modesTrigger());
    expect(screen.queryByRole('checkbox', { name: 'LeanOJ Proof Solver' })).not.toBeInTheDocument();

    await user.click(screen.getByRole('checkbox', { name: 'Batch translation' }));
    await waitFor(() => expect(translationAPI.saveSettings).toHaveBeenCalledTimes(1));
    expect(translationAPI.saveSettings.mock.calls[0][0].modes).toEqual(STANDARD_MODES);
    // Clicking Batch translation was outside the list, which closed it.
    expect(modesTrigger()).toHaveAttribute('aria-expanded', 'false');
    await user.click(modesTrigger());
    await user.click(screen.getByRole('checkbox', { name: 'Aggregator' }));
    await waitFor(() => expect(translationAPI.saveSettings).toHaveBeenCalledTimes(2));
    expect(translationAPI.saveSettings.mock.calls[1][0].modes).toEqual(['autonomous', 'compiler']);
  });

  test('a failed save shows the error and the saved modes again', async () => {
    translationAPI.saveSettings.mockRejectedValueOnce(Object.assign(
      new Error('HTTP 422 Unprocessable Entity: Settings could not be applied.'),
      { status: 422 },
    ));
    const { user } = renderModal();
    await screen.findByRole('list', { name: 'Most spoken' });
    await user.click(modesTrigger());
    await user.click(screen.getByRole('checkbox', { name: 'Aggregator' }));
    expect(await screen.findByRole('alert')).toHaveTextContent('Settings could not be applied.');
    expect(screen.getByRole('checkbox', { name: 'Aggregator' })).toBeChecked();
    expect(modesTrigger()).toHaveAccessibleName('Current runs: All modes');
  });

  test('is disabled until the saved settings load', async () => {
    const settings = deferred();
    translationAPI.getSettings.mockReturnValueOnce(settings.promise);
    renderModal();
    expect(modesTrigger()).toBeDisabled();
    settings.resolve({ success: true, settings: SAVED_SETTINGS });
    await waitFor(() => expect(modesTrigger()).toBeEnabled());
    expect(modesTrigger()).toHaveAccessibleName('Current runs: All modes');
  });

  const { modes: savedModes, ...SETTINGS_WITHOUT_MODES } = SAVED_SETTINGS;

  test('settings from a backend without modes leave the list unavailable and saves without modes', async () => {
    expect(savedModes).toEqual(STANDARD_MODES);
    translationAPI.getSettings.mockResolvedValue({ success: true, settings: SETTINGS_WITHOUT_MODES });
    const { user } = renderModal();
    await screen.findByRole('list', { name: 'Most spoken' });
    const restart = 'Restart MOTO to choose translated modes.';
    expect(screen.getByText(restart)).toBeVisible();
    expect(modesTrigger()).toBeDisabled();
    expect(modesTrigger()).toHaveAccessibleName('Current runs: Unavailable');
    expect(modesTrigger()).toHaveAccessibleDescription(`${restart} ${TRANSLATION_MODES_HELP}`);

    await user.click(screen.getByRole('checkbox', { name: 'Batch translation' }));
    await waitFor(() => expect(translationAPI.saveSettings).toHaveBeenCalledTimes(1));
    expect(translationAPI.saveSettings.mock.calls[0][0]).not.toHaveProperty('modes');
    expect(modesTrigger()).toBeDisabled();

    // A restarted backend replies with its modes, and later saves send them again.
    translationAPI.saveSettings.mockImplementationOnce(async (payload) => ({
      success: true,
      settings: { ...payload, modes: ['autonomous'] },
      status: status(),
    }));
    await user.click(screen.getByRole('checkbox', { name: 'Batch translation' }));
    await waitFor(() => expect(modesTrigger()).toBeEnabled());
    expect(screen.queryByText(restart)).not.toBeInTheDocument();
    expect(modesTrigger()).toHaveAccessibleName('Current runs: Autonomous');
    await user.click(screen.getByRole('checkbox', { name: 'Batch translation' }));
    await waitFor(() => expect(translationAPI.saveSettings).toHaveBeenCalledTimes(3));
    expect(translationAPI.saveSettings.mock.calls[2][0].modes).toEqual(['autonomous']);
  });

  test('a save reply without modes never puts back a mode the user unchecked', async () => {
    translationAPI.saveSettings.mockImplementation(async (payload) => {
      const { modes, ...older } = payload;
      return { success: true, settings: older, status: status() };
    });
    const { user } = renderModal();
    await screen.findByRole('list', { name: 'Most spoken' });
    await user.click(modesTrigger());
    await user.click(screen.getByRole('checkbox', { name: 'Aggregator' }));
    expect(await screen.findByText('Restart MOTO to choose translated modes.')).toBeVisible();
    expect(modesTrigger()).toHaveAccessibleName('Current runs: Unavailable');
    expect(modesTrigger()).toBeDisabled();
    expect(modesTrigger()).toHaveAttribute('aria-expanded', 'false');
    expect(modesPanel()).toHaveAttribute('hidden');

    await user.click(screen.getByRole('checkbox', { name: 'Batch translation' }));
    await waitFor(() => expect(translationAPI.saveSettings).toHaveBeenCalledTimes(2));
    expect(translationAPI.saveSettings.mock.calls[1][0]).not.toHaveProperty('modes');
  });
});

describe('translate all', () => {
  const translateAllButton = () => screen.getByRole('button', { name: 'Translate all older items' });

  test('asks first, starts through control(), and swaps to Cancel with keyboard focus kept', async () => {
    translationAPI.control.mockResolvedValueOnce({
      success: true,
      status: status({ generation: 4, translate_all: { ...TRANSLATE_ALL_OFF, requested: true, state: 'preparing' } }),
    });
    const { user } = renderModal();
    await screen.findByRole('list', { name: 'Most spoken' });
    const button = translateAllButton();
    expect(button).toBeEnabled();
    expect(button).toHaveAccessibleDescription(TRANSLATE_ALL_HELP);
    expect(TRANSLATE_ALL_HELP).toBe(
      'Translates everything else you can still open — papers, final answers, and proofs from older runs in your '
        + 'libraries and history, plus pruned papers — for the modes checked in the list. It runs in the background '
        + 'at a lower priority than current-run translations and anything you are viewing. When your translator '
        + 'connection can run more than one job at once, it runs alongside them; otherwise it waits until they '
        + 'finish. Content you cleared is never included. It stops by itself when everything is translated, and '
        + 'Cancel stops it early; finished translations are kept.',
    );

    await user.click(button);
    const confirm = screen.getByRole('group', { name: 'Confirm Translate all' });
    expect(confirm).toHaveTextContent(
      'Translate every older item still shown in your libraries for the checked modes? This can take a long time '
        + 'and uses translator model calls.',
    );
    expect(within(confirm).getByRole('button', { name: 'Not now' })).toHaveFocus();
    await user.click(within(confirm).getByRole('button', { name: 'Not now' }));
    expect(screen.queryByRole('group', { name: 'Confirm Translate all' })).not.toBeInTheDocument();
    expect(translateAllButton()).toHaveFocus();
    expect(translationAPI.control).not.toHaveBeenCalled();

    await user.click(translateAllButton());
    await user.click(screen.getByRole('button', { name: 'Translate all' }));
    expect(translationAPI.control).toHaveBeenCalledWith('translate_all');
    const cancel = await screen.findByRole('button', { name: 'Cancel Translate all' });
    expect(cancel).toHaveFocus();
    expect(cancel).toHaveAccessibleDescription(TRANSLATE_ALL_HELP);
    expect(screen.getByText('Finding older items…')).toHaveAttribute('role', 'status');
    expect(screen.queryByRole('group', { name: 'Confirm Translate all' })).not.toBeInTheDocument();
    expect(screen.queryByRole('button', { name: 'Translate all older items' })).not.toBeInTheDocument();
  });

  test.each([
    [
      'translation is off',
      { context: { status: status({ enabled: false, state: 'disabled' }) } },
      'Turn on translation first.',
    ],
    [
      'no languages are chosen',
      { settings: { ...SAVED_SETTINGS, languages: [] } },
      'Choose languages and translator models first.',
    ],
    [
      'translator models are missing',
      { settings: { ...SAVED_SETTINGS, validator: null } },
      'Choose languages and translator models first.',
    ],
    [
      'no mode is checked in Current runs',
      { settings: { ...SAVED_SETTINGS, modes: [] } },
      'Check at least one mode in Current runs first.',
    ],
    [
      'only a mode developer mode shows is checked',
      { settings: { ...SAVED_SETTINGS, modes: ['leanoj'] } },
      'Check at least one mode in Current runs first.',
    ],
  ])('is disabled and says why while %s', async (_, { context, settings }, reason) => {
    if (settings) {
      translationAPI.getSettings.mockResolvedValue({ success: true, settings });
    }
    renderModal(context ? { context: contextValue(context) } : {});
    await screen.findByRole('list', { name: 'Most spoken' });
    await waitFor(() => expect(translateAllButton()).toBeDisabled());
    expect(translateAllButton()).toHaveAccessibleDescription(`${reason} ${TRANSLATE_ALL_HELP}`);
    expect(screen.getByText(reason)).toBeVisible();
  });

  test.each([
    [
      'translating',
      {
        translate_all: {
          ...TRANSLATE_ALL_OFF,
          requested: true,
          state: 'translating',
          documents: 40,
          complete_documents: 12,
          pending_units: 3210,
        },
      },
      'Translating older items: 12 of 40 done.',
    ],
    [
      'waiting',
      { translate_all: { ...TRANSLATE_ALL_OFF, requested: true, state: 'waiting' } },
      'Waiting to translate older items.',
    ],
    [
      'waiting behind current-run work, as the backend says',
      {
        translate_all: {
          ...TRANSLATE_ALL_OFF,
          requested: true,
          state: 'waiting',
          message: 'Translate all waits for current-run and viewed translations.',
        },
      },
      'Translate all waits for current-run and viewed translations.',
    ],
    [
      'waiting while translation needs repair',
      {
        state: 'repair_required',
        translate_all: {
          ...TRANSLATE_ALL_OFF,
          requested: true,
          state: 'waiting',
          message: 'Translate all waits until translation can run again.',
        },
      },
      'Translate all waits until translation can run again.',
    ],
    [
      'preparing',
      { translate_all: { ...TRANSLATE_ALL_OFF, requested: true, state: 'preparing' } },
      'Finding older items…',
    ],
    [
      'paused',
      {
        paused: true,
        state: 'paused',
        translate_all: { ...TRANSLATE_ALL_OFF, requested: true, state: 'waiting' },
      },
      'Paused.',
    ],
  ])('while on and %s it shows its progress and Cancel stops it', async (_, overrides, line) => {
    translationAPI.control.mockResolvedValueOnce({ success: true, status: status({ generation: 4 }) });
    const { user } = renderModal({ context: contextValue({ status: status(overrides) }) });
    await screen.findByRole('list', { name: 'Most spoken' });
    expect(screen.getByText(line)).toHaveAttribute('role', 'status');
    expect(screen.queryByText(/Batch translation to/)).not.toBeInTheDocument();
    expect(screen.queryByRole('button', { name: 'Translate all older items' })).not.toBeInTheDocument();

    const cancel = screen.getByRole('button', { name: 'Cancel Translate all' });
    await user.click(cancel);
    expect(translationAPI.control).toHaveBeenCalledWith('cancel_translate_all');
    expect(await screen.findByRole('button', { name: 'Translate all older items' })).toHaveFocus();
    expect(screen.queryByText(line)).not.toBeInTheDocument();
  });

  test('status text is the backend message, else a line from the state that names no cause', () => {
    expect(translateAllStatusText({ state: 'translating', message: 'Translate all: 9 unit(s) left in the background.' }))
      .toBe('Translate all: 9 unit(s) left in the background.');
    expect(translateAllStatusText({ state: 'waiting', message: ' Translation is paused. ' }, { paused: true }))
      .toBe('Translation is paused.');
    expect(translateAllStatusText({ state: 'waiting', message: '' })).toBe('Waiting to translate older items.');
    expect(translateAllStatusText({ state: 'translating', documents: 2, complete_documents: 1, pending_units: 1 }))
      .toBe('Translating older items: 1 of 2 done.');
    expect(translateAllStatusText({ state: 'translating' }, { paused: true })).toBe('Paused.');
    expect(translateAllStatusText(null)).toBe('Finding older items…');
  });

  test('unchecking the last mode disables it at once; checking one again enables it', async () => {
    const { user } = renderModal();
    await screen.findByRole('list', { name: 'Most spoken' });
    await user.click(modesTrigger());
    for (const name of ['Autonomous Research', 'Aggregator', 'Single Paper Writer']) {
      await user.click(screen.getByRole('checkbox', { name }));
    }
    await waitFor(() => expect(translateAllButton()).toBeDisabled());
    expect(translateAllButton()).toHaveAccessibleDescription(
      `Check at least one mode in Current runs first. ${TRANSLATE_ALL_HELP}`,
    );

    await user.click(screen.getByRole('checkbox', { name: 'Aggregator' }));
    await waitFor(() => expect(translateAllButton()).toBeEnabled());
    expect(screen.queryByText('Check at least one mode in Current runs first.')).not.toBeInTheDocument();
  });

  test('settings from a backend without modes leave it enabled', async () => {
    const { modes, ...withoutModes } = SAVED_SETTINGS;
    translationAPI.getSettings.mockResolvedValue({ success: true, settings: withoutModes });
    renderModal();
    await screen.findByRole('list', { name: 'Most spoken' });
    await waitFor(() => expect(modesTrigger()).toHaveAccessibleName('Current runs: Unavailable'));
    expect(translateAllButton()).toBeEnabled();
  });

  test('a backend without the Translate all status block still shows the button', async () => {
    const { translate_all: omitted, ...olderStatus } = status();
    expect(omitted).toEqual(TRANSLATE_ALL_OFF);
    renderModal({ context: contextValue({ status: olderStatus }) });
    await screen.findByRole('list', { name: 'Most spoken' });
    expect(translateAllButton()).toBeEnabled();
    expect(screen.queryByRole('button', { name: 'Cancel Translate all' })).not.toBeInTheDocument();
  });

  test('the help shows its explanation on hover without starting anything', async () => {
    const { user } = renderModal();
    await screen.findByRole('list', { name: 'Most spoken' });
    const help = screen.getByRole('button', { name: 'Learn about Translate all' });
    await user.hover(help);
    expect(screen.getAllByText(TRANSLATE_ALL_HELP).some((node) => !node.classList.contains('visually-hidden'))).toBe(true);
    await user.click(help);
    expect(screen.queryByRole('group', { name: 'Confirm Translate all' })).not.toBeInTheDocument();
    expect(translationAPI.control).not.toHaveBeenCalled();
  });
});

describe('developer-only modes', () => {
  const LEANOJ_MODES = [...STANDARD_MODES, 'leanoj'];

  test('the settings payload keeps only known modes in order and leaves LeanOJ out without developer mode', () => {
    const settings = { ...SAVED_SETTINGS, modes: ['leanoj', 'compiler', 'future_mode', 'autonomous'] };
    const standard = buildTranslationSettingsPayload(settings, false);
    expect(Object.keys(standard).sort()).toEqual(
      ['batch_translation', 'enabled', 'languages', 'modes', 'paused', 'translator', 'validator'],
    );
    expect(standard.modes).toEqual(['autonomous', 'compiler']);
    expect(buildTranslationSettingsPayload(settings, true).modes).toEqual(['autonomous', 'compiler', 'leanoj']);
    // Settings from a backend that sent no modes save without them.
    expect(buildTranslationSettingsPayload({ ...SAVED_SETTINGS, modes: null }, true)).not.toHaveProperty('modes');
  });

  function renderGate({ developerModeEnabled = false, gateStatus = status({ modes: LEANOJ_MODES }) } = {}) {
    const refreshStatus = vi.fn(async () => {});
    const tree = (props) => (
      <AutoTranslateContext.Provider value={contextValue({ status: props.gateStatus, refreshStatus })}>
        <TranslationDeveloperModesGate developerModeEnabled={props.developerModeEnabled} />
      </AutoTranslateContext.Provider>
    );
    const view = render(tree({ developerModeEnabled, gateStatus }));
    const rerenderGate = (next) => view.rerender(tree({ developerModeEnabled, gateStatus, ...next }));
    return { ...view, refreshStatus, rerenderGate };
  }

  test('without developer mode a saved LeanOJ mode is removed as soon as the status shows it', async () => {
    translationAPI.getSettings.mockResolvedValue({
      success: true,
      settings: { ...SAVED_SETTINGS, modes: LEANOJ_MODES, translate_all: true },
    });
    const { refreshStatus } = renderGate();
    await waitFor(() => expect(translationAPI.saveSettings).toHaveBeenCalledTimes(1));
    const { translate_all: translateAll, ...expectedSettings } = SAVED_SETTINGS;
    expect(translateAll).toBe(false);
    expect(translationAPI.saveSettings).toHaveBeenCalledWith(expectedSettings);
    await waitFor(() => expect(refreshStatus).toHaveBeenCalledTimes(1));
  });

  test('turning developer mode off removes it, every time; developer mode keeps it', async () => {
    translationAPI.getSettings.mockResolvedValue({ success: true, settings: { ...SAVED_SETTINGS, modes: LEANOJ_MODES } });
    const { rerenderGate } = renderGate({ developerModeEnabled: true });
    await new Promise((resolve) => { setTimeout(resolve, 0); });
    expect(translationAPI.getSettings).not.toHaveBeenCalled();

    rerenderGate({ developerModeEnabled: false });
    await waitFor(() => expect(translationAPI.saveSettings).toHaveBeenCalledTimes(1));
    expect(translationAPI.saveSettings.mock.calls[0][0].modes).toEqual(STANDARD_MODES);

    rerenderGate({ developerModeEnabled: true });
    rerenderGate({ developerModeEnabled: false });
    await waitFor(() => expect(translationAPI.saveSettings).toHaveBeenCalledTimes(2));
  });

  test('leaves the settings alone without a developer-only mode, or with a status that has no modes', async () => {
    const { rerenderGate } = renderGate({ gateStatus: status({ modes: STANDARD_MODES }) });
    rerenderGate({ gateStatus: status() });
    await new Promise((resolve) => { setTimeout(resolve, 0); });
    expect(translationAPI.getSettings).not.toHaveBeenCalled();
    expect(translationAPI.saveSettings).not.toHaveBeenCalled();
  });

  test('saves nothing when the settings it reads already lack LeanOJ', async () => {
    const { refreshStatus } = renderGate();
    await waitFor(() => expect(translationAPI.getSettings).toHaveBeenCalledTimes(1));
    await new Promise((resolve) => { setTimeout(resolve, 0); });
    expect(translationAPI.saveSettings).not.toHaveBeenCalled();
    expect(refreshStatus).not.toHaveBeenCalled();
  });

  test('a failed cleanup is logged, not thrown, and runs again with the next status', async () => {
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => {});
    translationAPI.getSettings.mockResolvedValue({ success: true, settings: { ...SAVED_SETTINGS, modes: LEANOJ_MODES } });
    translationAPI.saveSettings.mockRejectedValueOnce(new Error('backend unavailable'));
    const { rerenderGate } = renderGate();
    await waitFor(() => expect(warn).toHaveBeenCalledWith(
      'Auto Top-p Translate modes could not be updated:',
      'backend unavailable',
    ));
    expect(translationAPI.saveSettings).toHaveBeenCalledTimes(1);

    rerenderGate({ gateStatus: status({ modes: LEANOJ_MODES }) });
    await waitFor(() => expect(translationAPI.saveSettings).toHaveBeenCalledTimes(2));
    expect(translationAPI.saveSettings.mock.calls[1][0].modes).toEqual(STANDARD_MODES);
    warn.mockRestore();
  });

  test('the cleanup waits for a window save in flight and builds on the settings it left', async () => {
    const languageSave = deferred();
    translationAPI.saveSettings.mockImplementationOnce((payload) => languageSave.promise.then(() => ({
      success: true,
      settings: payload,
      status: status(),
    })));
    const refreshStatus = vi.fn(async () => {});
    const tree = (modes) => (
      <AutoTranslateContext.Provider value={contextValue({ status: status({ modes }), refreshStatus })}>
        <AutoTranslateModal isOpen capabilities={{ lmStudioEnabled: true }} onClose={vi.fn()} />
        <TranslationDeveloperModesGate developerModeEnabled={false} />
      </AutoTranslateContext.Provider>
    );
    const user = userEvent.setup();
    const { rerender } = render(tree(STANDARD_MODES));
    await screen.findByRole('list', { name: 'Most spoken' });
    await user.click(screen.getByRole('checkbox', { name: /^French/ }));
    await waitFor(() => expect(translationAPI.saveSettings).toHaveBeenCalledTimes(1));

    // LeanOJ was checked in another browser while the language save is in flight.
    translationAPI.getSettings.mockResolvedValue({
      success: true,
      settings: { ...SAVED_SETTINGS, languages: [ENGLISH, FRENCH], modes: LEANOJ_MODES },
    });
    rerender(tree(LEANOJ_MODES));
    await new Promise((resolve) => { setTimeout(resolve, 0); });
    expect(translationAPI.getSettings).toHaveBeenCalledTimes(1);
    expect(translationAPI.saveSettings).toHaveBeenCalledTimes(1);

    languageSave.resolve();
    await waitFor(() => expect(translationAPI.saveSettings).toHaveBeenCalledTimes(2));
    // The window re-reads the settings the new modes came with, then the cleanup reads them.
    expect(translationAPI.getSettings).toHaveBeenCalledTimes(3);
    expect(translationAPI.saveSettings.mock.calls[1][0]).toMatchObject({
      languages: [ENGLISH, FRENCH],
      modes: STANDARD_MODES,
    });
  });

  // A fake backend whose status follows its saved modes, like the provider's refresh.
  function renderWindowWithGate(initialModes) {
    const backend = { settings: { ...SAVED_SETTINGS, modes: initialModes } };
    translationAPI.getSettings.mockImplementation(async () => ({ success: true, settings: backend.settings }));
    translationAPI.saveSettings.mockImplementation(async (payload) => {
      backend.settings = { ...backend.settings, ...payload };
      return { success: true, settings: backend.settings, status: status({ modes: backend.settings.modes }) };
    });
    function Harness({ developerModeEnabled }) {
      const [current, setCurrent] = useState(() => status({ modes: backend.settings.modes }));
      const refreshStatus = useCallback(async () => {
        setCurrent(status({ modes: backend.settings.modes }));
      }, []);
      return (
        <AutoTranslateContext.Provider value={contextValue({ status: current, refreshStatus })}>
          <AutoTranslateModal
            isOpen
            capabilities={{ lmStudioEnabled: true }}
            developerModeEnabled={developerModeEnabled}
            onClose={vi.fn()}
          />
          <TranslationDeveloperModesGate developerModeEnabled={developerModeEnabled} />
        </AutoTranslateContext.Provider>
      );
    }
    const view = render(<Harness developerModeEnabled />);
    const setDeveloperMode = (developerModeEnabled) => view.rerender(
      <Harness developerModeEnabled={developerModeEnabled} />,
    );
    return { backend, setDeveloperMode };
  }

  test('an open window shows the cleanup after developer mode turns off and on, and saves never undo it', async () => {
    const user = userEvent.setup();
    const { backend, setDeveloperMode } = renderWindowWithGate(LEANOJ_MODES);
    await screen.findByRole('list', { name: 'Most spoken' });
    await waitFor(() => expect(modesTrigger()).toBeEnabled());
    expect(modesTrigger()).toHaveAccessibleName('Current runs: All modes');

    setDeveloperMode(false);
    await waitFor(() => expect(backend.settings.modes).toEqual(STANDARD_MODES));
    await waitFor(() => expect(translationAPI.getSettings).toHaveBeenCalledTimes(3));

    setDeveloperMode(true);
    await waitFor(() => expect(modesTrigger()).toHaveAccessibleName(
      'Current runs: Autonomous, Aggregator, Single Paper Writer',
    ));
    await user.click(modesTrigger());
    expect(screen.getByRole('checkbox', { name: 'LeanOJ Proof Solver' })).not.toBeChecked();

    await user.click(screen.getByRole('checkbox', { name: 'Batch translation' }));
    await waitFor(() => expect(translationAPI.saveSettings).toHaveBeenCalledTimes(2));
    expect(translationAPI.saveSettings.mock.calls[1][0]).toMatchObject({
      batch_translation: true,
      modes: STANDARD_MODES,
    });
    // The window's own save already left the modes the status now shows.
    expect(translationAPI.getSettings).toHaveBeenCalledTimes(3);
  });

  test('a window reads the settings again only when the service modes change to ones it lacks', async () => {
    const refreshStatus = vi.fn(async () => {});
    const tree = (modes) => (
      <AutoTranslateContext.Provider value={contextValue({ status: status({ modes }), refreshStatus })}>
        <AutoTranslateModal isOpen capabilities={{ lmStudioEnabled: true }} onClose={vi.fn()} />
      </AutoTranslateContext.Provider>
    );
    const { rerender } = render(tree(STANDARD_MODES));
    await screen.findByRole('list', { name: 'Most spoken' });
    rerender(tree([...STANDARD_MODES].reverse()));
    rerender(tree(STANDARD_MODES));
    await new Promise((resolve) => { setTimeout(resolve, 0); });
    expect(translationAPI.getSettings).toHaveBeenCalledTimes(1);

    translationAPI.getSettings.mockResolvedValue({ success: true, settings: { ...SAVED_SETTINGS, modes: ['compiler'] } });
    rerender(tree(['compiler']));
    await waitFor(() => expect(modesTrigger()).toHaveAccessibleName('Current runs: Single Paper Writer'));
    expect(translationAPI.getSettings).toHaveBeenCalledTimes(2);
    rerender(tree(['compiler']));
    await new Promise((resolve) => { setTimeout(resolve, 0); });
    expect(translationAPI.getSettings).toHaveBeenCalledTimes(2);
  });
});

describe('saving settings', () => {
  test('language changes save the saved roles, never unsaved model drafts', async () => {
    const { user } = renderModal();
    await screen.findByRole('list', { name: 'Most spoken' });

    await user.click(screen.getByRole('tab', { name: 'Translator models' }));
    const validator = card('Translator Validator');
    await user.clear(fieldIn(validator, 'Max Output Tokens'));
    await user.type(fieldIn(validator, 'Max Output Tokens'), '8000');
    await user.click(screen.getByRole('tab', { name: 'Languages' }));
    await user.click(screen.getByRole('checkbox', { name: /^French/ }));

    await waitFor(() => expect(translationAPI.saveSettings).toHaveBeenCalledTimes(1));
    expect(translationAPI.saveSettings).toHaveBeenCalledWith({
      enabled: true,
      paused: false,
      batch_translation: false,
      languages: [ENGLISH, FRENCH],
      translator: ROLE('openrouter/alpha'),
      validator: ROLE('openrouter/beta'),
      modes: STANDARD_MODES,
    });

    // The model draft survives the tab switch and saves with the current languages.
    await user.click(screen.getByRole('tab', { name: 'Translator models' }));
    expect(fieldIn(card('Translator Validator'), 'Max Output Tokens')).toHaveValue(8000);
    await user.click(screen.getByRole('button', { name: 'Save translator models' }));
    await waitFor(() => expect(translationAPI.saveSettings).toHaveBeenCalledTimes(2));
    const payload = translationAPI.saveSettings.mock.calls[1][0];
    expect(Object.keys(payload).sort()).toEqual(
      ['batch_translation', 'enabled', 'languages', 'modes', 'paused', 'translator', 'validator'],
    );
    expect(payload.languages).toEqual([ENGLISH, FRENCH]);
    expect(payload.translator).toEqual(ROLE('openrouter/alpha'));
    expect(payload.validator).toEqual(ROLE('openrouter/beta', { max_output_tokens: 8000 }));
    expect(await screen.findByText('Translator models saved.')).toBeInTheDocument();
  });

  test('a saved Batch translation setting travels with language and model saves', async () => {
    translationAPI.getSettings.mockResolvedValue({ success: true, settings: { ...SAVED_SETTINGS, batch_translation: true } });
    const { user } = renderModal();
    await screen.findByRole('list', { name: 'Most spoken' });
    await user.click(screen.getByRole('checkbox', { name: /^French/ }));
    await waitFor(() => expect(translationAPI.saveSettings).toHaveBeenCalledTimes(1));
    expect(translationAPI.saveSettings.mock.calls[0][0].batch_translation).toBe(true);
  });

  test('a model save waits for an in-flight language save and keeps its languages', async () => {
    let finishLanguageSave;
    translationAPI.saveSettings.mockImplementationOnce((payload) => new Promise((resolve) => {
      finishLanguageSave = () => resolve({ success: true, settings: payload, status: status() });
    }));
    const { user } = renderModal();
    await screen.findByRole('list', { name: 'Most spoken' });
    await user.click(screen.getByRole('checkbox', { name: /^French/ }));
    await user.click(screen.getByRole('tab', { name: 'Translator models' }));
    const translator = card('Translator Submitter');
    await user.clear(fieldIn(translator, 'Max Output Tokens'));
    await user.type(fieldIn(translator, 'Max Output Tokens'), '16000');
    await user.click(screen.getByRole('button', { name: 'Save translator models' }));
    expect(translationAPI.saveSettings).toHaveBeenCalledTimes(1);

    finishLanguageSave();
    await waitFor(() => expect(translationAPI.saveSettings).toHaveBeenCalledTimes(2));
    const modelPayload = translationAPI.saveSettings.mock.calls[1][0];
    expect(modelPayload.languages).toEqual([ENGLISH, FRENCH]);
    expect(modelPayload.translator.max_output_tokens).toBe(16000);
    expect(await screen.findByText('Translator models saved.')).toBeInTheDocument();
  });

  test('enabled and paused in a settings save follow the latest status', async () => {
    const { user } = renderModal({
      context: contextValue({ status: status({ enabled: false, paused: true, state: 'disabled' }) }),
    });
    await screen.findByRole('list', { name: 'Most spoken' });
    await user.click(screen.getByRole('checkbox', { name: /^French/ }));
    await waitFor(() => expect(translationAPI.saveSettings).toHaveBeenCalledTimes(1));
    expect(translationAPI.saveSettings.mock.calls[0][0]).toMatchObject({ enabled: false, paused: true });
  });

  test('backend validation errors from a model save are displayed', async () => {
    translationAPI.saveSettings.mockRejectedValueOnce(Object.assign(
      new Error('HTTP 422 Unprocessable Entity: Hosted mode supports OpenRouter translator models only.'),
      { status: 422 },
    ));
    const { user } = renderModal();
    await user.click(screen.getByRole('tab', { name: 'Translator models' }));
    const validator = await screen.findByRole('heading', { name: 'Translator Validator' });
    const maxOutput = fieldIn(validator.closest('.submitter-config-section'), 'Max Output Tokens');
    await user.clear(maxOutput);
    await user.type(maxOutput, '9000');
    await user.click(screen.getByRole('button', { name: 'Save translator models' }));
    expect(await screen.findByRole('alert')).toHaveTextContent('Hosted mode supports OpenRouter translator models only.');
  });

  test('hosted mode shows OpenRouter-only translator roles', async () => {
    const { user } = renderModal({ capabilities: { lmStudioEnabled: false, genericMode: true } });
    await user.click(screen.getByRole('tab', { name: 'Translator models' }));
    expect(await screen.findAllByText('OpenRouter is required in this deployment.')).toHaveLength(2);
    expect(screen.queryByRole('button', { name: 'LM Studio' })).not.toBeInTheDocument();
    expect(screen.queryByText('Supercharge')).not.toBeInTheDocument();
    expect(api.getModels).not.toHaveBeenCalled();
  });
});

describe('progress tab', () => {
  test('shows per-language progress, the queue with positions and the running part, activity, and usage', async () => {
    const { user } = renderModal({ context: contextValue({ recentActivity: ACTIVITY }) });
    await user.click(screen.getByRole('tab', { name: 'Progress' }));
    const panel = screen.getByRole('tabpanel', { name: 'Progress' });

    const french = within(panel).getByRole('progressbar', { name: 'French translation progress' });
    expect(french).toHaveAttribute('aria-valuenow', '60');
    expect(french).toHaveAttribute('aria-valuemin', '0');
    expect(french).toHaveAttribute('aria-valuemax', '100');
    expect(french).toHaveAttribute('aria-valuetext', '60% translated');
    expect(within(panel).getByText('60% translated')).toBeInTheDocument();
    expect(within(panel).getByRole('progressbar', { name: 'Spanish translation progress' })).toHaveAttribute('aria-valuenow', '0');
    expect(within(panel).getByText('2 items pending')).toBeInTheDocument();
    expect(within(panel).getByText('1 item pending')).toBeInTheDocument();

    const queue = within(panel).getByRole('list', { name: 'Translation queue' });
    const [first, second, third] = within(queue).getAllByRole('listitem');
    expect(within(first).getByText('Translating now')).toBeInTheDocument();
    expect(within(first).getByText('French')).toBeInTheDocument();
    expect(within(first).getByText('Proof')).toBeInTheDocument();
    expect(within(first).getByText('Main Theorem')).toBeInTheDocument();
    expect(within(first).getByText('Viewing')).toBeInTheDocument();
    expect(within(first).getByText('40% done')).toBeInTheDocument();
    expect(within(first).getByText('Part 40–80% · attempt 2 · 1/3 accepted · keeps in a row 1')).toBeInTheDocument();
    expect(within(second).getByText('#1')).toBeInTheDocument();
    expect(within(second).getByText('Spanish')).toBeInTheDocument();
    expect(within(second).getByText('0% done')).toBeInTheDocument();
    expect(within(second).queryByText(/attempt/)).not.toBeInTheDocument();
    expect(within(third).getByText('#2')).toBeInTheDocument();
    expect(within(third).getByText('Final answer')).toBeInTheDocument();
    expect(within(third).getByText('11% done')).toBeInTheDocument();
    expect(within(third).queryByText('Viewing')).not.toBeInTheDocument();

    const activity = within(panel).getByRole('list', { name: 'Translation activity' });
    const accepted = within(activity)
      .getByText('French: attempt 2 for “Main Theorem” was accepted (1/3 accepted)')
      .closest('li');
    expect(accepted).toHaveClass('auto-translate-activity__item--accepted');
    const rejected = within(activity).getByText('French: attempt 1 for “Main Theorem” was rejected').closest('li');
    expect(rejected).toHaveClass('auto-translate-activity__item--rejected');
    expect(within(rejected).getByText('Keep the theorem name untranslated.')).toBeInTheDocument();
    expect(within(activity).getByText(
      'French: attempt 1 for “Paper A” was judged again and accepted (1/3 accepted)',
    )).toBeInTheDocument();
    const kept = within(activity)
      .getByText('French: the translator kept attempt 2 for “Main Theorem” (keeps in a row: 1)')
      .closest('li');
    expect(kept).toHaveClass('auto-translate-activity__item--kept');
    expect(within(activity).getByText(
      'French: the translator asked the validator to judge attempt 1 for “Paper A” again',
    )).toBeInTheDocument();
    const finalized = within(activity)
      .getByText('French: a part of “Main Theorem” is final with attempt 2 (kept twice)')
      .closest('li');
    expect(finalized).toHaveClass('auto-translate-activity__item--final');
    expect(within(activity).getByText(
      'French: a part of “Paper A” is final with attempt 3 (three accepted attempts)',
    )).toBeInTheDocument();
    const checkFailed = within(activity)
      .getByText('Spanish: attempt 4 for “Paper A” failed the automatic checks')
      .closest('li');
    expect(checkFailed).toHaveClass('auto-translate-activity__item--check');
    expect(within(checkFailed).getByText('Placeholder [[M1]] is missing from unit U5.')).toBeInTheDocument();
    const contractRetry = within(activity)
      .getByText('French: the validator reply for “Paper A” was not in the required format; checking again')
      .closest('li');
    expect(contractRetry).toHaveClass('auto-translate-activity__item--check');
    expect(within(contractRetry).getByText('decisions must list every attempt')).toBeInTheDocument();
    // The route leads the repair line once instead of being listed as a detail.
    const repair = within(activity)
      .getByText('Foreign Language Translation w/ Top-p · Translator Submitter · openrouter/alpha Repair required: Raise Max Output '
        + 'Tokens for the Translator Submitter.')
      .closest('li');
    expect(repair).toHaveClass('auto-translate-activity__item--repair');
    expect(within(repair).queryByRole('list')).not.toBeInTheDocument();
    const providerRetry = within(activity).getByText('Foreign Language Translation w/ Top-p · Translator Validator · openrouter/beta: retrying in 30s').closest('li');
    expect(providerRetry).toHaveClass('auto-translate-activity__item--provider');
    expect(within(activity).getByText('OpenRouter is busy; retrying soon.')).toBeInTheDocument();

    expect(within(panel).getByText('42 model calls')).toBeInTheDocument();
    expect(within(panel).getByText('900 prompt tokens')).toBeInTheDocument();
    expect(within(panel).getByText('300 completion tokens')).toBeInTheDocument();
    expect(within(panel).getByText('7 accepted attempts')).toBeInTheDocument();
    expect(within(panel).getByText('5 rejected attempts')).toBeInTheDocument();
    expect(within(panel).getByText('3 finished parts')).toBeInTheDocument();
    expect(within(panel).getByText('2 automatic check failures')).toBeInTheDocument();
    expect(within(panel).getByText(
      'New papers, proofs, final answers, completed brainstorms, and results from the current run of each mode you '
        + 'checked are translated automatically. Anything else translates while you have it open in a selected '
        + 'language, and Translate all older items translates the older runs of the checked modes in the '
        + 'background.',
    )).toBeInTheDocument();
  });

  test('queue positions update when a new status arrives', async () => {
    const context = contextValue();
    const { user, rerender } = renderModal({ context });
    await user.click(screen.getByRole('tab', { name: 'Progress' }));
    expect(within(screen.getByRole('list', { name: 'Translation queue' })).getByText('#2')).toBeInTheDocument();

    const next = status({
      generation: 4,
      queue: [{ ...status().queue[1], position: 1 }],
      active: [],
    });
    rerender(
      <AutoTranslateContext.Provider value={contextValue({ status: next })}>
        <AutoTranslateModal isOpen capabilities={{ lmStudioEnabled: true }} onClose={vi.fn()} />
      </AutoTranslateContext.Provider>,
    );
    const queue = screen.getByRole('list', { name: 'Translation queue' });
    expect(within(queue).getAllByRole('listitem')).toHaveLength(1);
    expect(within(queue).getByText('#1')).toBeInTheDocument();
    expect(within(queue).queryByText('Translating now')).not.toBeInTheDocument();
  });

  test('tags Translate all queue items and counts the older items waiting in it', async () => {
    const background = {
      doc_hash: 'h3',
      doc_key: 'paper:old-session:paper_002',
      kind: 'paper',
      title: 'Older Paper',
      language_key: 'fr',
      position: 3,
      viewing: false,
      background: true,
      translated_units: 0,
      total_units: 12,
    };
    const viewedBackground = { ...background, doc_hash: 'h4', title: 'Opened Older Paper', position: 4, viewing: true };
    const { user } = renderModal({
      context: contextValue({
        status: status({
          languages: [
            { ...status().languages[0], background_pending_units: 1200, background_pending_documents: 3 },
            { ...status().languages[1], background_pending_units: 1, background_pending_documents: 1 },
          ],
          queue: [...status().queue, background, viewedBackground],
        }),
      }),
    });
    await user.click(screen.getByRole('tab', { name: 'Progress' }));
    const panel = screen.getByRole('tabpanel', { name: 'Progress' });
    expect(within(panel).getByText('+ 3 older items waiting in Translate all')).toBeInTheDocument();
    expect(within(panel).getByText('+ 1 older item waiting in Translate all')).toBeInTheDocument();

    const rows = within(within(panel).getByRole('list', { name: 'Translation queue' })).getAllByRole('listitem');
    const older = rows.find((row) => within(row).queryByText('Older Paper'));
    expect(within(older).getByText('Background')).toHaveClass('auto-translate-background-tag');
    const opened = rows.find((row) => within(row).queryByText('Opened Older Paper'));
    expect(within(opened).getByText('Viewing')).toBeInTheDocument();
    expect(within(opened).queryByText('Background')).not.toBeInTheDocument();
    // Current-run rows carry neither the tag nor (without the new fields) a background count.
    expect(within(rows[0]).queryByText('Background')).not.toBeInTheDocument();
    expect(within(panel).getAllByText(/waiting in Translate all/)).toHaveLength(2);
  });

  test('an older status without background counts shows no Translate all line', async () => {
    const { user } = renderModal();
    await user.click(screen.getByRole('tab', { name: 'Progress' }));
    expect(screen.queryByText(/waiting in Translate all/)).not.toBeInTheDocument();
    expect(screen.queryByText('Background')).not.toBeInTheDocument();
  });

  test('an empty queue says nothing is waiting', async () => {
    const { user } = renderModal({ context: contextValue({ status: status({ queue: [], active: [] }) }) });
    await user.click(screen.getByRole('tab', { name: 'Progress' }));
    expect(screen.getByText('Nothing is waiting to be translated.')).toBeInTheDocument();
    expect(screen.queryByRole('list', { name: 'Translation queue' })).not.toBeInTheDocument();
  });

  test('renders translation retry waits and never prints raw, v1, or unknown event names', async () => {
    const recentActivity = [
      { event_type: 'translation_progress', language_key: 'fr', translated_units: 3, total_units: 4, timestamp: '2026-11-01T12:10:00Z' },
      { event_type: 'translation_queue_changed', queue_version: 7, timestamp: '2026-11-01T12:09:30Z' },
      { event_type: 'translation_status_changed', state: 'translating', message: 'Translating 3 unit(s).', timestamp: '2026-11-01T12:09:00Z' },
      { event_type: 'translation_segment_accepted', language_name: 'French', doc_title: 'Old', round: 1, timestamp: '2026-11-01T12:08:00Z' },
      { event_type: 'translation_future_event', message: 'Something new.', timestamp: '2026-11-01T12:07:00Z' },
      {
        activity_id: 'r1',
        event_type: 'translation_provider_retry',
        role_id: 'translation_validator',
        reason: 'translation_validator_contract',
        retry_after_seconds: 5,
        timestamp: '2026-11-01T12:06:00Z',
      },
      {
        activity_id: 'r2',
        event_type: 'translation_provider_retry',
        role_id: 'translation_submitter',
        reason: 'rate_limited',
        timestamp: '2026-11-01T12:05:00Z',
      },
      {
        activity_id: 'r3',
        event_type: 'translation_provider_retry',
        role_id: 'translation_submitter',
        message: 'Translator Submitter: an unexpected error interrupted the attempt; retrying in 20s.',
        timestamp: '2026-11-01T12:04:00Z',
      },
      { activity_id: 'w1', event_type: 'translation_waiting_for_credits', timestamp: '2026-11-01T12:03:00Z' },
    ];
    const { user } = renderModal({ context: contextValue({ recentActivity }) });
    await user.click(screen.getByRole('tab', { name: 'Progress' }));
    const activity = screen.getByRole('list', { name: 'Translation activity' });

    expect(within(activity).getAllByRole('listitem')).toHaveLength(4);
    const validatorRetry = within(activity)
      .getByText('Translator Validator: retrying in 5s (the validator reply was not in the required format)')
      .closest('li');
    expect(validatorRetry).toHaveClass('auto-translate-activity__item--provider');
    expect(within(activity).getByText('Translator Submitter: retrying (rate limited)')).toBeInTheDocument();
    expect(within(activity).getByText(
      'Translator Submitter: an unexpected error interrupted the attempt; retrying in 20s.',
    )).toBeInTheDocument();
    expect(within(activity).getByText('Waiting for provider credits to be reset.')).toBeInTheDocument();
    expect(activity).not.toHaveTextContent(/translation_[a-z_]+/);
    expect(activity).not.toHaveTextContent('Translating 3 unit(s).');
    expect(activity).not.toHaveTextContent('Something new.');
    expect(activity).not.toHaveTextContent('Old');
  });

  test('shows Translate all starting, cancelled, and finishing, also from status recent_activity alone', async () => {
    // Status copies keep the event type, time, activity id and message, state, and message.
    const recorded = (activityId, state, message, timestamp) => ({
      activity_id: activityId,
      event_type: 'translation_translate_all',
      timestamp,
      activity_message: message,
      state,
      message,
    });
    const recentActivity = [
      recorded('t3', 'finished', 'Translate all finished: 12 items in 2 languages.', '2026-11-01T12:03:00Z'),
      recorded('t2', 'cancelled', 'Translate all was turned off because stored translations were cleared.', '2026-11-01T12:02:00Z'),
      recorded('t1', 'started', 'Translate all started.', '2026-11-01T12:01:00Z'),
    ];
    const { user } = renderModal({ context: contextValue({ recentActivity }) });
    await user.click(screen.getByRole('tab', { name: 'Progress' }));
    const activity = screen.getByRole('list', { name: 'Translation activity' });
    expect(within(activity).getAllByRole('listitem')).toHaveLength(3);
    expect(within(activity).getByText('Translate all finished: 12 items in 2 languages.').closest('li'))
      .toHaveClass('auto-translate-activity__item--final');
    expect(within(activity).getByText('Translate all was turned off because stored translations were cleared.').closest('li'))
      .toHaveClass('auto-translate-activity__item--kept');
    expect(within(activity).getByText('Translate all started.').closest('li'))
      .toHaveClass('auto-translate-activity__item--accepted');
  });

  test('Translate all activity has a tone per state and falls back to its own text', () => {
    const item = (state, message) => ({ event_type: 'translation_translate_all', state, message });
    expect(describeTranslationActivity(item('started', 'Translate all started.')))
      .toEqual({ tone: 'accepted', text: 'Translate all started.', details: [] });
    expect(describeTranslationActivity(item('cancelled', 'Translate all cancelled.')))
      .toEqual({ tone: 'kept', text: 'Translate all cancelled.', details: [] });
    expect(describeTranslationActivity(item('finished', 'Nothing else to translate for the checked modes.')))
      .toEqual({ tone: 'final', text: 'Nothing else to translate for the checked modes.', details: [] });
    expect(describeTranslationActivity(item('finished')).text).toBe('Translate all finished.');
    expect(describeTranslationActivity(item('cancelled')).text).toBe('Translate all cancelled.');
    expect(describeTranslationActivity(item('started')).text).toBe('Translate all started.');
    expect(describeTranslationActivity(item('resumed', 'Translate all resumed.')))
      .toEqual({ tone: 'kept', text: 'Translate all resumed.', details: [] });
    expect(describeTranslationActivity(item('resumed'))).toBeNull();
  });

  test('describes only the activity types it knows', () => {
    expect(describeTranslationActivity({ event_type: 'translation_progress' })).toBeNull();
    expect(describeTranslationActivity({ event_type: 'translation_queue_changed', queue_version: 2 })).toBeNull();
    expect(describeTranslationActivity({ event_type: 'translation_status_changed', message: 'Idle.' })).toBeNull();
    expect(describeTranslationActivity({ event_type: 'translation_round_rejected', message: 'v1' })).toBeNull();
    expect(describeTranslationActivity({ event_type: 'unknown_event', message: 'Hi' })).toBeNull();
    expect(describeTranslationActivity(null)).toBeNull();
    expect(describeTranslationActivity({
      event_type: 'translation_provider_retry',
      role_id: 'translation_language_probe_1',
      reason: 'free_models_exhausted',
      retry_after_seconds: 120,
      route_identity: 'Foreign Language Translation w/ Top-p · Language check · openrouter/free',
    })).toEqual({
      tone: 'provider',
      text: 'Foreign Language Translation w/ Top-p · Language check · openrouter/free: retrying in 120s (every free model is rate-limited)',
      details: [],
    });
  });

  test('a recorded retry wait names its route once, so usage-limit and backoff waits show model and provider', () => {
    const route = '[Foreign Language Translation w/ Top-p · Translator Submitter · kimi-for-coding via Kimi Code]';
    const message = 'Usage limit reached; retrying in 3600s.';
    const recorded = {
      event_type: 'translation_provider_retry',
      role_id: 'translation_submitter',
      reason: 'usage_limit_reached',
      retry_after_seconds: 3600,
      route_identity: route,
    };
    // The service stores the line already prefixed with its route; the panel shows the route once.
    expect(describeTranslationActivity({ ...recorded, message: `${route} ${message}` })).toEqual({
      tone: 'provider',
      text: `${route} ${message}`,
      details: [],
    });
    // A line stored without its prefix still gets the route once.
    expect(describeTranslationActivity({ ...recorded, message }).text).toBe(`${route} ${message}`);
    const legacy = 'Translator Submitter: usage limit reached; retrying in 3600s.';
    expect(describeTranslationActivity({ ...recorded, route_identity: undefined, message: legacy }).text).toBe(legacy);
  });

  test('recorded backoff waits show the provider error, and repair and credit lines lead with their route', () => {
    const route = '[Foreign Language Translation w/ Top-p · Translator Submitter · alpha via OpenRouter]';
    expect(describeTranslationActivity({
      event_type: 'translation_provider_retry',
      role_id: 'translation_submitter',
      reason: 'transient',
      retry_after_seconds: 60,
      route_identity: route,
      message: `${route} Transient; retrying in 60s (retry 1).`,
      error_summary: 'upstream timeout',
    })).toEqual({ tone: 'provider', text: `${route} Transient; retrying in 60s (retry 1).`, details: ['upstream timeout'] });
    expect(describeTranslationActivity({
      event_type: 'translation_repair_required',
      route_identity: route,
      message: `${route} OpenRouter rejected the API key.`,
      guidance: 'Add a valid OpenRouter key.',
    })).toEqual({
      tone: 'repair',
      text: `${route} Repair required: OpenRouter rejected the API key.`,
      details: ['Add a valid OpenRouter key.'],
    });
    expect(describeTranslationActivity({
      event_type: 'translation_waiting_for_credits',
      route_identity: route,
      message: `${route} Waiting for provider credits to be reset.`,
    }).text).toBe(`${route} Waiting for provider credits to be reset.`);
  });

  test('a hung translator call names the waiting role and points to the translator models in this window', () => {
    const guidance = 'Possible hung model call (25+ min). It may still be thinking; you can keep waiting, or lower '
      + 'the reasoning effort for the translator models on the Translator models tab of this window if this repeats.';
    const route = '[Foreign Language Translation w/ Top-p · Translator Validator · beta via OpenRouter]';
    const hung = describeTranslationActivity({
      event_type: 'hung_connection_alert',
      workflow_mode: 'translation',
      role_id: 'translation_validator',
      elapsed_minutes: 25,
      route_identity: route,
      message: 'The model may still be thinking; you can keep waiting or lower reasoning effort in Settings if this repeats.',
    });
    expect(hung).toEqual({ tone: 'provider', text: `${route} ${guidance}`, details: [] });
    expect(describeTranslationActivity({
      event_type: 'hung_connection_alert',
      role_id: 'translation_submitter',
      elapsed_minutes: 25,
    }).text).toBe(`Translator Submitter: ${guidance}`);
  });

  test('live provider retry lines carry the route once; resumed events are not shown', () => {
    const route = '[Foreign Language Translation w/ Top-p · Translator Submitter · alpha via OpenRouter]';
    expect(describeTranslationActivity({
      event_type: 'provider_retry_cooldown',
      route_identity: route,
      message: 'OpenRouter server-side error: upstream timed out.',
    }).text).toBe(`${route} OpenRouter server-side error: upstream timed out.`);
    expect(describeTranslationActivity({
      event_type: 'provider_retry_cooldown',
      route_identity: route,
      message: `${route} OpenRouter is busy.`,
    }).text).toBe(`${route} OpenRouter is busy.`);
    expect(describeTranslationActivity({
      event_type: 'provider_retry_resumed',
      route_identity: route,
      message: 'Resumed.',
    })).toBeNull();
  });

  test('attempt numbers come from the live payload, the status copy, or the round', () => {
    const base = { event_type: 'translation_attempt_rejected', language_name: 'French', doc_title: 'D', part_start: 2, part_end: 2 };
    expect(describeTranslationActivity({ ...base, attempt_number: 4, round_index: 6, rejudged: true }).text)
      .toBe('French: attempt 4 for “D” was judged again and rejected');
    expect(describeTranslationActivity({ ...base, candidate_id: 'h:fr:2:attempt3:rejudge6', round_index: 6 }).text)
      .toBe('French: attempt 3 for “D” was judged again and rejected');
    expect(describeTranslationActivity({ ...base, round_index: 5 }).text)
      .toBe('French: attempt 5 for “D” was rejected');
    expect(describeTranslationActivity({
      event_type: 'translation_attempt_kept',
      language_key: 'es',
      doc_title: 'D',
      part_start: 1,
      part_end: 3,
      keep_attempts: [1, 2],
      keep_streak: 2,
    }, new Map([['es', 'Spanish']])).text).toBe('Spanish: the translator kept attempts 1, 2 for “D” (keeps in a row: 2)');
  });

  test('progress percentages, running parts, and queue row order', () => {
    expect(translationPercent(0, 4851)).toBe(0);
    expect(translationPercent(1815, 4851)).toBe(37);
    expect(translationPercent(4850, 4851)).toBe(99);
    expect(translationPercent(4851, 4851)).toBe(100);
    expect(formatPartRange(63, 79, 4851)).toBe('1–2%');
    expect(formatPartRange(5, 5, 1000)).toBe('0–1%');
    expect(formatPartRange(3, 3, 4)).toBe('50–75%');
    expect(formatPartRange(5, 5, 0)).toBeNull();
    expect(describeActivePart({ part_start: 5, part_end: 5, attempt: 1, accepted_count: 0, keep_streak: 0 }, 10))
      .toBe('Part 40–50% · attempt 1 · 0/3 accepted · keeps in a row 0');
    expect(describeActivePart({ part_start: 5, part_end: 5, attempt: 1, accepted_count: 0, keep_streak: 0 }))
      .toBe('Current part · attempt 1 · 0/3 accepted · keeps in a row 0');

    const running = { doc_hash: 'h9', doc_title: 'Not listed', language_key: 'fr', part_start: 1, part_end: 2, attempt: 3 };
    const rows = buildQueueRows(
      [
        { doc_hash: 'h2', language_key: 'fr', position: 1, title: 'Second' },
        { doc_hash: 'h1', language_key: 'fr', position: 0, title: 'First' },
      ],
      [{ doc_hash: 'h1', language_key: 'fr', part_start: 2, part_end: 2, attempt: 1 }, running],
    );
    expect(rows.map((row) => [row.item.title, row.position, Boolean(row.running)])).toEqual([
      ['Not listed', 0, true],
      ['First', 0, true],
      ['Second', 1, false],
    ]);
    expect(buildQueueRows(null, null)).toEqual([]);
  });

  test('shows at most 50 activity items', async () => {
    const many = Array.from({ length: 70 }, (_, index) => ({
      activity_id: `item-${index}`,
      event_type: 'translation_attempt_accepted',
      language_name: 'French',
      doc_title: `Doc ${index}`,
      part_start: 1,
      part_end: 1,
      round_index: 1,
      accepted_count: 1,
    }));
    const { user } = renderModal({ context: contextValue({ recentActivity: many }) });
    await user.click(screen.getByRole('tab', { name: 'Progress' }));
    expect(within(screen.getByRole('list', { name: 'Translation activity' })).getAllByRole('listitem')).toHaveLength(50);
  });

  test('clearing stored translations needs confirmation', async () => {
    const { user, onStatusChanged } = renderModal();
    await user.click(screen.getByRole('tab', { name: 'Progress' }));
    await user.click(screen.getByRole('button', { name: 'Clear stored translations' }));
    const confirm = screen.getByRole('group', { name: 'Confirm clearing stored translations' });
    expect(within(confirm).getByRole('button', { name: 'Cancel' })).toHaveFocus();
    await user.click(within(confirm).getByRole('button', { name: 'Cancel' }));
    expect(translationAPI.clearCache).not.toHaveBeenCalled();
    expect(screen.getByRole('button', { name: 'Clear stored translations' })).toHaveFocus();

    await user.click(screen.getByRole('button', { name: 'Clear stored translations' }));
    await user.click(screen.getByRole('button', { name: 'Clear translations' }));
    expect(translationAPI.clearCache).toHaveBeenCalledTimes(1);
    expect(await screen.findByText('Stored translations cleared.')).toBeInTheDocument();
    expect(onStatusChanged).toHaveBeenCalled();
  });
});

describe('accessibility (axe)', () => {
  test('the open modal on the Languages tab', async () => {
    const { container } = renderModal();
    await screen.findByRole('list', { name: 'Most spoken' });
    await expectNoAxeViolations(container);
  });

  test('the control row with Batch translation on and its help open', async () => {
    translationAPI.getSettings.mockResolvedValue({ success: true, settings: { ...SAVED_SETTINGS, batch_translation: true } });
    const { user, container } = renderModal({
      context: contextValue({ status: status({ batch_translation: true, max_parallel_jobs: 2 }) }),
    });
    await screen.findByRole('list', { name: 'Most spoken' });
    expect(screen.getByText('Up to 2 documents translate at once.')).toBeInTheDocument();
    await user.hover(screen.getByRole('button', { name: 'Learn about Batch translation' }));
    await expectNoAxeViolations(container);
  });

  test('the open current-runs list in developer mode with its help open', async () => {
    const { user, container } = renderModal({ developerModeEnabled: true });
    await screen.findByRole('list', { name: 'Most spoken' });
    await user.click(screen.getByRole('button', { name: /^Current runs:/ }));
    await user.hover(screen.getByRole('button', { name: 'Learn about which modes are translated' }));
    await expectNoAxeViolations(container);
  });

  test('the Translate all confirm, its help, and the disabled reason', async () => {
    const { user, container } = renderModal();
    await screen.findByRole('list', { name: 'Most spoken' });
    await user.click(screen.getByRole('button', { name: 'Translate all older items' }));
    await user.hover(screen.getByRole('button', { name: 'Learn about Translate all' }));
    await expectNoAxeViolations(container);
  });

  test('Translate all while it runs, and while translation is off', async () => {
    const running = renderModal({
      context: contextValue({
        status: status({ translate_all: { ...TRANSLATE_ALL_OFF, requested: true, state: 'translating', documents: 4 } }),
      }),
    });
    await screen.findByRole('list', { name: 'Most spoken' });
    await expectNoAxeViolations(running.container);
    running.unmount();

    const off = renderModal({ context: contextValue({ status: status({ enabled: false, state: 'disabled' }) }) });
    await screen.findByRole('list', { name: 'Most spoken' });
    await expectNoAxeViolations(off.container);
  });

  test('the Translator models tab including the shared role cards', async () => {
    const { user, container } = renderModal();
    await screen.findByRole('list', { name: 'Most spoken' });
    await user.click(screen.getByRole('tab', { name: 'Translator models' }));
    await waitFor(() => expect(openRouterAPI.getModels).toHaveBeenCalled());
    await expectNoAxeViolations(container);
  });

  test('the Progress tab with the queue, activity, and a repair banner', async () => {
    const { user, container } = renderModal({
      context: contextValue({
        recentActivity: ACTIVITY,
        status: status({
          state: 'repair_required',
          repair: { message: 'Repair needed.', guidance: 'Raise the limit.', route_identity: 'route' },
        }),
      }),
    });
    await screen.findByRole('list', { name: 'Most spoken' });
    await user.click(screen.getByRole('tab', { name: 'Progress' }));
    await expectNoAxeViolations(container);
  });
});
