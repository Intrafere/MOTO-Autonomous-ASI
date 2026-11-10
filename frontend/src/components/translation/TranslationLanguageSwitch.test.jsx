import React from 'react';
import { act, cleanup, fireEvent, render, screen, within } from '@testing-library/react';
import { afterEach, beforeEach, describe, expect, test, vi } from 'vitest';
import TranslationLanguageSwitch from './TranslationLanguageSwitch';
import TranslationPendingNotice from './TranslationPendingNotice';
import AutoTranslateProvider from './AutoTranslateProvider';
import { AutoTranslateContext, DISPLAY_LANGUAGE_STORAGE_KEY } from '../../hooks/useAutoTranslate';
import { translationAPI } from '../../services/api';
import useTranslatedDocuments from '../../hooks/useTranslatedDocuments';

vi.mock('../../services/websocket', () => ({
  websocket: { on: vi.fn(() => () => {}), off: vi.fn() },
}));

vi.mock('../../services/api', () => ({
  translationAPI: {
    getStatus: vi.fn(),
    registerDocuments: vi.fn(async () => ({ success: true, documents: [] })),
    lookupDocuments: vi.fn(async () => ({ success: true, documents: [] })),
    documentProgress: vi.fn(async () => ({ success: true, documents: [] })),
  },
}));

const LANGUAGES = [
  { key: 'es', name: 'Spanish', custom: false },
  { key: 'fr', name: 'French', custom: false },
];

function contextValue(overrides = {}) {
  return {
    available: true,
    status: { enabled: true, state: 'translating' },
    enabled: true,
    languages: LANGUAGES,
    displayLanguage: 'es',
    setDisplayLanguage: vi.fn(),
    refreshStatus: vi.fn(),
    subscribeProgress: () => () => {},
    recentActivity: [],
    ...overrides,
  };
}

function translation(status, overrides = {}, queuePosition = null) {
  const finished = status === 'complete';
  return {
    available: true,
    active: true,
    language: 'es',
    languageName: 'Spanish',
    selectedLanguage: 'es',
    languageOptions: [
      { key: 'es', name: 'Spanish', ready: finished, status: finished ? 'complete' : 'pending', translated: 2, total: 5 },
      { key: 'fr', name: 'French', ready: false, status: 'unknown', translated: 0, total: 0 },
    ],
    setLanguage: vi.fn(),
    byKey: {},
    overall: { status, translated: 2, total: 5, queuePosition },
    ...overrides,
  };
}

function offOptions() {
  return LANGUAGES.map((language) => ({
    key: language.key, name: language.name, ready: false, status: 'disabled', translated: 0, total: 0,
  }));
}

function optionLabels() {
  return [...screen.getByLabelText('Language').options].map((option) => option.textContent);
}

const TWO_ITEMS = [
  { docKey: 'proof:a', kind: 'proof', title: 'A', profile: 'fields', parts: { theorem_statement: 'Statement A' } },
  { docKey: 'proof:b', kind: 'proof', title: 'B', profile: 'fields', parts: { theorem_statement: 'Statement B' } },
];

// Two proof cards sharing one hook, as in the proof tabs.
function TwoItems() {
  const view = useTranslatedDocuments(TWO_ITEMS);
  return (
    <>
      <div data-testid="item-a"><TranslationLanguageSwitch translation={view.forKey('proof:a')} /></div>
      <div data-testid="item-b"><TranslationLanguageSwitch translation={view.forKey('proof:b')} /></div>
    </>
  );
}

function renderWith(ui, value = contextValue()) {
  return render(<AutoTranslateContext.Provider value={value}>{ui}</AutoTranslateContext.Provider>);
}

beforeEach(() => {
  localStorage.clear();
  translationAPI.getStatus.mockReset();
});

afterEach(() => {
  cleanup();
  vi.restoreAllMocks();
});

describe('TranslationLanguageSwitch', () => {
  test('is always shown inside the app, even with translation off or no languages; not without a provider', () => {
    const off = renderWith(
      <TranslationLanguageSwitch translation={translation('complete', { languageOptions: offOptions() })} />,
      contextValue({ enabled: false }),
    );
    expect(optionLabels()).toEqual(['Original', 'Spanish (translation off)', 'French (translation off)']);
    expect(screen.getByLabelText('Language')).toHaveAttribute('title', expect.stringContaining('Translation is off'));
    off.unmount();

    const noLanguages = renderWith(
      <TranslationLanguageSwitch translation={translation('complete', { languageOptions: [] })} />,
      contextValue({ languages: [] }),
    );
    expect(optionLabels()).toEqual(['Original']);
    expect(screen.getByLabelText('Language')).toHaveAttribute('title', expect.stringContaining('Choose languages'));
    noLanguages.unmount();

    const inert = render(<TranslationLanguageSwitch translation={translation('complete')} />);
    expect(inert.container).toBeEmptyDOMElement();
  });

  test('greys out languages not finished for this item and changes only this item', () => {
    const value = contextValue();
    const view = translation('pending');
    renderWith(<TranslationLanguageSwitch translation={view} />, value);

    const select = screen.getByLabelText('Language');
    expect(select).toHaveValue('es');
    expect(optionLabels()).toEqual(['Original', 'Spanish (40%)', 'French (not translated yet)']);
    const [, spanish, french] = select.options;
    expect(spanish).toHaveClass('translation-language-switch__option--unready');
    expect(french).toHaveAttribute('data-ready', 'false');
    // Unfinished languages stay selectable: choosing one translates this item first.
    expect(french).not.toBeDisabled();

    fireEvent.change(select, { target: { value: 'original' } });
    expect(view.setLanguage).toHaveBeenCalledWith('original');
    fireEvent.change(select, { target: { value: 'fr' } });
    expect(view.setLanguage).toHaveBeenCalledWith('fr');
    expect(value.setDisplayLanguage).not.toHaveBeenCalled();
  });

  test('a finished language is shown normally', () => {
    renderWith(<TranslationLanguageSwitch translation={translation('complete')} />);
    const [, spanish] = screen.getByLabelText('Language').options;
    expect(spanish).toHaveTextContent(/^Spanish$/);
    expect(spanish).toHaveAttribute('data-ready', 'true');
    expect(spanish).not.toHaveClass('translation-language-switch__option--unready');
  });

  test('using the switch never reaches a clickable ancestor, but Escape and Tab still bubble', () => {
    const value = contextValue();
    const view = translation('pending');
    const onAncestorClick = vi.fn();
    const onAncestorPointer = vi.fn();
    const onAncestorKey = vi.fn();
    renderWith(
      // Mirrors an expanded paper card that collapses on any click.
      <div
        data-testid="card"
        onClick={onAncestorClick}
        onMouseDown={onAncestorPointer}
        onMouseUp={onAncestorPointer}
        onPointerDown={onAncestorPointer}
        onPointerUp={onAncestorPointer}
        onKeyDown={(event) => onAncestorKey(event.key)}
      >
        <TranslationLanguageSwitch translation={view} />
      </div>,
      value,
    );
    const select = screen.getByLabelText('Language');

    fireEvent.pointerDown(select);
    fireEvent.mouseDown(select);
    fireEvent.pointerUp(select);
    fireEvent.mouseUp(select);
    fireEvent.click(select);
    fireEvent.click(screen.getByText('Language', { selector: 'label' }));
    fireEvent.click(screen.getByText('Pending 40%'));
    fireEvent.keyDown(select, { key: 'ArrowDown' });
    fireEvent.keyDown(select, { key: 'Enter' });
    fireEvent.keyDown(select, { key: ' ' });
    fireEvent.change(select, { target: { value: 'fr' } });

    expect(onAncestorClick).not.toHaveBeenCalled();
    expect(onAncestorPointer).not.toHaveBeenCalled();
    expect(onAncestorKey).not.toHaveBeenCalled();
    expect(view.setLanguage).toHaveBeenCalledWith('fr');

    fireEvent.keyDown(select, { key: 'Escape' });
    fireEvent.keyDown(select, { key: 'Tab' });
    expect(onAncestorKey.mock.calls.map(([key]) => key)).toEqual(['Escape', 'Tab']);
  });

  test('choosing a language for one item leaves other items and the default alone', async () => {
    localStorage.setItem(DISPLAY_LANGUAGE_STORAGE_KEY, 'es');
    translationAPI.getStatus.mockResolvedValue({ success: true, enabled: true, state: 'idle', languages: LANGUAGES });
    render(
      <AutoTranslateProvider>
        <TwoItems />
      </AutoTranslateProvider>,
    );
    const itemA = screen.getByTestId('item-a');
    const itemB = screen.getByTestId('item-b');
    await within(itemA).findByRole('option', { name: /French/ });
    expect(within(itemA).getByLabelText('Language')).toHaveValue('es');
    expect(within(itemB).getByLabelText('Language')).toHaveValue('es');

    act(() => {
      fireEvent.change(within(itemA).getByLabelText('Language'), { target: { value: 'fr' } });
    });
    expect(within(itemA).getByLabelText('Language')).toHaveValue('fr');
    expect(within(itemB).getByLabelText('Language')).toHaveValue('es');
    expect(localStorage.getItem(DISPLAY_LANGUAGE_STORAGE_KEY)).toBe('es');
  });

  test.each([
    ['complete', 'Translated', 'translation-status-chip--complete'],
    ['pending', 'Pending 40%', 'translation-status-chip--pending'],
    ['loading', 'Checking…', 'translation-status-chip--loading'],
    ['error', 'Translation unavailable', 'translation-status-chip--error'],
    ['unavailable', 'Not selected', 'translation-status-chip--unavailable'],
  ])('shows the %s status chip', (status, label, className) => {
    renderWith(<TranslationLanguageSwitch translation={translation(status)} />);
    expect(screen.getByText(label)).toHaveClass('translation-status-chip', className);
  });

  test.each([
    [0, 'Translating 40%'],
    [1, 'Queued #1'],
    [12, 'Queued #12'],
    [null, 'Pending 40%'],
  ])('a pending chip at queue position %s reads %s', (queuePosition, label) => {
    renderWith(<TranslationLanguageSwitch translation={translation('pending', {}, queuePosition)} />);
    expect(screen.getByRole('status')).toHaveTextContent(label);
    expect(screen.getByRole('status')).toHaveClass('translation-status-chip--pending');
  });

  test.each([
    ['not_configured', 'Needs translator models', 'translation-status-chip--waiting'],
    ['waiting_for_credits', 'Waiting for credits', 'translation-status-chip--waiting'],
    ['paused', 'Paused', 'translation-status-chip--waiting'],
    ['repair_required', 'Needs repair', 'translation-status-chip--error'],
  ])('a queued chip says what translation waits for while the service state is %s', (state, label, className) => {
    renderWith(
      <TranslationLanguageSwitch translation={translation('pending', {}, 1)} />,
      contextValue({ status: { enabled: true, state } }),
    );
    expect(screen.getByRole('status')).toHaveTextContent(label);
    expect(screen.getByRole('status')).toHaveClass(className);
    expect(screen.queryByText(/Queued|Translating/)).not.toBeInTheDocument();
  });

  test('a queue position never changes a finished chip', () => {
    renderWith(<TranslationLanguageSwitch translation={translation('complete', {}, 0)} />);
    expect(screen.getByText('Translated')).toBeInTheDocument();
    expect(screen.queryByText(/Translating|Queued/)).not.toBeInTheDocument();
  });

  test('announces pending progress and hides the chip while showing the original', () => {
    const { rerender } = renderWith(<TranslationLanguageSwitch translation={translation('pending')} />);
    expect(screen.getByRole('status')).toHaveTextContent('Pending 40%');

    rerender(
      <AutoTranslateContext.Provider value={contextValue({ displayLanguage: 'original' })}>
        <TranslationLanguageSwitch
          translation={{ ...translation('original'), active: false, selectedLanguage: 'original' }}
        />
      </AutoTranslateContext.Provider>,
    );
    expect(screen.queryByText(/Pending|Translated|Checking/)).not.toBeInTheDocument();
    expect(screen.getByLabelText('Language')).toHaveValue('original');
  });

  test('offers a translation request only while a pending document would stop with the viewer', async () => {
    const request = vi.fn(async () => {});
    const { rerender } = renderWith(
      <TranslationLanguageSwitch
        translation={translation('pending', { request, overall: { ...translation('pending').overall, requestable: true } })}
      />,
    );
    const button = screen.getByRole('button', { name: /Request translation/ });
    expect(button).toHaveAccessibleName(
      'Request translation. Translate into Spanish in the background, even after you leave this page.',
    );
    await act(async () => {
      fireEvent.click(button);
    });
    expect(request).toHaveBeenCalledTimes(1);
    expect(screen.getByText('Requested')).toBeInTheDocument();
    expect(screen.queryByRole('button', { name: /Request translation/ })).not.toBeInTheDocument();

    // A document already requested or in a current run needs no button.
    rerender(
      <AutoTranslateContext.Provider value={contextValue()}>
        <TranslationLanguageSwitch translation={translation('pending', { request })} key="other" />
      </AutoTranslateContext.Provider>,
    );
    expect(screen.queryByRole('button', { name: /Request translation/ })).not.toBeInTheDocument();
  });

  test('while the original is shown the request covers every selected language', async () => {
    const request = vi.fn(async () => {});
    renderWith(
      <TranslationLanguageSwitch
        translation={{ ...translation('original'), active: false, request, overall: { status: 'original', requestable: true } }}
      />,
      contextValue({ displayLanguage: 'original' }),
    );
    const button = screen.getByRole('button', { name: /Request translation/ });
    expect(button).toHaveAttribute('title', 'Translate into all your selected languages in the background');
    await act(async () => {
      fireEvent.click(button);
    });
    expect(request).toHaveBeenCalledTimes(1);
    expect(screen.getByText('Requested')).toBeInTheDocument();
  });

  test('a failed request says so and keeps the button for another try', async () => {
    vi.spyOn(console, 'warn').mockImplementation(() => {});
    const request = vi.fn(async () => {
      throw new Error('HTTP 500');
    });
    renderWith(
      <TranslationLanguageSwitch
        translation={translation('pending', { request, overall: { ...translation('pending').overall, requestable: true } })}
      />,
    );
    await act(async () => {
      fireEvent.click(screen.getByRole('button', { name: /Request translation/ }));
    });
    expect(screen.getByRole('alert')).toHaveTextContent('Request failed. Try again.');
    expect(screen.getByRole('button', { name: /Request translation/ })).toBeEnabled();
  });
});

describe('TranslationPendingNotice', () => {
  test('explains pending progress while the original is shown', () => {
    renderWith(<TranslationPendingNotice translation={translation('pending')} />);
    expect(screen.getByRole('status')).toHaveTextContent(
      'Spanish translation pending — 40% translated. Showing the original until it is ready.',
    );
  });

  test('shows the share translated so far while the document is translating now', () => {
    renderWith(<TranslationPendingNotice translation={translation('pending', {}, 0)} />);
    expect(screen.getByRole('status')).toHaveTextContent('Spanish: translating now — 40% done');
    expect(screen.getByRole('status')).toHaveClass('translation-pending-notice--pending');
  });

  test('shows the place in line while the document is queued, and updates when it moves', () => {
    const { rerender } = renderWith(<TranslationPendingNotice translation={translation('pending', {}, 4)} />);
    expect(screen.getByRole('status')).toHaveTextContent(
      'Spanish translation queued — #4 in line. Showing the original until it is ready.',
    );

    rerender(
      <AutoTranslateContext.Provider value={contextValue()}>
        <TranslationPendingNotice translation={translation('pending', {}, 1)} />
      </AutoTranslateContext.Provider>,
    );
    expect(screen.getByRole('status')).toHaveTextContent(
      'Spanish translation queued — #1 in line. Showing the original until it is ready.',
    );
  });

  test('mentions checking while the first status is loading', () => {
    renderWith(<TranslationPendingNotice translation={translation('loading', { overall: { status: 'loading', translated: 0, total: 0 } })} />);
    expect(screen.getByRole('status')).toHaveTextContent(
      'Checking for a Spanish translation. Showing the original until it is ready.',
    );
  });

  test('shows the repair guidance when translation needs repair', () => {
    renderWith(
      <TranslationPendingNotice translation={translation('pending', {}, 0)} />,
      contextValue({ status: { enabled: true, state: 'repair_required' } }),
    );
    expect(screen.getByRole('status')).toHaveTextContent(
      'Foreign Language Translation w/ Top-p is paused until its translator settings are repaired. Open Foreign Language Translation w/ Top-p from the Skills And Enhancements panel.',
    );
    expect(screen.queryByText(/in line|translating now/)).not.toBeInTheDocument();
    expect(screen.getByRole('status')).toHaveClass('translation-pending-notice--repair');
  });

  test('shows the paused message when translation is paused, even with a queue position', () => {
    renderWith(
      <TranslationPendingNotice translation={translation('pending', {}, 2)} />,
      contextValue({ status: { enabled: true, paused: true, state: 'paused' } }),
    );
    expect(screen.getByRole('status')).toHaveTextContent('Foreign Language Translation w/ Top-p is paused.');
  });

  test.each([
    [
      'not_configured',
      'Spanish translation is waiting for the translator models to be set up in Foreign Language Translation w/ Top-p. '
        + 'Open Foreign Language Translation w/ Top-p from the Skills And Enhancements panel.',
    ],
    ['waiting_for_credits', 'Spanish translation is waiting for provider credits. Showing the original until it is ready.'],
  ])('says what a queued translation waits for while the service state is %s', (state, text) => {
    renderWith(
      <TranslationPendingNotice translation={translation('pending', {}, 1)} />,
      contextValue({ status: { enabled: true, state } }),
    );
    expect(screen.getByRole('status')).toHaveTextContent(text);
    expect(screen.queryByText(/in line|translating now/)).not.toBeInTheDocument();
    expect(screen.getByRole('status')).toHaveClass('translation-pending-notice--paused');
  });

  const TRANSLATE_ALL_HINT = 'To translate your whole library in the background, use Translate all older items '
    + 'in the Foreign Language Translation w/ Top-p window (Skills And Enhancements).';

  test.each([
    [
      'queued',
      translation('pending', {}, 3),
      `Spanish translation queued — #3 in line. Showing the original until it is ready. ${TRANSLATE_ALL_HINT}`,
    ],
    [
      'pending',
      translation('pending'),
      'Spanish translation pending — 40% translated. Showing the original until it is ready. '
        + TRANSLATE_ALL_HINT,
    ],
  ])('a library item that is %s points to Translate all while it is off', (_, value, text) => {
    renderWith(
      <TranslationPendingNotice translation={value} translateAllModes={['autonomous']} />,
      contextValue({
        status: { enabled: true, state: 'translating', modes: ['autonomous', 'compiler'] },
        translateAll: { requested: false, state: 'off' },
      }),
    );
    expect(screen.getByRole('status')).toHaveTextContent(text);
  });

  test('a backend without the Translate all status block or saved modes still gets the hint', () => {
    renderWith(<TranslationPendingNotice translation={translation('pending', {}, 2)} translateAllModes={['leanoj']} />);
    expect(screen.getByRole('status')).toHaveTextContent(TRANSLATE_ALL_HINT);
  });

  test.each([
    [['autonomous'], 'Autonomous Research'],
    [['aggregator', 'compiler'], 'Aggregator and Single Paper Writer'],
    [['leanoj'], 'LeanOJ Proof Solver'],
  ])('while none of %j is checked the hint says to check %s in Current runs first', (viewerModes, names) => {
    renderWith(
      <TranslationPendingNotice translation={translation('pending', {}, 3)} translateAllModes={viewerModes} />,
      contextValue({ status: { enabled: true, state: 'translating', modes: [] } }),
    );
    expect(screen.getByRole('status')).toHaveTextContent(
      `Spanish translation queued — #3 in line. Showing the original until it is ready. To translate your whole `
        + `library in the background, check ${names} in Current runs, then use Translate all older items in the `
        + 'Foreign Language Translation w/ Top-p window (Skills And Enhancements).',
    );
  });

  test('with only some of a library\'s modes checked the hint names the items Translate all skips', () => {
    renderWith(
      <TranslationPendingNotice translation={translation('pending', {}, 3)} translateAllModes={['aggregator', 'compiler']} />,
      contextValue({ status: { enabled: true, state: 'translating', modes: ['compiler'] } }),
    );
    expect(screen.getByRole('status')).toHaveTextContent(
      'Spanish translation queued — #3 in line. Showing the original until it is ready. To translate your '
        + 'library in the background, use Translate all older items in the Foreign Language Translation w/ '
        + 'Top-p window (Skills And Enhancements). It skips Aggregator items until you check Aggregator in Current '
        + 'runs.',
    );
    expect(screen.getByRole('status')).not.toHaveTextContent('whole library');
  });

  test.each([
    ['Translate all is on', translation('pending', {}, 3), { translateAll: { requested: true, state: 'waiting' } }],
    ['the item is translating now', translation('pending', {}, 0), {}],
    ['the item is loading', translation('loading', { overall: { status: 'loading', translated: 0, total: 0 } }), {}],
    ['translation is paused', translation('pending', {}, 3), { status: { enabled: true, paused: true, state: 'paused' } }],
  ])('leaves the hint out while %s', (_, value, overrides) => {
    renderWith(<TranslationPendingNotice translation={value} translateAllModes={['autonomous']} />, contextValue(overrides));
    expect(screen.getByRole('status')).not.toHaveTextContent('Translate all');
  });

  test('viewers that are not libraries never show the hint', () => {
    renderWith(<TranslationPendingNotice translation={translation('pending', {}, 3)} />);
    expect(screen.getByRole('status')).not.toHaveTextContent('Translate all');
  });

  test.each([
    ['complete', translation('complete')],
    ['error', translation('error')],
    ['inactive', { ...translation('pending'), active: false }],
    ['missing', null],
  ])('renders nothing when %s', (_, value) => {
    const { container } = renderWith(<TranslationPendingNotice translation={value} />);
    expect(container).toBeEmptyDOMElement();
  });
});
