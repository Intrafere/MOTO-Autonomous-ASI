import React, { useState } from 'react';
import { render, screen, waitFor, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import TranslationLanguagePicker, {
  CUSTOM_LANGUAGE_TAG,
  UNCERTAIN_LANGUAGE_TAG,
  buildLanguageGroups,
  findLanguageMatch,
  formatRelativeTime,
  probeSummary,
  uncertainLanguageWarning,
} from './TranslationLanguagePicker';
import { translationAPI } from '../../services/api';

vi.mock('../../services/api', () => ({
  translationAPI: {
    getLanguages: vi.fn(),
    probeLanguages: vi.fn(),
  },
}));

const PINNED_KEYS = ['en', 'zh-Hans', 'hi', 'es', 'ar', 'fr'];

function catalog(overrides = {}) {
  return {
    success: true,
    pinned_keys: PINNED_KEYS,
    // Deliberately out of order: the pinned group must follow pinned_keys.
    languages: [
      { key: 'fr', name: 'French', native_name: 'Français', aliases: [], pinned: true, proficiency: 'proficient' },
      { key: 'yo', name: 'Yoruba', native_name: 'Yorùbá', aliases: [], pinned: false, proficiency: 'uncertain' },
      { key: 'en', name: 'English', native_name: '', aliases: [], pinned: true, proficiency: 'proficient' },
      { key: 'de', name: 'German', native_name: 'Deutsch', aliases: [], pinned: false, proficiency: 'proficient' },
      {
        key: 'zh-Hans',
        name: 'Mandarin Chinese',
        native_name: '简体中文',
        aliases: ['Chinese (Simplified)', 'Mandarin'],
        pinned: true,
        proficiency: 'proficient',
      },
      { key: 'hi', name: 'Hindi', native_name: 'हिन्दी', aliases: [], pinned: true, proficiency: 'proficient' },
      { key: 'af', name: 'Afrikaans', native_name: '', aliases: [], pinned: false, proficiency: 'unknown' },
      { key: 'es', name: 'Spanish', native_name: 'Español', aliases: [], pinned: true, proficiency: 'proficient' },
      {
        key: 'ar',
        name: 'Modern Standard Arabic',
        native_name: 'العربية',
        aliases: ['Arabic'],
        pinned: true,
        proficiency: 'proficient',
      },
      {
        key: 'pt-BR',
        name: 'Portuguese (Brazil)',
        native_name: 'Português (Brasil)',
        aliases: ['Brazilian Portuguese'],
        pinned: false,
        proficiency: 'proficient',
      },
    ],
    extras: [{ key: 'custom-abc1234567', name: 'Klingon', proficiency: 'proficient', reported: true }],
    probe: {
      status: 'complete',
      probed_at: new Date().toISOString(),
      models: [
        { role: 'validator', provider: 'openrouter', model_id: 'v', ok: true, error: '' },
        { role: 'translator', provider: 'openrouter', model_id: 's', ok: true, error: '' },
      ],
      stale: false,
    },
    ...overrides,
  };
}

const ENGLISH = { key: 'en', name: 'English', custom: false };
const YORUBA = { key: 'yo', name: 'Yoruba', custom: false };
const ELVISH = { key: 'custom-0123456789', name: 'Elvish', custom: true };

// Mirrors the modal: the saved list is replaced by what the backend returns.
function Harness({ initial = [], saveImpl, ...props }) {
  const [languages, setLanguages] = useState(initial);
  const onSaveLanguages = async (next) => {
    const saved = await saveImpl(next);
    setLanguages(saved || next);
    return { settings: { languages: saved || next } };
  };
  return (
    <TranslationLanguagePicker
      selectedLanguages={languages}
      onSaveLanguages={onSaveLanguages}
      modelsSaved
      rolesSignature="roles-1"
      {...props}
    />
  );
}

function renderPicker(props = {}) {
  const saveImpl = props.saveImpl || vi.fn(async (next) => next.map((language) => (
    language.key === 'custom-pending' ? { ...language, key: 'custom-0123456789' } : language
  )));
  const user = userEvent.setup();
  const view = render(<Harness {...props} saveImpl={saveImpl} />);
  return { ...view, saveImpl, user };
}

function rowNames(list) {
  return within(list).getAllByRole('listitem').map((item) => (
    item.querySelector('.auto-translate-language-row__name').textContent
  ));
}

function rowFor(name) {
  return screen.getByRole('checkbox', { name: new RegExp(`^${name}`) }).closest('li');
}

beforeEach(() => {
  vi.clearAllMocks();
  translationAPI.getLanguages.mockResolvedValue(catalog());
  translationAPI.probeLanguages.mockResolvedValue(catalog());
});

test('groups Most spoken in the pinned order, then every other language A–Z including model-reported extras', async () => {
  renderPicker();
  const mostSpoken = await screen.findByRole('list', { name: 'Most spoken' });
  expect(rowNames(mostSpoken)).toEqual([
    'English',
    'Mandarin Chinese',
    'Hindi',
    'Spanish',
    'Modern Standard Arabic',
    'French',
  ]);
  const all = screen.getByRole('list', { name: 'All languages (A–Z)' });
  expect(rowNames(all)).toEqual(['Afrikaans', 'German', 'Klingon', 'Portuguese (Brazil)', 'Yoruba']);
  expect(within(rowFor('French')).getByText('Français')).toBeInTheDocument();
});

test('uncertain languages are grey and carry a visible text tag; unknown and reported languages render normally', async () => {
  renderPicker();
  await screen.findByRole('list', { name: 'Most spoken' });
  const yoruba = rowFor('Yoruba');
  expect(yoruba).toHaveClass('auto-translate-language-row--uncertain');
  expect(within(yoruba).getByText(UNCERTAIN_LANGUAGE_TAG)).toBeInTheDocument();
  for (const name of ['Afrikaans', 'Klingon', 'German']) {
    const row = rowFor(name);
    expect(row).not.toHaveClass('auto-translate-language-row--uncertain');
    expect(within(row).queryByText(UNCERTAIN_LANGUAGE_TAG)).not.toBeInTheDocument();
  }
});

test('selecting an uncertain language saves it and shows the reliability warning', async () => {
  const { user, saveImpl } = renderPicker({ initial: [ENGLISH] });
  await screen.findByRole('list', { name: 'Most spoken' });
  await user.click(screen.getByRole('checkbox', { name: /^Yoruba/ }));
  expect(saveImpl).toHaveBeenCalledWith([ENGLISH, YORUBA]);
  expect(await screen.findByText(uncertainLanguageWarning('Yoruba'))).toBeInTheDocument();
  expect(screen.getByRole('checkbox', { name: /^Yoruba/ })).toBeChecked();
});

test('selecting a proficient language shows no warning and unchecking removes it', async () => {
  const { user, saveImpl } = renderPicker({ initial: [ENGLISH] });
  await screen.findByRole('list', { name: 'Most spoken' });
  await user.click(screen.getByRole('checkbox', { name: /^German/ }));
  expect(saveImpl).toHaveBeenLastCalledWith([ENGLISH, { key: 'de', name: 'German', custom: false }]);
  expect(screen.queryByText(/did not list/)).not.toBeInTheDocument();
  await waitFor(() => expect(screen.getByRole('checkbox', { name: /^German/ })).toBeChecked());
  await user.click(screen.getByRole('checkbox', { name: /^English/ }));
  expect(saveImpl).toHaveBeenLastCalledWith([{ key: 'de', name: 'German', custom: false }]);
});

test('model-reported extras save with their backend key as custom but display as proficient', async () => {
  const { user, saveImpl } = renderPicker();
  await screen.findByRole('list', { name: 'Most spoken' });
  await user.click(screen.getByRole('checkbox', { name: /^Klingon/ }));
  expect(saveImpl).toHaveBeenCalledWith([{ key: 'custom-abc1234567', name: 'Klingon', custom: true }]);
  const chips = await screen.findByRole('list', { name: 'Selected languages' });
  expect(within(chips).getByText('Klingon')).toBeInTheDocument();
  expect(within(chips).queryByText(CUSTOM_LANGUAGE_TAG)).not.toBeInTheDocument();
  expect(screen.queryByText(/did not list/)).not.toBeInTheDocument();
});

test('freeform add selects a matching catalog entry by name, native name, or alias', async () => {
  const { user, saveImpl } = renderPicker();
  await screen.findByRole('list', { name: 'Most spoken' });
  const input = screen.getByRole('textbox', { name: 'Add another language' });
  await user.type(input, '  arabic ');
  await user.click(screen.getByRole('button', { name: 'Add' }));
  expect(saveImpl).toHaveBeenLastCalledWith([{ key: 'ar', name: 'Modern Standard Arabic', custom: false }]);
  expect(input).toHaveValue('');

  await user.type(input, 'DEUTSCH');
  await user.keyboard('{Enter}');
  await waitFor(() => expect(saveImpl).toHaveBeenLastCalledWith([
    { key: 'ar', name: 'Modern Standard Arabic', custom: false },
    { key: 'de', name: 'German', custom: false },
  ]));
});

test('freeform add sends an unknown name as a pending custom language and shows the saved custom entry', async () => {
  const { user, saveImpl } = renderPicker({ initial: [ENGLISH] });
  await screen.findByRole('list', { name: 'Most spoken' });
  const input = screen.getByRole('textbox', { name: 'Add another language' });

  await user.click(screen.getByRole('button', { name: 'Add' }));
  await user.type(input, '   ');
  await user.click(screen.getByRole('button', { name: 'Add' }));
  expect(saveImpl).not.toHaveBeenCalled();

  await user.type(input, '  Elvish   Sindarin ');
  await user.click(screen.getByRole('button', { name: 'Add' }));
  expect(saveImpl).toHaveBeenCalledWith([
    ENGLISH,
    { key: 'custom-pending', name: 'Elvish Sindarin', custom: true },
  ]);
  expect(screen.getByText(uncertainLanguageWarning('Elvish Sindarin'))).toBeInTheDocument();
  const chips = await screen.findByRole('list', { name: 'Selected languages' });
  await waitFor(() => expect(screen.queryByText('Saving language changes…')).not.toBeInTheDocument());
  const customChip = within(chips).getByText('Elvish Sindarin').closest('li');
  expect(customChip).toHaveClass('auto-translate-chip--uncertain');
  expect(within(customChip).getByText(CUSTOM_LANGUAGE_TAG)).toBeInTheDocument();
});

test('chips show tags, reorder earlier, and remove; saves send only languages', async () => {
  const { user, saveImpl } = renderPicker({ initial: [ENGLISH, YORUBA, ELVISH] });
  await screen.findByRole('list', { name: 'Most spoken' });
  const chips = screen.getByRole('list', { name: 'Selected languages' });
  expect(screen.getByText('Languages are translated in this order.')).toBeInTheDocument();
  const items = within(chips).getAllByRole('listitem');
  expect(items.map((item) => item.querySelector('.auto-translate-chip__name').textContent))
    .toEqual(['English', 'Yoruba', 'Elvish']);
  expect(within(items[0]).queryByRole('button', { name: /earlier/ })).not.toBeInTheDocument();
  expect(within(items[0]).queryByText(UNCERTAIN_LANGUAGE_TAG)).not.toBeInTheDocument();
  expect(within(items[1]).getByText(UNCERTAIN_LANGUAGE_TAG)).toBeInTheDocument();
  expect(within(items[2]).getByText(CUSTOM_LANGUAGE_TAG)).toBeInTheDocument();

  await user.click(screen.getByRole('button', { name: 'Move Yoruba earlier' }));
  expect(saveImpl).toHaveBeenLastCalledWith([YORUBA, ENGLISH, ELVISH]);
  await waitFor(() => expect(screen.getByRole('button', { name: 'Move English earlier' })).toBeInTheDocument());

  await user.click(screen.getByRole('button', { name: 'Remove Elvish' }));
  await waitFor(() => expect(saveImpl).toHaveBeenLastCalledWith([YORUBA, ENGLISH]));
});

test('save errors are shown inline and the saved list is kept', async () => {
  const saveImpl = vi.fn(async () => {
    throw Object.assign(new Error('HTTP 422 Unprocessable Entity: Unknown language key.'), { status: 422 });
  });
  const { user } = renderPicker({ initial: [ENGLISH], saveImpl });
  await screen.findByRole('list', { name: 'Most spoken' });
  await user.click(screen.getByRole('checkbox', { name: /^German/ }));
  expect(await screen.findByRole('alert')).toHaveTextContent('Unknown language key.');
  expect(screen.getByRole('checkbox', { name: /^German/ })).not.toBeChecked();
});

test('search filters by name, native name, and alias', async () => {
  const { user } = renderPicker();
  await screen.findByRole('list', { name: 'Most spoken' });
  const search = screen.getByRole('searchbox', { name: 'Search languages' });
  await user.type(search, 'deut');
  expect(screen.queryByRole('list', { name: 'Most spoken' })).not.toBeInTheDocument();
  expect(rowNames(screen.getByRole('list', { name: 'All languages (A–Z)' }))).toEqual(['German']);
  await user.clear(search);
  await user.type(search, 'mandarin');
  expect(rowNames(screen.getByRole('list', { name: 'Most spoken' }))).toEqual(['Mandarin Chinese']);
  await user.clear(search);
  await user.type(search, 'zzz');
  expect(screen.getByText('No languages match “zzz”.')).toBeInTheDocument();
});

describe('model language check', () => {
  test('summarizes a completed check and runs a new check on request', async () => {
    const { user } = renderPicker();
    expect(await screen.findByText('Checked just now with 2 models.')).toBeInTheDocument();
    expect(translationAPI.probeLanguages).not.toHaveBeenCalled();
    translationAPI.probeLanguages.mockResolvedValueOnce(catalog({
      probe: { ...catalog().probe, models: [{ role: 'validator', ok: true }] },
    }));
    await user.click(screen.getByRole('button', { name: 'Check model languages' }));
    expect(translationAPI.probeLanguages).toHaveBeenCalledTimes(1);
    expect(await screen.findByText('Checked just now with 1 model.')).toBeInTheDocument();
  });

  test('runs automatically once when never checked and models are saved', async () => {
    translationAPI.getLanguages.mockResolvedValue(catalog({ probe: { status: 'never', models: [], stale: false } }));
    renderPicker();
    await waitFor(() => expect(translationAPI.probeLanguages).toHaveBeenCalledTimes(1));
    expect(await screen.findByText('Checked just now with 2 models.')).toBeInTheDocument();
    expect(translationAPI.probeLanguages).toHaveBeenCalledTimes(1);
  });

  test('runs automatically when the saved models changed since the last check', async () => {
    const staleCatalog = catalog({ probe: { ...catalog().probe, stale: true } });
    translationAPI.getLanguages.mockResolvedValue(staleCatalog);
    translationAPI.probeLanguages.mockImplementation(() => new Promise(() => {}));
    renderPicker();
    expect(await screen.findByText(/Models changed since the last check\./)).toBeInTheDocument();
    await waitFor(() => expect(translationAPI.probeLanguages).toHaveBeenCalledTimes(1));
    expect(screen.getByRole('button', { name: 'Checking model languages…' })).toBeDisabled();
  });

  test('does not check without saved models or while the tab is hidden', async () => {
    translationAPI.getLanguages.mockResolvedValue(catalog({ probe: { status: 'never', models: [], stale: false } }));
    const { unmount } = renderPicker({ modelsSaved: false });
    expect(await screen.findByText('Check which languages your translator models handle well.')).toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Check model languages' })).toBeDisabled();
    expect(screen.getByText('Save translator models first to check their languages.')).toBeInTheDocument();
    unmount();

    renderPicker({ active: false });
    await screen.findByText('Check which languages your translator models handle well.');
    expect(translationAPI.probeLanguages).not.toHaveBeenCalled();
  });

  test('shows a failed check inline', async () => {
    const { user } = renderPicker();
    await screen.findByText('Checked just now with 2 models.');
    translationAPI.probeLanguages.mockRejectedValueOnce(
      Object.assign(new Error('HTTP 422 Unprocessable Entity: Configure translator models first.'), { status: 422 }),
    );
    await user.click(screen.getByRole('button', { name: 'Check model languages' }));
    expect(await screen.findByRole('alert')).toHaveTextContent('Configure translator models first.');
  });
});

describe('pure helpers', () => {
  test('buildLanguageGroups falls back to pinned flags when pinned_keys is missing', () => {
    const groups = buildLanguageGroups({ ...catalog(), pinned_keys: undefined });
    expect(groups.pinned.map((row) => row.key)).toEqual(['fr', 'en', 'zh-Hans', 'hi', 'es', 'ar']);
  });

  test('findLanguageMatch matches case-insensitively and ignores unknown names', () => {
    expect(findLanguageMatch(catalog(), 'chinese (simplified)').key).toBe('zh-Hans');
    expect(findLanguageMatch(catalog(), 'klingon').reported).toBe(true);
    expect(findLanguageMatch(catalog(), 'Elvish')).toBeNull();
  });

  test('relative time and probe summaries', () => {
    const now = Date.parse('2026-11-01T12:00:00Z');
    expect(formatRelativeTime('2026-11-01T11:59:40Z', now)).toBe('just now');
    expect(formatRelativeTime('2026-11-01T11:55:00Z', now)).toBe('5 minutes ago');
    expect(formatRelativeTime('2026-11-01T10:00:00Z', now)).toBe('2 hours ago');
    expect(formatRelativeTime('2026-10-30T12:00:00Z', now)).toBe('2 days ago');
    expect(probeSummary({ status: 'failed', stale: true })).toBe(
      'The last language check failed. Models changed since the last check.',
    );
    expect(probeSummary({
      status: 'partial',
      probed_at: '2026-11-01T11:00:00Z',
      models: [{ ok: true }, { ok: false }],
    }, now)).toBe('Checked 1 hour ago with 1 model. Some models could not be checked.');
  });
});
