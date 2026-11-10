import React from 'react';
import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react';
import { afterEach, beforeEach, expect, test, vi } from 'vitest';
import BrainstormList from './BrainstormList';
import { AutoTranslateContext } from '../../hooks/useAutoTranslate';
import { translationAPI } from '../../services/api';
import { captureDownloads } from '../../test/downloads';

vi.mock('../../services/websocket', () => ({ websocket: { on: vi.fn(() => () => {}), off: vi.fn() } }));
vi.mock('../LatexRenderer', () => ({
  default: ({ content, documentId }) => <div data-testid="renderer" data-document={documentId}>{content}</div>,
}));
vi.mock('../../hooks/useProofCheckRuntime', () => ({
  isProofRunBusy: () => false,
  useProofCheckRuntime: () => ({ getSourceState: () => null, manualCheckEnabled: false }),
}));
vi.mock('./ProofRunStatusControls', () => ({ default: () => null }));
vi.mock('./ProofCheckModeModal', () => ({ default: () => null }));

const brainstorms = [{ topic_id: 'a', topic_prompt: 'Topic a', status: 'complete', submission_count: 2 }];
const ORIGINAL = 'Submission 1: the original brainstorm idea.';
const TRANSLATED = 'Propuesta 1: la idea original de la lluvia de ideas.';

function contextValue() {
  return {
    available: true,
    status: { enabled: true, state: 'translating' },
    enabled: true,
    languages: [{ key: 'es', name: 'Spanish', custom: false }],
    displayLanguage: 'es',
    setDisplayLanguage: vi.fn(),
    refreshStatus: vi.fn(),
    subscribeProgress: () => () => {},
    recentActivity: [],
  };
}

function renderList(content, { withProvider = true } = {}) {
  const api = { getBrainstorm: vi.fn().mockResolvedValue({ content }) };
  const list = <BrainstormList brainstorms={brainstorms} api={api} onRefresh={vi.fn()} />;
  render(withProvider
    ? <AutoTranslateContext.Provider value={contextValue()}>{list}</AutoTranslateContext.Provider>
    : list);
  fireEvent.click(screen.getByText('Topic a'));
  return api;
}

function respondWith(status) {
  return vi.spyOn(translationAPI, 'registerDocuments').mockImplementation(async ({ documents }) => ({
    success: true,
    documents: documents.map((document) => ({
      doc_key: document.docKey,
      doc_hash: `hash-${document.docKey}`,
      language: 'es',
      ...status,
    })),
  }));
}

function wait(ms) {
  return new Promise((resolve) => { setTimeout(resolve, ms); });
}

beforeEach(() => {
  vi.spyOn(console, 'warn').mockImplementation(() => {});
});

afterEach(() => {
  cleanup();
  vi.restoreAllMocks();
});

test('renders the translated brainstorm under a language-specific document identity', async () => {
  const register = respondWith({
    status: 'complete',
    translated_units: 1,
    total_units: 1,
    parts: { content: TRANSLATED },
  });
  renderList(ORIGINAL);

  await waitFor(() => {
    expect(screen.getByTestId('renderer')).toHaveTextContent(TRANSLATED);
  }, { timeout: 3000 });
  expect(screen.getByTestId('renderer')).toHaveAttribute('data-document', 'brainstorm:a:lang:es');
  expect(screen.getByTestId('renderer')).not.toHaveTextContent(ORIGINAL);
  expect(register).toHaveBeenCalledWith({
    language: 'es',
    viewing: true,
    documents: [{
      docKey: 'brainstorm:current:a',
      kind: 'brainstorm',
      title: 'Topic a',
      profile: 'document',
      parts: { content: ORIGINAL },
    }],
  });
  expect(screen.getByLabelText('Language')).toHaveValue('es');
  expect(screen.getByText('Translated')).toBeInTheDocument();
});

test('downloads the translated brainstorm being viewed, named for its language', async () => {
  respondWith({ status: 'complete', translated_units: 1, total_units: 1, parts: { content: TRANSLATED } });
  const saved = captureDownloads();
  renderList(ORIGINAL);
  await waitFor(() => {
    expect(screen.getByTestId('renderer')).toHaveTextContent(TRANSLATED);
  }, { timeout: 3000 });

  fireEvent.click(screen.getByRole('button', { name: 'Download' }));

  const [file] = await saved();
  expect(file.filename).toBe('a_brainstorm_es.txt');
  expect(file.text).toContain(TRANSLATED);
  expect(file.text).toContain('DISCLAIMER');
  expect(file.text).not.toContain(ORIGINAL);
});

test('downloads the original brainstorm while its translation is pending', async () => {
  respondWith({ status: 'pending', translated_units: 0, total_units: 2, parts: null });
  const saved = captureDownloads();
  renderList(ORIGINAL);
  expect(await screen.findByText(/Spanish translation pending/, {}, { timeout: 3000 })).toBeInTheDocument();

  fireEvent.click(screen.getByRole('button', { name: 'Download' }));

  const [file] = await saved();
  expect(file.filename).toBe('a_brainstorm.txt');
  expect(file.text).toContain(ORIGINAL);
});

test('keeps the original brainstorm while its translation is pending', async () => {
  respondWith({ status: 'pending', translated_units: 0, total_units: 2, parts: null });
  renderList(ORIGINAL);

  expect(await screen.findByText(
    'Spanish translation pending — 0% translated. Showing the original until it is ready.',
    {},
    { timeout: 3000 },
  )).toBeInTheDocument();
  expect(screen.getByTestId('renderer')).toHaveTextContent(ORIGINAL);
  expect(screen.getByTestId('renderer')).toHaveAttribute('data-document', 'brainstorm:a');
});

test('never sends the empty-brainstorm placeholder for translation', async () => {
  const register = respondWith({ status: 'complete', translated_units: 0, total_units: 0, parts: {} });
  renderList('');

  await waitFor(() => {
    expect(screen.getByTestId('renderer')).toHaveTextContent('No content yet...');
  });
  await wait(600);
  expect(register).not.toHaveBeenCalled();
});

test('renders no language switch and requests nothing without a translation provider', async () => {
  const register = vi.spyOn(translationAPI, 'registerDocuments');
  renderList(ORIGINAL, { withProvider: false });

  await waitFor(() => {
    expect(screen.getByTestId('renderer')).toHaveTextContent(ORIGINAL);
  });
  await wait(600);
  expect(screen.queryByLabelText('Language')).not.toBeInTheDocument();
  expect(screen.getByTestId('renderer')).toHaveAttribute('data-document', 'brainstorm:a');
  expect(register).not.toHaveBeenCalled();
});
