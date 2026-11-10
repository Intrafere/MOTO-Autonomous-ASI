import React from 'react';
import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react';
import { afterEach, beforeEach, expect, test, vi } from 'vitest';
import LiveResults from './LiveResults';
import { AutoTranslateContext } from '../../hooks/useAutoTranslate';
import { api, translationAPI } from '../../services/api';
import { captureDownloads } from '../../test/downloads';

vi.mock('../../services/websocket', () => ({ websocket: { on: vi.fn(() => () => {}), off: vi.fn() } }));
vi.mock('../LatexRenderer', () => ({
  default: ({ content, documentId }) => <div data-testid="renderer" data-document={documentId ?? ''}>{content}</div>,
}));
vi.mock('../../hooks/useProofCheckRuntime', () => ({
  isProofRunBusy: () => false,
  MANUAL_AGGREGATOR_PROOF_SOURCE_ID: 'manual_aggregator_current',
  useProofCheckRuntime: () => ({
    canQueueManualProofCheck: () => false,
    getManualCheckReason: () => '',
    getSourceState: () => null,
    queueManualProofCheck: vi.fn(),
    stopProofRun: vi.fn(),
  }),
}));
vi.mock('../autonomous/ProofRunStatusControls', () => ({ default: () => null }));
vi.mock('../autonomous/ProofCheckModeModal', () => ({ default: () => null }));

const ORIGINAL = 'Accepted submission 1: an original aggregator insight.';
const TRANSLATED = 'Propuesta aceptada 1: una idea original del agregador.';

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

function renderResults({ withProvider = true } = {}) {
  render(withProvider
    ? <AutoTranslateContext.Provider value={contextValue()}><LiveResults /></AutoTranslateContext.Provider>
    : <LiveResults />);
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
  vi.spyOn(api, 'getResults').mockResolvedValue({ results: ORIGINAL });
});

afterEach(() => {
  cleanup();
  vi.restoreAllMocks();
});

test('renders translated aggregator results under a language-specific document identity', async () => {
  const register = respondWith({
    status: 'complete',
    translated_units: 1,
    total_units: 1,
    parts: { content: TRANSLATED },
  });
  renderResults();

  await waitFor(() => {
    expect(screen.getByTestId('renderer')).toHaveTextContent(TRANSLATED);
  }, { timeout: 3000 });
  expect(screen.getByTestId('renderer')).toHaveAttribute('data-document', 'aggregator:results:lang:es');
  expect(screen.getByTestId('renderer')).not.toHaveTextContent(ORIGINAL);
  expect(register).toHaveBeenCalledWith({
    language: 'es',
    viewing: true,
    documents: [{
      docKey: 'aggregator:results',
      kind: 'aggregator_results',
      title: 'Live Results',
      profile: 'document',
      parts: { content: ORIGINAL },
    }],
  });
  expect(screen.getByLabelText('Language')).toHaveValue('es');
});

test('downloads the translated results being viewed, named for their language', async () => {
  respondWith({ status: 'complete', translated_units: 1, total_units: 1, parts: { content: TRANSLATED } });
  const saved = captureDownloads();
  renderResults();
  await waitFor(() => {
    expect(screen.getByTestId('renderer')).toHaveTextContent(TRANSLATED);
  }, { timeout: 3000 });

  fireEvent.click(screen.getByRole('button', { name: 'Download' }));

  const [file] = await saved();
  expect(file.filename).toMatch(/^aggregator_results_\d{4}-\d{2}-\d{2}_es\.txt$/);
  expect(file.text).toContain(TRANSLATED);
  expect(file.text).not.toContain(ORIGINAL);
});

test('keeps the original results while the translation is pending', async () => {
  respondWith({ status: 'pending', translated_units: 1, total_units: 4, parts: null });
  renderResults();

  expect(await screen.findByText(
    'Spanish translation pending — 25% translated. Showing the original until it is ready.',
    {},
    { timeout: 3000 },
  )).toBeInTheDocument();
  expect(screen.getByTestId('renderer')).toHaveTextContent(ORIGINAL);
  expect(screen.getByTestId('renderer')).toHaveAttribute('data-document', '');
});

test('never sends the no-results placeholder for translation', async () => {
  api.getResults.mockResolvedValue({ results: '' });
  const register = respondWith({ status: 'complete', translated_units: 0, total_units: 0, parts: {} });
  renderResults();

  await waitFor(() => {
    expect(screen.getByTestId('renderer')).toHaveTextContent('No accepted submissions yet.');
  });
  await wait(600);
  expect(register).not.toHaveBeenCalled();
});

test('never registers an Autonomous run\'s database shown here under the manual results key', async () => {
  api.getResults.mockResolvedValue({ results: ORIGINAL, manual_database: false });
  const register = respondWith({ status: 'complete', translated_units: 1, total_units: 1, parts: { content: TRANSLATED } });
  renderResults();

  await waitFor(() => {
    expect(screen.getByTestId('renderer')).toHaveTextContent(ORIGINAL);
  });
  await wait(600);
  expect(register).not.toHaveBeenCalled();
  expect(screen.getByTestId('renderer')).toHaveAttribute('data-document', '');
});

test.each([
  ['marks the manual database', { manual_database: true }],
  ['comes from a backend without the flag', {}],
])('registers the results when the response %s', async (_, flag) => {
  api.getResults.mockResolvedValue({ results: ORIGINAL, ...flag });
  const register = respondWith({ status: 'complete', translated_units: 1, total_units: 1, parts: { content: TRANSLATED } });
  renderResults();

  await waitFor(() => {
    expect(screen.getByTestId('renderer')).toHaveTextContent(TRANSLATED);
  }, { timeout: 3000 });
  expect(register.mock.calls[0][0].documents[0]).toMatchObject({ docKey: 'aggregator:results', parts: { content: ORIGINAL } });
});

test('stops registering once the results stop coming from the manual database', async () => {
  const register = respondWith({ status: 'pending', translated_units: 0, total_units: 1, parts: null });
  renderResults();
  await waitFor(() => expect(register).toHaveBeenCalledTimes(1), { timeout: 3000 });

  api.getResults.mockResolvedValue({ results: 'Autonomous brainstorm submission.', manual_database: false });
  fireEvent.click(screen.getByRole('button', { name: 'Refresh' }));
  await waitFor(() => {
    expect(screen.getByTestId('renderer')).toHaveTextContent('Autonomous brainstorm submission.');
  });
  await wait(600);
  expect(register).toHaveBeenCalledTimes(1);
  expect(screen.queryByText(/translation pending/)).not.toBeInTheDocument();
});

test('renders no language switch and requests nothing without a translation provider', async () => {
  const register = vi.spyOn(translationAPI, 'registerDocuments');
  renderResults({ withProvider: false });

  await waitFor(() => {
    expect(screen.getByTestId('renderer')).toHaveTextContent(ORIGINAL);
  });
  await wait(600);
  expect(screen.queryByLabelText('Language')).not.toBeInTheDocument();
  expect(register).not.toHaveBeenCalled();
});
