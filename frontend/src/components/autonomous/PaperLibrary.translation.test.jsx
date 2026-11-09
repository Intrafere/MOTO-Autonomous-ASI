import React from 'react';
import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react';
import { afterEach, beforeEach, expect, test, vi } from 'vitest';
import PaperLibrary from './PaperLibrary';
import { AutoTranslateContext } from '../../hooks/useAutoTranslate';
import { translationAPI } from '../../services/api';
import { captureDownloads } from '../../test/downloads';
import { downloadPDFViaBackend } from '../../utils/downloadHelpers';

vi.mock('../../services/websocket', () => ({ websocket: { on: vi.fn(() => () => {}), off: vi.fn() } }));
vi.mock('../LatexRenderer', () => ({
  default: ({ content, documentId }) => <div data-testid="renderer" data-document={documentId}>{content}</div>,
}));
vi.mock('../PaperCritiqueModal', () => ({ default: () => null }));
vi.mock('../../utils/downloadHelpers', async (importOriginal) => ({
  ...(await importOriginal()),
  downloadPDFViaBackend: vi.fn(async () => {}),
}));
vi.mock('../../hooks/useProofCheckRuntime', () => ({
  isProofRunBusy: () => false,
  useProofCheckRuntime: () => ({
    getSourceState: () => null,
    manualCheckEnabled: false,
    manualCheckReason: '',
    queueManualProofCheck: vi.fn(),
    stopProofRun: vi.fn(),
  }),
}));
vi.mock('./ProofRunStatusControls', () => ({ default: () => null }));
vi.mock('./ProofCheckModeModal', () => ({ default: () => null }));

const PAPER = { paper_id: 'p1', session_id: 's1', title: 'Paper One', abstract: 'An abstract.' };
const ORIGINAL = 'The original paper body.';

function contextValue() {
  return {
    available: true,
    status: { enabled: true, state: 'translating' },
    enabled: true,
    languages: [{ key: 'es', name: 'Spanish', custom: false }],
    displayLanguage: 'es',
    setDisplayLanguage: vi.fn(),
    setDocumentLanguages: vi.fn(),
    refreshStatus: vi.fn(),
    subscribeProgress: () => () => {},
    recentActivity: [],
  };
}

function renderLibrary(getAutonomousPaper) {
  const context = contextValue();
  render(
    <AutoTranslateContext.Provider value={context}>
      <PaperLibrary papers={[PAPER]} onRefresh={vi.fn()} api={{ getAutonomousPaper }} capabilities={{}} />
    </AutoTranslateContext.Provider>,
  );
  fireEvent.click(screen.getByText('Paper One'));
  return context;
}

function respondPending() {
  return vi.spyOn(translationAPI, 'registerDocuments').mockImplementation(async ({ documents }) => ({
    success: true,
    documents: documents.map((document) => ({
      doc_key: document.docKey,
      doc_hash: `hash-${document.docKey}`,
      language: 'es',
      status: 'pending',
      translated_units: 0,
      total_units: 1,
      parts: null,
    })),
  }));
}

function wait(ms) {
  return new Promise((resolve) => { setTimeout(resolve, ms); });
}

beforeEach(() => {
  vi.spyOn(console, 'warn').mockImplementation(() => {});
  vi.spyOn(console, 'error').mockImplementation(() => {});
});

afterEach(() => {
  cleanup();
  vi.restoreAllMocks();
});

test('choosing a language in an expanded paper keeps the paper open', async () => {
  const register = respondPending();
  const context = renderLibrary(vi.fn().mockResolvedValue({ content: ORIGINAL, outline: '', title: 'Paper One' }));

  const select = await screen.findByLabelText('Language');
  expect(await screen.findByText(/Spanish translation pending/, {}, { timeout: 3000 })).toBeInTheDocument();
  expect(register).toHaveBeenCalledWith(expect.objectContaining({
    documents: [expect.objectContaining({ docKey: 'paper:s1:p1', parts: { content: ORIGINAL } })],
  }));

  fireEvent.pointerDown(select);
  fireEvent.mouseDown(select);
  fireEvent.click(select);
  fireEvent.change(select, { target: { value: 'original' } });

  // The choice applies to this paper only, never the default language.
  expect(context.setDocumentLanguages).toHaveBeenCalledWith(['paper:s1:p1'], 'original');
  expect(context.setDisplayLanguage).not.toHaveBeenCalled();
  expect(screen.getByText('Paper Content')).toBeInTheDocument();
  expect(screen.getByLabelText('Language')).toBeInTheDocument();
});

test('never registers the load-failure placeholder for translation', async () => {
  const register = respondPending();
  renderLibrary(vi.fn().mockRejectedValue(new Error('backend busy')));

  expect(await screen.findByText('Failed to load content')).toBeInTheDocument();
  await wait(600);
  expect(register).not.toHaveBeenCalled();
  expect(screen.queryByLabelText('Language')).not.toBeInTheDocument();
});

test('raw and PDF downloads of a paper shown translated save that translation', async () => {
  const TRANSLATED = 'El cuerpo original del artículo.';
  vi.spyOn(translationAPI, 'registerDocuments').mockImplementation(async ({ documents }) => ({
    success: true,
    documents: documents.map((document) => ({
      doc_key: document.docKey,
      doc_hash: `hash-${document.docKey}`,
      language: 'es',
      status: 'complete',
      translated_units: 1,
      total_units: 1,
      parts: { content: TRANSLATED },
    })),
  }));
  const saved = captureDownloads();
  renderLibrary(vi.fn().mockResolvedValue({ content: ORIGINAL, outline: 'Outline', title: 'Paper One' }));
  expect(await screen.findByText(TRANSLATED, { exact: false }, { timeout: 3000 })).toBeInTheDocument();

  fireEvent.click(screen.getByRole('button', { name: 'Download Raw' }));
  const [file] = await saved();
  expect(file.filename).toBe('p1_Paper_One_es.txt');
  expect(file.text).toContain('Outline');
  expect(file.text).toContain(TRANSLATED);
  expect(file.text).not.toContain(ORIGINAL);

  fireEvent.click(screen.getByRole('button', { name: 'Download PDF' }));
  await waitFor(() => expect(downloadPDFViaBackend).toHaveBeenCalledTimes(1));
  const [content, metadata, filename] = downloadPDFViaBackend.mock.calls[0];
  expect(content).toBe(TRANSLATED);
  expect(filename).toBe('p1_Paper_One_es');
  expect(metadata.wordCount).toBeNull();
});
