import React from 'react';
import { cleanup, fireEvent, render, screen } from '@testing-library/react';
import { afterEach, beforeEach, expect, test, vi } from 'vitest';
import Stage2PaperHistory from './Stage2PaperHistory';
import { AutoTranslateContext } from '../../hooks/useAutoTranslate';
import { autonomousAPI, translationAPI } from '../../services/api';

vi.mock('../../services/websocket', () => ({ websocket: { on: vi.fn(() => () => {}), off: vi.fn() } }));
vi.mock('../LatexRenderer', () => ({
  default: ({ content, documentId }) => <div data-testid="renderer" data-document={documentId}>{content}</div>,
}));
vi.mock('../PaperCritiqueModal', () => ({ default: () => null }));
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

const PAPER = {
  history_id: 's1:p1',
  session_id: 's1',
  paper_id: 'p1',
  title: 'History Paper',
  abstract: 'An abstract.',
  status: 'complete',
};
const ORIGINAL = 'The original history paper body.';

function contextValue(status = { enabled: true, state: 'translating' }) {
  return {
    available: true,
    status,
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

async function renderHistory(status) {
  const context = contextValue(status);
  render(
    <AutoTranslateContext.Provider value={context}>
      <Stage2PaperHistory capabilities={{}} />
    </AutoTranslateContext.Provider>,
  );
  fireEvent.click(await screen.findByText('History Paper'));
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
  vi.spyOn(autonomousAPI, 'getPaperHistory').mockResolvedValue({ papers: [PAPER] });
  vi.spyOn(autonomousAPI, 'getPrunedPaperHistory').mockResolvedValue({ papers: [] });
  vi.spyOn(autonomousAPI, 'getSessions').mockResolvedValue({ sessions: [], current_session_id: null });
  vi.stubGlobal('fetch', vi.fn(async () => ({ ok: true, json: async () => ({ success: true, final_answers: [] }) })));
});

afterEach(() => {
  cleanup();
  vi.unstubAllGlobals();
  vi.restoreAllMocks();
});

test('choosing a language in an expanded history paper keeps the paper open', async () => {
  const register = respondPending();
  vi.spyOn(autonomousAPI, 'getHistoryPaper').mockResolvedValue({
    history_id: 's1:p1',
    session_id: 's1',
    paper_id: 'p1',
    title: 'History Paper',
    content: ORIGINAL,
    outline: '',
  });
  const context = await renderHistory();

  const select = await screen.findByLabelText('Language');
  // A history paper waiting for its translation points to Translate all.
  expect(await screen.findByText(/Spanish translation pending/, {}, { timeout: 3000 })).toHaveTextContent(
    'To translate your whole library in the background, use Translate all older items in the Foreign '
      + 'Language Translation w/ Top-p window (Skills And Enhancements).',
  );
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
});

test('while Autonomous Research is unchecked the hint says to check it in Current runs first', async () => {
  respondPending();
  vi.spyOn(autonomousAPI, 'getHistoryPaper').mockResolvedValue({
    history_id: 's1:p1',
    session_id: 's1',
    paper_id: 'p1',
    title: 'History Paper',
    content: ORIGINAL,
    outline: '',
  });
  await renderHistory({ enabled: true, state: 'translating', modes: ['aggregator', 'compiler'] });
  expect(await screen.findByText(/Spanish translation pending/, {}, { timeout: 3000 })).toHaveTextContent(
    'To translate your whole library in the background, check Autonomous Research in Current runs, then use '
      + 'Translate all older items in the Foreign Language Translation w/ Top-p window (Skills And Enhancements).',
  );
});

test('never registers the load-failure placeholder for translation', async () => {
  const register = respondPending();
  vi.spyOn(autonomousAPI, 'getHistoryPaper').mockRejectedValue(new Error('backend busy'));
  await renderHistory();

  expect(await screen.findByText('Failed to load content')).toBeInTheDocument();
  await wait(600);
  expect(register).not.toHaveBeenCalled();
  expect(screen.queryByLabelText('Language')).not.toBeInTheDocument();
});
