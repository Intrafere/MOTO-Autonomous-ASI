import React from 'react';
import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react';
import { afterEach, beforeEach, expect, test, vi } from 'vitest';
import FinalAnswerLibrary from './FinalAnswerLibrary';
import { AutoTranslateContext } from '../../hooks/useAutoTranslate';
import { autonomousAPI, translationAPI } from '../../services/api';
import { captureDownloads } from '../../test/downloads';

vi.mock('../../services/websocket', () => ({ websocket: { on: vi.fn(() => () => {}), off: vi.fn() } }));
vi.mock('../LatexRenderer', () => ({
  default: ({ content, documentId }) => <div data-testid="renderer" data-document={documentId}>{content}</div>,
}));
vi.mock('../PaperCritiqueModal', () => ({ default: () => null }));

const ANSWER = {
  answer_id: 'answer-1',
  session_id: 's1',
  title: 'Older Final Answer',
  format: 'short_form',
  certainty_level: 'partial_answer',
  user_prompt: 'An older research question',
  completion_date: '2026-10-01T12:00:00Z',
};
const PRUNED = {
  history_id: 's1:p2',
  session_id: 's1',
  paper_id: 'p2',
  title: 'Pruned History Paper',
  status: 'pruned',
};
const TRANSLATE_ALL_HINT = 'To translate your whole library in the background, use Translate all older items in '
  + 'the Foreign Language Translation w/ Top-p window (Skills And Enhancements).';

function contextValue(translateAll = null, status = { enabled: true, state: 'translating' }) {
  return {
    available: true,
    status,
    enabled: true,
    languages: [{ key: 'es', name: 'Spanish', custom: false }],
    displayLanguage: 'es',
    setDisplayLanguage: vi.fn(),
    refreshStatus: vi.fn(),
    subscribeProgress: () => () => {},
    recentActivity: [],
    translateAll,
  };
}

function renderLibrary(context = contextValue()) {
  render(
    <AutoTranslateContext.Provider value={context}>
      <FinalAnswerLibrary capabilities={{}} />
    </AutoTranslateContext.Provider>,
  );
}

beforeEach(() => {
  vi.spyOn(console, 'warn').mockImplementation(() => {});
  vi.spyOn(autonomousAPI, 'getSessions').mockResolvedValue({ sessions: [], current_session_id: null });
  vi.spyOn(autonomousAPI, 'getPaperHistory').mockResolvedValue({ papers: [] });
  vi.spyOn(autonomousAPI, 'getPrunedPaperHistory').mockResolvedValue({ papers: [PRUNED] });
  vi.spyOn(autonomousAPI, 'getPrunedHistoryPaper').mockResolvedValue({
    ...PRUNED,
    content: 'The pruned paper body.',
    outline: '',
  });
  vi.stubGlobal('fetch', vi.fn(async (url) => ({
    ok: true,
    json: async () => (url.endsWith('/final-answer-library')
      ? { success: true, final_answers: [ANSWER] }
      : { success: true, ...ANSWER, content: 'The older final answer body.' }),
  })));
  vi.spyOn(translationAPI, 'registerDocuments').mockImplementation(async ({ documents }) => ({
    success: true,
    documents: documents.map((document) => ({
      doc_key: document.docKey,
      doc_hash: `hash-${document.docKey}`,
      language: 'es',
      status: 'pending',
      translated_units: 0,
      total_units: 2,
      queue_position: 4,
      parts: null,
    })),
  }));
});

afterEach(() => {
  cleanup();
  vi.unstubAllGlobals();
  vi.restoreAllMocks();
});

test('older final answers and pruned papers point to Translate all while they wait', async () => {
  renderLibrary();
  fireEvent.click(await screen.findByText(/Older Final Answer/));
  const notice = await screen.findByText(/^Spanish translation queued/, {}, { timeout: 3000 });
  expect(notice).toHaveTextContent(
    `Spanish translation queued — #4 in line. Showing the original until it is ready. ${TRANSLATE_ALL_HINT}`,
  );
  expect(translationAPI.registerDocuments).toHaveBeenCalledWith(expect.objectContaining({
    documents: [expect.objectContaining({ docKey: 'final:answer-1', kind: 'final_answer' })],
  }));

  fireEvent.click(screen.getByText(/Pruned Papers \(1\)/));
  fireEvent.click(await screen.findByText('Pruned History Paper'));
  await waitFor(() => expect(screen.getAllByText(/^Spanish translation queued/)).toHaveLength(2), { timeout: 3000 });
  screen.getAllByText(/^Spanish translation queued/).forEach((node) => {
    expect(node).toHaveTextContent(TRANSLATE_ALL_HINT);
  });
  expect(translationAPI.registerDocuments).toHaveBeenCalledWith(expect.objectContaining({
    documents: [expect.objectContaining({ docKey: 'paper:s1:p2', kind: 'paper' })],
  }));
});

test('while Autonomous Research is unchecked the hint says to check it in Current runs first', async () => {
  renderLibrary(contextValue(null, { enabled: true, state: 'translating', modes: ['compiler'] }));
  const check = 'To translate your whole library in the background, check Autonomous Research in Current runs, '
    + 'then use Translate all older items in the Foreign Language Translation w/ Top-p window (Skills And '
    + 'Enhancements).';
  fireEvent.click(await screen.findByText(/Older Final Answer/));
  expect(await screen.findByText(/^Spanish translation queued/, {}, { timeout: 3000 })).toHaveTextContent(check);

  fireEvent.click(screen.getByText(/Pruned Papers \(1\)/));
  fireEvent.click(await screen.findByText('Pruned History Paper'));
  await waitFor(() => expect(screen.getAllByText(/^Spanish translation queued/)).toHaveLength(2), { timeout: 3000 });
  screen.getAllByText(/^Spanish translation queued/).forEach((node) => {
    expect(node).toHaveTextContent(check);
  });
});

test('the hint is left out while Translate all is already running', async () => {
  renderLibrary(contextValue({ requested: true, state: 'waiting' }));
  fireEvent.click(await screen.findByText(/Older Final Answer/));
  const notice = await screen.findByText(/^Spanish translation queued/, {}, { timeout: 3000 });
  expect(notice).not.toHaveTextContent('Translate all');
});

test('every download of an answer shown translated saves that translation', async () => {
  translationAPI.registerDocuments.mockImplementation(async ({ documents }) => ({
    success: true,
    documents: documents.map((document) => ({
      doc_key: document.docKey,
      doc_hash: `hash-${document.docKey}`,
      language: 'es',
      status: 'complete',
      translated_units: 2,
      total_units: 2,
      parts: { content: 'El cuerpo de la respuesta final anterior.' },
    })),
  }));
  const saved = captureDownloads();
  renderLibrary();
  fireEvent.click(await screen.findByText(/Older Final Answer/));
  await waitFor(() => {
    expect(screen.getByTestId('renderer')).toHaveTextContent('El cuerpo de la respuesta final anterior.');
  }, { timeout: 3000 });

  fireEvent.click(screen.getByRole('button', { name: /Download$/ }));
  fireEvent.click(screen.getByRole('button', { name: /Download Raw/ }));

  await waitFor(async () => expect(await saved()).toHaveLength(2));
  const files = await saved();
  expect(files.map((file) => file.filename)).toEqual([
    'final_answer_answer-1_es.txt',
    'Final_Answer_Older_Final_Answer_es.txt',
  ]);
  files.forEach((file) => {
    expect(file.text).toContain('El cuerpo de la respuesta final anterior.');
    expect(file.text).not.toContain('The older final answer body.');
  });
});
