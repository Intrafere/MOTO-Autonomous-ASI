import React from 'react';
import { act, cleanup, fireEvent, render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { afterEach, beforeEach, describe, expect, test, vi } from 'vitest';

vi.mock('./LatexRenderer', () => ({ default: ({ content, documentId }) => <div data-testid="latex-segment" data-document={documentId}>{content}</div> }));
vi.mock('../services/websocket', () => ({ websocket: { on: vi.fn(() => () => {}), off: vi.fn() } }));
vi.mock('../services/api', () => ({
  translationAPI: { registerDocuments: vi.fn(), lookupDocuments: vi.fn(), documentProgress: vi.fn() },
}));
import PaperProofViewer from './PaperProofViewer';
import useViewedTranslations from '../hooks/useViewedTranslations';
import { AutoTranslateContext } from '../hooks/useAutoTranslate';
import { translationAPI } from '../services/api';
import { PAPER_DISCLAIMER } from '../utils/disclaimerHelper';

const content = [
  'Readable paper body.',
  '=== PROOFS ATTACHED TO THIS PAPER (Lean 4 Verified) ===',
  'Proof 1: Addition identity',
  'Proof ID: proof_add',
  'Lean 4 Code:',
  'by omega',
  '---',
].join('\n');

const translatedContent = content.replace('Readable paper body.', 'Cuerpo legible del artículo con más palabras.');

describe('PaperProofViewer', () => {
  test('propagates document and segment identity across document switches', () => {
    const { rerender, container } = render(<PaperProofViewer documentId="session:a" prefixContent="Outline" content={content} />);
    const ids = () => screen.getAllByTestId('latex-segment').map(node => node.dataset.document);
    expect(ids()).toEqual(['session:a:prefix', 'session:a:paper:0', 'session:a:proof:proof_add:1']);
    container.querySelector('details').open = true;
    rerender(<PaperProofViewer documentId="session:b" prefixContent="New outline" content={content.replace('Readable', 'New')} />);
    expect(ids()).toEqual(['session:b:prefix', 'session:b:paper:0', 'session:b:proof:proof_add:1']);
    expect(container.querySelector('details').open).toBe(false);
    expect(screen.getByRole('button', { name: 'Rendered View' })).toHaveAttribute('aria-pressed', 'true');
  });
  test('defaults rendered with collapsed proofs and preserves complete raw source', async () => {
    const user = userEvent.setup();
    const { container } = render(<PaperProofViewer content={content} />);
    const proof = screen.getByText('Proof 1: proof_add').closest('details');
    expect(proof.open).toBe(false);
    expect(screen.getByRole('button', { name: 'Rendered View' })).toHaveAttribute('aria-pressed', 'true');
    expect(screen.getByLabelText('Paper content metrics')).toHaveTextContent('Proofs (1)');

    await user.click(screen.getByRole('button', { name: 'Raw View' }));
    expect(container.querySelector('.paper-proof-raw-content').textContent).toBe(content);
  });
});

describe('PaperProofViewer translation', () => {
  const LANGUAGES = [{ key: 'es', name: 'Spanish', custom: false }];

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

  function renderTranslated(ui, value = contextValue()) {
    const view = render(<AutoTranslateContext.Provider value={value}>{ui}</AutoTranslateContext.Provider>);
    return {
      ...view,
      rerender: (next, nextValue = value) => view.rerender(
        <AutoTranslateContext.Provider value={nextValue}>{next}</AutoTranslateContext.Provider>,
      ),
    };
  }

  function respondWith(overrides) {
    translationAPI.registerDocuments.mockImplementation(async ({ language, documents }) => ({
      success: true,
      documents: documents.map((document) => ({
        doc_key: document.docKey,
        doc_hash: `hash-${document.docKey}`,
        language,
        status: 'pending',
        translated_units: 0,
        total_units: 3,
        parts: null,
        ...overrides,
      })),
    }));
  }

  async function settle() {
    await act(async () => {
      await vi.advanceTimersByTimeAsync(500);
    });
    await act(async () => {
      for (let index = 0; index < 10; index += 1) {
        await Promise.resolve();
      }
    });
  }

  const ids = () => screen.getAllByTestId('latex-segment').map(node => node.dataset.document);
  const texts = () => screen.getAllByTestId('latex-segment').map(node => node.textContent);

  beforeEach(() => {
    vi.useFakeTimers();
    translationAPI.registerDocuments.mockReset();
    translationAPI.lookupDocuments.mockReset();
    translationAPI.documentProgress.mockReset();
    translationAPI.documentProgress.mockResolvedValue({ success: true, documents: [] });
  });

  afterEach(() => {
    vi.useRealTimers();
  });

  test('shows no switch and makes no translation requests without a provider', async () => {
    render(<PaperProofViewer documentId="paper:run:p1" content={content} />);
    await settle();

    expect(screen.queryByLabelText('Language')).not.toBeInTheDocument();
    expect(texts()).toContain('Readable paper body.\n');
    expect(translationAPI.registerDocuments).not.toHaveBeenCalled();
  });

  test('always shows the language switch for a keyed document, even with translation off', async () => {
    const original = contextValue({ displayLanguage: 'original' });
    const { rerender } = renderTranslated(<PaperProofViewer documentId="paper:run:p1" content={content} />, original);
    expect(screen.getByLabelText('Language')).toHaveValue('original');

    rerender(<PaperProofViewer documentId="paper:run:p1" content={content} />, contextValue({ enabled: false }));
    const select = screen.getByLabelText('Language');
    expect(select).toHaveValue('original');
    expect([...select.options].map((option) => option.textContent)).toEqual(['Original', 'Spanish (translation off)']);

    // Placeholders and unkeyed content have nothing to translate.
    rerender(<PaperProofViewer documentId="paper:run:p1" content={content} translationEnabled={false} />, original);
    expect(screen.queryByLabelText('Language')).not.toBeInTheDocument();
    rerender(<PaperProofViewer content={content} />, original);
    expect(screen.queryByLabelText('Language')).not.toBeInTheDocument();

    await settle();
    expect(translationAPI.registerDocuments).not.toHaveBeenCalled();
  });

  test('a language this paper has not finished is greyed out; a finished one is not', async () => {
    translationAPI.documentProgress.mockResolvedValue({
      success: true,
      documents: [{
        doc_key: 'paper:run:p1',
        doc_hash: 'hash-paper:run:p1',
        languages: [
          { language: 'es', status: 'complete', translated_units: 3, total_units: 3 },
          { language: 'fr', status: 'pending', translated_units: 1, total_units: 4 },
        ],
      }],
    });
    renderTranslated(
      <PaperProofViewer documentId="paper:run:p1" content={content} />,
      contextValue({ displayLanguage: 'original', languages: [...LANGUAGES, { key: 'fr', name: 'French', custom: false }] }),
    );
    await settle();

    // Read-only: showing the original never registers the paper for translation.
    expect(translationAPI.documentProgress).toHaveBeenCalledWith({
      documents: [expect.objectContaining({ docKey: 'paper:run:p1', parts: { content } })],
    });
    expect(translationAPI.registerDocuments).not.toHaveBeenCalled();
    const [, spanish, french] = screen.getByLabelText('Language').options;
    expect(spanish).toHaveTextContent(/^Spanish$/);
    expect(spanish).toHaveAttribute('data-ready', 'true');
    expect(french).toHaveTextContent('French (25%)');
    expect(french).toHaveClass('translation-language-switch__option--unready');
  });

  test('replaces the original with the complete translation and tags its identity with the language', async () => {
    respondWith({ status: 'complete', translated_units: 3, parts: { content: translatedContent } });
    const disclaimed = `${PAPER_DISCLAIMER}\n\n${content}`;
    const { container } = renderTranslated(
      <PaperProofViewer
        documentId="paper:run:p1"
        translationTitle="Addition paper"
        prefixContent="Outline stays original"
        content={disclaimed}
        metrics={{ paper_word_count: 999 }}
      />,
    );
    await settle();

    expect(translationAPI.registerDocuments).toHaveBeenCalledTimes(1);
    expect(translationAPI.registerDocuments.mock.calls[0][0]).toMatchObject({
      language: 'es',
      viewing: true,
      documents: [{
        docKey: 'paper:run:p1',
        kind: 'paper',
        title: 'Addition paper',
        profile: 'document',
        parts: { content },
      }],
    });
    expect(ids()).toEqual([
      'paper:run:p1:lang:es:prefix',
      'paper:run:p1:lang:es:paper:0',
      'paper:run:p1:lang:es:proof:proof_add:1',
    ]);
    expect(texts()[0]).toBe('Outline stays original');
    expect(texts()[1]).toContain('Cuerpo legible del artículo');
    expect(texts()[1]).toContain('DISCLAIMER');
    expect(texts()[1]).not.toContain('Readable paper body.');
    expect(screen.getByText('Translated')).toBeInTheDocument();
    expect(screen.queryByRole('status')).not.toBeInTheDocument();
    expect(screen.getByLabelText('Paper content metrics')).not.toHaveTextContent('999');

    fireEvent.click(screen.getByRole('button', { name: 'Raw View' }));
    expect(container.querySelector('.paper-proof-raw-content').textContent)
      .toBe(`Outline stays original${PAPER_DISCLAIMER}\n\n${translatedContent}`);
  });

  test('keeps the original with a pending notice until the translation is ready', async () => {
    respondWith({ status: 'pending', translated_units: 1, total_units: 3 });
    renderTranslated(<PaperProofViewer documentId="paper:run:p1" content={content} metrics={{ paper_word_count: 999 }} />);
    await settle();

    expect(ids()).toEqual(['paper:run:p1:paper:0', 'paper:run:p1:proof:proof_add:1']);
    expect(texts()[0]).toContain('Readable paper body.');
    expect(screen.getByText(/Spanish translation pending — 33% translated/)).toHaveAttribute('role', 'status');
    expect(screen.getByLabelText('Paper content metrics')).toHaveTextContent('999');
  });

  test('library viewers add the Translate all hint to the pending notice; others do not', async () => {
    respondWith({ status: 'pending', translated_units: 1, total_units: 3, queue_position: 2 });
    const { rerender } = renderTranslated(<PaperProofViewer documentId="paper:old:p1" content={content} translateAllModes={['autonomous']} />);
    await settle();
    expect(screen.getByText(/^Spanish translation queued/)).toHaveTextContent(
      'Spanish translation queued — #2 in line. Showing the original until it is ready. To translate your whole '
        + 'library in the background, use Translate all older items in the Foreign Language Translation w/ '
        + 'Top-p window (Skills And Enhancements).',
    );

    rerender(<PaperProofViewer documentId="paper:old:p1" content={content} />);
    expect(screen.getByText(/^Spanish translation queued/)).not.toHaveTextContent('Translate all');
  });

  test('shows the units done while the paper is translating now', async () => {
    respondWith({ status: 'pending', translated_units: 1, total_units: 3, queue_position: 0 });
    renderTranslated(<PaperProofViewer documentId="paper:run:p1" content={content} />);
    await settle();

    expect(texts()[0]).toContain('Readable paper body.');
    expect(screen.getByText('Spanish: translating now — 33% done')).toHaveAttribute('role', 'status');
    expect(screen.getByText('Translating 33%')).toBeInTheDocument();
  });

  test('registers under an explicit translation key and kind', async () => {
    respondWith({ status: 'complete', translated_units: 3, parts: { content: translatedContent } });
    renderTranslated(
      <PaperProofViewer
        documentId="tier3:run:final"
        translationKey="final:run"
        translationKind="final_answer"
        content={content}
      />,
    );
    await settle();

    expect(translationAPI.registerDocuments.mock.calls[0][0].documents[0]).toMatchObject({
      docKey: 'final:run',
      kind: 'final_answer',
    });
    expect(ids()).toEqual(['tier3:run:final:lang:es:paper:0', 'tier3:run:final:lang:es:proof:proof_add:1']);
  });

  test('keeps an expanded proof open when its translation arrives', async () => {
    const pending = { status: 'pending', translated_units: 0, total_units: 3 };
    respondWith(pending);
    const { container, rerender } = renderTranslated(<PaperProofViewer documentId="paper:run:p1" content={content} />);
    await settle();
    container.querySelector('details').open = true;

    respondWith({ status: 'complete', translated_units: 3, parts: { content: translatedContent } });
    rerender(<PaperProofViewer documentId="paper:run:p1" content={`${content}\n`} />);
    await settle();

    expect(ids()).toContain('paper:run:p1:lang:es:proof:proof_add:1');
    expect(container.querySelector('details').open).toBe(true);
  });

  describe('downloads save the language shown', () => {
    let downloads;

    // A parent whose download beside the viewer asks what the viewer shows.
    function DownloadingParent({ showViewer = true, viewerContent }) {
      const { reportView, downloadFor } = useViewedTranslations();
      downloads = downloadFor;
      return showViewer
        ? <PaperProofViewer documentId="paper:run:p1" content={viewerContent} onTranslationView={reportView} />
        : null;
    }

    test('saves the complete translation without the display-only disclaimer', async () => {
      respondWith({ status: 'complete', translated_units: 3, parts: { content: translatedContent } });
      const { rerender } = renderTranslated(<DownloadingParent viewerContent={`${PAPER_DISCLAIMER}

${content}`} />);
      await settle();

      expect(downloads(content, 'paper_p1')).toEqual({
        content: translatedContent,
        filename: 'paper_p1_es',
        language: 'es',
      });
      // Another document's download is never swapped for this translation.
      expect(downloads('Another paper.', 'paper_p2')).toEqual({
        content: 'Another paper.',
        filename: 'paper_p2',
        language: null,
      });

      rerender(<DownloadingParent showViewer={false} viewerContent={content} />);
      expect(downloads(content, 'paper_p1')).toEqual({ content, filename: 'paper_p1', language: null });
    });

    test('saves the original while the translation is pending or the original is chosen', async () => {
      respondWith({ status: 'pending', translated_units: 1, total_units: 3 });
      renderTranslated(<DownloadingParent viewerContent={content} />);
      await settle();
      expect(downloads(content, 'paper_p1')).toEqual({ content, filename: 'paper_p1', language: null });
      cleanup();

      respondWith({ status: 'complete', translated_units: 3, parts: { content: translatedContent } });
      renderTranslated(<DownloadingParent viewerContent={content} />, contextValue({ displayLanguage: 'original' }));
      await settle();
      expect(downloads(content, 'paper_p1')).toEqual({ content, filename: 'paper_p1', language: null });
    });
  });
});
