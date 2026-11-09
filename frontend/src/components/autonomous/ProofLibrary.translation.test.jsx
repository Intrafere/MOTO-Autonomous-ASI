import React from 'react';
import { render, screen, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { afterEach, beforeEach, expect, test, vi } from 'vitest';
import ProofLibrary from './ProofLibrary';
import { AutoTranslateContext } from '../../hooks/useAutoTranslate';
import { autonomousAPI, proofSearchAPI, translationAPI } from '../../services/api';

vi.mock('../../services/api', () => ({
  autonomousAPI: {
    getProofLibrary: vi.fn(),
    getSessions: vi.fn(),
    getLibraryProof: vi.fn(),
    getLibraryProofNoveltyGroup: vi.fn(),
  },
  syntheticLib4API: {
    getTradeEvidenceDiagnostics: vi.fn(),
  },
  proofSearchAPI: {
    getOverview: vi.fn(),
    search: vi.fn(),
    getProof: vi.fn(),
    reindex: vi.fn(),
  },
  translationAPI: {
    registerDocuments: vi.fn(),
    lookupDocuments: vi.fn(),
  },
}));

vi.mock('../../utils/researchRunHistory', () => ({
  buildResearchRunGroups: vi.fn(() => []),
  formatRunPromptPreview: vi.fn((value) => value || 'Unknown run'),
}));

vi.mock('../../utils/downloadHelpers', () => ({
  downloadTextFile: vi.fn(),
}));

const listProof = {
  proof_id: 'proof-a',
  library_id: 'session-a:proof-a',
  session_id: 'session-a',
  run_id: 'run-a',
  theorem_name: 'Lib.Helper',
  theorem_statement: 'Every library helper structure is inhabited.',
  novelty_tier: 'mathematical_discovery',
  novel: true,
};

const hydratedProof = {
  ...listProof,
  novelty_reasoning: 'The statement is absent from standard references.',
  novelty_scope: 'group',
  novelty_group_claim: 'Together these proofs settle the helper question.',
  novelty_group_proof_ids: ['proof-b'],
  formal_sketch: 'Construct the witness directly.',
  lean_code: 'theorem lib_helper : True := by trivial',
};

const TRANSLATED = {
  theorem_statement: 'Toda estructura auxiliar de la biblioteca está habitada.',
  novelty_reasoning: 'El enunciado no aparece en las referencias estándar.',
  novelty_group_claim: 'Juntas, estas demostraciones resuelven la cuestión auxiliar.',
  formal_sketch: 'Construir el testigo directamente.',
};

function contextValue(status = { enabled: true, state: 'translating' }) {
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
  };
}

function wait(ms) {
  return new Promise((resolve) => { setTimeout(resolve, ms); });
}

beforeEach(() => {
  vi.clearAllMocks();
  Element.prototype.scrollIntoView = vi.fn();
  autonomousAPI.getProofLibrary.mockResolvedValue({ proofs: [listProof], counts: { novel: 1 } });
  autonomousAPI.getSessions.mockResolvedValue({ sessions: [] });
  proofSearchAPI.getOverview.mockResolvedValue({ total_records: 1, result_cap: 7, corpora: [] });
  proofSearchAPI.getProof.mockResolvedValue(hydratedProof);
  translationAPI.registerDocuments.mockImplementation(async ({ documents }) => ({
    success: true,
    documents: documents.map((document) => ({
      doc_key: document.docKey,
      doc_hash: `hash-${document.docKey}`,
      language: 'es',
      status: 'complete',
      translated_units: 4,
      total_units: 4,
      parts: TRANSLATED,
    })),
  }));
});

afterEach(() => {
  vi.restoreAllMocks();
});

test('translates the opened library proof only and keeps Lean code original', async () => {
  const user = userEvent.setup();
  render(
    <AutoTranslateContext.Provider value={contextValue()}>
      <ProofLibrary />
    </AutoTranslateContext.Provider>
  );

  expect(await screen.findByText(listProof.theorem_statement)).toBeInTheDocument();
  expect(screen.getByLabelText('Language')).toHaveValue('es');
  await wait(600);
  expect(translationAPI.registerDocuments).not.toHaveBeenCalled();

  await user.click(screen.getByRole('button', { name: 'Expand proof Lib.Helper' }));

  const details = await screen.findByRole('region', { name: 'Details for Lib.Helper' });
  expect(await within(details).findByText(TRANSLATED.theorem_statement, {}, { timeout: 3000 })).toBeInTheDocument();
  expect(translationAPI.registerDocuments).toHaveBeenCalledWith({
    language: 'es',
    viewing: true,
    documents: [{
      docKey: 'proof:library:session-a:proof-a',
      kind: 'proof',
      title: 'Lib.Helper',
      profile: 'fields',
      parts: {
        theorem_statement: hydratedProof.theorem_statement,
        formal_sketch: hydratedProof.formal_sketch,
        novelty_reasoning: hydratedProof.novelty_reasoning,
        novelty_group_claim: hydratedProof.novelty_group_claim,
      },
    }],
  });
  expect(within(details).getByText(TRANSLATED.novelty_reasoning)).toBeInTheDocument();
  expect(within(details).getByText(TRANSLATED.novelty_group_claim)).toBeInTheDocument();
  expect(within(details).getByText(hydratedProof.lean_code)).toBeInTheDocument();
  expect(screen.getAllByText(TRANSLATED.theorem_statement)).toHaveLength(2);
  expect(screen.queryByText(listProof.theorem_statement)).not.toBeInTheDocument();
  expect(screen.getByText('Lib.Helper')).toBeInTheDocument();
  expect(screen.getByText('Translated')).toBeInTheDocument();
});

function respondQueued() {
  translationAPI.registerDocuments.mockImplementation(async ({ documents }) => ({
    success: true,
    documents: documents.map((document) => ({
      doc_key: document.docKey,
      doc_hash: `hash-${document.docKey}`,
      language: 'es',
      status: 'pending',
      translated_units: 0,
      total_units: 4,
      queue_position: 3,
      parts: null,
    })),
  }));
}

test('a library proof waiting for its translation points to Translate all', async () => {
  respondQueued();
  const user = userEvent.setup();
  render(
    <AutoTranslateContext.Provider value={contextValue()}>
      <ProofLibrary />
    </AutoTranslateContext.Provider>
  );
  await user.click(await screen.findByRole('button', { name: 'Expand proof Lib.Helper' }));
  expect(await screen.findByText(/^Spanish translation queued/, {}, { timeout: 3000 })).toHaveTextContent(
    'Spanish translation queued — #3 in line. Showing the original until it is ready. To translate your whole '
      + 'library in the background, use Translate all older items in the Foreign Language Translation w/ '
      + 'Top-p window (Skills And Enhancements).',
  );
});

test.each([
  ['autonomous', undefined, ['aggregator', 'compiler'], 'Autonomous Research'],
  ['manual', 'manual', ['autonomous'], 'Aggregator and Single Paper Writer'],
])('the %s library says which mode to check while none of its modes is', async (_, proofScope, modes, names) => {
  respondQueued();
  const user = userEvent.setup();
  render(
    <AutoTranslateContext.Provider value={contextValue({ enabled: true, state: 'translating', modes })}>
      <ProofLibrary proofScope={proofScope} />
    </AutoTranslateContext.Provider>
  );
  await user.click(await screen.findByRole('button', { name: 'Expand proof Lib.Helper' }));
  expect(await screen.findByText(/^Spanish translation queued/, {}, { timeout: 3000 })).toHaveTextContent(
    `To translate your whole library in the background, check ${names} in Current runs, then use Translate all `
      + 'older items in the Foreign Language Translation w/ Top-p window (Skills And Enhancements).',
  );
});

test('with only Aggregator checked, an opened Single Paper Writer proof says Translate all skips its kind', async () => {
  autonomousAPI.getProofLibrary.mockResolvedValue({
    proofs: [{ ...listProof, source_type: 'paper', source_id: 'manual_compiler_abc' }],
    counts: { novel: 1 },
  });
  respondQueued();
  const user = userEvent.setup();
  render(
    <AutoTranslateContext.Provider value={contextValue({ enabled: true, state: 'translating', modes: ['aggregator'] })}>
      <ProofLibrary proofScope="manual" />
    </AutoTranslateContext.Provider>
  );
  await user.click(await screen.findByRole('button', { name: 'Expand proof Lib.Helper' }));
  const notice = await screen.findByText(/^Spanish translation queued/, {}, { timeout: 3000 });
  expect(notice).toHaveTextContent(
    'To translate your library in the background, use Translate all older items in the Foreign Language '
      + 'Translation w/ Top-p window (Skills And Enhancements). It skips Single Paper Writer items until you check '
      + 'Single Paper Writer in Current runs.',
  );
  expect(notice).not.toHaveTextContent('whole library');
});

test('renders no language switch and requests nothing without a translation provider', async () => {
  const user = userEvent.setup();
  render(<ProofLibrary />);

  await user.click(await screen.findByRole('button', { name: 'Expand proof Lib.Helper' }));
  const details = await screen.findByRole('region', { name: 'Details for Lib.Helper' });
  expect(await within(details).findByText(hydratedProof.lean_code)).toBeInTheDocument();
  await wait(600);
  expect(screen.queryByLabelText('Language')).not.toBeInTheDocument();
  expect(translationAPI.registerDocuments).not.toHaveBeenCalled();
  expect(within(details).getByText(hydratedProof.theorem_statement)).toBeInTheDocument();
});
