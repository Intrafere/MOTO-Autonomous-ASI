import React from 'react';
import { render, screen, within } from '@testing-library/react';
import { afterEach, beforeEach, expect, test, vi } from 'vitest';
import LeanOJProofLibrary from './LeanOJProofLibrary';
import { AutoTranslateContext } from '../../hooks/useAutoTranslate';
import { translationAPI } from '../../services/api';

vi.mock('../../utils/downloadHelpers', () => ({
  downloadTextFile: vi.fn(),
}));

const listProof = {
  proof_id: 'leanoj-proof-1',
  session_id: 'leanoj-session-1',
  theorem_name: 'LeanOJ.Helper',
  theorem_statement: 'Every admissible helper sequence terminates.',
  proof_kind: 'subproof',
  created_at: '2026-06-12T00:00:00+00:00',
};

const hydratedProof = {
  ...listProof,
  novelty_scope: 'group',
  novelty_group_claim: 'Together these fragments bound the template.',
  novelty_reasoning: 'Not displayed in the library view.',
  lean_template: 'theorem template : True := sorry',
  lean_code: 'theorem leanoj_helper : True := by trivial',
};

const TRANSLATED = {
  theorem_statement: 'Toda sucesión auxiliar admisible termina.',
  novelty_group_claim: 'Juntos, estos fragmentos acotan la plantilla.',
};

function buildApi() {
  return {
    getProofLibrary: vi.fn().mockResolvedValue({
      proofs: [listProof],
      sessions: [{ session_id: 'leanoj-session-1', user_prompt: 'LeanOJ run', updated_at: '2026-06-12T00:00:00+00:00' }],
    }),
    getLibraryProof: vi.fn().mockResolvedValue(hydratedProof),
  };
}

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

beforeEach(() => {
  Element.prototype.scrollIntoView = vi.fn();
  vi.spyOn(console, 'warn').mockImplementation(() => {});
});

afterEach(() => {
  vi.restoreAllMocks();
});

test('translates the opened Proof Solver library entry and keeps Lean code and templates original', async () => {
  const register = vi.spyOn(translationAPI, 'registerDocuments').mockImplementation(async ({ documents }) => ({
    success: true,
    documents: documents.map((document) => ({
      doc_key: document.docKey,
      doc_hash: `hash-${document.docKey}`,
      language: 'es',
      status: 'complete',
      translated_units: 2,
      total_units: 2,
      parts: TRANSLATED,
    })),
  }));
  render(
    <AutoTranslateContext.Provider value={contextValue()}>
      <LeanOJProofLibrary api={buildApi()} selectedProofId="leanoj-proof-1" selectedSessionId="leanoj-session-1" />
    </AutoTranslateContext.Provider>
  );

  const details = await screen.findByRole('region', { name: 'Details for LeanOJ.Helper' });
  expect(await within(details).findByText(TRANSLATED.theorem_statement, {}, { timeout: 3000 })).toBeInTheDocument();
  expect(register).toHaveBeenCalledWith({
    language: 'es',
    viewing: true,
    documents: [{
      docKey: 'proof:leanoj:leanoj-proof-1',
      kind: 'proof',
      title: 'LeanOJ.Helper',
      profile: 'fields',
      parts: {
        theorem_statement: hydratedProof.theorem_statement,
        novelty_group_claim: hydratedProof.novelty_group_claim,
      },
    }],
  });
  expect(within(details).getByText(TRANSLATED.novelty_group_claim)).toBeInTheDocument();
  expect(within(details).getByText(hydratedProof.lean_code)).toBeInTheDocument();
  expect(within(details).getByText(hydratedProof.lean_template)).toBeInTheDocument();
  expect(screen.getAllByText(TRANSLATED.theorem_statement)).toHaveLength(2);
  expect(screen.queryByText(listProof.theorem_statement)).not.toBeInTheDocument();
  expect(screen.getByText('LeanOJ.Helper')).toBeInTheDocument();
});

function respondPending() {
  vi.spyOn(translationAPI, 'registerDocuments').mockImplementation(async ({ documents }) => ({
    success: true,
    documents: documents.map((document) => ({
      doc_key: document.docKey,
      doc_hash: `hash-${document.docKey}`,
      language: 'es',
      status: 'pending',
      translated_units: 1,
      total_units: 2,
      parts: null,
    })),
  }));
}

test('a Proof Solver library entry waiting for its translation points to Translate all', async () => {
  respondPending();
  render(
    <AutoTranslateContext.Provider value={contextValue()}>
      <LeanOJProofLibrary api={buildApi()} selectedProofId="leanoj-proof-1" selectedSessionId="leanoj-session-1" />
    </AutoTranslateContext.Provider>
  );
  expect(await screen.findByText(/^Spanish translation pending/, {}, { timeout: 3000 })).toHaveTextContent(
    'Spanish translation pending — 50% translated. Showing the original until it is ready. To translate '
      + 'your whole library in the background, use Translate all older items in the Foreign Language '
      + 'Translation w/ Top-p window (Skills And Enhancements).',
  );
});

test('while LeanOJ is unchecked the hint says to check it in Current runs first', async () => {
  respondPending();
  const status = { enabled: true, state: 'translating', modes: ['autonomous', 'aggregator', 'compiler'] };
  render(
    <AutoTranslateContext.Provider value={contextValue(status)}>
      <LeanOJProofLibrary api={buildApi()} selectedProofId="leanoj-proof-1" selectedSessionId="leanoj-session-1" />
    </AutoTranslateContext.Provider>
  );
  expect(await screen.findByText(/^Spanish translation pending/, {}, { timeout: 3000 })).toHaveTextContent(
    'To translate your whole library in the background, check LeanOJ Proof Solver in Current runs, then use '
      + 'Translate all older items in the Foreign Language Translation w/ Top-p window (Skills And Enhancements).',
  );
});

test('renders no language switch and requests nothing without a translation provider', async () => {
  const register = vi.spyOn(translationAPI, 'registerDocuments');
  render(<LeanOJProofLibrary api={buildApi()} selectedProofId="leanoj-proof-1" selectedSessionId="leanoj-session-1" />);

  const details = await screen.findByRole('region', { name: 'Details for LeanOJ.Helper' });
  expect(await within(details).findByText(hydratedProof.lean_code)).toBeInTheDocument();
  await new Promise((resolve) => { setTimeout(resolve, 600); });
  expect(screen.queryByLabelText('Language')).not.toBeInTheDocument();
  expect(register).not.toHaveBeenCalled();
});
