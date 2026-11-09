import React from 'react';
import { render, screen, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { afterEach, beforeEach, expect, test, vi } from 'vitest';
import MathematicalProofs from './MathematicalProofs';
import { AutoTranslateContext } from '../../hooks/useAutoTranslate';
import { translationAPI } from '../../services/api';

const proof = {
  proof_id: 'proof-1',
  run_id: 'run-1',
  theorem_name: 'prompt_level',
  theorem_statement: 'Every prompt-level structure is trivially inhabited.',
  formal_sketch: 'Reduce the goal to True and close it.',
  novelty_reasoning: 'No public reference proves this exact prompt-level statement.',
  novelty_group_claim: '',
  source_type: 'paper',
  source_id: 'paper-1',
  lean_code: 'theorem prompt_level : True := by trivial',
  novelty_tier: 'mathematical_discovery',
  novel: true,
};

const TRANSLATED = {
  theorem_statement: 'Toda estructura del enunciado está habitada trivialmente.',
  formal_sketch: 'Reducir el objetivo a True y cerrarlo.',
  novelty_reasoning: 'Ninguna referencia pública demuestra este enunciado exacto.',
};

function buildApi() {
  return {
    getProofs: vi.fn().mockResolvedValue({ proofs: [proof], proof_set_revision: 1 }),
    getProofStatus: vi.fn().mockResolvedValue({ lean4_enabled: false }),
    getBrainstorms: vi.fn().mockResolvedValue({ brainstorms: [] }),
    getPapers: vi.fn().mockResolvedValue({ papers: [] }),
  };
}

function contextValue(overrides = {}) {
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
    ...overrides,
  };
}

function renderTranslated(ui, value = contextValue()) {
  return render(<AutoTranslateContext.Provider value={value}>{ui}</AutoTranslateContext.Provider>);
}

function respondWith(statusFor) {
  return vi.spyOn(translationAPI, 'registerDocuments').mockImplementation(async ({ documents }) => ({
    success: true,
    documents: documents.map((document) => ({
      doc_key: document.docKey,
      doc_hash: `hash-${document.docKey}`,
      language: 'es',
      ...statusFor(document),
    })),
  }));
}

beforeEach(() => {
  vi.spyOn(console, 'warn').mockImplementation(() => {});
});

afterEach(() => {
  vi.restoreAllMocks();
});

test('shows translated proof prose once complete and leaves Lean code untouched', async () => {
  const register = respondWith(() => ({
    status: 'complete',
    translated_units: 3,
    total_units: 3,
    parts: TRANSLATED,
  }));
  const user = userEvent.setup();
  renderTranslated(<MathematicalProofs api={buildApi()} />);

  expect(await screen.findByText(TRANSLATED.theorem_statement, {}, { timeout: 3000 })).toBeInTheDocument();
  expect(register).toHaveBeenCalledWith({
    language: 'es',
    viewing: true,
    documents: [{
      docKey: 'proof:autonomous:run-1:proof-1',
      kind: 'proof',
      title: 'prompt_level',
      profile: 'fields',
      parts: {
        theorem_statement: proof.theorem_statement,
        formal_sketch: proof.formal_sketch,
        novelty_reasoning: proof.novelty_reasoning,
      },
    }],
  });
  expect(screen.queryByText(proof.theorem_statement)).not.toBeInTheDocument();
  expect(screen.getByText('Translated')).toBeInTheDocument();
  expect(screen.getByLabelText('Language')).toHaveValue('es');
  expect(screen.queryByRole('status')).not.toBeInTheDocument();

  await user.click(screen.getByRole('button', { name: 'View Details' }));
  const details = screen.getByRole('region', { name: /Details for prompt_level/ });
  expect(within(details).getByText(TRANSLATED.formal_sketch)).toBeInTheDocument();
  expect(within(details).getByText(TRANSLATED.novelty_reasoning)).toBeInTheDocument();
  expect(within(details).getByText('prompt_level')).toBeInTheDocument();
  expect(within(details).getByText(proof.lean_code)).toBeInTheDocument();
});

test('keeps the original and shows the queue position of a pending translation', async () => {
  respondWith(() => ({ status: 'pending', translated_units: 1, total_units: 3, queue_position: 2, parts: null }));
  const user = userEvent.setup();
  renderTranslated(<MathematicalProofs api={buildApi()} />);

  expect(await screen.findByText('Queued #2', {}, { timeout: 3000 })).toBeInTheDocument();
  expect(screen.getByText(proof.theorem_statement)).toBeInTheDocument();

  // The full explanation sits inside the opened card.
  await user.click(screen.getByRole('button', { name: 'View Details' }));
  expect(screen.getByText(
    'Spanish translation queued — #2 in line. Showing the original until it is ready.',
  )).toBeInTheDocument();
});

test('every proof card has its own language switch that changes only that proof', async () => {
  const second = { ...proof, proof_id: 'proof-2', theorem_name: 'second_level', theorem_statement: 'A second statement.' };
  const api = buildApi();
  api.getProofs.mockResolvedValue({ proofs: [proof, second], proof_set_revision: 1 });
  respondWith(() => ({ status: 'pending', translated_units: 0, total_units: 3, parts: null }));
  const value = contextValue({ setDocumentLanguages: vi.fn() });
  renderTranslated(<MathematicalProofs api={api} />, value);

  expect(await screen.findByText(second.theorem_statement)).toBeInTheDocument();
  const switches = screen.getAllByLabelText('Language');
  expect(switches).toHaveLength(2);
  switches.forEach((select) => expect(select).toHaveValue('es'));

  await userEvent.selectOptions(switches[1], 'original');
  expect(value.setDocumentLanguages).toHaveBeenCalledWith(['proof:autonomous:run-1:proof-2'], 'original');
  expect(value.setDisplayLanguage).not.toHaveBeenCalled();
});

test('a proof chosen as Original stays original while others follow the default', async () => {
  const second = { ...proof, proof_id: 'proof-2', theorem_name: 'second_level', theorem_statement: 'A second statement.' };
  const api = buildApi();
  api.getProofs.mockResolvedValue({ proofs: [proof, second], proof_set_revision: 1 });
  const register = respondWith(() => ({ status: 'complete', translated_units: 3, total_units: 3, parts: TRANSLATED }));
  renderTranslated(
    <MathematicalProofs api={api} />,
    contextValue({ documentLanguages: { 'proof:autonomous:run-1:proof-2': 'original' } }),
  );

  expect(await screen.findByText(TRANSLATED.theorem_statement, {}, { timeout: 3000 })).toBeInTheDocument();
  expect(screen.getByText(second.theorem_statement)).toBeInTheDocument();
  const [first, original] = screen.getAllByLabelText('Language');
  expect(first).toHaveValue('es');
  expect(original).toHaveValue('original');
  expect(register.mock.calls.flatMap(([body]) => body.documents.map((document) => document.docKey)))
    .toEqual(['proof:autonomous:run-1:proof-1']);
});

test('renders no language switch and requests nothing without a translation provider', async () => {
  const register = vi.spyOn(translationAPI, 'registerDocuments');
  render(<MathematicalProofs api={buildApi()} />);

  expect(await screen.findByText(proof.theorem_statement)).toBeInTheDocument();
  await new Promise((resolve) => { setTimeout(resolve, 600); });
  expect(screen.queryByLabelText('Language')).not.toBeInTheDocument();
  expect(register).not.toHaveBeenCalled();
});
