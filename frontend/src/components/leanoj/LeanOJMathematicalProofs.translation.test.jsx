import React from 'react';
import { render, screen, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { afterEach, beforeEach, expect, test, vi } from 'vitest';
import LeanOJMathematicalProofs from './LeanOJMathematicalProofs';
import { AutoTranslateContext } from '../../hooks/useAutoTranslate';
import { translationAPI } from '../../services/api';

const prompt = 'Solve the private LeanOJ template without displaying this prompt.';

const fragment = {
  library_id: 'run-1:fragment',
  proof_id: 'fragment',
  theorem_name: 'helper_bound',
  theorem_statement: 'Every admissible sequence is bounded by its first term.',
  user_prompt: prompt,
  source_title: '',
  novelty_reasoning: 'This bound is not stated in Mathlib.',
  formal_sketch: 'theorem helper_bound := by omega',
  lean_code: 'theorem helper_bound : True := by trivial',
  proof_kind: 'subproof',
  novelty_tier: 'mathematical_discovery',
  novel: true,
};

const finalProof = {
  library_id: 'run-1:final',
  proof_id: 'final',
  theorem_name: 'final_solution',
  theorem_statement: prompt,
  user_prompt: prompt,
  source_title: prompt,
  novelty_reasoning: 'The final submission closes the template.',
  lean_code: 'theorem final_solution : True := by trivial',
  proof_kind: 'final',
  novelty_tier: 'not_novel',
  novel: false,
};

const TRANSLATED = {
  'proof:leanoj:run-1:fragment': {
    theorem_statement: 'Toda sucesión admisible está acotada por su primer término.',
    novelty_reasoning: 'Esta cota no aparece en Mathlib.',
  },
  'proof:leanoj:run-1:final': {
    novelty_reasoning: 'La entrega final cierra la plantilla.',
  },
};

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

function buildApi() {
  return { getProofs: vi.fn().mockResolvedValue({ proofs: [fragment, finalProof] }) };
}

beforeEach(() => {
  vi.spyOn(console, 'warn').mockImplementation(() => {});
});

afterEach(() => {
  vi.restoreAllMocks();
});

test('shows translated proof prose, never sends the run prompt, and keeps Lean code original', async () => {
  const register = vi.spyOn(translationAPI, 'registerDocuments').mockImplementation(async ({ documents }) => ({
    success: true,
    documents: documents.map((document) => ({
      doc_key: document.docKey,
      doc_hash: `hash-${document.docKey}`,
      language: 'es',
      status: 'complete',
      translated_units: 1,
      total_units: 1,
      parts: TRANSLATED[document.docKey],
    })),
  }));
  const user = userEvent.setup();
  render(
    <AutoTranslateContext.Provider value={contextValue()}>
      <LeanOJMathematicalProofs api={buildApi()} status={{ session_id: 'run-1' }} />
    </AutoTranslateContext.Provider>
  );

  const fragmentTitle = TRANSLATED['proof:leanoj:run-1:fragment'].theorem_statement;
  expect(await screen.findByText(fragmentTitle, {}, { timeout: 3000 })).toBeInTheDocument();
  expect(register).toHaveBeenCalledWith({
    language: 'es',
    viewing: true,
    documents: [
      {
        docKey: 'proof:leanoj:run-1:fragment',
        kind: 'proof',
        title: 'helper_bound',
        profile: 'fields',
        parts: {
          theorem_statement: fragment.theorem_statement,
          novelty_reasoning: fragment.novelty_reasoning,
        },
      },
      {
        docKey: 'proof:leanoj:run-1:final',
        kind: 'proof',
        title: 'final_solution',
        profile: 'fields',
        parts: { novelty_reasoning: finalProof.novelty_reasoning },
      },
    ],
  });
  expect(screen.queryByText(fragment.theorem_statement)).not.toBeInTheDocument();
  expect(screen.getByText('final_solution')).toBeInTheDocument();
  expect(screen.queryByText(prompt)).not.toBeInTheDocument();

  const fragmentCard = screen.getByText(fragmentTitle).closest('article');
  await user.click(within(fragmentCard).getByRole('button', { name: 'View Details' }));
  expect(within(fragmentCard).getByText(TRANSLATED['proof:leanoj:run-1:fragment'].novelty_reasoning)).toBeInTheDocument();
  expect(within(fragmentCard).getByText(fragment.lean_code)).toBeInTheDocument();
});

test('renders no language switch and requests nothing without a translation provider', async () => {
  const register = vi.spyOn(translationAPI, 'registerDocuments');
  render(<LeanOJMathematicalProofs api={buildApi()} status={{ session_id: 'run-1' }} />);

  expect(await screen.findByText(fragment.theorem_statement)).toBeInTheDocument();
  await new Promise((resolve) => { setTimeout(resolve, 600); });
  expect(screen.queryByLabelText('Language')).not.toBeInTheDocument();
  expect(register).not.toHaveBeenCalled();
});
