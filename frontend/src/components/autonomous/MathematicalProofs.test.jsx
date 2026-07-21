import React from 'react';
import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import MathematicalProofs from './MathematicalProofs';

const proof = {
  proof_id: 'proof-1',
  run_id: 'run-1',
  user_prompt: 'Prove the prompt-level theorem',
  theorem_statement: 'theorem prompt_level : True',
  source_type: 'paper',
  source_id: 'paper-1',
  source_title: 'Paper one',
  lean_code: 'theorem prompt_level : True := by trivial',
  novelty_tier: 'mathematical_discovery',
  novel: true,
};

function buildApi() {
  return {
    getProofs: vi.fn().mockResolvedValue({ proofs: [proof] }),
    getProofStatus: vi.fn().mockResolvedValue({ lean4_enabled: false }),
    getBrainstorms: vi.fn().mockResolvedValue({ brainstorms: [] }),
    getPapers: vi.fn().mockResolvedValue({ papers: [] }),
  };
}

test('shows active proofs directly without a prompt-level collapse', async () => {
  const user = userEvent.setup();
  render(<MathematicalProofs api={buildApi()} />);

  expect(await screen.findByText(proof.theorem_statement)).toBeInTheDocument();
  expect(screen.queryByRole('button', { name: /Prove the prompt-level theorem/i })).not.toBeInTheDocument();
  expect(screen.queryByText(proof.user_prompt)).not.toBeInTheDocument();
  await user.click(screen.getByRole('button', { name: 'View Details' }));
  expect(screen.getByText(proof.lean_code)).toBeInTheDocument();
});

test('opens a selected active proof directly', async () => {
  render(<MathematicalProofs api={buildApi()} selectedProofId="proof-1" />);

  expect(await screen.findByText(proof.theorem_statement)).toBeInTheDocument();
  expect(screen.getByText(proof.lean_code)).toBeInTheDocument();
});
