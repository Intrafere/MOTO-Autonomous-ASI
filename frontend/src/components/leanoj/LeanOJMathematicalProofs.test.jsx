import React from 'react';
import { render, screen } from '@testing-library/react';
import LeanOJMathematicalProofs from './LeanOJMathematicalProofs';

test('shows current-run proofs directly without rendering the user prompt', async () => {
  const proof = {
    library_id: 'run-1:proof-1',
    proof_id: 'proof-1',
    theorem_statement: 'theorem current_run : True',
    user_prompt: 'A very long private current-run prompt that must not be displayed',
    source_title: '',
    lean_code: 'theorem current_run : True := by trivial',
    proof_kind: 'subproof',
    novelty_tier: 'not_novel',
    novel: false,
  };
  const api = {
    getProofs: vi.fn().mockResolvedValue({ proofs: [proof] }),
  };

  render(<LeanOJMathematicalProofs api={api} status={{ session_id: 'run-1' }} />);

  expect(await screen.findByText(proof.theorem_statement)).toBeInTheDocument();
  expect(screen.queryByText(proof.user_prompt)).not.toBeInTheDocument();
  expect(screen.queryByRole('button', { name: new RegExp(proof.user_prompt, 'i') })).not.toBeInTheDocument();
});

test('does not render the run prompt when a final proof record uses it as statement and source', async () => {
  const prompt = `Solve this run without displaying the prompt ${'x'.repeat(500)}`;
  const proof = {
    library_id: 'run-2:final',
    proof_id: 'final',
    theorem_name: 'final_solution',
    theorem_statement: prompt,
    user_prompt: prompt,
    source_title: prompt,
    lean_code: 'theorem final_solution : True := by trivial',
    proof_kind: 'final',
    novelty_tier: 'not_novel',
    novel: false,
  };
  const api = {
    getProofs: vi.fn().mockResolvedValue({ proofs: [proof] }),
  };

  render(<LeanOJMathematicalProofs api={api} status={{ session_id: 'run-2', user_prompt: prompt }} />);

  expect(await screen.findByText('final_solution')).toBeInTheDocument();
  expect(screen.queryByText(prompt)).not.toBeInTheDocument();
  expect(screen.getByText('Lean 4 verified this Proof Solver proof.')).toBeInTheDocument();
});
