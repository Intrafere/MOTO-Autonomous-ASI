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

const userPrunedProof = {
  ...proof,
  proof_id: 'proof-pruned',
  proof_set_revision: 4,
  live_context_status: 'pruned',
  live_context_owner_run_id: 'run-1',
  live_context_pruned_by: 'user',
  live_context_prune_reason: 'Not useful to this route',
};

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

test('shows independent prune provenance and user-only undo without hiding downloads', async () => {
  const api = buildApi();
  api.getProofs.mockResolvedValue({ proofs: [userPrunedProof], proof_set_revision: 4 });
  api.updateProofLiveContext = vi.fn().mockResolvedValue({
    proof_set_revision: 5,
    live_context_status: 'active',
  });
  const user = userEvent.setup();
  render(<MathematicalProofs api={api} />);

  expect(await screen.findByText('Pruned from live context')).toBeInTheDocument();
  expect(screen.getByText('Pruned by: User')).toBeInTheDocument();
  expect(screen.getByRole('button', { name: 'Download .lean' })).toBeInTheDocument();
  await user.click(screen.getByRole('button', { name: 'View Details' }));
  await user.click(screen.getByRole('button', { name: 'Undo user prune' }));
  expect(api.updateProofLiveContext).toHaveBeenCalledWith(
    expect.objectContaining({
      proofId: 'proof-pruned',
      scope: 'autonomous',
      status: 'active',
      runId: 'run-1',
      proofSetRevision: 4,
    })
  );
});

test('does not offer undo for automatic pruning', async () => {
  const api = buildApi();
  api.getProofs.mockResolvedValue({
    proofs: [{ ...userPrunedProof, live_context_pruned_by: 'automatic_proof_pruning' }],
    proof_set_revision: 4,
  });
  const user = userEvent.setup();
  render(<MathematicalProofs api={api} />);
  await screen.findByText('Pruned from live context');
  await user.click(screen.getByRole('button', { name: 'View Details' }));
  expect(screen.queryByRole('button', { name: 'Undo user prune' })).not.toBeInTheDocument();
});
