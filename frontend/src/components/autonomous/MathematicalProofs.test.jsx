import React from 'react';
import { render, screen, waitFor } from '@testing-library/react';
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
    getProofs: vi.fn().mockResolvedValue({ proofs: [proof], proof_set_revision: 4 }),
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

test('checks collapsed cards without expanding and selects all visible eligible proofs', async () => {
  const activeTwo = { ...proof, proof_id: 'proof-2', theorem_statement: 'theorem second : True' };
  const automaticPruned = {
    ...userPrunedProof,
    proof_id: 'proof-system-pruned',
    live_context_pruned_by: 'automatic_proof_pruning',
  };
  const api = buildApi();
  api.getProofs.mockResolvedValue({
    proofs: [proof, activeTwo, userPrunedProof, automaticPruned],
    proof_set_revision: 4,
  });
  const user = userEvent.setup();
  render(<MathematicalProofs api={api} />);

  await screen.findByRole('checkbox', { name: `Select proof ${proof.theorem_statement} (proof-1)` });
  await user.click(screen.getByRole('checkbox', { name: `Select proof ${proof.theorem_statement} (proof-1)` }));
  expect(screen.queryByText(proof.lean_code)).not.toBeInTheDocument();
  expect(screen.getByText(/1 selected · 1 prunable · 0 restorable/)).toBeInTheDocument();

  await user.click(screen.getByRole('checkbox', { name: /Select all visible eligible/ }));
  expect(screen.getByText(/3 selected · 2 prunable · 1 restorable/)).toBeInTheDocument();
  expect(screen.getByRole('checkbox', { name: /Select all visible eligible/ })).toBeChecked();
});

test('bulk prunes only eligible selected proofs and refreshes once after the batch', async () => {
  const activeTwo = { ...proof, proof_id: 'proof-2', theorem_statement: 'theorem second : True' };
  const api = buildApi();
  api.getProofs.mockResolvedValue({
    proofs: [proof, activeTwo, userPrunedProof],
    proof_set_revision: 4,
  });
  api.updateProofLiveContextBulk = vi.fn().mockResolvedValue({ proof_set_revision: 5 });
  api.refreshProofGraph = vi.fn().mockResolvedValue({});
  api.refreshLatestAssistantPack = vi.fn().mockResolvedValue({});
  const user = userEvent.setup();
  render(<MathematicalProofs api={api} />);

  await screen.findByRole('checkbox', { name: `Select proof ${proof.theorem_statement} (proof-1)` });
  expect(api.getProofs).toHaveBeenCalledTimes(1);
  await user.click(screen.getByRole('checkbox', { name: /Select all visible eligible/ }));
  await user.click(screen.getByRole('button', { name: 'Bulk prune' }));
  expect(screen.getByText(/Only proofs eligible for this action are included/)).toBeInTheDocument();
  await user.type(screen.getByLabelText(/Reason \(optional, shared by this batch\)/), 'Shared route cleanup');
  await user.click(screen.getByRole('button', { name: 'Confirm prune' }));

  await waitFor(() => expect(api.updateProofLiveContextBulk).toHaveBeenCalledTimes(1));
  expect(api.updateProofLiveContextBulk).toHaveBeenCalledWith({
    scope: 'autonomous',
    proofSetRevision: 4,
    items: expect.arrayContaining([
      expect.objectContaining({ proofId: 'proof-1', status: 'pruned', runId: 'run-1', reason: 'Shared route cleanup' }),
      expect.objectContaining({ proofId: 'proof-2', status: 'pruned', runId: 'run-1', reason: 'Shared route cleanup' }),
    ]),
  });
  expect(api.updateProofLiveContextBulk.mock.calls[0][0].items).toHaveLength(2);
  await waitFor(() => expect(api.getProofs).toHaveBeenCalledTimes(2));
  expect(api.refreshProofGraph).toHaveBeenCalledTimes(1);
  expect(api.refreshLatestAssistantPack).toHaveBeenCalledTimes(1);
  expect(await screen.findByText('2 proofs pruned from this run’s live model context.')).toBeInTheDocument();
});

test('clears selection and reports all-none bulk eligibility', async () => {
  const api = buildApi();
  api.getProofs.mockResolvedValue({ proofs: [proof], proof_set_revision: 4 });
  const user = userEvent.setup();
  render(<MathematicalProofs api={api} />);

  await screen.findByRole('checkbox', { name: `Select proof ${proof.theorem_statement} (proof-1)` });
  await user.click(screen.getByRole('checkbox', { name: `Select proof ${proof.theorem_statement} (proof-1)` }));
  await user.click(screen.getByRole('button', { name: 'Bulk restore' }));
  expect(screen.getByText('None of the selected proofs are user-pruned and eligible for restore.')).toBeInTheDocument();
  await user.click(screen.getByRole('button', { name: 'Clear selection' }));
  expect(screen.getByText('0 selected')).toBeInTheDocument();
});

test('bulk restores the frozen user-pruned occurrence identity', async () => {
  const api = buildApi();
  api.getProofs.mockResolvedValue({ proofs: [userPrunedProof], proof_set_revision: 4 });
  api.updateProofLiveContextBulk = vi.fn().mockResolvedValue({ proof_set_revision: 5 });
  api.refreshProofGraph = vi.fn().mockResolvedValue({});
  api.refreshLatestAssistantPack = vi.fn().mockResolvedValue({});
  const user = userEvent.setup();
  render(<MathematicalProofs api={api} />);

  await user.click(await screen.findByRole('checkbox', { name: /Select proof/ }));
  await user.click(screen.getByRole('button', { name: 'Bulk restore' }));
  await user.click(screen.getByRole('button', { name: 'Confirm restore' }));

  await waitFor(() => expect(api.updateProofLiveContextBulk).toHaveBeenCalledWith({
    scope: 'autonomous',
    proofSetRevision: 4,
    items: [expect.objectContaining({
      proofId: 'proof-pruned',
      runId: 'run-1',
      status: 'active',
    })],
  }));
});

test('clears stale bulk selection and refreshes current state', async () => {
  const api = buildApi();
  api.updateProofLiveContextBulk = vi.fn().mockRejectedValue(
    Object.assign(new Error('revision conflict'), { status: 409 })
  );
  const user = userEvent.setup();
  render(<MathematicalProofs api={api} />);

  await screen.findByRole('checkbox', { name: `Select proof ${proof.theorem_statement} (proof-1)` });
  await user.click(screen.getByRole('checkbox', { name: `Select proof ${proof.theorem_statement} (proof-1)` }));
  await user.click(screen.getByRole('button', { name: 'Bulk prune' }));
  await user.click(screen.getByRole('button', { name: 'Confirm prune' }));

  expect(await screen.findByText(/proof set changed before the batch completed/i)).toBeInTheDocument();
  expect(screen.getByText('0 selected')).toBeInTheDocument();
  await waitFor(() => expect(api.getProofs).toHaveBeenCalledTimes(2));
});
